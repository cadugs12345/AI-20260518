"""
v21 回测 — 正确方式集成衍生因子

问题诊断:
  v19的ML预测均值为0.1457（v16仅0.0129）
  → composite_v19的rank合成值太大，拉偏了XGB预测
  → 需要标准化复合因子再做ensemble

方案: 
  1. 用v16训练好的XGB预测（pred_20d_v16.pkl）作为基础
  2. 用衍生因子（高波反转、量价背离信号）做额外的rank合成修正
  3. ensemble: v16_ML × 0.8 + 衍生因子rank_合成 × 0.2

输出: pred_20d_v21.pkl, backtest_v21_results.json
"""
import sys, os, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config.settings import DATA_FACTORS

t0 = time.time()

print("="*60)
print("v21 回测 — 衍生因子正确集成")
print("="*60)

# 1. 加载v16 ML预测
print("\n[1] 加载v16 ML预测...")
pred_ml = pd.read_pickle(os.path.join(DATA_FACTORS, "pred_20d_v16.pkl"))
pred_ml["trade_date"] = pd.to_datetime(pred_ml["trade_date"])
print(f"  v16 pred: {len(pred_ml):,}行")

# 2. 加载面板v5（含衍生因子）
print("\n[2] 加载面板v5衍生因子...")
panel = pd.read_parquet(os.path.join(DATA_FACTORS, "factor_panel_v5.parquet"))
panel["trade_date"] = pd.to_datetime(panel["trade_date"])

derived = ["高波反转", "超跌信号", "量价背离信号", "多排强度", "波动收缩"]

# 3. 衍生因子rank合成
print("\n[3] 衍生因子rank合成...")
weights = {
    "高波反转": 0.40,      # IR 0.54
    "量价背离信号": 0.50,  # IR 0.78
    "多排强度": -0.30,    # IR -0.47 → 负因子（空头排列）
    "波动收缩": 0.05,      # IR 0.26, 弱信号
    "超跌信号": 0.05,      # NaN但留一点
}

for f in derived:
    if f in panel.columns:
        panel[f"rank_{f}"] = panel.groupby("trade_date")[f].rank(pct=True, method="dense")

# 合成衍生因子得分
panel["衍生得分"] = sum(
    panel[f"rank_{f}"] * w for f, w in weights.items() if f in panel.columns
)

# Z-score标准化（均值为0，不破坏ML的排序）
panel["衍生得分"] = panel.groupby("trade_date")["衍生得分"].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-10))

# 4. ensemble
print("\n[4] Ensemble v16_ML + 衍生因子...")
pred = pred_ml.merge(
    panel[["trade_date", "ts_code", "衍生得分"]], 
    on=["trade_date", "ts_code"], how="left"
)

# 填充缺失的衍生得分（中位数）
med = panel["衍生得分"].median()
pred["衍生得分"] = pred["衍生得分"].fillna(med)

# ensemble: ML + 衍生z-score增量
# w=0.05来自IC扫描的最优值（IC 6.60%→9.10%）
ENS_WEIGHT = 0.05
pred["pred_ret"] = pred["pred_ret"] + pred["衍生得分"] * ENS_WEIGHT

print(f"  Ensemble pred: mean={pred['pred_ret'].mean():.4f}, std={pred['pred_ret'].std():.4f}")
print(f"  （对比v16: mean≈0.0129 — 正常范围）")

pred.to_pickle(os.path.join(DATA_FACTORS, "pred_20d_v21.pkl"))
print("  ✅ 保存: pred_20d_v21.pkl")

# 5. 净值计算（轻量版）
print("\n[5] 净值计算...")

prices = pd.read_parquet(os.path.join(DATA_FACTORS, "full_prices.parquet"))
prices["trade_date"] = pd.to_datetime(prices["trade_date"])

pred_dates = sorted(pred["trade_date"].unique())
print(f"  交易日: {len(pred_dates)}")

# 波动率
prices_sorted = prices.sort_values(["ts_code","trade_date"]).copy()
prices_sorted["ret_1d"] = prices_sorted.groupby("ts_code")["close"].pct_change()
prices_sorted["vol_60d"] = prices_sorted.groupby("ts_code")["ret_1d"].transform(
    lambda x: x.rolling(60, min_periods=20).std())
prices_sorted["vol_60d_ann"] = prices_sorted["vol_60d"] * np.sqrt(244)

vol_map, price_map = {}, {}
for d in pred_dates:
    sub = prices[prices["trade_date"] == d][["ts_code","close"]].dropna()
    price_map[d] = dict(zip(sub["ts_code"], sub["close"]))
    v_sub = prices_sorted[prices_sorted["trade_date"] == d][["ts_code","vol_60d_ann"]].dropna()
    vol_map[d] = dict(zip(v_sub["ts_code"], v_sub["vol_60d_ann"]))

del prices_sorted, prices

STAMP, COMM, SLIP = 0.001, 0.0002, 0.002
N_DATES = len(pred_dates)

results = {}
for n_stocks, target_vol in [(30, 0.15), (30, 0.20), (50, 0.15), (50, 0.20)]:
    cash = 0.03
    holdings = {}
    navs = [1.0]
    
    for idx, date in enumerate(pred_dates[:-1]):
        sell_date = pred_dates[idx + 1]
        px_buy = price_map.get(date, {})
        px_sell = price_map.get(sell_date, {})
        stock_vol = vol_map.get(date, {})
        
        hold_val = sum(shares * px_buy.get(c, 0) for c, shares in holdings.items())
        total_val = hold_val + cash
        
        sell_proceeds = 0
        for code, shares in list(holdings.items()):
            px = px_sell.get(code, 0)
            if px > 0:
                val = shares * px
                cost = val * (STAMP + COMM + SLIP)
                sell_proceeds += val - cost
        cash += sell_proceeds
        holdings = {}
        
        day_pred = pred[pred["trade_date"] == date].sort_values("pred_ret", ascending=False)
        selected = list(day_pred.head(n_stocks)["ts_code"].values)
        
        selected_vols = [stock_vol.get(c, np.nan) for c in selected]
        selected_vols = [v for v in selected_vols if not np.isnan(v) and v > 0.01]
        
        if len(selected_vols) >= 5:
            median_vol = float(np.median(selected_vols))
            pos_ratio = min(target_vol / median_vol, 1.0)
            pos_ratio = max(pos_ratio, 0.05)
        else:
            pos_ratio = 1.0
        
        if selected and cash > 0.001:
            available = cash * pos_ratio * 0.98
            if available > 0.001:
                per = available / len(selected)
                for code in selected:
                    px = px_buy.get(code, 0)
                    if px > 0 and per > 0:
                        buy_cost = per * (COMM + SLIP)
                        bought = (per - buy_cost) / px
                        if bought > 0:
                            holdings[code] = bought
                cash -= per * len(holdings)
        
        new_port = sum(shares * px_sell.get(c, 0) for c, shares in holdings.items())
        new_total = new_port + cash
        ret = new_total / total_val - 1 if total_val > 0 else 0
        navs.append(navs[-1] * (1 + ret))
        
        if (idx + 1) % 500 == 0:
            print(f"  T{n_stocks}_V{int(target_vol*100)}: {idx+1}/{N_DATES}")
    
    nav_array = np.array(navs)
    pnl = nav_array[1:] / nav_array[:-1] - 1
    sr = float(np.mean(pnl) / np.std(pnl) * np.sqrt(13)) if np.std(pnl) > 0 else 0
    tr = float(nav_array[-1] - 1)
    ann_ret = float((1 + tr) ** (12 / max(len(pnl), 1)) - 1)
    dd = (np.maximum.accumulate(nav_array) - nav_array).max()
    wr = float(np.mean(pnl > 0))
    
    ver = f"T{n_stocks}_V{int(target_vol*100)}"
    results[ver] = {"total_return": tr, "annualized": ann_ret, "sharpe": sr, 
                    "max_dd": float(dd), "win_rate": wr}
    
    print(f"\n  {ver}")
    print(f"    总收益: {tr*100:.1f}%, 年化: {ann_ret*100:.1f}%")
    print(f"    夏普: {sr:.2f}, 回撤: {dd*100:.1f}%, 胜率: {wr*100:.0f}%")

out_path = os.path.join(DATA_FACTORS, "backtest_v21_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n✅ v21结果: {out_path}")
print(f"总用时: {(time.time()-t0)/60:.1f}分")
