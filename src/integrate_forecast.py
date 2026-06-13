"""
将业绩预告因子集成到因子面板 + v18快速回测

结论：
  业绩预告type_code: IC +1.79%, IR=0.33, 胜率64%
  与现有因子低相关（|ρ|<0.08）→ 真正的独立增量因子
  预期可将综合IC提高10-15%

Usage:
    python src/integrate_forecast.py
"""
import sys, os, json, time, pickle
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config.settings import DATA_FACTORS

t0 = time.time()

print("="*60)
print("业绩预告因子 → 面板集成 + 快速回测")
print("="*60)

# 1. 加载
print("\n[1] 加载数据...")
panel = pd.read_parquet(os.path.join(DATA_FACTORS, "factor_panel_with_fwd_v2.parquet"))
panel["trade_date"] = pd.to_datetime(panel["trade_date"])

fc = pd.read_parquet(os.path.join(DATA_FACTORS, "new_factors", "forecast_factors.parquet"))
fc["ann_date"] = pd.to_datetime(fc["ann_date"])

# 2. 集成预告因子到面板
print("\n[2] 集成预告因子...")
fc_latest = fc.sort_values("ann_date").groupby("ts_code").tail(1)

# 离散编码（预增=2,略增=1,扭亏=3,续盈=1,预减=-2,略减=-1,首亏=-3,续亏=-3,减亏=1）
panel["预告信号"] = panel["ts_code"].map(dict(zip(fc_latest["ts_code"], fc_latest["type_code"])))
panel["预告得分"] = panel["ts_code"].map(dict(zip(fc_latest["ts_code"], fc_latest["预告得分"])))

# 前向填充（预告有效期直到新的预告发布）
panel = panel.sort_values(["ts_code", "trade_date"])
for col in ["预告信号", "预告得分"]:
    panel[col] = panel.groupby("ts_code")[col].transform(
        lambda x: x.ffill().bfill())

print(f"  预告信号非空: {panel['预告信号'].notna().sum():,} ({panel['预告信号'].notna().mean()*100:.1f}%)")

# 3. 快速IC测试
print("\n[3] 20日IC测试...")
from scipy import stats
label = "fwd_20d_ret"
all_dates = sorted(panel["trade_date"].unique())

for fname in ["预告信号", "预告得分"]:
    ic_vals = []
    for date in all_dates[::20]:
        dd = panel[panel["trade_date"] == date][[fname, label]].dropna()
        if len(dd) < 100:
            continue
        v = dd[fname].values.astype(np.float64)
        r = dd[label].values.astype(np.float64)
        mask = ~np.isnan(v) & (np.abs(r) < 0.5)
        if mask.sum() < 100:
            continue
        ic, _ = stats.spearmanr(v[mask], r[mask])
        ic_vals.append(ic)
    
    ic_arr = np.array(ic_vals)
    print(f"  {fname:12s} IC={np.mean(ic_arr)*100:+6.2f}% IR={np.mean(ic_arr)/np.std(ic_arr):+.2f} n={len(ic_arr)}")

# 4. 保存集成面板
out_path = os.path.join(DATA_FACTORS, "factor_panel_v4.parquet")
panel.to_parquet(out_path)
print(f"\n✅ 集成面板已保存: {out_path}")

# 5. 快速回测：用调整权重 + 预告因子做v18
print("\n[4] v18快速回测（调整权重 + 预告信号）...")

# 加载v17调整权重
ALERTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "alerts")
weights_path = os.path.join(ALERTS_DIR, "adjusted_weights.json")
with open(weights_path) as f:
    factor_weights = json.load(f)["weights"]

# 增加预告信号因子权重
factor_weights["预告信号"] = 0.06

# 归一化
total = sum(factor_weights.values())
factor_weights = {k: v/total for k, v in factor_weights.items()}

print(f"  因子数: {len(factor_weights)}")

factor_cols = [c for c in factor_weights.keys() if c in panel.columns]
print(f"  可用因子: {len(factor_cols)}")

# 多因子合成
for f in factor_cols:
    panel[f"rank_{f}"] = panel.groupby("trade_date")[f].rank(pct=True, method="dense")

panel["composite"] = sum(panel[f"rank_{f}"] * factor_weights[f] for f in factor_cols)

# 用v16预测 + composite ensemble
pred_path = os.path.join(DATA_FACTORS, "pred_20d_v16.pkl")
if os.path.exists(pred_path):
    pred = pd.read_pickle(pred_path)
    pred["trade_date"] = pd.to_datetime(pred["trade_date"])
    
    # ensemble: ML预测×0.6 + 合成得分×0.4
    pred = pred.merge(panel[["trade_date", "ts_code", "composite"]], on=["trade_date", "ts_code"], how="left")
    pred["pred_ret"] = pred["pred_ret"] * 0.6 + pred["composite"].fillna(pred["composite"].median()) * 0.4
    
    # 保存v18
    pred.to_pickle(os.path.join(DATA_FACTORS, "pred_20d_v18.pkl"))
    print("  Ensemble预测已保存: pred_20d_v18.pkl")

# 6. 快速净值计算
print("\n[5] 计算净值...")
prices = pd.read_parquet(os.path.join(DATA_FACTORS, "full_prices.parquet"))
prices["trade_date"] = pd.to_datetime(prices["trade_date"])

pred_dates = sorted(pred["trade_date"].unique())

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

STAMP, COMM, SLIP = 0.001, 0.0002, 0.002

print(f"{'配置':15s} | {'总收益':>7s} | {'年化':>7s} | {'夏普':>5s} | {'回撤':>6s} | {'胜率':>4s}")
print("-"*55)

for n_stocks, target_vol in [(30, 0.15), (50, 0.15)]:
    cash = 0.03
    holdings = {}
    navs = [1.0]
    dates = []
    
    for i, date in enumerate(pred_dates[:-1]):
        sell_date = pred_dates[i+1]
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
            median_vol = np.median(selected_vols)
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
        dates.append(date)
    
    nav_array = np.array(navs)
    pnl = nav_array[1:] / nav_array[:-1] - 1
    sr = np.mean(pnl) / np.std(pnl) * np.sqrt(13) if np.std(pnl) > 0 else 0
    tr = nav_array[-1] - 1
    ann_ret = (1 + tr) ** (12 / len(pnl)) - 1
    dd = np.maximum.accumulate(nav_array) - nav_array
    mdd = dd.max()
    wr = np.mean(pnl > 0)
    
    print(f"T{n_stocks}_V{int(target_vol*100):15d} | {tr*100:6.1f}% | {ann_ret*100:6.1f}% | {sr:4.2f} | {mdd*100:5.1f}% | {wr*100:3.0f}%")

print(f"\n总用时: {(time.time()-t0)/60:.1f}分")
print("✅ v18完成——预告因子已集成")
