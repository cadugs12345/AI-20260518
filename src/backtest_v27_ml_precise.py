"""
v27 ML回测 — 复用v16框架，对比v12/RF/RF+风控
使用v16的完整回测引擎（印花税/佣金/滑点/目标波动率）
"""
import os, sys, time, json, pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

PROJECT = "/mnt/d/AI-20260518"
os.chdir(PROJECT)
sys.path.insert(0, '.')
import joblib

t0 = time.time()
print("="*60)
print("v27 ML对比回测 — 精算版")
print(f"{time.strftime('%F %H:%M')}")
print("="*60)

DATA_FACTORS = "data/factors"

# ===== 1. 加载 =====
print("\n[1] 加载数据...")
panel = pd.read_parquet(f"{DATA_FACTORS}/factor_panel_v6.parquet")
prices = pd.read_parquet(f"{DATA_FACTORS}/full_prices.parquet")
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
prices["trade_date"] = pd.to_datetime(prices["trade_date"])

factor_cols = [c for c in panel.columns 
               if c not in ["ts_code","trade_date","fwd_20d_ret","close","volume","ret_1d",
                   "__fragment_index","__batch_index","__last_in_fragment","__filename","board_break"]]
core15 = ["短期反转","20日动量","60日动量","120日动量","波动率",
          "换手率","量比","量能趋势","BP","EP","MACD","BOLL位置",
          "EMA5偏离","EMA10偏离","EMA20偏离"]

print(f"  面板: {len(panel):,}行, {len(factor_cols)}因子")

# ===== 2. 预计算 =====
print("\n[2] 预计算波动率...")
prices_sorted = prices.sort_values(["ts_code","trade_date"]).copy()
prices_sorted["ret_1d"] = prices_sorted.groupby("ts_code")["close"].pct_change()
prices_sorted["vol_60d"] = prices_sorted.groupby("ts_code")["ret_1d"].transform(
    lambda x: x.rolling(60, min_periods=20).std())
prices_sorted["vol_60d_ann"] = prices_sorted["vol_60d"] * np.sqrt(244)

# ===== 3. 周期节点 =====
all_dates = sorted(panel["trade_date"].unique())
period_dates = [all_dates[i] for i in range(0, len(all_dates), 20) 
                if all_dates[i] >= pd.Timestamp("2021-01-01")]
print(f"\n[3] 20日周期节点: {len(period_dates)}")

price_map, vol_map = {}, {}
for d in period_dates:
    sub = prices[prices["trade_date"] == d][["ts_code","close"]].dropna()
    price_map[d] = dict(zip(sub["ts_code"], sub["close"]))
    v_sub = prices_sorted[prices_sorted["trade_date"] == d][["ts_code","vol_60d_ann"]].dropna()
    vol_map[d] = dict(zip(v_sub["ts_code"], v_sub["vol_60d_ann"]))

# ===== 4. 生成各策略预测信号 =====
print("\n[4] 生成预测信号...")

# RF模型
rf_md = joblib.load("models/ml_ensemble_v1.joblib")
rf_model = rf_md["model"]
rf_factors = rf_md["factor_cols"]

# 策略A: v12等权
print("  v12等权...", end=" ", flush=True)
v12_z = panel[core15].rank(pct=True)
panel["score_v12"] = v12_z.mean(axis=1)

# 策略B: RF
print("RF...", end=" ", flush=True)
panel["score_rf"] = rf_model.predict_proba(panel[rf_factors].fillna(0))[:, 1]

# 策略C: RF+风控 (断板修复+高波反转+量价背离)
print("RF+风控...", flush=True)

# 构建预测表
pred_records = {"v12": [], "rf": [], "rf_risk": []}

for i, d in enumerate(period_dates):
    day = panel[panel["trade_date"] == d].copy()
    
    for strategy in ["v12", "rf", "rf_risk"]:
        score_col = "score_v12" if strategy == "v12" else "score_rf"
        
        # 基础候选
        candidates = day.dropna(subset=[score_col]).copy()
        
        # 风控
        if strategy == "rf_risk":
            # 逐行检查风控信号
            risk_mask = np.zeros(len(candidates), dtype=bool)
            for j, (_, row) in enumerate(candidates.iterrows()):
                r10 = row.get("repair_force_10d", np.nan)
                hv = row.get("高波反转", np.nan)
                dv = row.get("量价背离", np.nan)
                if (not np.isnan(r10) and r10 < -0.05) or \
                   (not np.isnan(hv) and hv < -0.03) or \
                   (not np.isnan(dv) and dv > 0.03):
                    risk_mask[j] = True
            
            safe = candidates[~risk_mask]
            if len(safe) >= 30:
                candidates = safe
        
        top = candidates.nlargest(50, score_col)
        for _, row in top.iterrows():
            pred_records[strategy].append({
                "trade_date": d, "ts_code": row["ts_code"],
                "pred_ret": row[score_col]
            })
    
    if (i+1) % 20 == 0:
        print(f"    [{i+1}/{len(period_dates)}]", flush=True)

pred = {}
for strategy in ["v12", "rf", "rf_risk"]:
    pred[strategy] = pd.DataFrame(pred_records[strategy])
    print(f"  {strategy}: {len(pred[strategy]):,}条, {pred[strategy]['trade_date'].nunique()}期", flush=True)

# ===== 5. 回测引擎 =====
print(f"\n[5] 回测引擎...")
STAMP, COMM, SLIP = 0.001, 0.0002, 0.002

def backtest_ml(pred_df, n_stocks=30, target_vol=0.15, label=""):
    pred_dates = sorted(pred_df["trade_date"].unique())
    cash = 0.03
    holdings = {}
    navs = [1.0]
    
    for i in range(len(pred_dates) - 1):
        date = pred_dates[i]
        sell_date = pred_dates[i+1]
        px_buy = price_map.get(date, {})
        px_sell = price_map.get(sell_date, {})
        stock_vol = vol_map.get(date, {})
        
        # 前一期资产
        hold_val = sum(shares * px_buy.get(c, 0) for c, shares in holdings.items())
        total_val = hold_val + cash
        
        # 卖出
        sell_proceeds = 0
        for code, shares in holdings.items():
            px = px_sell.get(code, 0)
            if px > 0:
                val = shares * px
                cost = val * (STAMP + COMM + SLIP)
                sell_proceeds += val - cost
        cash = cash + sell_proceeds
        holdings = {}
        
        # 选股
        day_pred = pred_df[pred_df["trade_date"] == date].sort_values("pred_ret", ascending=False)
        selected = list(day_pred.head(n_stocks)["ts_code"].values)
        
        # 截面波动率
        selected_vols = [stock_vol.get(c, np.nan) for c in selected]
        selected_vols = [v for v in selected_vols if not np.isnan(v) and v > 0.01]
        
        if len(selected_vols) >= 5:
            median_vol = np.median(selected_vols)
            pos_ratio = min(target_vol / median_vol, 1.0)
            pos_ratio = max(pos_ratio, 0.05)
        else:
            median_vol = np.nan
            pos_ratio = 1.0
        
        # 买入
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
        
        # 新净值
        new_port = sum(shares * px_sell.get(c, 0) for c, shares in holdings.items())
        new_total = new_port + cash
        ret = new_total / total_val - 1 if total_val > 0 else 0
        navs.append(navs[-1] * (1 + ret))
    
    pnl = np.array(navs[1:]) / np.array(navs[:-1]) - 1
    nav_arr = np.array(navs)
    tr = nav_arr[-1] - 1
    n_years = len(pnl) / 13
    ar = nav_arr[-1] ** (1/n_years) - 1 if n_years > 0 and nav_arr[-1] > 0 else 0
    vol = np.std(pnl) * np.sqrt(13)
    sr = np.mean(pnl) / np.std(pnl) * np.sqrt(13) if np.std(pnl) > 0 else 0
    dd = np.maximum.accumulate(nav_arr) - nav_arr
    mdd = dd.max()
    wr = np.mean(pnl > 0)
    calmar = ar / mdd if mdd > 0 else 0
    
    print(f"  {label:20s}: 年化{ar*100:+.1f}% | 夏普{sr:.2f} | 回撤{mdd*100:.1f}% | "
          f"胜率{wr*100:.0f}% | 卡玛{calmar:.2f} | 实波{vol*100:.1f}% | {len(pnl)}期")
    
    return {
        "ret": f"{ar*100:+.1f}%", "sharpe": f"{sr:.2f}", "mdd": f"{mdd*100:.1f}%",
        "wr": f"{wr*100:.0f}%", "calmar": f"{calmar:.2f}", "vol": f"{vol*100:.1f}%",
        "n": len(pnl), "_ar": ar, "_sr": sr, "_mdd": mdd
    }

print(f"\n回测 (T30 目波15%):")
r_v12 = backtest_ml(pred["v12"], 30, 0.15, "v12等权")
r_rf = backtest_ml(pred["rf"], 30, 0.15, "RF")
r_rf_risk = backtest_ml(pred["rf_risk"], 30, 0.15, "RF+风控")

# T50
print(f"\n回测 (T50 目波15%):")
r_v12_50 = backtest_ml(pred["v12"], 50, 0.15, "v12等权")
r_rf_50 = backtest_ml(pred["rf"], 50, 0.15, "RF")
r_rf_risk_50 = backtest_ml(pred["rf_risk"], 50, 0.15, "RF+风控")

# 汇总
result = {
    "T30_V15": {"v12": r_v12, "rf": r_rf, "rf_risk": r_rf_risk},
    "T50_V15": {"v12": r_v12_50, "rf": r_rf_50, "rf_risk": r_rf_risk_50},
}
json.dump(result, open("output/backtest_v27_ml_precise.json", "w"), indent=2, default=str)

print(f"\n{'='*60}")
print(f"{'策略':20s} {'T30年化':>8s} {'T30夏普':>8s} {'T30回撤':>8s} {'T50年化':>8s} {'T50夏普':>8s}")
print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for label, key in [("v12等权","v12"), ("RF","rf"), ("RF+风控","rf_risk")]:
    r30 = result["T30_V15"][key]
    r50 = result["T50_V15"][key]
    print(f"  {label:20s} {r30['ret']:>8s} {r30['sharpe']:>8s} {r30['mdd']:>8s} "
          f"{r50['ret']:>8s} {r50['sharpe']:>8s} {r50['mdd']:>8s}")

print(f"\n⏱ {(time.time()-t0)/60:.1f}分")
print("✅ output/backtest_v27_ml_precise.json")
