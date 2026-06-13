"""
📈 v38 Top10 实盘回测 — 月频调仓
直接用 fwd_20d_ret 作为每期持有收益
- 每月第一个交易日选 Top10
- Top10 等权未来20日平均收益 = 当月持仓收益
- 扣除换仓成本
- 2017~2026
"""

import sys, os, time, numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")

PROJECT = "/mnt/d/AI-20260518"
os.chdir(PROJECT)
import joblib

N_HOLD = 10
COST_PER_TRADE = 0.0032  # 单边 0.32%（印花税0.1+佣金0.02+滑点0.2）
OUTPUT = "backtest_results"
os.makedirs(OUTPUT, exist_ok=True)

t0 = time.time()
print("="*60)
print("📈 v38 月频调仓实盘回测")
print("="*60)

# 加载模型和因子
md = joblib.load("models/live_lgb_v38_final.joblib")
lgb_m = md["model"]
factor_cols = md["factor_cols"]

# 读面板（含 fwd_20d_ret）
panel = pd.read_parquet("data/factors/factor_panel_v5_final.parquet",
                        columns=["ts_code", "trade_date", "fwd_20d_ret"] + factor_cols)
panel = panel.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

# 预测
print(f"预测中 ({len(panel):,}行)...", end=" ", flush=True)
feat = panel[factor_cols].fillna(0).values.astype(np.float32)
panel["score"] = lgb_m.predict(feat)
print("done")

# 找到每月第一个交易日
dates = sorted(panel["trade_date"].unique())
df_dates = pd.DataFrame({"trade_date": dates})
df_dates["ym"] = df_dates["trade_date"].astype(str).str[:7]
monthly = df_dates.groupby("ym")["trade_date"].first().reset_index()
entry_dates = sorted(monthly["trade_date"].unique())
print(f"调仓日: {len(entry_dates)}个月 ({entry_dates[0]} ~ {entry_dates[-1]})")

# 回测
records = []
prev_codes = set()

for i, ed in enumerate(entry_dates):
    # 该日选 top10
    day = panel[panel["trade_date"] == ed].copy()
    if len(day) < 10:
        continue
    
    day = day.sort_values("score", ascending=False).head(N_HOLD)
    codes = set(day["ts_code"])
    
    # 换手率
    if i > 0:
        turnover = 1 - len(prev_codes & codes) / N_HOLD
    else:
        turnover = 1.0
    
    cost = turnover * COST_PER_TRADE
    
    # 每只股票的 fwd_20d_ret
    fwd_rets = day["fwd_20d_ret"].values
    # 如果有 NaN，用可用的算
    valid = fwd_rets[~pd.isna(fwd_rets)]
    if len(valid) == 0:
        period_ret = 0
    else:
        period_ret = valid.mean()  # 等权
    
    net_ret = (1 + period_ret) * (1 - cost) - 1
    
    records.append({
        "entry_date": ed,
        "period_ret": period_ret,
        "cost": cost,
        "net_ret": net_ret,
        "turnover": turnover,
        "avg_score": day["score"].mean(),
        "codes": list(codes),
    })
    
    prev_codes = codes
    
    if (i+1) % 24 == 0:
        print(f"  {i+1}/{len(entry_dates)} 期 ({ed})")

df = pd.DataFrame(records)
rets = df["net_ret"].values

# ====== 统计 ======
nav = np.cumprod(1 + rets)
peak = np.maximum.accumulate(nav)
dd = nav / peak - 1
total_years = (entry_dates[-1] - entry_dates[0]).days / 365.25

total_ret = nav[-1] - 1
annual_ret = nav[-1] ** (1 / total_years) - 1
annual_vol = rets.std() * np.sqrt(12)
sharpe = annual_ret / annual_vol if annual_vol > 0 else 0
max_dd = dd.min()
win_rate = (rets > 0).mean()
avg_turnover = df["turnover"].mean()

# 分年
df["year"] = df["entry_date"].astype(str).str[:4]
yearly = df.groupby("year").agg(
    期数=("net_ret", "count"),
    年收益=("net_ret", lambda x: np.prod(1 + x) - 1),
    月胜率=("net_ret", lambda x: (x > 0).mean()),
    平均换手=("turnover", "mean"),
    平均成本=("cost", "mean"),
).reset_index()

# 输出
print("\n" + "="*60)
print("📊 回测结果")
print("="*60)
print(f"回测期: {entry_dates[0]} ~ {entry_dates[-1]} ({total_years:.1f}年)")
print(f"频率: 月频 | 持仓: {N_HOLD}只等权 | 成本: 单边{COST_PER_TRADE*100:.2f}%")
print()
print("--- 核心指标 ---")
print(f"  累计净收益:  {total_ret*100:+.2f}%")
print(f"  年化收益:    {annual_ret*100:+.2f}%")
print(f"  年化波动:    {annual_vol*100:.2f}%")
print(f"  年化夏普:    {sharpe:.2f}")
print(f"  最大回撤:    {max_dd*100:.2f}%")
print(f"  月胜率:      {win_rate*100:.1f}%")
print(f"  平均换手:    {avg_turnover*100:.1f}%")
print(f"  平均月成本:  {df['cost'].mean()*100:.2f}%")
print()

print("--- 分年表现 ---")
print(f"{'年份':>6} | {'期数':>4} | {'年收益':>8} | {'月胜率':>7} | {'换手':>6} | {'成本':>6}")
print("-"*55)
for _, r in yearly.iterrows():
    print(f"{r['year']:>6} | {r['期数']:>4d} | {r['年收益']*100:>+7.2f}% | {r['月胜率']*100:>6.1f}% | {r['平均换手']*100:>5.1f}% | {r['平均成本']*100:>5.2f}%")

print()
print("--- 最近12个月 ---")
print(f"{'调仓日':>12} | {'收益':>8} | {'成本':>6} | {'净收益':>8} | {'换手':>6} | {'Score':>7}")
print("-"*60)
for _, r in df.tail(12).iterrows():
    codes_str = ",".join(r["codes"][:3])
    print(f"{str(r['entry_date'])[:10]:>12} | {r['period_ret']*100:>+7.2f}% | {r['cost']*100:>5.2f}% | {r['net_ret']*100:>+7.2f}% | {r['turnover']*100:>5.1f}% | {r['avg_score']:.4f}")

# 保存
nav_df = pd.DataFrame({"entry_date": entry_dates[:len(nav)], "nav": nav, "dd": dd})
nav_df.to_parquet(f"{OUTPUT}/v38_monthly_nav.parquet")

summary = {
    "version": "v38 月频Top10等权",
    "period": f"{entry_dates[0]} ~ {entry_dates[-1]}",
    "years": round(total_years, 1),
    "total_return_pct": round(total_ret * 100, 2),
    "annual_return_pct": round(annual_ret * 100, 2),
    "annual_vol_pct": round(annual_vol * 100, 2),
    "sharpe": round(sharpe, 2),
    "max_dd_pct": round(max_dd * 100, 2),
    "monthly_win_rate_pct": round(win_rate * 100, 1),
    "avg_turnover_pct": round(avg_turnover * 100, 1),
    "cost_per_trade_pct": COST_PER_TRADE * 100,
}
import json
json.dump(summary, open(f"{OUTPUT}/v38_monthly_summary.json", "w"), indent=2, ensure_ascii=False)

print(f"\n✅ 已保存: {OUTPUT}/v38_monthly_nav.parquet + v38_monthly_summary.json")
print(f"⏱ {time.time()-t0:.0f}s")
PYEOF
