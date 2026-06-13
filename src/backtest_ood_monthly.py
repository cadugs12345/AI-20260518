"""
📈 v38 月频调仓 · 真实样本外回测
- 每月初调仓日：用 t-730~t-5 训练 → 预测该日 → 选 Top10
- 持有到下月调仓日
- fwd_20d_ret 作为当期收益
- 扣除换仓成本（印花税+佣金+滑点）
- 2017~2026
"""

import sys, os, time, numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")

PROJECT = "/mnt/d/AI-20260518"
os.chdir(PROJECT)
from scipy import stats as ss
import joblib, lightgbm as lgb

N_HOLD = 10
COST_PER_TRADE = 0.0032  # 单边 0.32%
OUTPUT = "backtest_results"
os.makedirs(OUTPUT, exist_ok=True)

t0 = time.time()
print("="*60)
print("📈 v38 月频调仓 · 样本外回测")
print("="*60)

# 参考因子列
ref = joblib.load("models/ml_ensemble_v1.joblib")
factor_cols = ref["factor_cols"]

# 读面板
panel = pd.read_parquet("data/factors/factor_panel_v5_final.parquet",
                        columns=["ts_code", "trade_date", "fwd_20d_ret"] + factor_cols)
panel = panel.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
print(f"面板: {len(panel):,}行, {panel['trade_date'].min()} ~ {panel['trade_date'].max()}")

# rank标签
print("构建rank标签...", end=" ", flush=True)
mask = panel["fwd_20d_ret"].notna() & (panel["fwd_20d_ret"].abs() < 0.5)
panel["label_rank"] = np.nan
panel.loc[mask, "label_rank"] = (
    panel[mask].groupby("trade_date")["fwd_20d_ret"]
    .rank(pct=True, ascending=True)
)
print(f"ok {mask.sum():,.0f}条")

# 每月第一个交易日
dates = sorted(panel["trade_date"].unique())
df_dates = pd.DataFrame({"trade_date": dates})
df_dates["ym"] = df_dates["trade_date"].astype(str).str[:7]
monthly = df_dates.groupby("ym")["trade_date"].first().reset_index()
entry_dates = sorted(monthly["trade_date"].unique())
entry_dates = [d for d in entry_dates if d >= pd.Timestamp("2019-01-01")]  # 从2019开始，留够训练数据
print(f"调仓日: {len(entry_dates)}个 ({entry_dates[0]} ~ {entry_dates[-1]})")

# ====== 滚动回测 ======
records = []
prev_codes = set()

for i, ed in enumerate(entry_dates):
    # 训练
    train_end = ed - pd.Timedelta(days=5)
    train_start = train_end - pd.Timedelta(days=730)
    train_mask = (
        (panel["trade_date"] >= train_start) &
        (panel["trade_date"] <= train_end) &
        panel["label_rank"].notna()
    )
    train = panel[train_mask].copy()
    
    if len(train) < 10000:
        print(f"  ⚠️ {ed}: 训练数据不足 ({len(train)}行), 跳过")
        continue
    
    if len(train) > 200000:
        train = train.sample(200000, random_state=42)
    
    X_tr = train[factor_cols].fillna(0).values.astype(np.float32)
    y_tr = train["label_rank"].values.astype(np.float32)
    n_v = max(1, int(len(train) * 0.15))
    
    lgb_m = lgb.LGBMRegressor(
        n_estimators=500, max_depth=3, learning_rate=0.02,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.2, reg_lambda=1.0,
        min_child_weight=20, min_data_in_leaf=100,
        random_state=42, verbose=-1, n_jobs=8,
    )
    lgb_m.fit(
        X_tr[:-n_v], y_tr[:-n_v],
        eval_set=[(X_tr[-n_v:], y_tr[-n_v:])],
        callbacks=[lgb.early_stopping(30, verbose=False)],
        eval_metric="mse",
    )
    
    # 预测
    day = panel[panel["trade_date"] == ed].copy()
    if len(day) < 10:
        continue
    
    X_te = day[factor_cols].fillna(0).values.astype(np.float32)
    day["score"] = lgb_m.predict(X_te)
    day = day.sort_values("score", ascending=False).head(N_HOLD)
    
    codes = set(day["ts_code"])
    
    # 换手
    if i > 0:
        turnover = 1 - len(prev_codes & codes) / N_HOLD
    else:
        turnover = 1.0
    
    cost = turnover * COST_PER_TRADE
    
    # 当期收益
    fwd_rets = day["fwd_20d_ret"].values
    valid = fwd_rets[~pd.isna(fwd_rets)]
    period_ret = valid.mean() if len(valid) > 0 else 0
    net_ret = (1 + period_ret) * (1 - cost) - 1
    
    records.append({
        "entry_date": ed,
        "period_ret": period_ret,
        "cost": cost,
        "net_ret": net_ret,
        "turnover": turnover,
        "avg_score": day["score"].mean(),
        "best_iter": lgb_m.best_iteration_,
        "codes": list(codes),
    })
    
    prev_codes = codes
    
    if (i+1) % 20 == 0:
        elapsed = time.time() - t0
        print(f"  {i+1}/{len(entry_dates)} ({ed})  {elapsed:.0f}s")

# ====== 统计 ======
df = pd.DataFrame(records)
if len(df) == 0:
    print("❌ 无有效回测结果")
    sys.exit(1)

rets = df["net_ret"].values
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

# 滚动夏普（24个月）
rolling_sharpe = []
for j in range(24, len(rets)):
    rs = rets[j-24:j]
    s = rs.mean() / rs.std() * np.sqrt(12) if rs.std() > 0 else 0
    rolling_sharpe.append(s)

# 分年
df["year"] = df["entry_date"].astype(str).str[:4]
yearly = df.groupby("year").agg(
    N=("net_ret", "count"),
    return_=("net_ret", lambda x: np.prod(1 + x) - 1),
    win_rate=("net_ret", lambda x: (x > 0).mean()),
    avg_turnover=("turnover", "mean"),
    avg_cost=("cost", "mean"),
    avg_best_iter=("best_iter", "mean"),
).reset_index()

# 输出
print("\n" + "="*60)
print("📊 v38 月频调仓 · 样本外回测结果")
print("="*60)
print(f"回测期: {df['entry_date'].min().strftime('%Y-%m-%d')} ~ {df['entry_date'].max().strftime('%Y-%m-%d')} ({total_years:.1f}年)")
print(f"调仓: 月频 | 持仓: {N_HOLD}只等权 | 成本: 单边{COST_PER_TRADE*100:.2f}%")
print(f"总期数: {len(df)}个月")
print()

print("--- 核心指标 ---")
print(f"  累计净收益:     {total_ret*100:+.2f}%")
print(f"  年化收益:       {annual_ret*100:+.2f}%")
print(f"  年化波动(月):   {annual_vol*100:.2f}%")
print(f"  年化夏普:       {sharpe:.2f}")
print(f"  最大回撤:       {max_dd*100:.2f}%")
print(f"  月胜率:         {win_rate*100:.1f}%")
print(f"  平均换手:       {df['turnover'].mean()*100:.1f}%")
print(f"  平均月成本:     {df['cost'].mean()*100:.2f}%")
print(f"  平均训练迭代:   {df['best_iter'].mean():.0f}")
print()

if rolling_sharpe:
    print(f"  滚动夏普(24月): 均值 {np.mean(rolling_sharpe):.2f} | "
          f"最近 {rolling_sharpe[-1]:.2f} | "
          f"最低 {np.min(rolling_sharpe):.2f} | "
          f"最高 {np.max(rolling_sharpe):.2f}")
    print()

print("--- 分年表现 ---")
print(f"{'年份':>6} | {'期数':>4} | {'年收益':>10} | {'月胜率':>7} | {'换手':>6} | {'成本':>6} | {'迭代':>5}")
print("-"*65)
for _, r in yearly.iterrows():
    print(f"{r['year']:>6} | {r['N']:>4d} | {r['return_']*100:>+9.2f}% | {r['win_rate']*100:>6.1f}% | {r['avg_turnover']*100:>5.1f}% | {r['avg_cost']*100:>5.2f}% | {r['avg_best_iter']:>5.0f}")

print()
print("--- 最近12个月 ---")
print(f"{'调仓日':>12} | {'收益':>8} | {'成本':>6} | {'净收益':>8} | {'换手':>6} | {'Score':>7}")
print("-"*60)
for _, r in df.tail(12).iterrows():
    print(f"{str(r['entry_date'])[:10]:>12} | {r['period_ret']*100:>+7.2f}% | {r['cost']*100:>5.2f}% | {r['net_ret']*100:>+7.2f}% | {r['turnover']*100:>5.1f}% | {r['avg_score']:.4f}")

# 保存
nav_df = pd.DataFrame({
    "entry_date": df["entry_date"],
    "nav": nav,
    "dd": dd,
})
nav_df.to_parquet(f"{OUTPUT}/v38_ood_monthly_nav.parquet")
print(f"\n✅ 净值: {OUTPUT}/v38_ood_monthly_nav.parquet")

summary = {
    "version": "v38 月频Top10等权 · 样本外(OOD)",
    "period": f"{df['entry_date'].min().strftime('%Y-%m-%d')} ~ {df['entry_date'].max().strftime('%Y-%m-%d')}",
    "n_months": len(df),
    "years": round(total_years, 1),
    "total_return_pct": round(total_ret * 100, 2),
    "annual_return_pct": round(annual_ret * 100, 2),
    "annual_vol_pct": round(annual_vol * 100, 2),
    "sharpe": round(sharpe, 2),
    "max_dd_pct": round(max_dd * 100, 2),
    "monthly_win_rate_pct": round(win_rate * 100, 1),
    "avg_turnover_pct": round(df['turnover'].mean() * 100, 1),
    "cost_per_trade_pct": COST_PER_TRADE * 100,
}
import json
json.dump(summary, open(f"{OUTPUT}/v38_ood_monthly_summary.json", "w"), indent=2, ensure_ascii=False)
print(f"  摘要: {OUTPUT}/v38_ood_monthly_summary.json")
print(f"⏱ {time.time()-t0:.0f}s")
