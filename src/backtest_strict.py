"""
📈 v38 月频调仓 · 严格样本外回测
============================================
原则：
1. 每个调仓日 t：只用 t-730 ~ t-5 的数据训练
2. fwd_20d_ret 从原始价格数据实时计算（非面板预计算）
3. 训练集仅用 t-5 之前的，和预测日严格隔离
4. 每期不同随机种子
============================================
"""

import sys, os, time, numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")

PROJECT = "/mnt/d/AI-20260518"
os.chdir(PROJECT)
import joblib, lightgbm as lgb

N_HOLD = 10
COST_PER_TRADE = 0.0032
OUTPUT = "backtest_results"
os.makedirs(OUTPUT, exist_ok=True)
N_TRAIN_SAMPLE = 50000  # 每期训练采样行数

t0 = time.time()
print("="*60)
print("📈 v38 月频调仓 · 严格样本外")
print("="*60)

# ─── 1. 数据 ───
ref = joblib.load("models/ml_ensemble_v1.joblib")
factor_cols = ref["factor_cols"]

panel = pd.read_parquet("data/factors/factor_panel_v5_final.parquet",
                        columns=["ts_code", "trade_date"] + factor_cols)
panel = panel.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
print(f"面板: {len(panel):,}行 ({panel['trade_date'].min()} ~ {panel['trade_date'].max()})")

# 价格数据
prices = pd.read_parquet("data/factors/full_prices.parquet")
prices["trade_date"] = pd.to_datetime(prices["trade_date"])
prices = prices.drop_duplicates(subset=["ts_code", "trade_date"])
prices = prices.sort_values(["ts_code", "trade_date"])
print(f"价格: {len(prices):,}行, {prices['ts_code'].nunique()}只股票")

# 构建快速查找索引：股票→有序价格序列
print("构建价格索引...", end=" ", flush=True)
price_idx = {}
for code, grp in prices.groupby("ts_code"):
    grp = grp.sort_values("trade_date")
    price_idx[code] = {
        "dates": grp["trade_date"].values,
        "closes": grp["close"].values,
    }
print(f"{len(price_idx)}只")

def calc_fwd_ret_fast(code, trade_date):
    """从预索引快速计算未来20日收益"""
    idx = price_idx.get(code)
    if idx is None:
        return np.nan
    dates_arr = idx["dates"]
    closes_arr = idx["closes"]
    
    pos = np.searchsorted(dates_arr, trade_date, side="right") - 1
    if pos < 0 or pos >= len(dates_arr) or dates_arr[pos] != trade_date:
        return np.nan
    
    end_pos = pos + 20
    if end_pos >= len(dates_arr):
        return np.nan
    
    start_px = closes_arr[pos]
    if start_px <= 0:
        return np.nan
    end_px = closes_arr[end_pos]
    return end_px / start_px - 1

def batch_calc_fwd(df, col_code="ts_code", col_date="trade_date"):
    """批量计算一批股票的fwd_20d_ret"""
    results = []
    for _, row in df.iterrows():
        results.append(calc_fwd_ret_fast(row[col_code], row[col_date]))
    return np.array(results)

# ─── 2. 每月调仓日 ───
all_dates = sorted(panel["trade_date"].unique())
df_dates = pd.DataFrame({"trade_date": all_dates})
df_dates["ym"] = df_dates["trade_date"].astype(str).str[:7]
monthly_first = df_dates.groupby("ym")["trade_date"].first().reset_index()
entry_dates = sorted(monthly_first["trade_date"].unique())
entry_dates = [d for d in entry_dates if d >= pd.Timestamp("2019-01-01")]
print(f"调仓日: {len(entry_dates)}个 ({entry_dates[0]} ~ {entry_dates[-1]})")

# ─── 3. 回测 ───
records = []
prev_codes = set()
total_est = len(entry_dates) * (N_TRAIN_SAMPLE / 1000 * 1.2)  # 预估秒数

print(f"\n开始回测 (预估 {total_est/60:.0f}分钟)...")

for i, ed in enumerate(entry_dates):
    # 训练数据
    train_end = ed - pd.Timedelta(days=5)
    train_start = train_end - pd.Timedelta(days=730)
    train = panel[(panel["trade_date"] >= train_start) & 
                  (panel["trade_date"] <= train_end)].copy()
    
    if len(train) < 5000:
        print(f"  ⚠️ {str(ed)[:10]}: 训练不足 ({len(train)}行), 跳过")
        continue
    
    # 采样
    train_sample = train.sample(min(N_TRAIN_SAMPLE, len(train)), random_state=i)
    
    # 算 fwd_20d_ret
    train_fwd = batch_calc_fwd(train_sample)
    train_sample = train_sample.copy()
    train_sample["fwd_20d_ret"] = train_fwd
    
    mask = train_sample["fwd_20d_ret"].notna() & (train_sample["fwd_20d_ret"].abs() < 0.5)
    train_valid = train_sample[mask].copy()
    
    if len(train_valid) < 3000:
        continue
    
    # rank 标签
    train_valid["label_rank"] = (
        train_valid.groupby("trade_date")["fwd_20d_ret"]
        .rank(pct=True, ascending=True)
    )
    
    X_tr = train_valid[factor_cols].fillna(0).values.astype(np.float32)
    y_tr = train_valid["label_rank"].values.astype(np.float32)
    n_v = max(1, int(len(train_valid) * 0.15))
    
    # 训练（每期不同随机种子）
    lgb_m = lgb.LGBMRegressor(
        n_estimators=500, max_depth=3, learning_rate=0.02,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.2, reg_lambda=1.0,
        min_child_weight=20, min_data_in_leaf=100,
        random_state=i, verbose=-1, n_jobs=8,
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
    top10 = day.sort_values("score", ascending=False).head(N_HOLD)
    
    codes = set(top10["ts_code"])
    turnover = 1 - len(prev_codes & codes) / N_HOLD if i > 0 else 1.0
    cost = turnover * COST_PER_TRADE
    
    # 实时算 Top10 的未来20日收益
    period_rets = [calc_fwd_ret_fast(r["ts_code"], ed) for _, r in top10.iterrows()]
    valid_rets = [r for r in period_rets if not np.isnan(r)]
    period_ret = np.mean(valid_rets) if len(valid_rets) > 0 else 0
    net_ret = (1 + period_ret) * (1 - cost) - 1
    
    records.append({
        "entry_date": ed,
        "period_ret": period_ret,
        "cost": cost,
        "net_ret": net_ret,
        "turnover": turnover,
        "avg_score": top10["score"].mean(),
        "best_iter": lgb_m.best_iteration_,
        "codes": list(codes),
        "n_valid": len(valid_rets),
        "n_train": len(train_valid),
    })
    
    prev_codes = codes
    
    if (i+1) % 10 == 0:
        elapsed = time.time() - t0
        print(f"  [{i+1}/{len(entry_dates)}] {str(ed)[:10]}  "
              f"训练{len(train_valid)}行, ret={period_ret*100:+.2f}%  ({elapsed:.0f}s)")

# ─── 4. 统计 ───
df = pd.DataFrame(records)
if len(df) == 0:
    print("❌ 无结果")
    sys.exit(1)

rets = df["net_ret"].values
nav = np.cumprod(1 + rets)
n_months = len(rets)
first_date = df["entry_date"].iloc[0]
last_date = df["entry_date"].iloc[-1]
total_years = (last_date - first_date).days / 365.25

total_ret = nav[-1] - 1
annual_ret = nav[-1] ** (1 / total_years) - 1
annual_vol = rets.std() * np.sqrt(12)
sharpe = annual_ret / annual_vol if annual_vol > 0 else 0

peak = np.maximum.accumulate(nav)
dd = nav / peak - 1
max_dd = dd.min()
win_rate = (rets > 0).mean()

# 滚动夏普
rs_list = [rets[j-24:j].mean() / rets[j-24:j].std() * np.sqrt(12) 
           for j in range(24, len(rets)) if rets[j-24:j].std() > 0]

# 分年
df["year"] = df["entry_date"].astype(str).str[:4]
yearly = df.groupby("year").agg(
    N=("net_ret", "count"),
    ret=("net_ret", lambda x: np.prod(1 + x) - 1),
    win=("net_ret", lambda x: (x > 0).mean()),
    avg_turnover=("turnover", "mean"),
).reset_index()

print("\n" + "="*60)
print("📊 v38 严格样本外 · 回测结果")
print("="*60)
print(f"  期: {str(first_date)[:10]} ~ {str(last_date)[:10]} ({total_years:.1f}年)")
print(f"  成本: 单边{COST_PER_TRADE*100:.2f}% | 期数: {n_months}")
print()
print("--- 核心指标 ---")
print(f"  累计净收益:     {total_ret*100:+.2f}%")
print(f"  年化收益:       {annual_ret*100:+.2f}%")
print(f"  年化波动(月频): {annual_vol*100:.2f}%")
print(f"  年化夏普:       {sharpe:.2f}")
print(f"  最大回撤:       {max_dd*100:.2f}%")
print(f"  月胜率:         {win_rate*100:.1f}%")
print(f"  平均换手:       {df['turnover'].mean()*100:.1f}%")
print(f"  平均成本:       {df['cost'].mean()*100:.2f}%")
if rs_list:
    print(f"  24月滚动夏普:   均值 {np.mean(rs_list):.2f} | 当前 {rs_list[-1]:.2f} | 最低 {np.min(rs_list):.2f}")
print()
print("--- 分年 ---")
print(f"{'年':>4} | {'期':>3} | {'年收益':>9} | {'月胜率':>7} | {'换手':>5}")
print("-"*45)
for _, r in yearly.iterrows():
    print(f"{r['year']:>4} | {r['N']:>3d} | {r['ret']*100:>+8.2f}% | {r['win']*100:>6.1f}% | {r['avg_turnover']*100:>4.1f}%")
print()
print("--- 最近12个月 ---")
print(f"{'调仓日':>12} | {'收益':>7} | {'净收益':>8} | {'换手':>5} | {'Score':>6} | {'有效股':>5}")
print("-"*60)
for _, r in df.tail(12).iterrows():
    print(f"{str(r['entry_date'])[:10]:>12} | {r['period_ret']*100:>+6.2f}% | {r['net_ret']*100:>+7.2f}% | "
          f"{r['turnover']*100:>4.1f}% | {r['avg_score']:.4f} | {r['n_valid']:>4d}")

# 保存
nav_df = pd.DataFrame({"entry_date": df["entry_date"], "nav": nav, "dd": dd})
nav_df.to_parquet(f"{OUTPUT}/v38_strict_nav.parquet")
summary = {
    "version": "v38 严格样本外",
    "period": f"{str(first_date)[:10]} ~ {str(last_date)[:10]}",
    "n_months": n_months,
    "years": round(total_years, 1),
    "total_return_pct": round(total_ret*100, 2),
    "annual_return_pct": round(annual_ret*100, 2),
    "annual_vol_pct": round(annual_vol*100, 2),
    "sharpe": round(sharpe, 2),
    "max_dd_pct": round(max_dd*100, 2),
    "monthly_win_rate_pct": round(win_rate*100, 1),
    "avg_turnover_pct": round(df['turnover'].mean()*100, 1),
    "cost_per_trade_pct": COST_PER_TRADE*100,
}
import json
json.dump(summary, open(f"{OUTPUT}/v38_strict_summary.json", "w"), indent=2, ensure_ascii=False)
print(f"\n✅ {OUTPUT}/v38_strict_nav.parquet")
print(f"⏱ {time.time()-t0:.0f}s")
PYEOF
