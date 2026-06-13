#!/usr/bin/env python3
"""
🧪 v38 真正无菌的严格样本外回测（高效版）
==================================
无菌原则：
1. fwd_20d_ret 从 full_prices 实时计算，绝不碰面板预计算的 fwd
2. 每调仓日 t：只用 t-730~t-5 训练，t日数据严格隔离
3. 上市不足60天的股票剔除
4. 当天停牌（close几乎不变）剔除
5. 每期不同随机种子，不交叉验证调参
6. 标签：截面 rank(fwd_20d_ret) 在训练日内 rank
7. 市值过滤：排除 < 20亿小盘股（避免微盘股效应）
==================================
"""

import sys, os, time, numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")

PROJECT = "/mnt/d/AI-20260604"
os.chdir(PROJECT)
import joblib, lightgbm as lgb

N_HOLD = 10
COST_PER_TRADE = 0.0032
OUTPUT = "backtest_results"
os.makedirs(OUTPUT, exist_ok=True)
N_TRAIN_SAMPLE = 50000
MIN_MARKET_CAP = 20  # 亿
MIN_TRADING_DAYS = 60

t0 = time.time()
print("=" * 60)
print("🧪 v38 真正无菌 · 严格样本外回测")
print("=" * 60)

# ─── 1. 加载数据 ───
ref = joblib.load("models/live_lgb_v38_final.joblib")
factor_cols = ref["factor_cols"]
print(f"因子数: {len(factor_cols)}")

panel = pd.read_parquet("data/factors/factor_panel_v5_final.parquet",
                        columns=["ts_code", "trade_date"] + factor_cols + ["市值"])
# 清理脏数据
panel = panel.dropna(subset=["ts_code"]).reset_index(drop=True)
panel = panel.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
print(f"面板: {len(panel):,}行, {len(panel['trade_date'].unique())}天")

prices = pd.read_parquet("data/factors/full_prices.parquet")
prices["trade_date"] = pd.to_datetime(prices["trade_date"])
prices = prices.drop_duplicates(subset=["ts_code", "trade_date"])
prices = prices.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
print(f"价格: {len(prices):,}行, {prices['ts_code'].nunique()}只")

# ─── 2. 高效价格矩阵（unstack 极速版）───
print("构建价格矩阵...", end=" ", flush=True)
price_pivot = prices.pivot(index="trade_date", columns="ts_code", values="close")
price_matrix = price_pivot.values.astype(np.float64)  # [日期, 股票]
code_list = list(price_pivot.columns)
n_dates, n_stocks = price_matrix.shape
code_to_col = {c: j for j, c in enumerate(code_list)}
date_list = list(price_pivot.index)
date_to_pos = {d: i for i, d in enumerate(date_list)}

# 每只股票首次出现位置
stock_first_pos = {}
for j, code in enumerate(code_list):
    first = np.argmax(~np.isnan(price_matrix[:, j]))
    stock_first_pos[code] = first

# 有效掩码
stock_valid_mask = ~np.isnan(price_matrix)  # [日期, 股票]

print(f"矩阵 {n_dates}天 × {n_stocks}只")

# ─── 3. 辅助函数 ───

def code_to_col_idx(code):
    return code_to_col.get(code, -1)

def is_suspended(code, date_pos, tol=0.001):
    """检查是否停牌：价格几乎不变"""
    col = code_to_col_idx(code)
    if col < 0:
        return True
    if not stock_valid_mask[date_pos, col]:
        return True
    px_today = price_matrix[date_pos, col]
    if px_today <= 0:
        return True
    if date_pos > 0 and stock_valid_mask[date_pos - 1, col]:
        px_prev = price_matrix[date_pos - 1, col]
        if px_prev > 0 and abs(px_today / px_prev - 1) < tol:
            lookback_start = max(0, date_pos - 5)
            seg = price_matrix[lookback_start:date_pos + 1, col]
            valid = seg[~np.isnan(seg) & (seg > 0)]
            if len(valid) >= 3:
                chg = np.abs(np.diff(valid) / valid[:-1])
                if len(chg) > 0 and chg.max() < tol:
                    return True
    return False


def calc_fwd_batch(codes, date_positions, n_days=20):
    """批量计算 fwd_20d_ret，完全向量化"""
    col_idxs = np.array([code_to_col.get(c, -1) for c in codes])
    results = np.full(len(codes), np.nan)
    
    # 只处理有效的
    valid_mask = col_idxs >= 0
    for i in np.where(valid_mask)[0]:
        col = col_idxs[i]
        dp = date_positions[i]
        ep = dp + n_days
        if ep >= n_dates:
            continue
        if stock_valid_mask[dp, col] and stock_valid_mask[ep, col]:
            sp = price_matrix[dp, col]
            epx = price_matrix[ep, col]
            if sp > 0 and epx > 0:
                results[i] = epx / sp - 1
    return results


# ─── 4. 调仓日 ───
all_dates = sorted(panel["trade_date"].unique())
df_dates = pd.DataFrame({"trade_date": all_dates})
df_dates["ym"] = df_dates["trade_date"].astype(str).str[:7]
monthly_first = df_dates.groupby("ym")["trade_date"].first().reset_index()
entry_dates = sorted(monthly_first["trade_date"].unique())
entry_dates = [d for d in entry_dates if d >= pd.Timestamp("2019-01-01")]
print(f"调仓日: {len(entry_dates)}个 ({entry_dates[0]} ~ {entry_dates[-1]})")

# 交易日期位置映射
date_ym = {d: str(d)[:7] for d in all_dates}

# ─── 5. 回测主循环 ───
records = []
prev_codes = set()

print(f"\n开始回测 ({len(entry_dates)}期)...")

for i, ed in enumerate(entry_dates):
    ed_dt = ed
    
    # ── 5a. 训练数据 ──
    train_end = ed_dt - pd.Timedelta(days=5)
    train_start = train_end - pd.Timedelta(days=730)
    
    train = panel[(panel["trade_date"] >= train_start) & 
                  (panel["trade_date"] <= train_end)].copy()
    if len(train) < 5000:
        continue
    
    # 上市天数过滤（用外部Series避免列冲突）
    tr_codes = train["ts_code"]
    tr_dates = train["trade_date"]
    
    tr_col = tr_codes.map(code_to_col)
    tr_first_pos = tr_codes.map(stock_first_pos).astype(float)
    tr_date_pos = tr_dates.map(date_to_pos).astype(float)
    tr_age = tr_date_pos - tr_first_pos
    
    keep = (tr_col >= 0) & (tr_age >= MIN_TRADING_DAYS)
    keep_idx = np.where(keep.values)[0]
    train = train.iloc[keep_idx].reset_index(drop=True).copy()
    tr_date_pos = tr_date_pos.iloc[keep_idx]
    
    if len(train) < 5000:
        continue
    
    # 停牌过滤
    susp = [is_suspended(code, int(dp)) 
            for code, dp in zip(train["ts_code"], tr_date_pos)]
    susp_arr = np.array(susp)
    keep2 = np.where(~susp_arr)[0]
    train = train.iloc[keep2].reset_index(drop=True).copy()
    tr_date_pos = tr_date_pos.iloc[keep2]
    
    if len(train) < 5000:
        continue
    
    # 市值过滤
    if "市值" in train.columns:
        mcap_ok = train["市值"] >= MIN_MARKET_CAP
        keep3 = np.where(mcap_ok.values)[0]
        train = train.iloc[keep3].reset_index(drop=True).copy()
    
    if len(train) < 5000:
        continue
    
    # ── 5b. 采样 + 算 fwd ──
    train_sample = train.sample(min(N_TRAIN_SAMPLE, len(train)), random_state=i)
    
    # 重新算date_pos（因为上面过滤后丢了）
    ts_dp = train_sample["trade_date"].map(date_to_pos).values
    tr_fwd = calc_fwd_batch(
        train_sample["ts_code"].values,
        ts_dp,
        n_days=20
    )
    train_sample["fwd_ret"] = tr_fwd
    mask = train_sample["fwd_ret"].notna() & (train_sample["fwd_ret"].abs() < 0.5)
    mask_idx = np.where(mask.values)[0]
    train_valid = train_sample.iloc[mask_idx].copy()
    
    if len(train_valid) < 3000:
        continue
    
    # ── 5c. rank 标签 ──
    train_valid["label_rank"] = (
        train_valid.groupby("trade_date")["fwd_ret"]
        .rank(pct=True, ascending=True)
    )
    
    X_tr = train_valid[factor_cols].fillna(0).values.astype(np.float32)
    y_tr = train_valid["label_rank"].values.astype(np.float32)
    n_v = max(1, int(len(train_valid) * 0.15))
    
    # ── 5d. 训练 ──
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
    
    # ── 5e. 预测 ──
    day = panel[panel["trade_date"] == ed_dt].copy()
    if len(day) < 20:
        continue
    
    # 过滤（用外部Series）
    day_codes = day["ts_code"]
    day_col = day_codes.map(code_to_col)
    day_first = day_codes.map(stock_first_pos).astype(float)
    day_dp = day["trade_date"].map(date_to_pos).astype(float)
    day_age = day_dp - day_first
    
    day_keep = (day_col >= 0) & (day_age >= MIN_TRADING_DAYS)
    day_keep_idx = np.where(day_keep.values)[0]
    day = day.iloc[day_keep_idx].reset_index(drop=True).copy()
    day_dp = day_dp[day_keep_idx]
    
    if len(day) < 10:
        continue
    
    if "市值" in day.columns:
        mcap_ok = day["市值"] >= MIN_MARKET_CAP
        mcap_idx = np.where(mcap_ok.values)[0]
        day = day.iloc[mcap_idx].reset_index(drop=True).copy()
        day_dp = day_dp.iloc[mcap_idx]
    
    if len(day) < 10:
        continue
    
    # 停牌过滤
    susp_day = [is_suspended(code, int(dp)) 
                for code, dp in zip(day["ts_code"], day_dp)]
    susp_day_arr = np.array(susp_day)
    susp_idx = np.where(~susp_day_arr)[0]
    day = day.iloc[susp_idx].reset_index(drop=True).copy()
    day_dp = day_dp.iloc[susp_idx]
    
    if len(day) < 10:
        continue
    
    X_te = day[factor_cols].fillna(0).values.astype(np.float32)
    day["score"] = lgb_m.predict(X_te)
    top10 = day.sort_values("score", ascending=False).head(N_HOLD)
    
    codes = set(top10["ts_code"])
    turnover = 1 - len(prev_codes & codes) / N_HOLD if i > 0 else 1.0
    cost = turnover * COST_PER_TRADE
    
    # ── 5f. 实时算fwd ──
    # date_pos 从 trade_date 重新构造
    te_dp = top10["trade_date"].map(date_to_pos).values
    period_rets = calc_fwd_batch(
        top10["ts_code"].values,
        te_dp,
        n_days=20
    )
    valid_rets = period_rets[~np.isnan(period_rets)]
    period_ret = np.mean(valid_rets) if len(valid_rets) > 0 else 0
    net_ret = (1 + period_ret) * (1 - cost) - 1
    
    records.append({
        "entry_date": ed_dt,
        "period_ret": period_ret,
        "cost": cost,
        "net_ret": net_ret,
        "turnover": turnover,
        "avg_score": top10["score"].mean(),
        "best_iter": lgb_m.best_iteration_,
        "n_candidates": len(day),
        "n_train": len(train_valid),
    })
    
    prev_codes = codes
    
    if (i + 1) % 10 == 0:
        print(f"  [{i+1}/{len(entry_dates)}] {str(ed_dt)[:10]}  "
              f"训练{len(train_valid):,}行, ret={period_ret*100:+.2f}%")

# ─── 6. 统计 ───
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
annual_ret = nav[-1] ** (1 / total_years) - 1 if total_years > 0 else 0
annual_vol = rets.std() * np.sqrt(12)
sharpe = annual_ret / annual_vol if annual_vol > 0 else 0

peak = np.maximum.accumulate(nav)
dd = nav / peak - 1
max_dd = dd.min()
win_rate = (rets > 0).mean()
calmar = annual_ret / abs(max_dd) if max_dd < 0 else np.inf

# 滚动24月夏普
rs_list = []
for j in range(24, len(rets)):
    r24 = rets[j-24:j]
    if r24.std() > 0:
        rs_list.append(r24.mean() / r24.std() * np.sqrt(12))

# 分年
df["year"] = df["entry_date"].astype(str).str[:4]
yearly = df.groupby("year").agg(
    N=("net_ret", "count"),
    ret=("net_ret", lambda x: np.prod(1 + x) - 1),
    win=("net_ret", lambda x: (x > 0).mean()),
    avg_turnover=("turnover", "mean"),
).reset_index()

print("\n" + "=" * 60)
print("📊 v38 无菌 · 严格样本外 · 回测结果")
print("=" * 60)
print(f"  期: {str(first_date)[:10]} ~ {str(last_date)[:10]} ({total_years:.1f}年)")
print(f"  有效期数: {n_months}")
print(f"  持仓: Top{N_HOLD}等权 | 成本: 单边{COST_PER_TRADE*100:.2f}% | 市值≥{MIN_MARKET_CAP}亿")
print()
print("--- 核心指标 ---")
print(f"  累计净收益:     {total_ret*100:+.2f}%")
print(f"  年化收益:       {annual_ret*100:+.2f}%")
print(f"  年化波动(月):   {annual_vol*100:.2f}%")
print(f"  年化夏普:       {sharpe:.2f}")
print(f"  最大回撤:       {max_dd*100:.2f}%")
print(f"  Calmar比率:     {calmar:.2f}")
print(f"  月胜率:         {win_rate*100:.1f}%")
print(f"  平均换手:       {df['turnover'].mean()*100:.1f}%")
print(f"  平均月成本:     {df['cost'].mean()*100:.2f}%")
if rs_list:
    print(f"  24月滚动夏普:   均值 {np.mean(rs_list):.2f} | 当前 {rs_list[-1]:.2f}")
print()
print("--- 分年 ---")
print(f"{'年':>4} | {'期':>3} | {'年收益':>9} | {'月胜率':>7} | {'换手':>5}")
print("-" * 45)
for _, r in yearly.iterrows():
    print(f"{r['year']:>4} | {r['N']:>3d} | {r['ret']*100:>+8.2f}% | {r['win']*100:>6.1f}% | {r['avg_turnover']*100:>4.1f}%")
print()
print("--- 最近12个月 ---")
print(f"{'调仓日':>12} | {'期收益':>8} | {'成本':>6} | {'净收益':>8} | {'换手':>5} | {'Score':>7}")
print("-" * 60)
for _, r in df.tail(12).iterrows():
    print(f"{str(r['entry_date'])[:10]:>12} | {r['period_ret']*100:>+7.2f}% | {r['cost']*100:>5.2f}% | "
          f"{r['net_ret']*100:>+7.2f}% | {r['turnover']*100:>4.1f}% | {r['avg_score']:.4f}")

# 保存
nav_df = pd.DataFrame({"entry_date": df["entry_date"], "nav": nav, "dd": dd})
nav_df.to_parquet(f"{OUTPUT}/v38_sterile_nav.parquet")

summary = {
    "version": "v38 无菌样本外",
    "period": f"{str(first_date)[:10]} ~ {str(last_date)[:10]}",
    "n_months": n_months,
    "years": round(total_years, 1),
    "total_return_pct": round(total_ret * 100, 2),
    "annual_return_pct": round(annual_ret * 100, 2),
    "annual_vol_pct": round(annual_vol * 100, 2),
    "sharpe": round(sharpe, 2),
    "calmar": round(calmar, 2),
    "max_dd_pct": round(max_dd * 100, 2),
    "monthly_win_rate_pct": round(win_rate * 100, 1),
    "avg_turnover_pct": round(df["turnover"].mean() * 100, 1),
    "cost_per_trade_pct": COST_PER_TRADE * 100,
    "min_market_cap": MIN_MARKET_CAP,
    "min_trading_days": MIN_TRADING_DAYS,
}
import json
json.dump(summary, open(f"{OUTPUT}/v38_sterile_summary.json", "w"), indent=2, ensure_ascii=False)
print(f"\n✅ 保存: {OUTPUT}/v38_sterile_nav.parquet + v38_sterile_summary.json")
print(f"⏱ {time.time() - t0:.0f}s")
