"""
多因子合成 (集成版, 直接用含fwd收益的面板)
方案1: 等权基准
方案2: EWMA-IC加权 (滚动IC)
方案3: 均值-方差最优加权

对比三种方案的回测收益曲线
"""
import os, sys, time
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DATA_FACTORS, BACKTEST_START

print("=" * 60)
print("多因子合成与回测对比")
print("=" * 60)

t0 = time.time()

# 加载含未来收益的面板
panel = pd.read_parquet(os.path.join(DATA_FACTORS, "factor_panel_with_fwd.parquet"))
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
panel = panel[panel["trade_date"] >= BACKTEST_START].copy()
print(f"面板: {len(panel):,} 条")

# 因子列 (排除ID和收益)
factor_cols = [c for c in panel.columns
               if c not in ("ts_code","trade_date","fwd_20d_ret")
               and panel[c].dtype in ("float64","int64")]
n_factors = len(factor_cols)
print(f"因子: {n_factors} 个: {factor_cols[:5]}...", flush=True)

# 月度取样 (用每个月最后一个交易日)
dates_all = sorted(panel["trade_date"].unique())
dates_monthly = pd.date_range(start=dates_all[0], end=dates_all[-1], freq="ME")
dates_monthly = [d for d in dates_monthly if d in dates_all]
# 如果月初不在交易日中, 取最近的
monthly_dates = []
for m in dates_monthly:
    d = m
    while d not in dates_all and d >= dates_all[0]:
        d -= pd.Timedelta(days=1)
    if d in dates_all:
        monthly_dates.append(d)
monthly_dates = list(set(monthly_dates))
monthly_dates.sort()
print(f"月度节点: {len(monthly_dates)} 个", flush=True)

# ======== 方案1: 等权基准 ========
print("\n[方案1] 等权组合...", flush=True)
weights_equal = dict(zip(factor_cols, [1/n_factors]*n_factors))

# ======== 方案2: EWMA-IC加权 ========
print("[方案2] EWMA-IC滚动加权...", flush=True)
# 预计算所有因子的滚动IC (每个月底)
ic_records = {f: [] for f in factor_cols}

window = 60  # 滚动IC窗口 (交易日)
for i, date in enumerate(monthly_dates):
    if i == 0:
        for f in factor_cols:
            ic_records[f].append({"date": date, "ic": 0.05})
        continue
    
    # 取过去window个交易日的数据计算IC
    past = monthly_dates[max(0, i-window//21):i]  # 约60个月度
    if len(past) < 6:
        for f in factor_cols:
            ic_records[f].append({"date": date, "ic": 0.05})
        continue
    
    past_data = panel[panel["trade_date"].isin(past)]
    
    for f in factor_cols:
        sub = past_data[["trade_date","ts_code",f,"fwd_20d_ret"]].dropna(subset=[f,"fwd_20d_ret"])
        if len(sub) < 200:
            ic_records[f].append({"date": date, "ic": np.nan})
            continue
        # 按日期计算每期IC再平均
        ic_vals = []
        for d in sub["trade_date"].unique():
            dd = sub[sub["trade_date"] == d]
            fv = dd[f].values
            rv = dd["fwd_20d_ret"].values
            if len(fv) < 30:
                continue
            # 去极值+rank IC
            lo, hi = np.nanpercentile(fv, [1, 99])
            fv = np.clip(fv, lo, hi)
            from scipy.stats import spearmanr
            try:
                ic, _ = spearmanr(fv, rv)
                ic_vals.append(ic)
            except:
                continue
        
        mean_ic = np.mean(ic_vals) if ic_vals else np.nan
        ic_records[f].append({"date": date, "ic": mean_ic})
    
    if (i + 1) % 24 == 0:
        print(f"  IC计算: {i+1}/{len(monthly_dates)} 月度", flush=True)

# EWMA权重
lambda_decay = 0.94
def calc_ewma_weights(ic_list, date, min_obs=6):
    """给定因子的IC历史, 用EWMA合成权重"""
    ics = [r["ic"] for r in ic_list if r["date"] <= date and not np.isnan(r["ic"])]
    if len(ics) < min_obs:
        return np.nan
    ics = np.array(ics[-120:])  # 最多120期
    weights = np.array([(1-lambda_decay) * lambda_decay ** (len(ics)-1-i) for i in range(len(ics))])
    weights /= weights.sum()
    return np.sum(weights * ics)

ewma_weights_hist = {}
for i, date in enumerate(monthly_dates):
    wts = {}
    for f in factor_cols:
        ewma_ic = calc_ewma_weights(ic_records[f], date)
        if np.isnan(ewma_ic):
            wts[f] = 1/n_factors
        else:
            wts[f] = max(ewma_ic, 0)  # 截断负权
    total = sum(wts.values())
    if total > 0:
        for f in wts:
            wts[f] /= total
    ewma_weights_hist[date] = wts

# ======== 方案3: 均值-方差最优 ========
print("[方案3] 均值-方差约束优化...", flush=True)
mv_weights_hist = {}
lambda_risk = 0.5  # 风险厌恶系数

def max_utility(weights, cov, mu):
    port_var = weights @ cov @ weights
    port_ret = weights @ mu
    return -(port_ret - 0.5 * lambda_risk * port_var)

for i, date in enumerate(monthly_dates):
    past = monthly_dates[max(0, i-12):i+1]  # 最近12个月
    if len(past) < 6:
        mv_weights_hist[date] = dict(zip(factor_cols, [1/n_factors]*n_factors))
        continue
    
    past_data = panel[panel["trade_date"].isin(past)]
    
    # 因子收益 (IC)
    mu = np.array([np.nanmean([r["ic"] for r in ic_records[f] if r["date"] in past]) for f in factor_cols])
    mu = np.nan_to_num(mu, 0)
    
    # 因子协方差矩阵 (用IC序列代替原始值)
    ic_matrix = np.zeros((len(past), n_factors))
    for ji, d in enumerate(past):
        ic_vals = []
        for f in factor_cols:
            recs = [r["ic"] for r in ic_records[f] if r["date"] == d]
            ic_vals.append(recs[0] if recs else 0)
        ic_matrix[ji, :] = ic_vals
    ic_matrix = np.nan_to_num(ic_matrix, 0)
    cov = np.cov(ic_matrix.T) + np.eye(n_factors) * 1e-6
    
    # 约束优化: sum(w)=1, w>=0
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
    bounds = [(0, 0.3)] * n_factors
    x0 = np.array([1/n_factors]*n_factors)
    
    try:
        result = minimize(max_utility, x0, args=(cov, mu), 
                         method="SLSQP", bounds=bounds, constraints=constraints,
                         options={"maxiter": 200, "ftol": 1e-8})
        w = result.x if result.success else x0
    except:
        w = x0
    
    w = np.maximum(w, 0)
    w /= w.sum()
    mv_weights_hist[date] = dict(zip(factor_cols, w))
    
    if (i + 1) % 24 == 0:
        print(f"  均值方差: {i+1}/{len(monthly_dates)}", flush=True)

# ======== 回测对比 ========
print("\n[回测] 三种方案收益对比...", flush=True)

def calc_portfolio_ret(weights_dict, monthly_date_list):
    """用月度权重计算组合收益"""
    pnl = []
    for i, date in enumerate(monthly_date_list):
        wts = weights_dict.get(date)
        if wts is None or i == len(monthly_date_list) - 1:
            continue
        
        # 当月持仓
        month_data = panel[panel["trade_date"] == date].copy()
        if month_data.empty:
            continue
        
        # 计算复合得分
        scores = np.zeros(len(month_data))
        for f, w in wts.items():
            if f in month_data.columns:
                fv = month_data[f].fillna(0).values
                # 截面标准化
                fv = (fv - np.nanmean(fv)) / max(np.nanstd(fv), 1e-10)
                fv = np.nan_to_num(fv, 0)
                scores += fv * w
        
        # 选股: top 30%
        n_top = max(len(scores) // 3, 50)
        top_idx = np.argsort(-scores)[:n_top]
        
        # 等权持有, 下个月收益
        next_date = monthly_date_list[i+1]
        next_data = panel[panel["trade_date"] == next_date]
        if next_data.empty:
            continue
        
        next_returns = []
        for idx in top_idx:
            code = month_data.iloc[idx]["ts_code"]
            nd = next_data[next_data["ts_code"] == code]
            if not nd.empty and nd["fwd_20d_ret"].iloc[0] is not None and not np.isnan(nd["fwd_20d_ret"].iloc[0]):
                next_returns.append(nd["fwd_20d_ret"].iloc[0])
        
        if next_returns:
            pnl.append(np.mean(next_returns))
    
    return np.array(pnl)

# 计算三种方案收益
rets_equal = calc_portfolio_ret(dict(zip(monthly_dates, [weights_equal]*len(monthly_dates))), monthly_dates)
rets_ewma = calc_portfolio_ret(ewma_weights_hist, monthly_dates)
rets_mv = calc_portfolio_ret(mv_weights_hist, monthly_dates)

def summary(rets, name):
    if len(rets) == 0:
        print(f"{name}: 无回测结果")
        return
    cum = np.cumprod(1 + rets)
    total_ret = cum[-1] - 1
    ann_ret = (cum[-1]) ** (12/len(rets)) - 1
    vol = np.std(rets) * np.sqrt(12)
    sharpe = np.mean(rets) / np.std(rets) * np.sqrt(12) if np.std(rets) > 0 else 0
    max_dd = np.maximum.accumulate(cum) - cum
    max_dd = max_dd.max()
    win_rate = np.mean(rets > 0)
    print(f"\n{name}:")
    print(f"  总收益: {total_ret*100:.1f}%")
    print(f"  年化收益: {ann_ret*100:.1f}%")
    print(f"  年化波动: {vol*100:.1f}%")
    print(f"  夏普比率: {sharpe:.2f}")
    print(f"  最大回撤: {max_dd*100:.1f}%")
    print(f"  月度胜率: {win_rate*100:.0f}%")

summary(rets_equal, "方案1: 等权基准")
summary(rets_ewma, "方案2: EWMA-IC加权")
summary(rets_mv, "方案3: 均值-方差最优")

print(f"\n总用时: {(time.time()-t0)/60:.1f} 分钟")
print("Done!")
