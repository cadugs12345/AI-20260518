"""
多因子合成回测 (精简版, 快速验证)
"""
import os, sys, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DATA_FACTORS

t0 = time.time()
print("加载面板...", flush=True)
panel = pd.read_parquet(os.path.join(DATA_FACTORS, "factor_panel_with_fwd.parquet"))
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
panel = panel[panel["trade_date"] >= "2018-01-01"].copy()

factor_cols = [c for c in panel.columns if c not in ("ts_code","trade_date","fwd_20d_ret") and panel[c].dtype in ("float64","int64")]
n_f = len(factor_cols)
print(f"面板: {len(panel):,} 条, 因子: {n_f} 个", flush=True)

# 按月节点
dates = sorted(panel["trade_date"].unique())
m_dates = sorted(set(pd.Series(dates).dt.to_period("M").unique()))
monthly = sorted(set(
    panel[panel["trade_date"].dt.to_period("M").isin(m_dates)]
    .groupby(panel["trade_date"].dt.to_period("M"))["trade_date"].max().tolist()
))
print(f"月度: {len(monthly)} 个节点 ({monthly[0].date()} ~ {monthly[-1].date()})", flush=True)

# 预计算各因子截面IC (每个月)
print("计算因子月度IC...", flush=True)
ic_df_list = []
for i, date in enumerate(monthly):
    day = panel[panel["trade_date"] == date].dropna(subset=[factor_cols[0],"fwd_20d_ret"])
    if len(day) < 100: continue
    ic_row = {"trade_date": date}
    for f in factor_cols:
        sub = panel[panel["trade_date"] == date][[f,"fwd_20d_ret"]].dropna()
        if len(sub) < 50:
            ic_row[f] = np.nan
            continue
        fv, rv = sub[f].values, sub["fwd_20d_ret"].values
        lo, hi = np.nanpercentile(fv, [1,99])
        fv = np.clip(fv, lo, hi)
        ic_row[f] = spearmanr(fv, rv)[0] if len(fv) > 10 else np.nan
    ic_df_list.append(ic_row)
    if (i+1) % 20 == 0:
        print(f"  IC: {i+1}/{len(monthly)}", flush=True)

ic_df = pd.DataFrame(ic_df_list)
print(f"IC表: {len(ic_df)} 期 × {len(factor_cols)} 因子", flush=True)

# ===== 权重方案 =====
# 等权
w_equal = dict(zip(factor_cols, [1/n_f]*n_f))

# EWMA-IC
w_ewma = {}
ld = 0.94
for idx, row in ic_df.iterrows():
    wts = {}
    for f in factor_cols:
        hist = ic_df[f].values[:idx+1]
        hist = hist[~np.isnan(hist)]
        if len(hist) < 3:
            wts[f] = 1/n_f
        else:
            hist = hist[-60:]
            w = np.array([(1-ld)*ld**(len(hist)-1-j) for j in range(len(hist))])
            w /= w.sum()
            ew = np.nansum(w * hist)
            wts[f] = max(ew, 0) if not np.isnan(ew) else 0
    total = sum(wts.values())
    if total > 0:
        for f in wts: wts[f] /= total
    w_ewma[row["trade_date"]] = wts

# 均值-方差
w_mv = {}
lr = 0.5
for idx, row in ic_df.iterrows():
    date = row["trade_date"]
    m_idx = monthly.index(date)
    past = monthly[max(0, m_idx-12):m_idx+1]
    mu = np.array([ic_df[ic_df["trade_date"].isin(past)][f].mean() for f in factor_cols])
    mu = np.nan_to_num(mu, 0)
    cov = np.cov(np.nan_to_num(ic_df.iloc[max(0,idx-12):idx+1][factor_cols].values.copy(), 0).T) + np.eye(n_f)*1e-6
    
    def obj(w):
        return -(w @ mu - 0.5 * lr * w @ cov @ w)
    
    cons = [{"type": "eq", "fun": lambda x: np.sum(x)-1}]
    bounds = [(0, 0.3)] * n_f
    x0 = np.ones(n_f)/n_f
    try:
        res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter":200})
        w = res.x if res.success else x0
    except:
        w = x0
    w = np.maximum(w, 0); w /= w.sum()
    w_mv[date] = dict(zip(factor_cols, w))
    if (idx+1) % 20 == 0:
        print(f"  均值方差: {idx+1}/{len(ic_df)}", flush=True)

# ===== 回测 =====
def backtest(weight_func, name):
    pnl, cum = [], [1.0]
    dates_list = sorted(w_ewma.keys()) if name != "等权" else monthly
    
    for i, date in enumerate(dates_list):
        if i == len(dates_list)-1: continue
        wts = weight_func(date)
        if wts is None: continue
        
        day = panel[panel["trade_date"] == date]
        if day.empty: continue
        
        scores = np.zeros(len(day))
        for f, w in wts.items():
            if f not in day.columns: continue
            fv = day[f].fillna(0).values
            fv = (fv - fv.mean()) / max(fv.std(), 1e-10)
            scores += np.nan_to_num(fv, 0) * w
        
        n_top = max(len(scores)//3, 50)
        top_idx = np.argsort(-scores)[:n_top]
        
        ndate = dates_list[i+1]
        nday = panel[panel["trade_date"] == ndate]
        if nday.empty: continue
        
        rets = []
        for idx in top_idx:
            code = day.iloc[idx]["ts_code"]
            nd = nday[nday["ts_code"] == code]
            if not nd.empty and not np.isnan(nd["fwd_20d_ret"].iloc[0]):
                rets.append(nd["fwd_20d_ret"].iloc[0])
        
        if rets:
            r = np.mean(rets)
            pnl.append(r)
            cum.append(cum[-1] * (1+r))
    
    if not pnl:
        print(f"\n{name}: 无结果")
        return
    
    pnl = np.array(pnl)
    cum = np.array(cum)
    tr = cum[-1] - 1
    ar = (cum[-1])**(12/len(pnl)) - 1
    vol = np.std(pnl)*np.sqrt(12)
    sr = np.mean(pnl)/np.std(pnl)*np.sqrt(12)
    dd = np.maximum.accumulate(cum) - cum
    mdd = dd.max()
    wr = np.mean(pnl>0)
    
    print(f"\n{name}:")
    print(f"  总收益: {tr*100:.1f}% | 年化: {ar*100:.1f}% | 年化波动: {vol*100:.1f}%")
    print(f"  夏普: {sr:.2f} | 最大回撤: {mdd*100:.1f}% | 月度胜率: {wr*100:.0f}%")

print("\n===== 回测结果 =====", flush=True)
backtest(lambda d: w_equal, "等权基准")
backtest(lambda d: w_ewma.get(d), "EWMA-IC加权")
backtest(lambda d: w_mv.get(d), "均值-方差最优")

print(f"\n总用时: {(time.time()-t0)/60:.1f} 分钟", flush=True)
