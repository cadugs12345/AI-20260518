#!/usr/bin/env python3
"""
backtest_v36_label.py — 换标签体系回测

标签对比：
1. v31: fwd_20d_ret (等权20日收益)
2. v36a: fwd_20d_decay (指数衰减收益, e^{-0.1*t})
3. v36b: fwd_20d_sharpe (20日夏普 = 收益/波动率)
4. v36c: fwd_20d_rank (20日收益rank, 二分类导向)

回测框架：LGB 500棵 + 滚动训练 + 等权T30
"""
import os, sys, time, json, gc
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
os.chdir("/mnt/d/AI-20260518"); sys.path.insert(0, '.')
import joblib, lightgbm as lgb
from scipy.stats import spearmanr
tt = time.time()

STAMP, COMM, SLIP = 0.001, 0.0002, 0.002
print("="*60, flush=True); print("v36 换标签体系回测", flush=True); print(time.strftime('%F %H:%M'), flush=True); print("="*60, flush=True)

# ====== 数据 ======
panel = pd.read_parquet("data/factors/factor_panel_v6.parquet")
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
panel["ts_code"] = panel["ts_code"].astype(str)

# 需要日收益率数据来计算衰减标签
prices = pd.read_parquet("data/factors/full_prices.parquet")
prices["trade_date"] = pd.to_datetime(prices["trade_date"])
prices = prices.sort_values(["ts_code", "trade_date"])
prices["ret_1d"] = prices.groupby("ts_code")["close"].pct_change()

# 日收益率面板
ret_panel = prices[["ts_code","trade_date","ret_1d"]].dropna(subset=["ret_1d"]).copy()

print(f"日收益率范围: {ret_panel['trade_date'].min().date()} ~ {ret_panel['trade_date'].max().date()}", flush=True)

# ====== 构建新标签 ======
print("\n构建新标签...", flush=True)

# 等权20日收益（已有）
panel["fwd_20d_ret"] = panel["fwd_20d_ret"].astype(float)

# 已有 fwd_20d_ret 是等权20日收益，用已有的 pred_20d 数据
# 快速方法：只重新加权，不需要重新对齐
# 用panel已有数据算出衰减和夏普标签

# 1. 等权收益rank（0~1）
print("  计算收益rank...", flush=True)
panel["fwd_20d_rank"] = panel.groupby("trade_date")["fwd_20d_ret"].transform(
    lambda x: (x.rank() - 1) / (len(x) - 1) if len(x) > 1 else 0
)

# 2. 指数衰减收益 — 用numpy加速版
# 对每只股票，用rolling + shift技巧
print("  计算指数衰减收益...", flush=True)
alpha = np.log(2) / 20  # half-life = 20

# 预计算20日衰减权重
w = np.exp(-alpha * np.arange(1, 21))
w = w / w.sum()

def decay_ret_for_code(code_df):
    """对一只股票，用convolve算每个日期的衰减收益"""
    rets = code_df["ret_1d"].values.astype(np.float64)
    n = len(rets)
    if n < 25:
        return np.full(n, np.nan)
    # 用numpy的convolve算移动加权和
    # 需要反转权重，因为convolve是正序
    # conv = np.convolve(rets, w[::-1], mode='full')
    # 但这覆盖所有lag，需要取未来部分
    # 更简单：对每个i，rets[i+1:i+21] * w
    result = np.full(n, np.nan)
    for i in range(n - 1):
        end = min(i + 21, n)
        slice_len = end - i - 1
        if slice_len < 5: continue
        w_ = w[:slice_len] / w[:slice_len].sum()
        result[i] = (rets[i+1:end] * w_).sum()
    return result

# 用groupby apply
print(f"  面板: {len(ret_panel):,}行", flush=True)
decay = ret_panel.sort_values("trade_date").groupby("ts_code", group_keys=False).apply(
    lambda g: pd.Series(decay_ret_for_code(g), index=g.index, name="fwd_20d_decay")
)
print(f"  衰减收益计算完成: {decay.notna().sum():,}条", flush=True)
panel["fwd_20d_decay"] = decay

# 3. 20日夏普
print("  计算20日夏普...", flush=True)
def sharpe_for_code(code_df):
    rets = code_df["ret_1d"].values.astype(np.float64)
    n = len(rets)
    if n < 25:
        return np.full(n, np.nan)
    result = np.full(n, np.nan)
    for i in range(n - 1):
        end = min(i + 21, n)
        slice_len = end - i - 1
        if slice_len < 5: continue
        r = rets[i+1:end]
        vol = np.std(r) * np.sqrt(20)
        if vol < 1e-8: continue
        profit = (1 + r).prod() - 1
        result[i] = profit / vol
    return result

sharpe = ret_panel.sort_values("trade_date").groupby("ts_code", group_keys=False).apply(
    lambda g: pd.Series(sharpe_for_code(g), index=g.index, name="fwd_20d_sharpe")
)
print(f"  夏普计算完成: {sharpe.notna().sum():,}条", flush=True)
panel["fwd_20d_sharpe"] = sharpe

# 3. 等权收益rank（0~1）
panel["fwd_20d_rank"] = panel.groupby("trade_date")["fwd_20d_ret"].transform(
    lambda x: (x.rank() - 1) / (len(x) - 1) if len(x) > 1 else 0
)

# 检查标签质量
for label in ["fwd_20d_ret","fwd_20d_decay","fwd_20d_sharpe","fwd_20d_rank"]:
    v = panel[label].dropna()
    print(f"  {label:20s}: n={len(v):,}, 均值={v.mean():+.5f}, std={v.std():.5f}", flush=True)

# ====== 回测参数 ======
ps = prices.sort_values(["ts_code","trade_date"]).copy()
ps["vol_60d_ann"] = ps.groupby("ts_code")["ret_1d"].transform(lambda x: x.rolling(60, min_periods=20).std()) * np.sqrt(244)

pdts = sorted(panel["trade_date"].unique())
period_dates = [pdts[i] for i in range(0, len(pdts), 20) if pdts[i] >= pd.Timestamp("2021-01-01")]

half_dates = []
for d in period_dates:
    k=f"{d.year}{'H1' if d.month<=6 else 'H2'}"
    if not half_dates or half_dates[-1][0]!=k: half_dates.append((k,d))
    else: half_dates[-1]=(k,d)

rf_md=joblib.load("models/ml_ensemble_v1.joblib"); rf_factors=rf_md["factor_cols"]

# ====== 滚动训练（单个标签） ======
def train_lgb_with_label(label_col, name=""):
    """用特定标签训练LGB并回测"""
    preds={"trade_date":[],"ts_code":[],"pred_ret":[]}
    for hi,(hk,train_cutoff) in enumerate(half_dates):
        if hi<3: continue
        train_end=train_cutoff-pd.Timedelta(days=5)
        tr=panel[(panel["trade_date"]>=train_end-pd.Timedelta(days=730))&(panel["trade_date"]<=train_end)]
        tr=tr[tr[label_col].notna()].copy()
        # 去掉极端值
        if label_col != "fwd_20d_rank":  # rank已经是0~1
            lb=tr[label_col]
            lo,hi=lb.quantile(0.01),lb.quantile(0.99)
            tr=tr[(tr[label_col]>=lo)&(tr[label_col]<=hi)]
        if len(tr)<20000: continue
        if len(tr)>100000: tr=tr.sample(100000,random_state=42)
        X_tr=tr[rf_factors].fillna(0).values.astype(np.float32)
        y_tr=tr[label_col].values.astype(np.float32)
        nv=max(1,int(len(tr)*0.15))
        m=lgb.LGBMRegressor(n_estimators=500,max_depth=3,lr=0.02,subsample=0.7,colsample_bytree=0.7,
            reg_alpha=0.2,reg_lambda=1.0,min_child_weight=20,min_data_in_leaf=100,random_state=42,verbose=-1,n_jobs=8)
        m.fit(X_tr[:-nv],y_tr[:-nv],eval_set=[(X_tr[-nv:],y_tr[-nv:])],callbacks=[lgb.early_stopping(30,verbose=False)],eval_metric="mse")
        for d in period_dates:
            if d<=train_cutoff: continue
            day=panel[panel["trade_date"]==d]
            if len(day)==0: continue
            pp=m.predict(day[rf_factors].fillna(0).values.astype(np.float32))
            for j,code in enumerate(day["ts_code"].values):
                preds["trade_date"].append(d); preds["ts_code"].append(code); preds["pred_ret"].append(float(pp[j]))
        if (hi+1)%3==0: print(f"  {name}: {hk} ({hi+1}/{len(half_dates)-3})", flush=True)
    pdf=pd.DataFrame(preds)
    print(f"  {name}: {len(pdf):,}预测, {pdf['trade_date'].nunique()}期", flush=True)
    return pdf

# ====== 回测引擎 ======
def backtest_ml(pred_df, n_stocks=30, target_vol=0.15, label="", min_date=pd.Timestamp("2023-01-01")):
    pred_dates=sorted(pred_df["trade_date"].unique())
    pred_dates=[d for d in pred_dates if d>=min_date]
    if len(pred_dates)<2: return None
    cash=0.03; holdings={}; navs=[1.0]
    for i in range(len(pred_dates)-1):
        date=pred_dates[i]; sell_date=pred_dates[i+1]
        px_buy={}; px_sell={}; sv={}
        for _,r in ps[ps["trade_date"]==date].iterrows():
            px_buy[r["ts_code"]]=r["close"]; sv[r["ts_code"]]=r["vol_60d_ann"] if pd.notna(r.get("vol_60d_ann")) else 0.3
        for _,r in ps[ps["trade_date"]==sell_date].iterrows(): px_sell[r["ts_code"]]=r["close"]
        hv=sum(shares*px_buy.get(c,0) for c,shares in holdings.items()); tv_=hv+cash
        sp=0
        for c,shares in holdings.items():
            px=px_sell.get(c,0)
            if px>0: sp+=shares*px - shares*px*(STAMP+COMM+SLIP)
        cash+=sp; holdings={}
        day=pred_df[pred_df["trade_date"]==date].sort_values("pred_ret",ascending=False)
        selected=list(day.head(n_stocks)["ts_code"].values)
        sv_l=[sv.get(c,np.nan) for c in selected]; sv_l=[v for v in sv_l if not np.isnan(v) and v>0.01]
        pr=max(min(target_vol/np.median(sv_l),1.0) if len(sv_l)>=5 else 1.0,0.05)
        if selected and cash>0.001:
            al=cash*pr*0.98
            if al>0.001:
                per=al/len(selected)
                for code in selected:
                    px=px_buy.get(code,0)
                    if px>0 and per>0: b=(per-per*(COMM+SLIP))/px
                    if b>0: holdings[code]=b
                cash-=per*len(holdings)
        np_=sum(shares*px_sell.get(c,0) for c,shares in holdings.items())
        nt=np_+cash; ret=nt/tv_-1 if tv_>0 else 0
        navs.append(navs[-1]*(1+ret))
    pnl=np.array(navs[1:])/np.array(navs[:-1])-1; na=np.array(navs)
    ny=len(pnl)/13; ar=na[-1]**(1/ny)-1 if ny>0 and na[-1]>0 else 0
    sr=np.mean(pnl)/np.std(pnl)*np.sqrt(13) if np.std(pnl)>0 else 0
    dd=np.maximum.accumulate(na)-na; mdd=dd.max(); wr=np.mean(pnl>0); cal=ar/mdd if mdd>0 else 0
    print(f"  {label:32s}: 年化{ar*100:+7.1f}% | 夏普{sr:5.2f} | 回撤{mdd*100:5.1f}% | 胜率{wr*100:3.0f}% | 卡玛{cal:.2f} | {len(pnl)}期", flush=True)
    return {"ret":f"{ar*100:+.1f}%","sr":f"{sr:.2f}"}

# ====== 执行 ======
labels = [
    ("fwd_20d_ret", "v31 等权20d收益"),
    ("fwd_20d_decay", "v36a 指数衰减收益"),
    ("fwd_20d_sharpe", "v36b 20日夏普"),
    ("fwd_20d_rank", "v36c 收益rank"),
]

results = {}
for lcol, lname in labels:
    print(f"\n--- {lname} ---", flush=True)
    pp = train_lgb_with_label(lcol, lname)
    
    # IC vs fwd_20d_ret
    ics=[]
    for d in pp["trade_date"].unique():
        day=pp[pp["trade_date"]==d]
        pday=panel[panel["trade_date"]==d]
        m=day.merge(pday[["ts_code","fwd_20d_ret"]],on="ts_code")
        if len(m)>10:
            ic,_=spearmanr(m["pred_ret"],m["fwd_20d_ret"])
            if not np.isnan(ic): ics.append(ic)
    if ics:
        print(f"  IC vs 等权收益: {np.mean(ics)*100:+.2f}% IR: {np.mean(ics)/np.std(ics):.2f} ({len(ics)}期)", flush=True)
    
    results[lname] = pp

print(f"\n{'='*60}", flush=True)
print("回测: T30 目波15% 2023-2026", flush=True)
print("-"*60, flush=True)

for lname, pp in results.items():
    backtest_ml(pp, 30, 0.15, lname)

# ========== v37 rank + 指衰+行业 ==========
import tushare as ts
from config.settings import TS_TOKEN
ts.set_token(TS_TOKEN); pro=ts.pro_api()
stk=pro.query("stock_basic", exchange="", list_status="L", fields="ts_code,industry")
si=dict(zip(stk["ts_code"], stk["industry"]))

print(f"\n--- v37 rank+指衰+行业中性 ---", flush=True)
rank_preds = results["v36c 收益rank"]

def backtest_weighted(pred_df, n_stocks=30, target_vol=0.15, label="", min_date=pd.Timestamp("2023-01-01")):
    pred_dates=sorted(pred_df["trade_date"].unique())
    pred_dates=[d for d in pred_dates if d>=min_date]
    if len(pred_dates)<2: return None
    cash=0.03; holdings={}; navs=[1.0]
    for i in range(len(pred_dates)-1):
        date=pred_dates[i]; sell_date=pred_dates[i+1]
        px_buy={}; px_sell={}; sv={}
        for _,r in ps[ps["trade_date"]==date].iterrows():
            px_buy[r["ts_code"]]=r["close"]; sv[r["ts_code"]]=r["vol_60d_ann"] if pd.notna(r.get("vol_60d_ann")) else 0.3
        for _,r in ps[ps["trade_date"]==sell_date].iterrows(): px_sell[r["ts_code"]]=r["close"]
        hv=sum(shares*px_buy.get(c,0) for c,shares in holdings.items()); tv_=hv+cash
        sp=0
        for c,shares in holdings.items():
            px=px_sell.get(c,0)
            if px>0: sp+=shares*px - shares*px*(STAMP+COMM+SLIP)
        cash+=sp; holdings={}
        dp_=pred_df[pred_df["trade_date"]==date].sort_values("pred_ret",ascending=False)
        codes=list(dp_["ts_code"]); scores=dp_["pred_ret"].values
        sel=[]; ic={}
        order=np.argsort(-scores)
        for j in order:
            ind=si.get(codes[j],"其他")
            if ic.get(ind,0)<3: sel.append(codes[j]); ic[ind]=ic.get(ind,0)+1
            if len(sel)>=n_stocks: break
        r=np.arange(1,len(sel)+1); w=np.exp(-0.1*r); w=w/w.sum()
        sl=[sv.get(c,np.nan) for c in sel]; sl=[v for v in sl if not np.isnan(v) and v>0.01]
        pr=max(min(target_vol/np.median(sl),1.0) if len(sl)>=5 else 1.0,0.05)
        if sel and cash>0.001:
            al=cash*pr*0.98
            if al>0.001:
                for code,wt in zip(sel,w):
                    px=px_buy.get(code,0)
                    if px>0 and wt>0:
                        b=(al*wt)*(1-COMM-SLIP)/px
                        if b>0: holdings[code]=b
                cash-=al
        np_=sum(shares*px_sell.get(c,0) for c,shares in holdings.items())
        nt=np_+cash; ret=nt/tv_-1 if tv_>0 else 0
        navs.append(navs[-1]*(1+ret))
    pnl=np.array(navs[1:])/np.array(navs[:-1])-1; na=np.array(navs)
    ny=len(pnl)/13; ar=na[-1]**(1/ny)-1 if ny>0 and na[-1]>0 else 0
    sr=np.mean(pnl)/np.std(pnl)*np.sqrt(13) if np.std(pnl)>0 else 0
    dd=np.maximum.accumulate(na)-na; mdd=dd.max(); wr=np.mean(pnl>0); cal=ar/mdd if mdd>0 else 0
    print(f"  {label:32s}: 年化{ar*100:+7.1f}% | 夏普{sr:5.2f} | 回撤{mdd*100:5.1f}% | 胜率{wr*100:3.0f}% | 卡玛{cal:.2f} | {len(pnl)}期", flush=True)
    return {"ret":f"{ar*100:+.1f}%","sr":f"{sr:.2f}"}

# 等权对比
backtest_ml(rank_preds, 30, 0.15, "rank+等权")
backtest_weighted(rank_preds, 30, 0.15, "v37 rank+指衰+行业中性")

print(f"\n{'='*60}", flush=True)
print(f"✅ 完成 | ⏱ {(time.time()-tt)/60:.1f}分", flush=True)
