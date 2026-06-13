#!/usr/bin/env python3
"""
backtest_v37_rank_weighted.py — v36c rank标签 + v31指数衰减权重+行业中性

把最好的两个方向结合：
- 标签: fwd_20d_rank (收益分位排位)
- 权重: 指数衰减(e^{-0.1*r}) + 行业中性(每行业最多3只)
- ML: LGB 500棵滚动训练

对比：
1. v31基准（等权20d收益 + 指衰+行业）
2. v37a（rank + 等权分配）
3. v37b（rank + 指衰+行业）
4. v37c（rank + 指衰+行业 + capping 15%）
"""
import os, sys, time, json
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
os.chdir("/mnt/d/AI-20260518"); sys.path.insert(0, '.')
import joblib, lightgbm as lgb
from scipy.stats import spearmanr
tt = time.time()

STAMP, COMM, SLIP = 0.001, 0.0002, 0.002
print("="*60, flush=True); print("v37 rank标签 + 指数衰减+行业中性", flush=True); print(time.strftime('%F %H:%M'), flush=True); print("="*60, flush=True)

import tushare as ts
from config.settings import TS_TOKEN
ts.set_token(TS_TOKEN); pro=ts.pro_api()
stk=pro.query("stock_basic", exchange="", list_status="L", fields="ts_code,industry")
si=dict(zip(stk["ts_code"], stk["industry"]))

panel = pd.read_parquet("data/factors/factor_panel_v6.parquet")
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
prices = pd.read_parquet("data/factors/full_prices.parquet")
prices["trade_date"] = pd.to_datetime(prices["trade_date"])

ps = prices.sort_values(["ts_code","trade_date"]).copy()
ps["ret_1d"] = ps.groupby("ts_code")["close"].pct_change()
ps["vol_60d"] = ps.groupby("ts_code")["ret_1d"].transform(lambda x: x.rolling(60, min_periods=20).std())
ps["vol_60d_ann"] = ps["vol_60d"] * np.sqrt(244)

# rank标签
panel["fwd_20d_rank"] = panel.groupby("trade_date")["fwd_20d_ret"].transform(
    lambda x: (x.rank() - 1) / (len(x) - 1) if len(x) > 1 else 0
)

pdts = sorted(panel["trade_date"].unique())
period_dates = [pdts[i] for i in range(0, len(pdts), 20) if pdts[i] >= pd.Timestamp("2021-01-01")]

half_dates = []
for d in period_dates:
    k=f"{d.year}{'H1' if d.month<=6 else 'H2'}"
    if not half_dates or half_dates[-1][0]!=k: half_dates.append((k,d))
    else: half_dates[-1]=(k,d)

rf_md=joblib.load("models/ml_ensemble_v1.joblib"); rf_factors=rf_md["factor_cols"]

def train_lgb_rank():
    preds={"trade_date":[],"ts_code":[],"pred_ret":[]}
    for hi,(hk,train_cutoff) in enumerate(half_dates):
        if hi<3: continue
        train_end=train_cutoff-pd.Timedelta(days=5)
        tr=panel[(panel["trade_date"]>=train_end-pd.Timedelta(days=730))&(panel["trade_date"]<=train_end)]
        tr=tr[tr["fwd_20d_rank"].notna()].copy()
        if len(tr)<20000: continue
        if len(tr)>100000: tr=tr.sample(100000,random_state=42)
        X_tr=tr[rf_factors].fillna(0).values.astype(np.float32)
        y_tr=tr["fwd_20d_rank"].values.astype(np.float32)
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
        if (hi+1)%3==0: print(f"  {hk} ({hi+1}/{len(half_dates)-3})", flush=True)
    pdf=pd.DataFrame(preds)
    print(f"  总预测: {len(pdf):,}, {pdf['trade_date'].nunique()}期", flush=True)
    return pdf

print("\n滚动训练LGB(rank标签)...", flush=True)
preds = train_lgb_rank()

# IC
ics=[]
for d in preds["trade_date"].unique():
    day=preds[preds["trade_date"]==d]
    pday=panel[panel["trade_date"]==d]
    m=day.merge(pday[["ts_code","fwd_20d_ret"]],on="ts_code")
    if len(m)>10:
        ic,_=spearmanr(m["pred_ret"],m["fwd_20d_ret"])
        if not np.isnan(ic): ics.append(ic)
if ics: print(f"IC vs 等权收益: {np.mean(ics)*100:+.2f}% IR: {np.mean(ics)/np.std(ics):.2f} ({len(ics)}期)", flush=True)

# ====== 回测引擎：等权和指衰+行业 ======
def backtest_ml(pred_df, n_stocks=30, target_vol=0.15, label="", min_date=pd.Timestamp("2023-01-01"),
                weighted=False, industry_cap=False):
    pred_dates=sorted(pred_df["trade_date"].unique())
    pred_dates=[d for d in pred_dates if d>=min_date]
    if len(pred_dates)<2: return None
    cash=0.03; holdings={}; navs=[1.0]
    
    for i in range(len(pred_dates)-1):
        date=pred_dates[i]; sell_date=pred_dates[i+1]
        px_buy={}; px_sell={}; sv={}
        for _,r in ps[ps["trade_date"]==date].iterrows():
            px_buy[r["ts_code"]]=r["close"]
            sv[r["ts_code"]]=r["vol_60d_ann"] if pd.notna(r.get("vol_60d_ann")) else 0.3
        for _,r in ps[ps["trade_date"]==sell_date].iterrows(): px_sell[r["ts_code"]]=r["close"]
        
        hv=sum(shares*px_buy.get(c,0) for c,shares in holdings.items()); tv_=hv+cash
        sp=0
        for c,shares in holdings.items():
            px=px_sell.get(c,0)
            if px>0: sp+=shares*px - shares*px*(STAMP+COMM+SLIP)
        cash+=sp; holdings={}
        
        day=pred_df[pred_df["trade_date"]==date].sort_values("pred_ret",ascending=False).copy()
        
        if industry_cap and weighted:
            # 行业中性 + 指数衰减
            codes=list(day["ts_code"]); scores=day["pred_ret"].values
            sel=[]; ic={}
            order=np.argsort(-scores)
            for j in order:
                ind=si.get(codes[j],"其他")
                if ic.get(ind,0)<3: sel.append(codes[j]); ic[ind]=ic.get(ind,0)+1
                if len(sel)>=n_stocks: break
            selected=sel
        else:
            selected=list(day.head(n_stocks)["ts_code"].values)
        
        sv_l=[sv.get(c,np.nan) for c in selected]; sv_l=[v for v in sv_l if not np.isnan(v) and v>0.01]
        pr=max(min(target_vol/np.median(sv_l),1.0) if len(sv_l)>=5 else 1.0,0.05)
        
        if selected and cash>0.001:
            al=cash*pr*0.98
            if al>0.001:
                if weighted:
                    r=np.arange(1,len(selected)+1)
                    w=np.exp(-0.1*r); w=w/w.sum()
                    for code,wt in zip(selected,w):
                        px=px_buy.get(code,0)
                        if px>0 and wt>0:
                            b=(al*wt)*(1-COMM-SLIP)/px
                            if b>0: holdings[code]=b
                else:
                    per=al/len(selected)
                    for code in selected:
                        px=px_buy.get(code,0)
                        if px>0 and per>0:
                            b=(per-per*(COMM+SLIP))/px
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

print(f"\n{'='*60}", flush=True)
print("回测: T30 目波15% 2023-2026", flush=True)
print("-"*60, flush=True)

# 基准
backtest_ml(preds, 30, 0.15, "v31基准(20d等权+等权分配)", min_date=pd.Timestamp("2023-01-01"), weighted=False, industry_cap=False)
# rank+等权
backtest_ml(preds, 30, 0.15, "v37a rank+等权分配", min_date=pd.Timestamp("2023-01-01"), weighted=False, industry_cap=False)
# rank+指衰+行业
backtest_ml(preds, 30, 0.15, "v37b rank+指衰+行业中", min_date=pd.Timestamp("2023-01-01"), weighted=True, industry_cap=True)

print(f"\n{'='*60}", flush=True)
print(f"✅ 完成 | ⏱ {(time.time()-tt)/60:.1f}分", flush=True)
