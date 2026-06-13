#!/usr/bin/env python3
"""
v42 历史版本去重对比

原因：5/22发现滚动训练preds重复bug（重复率86%），导致v27/v29/v31夏普虚高
这里用统一回测框架+去重逻辑，对比3个历史版本的真实表现。

由于无法精确复现旧模型的训练方式，这里用统一方法（LGB rank标签）统一回测，
看不同组合方式（等权/指衰/行业中性）的真实差异。
"""
import os, sys, json, time
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
os.chdir("/mnt/d/AI-20260518"); sys.path.insert(0, '.')
import joblib, lightgbm as lgb
from scipy.stats import spearmanr
import tushare as ts; from config.settings import TS_TOKEN
ts.set_token(TS_TOKEN); pro = ts.pro_api()
stk = pro.query("stock_basic", exchange="", list_status="L", fields="ts_code,industry")
si = dict(zip(stk["ts_code"], stk["industry"]))
tt = time.time()

STAMP, COMM, SLIP = 0.001, 0.0002, 0.002
print("="*60, flush=True)
print("v42 历史版本去重对比 (统一LGB+rank标签)", flush=True)
print(time.strftime('%F %H:%M'), flush=True)
print("="*60, flush=True)

# 数据
panel = pd.read_parquet("data/factors/factor_panel_v6.parquet")
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
panel["ts_code"] = panel["ts_code"].astype(str)
prices = pd.read_parquet("data/factors/full_prices.parquet")
prices["trade_date"] = pd.to_datetime(prices["trade_date"])
prices["ret_1d"] = prices.groupby("ts_code")["close"].pct_change()
ps = prices.sort_values(["ts_code","trade_date"]).copy()
ps["vol_60d_ann"] = ps.groupby("ts_code")["ret_1d"].transform(lambda x: x.rolling(60, min_periods=20).std()) * np.sqrt(244)

# rank标签
mask = panel["fwd_20d_ret"].notna() & (panel["fwd_20d_ret"].abs() < 0.5)
panel["label_rank"] = np.nan
panel.loc[mask, "label_rank"] = panel[mask].groupby("trade_date")["fwd_20d_ret"].rank(pct=True, ascending=True)

pdts = sorted(panel["trade_date"].unique())
period_dates = [pdts[i] for i in range(0, len(pdts), 20) if pdts[i] >= pd.Timestamp("2021-01-01")]
half_dates = []
for d in period_dates:
    k = f"{d.year}{'H1' if d.month<=6 else 'H2'}"
    if not half_dates or half_dates[-1][0] != k: half_dates.append((k,d))
    else: half_dates[-1]=(k,d)

rf_md = joblib.load("models/ml_ensemble_v1.joblib")
factor_cols = rf_md["factor_cols"]

# 训练LGB
print("\n训练LGB (rank标签)...", flush=True)
preds = {"trade_date":[], "ts_code":[], "pred_ret":[], "fwd_20d_ret":[]}
for hi,(hk,train_cutoff) in enumerate(half_dates):
    if hi<3: continue
    train_end = train_cutoff - pd.Timedelta(days=5)
    tr = panel[(panel["trade_date"]>=train_end-pd.Timedelta(days=730))&(panel["trade_date"]<=train_end)&panel["label_rank"].notna()].copy()
    if len(tr)<20000: continue
    if len(tr)>100000: tr=tr.sample(100000,random_state=42)
    X_tr=tr[factor_cols].fillna(0).values.astype(np.float32)
    y_tr=tr["label_rank"].values.astype(np.float32)
    nv=max(1,int(len(tr)*0.15))
    m=lgb.LGBMRegressor(n_estimators=500,max_depth=3,lr=0.02,subsample=0.7,colsample_bytree=0.7,
        reg_alpha=0.2,reg_lambda=1.0,min_child_weight=20,min_data_in_leaf=100,random_state=42,verbose=-1,n_jobs=8)
    m.fit(X_tr[:-nv],y_tr[:-nv],eval_set=[(X_tr[-nv:],y_tr[-nv:])],callbacks=[lgb.early_stopping(30,verbose=False)],eval_metric="mse")
    for d in period_dates:
        if d<=train_cutoff: continue
        day = panel[panel["trade_date"]==d]
        if len(day)==0: continue
        pp = m.predict(day[factor_cols].fillna(0).values.astype(np.float32))
        for j,code in enumerate(day["ts_code"].values):
            preds["trade_date"].append(d); preds["ts_code"].append(code)
            preds["pred_ret"].append(float(pp[j]))
            preds["fwd_20d_ret"].append(float(day.iloc[j].get("fwd_20d_ret",np.nan)))
    if (hi+1)%3==0: print(f"  {hk} ({hi+1}/{len(half_dates)-3})", flush=True)

pdf = pd.DataFrame(preds)
pdf = pdf.drop_duplicates(subset=["trade_date","ts_code"], keep="first")
print(f"预测: {len(pdf):,}行, {pdf['trade_date'].nunique()}期", flush=True)


# ====== 回测引擎（支持各种组合） ======
def bt(pdf, label="", weighted=False, ind_neutral=False, risk=False,
       min_date=pd.Timestamp("2023-01-01"), n_stocks=30, target_vol=0.15):
    """通用回测"""
    pred_dates = sorted(pdf["trade_date"].unique())
    pred_dates = [d for d in pred_dates if d>=min_date]
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
        day = pdf[pdf["trade_date"]==date].sort_values("pred_ret",ascending=False).reset_index(drop=True)
        codes=list(day["ts_code"]);
        # 建入选列表
        sel=[]; ic={}
        order = list(range(len(codes)))
        # 风控
        if risk:
            r10 = [day.iloc[j].get("repair_force_10d",np.nan) for j in range(len(day))]
            hv_ = [day.iloc[j].get("高波反转",np.nan) for j in range(len(day))]
            safe = [j for j in range(len(day)) if not (
                (not np.isnan(r10[j]) and r10[j]<-0.05) or
                (not np.isnan(hv_[j]) and hv_[j]<-0.03))]
        else:
            safe = list(range(len(day)))
        for j in safe:
            ind = si.get(codes[j],"其他")
            if ind_neutral and ic.get(ind,0)>=3: continue
            if ind_neutral: ic[ind]=ic.get(ind,0)+1
            sel.append(j)
            if len(sel)>=n_stocks: break
        if len(sel)<n_stocks:
            for j in safe:
                if j not in sel: sel.append(j)
                if len(sel)>=n_stocks: break
        sel_codes = [codes[j] for j in sel]
        if weighted:
            rw=np.arange(1,len(sel_codes)+1); w=np.exp(-0.1*rw); w=w/w.sum()
        else:
            w=np.ones(len(sel_codes))/len(sel_codes)
        sl=[sv.get(c,np.nan) for c in sel_codes]; sl=[v for v in sl if not np.isnan(v) and v>0.01]
        pr=max(min(target_vol/np.median(sl),1.0) if len(sl)>=5 else 1.0,0.05)
        if sel_codes and cash>0.001:
            al=cash*pr*0.98
            if al>0.001:
                for code,wt in zip(sel_codes,w):
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
    print(f"  {label:32s}: 年化{ar*100:+7.1f}% | 夏普{sr:5.2f} | "
          f"回撤{mdd*100:5.1f}% | 胜率{wr*100:3.0f}% | 卡玛{cal:.2f} | {len(pnl)}期", flush=True)
    return {"ar":float(ar),"sr":float(sr),"mdd":float(mdd),"wr":float(wr),"cal":float(cal)}


# 加风控因子
panel2 = pd.read_parquet("data/factors/factor_panel_v6.parquet",
    columns=["ts_code","trade_date","repair_force_10d","高波反转"])
panel2["trade_date"] = pd.to_datetime(panel2["trade_date"])
pdf = pdf.merge(panel2, on=["ts_code","trade_date"], how="left")

# ====== 回测对比 ======
print(f"\n{'='*60}", flush=True)
print("历史版本去重对比 (统一LGB+rank标签, 2023-2026)", flush=True)
print(f"{'='*60}", flush=True)

cfgs = [
    # (名称, 权重, 行业中性, 风控)
    ("v12 等权(纯因子式)",      False, False, False),
    ("v27 RF+风控(等权)",      False, False, True),
    ("v29 LGB(等权+行业中性)", False, True,  False),
    ("v31 LGB+指衰+行业中性",  True,  True,  False),
    ("v38 rank+指衰+行业+风控", True,  True,  True),
]

results = {}
for name, weighted, ind_neutral, risk in cfgs:
    r = bt(pdf, name, weighted=weighted, ind_neutral=ind_neutral, risk=risk)
    if r: results[name] = r

# 汇总
print(f"\n{'='*60}", flush=True)
print("最终结果：", flush=True)
print(f"{'配置':32s} {'年化':>7s} {'夏普':>6s} {'回撤':>6s} {'胜率':>4s} {'卡玛':>5s}", flush=True)
print("-"*60, flush=True)
for name in [k for k in cfgs]:
    n = name[0]
    if n in results:
        r = results[n]
        print(f"{n:32s} {r['ar']*100:>6.1f}% {r['sr']:>5.2f} "
              f"{r['mdd']*100:>5.1f}% {r['wr']*100:>3.0f}% {r['cal']:>4.2f}", flush=True)

json.dump(results, open("output/backtest_v42_history.json","w"), indent=2, default=str)
print(f"\n✅ 完成 | ⏱ {(time.time()-tt)/60:.1f}分", flush=True)
print(f"  结果: output/backtest_v42_history.json", flush=True)
