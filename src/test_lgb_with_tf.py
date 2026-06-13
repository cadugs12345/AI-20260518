"""
test_lgb_with_tf.py — 加入龙虎榜因子后的LGB回测
在79 base因子 + toplist_inst_buy 上训练LGB
对比：v31 LGB(79因子) vs LGB(79+龙虎榜)
"""
import os, sys, time, json, gc
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
os.chdir("/mnt/d/AI-20260518"); sys.path.insert(0, '.')
import joblib, lightgbm as lgb
from scipy.stats import spearmanr
tt = time.time()

STAMP, COMM, SLIP = 0.001, 0.0002, 0.002
print("="*60, flush=True)
print("LGB + 龙虎榜因子 回测", flush=True)
print(time.strftime('%F %H:%M'), flush=True)
print("="*60, flush=True)

# 数据
panel = pd.read_parquet("data/factors/factor_panel_v6.parquet")
prices = pd.read_parquet("data/factors/full_prices.parquet")
prices["trade_date"] = pd.to_datetime(prices["trade_date"])
panel["trade_date"] = pd.to_datetime(panel["trade_date"])

# 合并龙虎榜
tf = pd.read_parquet("data/factors/toplist_factors.parquet")
tf["trade_date"] = pd.to_datetime(tf["trade_date"])
# 只取机构买入比率
tf_simp = tf[["ts_code","trade_date","toplist_inst_buy","toplist_net_rate","toplist_inst_net","toplist_freq_20d"]].copy()
# 填缺失（龙虎榜只有有上榜记录的才有数据）
panel = panel.merge(tf_simp, on=["ts_code","trade_date"], how="left")

# 资金流因子
mf = pd.read_parquet("data/factors/moneyflow_factors_v2.parquet")
mf = mf[["ts_code","trade_date","elg_net_ratio_ma10","elg_net_ratio_ma5","net_lg"]].copy()
panel = panel.merge(mf, on=["ts_code","trade_date"], how="left")

ps = prices.sort_values(["ts_code","trade_date"]).copy()
ps["ret_1d"] = ps.groupby("ts_code")["close"].pct_change()
ps["vol_60d"] = ps.groupby("ts_code")["ret_1d"].transform(lambda x: x.rolling(60, min_periods=20).std())
ps["vol_60d_ann"] = ps["vol_60d"] * np.sqrt(244)

pdts = sorted(panel["trade_date"].unique())
period_dates = [pdts[i] for i in range(0, len(pdts), 20) if pdts[i] >= pd.Timestamp("2021-01-01")]

half_dates = []
for d in period_dates:
    k = f"{d.year}{'H1' if d.month<=6 else 'H2'}"
    if not half_dates or half_dates[-1][0]!=k: half_dates.append((k,d))
    else: half_dates[-1]=(k,d)

import tushare as ts
from config.settings import TS_TOKEN
ts.set_token(TS_TOKEN); pro=ts.pro_api()
stk=pro.query("stock_basic",exchange="",list_status="L",fields="ts_code,industry")
si=dict(zip(stk["ts_code"],stk["industry"]))

rf_md=joblib.load("models/ml_ensemble_v1.joblib")
rf_factors=rf_md["factor_cols"]

# 基础因子 + 新因子
new_factors = ["toplist_inst_buy","toplist_net_rate","toplist_inst_net","toplist_freq_20d",
               "elg_net_ratio_ma10","elg_net_ratio_ma5","net_lg"]
all_factors = rf_factors + [f for f in new_factors if f in panel.columns]

print(f"\n基础因子: {len(rf_factors)}", flush=True)
print(f"新增因子: {[f for f in new_factors if f in panel.columns]}", flush=True)
print(f"总因子: {len(all_factors)}", flush=True)

def rolling_lgb(features, label=""):
    preds = {"trade_date":[],"ts_code":[],"pred_ret":[]}
    for hi,(hk,train_cutoff) in enumerate(half_dates):
        if hi < 3: continue
        train_end=train_cutoff-pd.Timedelta(days=5)
        tr=panel[(panel["trade_date"]>=train_end-pd.Timedelta(days=730))&(panel["trade_date"]<=train_end)]
        tr=tr[tr["fwd_20d_ret"].notna()&(tr["fwd_20d_ret"].abs()<0.5)]
        if len(tr)<20000: continue
        if len(tr)>100000: tr=tr.sample(100000,random_state=42)
        
        X_tr=tr[features].fillna(0).values.astype(np.float32)
        y_tr=np.clip(tr["fwd_20d_ret"].values.astype(np.float32),-0.3,0.3)
        nv=max(1,int(len(tr)*0.15))
        
        m=lgb.LGBMRegressor(n_estimators=500,max_depth=3,lr=0.02,
            subsample=0.7,colsample_bytree=0.7,reg_alpha=0.2,reg_lambda=1.0,
            min_child_weight=20,min_data_in_leaf=100,random_state=42,verbose=-1,n_jobs=8)
        m.fit(X_tr[:-nv],y_tr[:-nv],eval_set=[(X_tr[-nv:],y_tr[-nv:])],
              callbacks=[lgb.early_stopping(30, verbose=False)],eval_metric="mse")
        
        for d in period_dates:
            if d<=train_cutoff: continue
            day=panel[panel["trade_date"]==d]
            if len(day)==0: continue
            X_te=day[features].fillna(0).values.astype(np.float32)
            pp=m.predict(X_te)
            for j,code in enumerate(day["ts_code"].values):
                preds["trade_date"].append(d); preds["ts_code"].append(code); preds["pred_ret"].append(float(pp[j]))
        
        if (hi+1)%3==0: print(f"  {label}: {hk} ({hi+1}/{len(half_dates)-3})", flush=True)
    pdf=pd.DataFrame(preds); print(f"  {label}: {len(pdf):,}预测, {pdf['trade_date'].nunique()}期", flush=True)
    return pdf

def backtest_ml(pred_df, n_stocks=30, target_vol=0.15, label=""):
    pred_dates=sorted(pred_df["trade_date"].unique())
    pred_dates=[d for d in pred_dates if d>=pd.Timestamp("2023-01-01")]
    if len(pred_dates)<2: return None
    cash=0.03; holdings={}; navs=[1.0]
    for i in range(len(pred_dates)-1):
        date=pred_dates[i]; sell_date=pred_dates[i+1]
        px_buy={}; px_sell={}; sv={}
        for _,r in ps[ps["trade_date"]==date].iterrows():
            px_buy[r["ts_code"]]=r["close"]
            sv[r["ts_code"]]=r["vol_60d_ann"] if pd.notna(r.get("vol_60d_ann")) else 0.3
        for _,r in ps[ps["trade_date"]==sell_date].iterrows():
            px_sell[r["ts_code"]]=r["close"]
        hv=sum(shares*px_buy.get(c,0) for c,shares in holdings.items()); tv_=hv+cash
        sp=0
        for c,shares in holdings.items():
            px=px_sell.get(c,0)
            if px>0: sp+=shares*px - shares*px*(STAMP+COMM+SLIP)
        cash+=sp; holdings={}
        dp_=pred_df[pred_df["trade_date"]==date].sort_values("pred_ret",ascending=False)
        selected=list(dp_.head(n_stocks)["ts_code"].values)
        sv_l=[sv.get(c,np.nan) for c in selected]; sv_l=[v for v in sv_l if not np.isnan(v) and v>0.01]
        pr=max(min(target_vol/np.median(sv_l),1.0) if len(sv_l)>=5 else 1.0,0.05)
        if selected and cash>0.001:
            al=cash*pr*0.98
            if al>0.001:
                per=al/len(selected)
                for code in selected:
                    px=px_buy.get(code,0)
                    if px>0 and per>0:
                        b=(per-per*(COMM+SLIP))/px
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

# 训练
print(f"\n基准 LGB({len(rf_factors)}因子)...", flush=True)
base_pred = rolling_lgb(rf_factors, "基准")

print(f"\nLGB+龙虎榜({len(all_factors)}因子)...", flush=True)
new_pred = rolling_lgb(all_factors, "LGB+TF")

# IC
for pdf, label in [(base_pred,"基准"),(new_pred,"LGB+TF")]:
    ics=[]
    for d in pdf["trade_date"].unique():
        day=pdf[pdf["trade_date"]==d]
        pday=panel[panel["trade_date"]==d]
        m=day.merge(pday[["ts_code","fwd_20d_ret"]],on="ts_code")
        if len(m)>10:
            ic,_=spearmanr(m["pred_ret"],m["fwd_20d_ret"])
            if not np.isnan(ic): ics.append(ic)
    if ics:
        print(f"IC {label:10s}: {np.mean(ics)*100:+7.2f}% IR: {np.mean(ics)/np.std(ics):.2f} ({len(ics)}期)", flush=True)

# 回测
print(f"\n{'='*60}", flush=True)
print("回测: T30 目波15% 2023-2026", flush=True)
print("-"*60, flush=True)

backtest_ml(base_pred, 30, 0.15, "v31基准 LGB(79因子)")
backtest_ml(new_pred, 30, 0.15, "v34 LGB+龙虎榜+资金流")

# 权重回测（指数衰减+行业中性）
def backtest_weighted(pred_df, n_stocks=30, target_vol=0.15, label=""):
    pred_dates=sorted(pred_df["trade_date"].unique())
    pred_dates=[d for d in pred_dates if d>=pd.Timestamp("2023-01-01")]
    if len(pred_dates)<2: return None
    cash=0.03; holdings={}; navs=[1.0]
    for i in range(len(pred_dates)-1):
        date=pred_dates[i]; sell_date=pred_dates[i+1]
        px_buy={}; px_sell={}; sv={}
        for _,r in ps[ps["trade_date"]==date].iterrows():
            px_buy[r["ts_code"]]=r["close"]; sv[r["ts_code"]]=r["vol_60d_ann"] if pd.notna(r.get("vol_60d_ann")) else 0.3
        for _,r in ps[ps["trade_date"]==sell_date].iterrows():
            px_sell[r["ts_code"]]=r["close"]
        hv=sum(shares*px_buy.get(c,0) for c,shares in holdings.items()); tv_=hv+cash
        sp=0
        for c,shares in holdings.items():
            px=px_sell.get(c,0)
            if px>0: sp+=shares*px - shares*px*(STAMP+COMM+SLIP)
        cash+=sp; holdings={}
        dp_=pred_df[pred_df["trade_date"]==date].sort_values("pred_ret",ascending=False)
        codes=list(dp_["ts_code"]); scores=dp_["pred_ret"].values
        # 行业中性
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

print(f"\n--- 指数衰减+行业中性 ---", flush=True)
backtest_weighted(base_pred, 30, 0.15, "v31(指衰+行业) LGB=79")
backtest_weighted(new_pred, 30, 0.15, "v34(指衰+行业) LGB+7")

# 保存预测
base_pred.to_parquet("data/factors/pred_v34_base.parquet", index=False)
new_pred.to_parquet("data/factors/pred_v34_new.parquet", index=False)
print(f"\n预测已保存", flush=True)

# 有加权的版本（指数衰减+行业中性，用v27的回测引擎改一下似乎太复杂，先看等权的）
print(f"\n{'='*60}", flush=True)
print(f"⏱ {(time.time()-tt)/60:.1f}分", flush=True)
