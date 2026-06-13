#!/usr/bin/env python3
"""
backtest_v35_event.py — 事件驱动增强策略
架构：双层
- 底层：LGB滚动预测（79因子）
- 上层：事件信号（龙虎榜、断板修复等）触发调仓

版本策略：
v35a: LGB选Top30 + 龙虎榜机构净买入>0时加仓（等权，事件日买入+20%权重）
v35b: LGB选Top30 + 龙虎榜机构净买入>0的替换掉排名最低的
v35c: LGB选Top30 + 龙虎榜机构净买入排名前10的直接入选

基准：v31等权 LGB(79因子)
"""
import os, sys, time, json
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
os.chdir("/mnt/d/AI-20260518"); sys.path.insert(0, '.')
import joblib, lightgbm as lgb
from scipy.stats import spearmanr
tt = time.time()

STAMP, COMM, SLIP = 0.001, 0.0002, 0.002
print("="*60, flush=True); print("v35 事件驱动增强回测", flush=True); print(time.strftime('%F %H:%M'), flush=True); print("="*60, flush=True)

# 数据
panel = pd.read_parquet("data/factors/factor_panel_v6.parquet")
prices = pd.read_parquet("data/factors/full_prices.parquet")
panel["trade_date"] = pd.to_datetime(panel["trade_date"]); prices["trade_date"] = pd.to_datetime(prices["trade_date"])

# 事件信号
tf = pd.read_parquet("data/factors/toplist_factors.parquet")
tf["trade_date"] = pd.to_datetime(tf["trade_date"])
# 龙虎榜：机构净买入>=基准值=强信号
tf["tf_event"] = (tf["toplist_inst_net"] > 5).astype(int)  # 机构净买入>5%
tf["tf_strong"] = (tf["toplist_inst_buy"] > tf["toplist_inst_buy"].quantile(0.7)).astype(int)

panel = panel.merge(tf[["ts_code","trade_date","tf_event","tf_strong","toplist_inst_buy","toplist_inst_net"]], on=["ts_code","trade_date"], how="left")

ps = prices.sort_values(["ts_code","trade_date"]).copy()
ps["ret_1d"] = ps.groupby("ts_code")["close"].pct_change()
ps["vol_60d"] = ps.groupby("ts_code")["ret_1d"].transform(lambda x: x.rolling(60, min_periods=20).std())
ps["vol_60d_ann"] = ps["vol_60d"] * np.sqrt(244)

pdts = sorted(panel["trade_date"].unique())
period_dates = [pdts[i] for i in range(0, len(pdts), 20) if pdts[i] >= pd.Timestamp("2021-01-01")]

half_dates = []
for d in period_dates:
    k=f"{d.year}{'H1' if d.month<=6 else 'H2'}"
    if not half_dates or half_dates[-1][0]!=k: half_dates.append((k,d))
    else: half_dates[-1]=(k,d)

rf_md=joblib.load("models/ml_ensemble_v1.joblib"); rf_factors=rf_md["factor_cols"]

# 滚动训练LGB
def train_lgb_predict():
    preds={"trade_date":[],"ts_code":[],"pred_ret":[]}
    for hi,(hk,train_cutoff) in enumerate(half_dates):
        if hi<3: continue
        train_end=train_cutoff-pd.Timedelta(days=5)
        tr=panel[(panel["trade_date"]>=train_end-pd.Timedelta(days=730))&(panel["trade_date"]<=train_end)]
        tr=tr[tr["fwd_20d_ret"].notna()&(tr["fwd_20d_ret"].abs()<0.5)]
        if len(tr)<20000: continue
        if len(tr)>100000: tr=tr.sample(100000,random_state=42)
        X_tr=tr[rf_factors].fillna(0).values.astype(np.float32)
        y_tr=np.clip(tr["fwd_20d_ret"].values.astype(np.float32),-0.3,0.3)
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

print(f"\n滚动训练LGB...", flush=True)
lgb_preds = train_lgb_predict()

# 合并事件数据
lgb_preds = lgb_preds.merge(panel[["ts_code","trade_date","tf_event","tf_strong"]], on=["ts_code","trade_date"], how="left")
lgb_preds["tf_event"] = lgb_preds["tf_event"].fillna(0).astype(int)
lgb_preds["tf_strong"] = lgb_preds["tf_strong"].fillna(0).astype(int)

# IC
ics=[]; panel_ic=panel[["ts_code","trade_date","fwd_20d_ret"]]
for d in lgb_preds["trade_date"].unique():
    day=lgb_preds[lgb_preds["trade_date"]==d]
    mday=panel_ic[panel_ic["trade_date"]==d]
    m=day.merge(mday,on="ts_code")
    if len(m)>10:
        ic,_=spearmanr(m["pred_ret"],m["fwd_20d_ret"])
        if not np.isnan(ic): ics.append(ic)
if ics: print(f"IC: {np.mean(ics)*100:+.2f}% IR: {np.mean(ics)/np.std(ics):.2f} ({len(ics)}期)", flush=True)

# ========== 回测引擎 ==========
def backtest_event(pred_df, n_stocks=30, target_vol=0.15, label="", events=None, strategy="standard", min_date=pd.Timestamp("2023-01-01")):
    """
    events: list of event strategies
      "standard": LGB选Top30（无事件）
      "boost": LGB选Top30，有tf_event的股票额外加%权重
      "replace": LGB选Top30，有tf_event的替换掉排名最低的
      "priority": 有tf_event的股票优先入选
      "strong": 只选tf_strong的股票
    """
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
        
        day=pred_df[pred_df["trade_date"]==date].sort_values("pred_ret",ascending=False).copy()
        event_codes=set(day[day["tf_event"]==1]["ts_code"].values)
        strong_codes=set(day[day["tf_strong"]==1]["ts_code"].values)
        all_codes=list(day["ts_code"].values)
        
        if strategy=="standard":
            selected=all_codes[:n_stocks]
        elif strategy=="boost":
            # 标准Top30 + 有事件信号的额外加到最后（超出30但权重调整）
            selected=all_codes[:n_stocks]
            for code in event_codes:
                if code not in selected and len(selected)<n_stocks*1.5:
                    selected.append(code)
            selected=selected[:n_stocks]
        elif strategy=="replace":
            # 有事件的替换掉排名最低的
            top=all_codes[:n_stocks]
            event_in_top=[c for c in top if c in event_codes]
            non_event_ranked=[c for c in top if c not in event_codes]
            event_outside=[c for c in all_codes[n_stocks:] if c in event_codes]
            n_replace=min(len(event_outside), len(non_event_ranked))
            selected=event_in_top+non_event_ranked[:-n_replace] if n_replace>0 else top
            selected+=event_outside[:n_replace] if n_replace>0 else []
        elif strategy=="priority":
            # 有事件的先选，剩下的按LGB排名补齐
            event_selected=[c for c in all_codes if c in event_codes][:n_stocks]
            remaining=[c for c in all_codes if c not in event_selected]
            selected=event_selected+remaining[:n_stocks-len(event_selected)]
        elif strategy=="strong":
            strong_selected=[c for c in all_codes if c in strong_codes][:n_stocks]
            remaining=[c for c in all_codes if c not in strong_selected]
            selected=strong_selected+remaining[:max(0,n_stocks-len(strong_selected))]
        else:
            selected=all_codes[:n_stocks]
        
        sv_l=[sv.get(c,np.nan) for c in selected]
        sv_l=[v for v in sv_l if not np.isnan(v) and v>0.01]
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

print(f"\n{'='*60}", flush=True)
print("回测: T30 目波15% 2023-2026", flush=True)
print("-"*60, flush=True)

backtest_event(lgb_preds, 30, 0.15, "v31基准 LGB(79因子)", strategy="standard")
backtest_event(lgb_preds, 30, 0.15, "v35a 事件加仓(boost)", strategy="boost")
backtest_event(lgb_preds, 30, 0.15, "v35b 事件替换(replace)", strategy="replace")
backtest_event(lgb_preds, 30, 0.15, "v35c 事件优先(priority)", strategy="priority")
backtest_event(lgb_preds, 30, 0.15, "v35d 强信号优先(strong)", strategy="strong")

# 统计事件覆盖
event_days=lgb_preds[lgb_preds["tf_event"]==1]["trade_date"].nunique()
event_rows=len(lgb_preds[lgb_preds["tf_event"]==1])
print(f"\n事件覆盖: {event_days}天, {event_rows}条, {event_rows/len(lgb_preds)*100:.2f}%")

print(f"\n{'='*60}", flush=True)
print(f"✅ 完成 | ⏱ {(time.time()-tt)/60:.1f}分", flush=True)
