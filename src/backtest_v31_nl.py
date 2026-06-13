"""
backtest_v31_nl.py — 非线性衍生回测
基础: v29 LGB 79因子
对比: v31 NL LGB 43因子 (23基础+20非线性)
"""
import os, sys, time, json, gc
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
os.chdir("/mnt/d/AI-20260518"); sys.path.insert(0, '.')
import joblib, lightgbm as lgb
from scipy.stats import spearmanr
tt = time.time()

print("="*60, flush=True)
print("v31 NL回测: LGB基础 vs LGB+非线性", flush=True)
print(f"{time.strftime('%F %H:%M')}", flush=True)
print("="*60, flush=True)

# 读数据
panel = pd.read_parquet("data/factors/factor_panel_v6.parquet")
nl = pd.read_parquet("data/factors/auto_nl_features.parquet")

fac_base = [c for c in panel.columns if c not in ["ts_code","trade_date","fwd_20d_ret","close","ret_1d",
    "__fragment_index","__batch_index","__last_in_fragment","__filename","board_break"]]

# 拼接
panel_orig = panel.copy()
# 只加非基础列
nl_cols = [c for c in nl.columns if c not in fac_base]
panel = pd.concat([panel, nl[nl_cols].astype(np.float32)], axis=1)
del panel_orig, nl; gc.collect()

# 加载特征
af = json.load(open("models/auto_features_lgb.json"))
nl_sel = [c for c in af["nl"] if c in nl_cols]
base_sel = af["base"]
sel_features = base_sel + nl_sel
print(f"特征: {len(sel_features)} ({len(base_sel)}基础+{len(nl_sel)}NL)", flush=True)

# 价格
prices = pd.read_parquet("data/factors/full_prices.parquet")
prices["trade_date"] = pd.to_datetime(prices["trade_date"])
panel["trade_date"] = pd.to_datetime(panel["trade_date"])

ps = prices.sort_values(["ts_code","trade_date"]).copy()
ps["ret_1d"] = ps.groupby("ts_code")["close"].pct_change()
ps["vol_60d"] = ps.groupby("ts_code")["ret_1d"].transform(lambda x: x.rolling(60, min_periods=20).std())
ps["vol_60d_ann"] = ps["vol_60d"] * np.sqrt(244)

pdts = sorted(panel["trade_date"].unique())
period_dates = [pdts[i] for i in range(0, len(pdts), 20) if pdts[i] >= pd.Timestamp("2021-01-01")]

price_map, vol_map = {}, {}
for d in period_dates:
    s = prices[prices["trade_date"]==d][["ts_code","close"]].dropna()
    price_map[d] = dict(zip(s["ts_code"], s["close"]))
    v = ps[ps["trade_date"]==d][["ts_code","vol_60d_ann"]].dropna()
    vol_map[d] = dict(zip(v["ts_code"], v["vol_60d_ann"]))

# 半年度滚动训练
half_dates = []
for d in period_dates:
    k = f"{d.year}{'H1' if d.month<=6 else 'H2'}"
    if not half_dates or half_dates[-1][0]!=k: half_dates.append((k,d))
    else: half_dates[-1]=(k,d)

def run_rolling(features, label=""):
    preds = {"trade_date":[],"ts_code":[],"pred_ret":[]}
    for hi,(hk,train_cutoff) in enumerate(half_dates):
        if hi<3: continue
        train_end = train_cutoff - pd.Timedelta(days=5)
        tr = panel[(panel["trade_date"]>=train_end-pd.Timedelta(days=730)) & (panel["trade_date"]<=train_end)]
        tr = tr[tr["fwd_20d_ret"].notna() & (tr["fwd_20d_ret"].abs()<0.5)]
        if len(tr)<20000: continue
        if len(tr)>100000: tr=tr.sample(100000,random_state=42)
        
        X_tr = tr[features].fillna(0).values.astype(np.float32)
        y_tr = np.clip(tr["fwd_20d_ret"].values.astype(np.float32),-0.3,0.3)
        nv = max(1,int(len(tr)*0.15))
        
        m = lgb.LGBMRegressor(**dict(n_estimators=500,max_depth=3,lr=0.02,
            subsample=0.7,colsample_bytree=0.7,reg_alpha=0.2,reg_lambda=1.0,
            min_child_weight=20,min_data_in_leaf=100,random_state=42,verbose=-1,n_jobs=8))
        m.fit(X_tr[:-nv],y_tr[:-nv],eval_set=[(X_tr[-nv:],y_tr[-nv:])],
              callbacks=[lgb.early_stopping(30,verbose=False)],eval_metric="mse")
        
        for d in period_dates:
            if d<=train_cutoff: continue
            idx = panel["trade_date"]==d
            day = panel[idx]
            if len(day)==0: continue
            X_te = day[features].fillna(0).values.astype(np.float32)
            pp = m.predict(X_te)
            for j, code in enumerate(day["ts_code"].values):
                preds["trade_date"].append(d)
                preds["ts_code"].append(code)
                preds["pred_ret"].append(float(pp[j]))
        if (hi+1)%5==0:
            print(f"  {label}: {hi+1}/{len(half_dates)-3}期", flush=True)
    return pd.DataFrame(preds)

STAMP, COMM, SLIP = 0.001, 0.0002, 0.002

def bt(pdf, n=30, tv=0.15, label="", start_date=None):
    pds = sorted(pdf["trade_date"].unique())
    if start_date: pds=[d for d in pds if d>=start_date]
    if len(pds)<2: return None
    cash, hold, navs = 0.03, {}, [1.0]
    for i in range(len(pds)-1):
        d,sd = pds[i],pds[i+1]
        pb,psm = price_map.get(d,{}),price_map.get(sd,{})
        sv = vol_map.get(d,{})
        hv = sum(shares*pb.get(c,0) for c,shares in hold.items())
        tv_ = hv+cash; sp=0
        for c,shares in hold.items():
            px = psm.get(c,0)
            if px>0:
                v=shares*px
                sp+=v-v*(STAMP+COMM+SLIP)
        cash+=sp; hold={}
        dp_ = pdf[pdf["trade_date"]==d].sort_values("pred_ret",ascending=False)
        sel=list(dp_.head(n)["ts_code"])
        vs_=[sv.get(c,np.nan) for c in sel if not np.isnan(sv.get(c,np.nan)) and sv[c]>0.01]
        pr=max(min(tv/np.median(vs_),1.0) if len(vs_)>=5 else 1.0,0.05)
        if sel and cash>0.001:
            av=cash*pr*0.98
            if av>0.001:
                per=av/len(sel)
                for c in sel:
                    px=pb.get(c,0)
                    if px>0 and per>0:
                        b=(per-per*(COMM+SLIP))/px
                        if b>0: hold[c]=b
                cash-=per*len(hold)
        np_=sum(shares*psm.get(c,0) for c,shares in hold.items())
        nt=np_+cash
        r=nt/tv_-1 if tv_>0 else 0
        navs.append(navs[-1]*(1+r))
    pnl=np.array(navs[1:])/np.array(navs[:-1])-1
    na=np.array(navs); ny=len(pnl)/13
    ar=na[-1]**(1/ny)-1 if ny>0 and na[-1]>0 else 0
    sr=np.mean(pnl)/np.std(pnl)*np.sqrt(13) if np.std(pnl)>0 else 0
    dd=np.maximum.accumulate(na)-na; mdd=dd.max()
    wr=np.mean(pnl>0)
    print(f"  {label:22s}: 年化{ar*100:+7.1f}% | 夏普{sr:5.2f} | 回撤{mdd*100:5.1f}% | 胜率{wr*100:3.0f}% | {len(pnl)}期", flush=True)
    return {"ret":f"{ar*100:+.1f}%","sr":f"{sr:.2f}","mdd":f"{mdd*100:.1f}%","num":len(pnl)}

# 基础RF因子
rf_md = joblib.load("models/ml_ensemble_v1.joblib")
rf_factors = rf_md["factor_cols"]

# 滚动预测：两项
print(f"\n基线 LGB({len(rf_factors)}因子)...", flush=True)
base_p = run_rolling(rf_factors, "基准")

print(f"\n新 LGB({len(sel_features)}因子)...", flush=True)
nl_p = run_rolling(sel_features, "NL+")

# IC
for pdf,label in [(base_p,"基准"),(nl_p,"NL+")]:
    ics=[]
    for d in pdf["trade_date"].unique():
        day=pdf[pdf["trade_date"]==d]
        pday=panel[panel["trade_date"]==d]
        m=day.merge(pday[["ts_code","fwd_20d_ret"]],on="ts_code")
        if len(m)>10:
            ic,_=spearmanr(m["pred_ret"],m["fwd_20d_ret"])
            ics.append(ic)
    if ics:
        print(f"  {label} IC: {np.mean(ics)*100:+.2f}% IR: {np.mean(ics)/np.std(ics):.2f} ({len(ics)}期)", flush=True)

# 回测
print(f"\n{'='*60}", flush=True)
print("回测对比: T30 目波15% 2021-2026", flush=True)
print("-"*60, flush=True)

for pdf,label in [(base_p,"LGB(79因子)"),(nl_p,"LGB+NL(43)")]:
    bt(pdf,30,0.15,label)
    
print(f"\n{'='*60}", flush=True)
print(f"✅ 完成")
print(f"⏱ {(time.time()-tt)/60:.1f}分", flush=True)
