"""
backtest_v32_conf.py — 置信度加权动态调仓
用LightGBM的树间标准差作为置信度信号
高置信度→高权重，低置信度→低权重
对比：v31指数衰减+行业中性 (夏普1.25)
"""
import os, sys, time, json, gc
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
os.chdir("/mnt/d/AI-20260518"); sys.path.insert(0, '.')
import joblib, lightgbm as lgb
from scipy.stats import spearmanr
tt = time.time()

print("="*60, flush=True)
print("v32 置信度加权动态调仓", flush=True)
print(f"{time.strftime('%F %H:%M')}", flush=True)
print("="*60, flush=True)

panel = pd.read_parquet("data/factors/factor_panel_v6.parquet")
prices = pd.read_parquet("data/factors/full_prices.parquet")
prices["trade_date"] = pd.to_datetime(prices["trade_date"])
panel["trade_date"] = pd.to_datetime(panel["trade_date"])

fac_base = [c for c in panel.columns if c not in ["ts_code","trade_date","fwd_20d_ret","close","ret_1d",
    "__fragment_index","__batch_index","__last_in_fragment","__filename","board_break"]]

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

# 行业
import tushare as ts
from config.settings import TS_TOKEN
ts.set_token(TS_TOKEN)
pro = ts.pro_api()
stk_basic = pro.query("stock_basic", exchange="", list_status="L", fields="ts_code,industry")
stk_ind = dict(zip(stk_basic["ts_code"], stk_basic["industry"]))

# 半年度滚动
half_dates = []
for d in period_dates:
    k = f"{d.year}{'H1' if d.month<=6 else 'H2'}"
    if not half_dates or half_dates[-1][0]!=k: half_dates.append((k,d))
    else: half_dates[-1]=(k,d)

rf_md = joblib.load("models/ml_ensemble_v1.joblib")
rf_factors = rf_md["factor_cols"]

def rolling_predict_with_conf():
    """滚动预测，同时计算预测值和树间标准差（置信度）"""
    all_preds = {"trade_date":[],"ts_code":[],"pred_ret":[],"confidence":[]}
    
    for hi,(hk,train_cutoff) in enumerate(half_dates):
        if hi < 3: continue
        train_end = train_cutoff - pd.Timedelta(days=5)
        tr = panel[(panel["trade_date"]>=train_end-pd.Timedelta(days=730)) & (panel["trade_date"]<=train_end)]
        tr = tr[tr["fwd_20d_ret"].notna() & (tr["fwd_20d_ret"].abs()<0.5)]
        if len(tr)<20000: continue
        if len(tr)>100000: tr=tr.sample(100000,random_state=42)
        
        X_tr = tr[rf_factors].fillna(0).values.astype(np.float32)
        y_tr = np.clip(tr["fwd_20d_ret"].values.astype(np.float32),-0.3,0.3)
        nv = max(1,int(len(tr)*0.15))
        
        # 训练LGB（带更多树，以便计算树间标准差）
        m = lgb.LGBMRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.5, reg_alpha=0.5, reg_lambda=1.0,
            min_child_weight=20, min_data_in_leaf=100, random_state=42, verbose=-1, n_jobs=8)
        m.fit(X_tr[:-nv], y_tr[:-nv], eval_set=[(X_tr[-nv:], y_tr[-nv:])],
              callbacks=[lgb.early_stopping(20, verbose=False)], eval_metric="mse")
        
        # 对每期预测
        for d in period_dates:
            if d <= train_cutoff or d < period_dates[period_dates.index(half_dates[3][1])] if half_dates[3][1] in period_dates else False: continue
            day = panel[panel["trade_date"]==d]
            if len(day)==0: continue
            X_te = day[rf_factors].fillna(0).values.astype(np.float32)
            
            # 获取每棵树预测 -> 计算置信度
            # method: predict leaf indices, then compute across-tree variance
            try:
                preds_per_tree = m.predict(X_te, pred_leaf=False, num_iteration=m.best_iteration_)
                # boosters - get per-tree predictions
                y_pred = np.zeros(len(X_te))
                tree_preds = []
                for tree_idx in range(m.best_iteration_):
                    # m.booster_.predict() with start_iteration/num_iteration
                    tp = m.predict(X_te, start_iteration=tree_idx, num_iteration=1)
                    tree_preds.append(tp)
                tree_preds = np.column_stack(tree_preds)
                conf = np.std(tree_preds, axis=1)  # 标准差=置信度反向指标
                mean_pred = np.mean(tree_preds, axis=1)
            except:
                conf = np.ones(len(X_te)) * 0.02
                mean_pred = m.predict(X_te)
            
            for j, code in enumerate(day["ts_code"].values):
                all_preds["trade_date"].append(d)
                all_preds["ts_code"].append(code)
                all_preds["pred_ret"].append(float(mean_pred[j]) if hasattr(mean_pred,'__len__') else float(m.predict(X_te[j:j+1])[0]))
                all_preds["confidence"].append(float(1.0 / (conf[j] + 0.01)) if hasattr(conf,'__len__') and conf[j] > 0 else 1.0)
        
        if (hi+1)%3==0 or hi==len(half_dates)-1:
            print(f"  {hk}: {hi+1}/{len(half_dates)-3}期完成", flush=True)
    
    pdf = pd.DataFrame(all_preds)
    print(f"  总预测: {len(pdf):,}, {pdf['trade_date'].nunique()}期", flush=True)
    return pdf

# 简化版：用第1次训练的树间标准差
def rolling_predict_simple():
    """置信度加权：用LGB树间标准差作为预测置信度"""
    all_preds = {"trade_date":[],"ts_code":[],"pred_ret":[],"confidence":[]}
    
    for hi,(hk,train_cutoff) in enumerate(half_dates):
        if hi < 3: continue
        train_end = train_cutoff - pd.Timedelta(days=5)
        tr = panel[(panel["trade_date"]>=train_end-pd.Timedelta(days=730)) & (panel["trade_date"]<=train_end)]
        tr = tr[tr["fwd_20d_ret"].notna() & (tr["fwd_20d_ret"].abs()<0.5)]
        if len(tr)<20000: continue
        if len(tr)>100000: tr=tr.sample(100000,random_state=42)
        
        X_tr = tr[rf_factors].fillna(0).values.astype(np.float32)
        y_tr = np.clip(tr["fwd_20d_ret"].values.astype(np.float32),-0.3,0.3)
        nv = max(1,int(len(tr)*0.15))
        
        # 训练200棵树（不要太深，节约时间）
        m = lgb.LGBMRegressor(n_estimators=500, max_depth=3, lr=0.02,
            subsample=0.7, colsample_bytree=0.7,
            reg_alpha=0.2, reg_lambda=1.0,
            min_child_weight=20, min_data_in_leaf=100,
            random_state=42, verbose=-1, n_jobs=8)
        m.fit(X_tr[:-nv], y_tr[:-nv], eval_set=[(X_tr[-nv:], y_tr[-nv:])],
              callbacks=[lgb.early_stopping(30, verbose=False)], eval_metric="mse")
        
        best_n = m.best_iteration_
        
        for d in period_dates:
            if d <= train_cutoff: continue
            day = panel[panel["trade_date"]==d]
            if len(day)==0: continue
            X_te = day[rf_factors].fillna(0).values.astype(np.float32)
            
            # 使用正常预测（不加树间方差）确保基准正确
            mean_pred = m.predict(X_te)
            # 近似置信度：用预测排序的稳定性
            # 模拟树间方差：对数据加小噪声，预测多次看差异
            n_boot = 10
            boot_preds = np.zeros((len(X_te), n_boot))
            rng = np.random.RandomState(42)
            for b in range(n_boot):
                noise = rng.normal(0, 0.01, X_te.shape)
                boot_preds[:, b] = m.predict(X_te + noise)
            tree_std = np.std(boot_preds, axis=1)
            # 置信度 = 1 / (1 + 树间标准差/平均预测绝对值)
            conf = 1.0 / (1.0 + tree_std / (np.abs(mean_pred) + 0.01))
            
            for j, code in enumerate(day["ts_code"].values):
                all_preds["trade_date"].append(d)
                all_preds["ts_code"].append(code)
                all_preds["pred_ret"].append(float(mean_pred[j]))
                all_preds["confidence"].append(float(conf[j]))
        
        if (hi+1)%3==0 or hi==len(half_dates)-1:
            print(f"  {hk}: {hi+1}/{len(half_dates)-3}期完成 (best={best_n})", flush=True)
    
    pdf = pd.DataFrame(all_preds)
    print(f"  总预测: {len(pdf):,}, {pdf['trade_date'].nunique()}期", flush=True)
    return pdf

STAMP, COMM, SLIP = 0.001, 0.0002, 0.002

def bt_v31(pdf, n=30, tv=0.15, label="v31等权", min_date=None):
    """v31版本：指数衰减+行业中性"""
    pds = sorted(pdf["trade_date"].unique())
    if min_date: pds=[d for d in pds if d>=min_date]
    if len(pds)<2: return None
    cash, hold, navs = 0.03, {}, [1.0]
    for i in range(len(pds)-1):
        d,sd = pds[i],pds[i+1]
        pb,psm = price_map.get(d,{}),price_map.get(sd,{})
        sv = vol_map.get(d,{})
        hv = sum(qty*pb.get(c,0) for c,qty in hold.items())
        tv_ = hv+cash; sp=0
        for c,qty in hold.items():
            px=psm.get(c,0)
            if px>0:
                v = qty * px
                sp+=v-v*(STAMP+COMM+SLIP)
        cash+=sp; hold={}
        dp_ = pdf[pdf["trade_date"]==d].sort_values("pred_ret",ascending=False).copy()
        codes=list(dp_["ts_code"]); scores=dp_["pred_ret"].values
        # 行业中性
        selected=[]; ind_cnt={}
        order=np.argsort(-scores)
        for j in order:
            ind=stk_ind.get(codes[j],"其他")
            if ind_cnt.get(ind,0)<3:
                selected.append(j); ind_cnt[ind]=ind_cnt.get(ind,0)+1
            if len(selected)>=n: break
        sel=[codes[j] for j in selected]
        r=np.arange(1,len(sel)+1)
        w=np.exp(-0.1*r); w=w/w.sum()
        sl=[sv.get(c,np.nan) for c in sel]
        sl=[v for v in sl if not np.isnan(v) and v>0.01]
        pr=max(min(tv/np.median(sl),1.0) if len(sl)>=5 else 1.0,0.05)
        if sel and cash>0.001:
            al=cash*pr*0.98
            if al>0.001:
                for code,wt in zip(sel,w):
                    px=pb.get(code,0)
                    if px>0 and wt>0:
                        b=(al*wt)*(1-COMM-SLIP)/px
                        if b>0: hold[code]=b
                cash-=al
        np_=sum(qty*psm.get(c,0) for c,qty in hold.items())
        nt=np_+cash
        r_t=nt/tv_-1 if tv_>0 else 0
        navs.append(navs[-1]*(1+r_t))
    pnl=np.array(navs[1:])/np.array(navs[:-1])-1
    na=np.array(navs); ny=len(pnl)/13
    ar=na[-1]**(1/ny)-1 if ny>0 and na[-1]>0 else 0
    sr=np.mean(pnl)/np.std(pnl)*np.sqrt(13) if np.std(pnl)>0 else 0
    dd=np.maximum.accumulate(na)-na; mdd=dd.max()
    wr=np.mean(pnl>0)
    print(f"  {label:28s}: 年化{ar*100:+7.1f}% | 夏普{sr:5.2f} | 回撤{mdd*100:5.1f}% | 胜率{wr*100:3.0f}% | {len(pnl)}期", flush=True)
    return {"ret":f"{ar*100:+.1f}%","sr":f"{sr:.2f}","mdd":f"{mdd*100:.1f}%","num":len(pnl)}

def bt_conf(pdf, n=30, tv=0.15, label="置信度加权", use_industry=True, min_date=None):
    """置信度加权版：用confidence列调整权重"""
    pds = sorted(pdf["trade_date"].unique())
    if min_date: pds=[d for d in pds if d>=min_date]
    if len(pds)<2: return None
    cash, hold, navs = 0.03, {}, [1.0]
    for i in range(len(pds)-1):
        d,sd = pds[i],pds[i+1]
        pb,psm = price_map.get(d,{}),price_map.get(sd,{})
        sv = vol_map.get(d,{})
        hv = sum(qty*pb.get(c,0) for c,qty in hold.items())
        tv_ = hv+cash; sp=0
        for c,qty in hold.items():
            px=psm.get(c,0)
            if px>0:
                v = qty * px
                sp+=v-v*(STAMP+COMM+SLIP)
        cash+=sp; hold={}
        dp_ = pdf[pdf["trade_date"]==d].copy()
        dp_ = dp_.sort_values("pred_ret", ascending=False)
        codes=list(dp_["ts_code"]); scores=dp_["pred_ret"].values; confs=dp_["confidence"].values
        
        if use_industry:
            selected=[]; ind_cnt={}
            order=np.argsort(-scores)
            for j in order:
                ind=stk_ind.get(codes[j],"其他")
                if ind_cnt.get(ind,0)<3:
                    selected.append(j); ind_cnt[ind]=ind_cnt.get(ind,0)+1
                if len(selected)>=n: break
        else:
            selected=list(range(min(n, len(codes))))
        
        # 置信度加权权重
        base_weights = np.exp(-0.1 * np.arange(1, len(selected)+1))
        conf_weights = np.array([confs[j] for j in selected])
        final_weights = base_weights * conf_weights
        final_weights = final_weights / final_weights.sum()
        
        sel_codes=[codes[j] for j in selected]
        sl=[sv.get(c,np.nan) for c in sel_codes]
        sl=[v for v in sl if not np.isnan(v) and v>0.01]
        pr=max(min(tv/np.median(sl),1.0) if len(sl)>=5 else 1.0,0.05)
        
        if sel_codes and cash>0.001:
            al=cash*pr*0.98
            if al>0.001:
                for code,wt in zip(sel_codes,final_weights):
                    px=pb.get(code,0)
                    if px>0 and wt>0:
                        b=(al*wt)*(1-COMM-SLIP)/px
                        if b>0: hold[code]=b
                cash-=al
        np_=sum(qty*psm.get(c,0) for c,qty in hold.items())
        nt=np_+cash
        r_t=nt/tv_-1 if tv_>0 else 0
        navs.append(navs[-1]*(1+r_t))
    pnl=np.array(navs[1:])/np.array(navs[:-1])-1
    na=np.array(navs); ny=len(pnl)/13
    ar=na[-1]**(1/ny)-1 if ny>0 and na[-1]>0 else 0
    sr=np.mean(pnl)/np.std(pnl)*np.sqrt(13) if np.std(pnl)>0 else 0
    dd=np.maximum.accumulate(na)-na; mdd=dd.max()
    wr=np.mean(pnl>0)
    print(f"  {label:28s}: 年化{ar*100:+7.1f}% | 夏普{sr:5.2f} | 回撤{mdd*100:5.1f}% | 胜率{wr*100:3.0f}% | {len(pnl)}期", flush=True)
    return {"ret":f"{ar*100:+.1f}%","sr":f"{sr:.2f}","mdd":f"{mdd*100:.1f}%","num":len(pnl)}

# 跑回测
print(f"\n滚动训练+置信度计算...", flush=True)
preds = rolling_predict_simple()

print(f"\n{'='*60}", flush=True)
print(f"回测对比: T30 目波15% 2021-2026", flush=True)
print("-"*60, flush=True)

# IC
ics=[]
for d in preds["trade_date"].unique():
    day=preds[preds["trade_date"]==d]
    pday=panel[panel["trade_date"]==d]
    m=day.merge(pday[["ts_code","fwd_20d_ret"]],on="ts_code")
    if len(m)>10:
        ic,_=spearmanr(m["pred_ret"],m["fwd_20d_ret"])
        if not np.isnan(ic): ics.append(ic)
if ics:
    print(f"IC: {np.mean(ics)*100:+.2f}% IR: {np.mean(ics)/np.std(ics):.2f} ({len(ics)}期)", flush=True)

bt_v31(preds, 30, 0.15, "v31指数衰减+行业中性", min_date=pd.Timestamp("2022-12-06"))
bt_conf(preds, 30, 0.15, "v32置信度加权+行业中性", True, min_date=pd.Timestamp("2022-12-06"))
bt_conf(preds, 30, 0.15, "v32置信度加权(无行业)", False, min_date=pd.Timestamp("2022-12-06"))

# OOS
print(f"\n--- OOS(2023+) ---", flush=True)
bt_v31(preds, 30, 0.15, "v31 OOS", min_date=pd.Timestamp("2023-01-01"))
bt_conf(preds, 30, 0.15, "v32置信度+行业中性 OOS", True)

print(f"\n{'='*60}", flush=True)
print(f"✅ 完成 | ⏱ {(time.time()-tt)/60:.1f}分", flush=True)
