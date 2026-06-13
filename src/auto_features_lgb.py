"""
auto_features_lgb.py — 衍生因子LGB特征重要性筛选 + 回测
读auto_features_v1.parquet + factor_panel_v6, 80因子+800衍生=879特征
用LGB feature_importance筛选，然后做全量回测对比
"""
import os, sys, time, json
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
os.chdir("/mnt/d/AI-20260518")
sys.path.insert(0, '.')
import joblib, lightgbm as lgb
from scipy.stats import spearmanr
tt = time.time()

print("="*60)
print("Auto-Features LGB筛选+回测")
print(f"{time.strftime('%F %H:%M')}")
print("="*60, flush=True)

# 读面板
panel = pd.read_parquet("data/factors/factor_panel_v6.parquet")
factor_base = [c for c in panel.columns 
               if c not in ["ts_code","trade_date","fwd_20d_ret","close","volume","ret_1d",
                   "__fragment_index","__batch_index","__last_in_fragment","__filename","board_break"]]
print(f"基础因子: {len(factor_base)}", flush=True)

# 读衍生因子
print("读衍生因子...", flush=True)
auto = pd.read_parquet("data/factors/auto_features_v1.parquet")
auto_cols = list(auto.columns)
print(f"衍生因子: {len(auto_cols)}", flush=True)

# 拼接
panel = pd.concat([panel, auto], axis=1)
del auto
all_factors = factor_base + auto_cols
print(f"总特征: {len(all_factors)}", flush=True)

# ===== 1. LGB特征重要性 =====
print("\n[1] LGB特征重要性筛选...", flush=True)
train = panel[(panel["trade_date"] >= pd.Timestamp("2024-01-01")) & 
              (panel["trade_date"] < pd.Timestamp("2025-06-01"))]
train = train[train["fwd_20d_ret"].notna() & (train["fwd_20d_ret"].abs() < 0.5)]
if len(train) > 100000:
    train = train.sample(100000, random_state=42)

X = train[all_factors].fillna(0).values.astype(np.float32)
y = np.clip(train["fwd_20d_ret"].values.astype(np.float32), -0.3, 0.3)

nv = max(1, int(len(train) * 0.15))
lgb_fi = lgb.LGBMRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.5, reg_alpha=0.5, reg_lambda=1.0,
    min_child_weight=20, min_data_in_leaf=100, random_state=42, verbose=-1, n_jobs=8)
lgb_fi.fit(X[:-nv], y[:-nv], eval_set=[(X[-nv:], y[-nv:])],
           callbacks=[lgb.early_stopping(20, verbose=False)], eval_metric="mse")

imp = pd.DataFrame({"factor": all_factors, "importance": lgb_fi.feature_importances_})
imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)

imp_base = imp[imp["factor"].isin(factor_base)]
imp_auto = imp[~imp["factor"].isin(factor_base)]

print(f"  基础因子活跃(imp>0): {len(imp_base[imp_base['importance']>0])}")
print(f"  衍生因子活跃(imp>0): {len(imp_auto[imp_auto['importance']>0])}", flush=True)

# Top30%基础 + top衍生
n_base = max(1, int(len(factor_base) * 0.3))
top_base = imp_base.head(n_base)["factor"].tolist()

median_auto = imp_auto["importance"].median()
top_auto_list = imp_auto[imp_auto["importance"] >= median_auto].head(80)["factor"].tolist()

sel_features = list(dict.fromkeys(top_base + top_auto_list))
print(f"  入选: {len(sel_features)} ({n_base}基础+{len(top_auto_list)}衍生)", flush=True)

json.dump({"selected": sel_features, "n_base": n_base, "n_auto": len(top_auto_list)},
          open("models/auto_features_lgb.json","w"), indent=2)

# Top20
print(f"\n  Top20特征:")
for i, (_, row) in enumerate(imp.head(20).iterrows()):
    tag = " 🆕" if row["factor"] in auto_cols else ""
    print(f"    {i+1}. {row['factor'][:60]:60s} imp={row['importance']}{tag}", flush=True)

# ===== 2. 回测 =====
print(f"\n[2] 回测对比...", flush=True)

prices = pd.read_parquet("data/factors/full_prices.parquet")
prices["trade_date"] = pd.to_datetime(prices["trade_date"])
panel["trade_date"] = pd.to_datetime(panel["trade_date"])

ps = prices.sort_values(["ts_code","trade_date"]).copy()
ps["ret_1d"] = ps.groupby("ts_code")["close"].pct_change()
ps["vol_60d"] = ps.groupby("ts_code")["ret_1d"].transform(lambda x: x.rolling(60, min_periods=20).std())
ps["vol_60d_ann"] = ps["vol_60d"] * np.sqrt(244)

all_dates = sorted(p["trade_date"].unique() for p in [panel])[0]
period_dates = [all_dates[i] for i in range(0, len(all_dates), 20) 
                if all_dates[i] >= pd.Timestamp("2021-01-01")]
# If panel dates differ from prices - use panel dates
panel_dates = sorted(panel["trade_date"].unique())
period_dates = [panel_dates[i] for i in range(0, len(panel_dates), 20)
                if panel_dates[i] >= pd.Timestamp("2021-01-01")]

price_map, vol_map = {}, {}
for d in period_dates:
    sub = prices[prices["trade_date"]==d][["ts_code","close"]].dropna()
    price_map[d] = dict(zip(sub["ts_code"], sub["close"]))
    v_sub = ps[ps["trade_date"]==d][["ts_code","vol_60d_ann"]].dropna()
    vol_map[d] = dict(zip(v_sub["ts_code"], v_sub["vol_60d_ann"]))

STAMP, COMM, SLIP = 0.001, 0.0002, 0.002

def bt(pdf, n=30, tv=0.15, label="", start_date=None):
    pdts = sorted(pdf["trade_date"].unique())
    if start_date: pdts = [d for d in pdts if d >= start_date]
    if len(pdts) < 2: return None
    cash, hold, navs = 0.03, {}, [1.0]
    for i in range(len(pdts)-1):
        d, sd = pdts[i], pdts[i+1]
        pb, psm = price_map.get(d, {}), price_map.get(sd, {})
        sv = vol_map.get(d, {})
        hv = sum(shares * pb.get(c,0) for c,s in hold.items())
        tv_ = hv + cash
        sp = 0
        for c,shares in hold.items():
            px = psm.get(c,0)
            if px>0:
                v = shares*px
                sp += v - v*(STAMP+COMM+SLIP)
        cash += sp; hold = {}
        dp_ = pdf[pdf["trade_date"]==d].sort_values("pred_ret", ascending=False)
        sel = list(dp_.head(n)["ts_code"])
        vs_ = [sv.get(c,np.nan) for c in sel if not np.isnan(sv.get(c,np.nan)) and sv[c]>0.01]
        pr = max(min(tv/np.median(vs_),1.0) if len(vs_)>=5 else 1.0, 0.05)
        if sel and cash>0.001:
            av = cash*pr*0.98
            if av>0.001:
                per = av/len(sel)
                for c in sel:
                    px = pb.get(c,0)
                    if px>0 and per>0:
                        bc = per*(COMM+SLIP)
                        b = (per-bc)/px
                        if b>0: hold[c]=b
                cash -= per*len(hold)
        np_ = sum(shares * psm.get(c,0) for c,shares in hold.items())
        nt = np_ + cash
        r = nt/tv_ - 1 if tv_>0 else 0
        navs.append(navs[-1]*(1+r))
    pnl = np.array(navs[1:])/np.array(navs[:-1])-1
    na = np.array(navs)
    ny = len(pnl)/13
    ar = na[-1]**(1/ny)-1 if ny>0 and na[-1]>0 else 0
    sr = np.mean(pnl)/np.std(pnl)*np.sqrt(13) if np.std(pnl)>0 else 0
    dd = np.maximum.accumulate(na)-na
    mdd = dd.max()
    wr = np.mean(pnl>0)
    print(f"  {label:25s}: 年化{ar*100:+7.1f}% | 夏普{sr:6.2f} | 回撤{mdd*100:6.1f}% | 胜率{wr*100:3.0f}% | {len(pnl)}期", flush=True)
    return {"ret":f"{ar*100:+.1f}%","sr":f"{sr:.2f}","mdd":f"{mdd*100:.1f}%","num":len(pnl)}

# 滚动训练：基准（v29: 79因子基础）vs 新特征集
rf_md = joblib.load("models/ml_ensemble_v1.joblib")
rf_factors = rf_md["factor_cols"]

# 用来做baseline的IC
half_dates = []
for d in period_dates:
    k = f"{d.year}{'H1' if d.month<=6 else 'H2'}"
    if not half_dates or half_dates[-1][0]!=k:
        half_dates.append((k,d))
    else:
        half_dates[-1]=(k,d)

def rolling_predict(feature_list, label=""):
    """滚动预测所有期"""
    preds_list = []
    for hi, (hk, train_cutoff) in enumerate(half_dates):
        if hi < 3: continue
        train_end = train_cutoff - pd.Timedelta(days=5)
        train_data = panel[(panel["trade_date"]>=train_end-pd.Timedelta(days=730)) & (panel["trade_date"]<=train_end)]
        train_data = train_data[train_data["fwd_20d_ret"].notna() & (train_data["fwd_20d_ret"].abs()<0.5)]
        if len(train_data)<20000: continue
        if len(train_data)>100000: train_data=train_data.sample(100000,random_state=42)
        
        X_tr = train_data[feature_list].fillna(0).values.astype(np.float32)
        y_tr = np.clip(train_data["fwd_20d_ret"].values.astype(np.float32),-0.3,0.3)
        nv = max(1,int(len(train_data)*0.15))
        
        lgb_m = lgb.LGBMRegressor(**dict(n_estimators=500,max_depth=3,lr=0.02,
            subsample=0.7,colsample_bytree=0.7,reg_alpha=0.2,reg_lambda=1.0,
            min_child_weight=20,min_data_in_leaf=100,random_state=42,verbose=-1,n_jobs=8))
        lgb_m.fit(X_tr[:-nv],y_tr[:-nv],eval_set=[(X_tr[-nv:],y_tr[-nv:])],
                  callbacks=[lgb.early_stopping(30,verbose=False)],eval_metric="mse")
        
        for d in period_dates:
            if d <= train_cutoff: continue
            idx = panel["trade_date"]==d
            day = panel[idx]
            if len(day)==0: continue
            X_te = day[feature_list].fillna(0).values.astype(np.float32)
            preds = lgb_m.predict(X_te)
            for j, code in enumerate(day["ts_code"].values):
                preds_list.append({"trade_date": d, "ts_code": code, "pred_ret": float(preds[j])})
        
        if (hi+1)%5==0:
            print(f"  {label}: {hi+1}/{len(half_dates)-3} 期期完成", flush=True)
    
    pdf = pd.DataFrame(preds_list)
    return pdf

print(f"\n基线 LGB基础因子({len(rf_factors)})...", flush=True)
base_pred = rolling_predict(rf_factors, "基线条")
print(f"  基线: {len(base_pred):,}预测", flush=True)

print(f"\n新特征集 LGB({len(sel_features)})...", flush=True)
# 用一部分衍生因子（top 40）加上基础
new_features = list(dict.fromkeys(top_base + top_auto_list[:40]))
print(f"  新特征集: {len(new_features)} ({len(top_base)}基础+{len(top_auto_list[:40])}衍生)", flush=True)
auto_pred = rolling_predict(new_features, "新特征")

print(f"\n{'='*60}")
print("回测对比: LGB基础 vs LGB+AutoFeat")
print(f"{'='*60}")
print(f"\n{'策略':30s} {'T30年化':>8s} {'T30夏普':>8s} {'T30回撤':>8s} {'T30胜率':>8s}")
print("-"*65)

for pdf, name in [(base_pred,"LGB基础(79因子)"), (auto_pred,"LGB+AutoFeat")]:
    r = bt(pdf, 30, 0.15, name)
    if name == "LGB+AutoFeat":
        r_o = bt(pdf, 30, 0.15, name+" OOS", start_date=pd.Timestamp("2023-01-01"))

print(f"\n{'='*60}")
print(f"✅ 完成")
print(f"⏱ {(time.time()-tt)/60:.1f}分", flush=True)
