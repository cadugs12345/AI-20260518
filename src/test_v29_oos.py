"""
v29 样本外验证 — LightGBM夏普1.03是否真实？
方法：用2023之前数据训练，预测2023之后所有period
        2022H2训练 → 2023以后预测
        对比 full-sample vs out-of-sample
"""
import os, sys, time, json
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
os.chdir("/mnt/d/AI-20260518")
import joblib, lightgbm as lgb
from scipy.stats import spearmanr
tt = time.time()

print("="*60)
print("v29 样本外验证 (Out-of-Sample)")
print(f"{time.strftime('%F %H:%M')}")
print("="*60)

panel = pd.read_parquet("data/factors/factor_panel_v6.parquet")
prices = pd.read_parquet("data/factors/full_prices.parquet")
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
prices["trade_date"] = pd.to_datetime(prices["trade_date"])

factor_cols = [c for c in panel.columns 
               if c not in ["ts_code","trade_date","fwd_20d_ret","close","volume","ret_1d",
                   "__fragment_index","__batch_index","__last_in_fragment","__filename","board_break"]]
core15 = ["短期反转","20日动量","60日动量","120日动量","波动率",
          "换手率","量比","量能趋势","BP","EP","MACD","BOLL位置",
          "EMA5偏离","EMA10偏离","EMA20偏离"]

# 波动率
ps = prices.sort_values(["ts_code","trade_date"]).copy()
ps["ret_1d"] = ps.groupby("ts_code")["close"].pct_change()
ps["vol_60d"] = ps.groupby("ts_code")["ret_1d"].transform(lambda x: x.rolling(60, min_periods=20).std())
ps["vol_60d_ann"] = ps["vol_60d"] * np.sqrt(244)

all_dates = sorted(panel["trade_date"].unique())
period_dates = [all_dates[i] for i in range(0, len(all_dates), 20)
                if all_dates[i] >= pd.Timestamp("2021-01-01")]

price_map, vol_map = {}, {}
for d in period_dates:
    sub = prices[prices["trade_date"]==d][["ts_code","close"]].dropna()
    price_map[d] = dict(zip(sub["ts_code"], sub["close"]))
    v_sub = ps[ps["trade_date"]==d][["ts_code","vol_60d_ann"]].dropna()
    vol_map[d] = dict(zip(v_sub["ts_code"], v_sub["vol_60d_ann"]))

# 基准RF
rf_md = joblib.load("models/ml_ensemble_v1.joblib")
rf_model, rf_factors = rf_md["model"], rf_md["factor_cols"]

v12_z = panel[core15].rank(pct=True)
panel["s_v12"] = v12_z.mean(axis=1)
panel["s_rf"] = 0.0
for date in period_dates:
    idx = panel["trade_date"]==date
    day = panel[idx]
    X = day[rf_factors].fillna(0).values.astype(np.float32)
    panel.loc[idx, "s_rf"] = rf_model.predict_proba(X)[:,1]
print(f"基准计算完成: {len(period_dates)}期", flush=True)

# ===== 样本外验证 =====
# 只用一个模型：用2023-01-01之前的数据训练，预测之后所有
print("\n[样本外] 单次训练（2023前训练→2023后预测）...")

train = panel[(panel["trade_date"] >= "2021-01-01") & (panel["trade_date"] < "2023-01-01")]
train = train[train["fwd_20d_ret"].notna() & (train["fwd_20d_ret"].abs() < 0.5)]
if len(train) > 150000:
    train = train.sample(150000, random_state=42)

X_tr = train[rf_factors].fillna(0).values.astype(np.float32)
y_tr = np.clip(train["fwd_20d_ret"].values.astype(np.float32), -0.3, 0.3)
n_val = max(1, int(len(train) * 0.15))

lgb_oos = lgb.LGBMRegressor(
    n_estimators=500, max_depth=3, learning_rate=0.02,
    subsample=0.7, colsample_bytree=0.7,
    reg_alpha=0.2, reg_lambda=1.0,
    min_child_weight=20, min_data_in_leaf=100,
    random_state=42, verbose=-1, n_jobs=8
)
lgb_oos.fit(X_tr[:-n_val], y_tr[:-n_val],
            eval_set=[(X_tr[-n_val:], y_tr[-n_val:])],
            callbacks=[lgb.early_stopping(30, verbose=False)],
            eval_metric="mse")
print(f"  训练完成: {len(train):,}条, best={lgb_oos.best_iteration_}", flush=True)

# 预测2023之后全部
print("  预测样本外...", flush=True)
oos_preds = {"trade_date": [], "ts_code": [], "pred_ret": []}
for d in period_dates:
    if d < pd.Timestamp("2023-01-01"):
        continue
    idx = panel["trade_date"] == d
    day = panel[idx]
    X_te = day[rf_factors].fillna(0).values.astype(np.float32)
    preds = lgb_oos.predict(X_te)
    oos_preds["trade_date"].extend([d] * len(day))
    oos_preds["ts_code"].extend(day["ts_code"].values)
    oos_preds["pred_ret"].extend(preds.astype(np.float64))

oos_df = pd.DataFrame(oos_preds)
print(f"  样本外: {len(oos_df):,}条, {oos_df['trade_date'].nunique()}期", flush=True)

# IC
ic_list = []
for d in oos_df["trade_date"].unique():
    lgb_day = oos_df[oos_df["trade_date"]==d]
    panel_day = panel[panel["trade_date"]==d]
    m = lgb_day.merge(panel_day[["ts_code","fwd_20d_ret"]], on="ts_code")
    if len(m) > 10:
        ic, _ = spearmanr(m["pred_ret"], m["fwd_20d_ret"])
        ic_list.append(ic)
if ic_list:
    ic_m = np.mean(ic_list)
    ic_s = np.std(ic_list)
    print(f"  IC: {ic_m*100:+.2f}% | IR: {ic_m/ic_s:.2f} | {len(ic_list)}期", flush=True)

# ===== 回测 =====
print("\n[回测]   样本外(2023-2026) vs 全样本(2021-2026)")
STAMP, COMM, SLIP = 0.001, 0.0002, 0.002

def bt(pdf, n=30, tv=0.15, label="", start_date=None):
    pdts = sorted(pdf["trade_date"].unique())
    if start_date:
        pdts = [d for d in pdts if d >= start_date]
    if len(pdts) < 2:
        print(f"  {label:20s}: 期数不足({len(pdts)})")
        return None
    cash, hold, navs = 0.03, {}, [1.0]
    for i in range(len(pdts)-1):
        d, sd = pdts[i], pdts[i+1]
        pb, psm = price_map.get(d, {}), price_map.get(sd, {})
        sv = vol_map.get(d, {})
        hv = sum(shares * pb.get(c,0) for c,shares in hold.items())
        tv_ = hv + cash
        sp = 0
        for c,shares in hold.items():
            px = psm.get(c,0)
            if px>0:
                v = shares*px
                sp += v - v*(STAMP+COMM+SLIP)
        cash += sp
        hold = {}
        dp_ = pdf[pdf["trade_date"]==d].sort_values("pred_ret",ascending=False)
        sel = list(dp_.head(n)["ts_code"])
        vs_ = [sv.get(c,np.nan) for c in sel if not np.isnan(sv.get(c,np.nan)) and sv[c]>0.01]
        pr = max(min(tv/np.median(vs_),1.0) if len(vs_)>=5 else 1.0,0.05)
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
    vv = np.std(pnl)*np.sqrt(13)
    sr = np.mean(pnl)/np.std(pnl)*np.sqrt(13) if np.std(pnl)>0 else 0
    dd = np.maximum.accumulate(na)-na
    mdd = dd.max()
    wr = np.mean(pnl>0)
    cal = ar/mdd if mdd>0 else 0
    print(f"  {label:20s}: 年化{ar*100:+7.1f}% | 夏普{sr:6.2f} | 回撤{mdd*100:6.1f}% | 胜率{wr*100:3.0f}% | {len(pnl)}期")
    return {"ret":f"{ar*100:+.1f}%","sr":f"{sr:.2f}","mdd":f"{mdd*100:.1f}%","num":len(pnl)}

# 构建回测信号
print("\n构建信号...", flush=True)
pred_records = {k:[] for k in ["v12","rf","rf_risk","lgb_full","lgb_oos"]}
lgb_full = pd.DataFrame({"trade_date": pd.Series(dtype="datetime64[ns]"),
    "ts_code": pd.Series(dtype="str"), "pred_ret": pd.Series(dtype="float64")})

# 用滚动训练（和v29完全相同）做全样本
half_dates = []
for d in period_dates:
    k = f"{d.year}{'H1' if d.month<=6 else 'H2'}"
    if not half_dates or half_dates[-1][0]!=k:
        half_dates.append((k,d))
    else:
        half_dates[-1]=(k,d)

for hi,(hk,train_cutoff) in enumerate(half_dates):
    if hi<3: continue
    train_end = train_cutoff - pd.Timedelta(days=5)
    train = panel[(panel["trade_date"]>=train_end-pd.Timedelta(days=730)) & (panel["trade_date"]<=train_end)]
    train = train[train["fwd_20d_ret"].notna() & (train["fwd_20d_ret"].abs()<0.5)]
    if len(train)<20000: continue
    if len(train)>150000: train=train.sample(150000,random_state=42)
    
    X_tr = train[rf_factors].fillna(0).values.astype(np.float32)
    y_tr = np.clip(train["fwd_20d_ret"].values.astype(np.float32),-0.3,0.3)
    nv = max(1,int(len(train)*0.15))
    
    lgb_m = lgb.LGBMRegressor(**dict(n_estimators=500,max_depth=3,learning_rate=0.02,
        subsample=0.7,colsample_bytree=0.7,reg_alpha=0.2,reg_lambda=1.0,
        min_child_weight=20,min_data_in_leaf=100,random_state=42,verbose=-1,n_jobs=8))
    lgb_m.fit(X_tr[:-nv],y_tr[:-nv],eval_set=[(X_tr[-nv:],y_tr[-nv:])],
              callbacks=[lgb.early_stopping(30,verbose=False)],eval_metric="mse")
    
    for d in period_dates:
        if d<=train_cutoff: continue
        idx = panel["trade_date"]==d
        day = panel[idx]
        X_te = day[rf_factors].fillna(0).values.astype(np.float32)
        preds = lgb_m.predict(X_te)
        lgb_full = pd.concat([lgb_full,pd.DataFrame({"trade_date":d,"ts_code":day["ts_code"].values,
            "pred_ret":preds.astype(np.float64)})],ignore_index=True)

# 构建各策略信号
lgb_full_map = {}
for d in lgb_full["trade_date"].unique():
    lgb_full_map[d] = dict(zip(lgb_full[lgb_full["trade_date"]==d]["ts_code"],
                               lgb_full[lgb_full["trade_date"]==d]["pred_ret"]))
lgb_oos_map = {}
for d in oos_df["trade_date"].unique():
    lgb_oos_map[d] = dict(zip(oos_df[oos_df["trade_date"]==d]["ts_code"],
                              oos_df[oos_df["trade_date"]==d]["pred_ret"]))

for d in period_dates:
    day = panel[panel["trade_date"]==d].copy()
    if len(day)==0: continue
    
    sc_v12 = day["s_v12"].values
    sc_rf = day["s_rf"].values
    sc_rf_risk = day["s_rf"].values.copy()
    
    for j,(_,r) in enumerate(day.iterrows()):
        triggered = sum([not np.isnan(r.get("repair_force_10d",np.nan)) and r["repair_force_10d"]<-0.05,
                         not np.isnan(r.get("高波反转",np.nan)) and r["高波反转"]<-0.03,
                         not np.isnan(r.get("量价背离",np.nan)) and r["量价背离"]>0.03])>0
        if triggered:
            sc_rf_risk[j] = -999
    
    # LGB全样本
    full_d = lgb_full_map.get(d, {})
    sc_full = np.array([full_d.get(code, sc_rf[j]) for j,code in enumerate(day["ts_code"].values)])
    
    # LGB样本外
    oos_d = lgb_oos_map.get(d, {})
    sc_oos = np.array([oos_d.get(code, sc_rf[j]) for j,code in enumerate(day["ts_code"].values)])
    
    for strategy,sc in [("v12",sc_v12),("rf",sc_rf),("rf_risk",sc_rf_risk),
                        ("lgb_full",sc_full),("lgb_oos",sc_oos)]:
        order = np.argsort(-sc)[:50]
        for j in range(50):
            idx = order[j]
            pred_records[strategy].append({"trade_date":d,"ts_code":day["ts_code"].values[idx],"pred_ret":float(sc[idx])})

# 回测：全样本 vs 样本外
predicts = {k:pd.DataFrame(pred_records[k]) for k in pred_records}

print(f"\n{'='*60}")
print("回测对比: 全样本(2021-2026) vs 样本外(2023-2026)")
print(f"{'='*60}")
print(f"\n{'策略':22s} {'T30全样年化':>10s} {'T30全样夏普':>10s} {'T30OOS年化':>10s} {'T30OOS夏普':>10s}")
print("-"*65)

results_all = {}
results_oos = {}
labels = [("v12","v12等权"),("rf","RF"),("rf_risk","RF+风控"),
          ("lgb_full","LGB全样"),("lgb_oos","LGB OOS")]

for k,name in labels:
    a = bt(predicts[k], 30, 0.15, name+"(全样)")
    o = bt(predicts[k], 30, 0.15, name+"(OOS)", start_date=pd.Timestamp("2023-01-01"))
    results_all[k] = a
    results_oos[k] = o

print(f"\n{'='*60}")
print(f"汇总:")
for k,name in labels:
    if results_all[k] and results_oos[k]:
        a, o = results_all[k], results_oos[k]
        d_sr = float(a["sr"]) - float(o["sr"]) if o["sr"] != "N/A" and a["sr"] != "N/A" else 0
        chk = "✅ 真实" if abs(d_sr) < 0.3 else "⚠️ 存疑"
        print(f"  {name:12s}: 全样夏普 {a['sr']:>6s} | OOS夏普 {o['sr']:>6s} | 差距 {d_sr:+.2f} | {chk}")

result = {"full": {k:results_all[k] for k in results_all if results_all[k]},
          "oos": {k:results_oos[k] for k in results_oos if results_oos[k]}}
json.dump(result, open("output/backtest_v29_oos.json","w"), indent=2, default=str)
print(f"\n✅ output/backtest_v29_oos.json")
print(f"⏱ {(time.time()-tt)/60:.1f}分")
