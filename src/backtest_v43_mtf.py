#!/usr/bin/env python3
"""
v43 多时间框架信号融合

核心思路：
训练 3 个 LGB 模型（5日/10日/20日预测），用它们的共识/分歧来判断信号质量。
- 共识高 → 权重上调
- 分歧大 → 权重下修

分三步：
1. 构建 5日/10日/20日 forward returns
2. 训练 3 个模型
3. 回测验证融合效果
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
print("v43 多时间框架信号融合", flush=True)
print(time.strftime('%F %H:%M'), flush=True)
print("="*60, flush=True)

# ====== 数据 ======
panel = pd.read_parquet("data/factors/factor_panel_v6.parquet")
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
panel["ts_code"] = panel["ts_code"].astype(str)

# 从价格数据构建5日/10日 forward returns
prices = pd.read_parquet("data/factors/full_prices.parquet")
prices["trade_date"] = pd.to_datetime(prices["trade_date"])
prices = prices.sort_values(["ts_code","trade_date"])
prices["ret_1d"] = prices.groupby("ts_code")["close"].pct_change()
prices["fwd_1d"] = prices.groupby("ts_code")["ret_1d"].shift(-1)
prices["fwd_5d"] = prices.groupby("ts_code")["ret_1d"].transform(lambda x: x.shift(-1).rolling(5, min_periods=3).sum())
prices["fwd_10d"] = prices.groupby("ts_code")["ret_1d"].transform(lambda x: x.shift(-1).rolling(10, min_periods=5).sum())

# merge到panel
fwd = prices[["ts_code","trade_date","fwd_1d","fwd_5d","fwd_10d"]].copy()
panel = panel.merge(fwd, on=["ts_code","trade_date"], how="left")

# 已有 fwd_20d_ret
print(f"面板: {len(panel):,}行", flush=True)
print(f"fwd_5d: {panel['fwd_5d'].notna().sum():,}", flush=True)
print(f"fwd_10d: {panel['fwd_10d'].notna().sum():,}", flush=True)
print(f"fwd_20d_ret: {panel['fwd_20d_ret'].notna().sum():,}", flush=True)

# ====== 构建rank标签（3个时间框架） ======
print("\n构建rank标签...", flush=True)
for period, col in [(5, "fwd_5d"), (10, "fwd_10d"), (20, "fwd_20d_ret")]:
    label = f"label_rank_{period}d"
    mask = panel[col].notna() & (panel[col].abs() < 0.5)
    panel[label] = np.nan
    panel.loc[mask, label] = panel[mask].groupby("trade_date")[col].rank(pct=True, ascending=True)
    print(f"  {label}: {mask.sum():,}条", flush=True)

# ====== 回测参数 ======
ps = prices.sort_values(["ts_code","trade_date"]).copy()
ps["vol_60d_ann"] = ps.groupby("ts_code")["ret_1d"].transform(lambda x: x.rolling(60, min_periods=20).std()) * np.sqrt(244)

pdts = sorted(panel["trade_date"].unique())
period_dates = [pdts[i] for i in range(0, len(pdts), 20) if pdts[i] >= pd.Timestamp("2021-01-01")]

half_dates = []
for d in period_dates:
    k = f"{d.year}{'H1' if d.month<=6 else 'H2'}"
    if not half_dates or half_dates[-1][0] != k: half_dates.append((k,d))
    else: half_dates[-1]=(k,d)

rf_md = joblib.load("models/ml_ensemble_v1.joblib")
factor_cols = rf_md["factor_cols"]
print(f"\n因子: {len(factor_cols)}个 | 换仓期: {len(period_dates)}", flush=True)


# ====== 训练3个模型（5d/10d/20d） ======
def train_mtf(label_col, name=""):
    """训练模型并预测"""
    all_preds = {"trade_date":[],"ts_code":[],"pred_ret":[]}
    for hi,(hk,train_cutoff) in enumerate(half_dates):
        if hi<3: continue
        train_end = train_cutoff - pd.Timedelta(days=5)
        tr = panel[(panel["trade_date"]>=train_end-pd.Timedelta(days=730))&(panel["trade_date"]<=train_end)&panel[label_col].notna()].copy()
        if len(tr)<20000: continue
        if len(tr)>100000: tr=tr.sample(100000,random_state=42)
        X_tr=tr[factor_cols].fillna(0).values.astype(np.float32)
        y_tr=tr[label_col].values.astype(np.float32)
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
                all_preds["trade_date"].append(d); all_preds["ts_code"].append(code)
                all_preds["pred_ret"].append(float(pp[j]))
        if (hi+1)%3==0: print(f"  {name}: {hk} ({hi+1}/{len(half_dates)-3})", flush=True)
    pdf = pd.DataFrame(all_preds)
    pdf = pdf.drop_duplicates(subset=["trade_date","ts_code"], keep="first")
    print(f"  {name}: {len(pdf):,}预测, {pdf['trade_date'].nunique()}期", flush=True)
    return pdf

print("\n训练 5日模型...", flush=True)
p5 = train_mtf("label_rank_5d", "5d")
print("训练 10日模型...", flush=True)
p10 = train_mtf("label_rank_10d", "10d")
print("训练 20日模型...", flush=True)
p20 = train_mtf("label_rank_20d", "20d")

# 合并
p5.rename(columns={"pred_ret":"pred_5d"}, inplace=True)
p10.rename(columns={"pred_ret":"pred_10d"}, inplace=True)
p20.rename(columns={"pred_ret":"pred_20d"}, inplace=True)

merged = p5.merge(p10, on=["trade_date","ts_code"], how="outer")
merged = merged.merge(p20, on=["trade_date","ts_code"], how="outer")
merged = merged.dropna(subset=["pred_5d","pred_10d","pred_20d"])

# ==== 共识信号 ====
# 对每行rank标准化后取均值作为共识
for col in ["pred_5d","pred_10d","pred_20d"]:
    merged[f"{col}_rank"] = merged.groupby("trade_date")[col].rank(pct=True)

merged["consensus"] = merged[["pred_5d_rank","pred_10d_rank","pred_20d_rank"]].mean(axis=1)
merged["disagreement"] = merged[["pred_5d_rank","pred_10d_rank","pred_20d_rank"]].std(axis=1)
# 分歧调整：共识权重 × (1 - 分歧)
merged["consensus_weighted"] = merged["consensus"] * (1 - merged["disagreement"])

print(f"\n融合信号: {len(merged):,}行", flush=True)
print(f"  共识均值: {merged['consensus'].mean():.4f} ± {merged['consensus'].std():.4f}", flush=True)
print(f"  分歧均值: {merged['disagreement'].mean():.4f} ± {merged['disagreement'].std():.4f}", flush=True)

# IC对比
panel_for_ic = panel[["trade_date","ts_code","fwd_20d_ret"]].copy()
ic_df = merged.merge(panel_for_ic, on=["trade_date","ts_code"], how="left")
for col in ["pred_5d","pred_10d","pred_20d","consensus","consensus_weighted"]:
    ics = []
    for d in ic_df["trade_date"].unique():
        day = ic_df[ic_df["trade_date"]==d]
        if len(day)>10:
            ic,_ = spearmanr(day[col], day["fwd_20d_ret"])
            if not np.isnan(ic): ics.append(ic)
    if ics:
        print(f"  IC({col:23s}): {np.mean(ics)*100:+.2f}%, IR={np.mean(ics)/np.std(ics):.2f}", flush=True)


# ====== 回测 ======
def bt(pdf, score_col="consensus", label="", risk=True, min_date=pd.Timestamp("2023-01-01")):
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
        day = pdf[pdf["trade_date"]==date].sort_values(score_col,ascending=False).reset_index(drop=True)
        codes=list(day["ts_code"])
        safe = list(range(len(codes)))
        sel=[]; ic={}
        for j in safe:
            ind=si.get(codes[j],"其他")
            if ic.get(ind,0)<3: sel.append(j); ic[ind]=ic.get(ind,0)+1
            if len(sel)>=30: break
        if len(sel)<30:
            for j in safe:
                if j not in sel: sel.append(j)
                if len(sel)>=30: break
        sel_codes=[codes[j] for j in sel]
        rw=np.arange(1,len(sel_codes)+1); w=np.exp(-0.1*rw); w=w/w.sum()
        sl=[sv.get(c,np.nan) for c in sel_codes]; sl=[v for v in sl if not np.isnan(v) and v>0.01]
        pr=max(min(0.15/np.median(sl),1.0) if len(sl)>=5 else 1.0,0.05)
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
    print(f"  {label:32s}: 年化{ar*100:+7.1f}% | 夏普{sr:5.2f} | 回撤{mdd*100:5.1f}% | 胜率{wr*100:3.0f}% | 卡玛{cal:.2f} | {len(pnl)}期", flush=True)
    return {"ar":float(ar),"sr":float(sr),"mdd":float(mdd),"wr":float(wr),"cal":float(cal)}


print(f"\n{'='*60}", flush=True)
print("多时间框架信号对比 (T30 指衰+行业, 2023-2026)", flush=True)
print(f"{'='*60}", flush=True)

cfgs = [
    (merged, "pred_20d", "基准: 仅20d预测"),
    (merged, "consensus", "共识(5d+10d+20d均值)"),
    (merged, "consensus_weighted", "共识×(1-分歧)"),
    # 单时间框架对比
]
for pdf, sc, label in cfgs:
    bt(pdf, sc, label)

print(f"\n{'='*60}", flush=True)
print("各单时间框架对比", flush=True)
print(f"{'='*60}", flush=True)
for pdf, sc, label in [
    (merged, "pred_5d", "仅5d预测"),
    (merged, "pred_10d", "仅10d预测"),
    (merged, "pred_20d", "仅20d预测(基准)"),
]:
    bt(pdf, sc, label)

print(f"\n✅ 完成 | ⏱ {(time.time()-tt)/60:.1f}分", flush=True)
