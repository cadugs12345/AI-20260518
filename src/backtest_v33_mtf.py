"""
backtest_v33_mtf.py — 多时间框架信号融合 (Multi-Time-Frame)
用LGB分别预测 fwd_5d_ret / fwd_10d_ret / fwd_20d_ret
然后融合策略：
  1. 共识增强（三框架一致看多→重仓）
  2. 分歧信号（短期弱但长期强→反转机会 / 短期强但长期弱→警惕）
对比基准：v31 指数衰减+行业中性 (夏普1.25)
"""
import os, sys, time, json, gc
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
os.chdir("/mnt/d/AI-20260518"); sys.path.insert(0, '.')
import joblib, lightgbm as lgb
from scipy.stats import spearmanr
from datetime import datetime, timedelta
tt = time.time()

print("="*60, flush=True)
print("v33 多时间框架信号融合 (5d/10d/20d)", flush=True)
print(datetime.now().strftime('%F %H:%M'), flush=True)
print("="*60, flush=True)

# ========== 数据加载 ==========
panel = pd.read_parquet("data/factors/factor_panel_v6.parquet")
prices = pd.read_parquet("data/factors/full_prices.parquet")
prices["trade_date"] = pd.to_datetime(prices["trade_date"])
panel["trade_date"] = pd.to_datetime(panel["trade_date"])

fac_base = [c for c in panel.columns if c not in ["ts_code","trade_date","fwd_20d_ret","close","ret_1d",
    "__fragment_index","__batch_index","__last_in_fragment","__filename","board_break"]]

# ========== 构建多时间框架标签 ==========
print("\n构建多时间框架标签...", flush=True)
ps = prices.sort_values(["ts_code","trade_date"]).copy()
ps["ret_1d"] = ps.groupby("ts_code")["close"].pct_change()
ps["vol_60d"] = ps.groupby("ts_code")["ret_1d"].transform(lambda x: x.rolling(60, min_periods=20).std())
ps["vol_60d_ann"] = ps["vol_60d"] * np.sqrt(244)

# 计算5d/10d/20d前向收益
for fwd in [5, 10, 20]:
    ps[f"fwd_{fwd}d_ret"] = ps.groupby("ts_code")["ret_1d"].transform(
        lambda x: x.shift(-fwd).rolling(fwd, min_periods=max(3, fwd//2)).sum()
    )

# 合并到panel（排除panel_v6已有的fwd_20d_ret）
ps_merge = ps[["ts_code","trade_date","fwd_5d_ret","fwd_10d_ret","vol_60d_ann"]].copy()
ps_merge = ps_merge.rename(columns={"fwd_5d_ret":"fwd_5d_ret","fwd_10d_ret":"fwd_10d_ret"})
panel = panel.merge(ps_merge, on=["ts_code","trade_date"], how="left")

# 构造panel日期序列
pdts = sorted(panel["trade_date"].unique())
period_dates = [pdts[i] for i in range(0, len(pdts), 20) if pdts[i] >= pd.Timestamp("2021-01-01")]
print(f"回测期数: {len(period_dates)}", flush=True)

# 半年度滚动
half_dates = []
for d in period_dates:
    k = f"{d.year}{'H1' if d.month<=6 else 'H2'}"
    if not half_dates or half_dates[-1][0]!=k: half_dates.append((k,d))
    else: half_dates[-1]=(k,d)
print(f"训练窗口: {len(half_dates)} 半年度 ({half_dates[0][0]}~{half_dates[-1][0]})", flush=True)

# 行业
import tushare as ts
from config.settings import TS_TOKEN
ts.set_token(TS_TOKEN)
pro = ts.pro_api()
stk_basic = pro.query("stock_basic", exchange="", list_status="L", fields="ts_code,industry")
stk_ind = dict(zip(stk_basic["ts_code"], stk_basic["industry"]))

# 基础因子
rf_md = joblib.load("models/ml_ensemble_v1.joblib")
rf_factors = rf_md["factor_cols"]

# ========== 多时间框架LGB训练 ==========
def train_lgb(tr_X, tr_y, val_X, val_y):
    m = lgb.LGBMRegressor(n_estimators=500, max_depth=3, lr=0.02,
        subsample=0.7, colsample_bytree=0.7, reg_alpha=0.2, reg_lambda=1.0,
        min_child_weight=20, min_data_in_leaf=100, random_state=42, verbose=-1, n_jobs=8)
    m.fit(tr_X, tr_y, eval_set=[(val_X, val_y)],
          callbacks=[lgb.early_stopping(30, verbose=False)], eval_metric="mse")
    return m

def rolling_predict_mtf():
    """滚动训练三个框架的LGB模型"""
    all_preds = {"trade_date":[],"ts_code":[],"pred_5d":[],"pred_10d":[],"pred_20d":[]}
    
    for hi,(hk,train_cutoff) in enumerate(half_dates):
        if hi < 3: continue
        train_end = train_cutoff - pd.Timedelta(days=5)
        tr = panel[(panel["trade_date"]>=train_end-pd.Timedelta(days=730)) & (panel["trade_date"]<=train_end)]
        tr = tr[tr["fwd_20d_ret"].notna() & (tr["fwd_20d_ret"].abs()<0.5)]
        if len(tr)<20000: continue
        if len(tr)>100000: tr=tr.sample(100000,random_state=42)
        
        X_tr_full = tr[rf_factors].fillna(0).values.astype(np.float32)
        nv = max(1,int(len(tr)*0.15))
        
        models = {}
        for fwd_label, fwd_col in [("5d","fwd_5d_ret"), ("10d","fwd_10d_ret"), ("20d","fwd_20d_ret")]:
            y_vals = tr[fwd_col].values.astype(np.float32)
            valid = ~np.isnan(y_vals)
            y = np.clip(y_vals[valid], -0.3, 0.3)
            X_local = X_tr_full[valid]
            nv_local = max(1, int(len(y)*0.15))
            models[fwd_label] = train_lgb(X_local[:-nv_local], y[:-nv_local], X_local[-nv_local:], y[-nv_local:])
        
        for d in period_dates:
            if d <= train_cutoff: continue
            day = panel[panel["trade_date"]==d]
            if len(day)==0: continue
            X_te = day[rf_factors].fillna(0).values.astype(np.float32)
            
            p5 = models["5d"].predict(X_te)
            p10 = models["10d"].predict(X_te)
            p20 = models["20d"].predict(X_te)
            
            for j, code in enumerate(day["ts_code"].values):
                all_preds["trade_date"].append(d)
                all_preds["ts_code"].append(code)
                all_preds["pred_5d"].append(float(p5[j]))
                all_preds["pred_10d"].append(float(p10[j]))
                all_preds["pred_20d"].append(float(p20[j]))
        
        if (hi+1)%3==0:
            print(f"  {hk}: {hi+1}/{len(half_dates)-3}期完成", flush=True)
    
    pdf = pd.DataFrame(all_preds)
    print(f"  总预测: {len(pdf):,}, {pdf['trade_date'].nunique()}期", flush=True)
    return pdf

print(f"\n滚动训练中 (3个LGB×{len(half_dates)-3}个窗口)...", flush=True)
tt2 = time.time()
preds_mtf = rolling_predict_mtf()
print(f"  训练用时: {(time.time()-tt2)/60:.1f}分", flush=True)

# ========== 融合信号 ==========
# 将三个时间框架的预测归一化后融合
preds_mtf["pred_5d_z"] = preds_mtf.groupby("trade_date")["pred_5d"].transform(
    lambda x: (x - x.mean()) / (x.std() + 0.001))
preds_mtf["pred_10d_z"] = preds_mtf.groupby("trade_date")["pred_10d"].transform(
    lambda x: (x - x.mean()) / (x.std() + 0.001))
preds_mtf["pred_20d_z"] = preds_mtf.groupby("trade_date")["pred_20d"].transform(
    lambda x: (x - x.mean()) / (x.std() + 0.001))

# 融合策略1: 等权平均
preds_mtf["ensemble_mean"] = (preds_mtf["pred_5d_z"] + preds_mtf["pred_10d_z"] + preds_mtf["pred_20d_z"]) / 3

# 融合策略2: 共识增强（三框架一致性加权）
std_cols = preds_mtf[["pred_5d_z","pred_10d_z","pred_20d_z"]].values
mean_cols = preds_mtf[["pred_5d_z","pred_10d_z","pred_20d_z"]].mean(axis=1)
# 共识度：框架间标准差倒数
consensus = 1.0 / (np.std(std_cols, axis=1) + 0.5)
preds_mtf["consensus_weighted"] = mean_cols * consensus

# 融合策略3: 分歧信号（短期vs长期背离）
preds_mtf["divergence_5v20"] = (preds_mtf["pred_5d_z"] - preds_mtf["pred_20d_z"])  # >0: 短期强于长期
# 分歧策略：短期走强长期走弱=警惕(负信号)；短期走弱长期走强=低估机会(正信号)
preds_mtf["divergence_5v20_signal"] = -(preds_mtf["pred_5d_z"] - preds_mtf["pred_20d_z"]) * preds_mtf["pred_20d_z"]

# 融合策略4: 仅用20d（v31对照）
preds_mtf["v31_ref"] = preds_mtf["pred_20d_z"]

# ========== IC分析 ==========
for label, col in [("5d","pred_5d"),("10d","pred_10d"),("20d","pred_20d"),
                   ("ensemble","ensemble_mean"),("consensus","consensus_weighted"),
                   ("divergence","divergence_5v20_signal"),("v31_ref","v31_ref")]:
    fwd_map = {"5d":"fwd_5d_ret","10d":"fwd_10d_ret","20d":"fwd_20d_ret","ensemble":"fwd_20d_ret",
               "consensus":"fwd_20d_ret","divergence":"fwd_20d_ret","v31_ref":"fwd_20d_ret"}
    ics=[]
    for d in preds_mtf["trade_date"].unique():
        day=preds_mtf[preds_mtf["trade_date"]==d]
        pday=panel[panel["trade_date"]==d]
        m=day.merge(pday[["ts_code",fwd_map[label]]],on="ts_code")
        if len(m)>10:
            ic,_=spearmanr(m[col],m[fwd_map[label]])
            if not np.isnan(ic): ics.append(ic)
    if ics:
        print(f"IC {label:15s}: {np.mean(ics)*100:+7.2f}% IR: {np.mean(ics)/np.std(ics):.2f} ({len(ics)}期)", flush=True)

# ========== 回测 ==========
STAMP, COMM, SLIP = 0.001, 0.0002, 0.002

def backtest_v27_engine(pred_df, n_stocks=30, target_vol=0.15, label="", min_date=None):
    """复用v27已验证的回测引擎"""
    pred_dates = sorted(pred_df["trade_date"].unique())
    if min_date: pred_dates=[d for d in pred_dates if d>=min_date]
    if len(pred_dates)<2: return None
    cash = 0.03; holdings = {}; navs = [1.0]
    
    for i in range(len(pred_dates)-1):
        date = pred_dates[i]; sell_date = pred_dates[i+1]
        px_buy = {}
        for _,r in ps[ps["trade_date"]==date].iterrows():
            px_buy[r["ts_code"]] = r["close"]
        px_sell = {}
        for _,r in ps[ps["trade_date"]==sell_date].iterrows():
            px_sell[r["ts_code"]] = r["close"]
        stock_vol = {}
        for _,r in ps[ps["trade_date"]==date].iterrows():
            stock_vol[r["ts_code"]] = r["vol_60d_ann"] if pd.notna(r.get("vol_60d_ann")) else 0.3
        
        hold_val = sum(shares * px_buy.get(c,0) for c,shares in holdings.items())
        total_val = hold_val + cash
        sell_proceeds = 0
        for code, shares in holdings.items():
            px = px_sell.get(code, 0)
            if px > 0:
                val = shares * px
                sell_proceeds += val - val*(STAMP+COMM+SLIP)
        cash = cash + sell_proceeds
        holdings = {}
        
        day_pred = pred_df[pred_df["trade_date"]==date].sort_values("pred_ret", ascending=False)
        selected = list(day_pred.head(n_stocks)["ts_code"].values)
        
        selected_vols = [stock_vol.get(c, np.nan) for c in selected]
        selected_vols = [v for v in selected_vols if not np.isnan(v) and v > 0.01]
        if len(selected_vols) >= 5:
            pos_ratio = max(min(target_vol / np.median(selected_vols), 1.0), 0.05)
        else:
            pos_ratio = 1.0
        
        if selected and cash > 0.001:
            available = cash * pos_ratio * 0.98
            if available > 0.001:
                per = available / len(selected)
                for code in selected:
                    px = px_buy.get(code, 0)
                    if px > 0 and per > 0:
                        bought = (per - per*(COMM+SLIP)) / px
                        if bought > 0: holdings[code] = bought
                cash -= per * len(holdings)
        
        new_port = sum(shares * px_sell.get(c,0) for c,shares in holdings.items())
        new_total = new_port + cash
        ret = new_total/total_val - 1 if total_val > 0 else 0
        navs.append(navs[-1] * (1+ret))
    
    pnl = np.array(navs[1:])/np.array(navs[:-1])-1
    nav_arr = np.array(navs)
    n_years = len(pnl)/13
    ar = nav_arr[-1]**(1/n_years)-1 if n_years>0 and nav_arr[-1]>0 else 0
    sr = np.mean(pnl)/np.std(pnl)*np.sqrt(13) if np.std(pnl)>0 else 0
    dd = np.maximum.accumulate(nav_arr) - nav_arr; mdd = dd.max()
    wr = np.mean(pnl>0); cal = ar/mdd if mdd>0 else 0
    print(f"  {label:32s}: 年化{ar*100:+7.1f}% | 夏普{sr:5.2f} | 回撤{mdd*100:5.1f}% | 胜率{wr*100:3.0f}% | 卡玛{cal:.2f} | {len(pnl)}期", flush=True)
    return {"ret":f"{ar*100:+.1f}%","sr":f"{sr:.2f}","mdd":f"{mdd*100:.1f}%","num":len(pnl)}

print(f"\n{'='*60}", flush=True)
print("回测: T30 指数衰减+行业中性 目波15%", flush=True)
print("-"*60, flush=True)

# 基准
bt_input = preds_mtf.rename(columns={"v31_ref":"pred_ret"})[["trade_date","ts_code","pred_ret"]]
backtest_v27_engine(bt_input, 30, 0.15, "v31 20d-only基准", pd.Timestamp("2023-01-01"))

bt_input_eq = preds_mtf.rename(columns={"ensemble_mean":"pred_ret"})[["trade_date","ts_code","pred_ret"]]
backtest_v27_engine(bt_input_eq, 30, 0.15, "v33 等权融合(5d/10d/20d)", pd.Timestamp("2023-01-01"))

bt_input_cw = preds_mtf.rename(columns={"consensus_weighted":"pred_ret"})[["trade_date","ts_code","pred_ret"]]
backtest_v27_engine(bt_input_cw, 30, 0.15, "v33 共识加权增强", pd.Timestamp("2023-01-01"))

bt_input_div = preds_mtf.rename(columns={"divergence_5v20_signal":"pred_ret"})[["trade_date","ts_code","pred_ret"]]
backtest_v27_engine(bt_input_div, 30, 0.15, "v33 分歧信号(5v20背离)", pd.Timestamp("2023-01-01"))

bt_input_5d = preds_mtf.rename(columns={"pred_5d":"pred_ret"})[["trade_date","ts_code","pred_ret"]]
backtest_v27_engine(bt_input_5d, 30, 0.15, "v33 5d-only参考", pd.Timestamp("2023-01-01"))

print(f"\n{'='*60}", flush=True)
print(f"✅ 完成 | ⏱ {(time.time()-tt)/60:.1f}分", flush=True)
