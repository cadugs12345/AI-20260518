#!/usr/bin/env python3
"""
backtest_v39_label_compare.py — v31 等权收益标签 去重后重测

目的：确认 rank 标签 vs 等权收益标签的真实差距（去重后）
- v31: fwd_20d_ret（等权 20 日收益标签）
- v36c: fwd_20d_rank（截面 rank 标签）

两者都用同一套回测框架：指数衰减权重 + 行业中性
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
print("v39 标签对比（去重后）: rank vs 等权收益", flush=True)
print(time.strftime('%F %H:%M'), flush=True)
print("="*60, flush=True)

# ====== 数据 ======
print("\n加载数据...", flush=True)
panel = pd.read_parquet("data/factors/factor_panel_v6.parquet")
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
panel["ts_code"] = panel["ts_code"].astype(str)

prices = pd.read_parquet("data/factors/full_prices.parquet")
prices["trade_date"] = pd.to_datetime(prices["trade_date"])
prices = prices.sort_values(["ts_code", "trade_date"])
prices["ret_1d"] = prices.groupby("ts_code")["close"].pct_change()
print(f"  面板: {len(panel):,}行 | 价格: {len(prices):,}行", flush=True)

# ====== 构建两种标签 ======
print("\n构建标签...", flush=True)

# 法务标签（分类用）
mask = panel["fwd_20d_ret"].notna() & (panel["fwd_20d_ret"].abs() < 0.5)

# 标签A: 等权 20 日收益（v31 原版）
panel["label_ret"] = panel["fwd_20d_ret"].astype(float)

# 标签B: 截面 rank（v36c）
panel["label_rank"] = np.nan
panel.loc[mask, "label_rank"] = (
    panel[mask].groupby("trade_date")["fwd_20d_ret"]
    .rank(pct=True, ascending=True)
)
print(f"  label_ret: 有标签={panel['label_ret'].notna().sum():,}")
print(f"  label_rank: 有标签={panel['label_rank'].notna().sum():,}", flush=True)

# ====== 回测参数 ======
ps = prices.sort_values(["ts_code","trade_date"]).copy()
ps["vol_60d_ann"] = ps.groupby("ts_code")["ret_1d"].transform(
    lambda x: x.rolling(60, min_periods=20).std()) * np.sqrt(244)

pdts = sorted(panel["trade_date"].unique())
period_dates = [pdts[i] for i in range(0, len(pdts), 20)
                if pdts[i] >= pd.Timestamp("2021-01-01")]

half_dates = []
for d in period_dates:
    k = f"{d.year}{'H1' if d.month <= 6 else 'H2'}"
    if not half_dates or half_dates[-1][0] != k:
        half_dates.append((k, d))
    else:
        half_dates[-1] = (k, d)

rf_md = joblib.load("models/ml_ensemble_v1.joblib")
factor_cols = rf_md["factor_cols"]
print(f"  因子: {len(factor_cols)}个 | 换仓期: {len(period_dates)}", flush=True)


# ====== 滚动训练 ======
def train_lgb(label_col, name=""):
    """用指定标签训练，返回去重后的预测表"""
    preds = {"trade_date": [], "ts_code": [], "pred_ret": []}
    
    for hi, (hk, train_cutoff) in enumerate(half_dates):
        if hi < 3:
            continue
        
        train_end = train_cutoff - pd.Timedelta(days=5)
        tr = panel[
            (panel["trade_date"] >= train_end - pd.Timedelta(days=730)) &
            (panel["trade_date"] <= train_end) &
            panel[label_col].notna()
        ].copy()
        
        # 对收益标签做极端值裁剪
        if label_col != "label_rank":
            lb = tr[label_col]
            lo_, hi_ = lb.quantile(0.01), lb.quantile(0.99)
            tr = tr[(tr[label_col] >= lo_) & (tr[label_col] <= hi_)]
        
        if len(tr) < 20000:
            continue
        if len(tr) > 100000:
            tr = tr.sample(100000, random_state=42)
        
        X_tr = tr[factor_cols].fillna(0).values.astype(np.float32)
        y_tr = tr[label_col].values.astype(np.float32)
        nv = max(1, int(len(tr) * 0.15))
        
        m = lgb.LGBMRegressor(
            n_estimators=500, max_depth=3, learning_rate=0.02,
            subsample=0.7, colsample_bytree=0.7,
            reg_alpha=0.2, reg_lambda=1.0,
            min_child_weight=20, min_data_in_leaf=100,
            random_state=42, verbose=-1, n_jobs=8,
        )
        m.fit(
            X_tr[:-nv], y_tr[:-nv],
            eval_set=[(X_tr[-nv:], y_tr[-nv:])],
            callbacks=[lgb.early_stopping(30, verbose=False)],
            eval_metric="mse",
        )
        
        for d in period_dates:
            if d <= train_cutoff:
                continue
            day = panel[panel["trade_date"] == d]
            if len(day) == 0:
                continue
            pp = m.predict(day[factor_cols].fillna(0).values.astype(np.float32))
            for j, code in enumerate(day["ts_code"].values):
                preds["trade_date"].append(d)
                preds["ts_code"].append(code)
                preds["pred_ret"].append(float(pp[j]))
        
        if (hi + 1) % 3 == 0:
            print(f"  {name}: {hk} ({hi+1}/{len(half_dates)-3})", flush=True)
    
    pdf = pd.DataFrame(preds)
    # 去重：同一 date+code 保留首次预测
    pdf = pdf.drop_duplicates(subset=["trade_date", "ts_code"], keep="first")
    print(f"  {name}: {len(pdf):,}预测, {pdf['trade_date'].nunique()}期", flush=True)
    return pdf


# ====== 回测引擎 ======
def backtest(pred_df, n_stocks=30, target_vol=0.15, label="",
             min_date=pd.Timestamp("2023-01-01")):
    import tushare as ts
    from config.settings import TS_TOKEN
    ts.set_token(TS_TOKEN)
    pro = ts.pro_api()
    stk = pro.query("stock_basic", exchange="", list_status="L",
                    fields="ts_code,industry")
    si = dict(zip(stk["ts_code"], stk["industry"]))
    
    pred_dates = sorted(pred_df["trade_date"].unique())
    pred_dates = [d for d in pred_dates if d >= min_date]
    if len(pred_dates) < 2:
        return None
    
    cash = 0.03
    holdings = {}
    navs = [1.0]
    
    for i in range(len(pred_dates) - 1):
        date = pred_dates[i]
        sell_date = pred_dates[i + 1]
        
        px_buy = {}
        px_sell = {}
        sv = {}
        for _, r in ps[ps["trade_date"] == date].iterrows():
            px_buy[r["ts_code"]] = r["close"]
            sv[r["ts_code"]] = (r["vol_60d_ann"]
                                if pd.notna(r.get("vol_60d_ann")) else 0.3)
        for _, r in ps[ps["trade_date"] == sell_date].iterrows():
            px_sell[r["ts_code"]] = r["close"]
        
        hv = sum(shares * px_buy.get(c, 0) for c, shares in holdings.items())
        tv_ = hv + cash
        sp = 0
        for c, shares in holdings.items():
            px = px_sell.get(c, 0)
            if px > 0:
                sp += shares * px - shares * px * (STAMP + COMM + SLIP)
        cash += sp
        holdings = {}
        
        day = pred_df[pred_df["trade_date"] == date].sort_values(
            "pred_ret", ascending=False).reset_index(drop=True)
        codes = list(day["ts_code"])
        scores = day["pred_ret"].values
        
        # 行业中性选择
        selected_idx = []
        ind_count = {}
        order = np.argsort(-scores)
        for j in order:
            ind = si.get(codes[j], "其他")
            if ind_count.get(ind, 0) < 3:
                selected_idx.append(j)
                ind_count[ind] = ind_count.get(ind, 0) + 1
            if len(selected_idx) >= n_stocks:
                break
        if len(selected_idx) < n_stocks:
            for j in order:
                if j not in selected_idx:
                    selected_idx.append(j)
                    if len(selected_idx) >= n_stocks:
                        break
        
        sel_codes = [codes[j] for j in selected_idx]
        
        # 指数衰减权重
        r = np.arange(1, len(sel_codes) + 1)
        weights = np.exp(-0.1 * r)
        weights = weights / weights.sum()
        
        sl = [sv.get(c, np.nan) for c in sel_codes]
        sl = [v for v in sl if not np.isnan(v) and v > 0.01]
        pr = max(min(target_vol / np.median(sl), 1.0)
                 if len(sl) >= 5 else 1.0, 0.05)
        
        if sel_codes and cash > 0.001:
            alloc = cash * pr * 0.98
            if alloc > 0.001:
                for code, wt in zip(sel_codes, weights):
                    px = px_buy.get(code, 0)
                    if px > 0 and wt > 0:
                        b = (alloc * wt) * (1 - COMM - SLIP) / px
                        if b > 0:
                            holdings[code] = b
                cash -= alloc
        
        np_ = sum(shares * px_sell.get(c, 0)
                  for c, shares in holdings.items())
        nt = np_ + cash
        ret = nt / tv_ - 1 if tv_ > 0 else 0
        navs.append(navs[-1] * (1 + ret))
    
    pnl = np.array(navs[1:]) / np.array(navs[:-1]) - 1
    na = np.array(navs)
    ny = len(pnl) / 13
    ar = na[-1] ** (1 / ny) - 1 if ny > 0 and na[-1] > 0 else 0
    sr = (np.mean(pnl) / np.std(pnl) * np.sqrt(13)) if np.std(pnl) > 0 else 0
    dd = np.maximum.accumulate(na) - na
    mdd = dd.max()
    wr = np.mean(pnl > 0)
    cal = ar / mdd if mdd > 0 else 0
    
    print(f"  {label:32s}: 年化{ar*100:+7.1f}% | 夏普{sr:5.2f} | "
          f"回撤{mdd*100:5.1f}% | 胜率{wr*100:3.0f}% | 卡玛{cal:.2f} | "
          f"{len(pnl)}期", flush=True)
    
    return {
        "label": label,
        "navs": [float(x) for x in navs],
        "ar": float(ar),
        "sr": float(sr),
        "mdd": float(mdd),
        "wr": float(wr),
        "cal": float(cal),
        "n_periods": len(pnl),
    }


# ====== 执行 ======
# 1. 训练两个标签
print("\n--- 训练: 等权收益标签 ---", flush=True)
pred_ret = train_lgb("label_ret", "v31 收益标签")

print("\n--- 训练: rank标签 ---", flush=True)
pred_rank = train_lgb("label_rank", "v36c rank标签")

# IC分析
print("\n--- IC对比 ---", flush=True)
for name, pdf in [("v31 收益标签", pred_ret), ("v36c rank标签", pred_rank)]:
    ics = []
    for d in pdf["trade_date"].unique():
        day = pdf[pdf["trade_date"] == d]
        pday = panel[panel["trade_date"] == d]
        m = day.merge(pday[["ts_code", "fwd_20d_ret"]], on="ts_code")
        if len(m) > 10:
            ic, _ = spearmanr(m["pred_ret"], m["fwd_20d_ret"])
            if not np.isnan(ic):
                ics.append(ic)
    if ics:
        print(f"  {name:20s}: IC={np.mean(ics)*100:+.2f}%, IR={np.mean(ics)/np.std(ics):.2f} ({len(ics)}期)", flush=True)

# 保存预测
pred_ret.to_parquet("output/pred_v39_label_ret.parquet", index=False)
pred_rank.to_parquet("output/pred_v39_label_rank.parquet", index=False)
print("\n  预测已保存", flush=True)

# 2. 等权对比
print(f"\n{'='*60}", flush=True)
print("T30 指衰+行业中性 2023-2026 (去重后)", flush=True)
print(f"{'='*60}", flush=True)

results = {}
for lname, pdf in [("v31 等权收益(去重)", pred_ret), ("v36c rank标签(去重)", pred_rank)]:
    r = backtest(pdf, 30, 0.15, lname)
    if r:
        results[lname] = r

# 汇总
print(f"\n{'='*60}", flush=True)
print("结果汇总：", flush=True)
print(f"{'配置':32s} {'年化':>7s} {'夏普':>6s} {'回撤':>6s} {'胜率':>4s} {'卡玛':>5s}", flush=True)
print("-"*60, flush=True)
for lname, r in results.items():
    print(f"{lname:32s} {r['ar']*100:>6.1f}% {r['sr']:>5.2f} "
          f"{r['mdd']*100:>5.1f}% {r['wr']*100:>3.0f}% {r['cal']:>4.2f}", flush=True)

# 保存
json.dump(results, open("output/backtest_v39_label_compare.json", "w"),
          indent=2, default=str)
print(f"\n✅ 完成 | ⏱ {(time.time()-tt)/60:.1f}分", flush=True)
print(f"  结果: output/backtest_v39_label_compare.json", flush=True)
