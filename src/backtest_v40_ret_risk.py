#!/usr/bin/env python3
"""
v40 收益标签+风控对比

复用 v39 已训练的预测数据，对收益标签加风控看效果
"""
import os, sys, json, time
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
os.chdir("/mnt/d/AI-20260518"); sys.path.insert(0, '.')
import joblib
tt = time.time()

STAMP, COMM, SLIP = 0.001, 0.0002, 0.002
print("="*60, flush=True)
print("v40 收益标签+风控对比", flush=True)
print(time.strftime('%F %H:%M'), flush=True)
print("="*60, flush=True)

# 加载预测
pred_ret = pd.read_parquet("output/pred_v39_label_ret.parquet")
pred_rank = pd.read_parquet("output/pred_v39_label_rank.parquet")

# 为预测加上风控因子
panel = pd.read_parquet("data/factors/factor_panel_v6.parquet",
    columns=["ts_code","trade_date","repair_force_10d","高波反转"])
panel["trade_date"] = pd.to_datetime(panel["trade_date"])

pred_ret = pred_ret.merge(panel, on=["ts_code","trade_date"], how="left")
pred_rank = pred_rank.merge(panel, on=["ts_code","trade_date"], how="left")

print(f"收益标签预测: {len(pred_ret):,}行", flush=True)
print(f"rank标签预测: {len(pred_rank):,}行", flush=True)

# 价格 + vol数据
prices = pd.read_parquet("data/factors/full_prices.parquet")
prices["trade_date"] = pd.to_datetime(prices["trade_date"])
prices["ret_1d"] = prices.groupby("ts_code")["close"].pct_change()
ps = prices.sort_values(["ts_code","trade_date"]).copy()
ps["vol_60d_ann"] = ps.groupby("ts_code")["ret_1d"].transform(
    lambda x: x.rolling(60, min_periods=20).std()) * np.sqrt(244)


def backtest(pred_df, n_stocks=30, target_vol=0.15, label="",
             min_date=pd.Timestamp("2023-01-01"),
             risk_enabled=True):
    """回测引擎（可选风控）"""
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
    risk_log = {"total": 0, "removed": 0}
    
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
        
        # 风控过滤
        risk_log["total"] += 1
        if risk_enabled:
            r10 = day["repair_force_10d"].values.astype(float)
            hv_ = day["高波反转"].values.astype(float)
            rmask = (r10 < -0.05) | (hv_ < -0.03)
            safe_idx = np.where(~rmask)[0]
            risk_log["removed"] += int(rmask.sum())
        else:
            safe_idx = np.arange(len(day))
        
        # 行业中性
        selected_idx = []
        ind_count = {}
        for j in safe_idx:
            ind = si.get(codes[j], "其他")
            if ind_count.get(ind, 0) < 3:
                selected_idx.append(j)
                ind_count[ind] = ind_count.get(ind, 0) + 1
            if len(selected_idx) >= n_stocks:
                break
        if len(selected_idx) < n_stocks:
            for j in safe_idx:
                if j not in selected_idx:
                    selected_idx.append(j)
                    if len(selected_idx) >= n_stocks:
                        break
        
        sel_codes = [codes[j] for j in selected_idx]
        
        r_w = np.arange(1, len(sel_codes) + 1)
        weights = np.exp(-0.1 * r_w)
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
    
    risk_note = ""
    if risk_enabled:
        risk_ratio = risk_log["removed"] / max(risk_log["total"], 1)
        risk_note = f" [风控剔除{risk_log['removed']}次]"
    
    print(f"  {label:32s}: 年化{ar*100:+7.1f}% | 夏普{sr:5.2f} | "
          f"回撤{mdd*100:5.1f}% | 胜率{wr*100:3.0f}% | 卡玛{cal:.2f} | "
          f"{len(pnl)}期{risk_note}", flush=True)
    
    return {
        "label": label,
        "ar": float(ar), "sr": float(sr), "mdd": float(mdd),
        "wr": float(wr), "cal": float(cal), "n_periods": len(pnl),
    }


# ====== 回测对比 ======
print(f"\n{'='*60}", flush=True)
print("T30 指衰+行业中性 2023-2026", flush=True)
print(f"{'='*60}", flush=True)

configs = [
    # (预测数据, 风控, 名字)
    (pred_rank, False, "rank+无风控"),
    (pred_rank, True,  "rank+风控(修复+高波)"),
    (pred_ret,  False, "收益+无风控"),
    (pred_ret,  True,  "收益+风控(修复+高波)"),
]

results = []
for pdf, risk, name in configs:
    r = backtest(pdf, 30, 0.15, name, risk_enabled=risk)
    if r:
        results.append(r)

# 汇总
print(f"\n{'='*60}", flush=True)
print("结果汇总：", flush=True)
print(f"{'配置':32s} {'年化':>7s} {'夏普':>6s} {'回撤':>6s} {'胜率':>4s} {'卡玛':>5s}", flush=True)
print("-"*60, flush=True)
for r in results:
    print(f"{r['label']:32s} {r['ar']*100:>6.1f}% {r['sr']:>5.2f} "
          f"{r['mdd']*100:>5.1f}% {r['wr']*100:>3.0f}% {r['cal']:>4.2f}", flush=True)

json.dump(results, open("output/backtest_v40_ret_risk.json", "w"), indent=2)
print(f"\n✅ 完成 | ⏱ {(time.time()-tt):.1f}s", flush=True)
