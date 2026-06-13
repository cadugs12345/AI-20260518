#!/usr/bin/env python3
"""快速分年分析：rank+风控 vs 收益标签"""
import os, sys, json, time
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
os.chdir("/mnt/d/AI-20260518"); sys.path.insert(0, '.')
import tushare as ts; from config.settings import TS_TOKEN
ts.set_token(TS_TOKEN); pro = ts.pro_api()
stk = pro.query("stock_basic", exchange="", list_status="L", fields="ts_code,industry")
si = dict(zip(stk["ts_code"], stk["industry"]))
tt = time.time()
STAMP, COMM, SLIP = 0.001, 0.0002, 0.002

prices = pd.read_parquet("data/factors/full_prices.parquet")
prices["trade_date"] = pd.to_datetime(prices["trade_date"])
prices["ret_1d"] = prices.groupby("ts_code")["close"].pct_change()
ps = prices.sort_values(["ts_code","trade_date"]).copy()
ps["vol_60d_ann"] = ps.groupby("ts_code")["ret_1d"].transform(lambda x: x.rolling(60, min_periods=20).std()) * np.sqrt(244)

pred_ret = pd.read_parquet("output/pred_v39_label_ret.parquet")
pred_rank = pd.read_parquet("output/pred_v39_label_rank.parquet")
panel = pd.read_parquet("data/factors/factor_panel_v6.parquet", columns=["ts_code","trade_date","repair_force_10d","高波反转"])
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
pred_ret = pred_ret.merge(panel, on=["ts_code","trade_date"], how="left")
pred_rank = pred_rank.merge(panel, on=["ts_code","trade_date"], how="left")

def backtest(pred_df, risk=True, label=""):
    pred_dates = sorted(pred_df["trade_date"].unique())
    pred_dates = [d for d in pred_dates if d >= pd.Timestamp("2021-01-01")]
    if len(pred_dates) < 2: return {}
    
    results = {}
    years = sorted(set(d.year for d in pred_dates))
    
    for yr in years:
        year_dates = [d for d in pred_dates if d.year == yr]
        if len(year_dates) < 2:
            continue
        
        cash = 0.03; holdings = {}; navs = [1.0]
        for i in range(len(year_dates) - 1):
            date = year_dates[i]; sell_date = year_dates[i+1]
            px_buy = {}; px_sell = {}; sv = {}
            for _, r in ps[ps["trade_date"] == date].iterrows():
                px_buy[r["ts_code"]] = r["close"]
                sv[r["ts_code"]] = r["vol_60d_ann"] if pd.notna(r.get("vol_60d_ann")) else 0.3
            for _, r in ps[ps["trade_date"] == sell_date].iterrows():
                px_sell[r["ts_code"]] = r["close"]
            hv = sum(shares * px_buy.get(c,0) for c,shares in holdings.items()); tv_ = hv + cash
            sp = 0
            for c,shares in holdings.items():
                px = px_sell.get(c,0)
                if px > 0: sp += shares*px - shares*px*(STAMP+COMM+SLIP)
            cash += sp; holdings = {}
            day = pred_df[pred_df["trade_date"]==date].sort_values("pred_ret",ascending=False).reset_index(drop=True)
            codes = list(day["ts_code"])
            if risk:
                r10 = day["repair_force_10d"].values.astype(float)
                hv_ = day["高波反转"].values.astype(float)
                rmask = (r10 < -0.05) | (hv_ < -0.03)
                safe_idx = np.where(~rmask)[0]
            else:
                safe_idx = np.arange(len(day))
            sel=[]; ic={}
            for j in safe_idx:
                ind = si.get(codes[j],"其他")
                if ic.get(ind,0)<3: sel.append(codes[j]); ic[ind]=ic.get(ind,0)+1
                if len(sel)>=30: break
            if len(sel)<30:
                for j in safe_idx:
                    if len(sel)>=30: break
                    if codes[j] not in sel: sel.append(codes[j])
            rw = np.arange(1,len(sel)+1); w = np.exp(-0.1*rw); w = w/w.sum()
            sl = [sv.get(c,np.nan) for c in sel]; sl = [v for v in sl if not np.isnan(v) and v>0.01]
            pr = max(min(0.15/np.median(sl),1.0) if len(sl)>=5 else 1.0, 0.05)
            if sel and cash>0.001:
                al = cash*pr*0.98
                if al>0.001:
                    for code,wt in zip(sel,w):
                        px = px_buy.get(code,0)
                        if px>0 and wt>0:
                            b = (al*wt)*(1-COMM-SLIP)/px
                            if b>0: holdings[code]=b
                    cash -= al
            np_ = sum(shares*px_sell.get(c,0) for c,shares in holdings.items())
            nt = np_+cash; ret = nt/tv_-1 if tv_>0 else 0
            navs.append(navs[-1]*(1+ret))
        pnl = np.array(navs[1:])/np.array(navs[:-1])-1; na = np.array(navs)
        n_y = len(pnl)/13; ar = na[-1]**(1/n_y)-1 if n_y>0 and na[-1]>0 else 0
        sr = np.mean(pnl)/np.std(pnl)*np.sqrt(13) if np.std(pnl)>0 else 0
        dd = np.maximum.accumulate(na)-na; mdd = dd.max()
        wr = np.mean(pnl>0)
        results[yr] = {"ar": ar, "sr": sr, "mdd": mdd, "wr": wr, "n": len(pnl)}
        print(f"  {yr}: 年化{ar*100:+6.1f}% 夏普{sr:5.2f} 回撤{mdd*100:5.1f}% 胜率{wr*100:2.0f}% ({len(pnl)}期)", flush=True)
    return results

print("=== rank+风控 分年 ===", flush=True)
r1 = backtest(pred_rank, risk=True, label="rank+风控")

print("\n=== rank+无风控 分年 ===", flush=True)
r2 = backtest(pred_rank, risk=False, label="rank+无风控")

print("\n=== 收益标签+风控 分年 ===", flush=True)
r3 = backtest(pred_ret, risk=True, label="收益+风控")

# 汇总
print(f"\n{'='*65}", flush=True)
print(f"{'年份':>4s} {'rank风控夏普':>12s} {'rank无风控夏普':>14s} {'收益风控夏普':>14s} {'rank风控回撤':>12s} {'收益风控回撤':>14s}", flush=True)
print("-"*65, flush=True)
for yr in sorted(set(list(r1.keys()) + list(r2.keys()) + list(r3.keys()))):
    s1 = f"{r1.get(yr,{}).get('sr',0):.2f}" if yr in r1 else "-"
    s2 = f"{r2.get(yr,{}).get('sr',0):.2f}" if yr in r2 else "-"
    s3 = f"{r3.get(yr,{}).get('sr',0):.2f}" if yr in r3 else "-"
    d1 = f"{r1.get(yr,{}).get('mdd',0)*100:.1f}%" if yr in r1 else "-"
    d3 = f"{r3.get(yr,{}).get('mdd',0)*100:.1f}%" if yr in r3 else "-"
    print(f"{yr:>4s} {s1:>12s} {s2:>14s} {s3:>14s} {d1:>12s} {d3:>14s}", flush=True)

print(f"\n⏱ {(time.time()-tt)/60:.1f}分", flush=True)
