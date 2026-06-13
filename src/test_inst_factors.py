"""机构行为因子：股东户数+十大股东 IC测试"""
import sys, os, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings("ignore")
os.chdir("/mnt/d/AI-20260518")

import tushare as ts
sys.path.insert(0, '.')
from config.settings import TS_TOKEN
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

t0 = time.time()
codes = pd.read_parquet("data/raw/stock_list.parquet")["ts_code"].tolist()[:300]
print(f"测试: {len(codes)}只", flush=True)

all_holder, all_top10 = [], []
for i, code in enumerate(codes):
    try:
        dh = pro.query("stk_holdernumber", ts_code=code, start_date="20200101", end_date="20260519")
        if dh is not None and len(dh) > 0:
            dh["ts_code"] = code; all_holder.append(dh)
    except: pass
    try:
        dt = pro.query("top10_holders", ts_code=code, start_date="20220101", end_date="20260519")
        if dt is not None and len(dt) > 0:
            dt["ts_code"] = code; all_top10.append(dt)
    except: pass
    time.sleep(0.25)
    if (i+1) % 50 == 0:
        print(f"  {i+1}/{len(codes)} holder={sum(len(x) for x in all_holder)} top10={sum(len(x) for x in all_top10)} t={time.time()-t0:.0f}s", flush=True)

# 缓存
if all_holder:
    pd.concat(all_holder, ignore_index=True).to_parquet("data/new_factors/holder_test.parquet")
if all_top10:
    pd.concat(all_top10, ignore_index=True).to_parquet("data/new_factors/top10_test.parquet")
print(f"下载完成 t={time.time()-t0:.0f}s", flush=True)

# IC测试
panel = pd.read_parquet("data/factors/factor_panel_v6.parquet", columns=["ts_code","trade_date","fwd_20d_ret"])

if all_holder:
    h = pd.concat(all_holder, ignore_index=True)
    h["end_date"] = pd.to_datetime(h["end_date"])
    h = h.sort_values(["ts_code","end_date"])
    h["holder_change"] = h.groupby("ts_code")["holder_num"].pct_change()
    h["holder_qoq"] = h.groupby("ts_code")["holder_num"].pct_change(3)
    h["trade_date"] = (h["end_date"] + pd.Timedelta(days=45)).dt.to_timestamp().dt.to_period("M").dt.to_timestamp()
    merged = h.merge(panel, on=["ts_code","trade_date"], how="inner")
    
    print("\n📊 股东户数:", flush=True)
    for fact in ["holder_change","holder_qoq"]:
        ics = []
        for ym, g in merged.dropna(subset=[fact,"fwd_20d_ret"]).groupby(merged["trade_date"].dt.to_period("M")):
            gv = g[[fact,"fwd_20d_ret"]].dropna()
            if len(gv) < 10: continue
            r, _ = spearmanr(gv[fact], gv["fwd_20d_ret"])
            if not np.isnan(r): ics.append(r)
        if ics:
            ic_m = float(np.mean(ics)); ic_s = float(np.std(ics))
            print(f"  {fact}: IC={ic_m*100:+.2f}% IR={ic_m/ic_s if ic_s>0 else 0:.2f} ({len(ics)}个月)", flush=True)

if all_top10:
    t10 = pd.concat(all_top10, ignore_index=True)
    t10["end_date"] = pd.to_datetime(t10["end_date"])
    inst = ["金融机构","保险公司","基金","证券","投资"]
    t10["is_inst"] = t10["holder_type"].apply(lambda x: any(it in str(x) for it in inst))
    isum = t10.groupby(["ts_code","end_date"]).agg(
        inst_hold_ratio=("hold_ratio", lambda x: x[t10.loc[x.index,"is_inst"]].sum()),
        inst_change=("hold_change", lambda x: x[t10.loc[x.index,"is_inst"]].sum()),
    ).reset_index()
    isum["inst_ratio_change"] = isum.groupby("ts_code")["inst_hold_ratio"].diff()
    isum["trade_date"] = (isum["end_date"] + pd.Timedelta(days=45)).dt.to_timestamp().dt.to_period("M").dt.to_timestamp()
    merged2 = isum.merge(panel, on=["ts_code","trade_date"], how="inner")
    
    print("\n📊 十大股东机构:", flush=True)
    for fact in ["inst_ratio_change","inst_change"]:
        ics = []
        for ym, g in merged2.dropna(subset=[fact,"fwd_20d_ret"]).groupby(merged2["trade_date"].dt.to_period("M")):
            gv = g[[fact,"fwd_20d_ret"]].dropna()
            if len(gv) < 10: continue
            r, _ = spearmanr(gv[fact], gv["fwd_20d_ret"])
            if not np.isnan(r): ics.append(r)
        if ics:
            ic_m = float(np.mean(ics)); ic_s = float(np.std(ics))
            print(f"  {fact}: IC={ic_m*100:+.2f}% IR={ic_m/ic_s if ic_s>0 else 0:.2f} ({len(ics)}个月)", flush=True)

print(f"\n⏱ {time.time()-t0:.0f}s", flush=True)
