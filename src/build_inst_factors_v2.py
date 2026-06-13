"""
全量下载股东户数+十大股东，构建机构行为因子，合并到面板
"""
import sys, os, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

PROJECT = "/mnt/d/AI-20260518"
os.chdir(PROJECT)

sys.path.insert(0, '.')
from config.settings import TS_TOKEN
import tushare as ts
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

t0 = time.time()
OUT = "data/new_factors"
os.makedirs(OUT, exist_ok=True)

# 1. 加载股票列表
codes = pd.read_parquet("data/raw/stock_list.parquet")["ts_code"].tolist()
print(f"全市场: {len(codes)}只", flush=True)

# 2. 下载全量股东户数（tushare限速~0.25s/股）
print("下载股东户数...", flush=True)
all_holder = []
for i, code in enumerate(codes):
    try:
        dh = pro.query("stk_holdernumber", ts_code=code, start_date="20180101", end_date="20260519")
        if dh is not None and len(dh) > 0:
            dh["ts_code"] = code
            all_holder.append(dh)
    except:
        pass
    time.sleep(0.15)
    if (i + 1) % 500 == 0:
        nh = sum(len(x) for x in all_holder)
        print(f"  holder: {i+1}/{len(codes)}, {nh}条, t={time.time()-t0:.0f}s", flush=True)
        pd.concat(all_holder, ignore_index=True).to_parquet(f"{OUT}/holder_checkpoint.parquet")

if all_holder:
    holder = pd.concat(all_holder, ignore_index=True)
    holder.to_parquet(f"{OUT}/holder_all.parquet")
    print(f"  股东户数完成: {len(holder)}条, t={time.time()-t0:.0f}s", flush=True)

# 3. 下载全量十大股东
print("下载十大股东...", flush=True)
all_top10 = []
for i, code in enumerate(codes):
    try:
        dt = pro.query("top10_holders", ts_code=code, start_date="20200101", end_date="20260519")
        if dt is not None and len(dt) > 0:
            dt["ts_code"] = code
            all_top10.append(dt)
    except:
        pass
    time.sleep(0.2)
    if (i + 1) % 500 == 0:
        nt = sum(len(x) for x in all_top10)
        print(f"  top10: {i+1}/{len(codes)}, {nt}条, t={time.time()-t0:.0f}s", flush=True)
        pd.concat(all_top10, ignore_index=True).to_parquet(f"{OUT}/top10_checkpoint.parquet")

if all_top10:
    t10 = pd.concat(all_top10, ignore_index=True)
    t10.to_parquet(f"{OUT}/top10_all.parquet")
    print(f"  十大股东完成: {len(t10)}条, t={time.time()-t0:.0f}s", flush=True)

print(f"\n下载完成 t={time.time()-t0:.0f}s", flush=True)
