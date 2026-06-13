#!/usr/bin/env python3
"""
增量补今天日K线（多线程）— 只补缺失的06-11
"""
import tushare as ts
import pandas as pd
import numpy as np
import os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJ = "/mnt/d/AI-20260604"
DATA_DIR = os.path.join(PROJ, "data", "raw", "daily")
FACTORS_DIR = os.path.join(PROJ, "data", "factors")
SL_FILE = os.path.join(PROJ, "data", "raw", "stock_list.parquet")
TODAY = "20260611"

ts.set_token("3e8953587c4c717c26e5cb99d028a66e044d184f2d464cab0950000e")
t0 = time.time()

print(f"📥 多线程增量下载 — {TODAY}")
print("=" * 50)

sl = pd.read_parquet(SL_FILE)
codes = sl["ts_code"].tolist()
total = len(codes)
print(f"总股票数: {total}")

today_dt = pd.Timestamp(TODAY)

def download_one(code):
    """单只股票下载今天的数据并追加到已有文件"""
    fpath = os.path.join(DATA_DIR, f"{code}.parquet")
    try:
        pro = ts.pro_api()
        df = pro.daily(ts_code=code, start_date=TODAY, end_date=TODAY,
                      fields="trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount")
        if df is None or len(df) == 0:
            return code, "empty"
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        
        if os.path.exists(fpath):
            old = pd.read_parquet(fpath)
            # 如果已有今天的数据则跳过
            if today_dt in old["trade_date"].values:
                return code, "exists"
            combined = pd.concat([old, df]).sort_values("trade_date").reset_index(drop=True)
            combined.to_parquet(fpath, index=False)
        else:
            df = df.sort_values("trade_date").reset_index(drop=True)
            df.to_parquet(fpath, index=False)
        return code, "ok"
    except Exception as e:
        return code, f"fail:{e}"

ok_count = 0
empty_count = 0
exists_count = 0
fail_count = 0
batch_size = 300

for batch_start in range(0, total, batch_size):
    batch = codes[batch_start:batch_start+batch_size]
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(download_one, code): code for code in batch}
        for future in as_completed(futures):
            code, status = future.result()
            if status == "ok":
                ok_count += 1
            elif status == "empty":
                empty_count += 1
            elif status == "exists":
                exists_count += 1
            else:
                fail_count += 1
                if fail_count <= 3:
                    print(f"  ❌ {code}: {status}")
    
    progress = min(batch_start+batch_size, total)
    elapsed = time.time() - t0
    print(f"  进度: {progress}/{total} | OK={ok_count} | EMPTY={empty_count} | EXISTS={exists_count} | FAIL={fail_count} | {elapsed:.0f}s", flush=True)
    time.sleep(0.5)

print(f"\n✅ 完成! OK={ok_count} | EMPTY={empty_count}(停牌/无数据) | EXISTS={exists_count}(已有) | FAIL={fail_count}")
print(f"⏱ 耗时: {time.time()-t0:.0f}秒")

# 更新 full_prices
print("\n📦 更新 full_prices...")
today_rows = []
for code in codes:
    fpath = os.path.join(DATA_DIR, f"{code}.parquet")
    try:
        df = pd.read_parquet(fpath)
        row = df[df["trade_date"] == today_dt]
        if len(row) > 0:
            today_rows.append(row.iloc[-1:])
    except:
        pass

if today_rows:
    new_px = pd.concat(today_rows, ignore_index=True)
    fp_path = os.path.join(FACTORS_DIR, "full_prices.parquet")
    fp_old = pd.read_parquet(fp_path)
    fp_new = pd.concat([fp_old, new_px]).drop_duplicates(subset=["ts_code","trade_date"]).sort_values(["ts_code","trade_date"]).reset_index(drop=True)
    fp_new.to_parquet(fp_path, index=False)
    print(f"  ✅ full_prices: {len(fp_new):,}行 (新增{len(new_px)}行, 最新日期{TODAY})")
else:
    print(f"  ⚠️ 无{TODAY}数据")
