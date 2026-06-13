#!/usr/bin/env python3
"""批量下载分时数据 (快速版)"""
import os, sys, time, json
import requests as req
import pandas as pd

API_URL = "https://data.diemeng.chat/api/stock/history"
API_KEY = "4b4d5c2093ec2260967007116f09a5732e5cbab7f8a17d00da"
DATA_DIR = "/mnt/d/AI-20260604/data/minute"
DAILY_DIR = "/mnt/d/AI-20260604/data/raw/daily"
STOCK_LIST_FILE = "/mnt/d/AI-20260604/data/raw/stock_list.parquet"
os.makedirs(DATA_DIR, exist_ok=True)

MAX_STOCKS = 200
MIN_DAYS = 60

# 筛选：近60日有涨停 + 平均成交额排前MAX_STOCKS
stock_df = pd.read_parquet(STOCK_LIST_FILE)
stocks = stock_df['ts_code'].tolist()

active = []
for code in stocks:
    fpath = os.path.join(DAILY_DIR, f"{code}.parquet")
    if not os.path.exists(fpath):
        continue
    try:
        df = pd.read_parquet(fpath)
    except:
        continue
    if len(df) < MIN_DAYS:
        continue
    recent = df.tail(60)
    limit_cnt = ((recent['close'] / recent['close'].shift(1) > 1.095) & 
                 (abs(recent['high'] - recent['close']) < 1e-6)).sum()
    if limit_cnt >= 1:
        active.append((code, recent['amount'].mean()))

active.sort(key=lambda x: -x[1])  # 按成交额排序
top = [x[0] for x in active[:MAX_STOCKS]]
print(f"筛选: {len(active)}只活跃股, 取前{MAX_STOCKS}只 (按成交额)", flush=True)

headers = {"apiKey": API_KEY, "Content-Type": "application/json"}
already = set(f.replace(".parquet","") for f in os.listdir(DATA_DIR) if f.endswith(".parquet"))

for i, code in enumerate(top):
    if code in already:
        print(f"[{i+1}/{MAX_STOCKS}] {code} 已有缓存", flush=True)
        continue
    payload = {
        "stock_code": code,
        "level": "1min",
        "start_time": "2026-05-01",
        "end_time": "2026-06-12",
        "page": 0,
        "page_size": 10000,
    }
    try:
        resp = req.post(API_URL, headers=headers, json=payload, timeout=30)
        data = resp.json()
    except Exception as e:
        print(f"[{i+1}/{MAX_STOCKS}] {code} ❌ {e}", flush=True)
        time.sleep(0.5)
        continue
    if data.get("code") != 200:
        print(f"[{i+1}/{MAX_STOCKS}] {code} ❌ {data.get('msg','unknown')}", flush=True)
        time.sleep(0.5)
        continue
    records = data.get("data",{}).get("list",[])
    if not records:
        print(f"[{i+1}/{MAX_STOCKS}] {code} 空", flush=True)
        time.sleep(0.5)
        continue
    df = pd.DataFrame(records)
    df["trade_time"] = pd.to_datetime(df["trade_time"])
    df = df.sort_values("trade_time").reset_index(drop=True)
    df.to_parquet(os.path.join(DATA_DIR, f"{code}.parquet"), index=False)
    print(f"[{i+1}/{MAX_STOCKS}] {code} ✅ {len(df)}条", flush=True)
    time.sleep(0.3)

done = len([f for f in os.listdir(DATA_DIR) if f.endswith(".parquet")])
print(f"\n✅ 完成! 缓存: {done}只", flush=True)
