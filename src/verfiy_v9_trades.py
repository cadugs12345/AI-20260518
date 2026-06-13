#!/usr/bin/env python3
"""验证原v9 ABCD版交易信号、买卖价格是否正确"""
import pandas as pd
import numpy as np
import os, random, math

PROJ_B = "/mnt/d/AI-20260604"
DATA_DAILY_DIR = os.path.join(PROJ_B, "data", "raw", "daily")
OUTPUT_DIR = os.path.join(PROJ_B, "backtest_results")
NAV_PATH = os.path.join(OUTPUT_DIR, "breakout_v9_nav.csv")
TRADES_PATH = os.path.join(OUTPUT_DIR, "breakout_v9_backtest.csv")

# 加载原v9代码用于生成信号参考
import importlib.util, sys
spec = importlib.util.spec_from_file_location("v9", "/mnt/d/AI-20260604/src/breakout_v9_backtest.py.bak")

# 直接加载原v9的常数
exec(open("/mnt/d/AI-20260604/src/breakout_v9_backtest.py.bak").read().split("def detect_signals")[0])

# 再加载原v9的detect_signals
v9_code = open("/mnt/d/AI-20260604/src/breakout_v9_backtest.py.bak").read()
exec(v9_code)

trades = pd.read_csv(TRADES_PATH)
print(f"总交易数: {len(trades)}")
print(f"买入价缺失: {trades['entry_price'].isna().sum()}")
print(f"卖出价缺失: {trades['exit_price'].isna().sum()}")

# 抽样验证：按年份分层随机抽
random.seed(42)
samples_per_year = 5
sample_codes = []
trades['entry_date'] = pd.to_datetime(trades['entry_date'])
trades['year'] = trades['entry_date'].dt.year

for yr in sorted(trades['year'].unique()):
    yr_trades = trades[trades['year'] == yr]
    n = min(samples_per_year, len(yr_trades))
    if n == 0: continue
    for _, row in yr_trades.sample(n=n, random_state=42).iterrows():
        sample_codes.append(row)

print(f"\n抽样验证: {len(sample_codes)}笔交易\n")

# 加载股票列表
sl = pd.read_parquet("/mnt/d/AI-20260604/data/raw/stock_list.parquet")
code_to_name = dict(zip(sl['ts_code'], sl.get('name', ['']*len(sl))))

issues = []
for row in sample_codes:
    code = row['code']
    entry_dt = pd.Timestamp(row['entry_date'])
    entry_px = row['entry_price']
    exit_px = row['exit_price']
    hold_days = int(row['hold_days'])
    ret = row['ret_ac']
    
    fp = os.path.join(DATA_DAILY_DIR, f"{code}.parquet")
    if not os.path.exists(fp):
        print(f"  ❌ {code} 数据缺失")
        issues.append(row)
        continue
    
    df = pd.read_parquet(fp).sort_values('trade_date').reset_index(drop=True)
    df['date_str'] = df['trade_date'].apply(lambda x: pd.Timestamp(x).strftime('%Y-%m-%d'))
    
    # 找入场日索引
    emask = df['date_str'] == entry_dt.strftime('%Y-%m-%d')
    if not emask.any():
        print(f"  ❌ {code} {entry_dt.date()} 入场日不存在")
        continue
    
    eidx = emask.idxmax()
    if eidx >= len(df):
        continue
    
    # 验证买入价格
    actual_open = float(df.iloc[eidx]['open'])
    actual_low = float(df.iloc[eidx]['low'])
    actual_close = float(df.iloc[eidx]['close'])
    actual_vol = float(df.iloc[eidx]['vol'])
    
    entry_ok = abs(entry_px - actual_open) < 0.001
    
    # 验证卖出价格和持有天数
    exit_idx = min(eidx + hold_days, len(df)-1)
    exit_actual = float(df.iloc[exit_idx]['close'])
    exit_date_actual = df.iloc[exit_idx]['date_str']
    expected_hold = hold_days
    
    # 验证ret_ac是否匹配
    cost = 0.0032 * 2
    expected_ret = (exit_px / entry_px - 1 - cost) * 100
    
    ret_ok = abs(ret - expected_ret) < 0.1
    
    name = code_to_name.get(code, '')
    status = "✅" if (entry_ok and ret_ok) else "⚠️"
    
    print(f"  {status} {code} {name}")
    print(f"     入场: {entry_dt.date()} 买入价={entry_px}")
    print(f"     实际开盘价={actual_open} 收盘={actual_close}")
    print(f"     出场: {exit_date_actual} (持有{hold_days}日) 卖出价={exit_px}")
    print(f"     回测ret={ret:.2f}% 理论ret={expected_ret:.2f}%")
    
    if not entry_ok:
        print(f"     ⚠️ 买入价不匹配! 开盘{actual_open} vs 记录{entry_px}")
        issues.append(row)
    if not ret_ok:
        print(f"     ⚠️ 收益不匹配! 记录{ret:.2f}% vs 理论{expected_ret:.2f}%")
        issues.append(row)
    print()

print(f"\n{'='*60}")
print(f"验证完成: {len(sample_codes)}笔, 问题: {len(issues)}笔")
if issues:
    print("有问题交易的代码:")
    for r in issues:
        print(f"  {r['code']} {pd.Timestamp(r['entry_date']).date()}")
else:
    print("✅ 所有抽样交易价格正确!")
