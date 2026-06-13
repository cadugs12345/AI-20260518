"""
快速重建 factor_panel_with_fwd_v2.parquet
从 v6 面板取数据 + 计算 fwd_20d_ret
"""
import sys, os, time, gc
import numpy as np
import pandas as pd

os.chdir("/mnt/d/AI-20260518"); sys.path.insert(0, '.')
t0 = time.time()
print(f"🔄 重建 with_fwd_v2 — {time.strftime('%F %H:%M')}")
print("="*60)

# 1. 读v6面板（取基础因子列，与原始with_fwd_v2对齐）
print("[1/4] 加载v6面板...")
v2_cols = ['ts_code', 'trade_date', '短期反转', '20日动量', '60日动量', '120日动量', 
           '波动率', 'RSI_6', 'RSI_12', 'RSI_24', 'EMA5偏离', 'EMA10偏离', 'EMA20偏离',
           'BOLL位置', 'MACD', '量能趋势', 'OBV', 
           '市值', '流通市值', 'BP', 'EP', 'SP', '股息率', '换手率', '量比',
           'ROE', '毛利率', '净利率', '杠杆']
panel = pd.read_parquet("data/factors/factor_panel_v6.parquet", columns=v2_cols)
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
panel = panel.sort_values(["trade_date","ts_code"]).reset_index(drop=True)
print(f"  基础面板: {len(panel):,}行, {panel['trade_date'].nunique()}个交易日")
print(f"  最新日期: {panel['trade_date'].max()}")

# 2. 读取fwd收益数据 - 从原始日K线计算
print("[2/4] 计算fwd_20d_ret...")
daily_files = sorted(os.listdir("data/raw/daily"))
rows = []
for fname in daily_files:
    df = pd.read_parquet(os.path.join("data/raw/daily", fname))
    rows.append(df)
daily = pd.concat(rows, ignore_index=True)
daily["trade_date"] = pd.to_datetime(daily["trade_date"])
daily = daily.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
del rows; gc.collect()
print(f"  日K线: {len(daily):,}行, {daily['ts_code'].nunique()}只")

# 计算fwd收益: 今天close / 20天后close - 1
def calc_fwd_ret(grp):
    grp = grp.sort_values("trade_date").reset_index(drop=True)
    close = grp["close"].values
    fwd = np.full(len(grp), np.nan)
    if len(grp) > 20:
        fwd[:-20] = close[20:] / close[:-20] - 1
    grp["fwd_20d_ret"] = fwd
    return grp[["ts_code", "trade_date", "fwd_20d_ret"]]

fwd_list = []
for code, grp in daily.groupby("ts_code", group_keys=False):
    fwd_list.append(calc_fwd_ret(grp))
fwd = pd.concat(fwd_list, ignore_index=True)
del daily, fwd_list; gc.collect()
print(f"  fwd收益: {fwd['fwd_20d_ret'].notna().sum():,}条非空")

# 3. 合并fwd收益到面板
print("[3/4] 合并fwd收益...")
panel = panel.merge(fwd, on=["ts_code", "trade_date"], how="left")
print(f"  合并后: {len(panel):,}行")

# 4. 保存
print("[4/4] 保存...")
panel.to_parquet("data/factors/factor_panel_with_fwd_v2.parquet", index=False)
print(f"  ✅ 已保存: {len(panel):,}行")
print(f"  最新日期: {panel['trade_date'].max()}")
print(f"  fwd_20d_ret非空: {panel['fwd_20d_ret'].notna().sum():,}")
print(f"\n✅ 完成! {time.time()-t0:.0f}秒")
