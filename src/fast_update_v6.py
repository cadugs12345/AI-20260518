#!/usr/bin/env python3
"""
快速补齐 v6 面板：读原始日K线，只计算 5/18~5/22 的因子
用 build_factors.py 的核心逻辑
"""
import sys, os, json, glob, time
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")
os.chdir("/mnt/d/AI-20260518"); sys.path.insert(0, '.')

from config.settings import DATA_RAW, DATA_FACTORS
t0 = time.time()

print(f"\n🔄 快速补齐面板v6 (5/18~5/22)")
print("="*50)

# 1. 加载v6
print("[1/4] 加载v6面板...")
old = pd.read_parquet(f"{DATA_FACTORS}/factor_panel_v6.parquet")
old["trade_date"] = pd.to_datetime(old["trade_date"])
print(f"  原有: {len(old):,}行, 最新: {old['trade_date'].max().date()}")

# 2. 加载原始日K线（只取 2025-06-01 之后够算因子的）
print("[2/4] 加载原始日K线...")
files = sorted(glob.glob(f"{DATA_RAW}/daily/*.parquet"))
chunks = []
for f in files:
    try:
        df = pd.read_parquet(f)
        chunks.append(df)
    except: pass
daily = pd.concat(chunks, ignore_index=True)
daily["trade_date"] = pd.to_datetime(daily["trade_date"])
daily = daily.sort_values(["ts_code","trade_date"]).reset_index(drop=True)
print(f"  日K线: {len(daily):,}行, {daily['ts_code'].nunique()}只")

need_dates = sorted(d for d in daily["trade_date"].unique() if d > pd.Timestamp("2026-05-15"))
print(f"\n[3/4] 需补交易日: {len(need_dates)}个 — {need_dates[0].date()} ~ {need_dates[-1].date()}")

def calc(code_grp):
    """计算一只股票的因子（与 build_factors.py 一致）"""
    if len(code_grp) < 60:
        return []
    c = code_grp.set_index("trade_date")
    close, high, low, vol, pct_chg, amount = c["close"], c["high"], c["low"], c["vol"], c["pct_chg"], c["amount"]
    ret = []
    
    for dt in need_dates:
        if dt not in c.index:
            continue
        pos = c.index.get_loc(dt)
        n = pos + 1  # 当前行数（含今天）
        
        row = {"ts_code": code_grp["ts_code"].iloc[0], "trade_date": dt}
        
        # === 基础因子 ===
        row["短期反转"] = -close.iloc[pos] / close.iloc[max(0,pos-5)] + 1 if n >= 6 else np.nan
        row["20日动量"] = close.iloc[pos] / close.iloc[max(0,pos-21)] - 1 if n >= 21 else np.nan
        row["60日动量"] = close.iloc[pos] / close.iloc[max(0,pos-61)] - 1 if n >= 61 else np.nan
        row["120日动量"] = close.iloc[pos] / close.iloc[max(0,pos-121)] - 1 if n >= 121 else np.nan
        row["波动率"] = np.std(close.iloc[max(0,pos-60):pos].pct_change().dropna()) if n >= 60 else np.nan
        
        # RSI
        for period in [6, 12, 24]:
            if n >= period + 2:
                dts = close.iloc[pos-period:pos+1].diff().dropna()
                gains = dts[dts > 0].sum()
                losses = -dts[dts < 0].sum()
                rs = gains / losses if losses != 0 else 999
                row[f"RSI_{period}"] = 100 - 100 / (1 + rs)
            else:
                row[f"RSI_{period}"] = np.nan
        
        # 均线偏离
        for ma_p in [5, 10, 20, 60]:
            if n >= ma_p:
                ma = close.iloc[pos-ma_p+1:pos+1].mean()
                row[f"EMA{ma_p}偏离"] = close.iloc[pos] / ma - 1
            else:
                row[f"EMA{ma_p}偏离"] = np.nan
        
        # 换手率（估算）
        if n >= 21:
            avg_vol = vol.iloc[max(0,pos-20):pos].mean()
            row["换手率"] = vol.iloc[pos] / avg_vol if avg_vol > 0 else np.nan
        row["成交额"] = amount.iloc[pos]
        
        # 高波反转
        if n >= 21:
            p20 = pct_chg.iloc[max(0,pos-20):pos]
            vol_20d_std = np.std(p20) if len(p20) >= 10 else np.nan
            ret_20 = close.iloc[pos] / close.iloc[max(0,pos-20)] - 1
            row["高波反转"] = vol_20d_std * (-ret_20) if pd.notna(vol_20d_std) else np.nan
        
        # 量价背离
        if n >= 21:
            ret_20 = close.iloc[pos] / close.iloc[max(0,pos-20)] - 1
            avg_vol_20 = vol.iloc[max(0,pos-20):pos].mean()
            vol_change = vol.iloc[pos] / avg_vol_20 - 1 if avg_vol_20 > 0 else 0
            row["量价背离"] = abs(ret_20) - abs(vol_change)
        
        # big_order_ratio（大单净量估算）
        if n >= 21:
            avg_amt = amount.iloc[max(0,pos-20):pos].mean()
            row["big_order_ratio"] = (amount.iloc[pos] / avg_amt - 1) * abs(pct_chg.iloc[pos]) if avg_amt > 0 else np.nan
        
        ret.append(row)
    return ret

# 逐只计算（只算 need_dates 的因子，跳过已有的）
all_new = []
codes = daily["ts_code"].unique()
for i, code in enumerate(codes):
    grp = daily[daily["ts_code"] == code]
    rows = calc(grp)
    if rows:
        all_new.extend(rows)
    if (i+1) % 500 == 0:
        print(f"  {i+1}/{len(codes)}只, 已生成{len(all_new)}行", flush=True)

if not all_new:
    print("  ⚠️ 无新数据")
    sys.exit(0)

new_df = pd.DataFrame(all_new)
print(f"  生成: {len(new_df):,}行")

# 4. 合并到v6
print("[4/4] 合并到v6...")
# 补全 v6 有但 new 没有的列
for c in old.columns:
    if c not in new_df.columns:
        new_df[c] = np.nan
# 补全 new 有但 v6 没有的列（跳过）
cols = [c for c in old.columns if c in new_df.columns]
new_df = new_df[cols]

combined = pd.concat([old, new_df], ignore_index=True)
combined = combined.drop_duplicates(subset=["ts_code","trade_date"], keep="last")
combined = combined.sort_values(["trade_date","ts_code"]).reset_index(drop=True)
combined.to_parquet(f"{DATA_FACTORS}/factor_panel_v6.parquet", index=False)
print(f"  ✅ v6面板更新: {len(combined):,}行, {combined['trade_date'].nunique()}日")
print(f"  最新: {combined['trade_date'].max().date()}")
print(f"  ⏱ {time.time()-t0:.0f}s")
