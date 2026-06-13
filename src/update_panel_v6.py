#!/usr/bin/env python3
"""
增量更新因子面板 v6 — 只补缺失的交易周 5/18~5/22
"""
import sys, os, json, glob
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")
from datetime import datetime
os.chdir("/mnt/d/AI-20260518"); sys.path.insert(0, '.')

print(f"\n🔄 增量更新面板v6 — {datetime.now().strftime('%F %H:%M')}")
print("="*60)

# 1. 读原有v6面板
print("[1/5] 加载v6面板...")
old_panel = pd.read_parquet("data/factors/factor_panel_v6.parquet")
old_panel["trade_date"] = pd.to_datetime(old_panel["trade_date"])
print(f"  原有: {len(old_panel):,}行, {old_panel['trade_date'].nunique()}个交易日")
print(f"  最新日期: {old_panel['trade_date'].max()}")
old_dates = set(old_panel["trade_date"].unique())

# 2. 读所有原始日K线
print("[2/5] 加载原始日K线...")
files = sorted(glob.glob("data/raw/daily/*.parquet"))
all_rows = []
for f in files:
    try:
        df = pd.read_parquet(f)
        all_rows.append(df)
    except:
        pass
daily = pd.concat(all_rows, ignore_index=True)
daily["trade_date"] = pd.to_datetime(daily["trade_date"])
daily = daily.sort_values(["ts_code","trade_date"]).reset_index(drop=True)
print(f"  日K线: {len(daily):,}行, {daily['ts_code'].nunique()}只, {daily['trade_date'].nunique()}日")

# 3. 找出缺失的交易日
all_dates = set(daily["trade_date"].unique())
# 只关注2026年5月及以后的缺失日期
missing_dates = sorted(all_dates - old_dates)
print(f"[3/5] 缺失交易日: {len(missing_dates)}个")
if len(missing_dates) > 20:
    # 只补最近20天，避免补老数据
    missing_dates = missing_dates[-20:]
    print(f"  截取最近20天: {missing_dates[0].date()} ~ {missing_dates[-1].date()}")
if len(missing_dates) == 0:
    print("  ✅ 面板已是最新，无需更新")
    sys.exit(0)
for d in missing_dates:
    print(f"  ➖ {d.date()}")

# 4. 为每个缺失交易日计算因子
print("[4/5] 计算缺失因子...")
new_rows = []
for target_date in missing_dates:
    end_dt = target_date
    start_dt = target_date - pd.Timedelta(days=730)  # 2年回溯
    subset = daily[(daily["trade_date"] >= start_dt) & (daily["trade_date"] <= end_dt)].copy()
    
    for code, grp in subset.groupby("ts_code"):
        grp = grp.iloc[-252:]  # 最多252日K线（1年）
        if len(grp) < 60:
            continue
        
        close = grp["close"].values
        high = grp["high"].values
        low = grp["low"].values
        vol = grp["vol"].values
        pct = grp["pct_chg"].values
        amount = grp["amount"].values
        n = len(grp)
        
        row = {"ts_code": code, "trade_date": end_dt}
        
        # 基础因子
        row["短期反转"] = - (close[-1] / close[-6] - 1) if n >= 6 else np.nan
        row["20日动量"] = close[-1] / close[-21] - 1 if n >= 21 else np.nan
        row["60日动量"] = close[-1] / close[-61] - 1 if n >= 61 else np.nan
        row["120日动量"] = close[-1] / close[-121] - 1 if n >= 121 else np.nan
        row["波动率"] = np.std(pct[-60:]) if n >= 60 else np.nan
        
        # RSI
        for period in [6, 12, 24]:
            if n >= period + 1:
                deltas = np.diff(close[-period-1:])
                gains = np.sum(np.maximum(deltas, 0))
                losses = -np.sum(np.minimum(deltas, 0))
                rs = gains / losses if losses != 0 else 999
                row[f"RSI_{period}"] = 100 - 100 / (1 + rs)
            else:
                row[f"RSI_{period}"] = np.nan
        
        # 均线偏离
        for ma_p in [5, 10, 20, 60]:
            if n >= ma_p:
                ma = np.mean(close[-ma_p:])
                row[f"EMA{ma_p}偏离"] = close[-1] / ma - 1
            else:
                row[f"EMA{ma_p}偏离"] = np.nan
        
        # 换手率估算
        row["换手率"] = (vol[-1] * 100 / (sum(vol[-20:])/20*1000)) if n >= 20 else np.nan
        row["成交额"] = amount[-1] if len(amount) > 0 else 0
        
        # 量价关系
        if n >= 21:
            ret_20d = close[-1] / close[-21] - 1
            vol_20d_change = vol[-1] / np.mean(vol[-21:-1]) - 1 if np.mean(vol[-21:-1]) > 0 else 0
            row["量价背离"] = ret_20d - vol_20d_change
        
        # 高波反转
        if n >= 21:
            vol_20d_std = np.std(pct[-20:])
            ret_next_5d = close[-1] / close[-6] - 1 if n >= 6 else 0
            row["高波反转"] = vol_20d_std * (-ret_20d)
        
        # 大单净量（用价格加权金额估算）
        if n >= 60:
            amt_chg = (amount[-1] / np.mean(amount[-21:-1]) - 1) if np.mean(amount[-21:-1]) > 0 else 0
            ret_1d = pct[-1]
            row["big_order_ratio"] = amt_chg * abs(ret_1d)
        
        new_rows.append(row)

if not new_rows:
    print("  ⚠️ 无新数据行")
    sys.exit(0)

new_df = pd.DataFrame(new_rows)
print(f"  新增: {len(new_df):,}行")

# 5. 合并回v6
print("[5/5] 合并到v6面板...")
v6_old_cols = set(old_panel.columns)
new_cols = set(new_df.columns)

# 找出原v6有但新数据没有的列（可能是财务数据，补NaN）
for c in v6_old_cols - new_cols:
    new_df[c] = np.nan

# 保持列顺序一致
ordered_cols = [c for c in old_panel.columns if c in new_df.columns] + \
               [c for c in new_df.columns if c not in old_panel.columns]
new_df = new_df[ordered_cols]

# 合并并去重
combined = pd.concat([old_panel, new_df], ignore_index=True)
combined = combined.drop_duplicates(subset=["ts_code","trade_date"], keep="last")
combined = combined.sort_values(["trade_date","ts_code"]).reset_index(drop=True)

combined.to_parquet("data/factors/factor_panel_v6.parquet", index=False)
print(f"  ✅ v6面板已更新: {len(combined):,}行, {combined['trade_date'].nunique()}个交易日")
print(f"  最新日期: {combined['trade_date'].max()}")

# 也更新v5（只保留基础列方便后面使用）
v5_cols = [c for c in combined.columns if c not in ["成交额","量价背离","高波反转","big_order_ratio",
                                                       "ROE","毛利率","净利率","利润增速","营收增速","杠杆",
                                                       "pe","pe_ttm","pb","ps","ps_ttm","dv_ratio","dv_ttm","total_mv","circ_mv",
                                                       "fwd_20d_ret","label_rank"]]
v5_df = combined[v5_cols].copy()
v5_df.to_parquet("data/factors/factor_panel_v5.parquet", index=False)
print(f"  ✅ v5面板同步更新: {len(v5_df):,}行")

print(f"\n✅ 完成! {(datetime.now() - datetime(2026,5,23,19,48)).seconds//60}分钟")
