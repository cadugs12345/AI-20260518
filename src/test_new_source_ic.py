#!/usr/bin/env python3
"""
test_new_source_ic.py — 新数据源因子单因子IC测试
测试：资金流v2因子、龙虎榜因子、业绩预告因子
"""
import os, sys
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
os.chdir("/mnt/d/AI-20260518")
from scipy.stats import spearmanr

print("="*60, flush=True)
print("新数据源因子IC测试", flush=True)
print("="*60, flush=True)

panel = pd.read_parquet("data/factors/factor_panel_v6.parquet", columns=["trade_date","ts_code","fwd_20d_ret"])
panel["trade_date"] = pd.to_datetime(panel["trade_date"])

# 1. 业绩预告因子（截面）
fc = pd.read_parquet("data/factors/forecast_factors.parquet")
print(f"\n1. 业绩预告因子: {len(fc)}行, 列={list(fc.columns)}")
fc_cols = [c for c in fc.columns if c != "ts_code"]

# 2. 资金流因子（时间序列）
mf = pd.read_parquet("data/factors/moneyflow_factors_v2.parquet")
mf["trade_date"] = pd.to_datetime(mf["trade_date"])
print(f"\n2. 资金流因子: {len(mf):,}行, {len(mf['trade_date'].unique())}天, 列={[c for c in mf.columns if c not in ['ts_code','trade_date']]}")

# 3. 龙虎榜因子
tf = pd.read_parquet("data/factors/toplist_factors.parquet")
tf["trade_date"] = pd.to_datetime(tf["trade_date"])
print(f"\n3. 龙虎榜因子: {len(tf):,}行, 列={[c for c in tf.columns if c not in ['ts_code','trade_date']]}")

results = []

# 测试截面因子（业绩预告，只有最近一期）
print(f"\n{'='*60}", flush=True)
print("业绩预告因子（截面，用最近日期测试）", flush=True)
latest_date = panel["trade_date"].max()
latest_panel = panel[panel["trade_date"]==latest_date].copy()
merged = latest_panel.merge(fc, on="ts_code", how="inner")
for col in fc_cols:
    valid = merged[col].notna() & merged["fwd_20d_ret"].notna() & (merged["fwd_20d_ret"].abs()<0.5)
    if valid.sum() > 20:
        ic, _ = spearmanr(merged.loc[valid, col], merged.loc[valid, "fwd_20d_ret"])
        if not np.isnan(ic):
            ics = [ic]
            results.append({"factor": col, "type": "截面", "ic_avg": ic, "ir": None, "n": valid.sum()})
            print(f"  {col:25s}: IC={ic*100:+6.2f}% (n={valid.sum()}) {'🟢' if abs(ic)>0.02 else ''}", flush=True)

# 测试时间序列因子（资金流）
print(f"\n{'='*60}", flush=True)
print("资金流因子（时间序列）", flush=True)
mf_cols = [c for c in mf.columns if c not in ["ts_code","trade_date"]]

# 和panel按日期+股票合并
merged = panel.merge(mf[["ts_code","trade_date"]+mf_cols], on=["ts_code","trade_date"], how="inner")
print(f"  合并后: {len(merged):,}行, {merged['trade_date'].nunique()}期", flush=True)

for col in mf_cols:
    ics = []
    for d in merged["trade_date"].unique():
        day = merged[merged["trade_date"]==d]
        valid = day[col].notna() & day["fwd_20d_ret"].notna() & (day["fwd_20d_ret"].abs()<0.5)
        if valid.sum() > 20:
            ic, _ = spearmanr(day.loc[valid, col], day.loc[valid, "fwd_20d_ret"])
            if not np.isnan(ic):
                ics.append(ic)
    if ics:
        ir = np.mean(ics) / np.std(ics) if np.std(ics) > 0 else 0
        results.append({"factor": col, "type": "资金流", "ic_avg": np.mean(ics), "ir": ir, "n": len(ics)})
        print(f"  {col:25s}: IC={np.mean(ics)*100:+6.2f}% IR={ir:.2f} ({len(ics)}期) {'🟢' if abs(np.mean(ics))>0.01 else ''}", flush=True)

# 测试龙虎榜因子
print(f"\n{'='*60}", flush=True)
print("龙虎榜因子（时间序列）", flush=True)
tf_cols = [c for c in tf.columns if c not in ["ts_code","trade_date"]]

merged_tf = panel.merge(tf[["ts_code","trade_date"]+tf_cols], on=["ts_code","trade_date"], how="inner")
print(f"  合并后: {len(merged_tf):,}行, {merged_tf['trade_date'].nunique()}期", flush=True)

for col in tf_cols:
    ics = []
    for d in merged_tf["trade_date"].unique():
        day = merged_tf[merged_tf["trade_date"]==d]
        valid = day[col].notna() & day["fwd_20d_ret"].notna() & (day["fwd_20d_ret"].abs()<0.5)
        if valid.sum() > 20:
            ic, _ = spearmanr(day.loc[valid, col], day.loc[valid, "fwd_20d_ret"])
            if not np.isnan(ic):
                ics.append(ic)
    if ics:
        ir = np.mean(ics) / np.std(ics) if np.std(ics) > 0 else 0
        results.append({"factor": col, "type": "龙虎榜", "ic_avg": np.mean(ics), "ir": ir, "n": len(ics)})
        print(f"  {col:25s}: IC={np.mean(ics)*100:+6.2f}% IR={ir:.2f} ({len(ics)}期) {'🟢' if abs(np.mean(ics))>0.01 else ''}", flush=True)

# 汇总
print(f"\n{'='*60}", flush=True)
print("IC结果汇总", flush=True)
print(f"{'因子':30s} {'类型':8s} {'IC':8s} {'IR':6s} {'期数':6s}", flush=True)
print("-"*60, flush=True)
for r in sorted(results, key=lambda x: abs(x["ic_avg"]), reverse=True):
    ics = f"{r['ic_avg']*100:+.2f}%"
    irs = f"{r['ir']:.2f}" if r['ir'] is not None else "N/A"
    print(f"{r['factor']:30s} {r['type']:8s} {ics:8s} {irs:6s} {r['n']:>6}", flush=True)

print(f"\n{'='*60}", flush=True)
print("完成", flush=True)
