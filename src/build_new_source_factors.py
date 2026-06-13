#!/usr/bin/env python3
"""
build_new_source_factors.py — 将新数据源整合为选股因子
数据源:
1. 业绩预告 → 超预期因子、修正因子
2. 个股资金流 → 大单净流入因子（新版，分大小单）
3. 龙虎榜 → 机构净买入因子
输出: data/factors/new_source_factors.parquet (可合并到面板)
"""
import os, sys, time, json
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
os.chdir("/mnt/d/AI-20260518")
tt = time.time()

print("="*60, flush=True)
print("新数据源因子构建", flush=True)
print(time.strftime('%F %H:%M'), flush=True)
print("="*60, flush=True)

# ========== 1. 业绩预告因子 ==========
print("\n[1/3] 业绩预告因子...", flush=True)
fc = pd.read_parquet("data/raw/new_sources/forecast.parquet")
print(f"  原始: {len(fc):,}条", flush=True)

# 取最新的预告（按ann_date去重，每只股票取最近一条）
fc = fc.sort_values(["ts_code", "ann_date"], ascending=[True, False])
fc = fc.drop_duplicates(subset=["ts_code", "end_date"], keep="first")

# 关键列：p_change_min/p_change_max = 净利润变动幅度（%）
# type = 预增/预减/扭亏/首亏/续亏/续盈等
# 构造因子:
fc["p_change_avg"] = (fc["p_change_min"] + fc["p_change_max"]) / 2  # 均值

# 业绩变动方向（1预增扭亏=好，-1预减首亏=差）
type_map = {"预增":1, "略增":0.5, "扭亏":1, "续盈":0.3, "预减":-1, "略减":-0.5, "首亏":-1, "续亏":-0.5, "减亏":0.2, "不确定":0}
fc["type_score"] = fc["type"].map(type_map).fillna(0)

# 超预期：p_change_avg > 50% 且 net_profit > 上期 = 超预期
fc["surprise_ratio"] = np.where(
    (fc["last_parent_net"] != 0) & fc["last_parent_net"].notna(),
    (fc["net_profit_min"] - fc["last_parent_net"].abs()) / fc["last_parent_net"].abs(),
    np.nan
)
fc["surprise_score"] = fc["type_score"] * np.clip(fc["p_change_avg"] / 50, 0, 2)

# 按ann_date取最新
fc_latest = fc.sort_values("ann_date", ascending=False).drop_duplicates("ts_code")
fc_factors = pd.DataFrame({
    "ts_code": fc_latest["ts_code"],
    "forecast_type": fc_latest["type_score"],
    "forecast_surprise": fc_latest["surprise_score"],
    "forecast_change": fc_latest["p_change_avg"],
})

# 日期标注
fc_latest["trade_date"] = pd.to_datetime(fc_latest["ann_date"])

print(f"  因子: {len(fc_factors):,}只股票", flush=True)

# ========== 2. 个股资金流因子 ==========
print("\n[2/3] 个股资金流因子...", flush=True)
mf = pd.read_parquet("data/raw/new_sources/individual_moneyflow.parquet")
print(f"  原始: {len(mf):,}条", flush=True)

mf["trade_date"] = pd.to_datetime(mf["trade_date"])

# 计算净指标
mf["net_sm"] = mf["buy_sm_amount"] - mf["sell_sm_amount"]  # 小单净额
mf["net_md"] = mf["buy_md_amount"] - mf["sell_md_amount"]  # 中单
mf["net_lg"] = mf["buy_lg_amount"] - mf["sell_lg_amount"]  # 大单
mf["net_elg"] = mf["buy_elg_amount"] - mf["sell_elg_amount"]  # 特大单
mf["net_total"] = mf["net_mf_amount"]  # 总净额

# 分母：成交额
mf["total_amount"] = (
    mf["buy_sm_amount"] + mf["buy_md_amount"] + mf["buy_lg_amount"] + mf["buy_elg_amount"] +
    mf["sell_sm_amount"] + mf["sell_md_amount"] + mf["sell_lg_amount"] + mf["sell_elg_amount"]
) / 2  # 近似成交额

# 大单+特大单净占比（最有效的大资金信号）
mf["big_net_ratio"] = (mf["net_lg"] + mf["net_elg"]) / (mf["total_amount"] + 1) * 100

# 特大单净占比
mf["elg_net_ratio"] = mf["net_elg"] / (mf["total_amount"] + 1) * 100

# 滚动平均（5日平滑）
mf = mf.sort_values(["ts_code", "trade_date"])
for col in ["big_net_ratio", "elg_net_ratio", "net_total", "big_net_ratio"]:
    mf[f"{col}_ma5"] = mf.groupby("ts_code")[col].transform(lambda x: x.rolling(5, min_periods=3).mean())
    mf[f"{col}_ma10"] = mf.groupby("ts_code")[col].transform(lambda x: x.rolling(10, min_periods=5).mean())

# 资金流的变化方向
mf["big_net_direction"] = np.sign(mf["big_net_ratio"])
# 连续资金流
mf["big_net_accel"] = mf["big_net_ratio"] - mf.groupby("ts_code")["big_net_ratio"].shift(1)

# 保留因子列
mf_cols = ["ts_code", "trade_date",
    "big_net_ratio", "elg_net_ratio", 
    "big_net_ratio_ma5", "elg_net_ratio_ma5",
    "big_net_ratio_ma10", "elg_net_ratio_ma10",
    "big_net_direction", "big_net_accel",
    "net_total", "net_lg", "net_elg"
]
mf_factors = mf[mf_cols].copy()

print(f"  因子列: {len(mf_cols)-2}个, 行数: {len(mf_factors):,}", flush=True)

# ========== 3. 龙虎榜因子 ==========
print("\n[3/3] 龙虎榜因子...", flush=True)
tf = pd.read_parquet("data/raw/new_sources/top_list.parquet")
print(f"  原始: {len(tf):,}条", flush=True)

tf["trade_date"] = pd.to_datetime(tf["trade_date"])

# 净买入率
tf["net_rate"] = tf["net_amount"] / (tf["amount"] + 1) * 100

# 机构参与度
tf["inst_buy_ratio"] = tf["l_buy"] / (tf["amount"] + 1) * 100
tf["inst_sell_ratio"] = tf["l_sell"] / (tf["amount"] + 1) * 100
tf["inst_net_ratio"] = (tf["l_buy"] - tf["l_sell"]) / (tf["amount"] + 1) * 100

# 近20日上榜频次和累计净买入
tf = tf.sort_values(["ts_code", "trade_date"])
tf["toplist_20d"] = tf.groupby("ts_code")["trade_date"].transform(
    lambda x: x.rolling(20, min_periods=1).count()
)
tf["net_20d"] = tf.groupby("ts_code")["net_amount"].transform(
    lambda x: x.rolling(20, min_periods=1).sum()
)
tf["inst_net_20d"] = tf.groupby("ts_code")["inst_net_ratio"].transform(
    lambda x: x.rolling(20, min_periods=1).mean()
)

# 每天最新（同一只股票可能多个原因上榜，取平均）
tf_day = tf.groupby(["ts_code", "trade_date"], as_index=False).agg({
    "net_rate": "mean",
    "inst_buy_ratio": "mean",
    "inst_net_ratio": "mean",
    "toplist_20d": "max",
    "net_20d": "max",
    "inst_net_20d": "mean"
})

tf_factors = tf_day.rename(columns={
    "net_rate": "toplist_net_rate",
    "inst_buy_ratio": "toplist_inst_buy",
    "inst_net_ratio": "toplist_inst_net",
    "toplist_20d": "toplist_freq_20d",
    "net_20d": "toplist_net_20d",
    "inst_net_20d": "toplist_inst_net_20d"
})

print(f"  因子列: 6个, 行数: {len(tf_factors):,}", flush=True)

# ========== 输出 ==========
print(f"\n{'='*60}", flush=True)

# 保存资金流因子（时间序列数据，单独存）
mf_factors.to_parquet("data/factors/moneyflow_factors_v2.parquet", index=False)
print(f"✅ 资金流因子: moneyflow_factors_v2.parquet ({len(mf_factors):,}行)", flush=True)

# 保存龙虎榜因子
tf_factors.to_parquet("data/factors/toplist_factors.parquet", index=False)
print(f"✅ 龙虎榜因子: toplist_factors.parquet ({len(tf_factors):,}行)", flush=True)

# 业绩预告是截面因子（每个股票一条最新），保存
fc_factors.to_parquet("data/factors/forecast_factors.parquet", index=False)
print(f"✅ 业绩预告因子: forecast_factors.parquet ({len(fc_factors):,}行)", flush=True)

print(f"\n⏱ {(time.time()-tt):.0f}秒", flush=True)
print("完成", flush=True)
