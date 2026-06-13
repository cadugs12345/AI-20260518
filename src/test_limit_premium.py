"""
昨日涨停溢价残差因子 (Yesterday's Limit-up Premium Residual)
核心逻辑: 昨日涨停股今日的涨跌幅 - 今日市场整体涨跌幅（剔除系统性风险）

因子:
  1. limit_premium_raw: 昨日涨停股今日的平均涨幅
  2. limit_premium_excess: 昨日涨停股今日涨幅 - 全市场平均涨幅（超额溢价）
  3. limit_premium_residual: 回归残差（涨停溢价 = alpha + beta*市场 + residual）
  4. limit_premium_zscore: 溢价残差的截面z-score
  5. limit_premium_persistence: 昨日涨停且今天继续板的比率
"""
import sys, os, json, glob
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, linregress
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config.settings import DATA_FACTORS

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT)

print(f"\n🏆 昨日涨停溢价残差因子 — {datetime.now().strftime('%F %H:%M')}")
print("=" * 60)

RAW_DIR = "data/raw/daily"
OUTPUT_DIR = "data/new_factors"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 加载日线
print("[1/4] 加载日线...")
files = sorted(glob.glob(f"{RAW_DIR}/*.parquet"))
codes = [os.path.basename(f).replace('.parquet','') for f in files]
print(f"  股票数: {len(codes)}")

chunks = []
for i, (f, code) in enumerate(zip(files, codes)):
    try:
        df = pd.read_parquet(f)
        df["ts_code"] = code
        chunks.append(df[["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "pct_chg", "vol", "amount"]])
    except:
        pass
    if (i+1) % 2000 == 0:
        print(f"  进度: {i+1}/{len(codes)}")

daily = pd.concat(chunks, ignore_index=True)
daily["trade_date"] = pd.to_datetime(daily["trade_date"])
daily = daily.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
print(f"  数据: {len(daily):,}行, {daily['ts_code'].nunique()}只")

# 2. 识别涨停
print("\n[2/4] 识别涨停股...")
def get_limit_pct(code):
    if code.startswith('30') or code.startswith('688'):
        return 0.20
    elif code.startswith('8') or code.startswith('4'):
        return 0.30
    else:
        return 0.10

limit_pcts = daily["ts_code"].map(get_limit_pct)
daily["limit_up"] = daily["close"] >= daily["pre_close"] * (1 + limit_pcts) * 0.995
daily["limit_down"] = daily["close"] <= daily["pre_close"] * (1 - limit_pcts) * 0.995

# 昨日是否涨停
daily["yesterday_limit"] = daily.groupby("ts_code")["limit_up"].shift(1)
daily["yesterday_close"] = daily.groupby("ts_code")["close"].shift(1)

print(f"  涨停样本: {daily['limit_up'].sum():,}次")
print(f"  涨停率: {daily['limit_up'].mean()*100:.2f}%")

# 3. 计算溢价残差
print("\n[3/4] 计算溢价残差...")

# 每日市场平均涨幅（等权）
market_ret = daily.groupby("trade_date")["pct_chg"].mean().to_dict()
daily["market_ret"] = daily["trade_date"].map(market_ret)

# 逐日计算
daily["_ym"] = daily["trade_date"].dt.to_period("M")

print("  逐日计算板块涨停溢价...")
all_results = []
for ym, g in daily.groupby("_ym"):
    g = g.copy()
    for date, day_df in g.groupby("trade_date"):
        if len(day_df) < 50:
            continue
        
        # 昨日涨停股的今日表现
        yest_limit = day_df["yesterday_limit"].values.astype(bool)
        if yest_limit.sum() < 3:
            # 昨日涨停样本太少，填充0
            day_df["limit_premium_raw"] = 0.0
            day_df["limit_premium_excess"] = 0.0
            day_df["limit_premium_residual"] = 0.0
            day_df["limit_premium_zscore"] = 0.0
            day_df["limit_premium_persistence"] = 0.0
            all_results.append(day_df)
            continue
        
        # 昨日涨停股今天的平均涨幅
        today_pct = day_df["pct_chg"].values / 100
        limit_ret = today_pct[yest_limit]
        premium_raw = np.mean(limit_ret)
        
        # 市场平均
        mkt = day_df["market_ret"].iloc[0] / 100 if pd.notna(day_df["market_ret"].iloc[0]) else 0
        
        # 所有昨日涨停股今天的个体表现
        limit_individual_residual = today_pct - mkt
        limit_individual_excess = today_pct - mkt  # = alpha
        
        # 对非涨停股，用回归残差（但这里简化：直接给0）
        limit_individual_residual[~yest_limit] = 0.0
        limit_individual_excess[~yest_limit] = 0.0
        
        # 计算昨日涨停股之间的z-score
        if yest_limit.sum() > 5:
            limit_z = (limit_ret - np.mean(limit_ret)) / (np.std(limit_ret) + 1e-8)
            # 缩放到个股
            z_scores = np.zeros(len(day_df))
            z_scores[yest_limit] = limit_z
            z_scores[~yest_limit] = 0.0
        else:
            z_scores = np.zeros(len(day_df))
        
        # 连板率
        persistence = np.mean(day_df["limit_up"].values[yest_limit]) if yest_limit.sum() > 0 else 0
        persistence_scores = np.zeros(len(day_df))
        persistence_scores[yest_limit] = persistence
        
        # 赋值
        day_df["limit_premium_raw"] = premium_raw
        day_df["limit_premium_excess"] = premium_raw - mkt
        day_df["limit_premium_residual"] = limit_individual_residual
        day_df["limit_premium_zscore"] = z_scores
        day_df["limit_premium_persistence"] = persistence_scores
        
        # 对非昨日涨停的股票，这个因子也重要——说明市场情绪好
        # 如果一个非涨停股处于昨日涨停股聚集的板块 = 板块效应
        # 这里简化为全员赋值
        day_df["limit_market_sentiment"] = premium_raw - mkt
        
        all_results.append(day_df)
    
    print(f"    处理 {ym}...")

daily = pd.concat(all_results, ignore_index=True)

factor_cols = ["limit_premium_raw", "limit_premium_excess", "limit_premium_residual",
               "limit_premium_zscore", "limit_premium_persistence", "limit_market_sentiment"]
factor_cols = [c for c in factor_cols if c in daily.columns]

print(f"\n  因子统计:")
for col in factor_cols:
    valid = daily[col].notna().sum()
    print(f"    {col:30s}: {valid:>10,} 非空, 均值={daily[col].mean():.4f}")

# 4. IC测试
print(f"\n[4/4] IC测试...")
panel = pd.read_parquet(os.path.join(DATA_FACTORS, "factor_panel_with_fwd_v2.parquet"))
fwd = panel[["ts_code", "trade_date", "fwd_20d_ret"]].copy()
fwd["trade_date"] = pd.to_datetime(fwd["trade_date"])
daily["trade_date"] = pd.to_datetime(daily["trade_date"])

merged = daily.merge(fwd, on=["ts_code", "trade_date"], how="inner")
merged["_ym2"] = merged["trade_date"].dt.to_period("M")
print(f"  合并: {len(merged):,}行, {merged['_ym2'].nunique()}个月")

print(f"\n{'='*60}")
print(f"  {'因子':30s} {'IC均值':>8s} {'IC_IR':>8s} {'月数':>5s}")
print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*5}")

ic_results = {}
for factor in factor_cols:
    ics = []
    for ym, g in merged.groupby("_ym2"):
        gv = g[[factor, "fwd_20d_ret"]].dropna()
        if len(gv) < 50:
            continue
        r, _ = spearmanr(gv[factor], gv["fwd_20d_ret"])
        if not np.isnan(r):
            ics.append(r)
    
    if ics:
        ic_mean = float(np.mean(ics))
        ic_std = float(np.std(ics))
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0
        ic_results[factor] = {
            "ic_mean": ic_mean, "ic_std": ic_std, "ic_ir": ic_ir, "n_months": len(ics),
        }
        print(f"  {factor:30s}: {ic_mean*100:+7.2f}% {ic_ir:7.2f} {len(ics):5d}")

# 保存
with open(f"{OUTPUT_DIR}/limit_premium_ic.json", "w") as f:
    json.dump(ic_results, f, indent=2, default=str)

# 合并到面板v6
merge_cols = ["ts_code", "trade_date"] + factor_cols
to_merge = daily[merge_cols].copy().drop_duplicates(subset=["ts_code", "trade_date"])

panel_v6 = pd.read_parquet(os.path.join(DATA_FACTORS, "factor_panel_v6.parquet"))
panel_v6["trade_date"] = pd.to_datetime(panel_v6["trade_date"])
panel_v6 = panel_v6.merge(to_merge, on=["ts_code", "trade_date"], how="left")
panel_v6.to_parquet(os.path.join(DATA_FACTORS, "factor_panel_v6.parquet"))
print(f"\n  ✅ 面板v6已更新: {len(panel_v6.columns)}列")

print(f"\n{'='*60}")
print(f"✅ 昨日涨停溢价残差因子完成")
print(f"{'='*60}")
