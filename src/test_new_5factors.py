"""
5因子批量构建（高效向量化版本）
1. 隔夜动量: (open - pre_close) / pre_close
2. 资金流强度: (close - (h+l)/2) / (h-l)  * sign(pct_chg)
3. 残差波动率: close偏离20日均线的残差std
4. 换手率乖离: 换手率偏离20日均线的z-score
5. 分析师预期上调: 用短/长均线比作为代理（无外部数据时）
"""
import sys, os, json, glob
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config.settings import DATA_FACTORS

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT)

print(f"\n🔥 5因子批量构建（高效版）— {datetime.now().strftime('%F %H:%M')}")
print("=" * 60)

RAW_DIR = "data/raw/daily"
OUTPUT_DIR = "data/new_factors"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: 合并所有日线数据
print("[1/4] 合并日线数据...")
files = sorted(glob.glob(f"{RAW_DIR}/*.parquet"))
codes = [os.path.basename(f).replace('.parquet','') for f in files]
print(f"  股票数: {len(codes)}")

# 逐个读取并追加（避免一次读5000个文件爆内存）
chunks = []
for i, (f, code) in enumerate(zip(files, codes)):
    try:
        df = pd.read_parquet(f)
        df["ts_code"] = code
        chunks.append(df[["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "pct_chg", "vol", "amount"]])
    except:
        pass
    
    if (i+1) % 1000 == 0:
        print(f"  读取: {i+1}/{len(codes)}...")

print(f"  合并中...")
daily = pd.concat(chunks, ignore_index=True)
daily["trade_date"] = pd.to_datetime(daily["trade_date"])
daily = daily.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
print(f"  日线数据: {len(daily):,}行, {daily['ts_code'].nunique()}只股票")
print(f"  日期范围: {daily['trade_date'].min().date()} ~ {daily['trade_date'].max().date()}")

# Step 2: 向量化计算因子
print("\n[2/4] 计算因子...")

# 1. 隔夜动量
print("  1/5 隔夜动量...")
daily["overnight_ret"] = np.where(
    daily["pre_close"] > 0,
    daily["open"] / daily["pre_close"] - 1,
    np.nan
)

# 2. 资金流强度 (简化版)
print("  2/5 资金流强度...")
# 使用价格位置判断资金流向
hl_range = daily["high"] - daily["low"]
daily["moneyflow_raw"] = np.where(
    hl_range > 0,
    (daily["close"] - (daily["high"] + daily["low"]) / 2) / hl_range,
    0
)
# 乘上涨跌幅方向（涨时正流入×正，跌时正流入×负 = 背离）
daily["moneyflow_strength"] = daily["moneyflow_raw"] * daily["pct_chg"].clip(-10, 10)

# 3. 残差波动率
print("  3/5 残差波动率...")
daily["ma20"] = daily.groupby("ts_code")["close"].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)
daily["resid"] = daily["close"] - daily["ma20"]
daily["idvol"] = daily.groupby("ts_code")["resid"].transform(
    lambda x: x.rolling(20, min_periods=10).std()
) / (daily["ma20"] + 1e-8) * 100

# 4. 换手率乖离
print("  4/5 换手率乖离...")
# vol本身不是换手率，但vol的20日均线可以做标准化
vol_ma20 = daily.groupby("ts_code")["vol"].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)
vol_std20 = daily.groupby("ts_code")["vol"].transform(
    lambda x: x.rolling(20, min_periods=10).std()
)
daily["turnover_bias"] = np.where(
    vol_std20 > 0,
    (daily["vol"] - vol_ma20) / vol_std20,
    0
)

# 5. 分析师预期上调（代理因子）
print("  5/5 预期上调代理...")
# 使用短/长均线比作为代理：5日均线/20日均线
ma5 = daily.groupby("ts_code")["close"].transform(
    lambda x: x.rolling(5, min_periods=3).mean()
)
daily["revise_up_proxy"] = ma5 / daily["ma20"] - 1

# 清理
new_factor_cols = ["overnight_ret", "moneyflow_raw", "moneyflow_strength", 
                   "idvol", "turnover_bias", "revise_up_proxy"]
print(f"\n  因子统计:")
for col in new_factor_cols:
    valid = daily[col].notna().sum()
    print(f"    {col:25s}: {valid:>10,} 非空, 均值={daily[col].mean():.4f}")

# Step 3: 保存并IC测试
print("\n[3/4] 保存 & IC测试...")

# 保存新因子
new_factors = daily[["ts_code", "trade_date"] + new_factor_cols].copy()
new_factors.to_parquet(f"{OUTPUT_DIR}/five_factors.parquet")
print(f"  已保存: {OUTPUT_DIR}/five_factors.parquet")

# 加载面板的未来收益
print("  加载未来收益...")
panel = pd.read_parquet(os.path.join(DATA_FACTORS, "factor_panel_with_fwd_v2.parquet"))
fwd = panel[["ts_code", "trade_date", "fwd_20d_ret"]].copy()
fwd["trade_date"] = pd.to_datetime(fwd["trade_date"])
new_factors["trade_date"] = pd.to_datetime(new_factors["trade_date"])

# 合并
merged = new_factors.merge(fwd, on=["ts_code", "trade_date"], how="inner")
merged["_ym"] = merged["trade_date"].dt.to_period("M")
print(f"  合并后: {len(merged):,}行")

# IC计算
print("\n  IC测试结果:")
print(f"  {'='*60}")
ic_results = {}
for factor in new_factor_cols:
    if factor not in merged.columns:
        continue
    ics = []
    for ym, g in merged.groupby("_ym"):
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
            "abs_ic_gt_2pct": float(np.mean([abs(x) > 0.02 for x in ics])),
        }
        print(f"  {factor:25s}: IC={ic_mean*100:+.2f}%  IR={ic_ir:.2f}  ({len(ics)}个月)")

# 保存IC结果
with open(f"{OUTPUT_DIR}/ic_results.json", "w") as f:
    json.dump(ic_results, f, indent=2, default=str)

# Step 4: 合并到面板v6
print(f"\n[4/4] 合并到面板v6...")
panel_v6 = pd.read_parquet(os.path.join(DATA_FACTORS, "factor_panel_v6.parquet"))
panel_v6["trade_date"] = pd.to_datetime(panel_v6["trade_date"])

merge_data = new_factors.copy()
merge_data["trade_date"] = pd.to_datetime(merge_data["trade_date"])

panel_v6 = panel_v6.merge(merge_data, on=["ts_code", "trade_date"], how="left")
panel_v6.to_parquet(os.path.join(DATA_FACTORS, "factor_panel_v6.parquet"))
print(f"  ✅ 面板v6已更新: {len(panel_v6.columns)}列")
print(f"  新增列: {[c for c in new_factor_cols if c in panel_v6.columns]}")

print(f"\n{'='*60}")
print(f"✅ 5因子批量测试完成")
print(f"{'='*60}")
