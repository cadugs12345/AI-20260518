"""
第2批4因子构建 + IC测试
1. 换手率持续性因子       — 换手率动量/趋势持续性（使用vol滑动窗口）
2. 大单净量分位数因子     — 用amount的大额交易占比做近似
3. 融资余额边际变化因子   — 需外部数据，用price momentum做替代验证
4. 龙虎榜席位溢价因子     — 龙虎榜买入后N日收益溢价
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

print(f"\n🔥 第2批4因子构建 + IC测试 — {datetime.now().strftime('%F %H:%M')}")
print("=" * 60)

RAW_DIR = "data/raw/daily"
OUTPUT_DIR = "data/new_factors"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载已有日线数据（直接读之前做好的）
print("[1/4] 加载日线数据...")
files = sorted(glob.glob(f"{RAW_DIR}/*.parquet"))
codes = [os.path.basename(f).replace('.parquet','') for f in files]
print(f"  股票数: {len(codes)}")

# 上次已经合并过，这次直接从new_factors读取日线缓存，或者重新读并算
# 更高效: 直接从five_factors.parquet加载基础日线
daily_path = f"{OUTPUT_DIR}/five_factors.parquet"
if os.path.exists(daily_path):
    # 重新读取raw做计算（因为需要原始open/high/low/close/vol）
    pass

print("  读取日K线...")
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

# Step 2: 计算因子
print("\n[2/4] 计算4因子...")

# 1. 换手率持续性因子
# 定义: 换手率时间序列的自相关性/趋势持续性
# 用vol的短期(5日)均值/长期(60日)均值比来衡量换手率是否持续偏高
print("  1/4 换手率持续性因子...")
vol_ma5 = daily.groupby("ts_code")["vol"].transform(lambda x: x.rolling(5, min_periods=3).mean())
vol_ma60 = daily.groupby("ts_code")["vol"].transform(lambda x: x.rolling(60, min_periods=30).mean())
vol_ma20 = daily.groupby("ts_code")["vol"].transform(lambda x: x.rolling(20, min_periods=10).mean())

# 持续性: 5日均量 / 60日均量（换手率持续放量=正，持续缩量=负）
daily["turnover_persistence"] = vol_ma5 / vol_ma60 - 1

# 2. 大单净量分位数因子
# 没有逐笔数据，用amount的日间变化做近似
# 核心思路: 量价配合度 — 涨时放量(大单买入)、跌时放量(大单卖出)
# 使用: (涨跌幅 × vol偏离) / 截面分位数
daily["vol_ma20_ratio"] = daily["vol"] / vol_ma20

# 大单净量近似: 如果是上涨日且放量 > 1.5倍 = 正大单
# 如果是下跌日且放量 > 1.5倍 = 负大单
daily["large_trade_raw"] = np.where(
    daily["pct_chg"] > 1.5,
    daily["vol_ma20_ratio"].clip(0, 3),  # 涨且放量
    np.where(
        daily["pct_chg"] < -1.5,
        -daily["vol_ma20_ratio"].clip(0, 3),  # 跌且放量
        daily["pct_chg"] / 10 * daily["vol_ma20_ratio"].clip(0, 2)  # 正常: 涨跌幅度调整
    )
)

# 截面分位数（每月滚动）
def calc_percentile_rank(series):
    """将series转换为截面百分位排名 [0,1]"""
    valid = series.notna()
    if valid.sum() < 10:
        return series * np.nan
    ranks = series.rank(pct=True)
    return ranks

# 按月计算百分位
daily["_ym"] = daily["trade_date"].dt.to_period("M")
daily["large_trade_quantile"] = daily.groupby("_ym")["large_trade_raw"].transform(
    lambda x: x.rank(pct=True)
)

# 3. 融资余额边际变化因子（代理版本）
# 融资余额无法获取，用以下替代:
# - 价格动量 + 换手率变化 代理"加杠杆"行为
# - 上涨且放量 = 可能加杠杆（融资买入增加）
# - 下跌且放量 = 可能去杠杆（融资偿还增加）
daily["margin_proxy"] = daily["pct_chg"] * daily["vol_ma20_ratio"] / 100

# 4. 龙虎榜席位溢价因子
print("  4/4 龙虎榜席位溢价因子...")

# 用已有的toplist数据
toplist_path = os.path.join(DATA_FACTORS, "new_factors/toplist_factors.parquet")
if os.path.exists(toplist_path):
    toplist = pd.read_parquet(toplist_path)
    toplist["trade_date"] = pd.to_datetime(toplist["trade_date"])
    
    # 仅在龙虎榜上榜后才有信号
    # 席位溢价: 上榜后N日是否有超额
    # 这里保守一点，直接用"总净买入"因子
    daily = daily.merge(
        toplist[["ts_code", "trade_date", "总净买入_20d", "上榜次数_20d"]],
        on=["ts_code", "trade_date"], how="left"
    )
    # 席位溢价: 如果机构/知名席位买入多 => 正面信号
    # 我们没有细分席位数据，用净买入方向做简单代理
    daily["seat_premium"] = np.sign(daily["总净买入_20d"]) * daily["上榜次数_20d"]
    daily["seat_premium"] = daily["seat_premium"].fillna(0)
else:
    print("  ⚠️ 无龙虎榜数据，席位溢价因子跳过")
    daily["seat_premium"] = 0

daily.drop(columns=["_ym"], inplace=True, errors="ignore")

# 因子列
new_factor_cols = ["turnover_persistence", "large_trade_raw", "large_trade_quantile", 
                   "margin_proxy", "seat_premium"]
print(f"\n  因子统计:")
for col in new_factor_cols:
    valid = daily[col].notna().sum() if col in daily.columns else 0
    if valid > 0:
        print(f"    {col:30s}: {valid:>10,} 非空, 均值={daily[col].mean():.4f}")

# Step 3: IC测试
print(f"\n[3/4] IC测试...")

# 合并未来收益
panel = pd.read_parquet(os.path.join(DATA_FACTORS, "factor_panel_with_fwd_v2.parquet"))
fwd = panel[["ts_code", "trade_date", "fwd_20d_ret"]].copy()
fwd["trade_date"] = pd.to_datetime(fwd["trade_date"])
daily["trade_date"] = pd.to_datetime(daily["trade_date"])

merged = daily.merge(fwd, on=["ts_code", "trade_date"], how="inner")
merged["_ym"] = merged["trade_date"].dt.to_period("M")
print(f"  合并: {len(merged):,}行, {merged['_ym'].nunique()}个月")

print(f"\n{'='*60}")
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
        }
        print(f"  {factor:30s}: IC={ic_mean*100:+.2f}%  IR={ic_ir:.2f}  ({len(ics)}个月)")

# 保存IC结果
with open(f"{OUTPUT_DIR}/ic_results_batch2.json", "w") as f:
    json.dump(ic_results, f, indent=2, default=str)

# Step 4: 合并到面板v6
print(f"\n[4/4] 保存到面板v6...")
merge_cols = ["ts_code", "trade_date"] + [c for c in new_factor_cols if c in daily.columns]
to_merge = daily[merge_cols].copy().drop_duplicates(subset=["ts_code", "trade_date"])

panel_v6 = pd.read_parquet(os.path.join(DATA_FACTORS, "factor_panel_v6.parquet"))
panel_v6["trade_date"] = pd.to_datetime(panel_v6["trade_date"])
panel_v6 = panel_v6.merge(to_merge, on=["ts_code", "trade_date"], how="left")
panel_v6.to_parquet(os.path.join(DATA_FACTORS, "factor_panel_v6.parquet"))
print(f"  ✅ 面板v6已更新: {len(panel_v6.columns)}列")
print(f"  新增: {[c for c in merge_cols if c not in ['ts_code','trade_date'] and c in panel_v6.columns]}")

print(f"\n{'='*60}")
print(f"✅ 第2批4因子完成")
print(f"{'='*60}")
