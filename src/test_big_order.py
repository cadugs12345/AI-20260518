"""
大单净量因子 (大单判定升级版)
没有逐笔数据，用日级量价结合推断大单方向

核心改进点:
  1. 量价背离度 — 涨时缩量/跌时放量 = 大单出货
  2. 价格位置 + 量能突变 — 在关键位置突然放量 = 大单介入
  3. 日内均价偏离 — (amount/vol) vs (h+l)/2 的偏离判断主导方向
  4. 超大单近似 — 用极值量(vol > 99%分位) + 特殊价格行为

输出因子:
  - net_big_order_raw: 大单净量（正=买入）
  - net_big_order_smooth: 5日平滑
  - big_order_ratio: 大单成交占比
  - big_order_divergence: 量价背离得分
  - big_order_momentum: 大单动量（连续净买入强度）
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

print(f"\n💰 大单净量因子（升级版）— {datetime.now().strftime('%F %H:%M')}")
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

# 2. 核心因子计算
print("\n[2/4] 计算大单净量子因子...")

# --- 2a. 日内资金强度 ---
# 均价 = amount/vol（代表平均成交价），对比收盘价决定当日资金方向
# 均价 < close 且 close偏上轨 = 大单在买 / 均价 > close 且 close偏下轨 = 大单在砸
daily["vwap"] = daily["amount"] / (daily["vol"] + 1e-8)
daily["hl_mid"] = (daily["high"] + daily["low"]) / 2
daily["hl_range"] = daily["high"] - daily["low"]
daily["close_position"] = np.where(
    daily["hl_range"] > 0,
    (daily["close"] - daily["low"]) / daily["hl_range"],  # 0=最低 1=最高
    0.5
)

# 大单方向判定:
# close > vwap 且 close偏上轨(>0.6) → 买方主导
# close < vwap 且 close偏下轨(<0.4) → 卖方主导
# 强度 = 位置偏差 * 成交量偏离
daily["vol_ma20"] = daily.groupby("ts_code")["vol"].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)
daily["vol_deviation"] = np.where(
    daily["vol_ma20"] > 0,
    daily["vol"] / daily["vol_ma20"],
    1.0
)

# 大单净量 = 价格位置 vs 均价 的方向 * 量能放大倍数
price_dir = np.where(
    daily["close"] > daily["vwap"],
    daily["close_position"] * 2 - 1,  # [0,1]映射到[-1,1]，close顶=+1
    daily["close_position"] * 2 - 1   # 同上
)
daily["net_order1"] = price_dir * np.log1p(daily["vol_deviation"].clip(0.1, 10))

# --- 2b. 量价背离判定 ---
# 涨时缩量 = 假涨(大单在卖) / 跌时放量 = 真跌(大单在卖)
# 涨时放量 = 真涨(大单在买) / 跌时缩量 = 假跌(大单在吸筹)
pct = daily["pct_chg"] / 100
vol_ratio = daily["vol_deviation"]

# 量价背离得分: 
# 涨(+) × 放量(+) = 正正常 → 0
# 涨(+) × 缩量(-) = 背离 → 负(大单在出货)
# 跌(-) × 放量(+) = 正常 → 0  
# 跌(-) × 缩量(-) = 背离 → 正(大单在吸筹)
daily["volume_price_div"] = -np.sign(pct) * np.sign(vol_ratio - 1) * np.abs(pct) * np.log1p(np.abs(vol_ratio - 1))
daily["volume_price_div"] = daily["volume_price_div"].clip(-0.05, 0.05)

# --- 2c. 超大单信号 ---
# 找出vol超过99%分位数的交易日 = 异常放量
print("  计算量能阈值...")
daily["vol_99pct"] = daily.groupby("ts_code")["vol"].transform(
    lambda x: x.rolling(120, min_periods=60).quantile(0.99)
)
daily["vol_95pct"] = daily.groupby("ts_code")["vol"].transform(
    lambda x: x.rolling(120, min_periods=60).quantile(0.95)
)

# 异常放量 + 价格行为 => 大单判定
daily["abnormal_vol"] = daily["vol"] > daily["vol_95pct"]
daily["extreme_vol"] = daily["vol"] > daily["vol_99pct"]

# 异常放量日，大单方向 = 收盘位置 vs 均价
daily["big_order_signal"] = np.where(
    daily["abnormal_vol"],
    np.where(daily["close"] > daily["vwap"], daily["close_position"], -daily["close_position"]),
    0
)

# --- 2d. 汇总因子 ---
daily["net_big_order_raw"] = daily["net_order1"].clip(-5, 5)

# 5日/10日平滑
daily["net_big_order_smooth5"] = daily.groupby("ts_code")["net_big_order_raw"].transform(
    lambda x: x.rolling(5, min_periods=3).mean()
)
daily["net_big_order_smooth10"] = daily.groupby("ts_code")["net_big_order_raw"].transform(
    lambda x: x.rolling(10, min_periods=5).mean()
)

# 大单占比: 异常放量日的比例
daily["big_order_ratio"] = daily.groupby("ts_code")["abnormal_vol"].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)

# 大单动量: 连续净买入天数
# 大单动量: 连续净买入强度
daily["big_order_dir"] = np.sign(daily["net_big_order_raw"])
daily["big_order_momentum"] = (
    daily.groupby("ts_code")["big_order_dir"].transform(
        lambda x: x.rolling(10, min_periods=5).mean()
    ) *
    daily.groupby("ts_code")["net_big_order_raw"].transform(
        lambda x: x.rolling(10, min_periods=5).mean()
    )
)

factor_cols = ["net_big_order_raw", "net_big_order_smooth5", "net_big_order_smooth10",
               "big_order_ratio", "volume_price_div", "big_order_momentum"]
factor_cols = [c for c in factor_cols if c in daily.columns]

print(f"\n  因子统计:")
for col in factor_cols:
    valid = daily[col].notna().sum()
    print(f"    {col:30s}: {valid:>10,} 非空, 均值={daily[col].mean():.4f}")

# 3. IC测试
print(f"\n[3/4] IC测试...")
panel = pd.read_parquet(os.path.join(DATA_FACTORS, "factor_panel_with_fwd_v2.parquet"))
fwd = panel[["ts_code", "trade_date", "fwd_20d_ret"]].copy()
fwd["trade_date"] = pd.to_datetime(fwd["trade_date"])
daily["trade_date"] = pd.to_datetime(daily["trade_date"])

merged = daily.merge(fwd, on=["ts_code", "trade_date"], how="inner")
merged["_ym"] = merged["trade_date"].dt.to_period("M")
print(f"  合并: {len(merged):,}行, {merged['_ym'].nunique()}个月")

print(f"\n{'='*60}")
print(f"  {'因子':30s} {'IC均值':>8s} {'IC_IR':>8s} {'月数':>5s}")
print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*5}")

ic_results = {}
for factor in factor_cols:
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
        print(f"  {factor:30s}: {ic_mean*100:+7.2f}% {ic_ir:7.2f} {len(ics):5d}")

# 保存
with open(f"{OUTPUT_DIR}/big_order_ic.json", "w") as f:
    json.dump(ic_results, f, indent=2, default=str)

# 4. 合并到面板v6
print(f"\n[4/4] 保存到面板v6...")
merge_cols = ["ts_code", "trade_date"] + factor_cols
to_merge = daily[merge_cols].copy().drop_duplicates(subset=["ts_code", "trade_date"])

panel_v6 = pd.read_parquet(os.path.join(DATA_FACTORS, "factor_panel_v6.parquet"))
panel_v6["trade_date"] = pd.to_datetime(panel_v6["trade_date"])
panel_v6 = panel_v6.merge(to_merge, on=["ts_code", "trade_date"], how="left")
panel_v6.to_parquet(os.path.join(DATA_FACTORS, "factor_panel_v6.parquet"))
print(f"  ✅ 面板v6已更新: {len(panel_v6.columns)}列")

print(f"\n{'='*60}")
print(f"✅ 大单净量因子（升级版）完成")
print(f"{'='*60}")
