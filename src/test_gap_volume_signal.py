"""
放量高开选股因子 (通达信公式复刻)
逻辑: JJLB>4 AND JE>ZB*0.008 AND KG>1.002 AND KG<1.07 AND NOT ST

子因子分解:
  1. volume_ratio > 4 (量比>4)
  2. volume_surge (今日量>昨日*0.008)
  3. gap_up = open/pre_close - 1
  4. signal_raw: 全部条件满足=1
  5. signal_score: 综合强度（量比×高开幅度，条件内连续值）
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

print(f"\n📈 放量高开选股因子 (通达信复刻) — {datetime.now().strftime('%F %H:%M')}")
print("=" * 60)

RAW_DIR = "data/raw/daily"
OUTPUT_DIR = "data/new_factors"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 加载日线+ST标记
print("[1/4] 加载数据...")
files = sorted(glob.glob(f"{RAW_DIR}/*.parquet"))
codes = [os.path.basename(f).replace('.parquet','') for f in files]
print(f"  股票数: {len(codes)}")

# 加载股票列表获取ST标记
stock_list = pd.read_parquet("data/raw/stock_list.parquet")
st_codes = set()
if "name" in stock_list.columns:
    for _, row in stock_list.iterrows():
        name = str(row.get("name", ""))
        if name and ("ST" in name or "*ST" in name or "退" in name):
            st_codes.add(row["ts_code"])
print(f"  ST股票: {len(st_codes)}只")

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

# 2. 计算因子
print("\n[2/4] 计算因子...")

# 基础因子: 量比 = vol / 5日均量
vol_ma5 = daily.groupby("ts_code")["vol"].transform(
    lambda x: x.rolling(5, min_periods=4).mean()
)
# 通达信量比= vol / MA(vol, 5)
daily["volume_ratio"] = np.where(vol_ma5 > 0, daily["vol"] / vol_ma5, 1.0)

# 高开幅度 KG = open / pre_close - 1
daily["gap_up"] = np.where(
    daily["pre_close"] > 0,
    daily["open"] / daily["pre_close"] - 1,
    np.nan
)

# 昨日成交量 ZB
daily["vol_yesterday"] = daily.groupby("ts_code")["vol"].shift(1)

# ST标记
daily["is_st"] = daily["ts_code"].isin(st_codes)

# 个股流通市值近似 (用 close * vol_ma20 / 换手率 或 直接使用amount)
# 简化: CAPITAL/1000000 用市值代替
# 条件 JJJE > ZB*0.008 实际是 > 昨日量*0.8%
# 通达信JJJE是成交量(手)，VOL是股数(手已除以100)
# 简化：条件 = vol > vol_yesterday * 0.008
daily["volume_min_cond"] = daily["vol"] > daily["vol_yesterday"] * 0.008

# 选股条件
daily["cond_volume_ratio"] = daily["volume_ratio"] > 4.0
daily["cond_volume_min"] = daily["volume_min_cond"]
daily["cond_gap_up_low"] = daily["gap_up"] > 0.002  # >0.2%
daily["cond_gap_up_high"] = daily["gap_up"] < 0.07  # <7%
daily["cond_not_st"] = ~daily["is_st"]

# 原始信号: 全部条件满足=1
daily["signal_raw"] = (
    daily["cond_volume_ratio"] 
    & daily["cond_volume_min"]
    & daily["cond_gap_up_low"] 
    & daily["cond_gap_up_high"] 
    & daily["cond_not_st"]
).astype(int)

# 综合强度分: 在满足条件的样本内，量化比×高开幅度
daily["signal_score"] = np.where(
    daily["signal_raw"] == 1,
    daily["volume_ratio"].clip(4, 20) * daily["gap_up"].clip(0.002, 0.07) * 100,
    0
)

# 独立子因子
daily["gap_up_intensity"] = np.where(
    daily["cond_gap_up_low"] & daily["cond_gap_up_high"] & daily["cond_not_st"],
    daily["gap_up"],
    0
)

daily["vol_ratio_intensity"] = np.where(
    daily["cond_volume_min"] & daily["cond_not_st"],
    daily["volume_ratio"].clip(0, 20) - 1,
    0
)

print(f"\n  条件统计:")
for cond in ["cond_volume_ratio", "cond_volume_min", "cond_gap_up_low", "cond_gap_up_high", "cond_not_st"]:
    print(f"    {cond:25s}: {daily[cond].sum():>10,}")

print(f"\n  信号统计:")
print(f"    signal_raw        : {daily['signal_raw'].sum():>10,}  ({daily['signal_raw'].mean()*100:.4f}%)")
print(f"    signal_score非零  : {(daily['signal_score'] > 0).sum():>10,}")

factor_cols = ["signal_raw", "signal_score", "gap_up_intensity", "vol_ratio_intensity"]
factor_cols = [c for c in factor_cols if c in daily.columns]

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
with open(f"{OUTPUT_DIR}/gap_volume_signal_ic.json", "w") as f:
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
print(f"✅ 放量高开选股因子完成")
print(f"{'='*60}")
