"""
放量高开因子 - 参数优化版
原公式 signal_raw IR -0.45，样本太少(416次)
用放宽版和连续版测试
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

print(f"\n📈 放量高开因子优化版 — {datetime.now().strftime('%F %H:%M')}")
print("=" * 60)

RAW_DIR = "data/raw/daily"
OUTPUT_DIR = "data/new_factors"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 加载
print("[1/4] 加载数据...")
files = sorted(glob.glob(f"{RAW_DIR}/*.parquet"))
codes = [os.path.basename(f).replace('.parquet','') for f in files]
print(f"  股票数: {len(codes)}")

# ST标记
stock_list = pd.read_parquet("data/raw/stock_list.parquet")
st_codes = set()
if "name" in stock_list.columns:
    for _, row in stock_list.iterrows():
        name = str(row.get("name", ""))
        if name and ("ST" in name or "*ST" in name or "退" in name):
            st_codes.add(row["ts_code"])

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

# 2. 基础计算
daily["is_st"] = daily["ts_code"].isin(st_codes)
vol_ma5 = daily.groupby("ts_code")["vol"].transform(lambda x: x.rolling(5, min_periods=4).mean())
daily["volume_ratio"] = np.where(vol_ma5 > 0, daily["vol"] / vol_ma5, 1.0)
daily["gap_up"] = np.where(daily["pre_close"] > 0, daily["open"] / daily["pre_close"] - 1, np.nan)
daily["vol_yesterday"] = daily.groupby("ts_code")["vol"].shift(1)

# 3. 多组参数测试
print("\n[2/4] 参数扫描...")

params = [
    # (量比下限, 量比上限, 高开下限, 高开上限, 名称)
    (4.0, 999, 0.002, 0.07, "原版"),      # 原始条件
    (2.0, 999, 0.002, 0.07, "量比2"),      # 放宽量比
    (1.5, 999, 0.002, 0.07, "量比1.5"),    # 更宽
    (1.5, 20, 0.002, 0.05, "量比1.5-高开0.5"), # 去掉极端值
    (2.0, 20, 0.003, 0.05, "量比2-高开0.3"),  # 适中
    (3.0, 999, 0.002, 0.07, "量比3"),      # 折中
    (2.0, 10, 0.002, 0.04, "量比2-高开0.4"),  # 严格
    (1.0, 999, 0.002, 0.07, "量比1"),      # 几乎无条件
]

panel = pd.read_parquet(os.path.join(DATA_FACTORS, "factor_panel_with_fwd_v2.parquet"))
fwd = panel[["ts_code", "trade_date", "fwd_20d_ret"]].copy()
fwd["trade_date"] = pd.to_datetime(fwd["trade_date"])

# 逐参数测试
results = []
for vr_low, vr_high, gap_low, gap_high, name in params:
    # 条件
    signal = (
        (daily["volume_ratio"] > vr_low) 
        & (daily["volume_ratio"] < vr_high)
        & (daily["gap_up"] > gap_low) 
        & (daily["gap_up"] < gap_high)
        & (daily["vol"] > daily["vol_yesterday"].fillna(0) * 0.008)
        & ~daily["is_st"]
    ).astype(int)
    
    daily[f"signal_{name}"] = signal
    
    sample_count = signal.sum()
    daily[f"signal_{name}_score"] = np.where(
        signal == 1,
        daily["volume_ratio"].clip(vr_low, 20) * daily["gap_up"].clip(gap_low, 0.07) * 100,
        0
    )
    
    # 合并
    merged = daily[["ts_code", "trade_date", f"signal_{name}", f"signal_{name}_score"]].merge(
        fwd, on=["ts_code", "trade_date"], how="inner"
    )
    merged["_ym"] = merged["trade_date"].dt.to_period("M")
    
    # 分别测signal和score
    for col_key, col_name in [(f"signal_{name}", "raw"), (f"signal_{name}_score", "score")]:
        ics = []
        for ym, g in merged.groupby("_ym"):
            gv = g[[col_key, "fwd_20d_ret"]].dropna()
            if len(gv) < 50:
                continue
            r, _ = spearmanr(gv[col_key], gv["fwd_20d_ret"])
            if not np.isnan(r):
                ics.append(r)
        
        if ics:
            ic_mean = float(np.mean(ics))
            ic_std = float(np.std(ics))
            ic_ir = ic_mean / ic_std if ic_std > 0 else 0
            results.append({
                "param": name,
                "type": col_name,
                "sample_count": int(sample_count),
                "ic_mean": ic_mean,
                "ic_ir": ic_ir,
                "n_months": len(ics),
                "abs_gt_2pct": float(np.mean([abs(x) > 0.02 for x in ics])),
            })

# 打印结果
print(f"\n{'='*80}")
print(f"  {'参数名':15s} {'类型':8s} {'样本':>6s} {'IC均值':>8s} {'IC_IR':>8s} {'月数':>5s} {'IC>2%':>8s}")
print(f"  {'-'*15} {'-'*8} {'-'*6} {'-'*8} {'-'*8} {'-'*5} {'-'*8}")
results.sort(key=lambda x: -abs(x["ic_ir"]))
for r in results:
    print(f"  {r['param']:15s} {r['type']:8s} {r['sample_count']:6d} {r['ic_mean']*100:+7.2f}% {r['ic_ir']:7.2f} {r['n_months']:5d} {r['abs_gt_2pct']*100:7.0f}%")

# 选择最优版纳入面板
print(f"\n[3/4] 选择最优版...")
best_raw = results[0] if results else None
print(f"  最优原始: {best_raw['param']} IR={best_raw['ic_ir']:.2f}")

# 取2个最优信号纳入面板
best_param = best_raw['param']
factor_cols = [f"signal_{best_param}", f"signal_{best_param}_score"]

# 同时也放一个连续组合因子：量比×高开的连续值（不含硬阈值）
daily["gap_volume_combo"] = np.where(
    ~daily["is_st"],
    daily["volume_ratio"].clip(0, 10) * (daily["gap_up"].clip(-0.05, 0.10) + 0.05) * 10,
    0
)

factor_cols.append("gap_volume_combo")

# 4. IC测试（仅最优版展示）
print(f"\n[4/4] 最终IC测试 + 合并面板...")
merged_full = daily[["ts_code", "trade_date"] + factor_cols].merge(
    fwd, on=["ts_code", "trade_date"], how="inner"
)
merged_full["_ym"] = merged_full["trade_date"].dt.to_period("M")

print(f"\n  最终因子:")
for factor in factor_cols:
    ics = []
    for ym, g in merged_full.groupby("_ym"):
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
        print(f"    {factor:30s}: IC={ic_mean*100:+7.2f}%  IR={ic_ir:7.2f}  ({len(ics)}个月)")

# 合并面板
merge_cols = ["ts_code", "trade_date"] + factor_cols
to_merge = daily[merge_cols].copy().drop_duplicates(subset=["ts_code", "trade_date"])

panel_v6 = pd.read_parquet(os.path.join(DATA_FACTORS, "factor_panel_v6.parquet"))
panel_v6["trade_date"] = pd.to_datetime(panel_v6["trade_date"])
panel_v6 = panel_v6.merge(to_merge, on=["ts_code", "trade_date"], how="left")
panel_v6.to_parquet(os.path.join(DATA_FACTORS, "factor_panel_v6.parquet"))
print(f"\n  ✅ 面板v6已更新: {len(panel_v6.columns)}列 ({len(factor_cols)}个新增)")

print(f"\n{'='*60}")
print(f"✅ 放量高开因子优化完成")
print(f"{'='*60}")
