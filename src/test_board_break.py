"""
断板修复力度因子
定义: 涨停炸板后第二天的修复能力
核心逻辑:
  1. 识别炸板日: 盘中曾涨停(high≈limit up)但收盘未封住(close < high*0.99)
  2. 修复力度: (第二天高开幅度 + 第二天涨幅) / 炸板跌幅
  3. 炸板深度: (最高价-收盘价)/最高价  越深说明抛压越大
  4. n日修复: 炸板后N个交易日的累计反弹幅度

输出因子:
  - board_break_depth: 炸板深度 (负值，越大说明当日抛压越重)
  - board_repair_1d: 1日修复力度
  - board_repair_5d: 5日修复力度  
  - board_break_signal: 炸板预警信号 (综合)
"""
import sys, os, json, glob
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config.settings import DATA_FACTORS

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT)

print(f"\n💥 断板修复力度因子 — {datetime.now().strftime('%F %H:%M')}")
print("=" * 60)

RAW_DIR = "data/raw/daily"
OUTPUT_DIR = "data/new_factors"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 加载日线
print("[1/4] 加载日线...")
files = sorted(glob.glob(f"{RAW_DIR}/*.parquet"))
codes = [os.path.basename(f).replace('.parquet','') for f in files]
print(f"  股票数: {len(codes)}")

# 计算涨停价（按10%和20%分板块识别）
chunks = []
for i, (f, code) in enumerate(zip(files, codes)):
    try:
        df = pd.read_parquet(f)
        df["ts_code"] = code
        columns = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "pct_chg", "vol", "amount"]
        chunks.append(df[[c for c in columns if c in df.columns]])
    except:
        pass
    if (i+1) % 2000 == 0:
        print(f"  进度: {i+1}/{len(codes)}")

daily = pd.concat(chunks, ignore_index=True)
daily["trade_date"] = pd.to_datetime(daily["trade_date"])
daily = daily.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
print(f"  数据: {len(daily):,}行, {daily['ts_code'].nunique()}只")

# 2. 识别涨停炸板
print("\n[2/4] 识别涨停炸板...")

# 计算涨停价: 主板10%, 创业板/科创板20%, 北交所30%
def get_limit_pct(code):
    if code.startswith('30') or code.startswith('688'):
        return 0.20
    elif code.startswith('8') or code.startswith('4'):
        return 0.30
    else:
        return 0.10

limit_pcts = daily["ts_code"].map(get_limit_pct)
daily["limit_high"] = daily["pre_close"] * (1 + limit_pcts)  # 涨停价
daily["limit_low"] = daily["pre_close"] * (1 - limit_pcts)   # 跌停价

# 炸板识别: 最高价接近涨停(>=涨停价*0.99)但收盘未封住(<涨停价*0.995)
daily["touched_limit"] = daily["high"] >= daily["limit_high"] * 0.99
daily["closed_limit"] = daily["close"] >= daily["limit_high"] * 0.995
daily["board_break"] = daily["touched_limit"] & ~daily["closed_limit"]

# 炸板深度: 从最高点的回落幅度
daily["board_break_depth"] = np.where(
    daily["board_break"],
    (daily["high"] - daily["close"]) / (daily["high"] - daily["low"] + 1e-8),  # 0=浅, 1=深
    0
)

# 炸板幅: 从涨停到收盘的绝对跌幅
daily["board_break_pct"] = np.where(
    daily["board_break"],
    (daily["limit_high"] - daily["close"]) / daily["limit_high"] * 100,
    0
)

print(f"  炸板样本: {daily['board_break'].sum():,}次")
print(f"  炸板率: {daily['board_break'].sum()/daily['touched_limit'].sum()*100:.1f}%")

# 3. 修复力度计算
print("\n[3/4] 计算修复力度...")

# 向前滚动计算修复
# 修复1: 次日高开幅度
# 修复2: 次日到5日累计涨幅

# 在股票分组内shift(-1)拿次日数据
daily["next_open"] = daily.groupby("ts_code")["open"].shift(-1)
daily["next_close"] = daily.groupby("ts_code")["close"].shift(-1)
daily["next_pct"] = daily.groupby("ts_code")["pct_chg"].shift(-1)

# 1日修复: 炸板后第2天(次日)收盘 vs 炸板日收盘
daily["repair_1d"] = daily.groupby("ts_code")["pct_chg"].shift(-1) / 100  # 次日涨跌幅

# 修正: 正的修复 = 次日上涨 => good
# 修复力度 = 修复幅度 / 炸板深度
daily["repair_force_1d"] = np.where(
    daily["board_break"] & (daily["board_break_depth"] > 0.01),
    daily["repair_1d"] / daily["board_break_depth"],
    0
)

# 5日累计修复 (shift(-1)到shift(-5)的累计)
for d in [3, 5, 10]:
    cum_ret = np.zeros(len(daily))
    for j in range(1, d+1):
        cum_ret += daily.groupby("ts_code")["pct_chg"].shift(-j).fillna(0) / 100
    daily[f"repair_{d}d"] = cum_ret
    daily[f"repair_force_{d}d"] = np.where(
        daily["board_break"] & (daily["board_break_depth"] > 0.01),
        cum_ret / daily["board_break_depth"],
        0
    )

# 综合信号: 修复力度加权
daily["board_repair_score"] = (
    daily["repair_force_1d"].clip(-5, 5) * 0.4 +
    daily["repair_force_3d"].clip(-5, 5) * 0.3 +
    daily["repair_force_5d"].clip(-5, 5) * 0.2 +
    daily["repair_force_10d"].clip(-5, 5) * 0.1
)

# 也做一个非炸板日的默认值: 用"是否触碰涨停"替代
# 对于没炸板但触碰过涨停的 = 强势封板 = 好信号
daily["limit_up_quality"] = np.where(
    daily["closed_limit"] & daily["touched_limit"],
    1.0,  # 强势封板
    np.where(
        daily["board_break"],
        daily["board_repair_score"],  # 炸板修复力度
        0  # 普通日
    )
)

factor_cols = [
    "board_break", "board_break_depth", "board_break_pct",
    "repair_force_1d", "repair_force_3d", "repair_force_5d", "repair_force_10d",
    "board_repair_score", "limit_up_quality"
]

print(f"\n  因子统计:")
for col in factor_cols:
    if col in daily.columns:
        valid = daily[col].notna().sum()
        print(f"    {col:25s}: {valid:>10,} 非空, 均值={daily[col].mean():.4f}")

# 4. IC测试
print(f"\n[4/4] IC测试...")
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
test_cols = [c for c in factor_cols if c in merged.columns and c != "board_break"]
# board_break是布尔值，单独处理
test_cols += ["board_break"] if "board_break" in merged.columns else []

for factor in test_cols:
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
with open(f"{OUTPUT_DIR}/board_break_ic.json", "w") as f:
    json.dump(ic_results, f, indent=2, default=str)

# 合并到面板v6
merge_cols = ["ts_code", "trade_date"] + [c for c in factor_cols if c in daily.columns]
to_merge = daily[merge_cols].copy().drop_duplicates(subset=["ts_code", "trade_date"])

panel_v6 = pd.read_parquet(os.path.join(DATA_FACTORS, "factor_panel_v6.parquet"))
panel_v6["trade_date"] = pd.to_datetime(panel_v6["trade_date"])
panel_v6 = panel_v6.merge(to_merge, on=["ts_code", "trade_date"], how="left")
panel_v6.to_parquet(os.path.join(DATA_FACTORS, "factor_panel_v6.parquet"))
print(f"\n  ✅ 面板v6已更新: {len(panel_v6.columns)}列")

print(f"\n{'='*60}")
print(f"✅ 断板修复力度因子完成")
print(f"{'='*60}")
