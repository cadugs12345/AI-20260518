"""
高低切情绪因子 (High-to-Low Rotation)
定义: 资金从高位股切换到低位股的强度

子因子:
  1. hilo_spread: 高位-低位组的收益率差（负值=资金切向低位）
  2. hilo_momentum_ratio: 低位动量/高位动量 比
  3. hilo_volume_ratio: 低位放量/高位放量 比
  4. hilo_count_ratio: 低位涨停数/高位涨停数 比
  5. rotation_intensity: 综合轮动强度
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

print(f"\n🔄 高低切情绪因子 — {datetime.now().strftime('%F %H:%M')}")
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

# 2. 计算各股高/低位特征
print("\n[2/4] 计算高低位特征...")

# 用20日涨幅定义高/低位股
daily["ret_20d"] = daily.groupby("ts_code")["close"].transform(
    lambda x: x.pct_change(20)
)
daily["ret_5d"] = daily.groupby("ts_code")["close"].transform(
    lambda x: x.pct_change(5)
)

# 涨停识别
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

# 3. 截面滚动计算高低切指标
print("\n[3/4] 截面高低切指标...")

# 每月计算一次截面分组
daily["_ym"] = daily["trade_date"].dt.to_period("M")

def calc_rotation_metrics(group):
    """对一个月内所有股票计算高低切指标"""
    if len(group) < 100:
        return group
    
    # 同一天内的截面分组
    results = []
    for date, day_df in group.groupby("trade_date"):
        day_df = day_df.copy()
        if len(day_df) < 100:
            continue
        
        # 按20日涨幅分高/中/低三组
        ret_col = "ret_20d"
        if day_df[ret_col].notna().sum() < 50:
            continue
        
        # 三等分
        try:
            q_high = day_df[ret_col].quantile(0.67)
            q_low = day_df[ret_col].quantile(0.33)
        except:
            continue
        
        high_mask = day_df[ret_col] >= q_high
        mid_mask = (day_df[ret_col] < q_high) & (day_df[ret_col] > q_low)
        low_mask = day_df[ret_col] <= q_low
        
        # 当日收益率
        day_pct = day_df["pct_chg"].values / 100
        
        # 高位/低位组的平均当日收益
        high_mean_ret = np.mean(day_pct[high_mask.values]) if high_mask.any() else 0
        low_mean_ret = np.mean(day_pct[low_mask.values]) if low_mask.any() else 0
        mid_mean_ret = np.mean(day_pct[mid_mask.values]) if mid_mask.any() else 0
        
        # 均值偏离（整个市场的均值）
        market_mean = np.mean(day_pct)
        
        # 放量比: 低位组放量/高位组放量
        vol_col = "vol"
        high_vol_ratio = np.mean(day_df.loc[high_mask.values, vol_col] / 
                                 day_df.loc[high_mask.values, vol_col].rolling(20).mean().fillna(1)) if high_mask.any() else 1
        low_vol_ratio = np.mean(day_df.loc[low_mask.values, vol_col] / 
                                day_df.loc[low_mask.values, vol_col].rolling(20).mean().fillna(1)) if low_mask.any() else 1
        
        # 涨停比: 低位涨停数/高位涨停数
        high_limit = day_df.loc[high_mask.values, "limit_up"].sum() if high_mask.any() else 0
        low_limit = day_df.loc[low_mask.values, "limit_up"].sum() if low_mask.any() else 0
        
        # 每个股票都赋予相同的当日截面指标
        day_df["hilo_spread"] = high_mean_ret - low_mean_ret  # 正=高位强, 负=高低切换
        day_df["hilo_vol_ratio"] = low_vol_ratio / (high_vol_ratio + 1e-8)
        day_df["hilo_limit_ratio"] = (low_limit + 1) / (high_limit + 1)
        
        # 综合轮动: 低位强于高位 = 高低切正信号
        day_df["rotation_intensity"] = (
            -np.clip(day_df["hilo_spread"], -0.05, 0.05) * 10 * 0.4 +  # 负spread=能切入低位
            np.clip(day_df["hilo_vol_ratio"] - 1, -1, 3) * 0.3 +        # 低位放量=好
            np.clip(day_df["hilo_limit_ratio"] - 1, -5, 10) * 0.03      # 低位涨停多=好
        )
        
        # 个股层面的因子: 与自身高位/低位的关系
        # 如果该股本身属于低位组，且高低切正在发生 → 加分
        # 高位组且高低切正在发生 → 减分
        low_score = low_mask.astype(float) * day_df["rotation_intensity"]
        high_score = high_mask.astype(float) * (-day_df["rotation_intensity"])
        
        day_df["hilo_signal"] = low_score + high_score
        
        results.append(day_df)
    
    if results:
        return pd.concat(results, ignore_index=True)
    return group

print("  计算截面指标（按月分组合并）...")
# 用groupby.apply太慢，改成逐月处理
all_results = []
for ym, g in daily.groupby("_ym"):
    result = calc_rotation_metrics(g)
    if result is not None and len(result) > 0:
        all_results.append(result)
    # 只处理最近60个月以节省时间
    print(f"    处理 {ym}...")

daily = pd.concat(all_results, ignore_index=True) if all_results else daily

factor_cols = ["hilo_spread", "hilo_vol_ratio", "hilo_limit_ratio", 
               "rotation_intensity", "hilo_signal"]
factor_cols = [c for c in factor_cols if c in daily.columns]

print(f"\n  因子统计:")
for col in factor_cols:
    valid = daily[col].notna().sum()
    print(f"    {col:25s}: {valid:>10,} 非空, 均值={daily[col].mean():.4f}")

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
with open(f"{OUTPUT_DIR}/hilo_rotation_ic.json", "w") as f:
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
print(f"✅ 高低切情绪因子完成")
print(f"{'='*60}")
