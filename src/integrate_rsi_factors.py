"""
将RSI超买超卖因子集成到主因子面板
并测试与现有因子的相关性，以及v12合成效果
"""
import sys, os, json
from datetime import datetime
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config.settings import DATA_FACTORS
from scipy.stats import spearmanr

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT)

print(f"\n🔧 RSI因子集成 & 合成测试 — {datetime.now().strftime('%F %H:%M')}")
print("=" * 60)

# 1. 加载面板v5
print("[1/5] 加载面板v5...")
panel = pd.read_parquet(os.path.join(DATA_FACTORS, "factor_panel_v5.parquet"))
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
print(f"  面板v5: {len(panel):,}行, {panel['trade_date'].nunique()}个交易日")
print(f"  已有因子列(部分): {[c for c in panel.columns if c not in ['ts_code','trade_date','name','close','volume']][:5]}...")

# 2. 构建RSI衍生因子
print("\n[2/5] 构建RSI衍生因子...")
rsi_cols = [c for c in panel.columns if c.startswith("RSI")]
print(f"  基础RSI: {rsi_cols}")

new_factors = {}

# A: RSI反转强度
for col in rsi_cols:
    name = f"{col}_反转强度"
    vals = panel[col].values.copy()
    factor = np.zeros(len(vals))
    mask = ~np.isnan(vals)
    ob_mask = (vals < 30) & mask
    factor[ob_mask] = (30 - vals[ob_mask]) / 30
    os_mask = (vals > 70) & mask
    factor[os_mask] = (70 - vals[os_mask]) / 30
    new_factors[name] = factor

# B: 多RSI共振
rsi_matrix = np.column_stack([panel[c].values for c in rsi_cols])
mask_all = ~np.any(np.isnan(rsi_matrix), axis=1)

rsi_mean = np.mean(rsi_matrix, axis=1)  # 平均RSI值
factor_resonance = np.zeros(len(panel))

# RSI < 35 超卖→正信号, RSI > 65 超买→负信号, 线性插值
for i in range(len(panel)):
    if np.isnan(rsi_mean[i]):
        continue
    if rsi_mean[i] < 35:
        factor_resonance[i] = (35 - rsi_mean[i]) / 35 * (1 + (mask_all[i] * 0.2))
    elif rsi_mean[i] > 65:
        factor_resonance[i] = (65 - rsi_mean[i]) / 35 * (1 + (mask_all[i] * 0.2))

new_factors["RSI_共振净信号"] = factor_resonance

# 共振超卖（纯超卖，不包含超买）
resonance_os = np.zeros(len(panel))
os_resonance = np.all(rsi_matrix < 35, axis=1) & mask_all
resonance_os[os_resonance] = (35 - np.mean(rsi_matrix[os_resonance], axis=1)) / 35
new_factors["RSI_共振超卖"] = resonance_os

# C: 背离
for col in rsi_cols:
    rsi_max = panel.groupby("ts_code")[col].transform(lambda x: x.rolling(20, min_periods=5).max())
    rsi_min = panel.groupby("ts_code")[col].transform(lambda x: x.rolling(20, min_periods=5).min())
    
    is_rsi_high = panel[col] >= rsi_max * 0.98
    is_rsi_low = panel[col] <= rsi_min * 1.02
    
    new_factors[f"{col}_顶背离"] = np.where(is_rsi_high, -1.0, 0.0)
    new_factors[f"{col}_底背离"] = np.where(is_rsi_low, 1.0, 0.0)

# 加到panel
for name, vals in new_factors.items():
    panel[name] = vals

print(f"  新增因子: {len(new_factors)}个")
for name in new_factors:
    print(f"    {name:25s} 均值={np.mean(new_factors[name]):.4f}")

# 3. 相关性分析
print(f"\n[3/5] 因子相关性分析...")
# 选最近3年数据
recent = panel[panel["trade_date"] >= "2023-01-01"].copy()
print(f"  近3年: {len(recent):,}行")

# 所有因子列
factor_cols = [c for c in panel.columns if c not in ['ts_code','trade_date','name','close','volume'] 
               and not c.startswith('fwd')]
all_factors = factor_cols + list(new_factors.keys())

# 相关性矩阵（采样日频截面）
corr_data = recent[all_factors].dropna(how='all')
corr_matrix = corr_data.corr(method='spearman')

# RSI新因子与已有因子的最大相关
print(f"\n  RSI新因子与现有因子的最大相关性:")
for nf in new_factors:
    if nf not in corr_matrix.columns:
        continue
    cols = [c for c in corr_matrix.columns if c != nf and c not in new_factors]
    if cols:
        max_corr = corr_matrix.loc[nf, cols].abs().max()
        max_with = corr_matrix.loc[nf, cols].abs().idxmax()
        print(f"    {nf:25s}: 最大ρ={max_corr:.3f} (与{max_with})")

# 4. v12合成测试
print(f"\n[4/5] v12合成 + RSI新因子测试...")

# v12原有权重（归一化后）
v12_weights = {
    "60日动量": 0.127, "20日动量": 0.120, "市值": 0.117,
    "EMA20偏离": 0.059, "120日动量": 0.058, "换手率": 0.050,
    "波动率": 0.049, "EMA5偏离": 0.049, "RSI_24": 0.046,
    "MACD": 0.044, "OBV": 0.041, "BOLL位置": 0.041,
    "RSI_12": 0.039, "RSI_6": 0.039, "量能趋势": 0.038,
    "EMA10偏离": 0.038,
}

# v12方向
v12_direction = {k: -1 for k in v12_weights}

# 测试不同的RSI新因子替换方案
test_configs = [
    ("v12 + RSI_共振净信号替代原始RSI", {
        **{k: v for k, v in v12_weights.items() if not k.startswith("RSI")},
        "RSI_共振净信号": 0.05,
    }),
    ("v12 + RSI_6_反转强度替代RSI_6", {
        **{k: v for k, v in v12_weights.items() if k != "RSI_6"},
        "RSI_6_反转强度": 0.04,
    }),
    ("v12 + 全部RSI替换为反转强度", {
        **{k: v for k, v in v12_weights.items() if not k.startswith("RSI")},
        "RSI_6_反转强度": 0.025,
        "RSI_12_反转强度": 0.025,
        "RSI_24_反转强度": 0.025,
    }),
    ("v12 + 共振信号替换全部RSI", {
        **{k: v for k, v in v12_weights.items() if not k.startswith("RSI")},
        "RSI_共振净信号": 0.08,
        "RSI_共振超卖": 0.04,
    }),
]

for cfg_name, weights in test_configs:
    print(f"\n  📊 {cfg_name}")
    
    # 月频IC测试
    panel["_ym"] = panel["trade_date"].dt.to_period("M")
    ics = []
    for ym, g in panel.groupby("_ym"):
        g = g.dropna(subset=["fwd_20d_ret"])
        if len(g) < 100:
            continue
        
        score = np.zeros(len(g))
        for fn, w in weights.items():
            if fn not in g.columns:
                continue
            direction = v12_direction.get(fn, -1)
            vals = g[fn].values
            mask = ~np.isnan(vals)
            if mask.sum() < 10:
                continue
            std = np.nanstd(vals[mask])
            if std < 1e-10:
                continue
            normed = (vals - np.nanmean(vals[mask])) / std
            normed[~mask] = 0
            score += normed * direction * w
        
        valid = ~np.isnan(score)
        if valid.sum() < 50:
            continue
        r = spearmanr(score[valid], g["fwd_20d_ret"].values[valid])[0]
        if not np.isnan(r):
            ics.append(r)
    
    if ics:
        ic_mean = float(np.mean(ics))
        ic_std = float(np.std(ics))
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0
        print(f"    IC均值={ic_mean*100:+.2f}%  IC_IR={ic_ir:.2f}  ({len(ics)}个月)")
        
        # 对比原始v12
        if "原始" not in cfg_name:
            # 简单对比：IR提升百分比
            if abs(ic_ir) > 0:
                print(f"    {'✅' if ic_ir > 0.8 else '⚠️'} IR {'>' if abs(ic_ir) > 0.65 else '<'} 0.65")
    else:
        print(f"    无有效IC数据")

panel.drop(columns=["_ym"], inplace=True, errors="ignore")

# 5. 保存面板v6
print(f"\n[5/5] 保存面板v6...")
v6_path = os.path.join(DATA_FACTORS, "factor_panel_v6.parquet")
panel.to_parquet(v6_path)
print(f"  保存: {v6_path}")
print(f"  总列数: {len(panel.columns)}")
print(f"  因子列: {len([c for c in panel.columns if c not in ['ts_code','trade_date','name','close','volume'] and not c.startswith('fwd')])}")

print(f"\n{'='*60}")
print(f"✅ RSI因子集成完成")
print(f"{'='*60}")
