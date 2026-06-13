"""
快速测试：RSI新因子与现有因子相关性 + v12合成回测
使用采样方式降低计算量
"""
import sys, os, json
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config.settings import DATA_FACTORS

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT)

print(f"\n🚀 RSI新因子快速合成测试 — {datetime.now().strftime('%F %H:%M')}")
print("=" * 60)

# 1. 加载面板
print("[1/4] 加载面板...")
panel = pd.read_parquet(os.path.join(DATA_FACTORS, "factor_panel_v5.parquet"))
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
print(f"  面板: {len(panel):,}行")

# 2. 构建RSI新因子（只构建测试所需的，不存到面板）
print("\n[2/4] 快速构建RSI因子 + 相关性...")
rsi_cols = ["RSI_6", "RSI_12", "RSI_24"]

# 只取最近3年数据做采样
recent = panel[panel["trade_date"] >= "2023-01-01"].copy()
# 每月随机采样200只股票（加速）
sampled = recent.groupby(recent["trade_date"].dt.to_period("M")).apply(
    lambda g: g.sample(min(200, len(g)), random_state=42)
).reset_index(drop=True)
print(f"  采样: {len(sampled):,}行")

rsi_vals = sampled[rsi_cols].values

# 构建RSI_6_反转强度
factor_rsi6_reversal = np.zeros(len(sampled))
mask6 = ~np.isnan(rsi_vals[:, 0])
ob6 = rsi_vals[:, 0] < 30
os6 = rsi_vals[:, 0] > 70
factor_rsi6_reversal[ob6 & mask6] = (30 - rsi_vals[ob6 & mask6, 0]) / 30
factor_rsi6_reversal[os6 & mask6] = (70 - rsi_vals[os6 & mask6, 0]) / 30

# 构建RSI_共振净信号
rsi_mean = np.nanmean(rsi_vals, axis=1)
factor_resonance = np.zeros(len(sampled))
mask_r = ~np.isnan(rsi_mean)
factor_resonance[mask_r & (rsi_mean < 35)] = (35 - rsi_mean[mask_r & (rsi_mean < 35)]) / 35
factor_resonance[mask_r & (rsi_mean > 65)] = (65 - rsi_mean[mask_r & (rsi_mean > 65)]) / 35

# 相关性
existing_factor_cols = ['短期反转','20日动量','60日动量','120日动量','波动率','换手率',
                        '量能趋势','市值','EMA5偏离','EMA10偏离','EMA20偏离',
                        'MACD','BOLL位置','OBV','量价背离信号','高波反转']
existing = {c: sampled[c].values for c in existing_factor_cols if c in sampled.columns}

print(f"\n  因子相关性 (Spearman ρ):")
print(f"  {'新因子':25s} {'现有因子':15s} {'ρ':>8s}")
print(f"  {'-'*25} {'-'*15} {'-'*8}")

for new_name, new_vals in [("RSI_6_反转强度", factor_rsi6_reversal), ("RSI_共振净信号", factor_resonance)]:
    valid_new = ~np.isnan(new_vals)
    for old_name, old_vals in existing.items():
        valid_old = ~np.isnan(old_vals)
        valid = valid_new & valid_old
        if valid.sum() > 100:
            r, _ = spearmanr(new_vals[valid], old_vals[valid])
            print(f"  {new_name:25s} {old_name:15s} {r:8.3f}")
    print()

# 3. 合成因子回测
print("\n[3/4] 合成因子IC对比...")

# v12权重
v12_weights = {
    "60日动量": 0.127, "20日动量": 0.120, "市值": 0.117,
    "EMA20偏离": 0.059, "120日动量": 0.058, "换手率": 0.050,
    "波动率": 0.049, "EMA5偏离": 0.049, "RSI_24": 0.046,
    "MACD": 0.044, "OBV": 0.041, "BOLL位置": 0.041,
    "RSI_12": 0.039, "RSI_6": 0.039, "量能趋势": 0.038,
    "EMA10偏离": 0.038,
}

# 月频IC计算函数
def calc_composite_ic(panel, weights, direction=-1):
    panel_data = panel.copy()
    panel_data["_ym"] = panel_data["trade_date"].dt.to_period("M")
    ics = []
    for ym, g in panel_data.groupby("_ym"):
        g = g.dropna(subset=["fwd_20d_ret"])
        if len(g) < 100:
            continue
        score = np.zeros(len(g))
        for fn, w in weights.items():
            if fn not in g.columns:
                continue
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
        r, _ = spearmanr(score[valid], g["fwd_20d_ret"].values[valid])
        if not np.isnan(r):
            ics.append(r)
    return ics

# 测试各种组合
configs = [
    ("v12原始", v12_weights),
    ("v12 + RSI_6_反转强度(取代RSI_6)", {
        **{k: v for k, v in v12_weights.items() if k != "RSI_6"},
        "RSI_6_反转强度": 0.04,
    }),
    ("v12 + RSI_共振净信号(取代全部RSI)", {
        **{k: v for k, v in v12_weights.items() if not k.startswith("RSI")},
        "RSI_共振净信号": 0.08,
    }),
    ("v12 + 共振信号(取代全部RSI) + 量价背离", {
        **{k: v for k, v in v12_weights.items() if not k.startswith("RSI")},
        "RSI_共振净信号": 0.06,
        "量价背离信号": 0.04,
    }),
    ("v12 + RSI反转 + 共振 + 量价背离", {
        **{k: v for k, v in v12_weights.items() if not k.startswith("RSI")},
        "RSI_6_反转强度": 0.03,
        "RSI_12_反转强度": 0.02,
        "RSI_共振净信号": 0.04,
        "量价背离信号": 0.03,
    }),
]

# 在RSI因子列里构建（全量）
print("  构建RSI因子（全量面板）...")
# RSI_6_反转强度
vals_rsi6 = panel["RSI_6"].values
f_rsi6_rev = np.zeros(len(panel))
mask = ~np.isnan(vals_rsi6)
f_rsi6_rev[(vals_rsi6 < 30) & mask] = (30 - vals_rsi6[(vals_rsi6 < 30) & mask]) / 30
f_rsi6_rev[(vals_rsi6 > 70) & mask] = (70 - vals_rsi6[(vals_rsi6 > 70) & mask]) / 30
panel["RSI_6_反转强度"] = f_rsi6_rev

# RSI_12_反转强度
vals_rsi12 = panel["RSI_12"].values
f_rsi12_rev = np.zeros(len(panel))
mask = ~np.isnan(vals_rsi12)
f_rsi12_rev[(vals_rsi12 < 30) & mask] = (30 - vals_rsi12[(vals_rsi12 < 30) & mask]) / 30
f_rsi12_rev[(vals_rsi12 > 70) & mask] = (70 - vals_rsi12[(vals_rsi12 > 70) & mask]) / 30
panel["RSI_12_反转强度"] = f_rsi12_rev

# 共振净信号
rsi_all = panel[["RSI_6","RSI_12","RSI_24"]].values
rsi_m = np.nanmean(rsi_all, axis=1)
f_res = np.zeros(len(panel))
m = ~np.isnan(rsi_m)
f_res[m & (rsi_m < 35)] = (35 - rsi_m[m & (rsi_m < 35)]) / 35
f_res[m & (rsi_m > 65)] = (65 - rsi_m[m & (rsi_m > 65)]) / 35
panel["RSI_共振净信号"] = f_res

panel.drop(columns=["_ym"], inplace=True, errors="ignore")

# 跑每个配置
results = []
for name, weights in configs:
    print(f"\n  📊 {name}")
    ics = calc_composite_ic(panel, weights, direction=-1)
    if ics:
        ic_mean = float(np.mean(ics))
        ic_std = float(np.std(ics))
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0
        print(f"    IC均值={ic_mean*100:+.2f}%  IC_IR={ic_ir:.2f}  ({len(ics)}个月)")
        results.append({"config": name, "ic_mean": ic_mean, "ic_ir": ic_ir, "n_months": len(ics)})

# 打印对比
print(f"\n{'='*60}")
print(f"  合成因子IC对比总结")
print(f"{'='*60}")
print(f"  {'配置':40s} {'IC均值':>8s} {'IC_IR':>8s} {'月数':>5s}")
print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*5}")
for r in results:
    print(f"  {r['config']:40s} {r['ic_mean']*100:+7.2f}% {r['ic_ir']:7.2f} {r['n_months']:5d}")

# 保存结果
rsi_dir = "data/rsi_new"
os.makedirs(rsi_dir, exist_ok=True)
with open(os.path.join(rsi_dir, "synthesis_results.json"), "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# 也保存面板v6（含RSI新因子）
print(f"\n[4/4] 保存面板v6...")
panel.to_parquet(os.path.join(DATA_FACTORS, "factor_panel_v6.parquet"))
print(f"  ✅ 已保存: factor_panel_v6.parquet ({len(panel.columns)}列)")

print(f"\n{'='*60}")
print(f"✅ 测试完成")
print(f"{'='*60}")
