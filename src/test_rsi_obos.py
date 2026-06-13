"""
RSI超买超卖因子构建与IC测试
测试不同RSI超买超卖编码方式的选股效果

子因子：
  A: RSI超买超卖强度 — >80时负值(反转)，<30时正值(反弹)
  B: RSI极值共振 — RSI_6/12/24同时<30或>80时的信号强度
  C: RSI斜率 — RSI的变化速度（加速超买=危险，加速超卖=机会）
  D: RSI背离 — 价格创新高但RSI不创新高
"""
import sys, os
from datetime import datetime
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config.settings import DATA_FACTORS

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT)

print(f"\n📊 RSI超买超卖因子测试 — {datetime.now().strftime('%F %H:%M')}")
print("=" * 50)

# 1. 加载面板
print("[1/4] 加载因子面板...")
panel = pd.read_parquet(os.path.join(DATA_FACTORS, "factor_panel_with_fwd_v2.parquet"))
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
print(f"  面板: {len(panel):,}行, {panel['trade_date'].nunique()}个交易日")

# 2. 构建RSI超买超卖因子
print("\n[2/4] 构建RSI衍生因子...")

# 基础RSI列
rsi_cols = [c for c in panel.columns if c.startswith("RSI")]
print(f"  已有RSI: {rsi_cols}")

codes_rsi = []  # 记录因子代码

# A: 超买超卖强度 (单RSI)
for col in rsi_cols:
    name = f"{col}_反转强度"
    vals = panel[col].values.copy()
    mask = ~np.isnan(vals)
    
    # RSI > 70 超买 → 负向（预期下跌）
    # RSI < 30 超卖 → 正向（预期上涨）
    # 中间区域线性插值到0
    factor = np.zeros(len(vals))
    
    # 超卖区域: RSI < 30 → 正向 (30-RSI)/30
    ob_mask = (vals < 30) & mask
    factor[ob_mask] = (30 - vals[ob_mask]) / 30
    
    # 超买区域: RSI > 70 → 负向 (70-RSI)/30
    os_mask = (vals > 70) & mask
    factor[os_mask] = (70 - vals[os_mask]) / 30  # 负值
    
    panel[name] = factor
    codes_rsi.append(name)

# B: 多RSI共振
rsi_list = [panel[c].values for c in rsi_cols]
rsi_matrix = np.column_stack(rsi_list)
mask_all = ~np.any(np.isnan(rsi_matrix), axis=1)

# 共振超卖: 所有RSI都<30
resonance_os = np.zeros(len(panel))
os_resonance = np.all(rsi_matrix < 30, axis=1) & mask_all
resonance_os[os_resonance] = (30 - np.mean(rsi_matrix[os_resonance], axis=1)) / 30

# 共振超买: 所有RSI都>70  
resonance_ob = np.zeros(len(panel))
ob_resonance = np.all(rsi_matrix > 70, axis=1) & mask_all
resonance_ob[ob_resonance] = (70 - np.mean(rsi_matrix[ob_resonance], axis=1)) / 30  # 负值

panel["RSI_共振超卖"] = resonance_os
panel["RSI_共振超买"] = resonance_ob
panel["RSI_共振净信号"] = resonance_os + resonance_ob
codes_rsi += ["RSI_共振超卖", "RSI_共振超买", "RSI_共振净信号"]

# C: RSI斜率（变化速度）
for col in rsi_cols:
    name = f"{col}_斜率"
    panel[name] = panel.groupby("ts_code")[col].diff(1)
    codes_rsi.append(name)
    
    # 加速版本：斜率的导数（二阶差分）
    name2 = f"{col}_加速度"
    panel[name2] = panel.groupby("ts_code")[name].diff(1)
    codes_rsi.append(name2)

# D: RSI背离（价格新高但RSI未新高 → 顶背离看跌）
# 使用20日窗口
for col in rsi_cols:
    name_d = f"{col}_顶背离"
    name_b = f"{col}_底背离"
    
    # 滚动20日最大值
    rsi_max = panel.groupby("ts_code")[col].transform(lambda x: x.rolling(20, min_periods=5).max())
    rsi_min = panel.groupby("ts_code")[col].transform(lambda x: x.rolling(20, min_periods=5).min())
    
    # 当前日RSI = 20日最高 → 超买状态
    is_rsi_high = panel[col] >= rsi_max * 0.98
    # 当前日RSI = 20日最低 → 超卖状态
    is_rsi_low = panel[col] <= rsi_min * 1.02
    
    # 顶背离信号: RSI正在从高位回落，但价格还在涨
    panel[name_d] = np.where(is_rsi_high, -panel[col] * 0.01, 0)
    panel[name_b] = np.where(is_rsi_low, panel[col] * 0.01, 0)
    codes_rsi += [name_d, name_b]

# 去掉无限值
panel = panel.replace([np.inf, -np.inf], np.nan)

print(f"  构建因子: {len(codes_rsi)}个")
for name in codes_rsi:
    if name in panel.columns:
        print(f"    {name:30s} 存在={panel[name].notna().sum():>8,} 样本")

# 3. IC测试
print(f"\n[3/4] IC测试 (与未来20日收益)...")

# 找到未来收益列
fwd_cols = [c for c in panel.columns if c.startswith("fwd")]
print(f"  未来收益列: {fwd_cols}")

results = []
for factor in codes_rsi:
    if factor not in panel.columns:
        continue
    if panel[factor].nunique() < 10:
        print(f"  ⚠️ {factor}: 仅有{panel[factor].nunique()}个唯一值，跳过")
        continue
    
    for fwd in fwd_cols:
        if fwd not in panel.columns:
            continue
        
        valid = panel[[factor, fwd]].dropna()
        if len(valid) < 1000:
            continue
        
        # 截面IC（按月）
        panel["_tmp_date"] = panel["trade_date"].dt.to_period("M")
        ics = []
        for period, g in panel.groupby("_tmp_date"):
            gv = g[[factor, fwd]].dropna()
            if len(gv) < 50:
                continue
            r = gv[factor].corr(gv[fwd], method="spearman")
            if not np.isnan(r):
                ics.append(r)
        panel.drop(columns=["_tmp_date"], inplace=True)
        
        if len(ics) > 5:
            ic_mean = np.mean(ics)
            ic_std = np.std(ics)
            ic_ir = ic_mean / ic_std if ic_std > 0 else 0
            results.append({
                "factor": factor,
                "fwd": fwd,
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "ic_ir": ic_ir,
                "n_months": len(ics),
            })

# 4. 输出结果
print(f"\n[4/4] 结果分析...")
results.sort(key=lambda x: -abs(x["ic_ir"]))

print(f"\n{'='*80}")
print(f"RSI超买超卖因子IC测试结果")
print(f"{'='*80}")
print(f"{'因子':30s} {'未来收益':10s} {'IC均值':>8s} {'IC_IR':>8s} {'月数':>5s}")
print(f"{'—'*30} {'—'*10} {'—'*8} {'—'*8} {'—'*5}")
for r in results[:30]:
    print(f"{r['factor']:30s} {r['fwd']:10s} {r['ic_mean']*100:7.2f}% {r['ic_ir']:7.2f} {r['n_months']:5d}")

# 汇总最优
print(f"\n{'='*80}")
print(f"最佳子因子 TOP5")
print(f"{'='*80}")
best_ob = [r for r in results if abs(r["ic_mean"]) > 0.01]
best_ob.sort(key=lambda x: -abs(x["ic_ir"]))
for r in best_ob[:10]:
    direction = "正(超卖→涨)" if r["ic_mean"] > 0 else "负(超买→跌)"
    print(f"  {r['factor']:30s} {direction:12s} IC={r['ic_mean']*100:+.2f}% IR={r['ic_ir']:.2f} ({r['n_months']}个月)")

# 与现有RSI因子对比
print(f"\n{'='*80}")
print(f"对比现有RSI因子")
print(f"{'='*80}")
existing_rsi = [r for r in results if any(c in r['factor'] for c in ['RSI_','RSI_6','RSI_12','RSI_24'])]
existing_rsi.sort(key=lambda x: -abs(x["ic_ir"]))
for r in existing_rsi[:15]:
    print(f"  {r['factor']:30s} IC={r['ic_mean']*100:+.2f}% IR={r['ic_ir']:.2f}")

# 保存结果
import json
output = {
    "test_time": datetime.now().isoformat(),
    "n_factors_tested": len(results),
    "results": [{k: round(v, 6) if isinstance(v, float) else v for k, v in r.items()} 
                for r in results],
}
with open("alerts/rsi_obos_test.json", "w") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"\n  结果已保存: alerts/rsi_obos_test.json")
