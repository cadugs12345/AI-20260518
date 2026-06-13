"""
相关性分析 + 合成测试
选今日新增因子中真正的增量因子
"""
import sys, os, json
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT)
import subprocess
subprocess.run(['bash', '-c', 'cd /mnt/d/AI-20260518 && source .venv/bin/activate && which python'], capture_output=True)

# 用当前python（已在venv中运行）
sys.path.insert(0, PROJECT)
from config.settings import DATA_FACTORS

print(f"\n🧬 因子相关性分析 + 合成测试 — {datetime.now().strftime('%F %H:%M')}")
print("=" * 60)

# 1. 加载面板
print("[1/4] 加载面板v6...")
panel = pd.read_parquet(os.path.join(DATA_FACTORS, "factor_panel_v6.parquet"))
factor_cols = [c for c in panel.columns if c not in ['ts_code','trade_date','name','close','volume','fwd_20d_ret']]
print(f"  总因子数: {len(factor_cols)}")

# 2. 每个因子单独IC
print("\n[2/4] 全因子IC扫描...")
panel["_ym"] = panel["trade_date"].dt.to_period("M")

ic_all = {}
for factor in factor_cols:
    ics = []
    for ym, g in panel.groupby("_ym"):
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
        ic_all[factor] = {"ic": ic_mean, "ir": ic_ir}

# 排序
sorted_factors = sorted(ic_all.items(), key=lambda x: -abs(x[1]["ir"]))
print(f"  IC_IR 排名前15:")
print(f"  {'因子':30s} {'IC':>8s} {'IR':>8s}")
print(f"  {'-'*30} {'-'*8} {'-'*8}")
for f, v in sorted_factors[:15]:
    print(f"  {f:30s} {v['ic']*100:+7.2f}% {v['ir']:7.2f}")

# 3. 相关性矩阵（核心因子vs新因子）
print(f"\n[3/4] 相关性分析...")
# 原v12核心因子
core_factors = ["短期反转", "20日动量", "60日动量", "120日动量", "波动率", 
                "换手率", "量比", "量能趋势", "BP", "EP", "MACD", "BOLL位置",
                "EMA5偏离", "EMA10偏离", "EMA20偏离"]

# 今日新增候选因子（IR>0.5的）
new_candidates = [f for f, v in ic_all.items() if abs(v["ir"]) > 0.5 
                  and f not in core_factors
                  and f not in ["_ym"]]

print(f"  原核心: {len(core_factors)}个")
print(f"  候选新因子(IR>0.5): {len(new_candidates)}个")

# 采样计算相关性（全量太大，抽样20%）
panel_sample = panel.sample(frac=0.2, random_state=42)
all_test = core_factors + [c for c in new_candidates if c in panel_sample.columns]
test_data = panel_sample[all_test].dropna()
corr = test_data.corr(method="spearman")

# 对新因子，找与核心因子最大相关性
print(f"\n  新因子与核心因子最大相关性:")
print(f"  {'新因子':30s} {'IR':>6s} {'最大|ρ|':>8s} {'与谁相关':>30s}")
print(f"  {'-'*30} {'-'*6} {'-'*8} {'-'*30}")

incremental_candidates = []
for nf in new_candidates:
    if nf not in corr.columns:
        continue
    max_corr = 0
    max_with = ""
    for cf in core_factors:
        if cf in corr.index:
            c = abs(corr.loc[cf, nf])
            if c > max_corr:
                max_corr = c
                max_with = cf
    ir = ic_all.get(nf, {}).get("ir", 0)
    print(f"  {nf:30s} {ir:6.2f} {max_corr:7.3f}  {max_with:30s}")
    
    if max_corr < 0.5 or ir > 1.5:
        # ρ<0.5（独立性好）或 IR极高（即使相关也可能有增量）
        incremental_candidates.append(nf)

print(f"\n  增量候选因子 ({len(incremental_candidates)}个):")
for f in incremental_candidates:
    print(f"    {f:30s} IR={ic_all[f]['ir']:.2f}")

# 4. 合成测试
print(f"\n[4/4] v12合成测试...")

# 加载v12权重（从原始权重或从现有面板反推）
weights_path = "config/v12_weights.json"
if os.path.exists(weights_path):
    with open(weights_path) as f:
        weights = json.load(f)
    print(f"  加载v12权重: {weights}")
else:
    # 用IC_IR作为权重近似
    weights = {f: v["ir"] for f, v in sorted_factors if abs(v["ir"]) > 0.3}
    weights = {k: v for k, v in list(weights.items())[:22]}
    print(f"  用IC_IR权重近似: {len(weights)}个因子")

# 合成原始v12
base_factors = list(weights.keys())
available = [f for f in base_factors if f in panel.columns]

# 等权合成合成测试
def calc_synthetic_ir(panel_df, factors):
    """计算因子合成的截面IC_IR"""
    if len(factors) < 3:
        return 0, 0, 0
    # 去量纲等权合成
    panel_df = panel_df.copy()
    valid_f = [f for f in factors if f in panel_df.columns]
    if len(valid_f) < 3:
        return 0, 0, 0
    
    zdf = panel_df[valid_f].rank(pct=True)
    panel_df["zs"] = zdf.mean(axis=1)
    
    ics = []
    for ym, g in panel_df.groupby("_ym"):
        gv = g[["zs", "fwd_20d_ret"]].dropna()
        if len(gv) < 50:
            continue
        r, _ = spearmanr(gv["zs"], gv["fwd_20d_ret"])
        if not np.isnan(r):
            ics.append(r)
    
    if not ics:
        return 0, 0, 0
    ic_mean = float(np.mean(ics))
    ic_std = float(np.std(ics))
    ir = ic_mean / ic_std if ic_std > 0 else 0
    return ic_mean, ic_std, ir

# 基线
panel_ym = panel.copy()
panel_ym["_ym"] = panel_ym["trade_date"].dt.to_period("M")

print(f"\n  ┌{'='*50}┐")
print(f"  │ 合成IC测试（等权截面合成）")
print(f"  ├{'='*50}┤")

# 基线: 原核心因子
ic_base, _, ir_base = calc_synthetic_ir(panel_ym, core_factors)
print(f"  │ 原核心因子 ({len(core_factors)}个): IC={ic_base*100:.2f}% IR={ir_base:.2f}")

# 加入增量候选
if incremental_candidates:
    combined = core_factors + incremental_candidates[:10]
    ic_new, _, ir_new = calc_synthetic_ir(panel_ym, combined)
    print(f"  │ +新候选({len(incremental_candidates[:10])}个): IC={ic_new*100:.2f}% IR={ir_new:.2f}")
    print(f"  │ 提升: IR {ir_base:.2f}→{ir_new:.2f} ({(ir_new/ir_base-1)*100:+.1f}%)")

    # 最佳子集测试
    best_ir = ir_base
    best_set = core_factors.copy()
    for nf in incremental_candidates[:10]:
        test_set = core_factors + [nf]
        _, _, test_ir = calc_synthetic_ir(panel_ym, test_set)
        if test_ir > best_ir + 0.02:
            best_ir = test_ir
            best_set = test_set
            print(f"  │ ✅ {nf:30s} 提升 IR→{test_ir:.2f}")

print(f"  └{'='*50}┘")

# 保存增量候选清单
with open("config/incremental_factors.json", "w") as f:
    json.dump({
        "incremental_candidates": incremental_candidates,
        "ic_results": {k: v for k, v in ic_all.items() if k in incremental_candidates},
        "best_core": best_set if 'best_set' in dir() else core_factors,
    }, f, indent=2, default=str)

print(f"\n✅ 分析完成")
print(f"  → 增量候选已保存: config/incremental_factors.json")
