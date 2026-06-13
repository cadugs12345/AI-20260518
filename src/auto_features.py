"""
auto_features_v2.py — 自动衍生因子 v2
修复内存问题：list收集后一次性concat
"""
import os, sys, time, json
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
os.chdir("/mnt/d/AI-20260518"); sys.path.insert(0, '.')
import joblib, lightgbm as lgb
from scipy.stats import spearmanr
tt = time.time()

print("="*60)
print("Auto-Features v2 自动衍生因子")
print(f"{time.strftime('%F %H:%M')}")
print("="*60)

panel = pd.read_parquet("data/factors/factor_panel_v6.parquet")
factor_cols_raw = [c for c in panel.columns 
               if c not in ["ts_code","trade_date","fwd_20d_ret","close","volume","ret_1d",
                   "__fragment_index","__batch_index","__last_in_fragment","__filename","board_break"]]
print(f"基础因子: {len(factor_cols_raw)}")

# ==== 1. 低相关因子对 ====
print("\n[1] 筛选低相关因子对...", flush=True)
panel_small = panel.sample(200000, random_state=42)
corr_mat = panel_small[factor_cols_raw].corr(method="spearman")

low_corr_pairs = []
for i in range(len(factor_cols_raw)):
    for j in range(i+1, len(factor_cols_raw)):
        rho = abs(corr_mat.iloc[i, j])
        if rho < 0.3:
            low_corr_pairs.append((factor_cols_raw[i], factor_cols_raw[j], rho))

low_corr_pairs.sort(key=lambda x: x[2])
# 前200对（800衍生因子）就够了
n_pairs = min(200, len(low_corr_pairs))
selected_pairs = low_corr_pairs[:n_pairs]
print(f"  选做衍生的: {n_pairs}对 → {n_pairs*4}衍生因子")

# ==== 2. 生成衍生因子（一次性concat）====
print("\n[2] 生成衍生因子...", flush=True)

# 预计算所有rank
ranks = {}
for fc in factor_cols_raw:
    ranks[fc] = panel[fc].rank(pct=True).values.astype(np.float32)
print("  rank预计算完成", flush=True)

# 生成列
new_series = []
new_names = []
for fi, fj, _ in selected_pairs:
    sa, sb = ranks[fi], ranks[fj]
    n_base_i = factor_cols_raw.index(fi)
    n_base_j = factor_cols_raw.index(fj)
    prefix = f"af_r{n_base_i}-r{n_base_j}"
    
    new_series.append(pd.Series(sa - sb, dtype=np.float32))
    new_names.append(f"{prefix}_diff")
    
    new_series.append(pd.Series(sa + sb, dtype=np.float32))
    new_names.append(f"{prefix}_sum")
    
    div = np.divide(sa, sb + 0.001)
    new_series.append(pd.Series(div, dtype=np.float32))
    new_names.append(f"{prefix}_div")
    
    new_series.append(pd.Series(sa * sb, dtype=np.float32))
    new_names.append(f"{prefix}_mul")
    
    if (len(new_names)) % 200 == 0:
        print(f"  {len(new_names)}衍生因子计算完成", flush=True)

print(f"  总衍生因子: {len(new_names)}")

# 一次性concat
print("  拼接衍生因子到面板...", flush=True)
derived_panel = pd.concat(new_series, axis=1)
derived_panel.columns = new_names

# 加到panel
panel = pd.concat([panel, derived_panel], axis=1)
del derived_panel, new_series, ranks

all_new_cols = new_names
all_factors = factor_cols_raw + all_new_cols
print(f"  面板: {panel.shape}, 总特征: {len(all_factors)}")

# ==== 3. LGB训练筛选 ====
print(f"\n[3] LGB特征重要性筛选...", flush=True)

train = panel[panel["trade_date"] >= pd.Timestamp("2024-01-01")]
train = train[train["fwd_20d_ret"].notna() & (train["fwd_20d_ret"].abs() < 0.5)]
if len(train) > 200000:
    train = train.sample(200000, random_state=42)

X = train[all_factors].fillna(0).values.astype(np.float32)
y = np.clip(train["fwd_20d_ret"].values.astype(np.float32), -0.3, 0.3)

nv = max(1, int(len(train) * 0.15))
lgb_fi = lgb.LGBMRegressor(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.5,
    reg_alpha=0.5, reg_lambda=1.0,
    min_child_weight=20, min_data_in_leaf=100,
    random_state=42, verbose=-1, n_jobs=8
)
lgb_fi.fit(X[:-nv], y[:-nv], eval_set=[(X[-nv:], y[-nv:])],
           callbacks=[lgb.early_stopping(20, verbose=False)], eval_metric="mse")

imp = pd.DataFrame({"factor": all_factors, "importance": lgb_fi.feature_importances_})
imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)

imp_base = imp[imp["factor"].isin(factor_cols_raw)]
imp_auto = imp[~imp["factor"].isin(factor_cols_raw)]

median_imp = imp_auto["importance"].median()
top_auto = imp_auto[imp_auto["importance"] >= median_imp]
print(f"\n  基础因子活跃: {len(imp_base[imp_base['importance']>0])}")
print(f"  衍生因子活跃: {len(imp_auto[imp_auto['importance']>0])}")
print(f"  Top衍生(>中位): {len(top_auto)}")

# Top30%基础 + 衍生
n_base = max(1, int(len(factor_cols_raw) * 0.3))
top_base = imp_base.head(n_base)["factor"].tolist()
top_auto_list = top_auto.head(80)["factor"].tolist() if len(top_auto) > 0 else []

selected_features = list(dict.fromkeys(top_base + top_auto_list))
print(f"  ✅ 最终特征集: {len(selected_features)} ({n_base}基础+{len(top_auto_list)}衍生)")

json.dump({"selected": selected_features, "n_base": n_base, "n_auto": len(top_auto_list),
           "all_base": factor_cols_raw, "all_auto": all_new_cols}, 
          open("models/auto_features_v1.json","w"), indent=2)

print(f"\n  Top20特征重要性:")
for i, (_, row) in enumerate(imp.head(20).iterrows()):
    tag = " 🆕" if row["factor"] in all_new_cols else ""
    print(f"    {i+1}. {row['factor'][:70]:70s} {row['importance']:>5d}{tag}")

print(f"\n  ✅ models/auto_features_v1.json")
print(f"  ⏱ {(time.time()-tt)/60:.1f}分")
