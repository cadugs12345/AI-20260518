"""
生成auto_features_v1.parquet — 800衍生因子
在tmux里跑，避免超时
"""
import os, json, time
import pandas as pd, numpy as np
import warnings; warnings.filterwarnings("ignore")

os.chdir("/mnt/d/AI-20260518")
tt = time.time()

print("生成auto_features_v1.parquet")
print("="*50, flush=True)

panel = pd.read_parquet("data/factors/factor_panel_v6.parquet")
factor_cols_raw = [c for c in panel.columns 
               if c not in ["ts_code","trade_date","fwd_20d_ret","close","volume","ret_1d",
                   "__fragment_index","__batch_index","__last_in_fragment","__filename","board_break"]]
print(f"基础因子: {len(factor_cols_raw)}", flush=True)

# 低相关因子对
panel_small = panel.sample(200000, random_state=42)
corr_mat = panel_small[factor_cols_raw].corr(method="spearman")
print("相关性矩阵完成", flush=True)

low_corr_pairs = []
for i in range(len(factor_cols_raw)):
    for j in range(i+1, len(factor_cols_raw)):
        if abs(corr_mat.iloc[i, j]) < 0.3:
            low_corr_pairs.append((factor_cols_raw[i], factor_cols_raw[j]))
print(f"低相关对: {len(low_corr_pairs)}", flush=True)

n_pairs = min(200, len(low_corr_pairs))
selected_pairs = low_corr_pairs[:n_pairs]

# 衍生
new_series, new_names = [], []
for idx, (fi, fj) in enumerate(selected_pairs):
    sa = panel[fi].rank(pct=True).values.astype(np.float32)
    sb = panel[fj].rank(pct=True).values.astype(np.float32)
    pre = f"af{idx:04d}"
    new_series.append(pd.Series(sa - sb, dtype=np.float32))
    new_names.append(f"{pre}_diff")
    new_series.append(pd.Series(sa + sb, dtype=np.float32))
    new_names.append(f"{pre}_sum")
    new_series.append(pd.Series(np.divide(sa, sb + 0.001), dtype=np.float32))
    new_names.append(f"{pre}_div")
    new_series.append(pd.Series(sa * sb, dtype=np.float32))
    new_names.append(f"{pre}_mul")
    
    if (idx + 1) % 50 == 0:
        print(f"  {idx+1}/{n_pairs}对 → {len(new_names)}衍生因子", flush=True)

print(f"拼接 {len(new_names)}列...", flush=True)
df_new = pd.concat(new_series, axis=1)
df_new.columns = new_names

print(f"保存 parquet...", flush=True)
df_new.to_parquet("data/factors/auto_features_v1.parquet")
print(f"✅ data/factors/auto_features_v1.parquet ({df_new.shape})", flush=True)

json.dump({"n": n_pairs*4, "cols": new_names, "base_pairs": list(selected_pairs)},
          open("models/auto_features_meta.json","w"), indent=2)
print(f"✅ models/auto_features_meta.json", flush=True)
print(f"⏱ {(time.time()-tt)/60:.1f}分", flush=True)
