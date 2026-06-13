"""
build_auto_light.py — 轻量版自动衍生因子
不保存全部衍生因子，直接在生成过程中做方差+IC筛选
只保存top 100-200个有效衍生因子（float16节省空间）
"""
import os, sys, time, json
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
os.chdir("/mnt/d/AI-20260518"); sys.path.insert(0, '.')
from scipy.stats import spearmanr
import gc, joblib, lightgbm as lgb
tt = time.time()

print("="*60)
print("AutoFeat Light — 生成+筛选一体化")
print(f"{time.strftime('%F %H:%M')}")
print("="*60, flush=True)

panel = pd.read_parquet("data/factors/factor_panel_v6.parquet")
factor_base = [c for c in panel.columns 
               if c not in ["ts_code","trade_date","fwd_20d_ret","close","volume","ret_1d",
                   "__fragment_index","__batch_index","__last_in_fragment","__filename","board_break"]]
print(f"基础因子: {len(factor_base)}", flush=True)

# 用10万样本算相关性
panel_small = panel.sample(100000, random_state=42)
corr_mat = panel_small[factor_base].corr(method="spearman")
print("相关性矩阵完成", flush=True)

low_corr = []
for i in range(len(factor_base)):
    for j in range(i+1, len(factor_base)):
        if abs(corr_mat.iloc[i,j]) < 0.3:
            low_corr.append((factor_base[i], factor_base[j]))
print(f"低相关对: {len(low_corr)}", flush=True)

# 取200对
selected_pairs = low_corr[:200]

# 每个因子预计算rank
ranks = {}
for fc in factor_base:
    ranks[fc] = panel[fc].rank(pct=True).values
print("rank预计算完毕", flush=True)

# ===== 流式生成+筛选 =====
# 目标：只保留衍生因子中IC>=0.02或方差>0.005的
selected_cols = []
generated = []
auto_data = {}  # {col_name: array}
n_candidates = 0, 0

train_idx = panel[panel["trade_date"] >= "2024-01-01"].sample(30000, random_state=42).index

for pi, (fi, fj) in enumerate(selected_pairs):
    sa, sb = ranks[fi], ranks[fj]
    prefix = f"af{pi:04d}"
    
    for op, op_name in [(sa-sb, "diff"), (sa+sb, "sum"), 
                         (np.divide(sa, sb+0.001), "div"), (sa*sb, "mul")]:
        name = f"{prefix}_{op_name}"
        varr = np.var(op)
        if varr < 0.0005:
            continue  # 方差太低，跳过
        
        # 方差足够就保留（LGB会自动判重要性）
    auto_data[name] = op
    selected_cols.append(name)
    
    # 保持衍生因子数量可控
    if len(selected_cols) >= 320:
        break
    
    if (pi+1)%50 == 0:
        print(f"  {pi+1}/{len(selected_pairs)}对 → 积累{len(selected_cols)}候选", flush=True)

print(f"  候选衍生因子: {len(selected_cols)}", flush=True)

if len(selected_cols) == 0:
    print("❌ 没有找到有效的衍生因子!", flush=True)
    exit()

# ===== LGB特征重要性 =====
print("\nLGB特征重要性筛选...", flush=True)

# 构建训练数据
X_parts = [panel[factor_base].fillna(0)]
for col in selected_cols:
    X_parts.append(pd.DataFrame({col: auto_data[col]}))
    
train_X = pd.concat(X_parts, axis=1).loc[train_idx].fillna(0)
y_tr = panel.loc[train_idx, "fwd_20d_ret"].values
y_tr = np.clip(y_tr[~np.isnan(y_tr) & (y_tr<0.5)], -0.3, 0.3)

# 取有效行
valid = panel.loc[train_idx, "fwd_20d_ret"].notna().values & (panel.loc[train_idx, "fwd_20d_ret"].abs()<0.5).values
train_X = train_X.iloc[np.where(valid)[0]]
y_tr = y_tr[np.where(valid)[0]]

print(f"  有效训练: {len(train_X)}", flush=True)
all_features = list(train_X.columns)
X = train_X.values.astype(np.float32)

nv = max(1, int(len(X)*0.15))
lgb_fi = lgb.LGBMRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.5, reg_alpha=0.5, reg_lambda=1.0,
    min_child_weight=20, min_data_in_leaf=100, random_state=42, verbose=-1, n_jobs=8)
lgb_fi.fit(X[:-nv], y_tr[:-nv], eval_set=[(X[-nv:], y_tr[-nv:])],
           callbacks=[lgb.early_stopping(20, verbose=False)], eval_metric="mse")

imp = pd.DataFrame({"factor": all_features, "importance": lgb_fi.feature_importances_})
imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
imp_base = imp[imp["factor"].isin(factor_base)]
imp_auto = imp[~imp["factor"].isin(factor_base)]

n_base = max(1, int(len(factor_base)*0.3))
top_base = imp_base.head(n_base)["factor"].tolist()

# 取importance>0的衍生因子
top_auto_list = imp_auto[imp_auto["importance"]>0].head(40)["factor"].tolist()
sel_features = list(dict.fromkeys(top_base + top_auto_list))

print(f"\n  入选: {len(sel_features)} ({len(top_base)}基础+{len(top_auto_list)}衍生)", flush=True)
print(f"\n  Top20:", flush=True)
for i,(_,row) in enumerate(imp.head(20).iterrows()):
    tag = " 🆕" if row["factor"] in selected_cols else ""
    print(f"    {i+1}. {row['factor'][:60]:60s} imp={row['importance']}{tag}", flush=True)

# ===== 保存 =====
# 全量计算所有入选衍生因子 + 保存为float16 parquet
print(f"\n全量计算衍生因子...", flush=True)
new_series = []
for col in selected_cols:
    if col in auto_data and col in top_auto_list:
        new_series.append(pd.Series(auto_data[col].astype(np.float16), name=col))

if new_series:
    df_auto = pd.concat(new_series, axis=1)
    df_auto.to_parquet("data/factors/auto_features_light.parquet")
    print(f"  保存: {df_auto.shape}", flush=True)

# 保存筛选出的特征
out = {"selected": sel_features, "auto_cols": [c for c in top_auto_list if c in selected_cols],
       "n_base": len(top_base), "n_auto": len(top_auto_list)}
json.dump(out, open("models/auto_features_lgb.json","w"), indent=2)
print(f"✅ models/auto_features_lgb.json", flush=True)
print(f"✅ data/factors/auto_features_light.parquet", flush=True)
print(f"⏱ {(time.time()-tt)/60:.1f}分", flush=True)
