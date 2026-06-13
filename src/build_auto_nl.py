"""
auto_nl_features.py — 非线性衍生因子
做真正的增量特征：动量乖离、波动调整、滚动相关、量价关系
而非线性rank组合（LGB自己就能学）
"""
import os, sys, time, json, gc
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
os.chdir("/mnt/d/AI-20260518"); sys.path.insert(0, '.')
import joblib, lightgbm as lgb
tt = time.time()

print("="*60, flush=True)
print("Auto NL Features — 非线性衍生因子", flush=True)
print(f"{time.strftime('%F %H:%M')}", flush=True)
print("="*60, flush=True)

panel = pd.read_parquet("data/factors/factor_panel_v6.parquet",
    columns=["ts_code","trade_date","fwd_20d_ret","close","ret_1d"])
fac_base = [c for c in pd.read_parquet("data/factors/factor_panel_v6.parquet").columns 
            if c not in ["ts_code","trade_date","fwd_20d_ret","close","ret_1d",
                "__fragment_index","__batch_index","__last_in_fragment","__filename","board_break"]]
panel_b = pd.read_parquet("data/factors/factor_panel_v6.parquet", columns=fac_base).astype(np.float32)
panel = pd.concat([panel, panel_b], axis=1); del panel_b; gc.collect()
print(f"面板: {panel.shape}, {len(fac_base)}因子", flush=True)

# ===== 6类非线性衍生 =====
# 动量类
momo_cols = [c for c in fac_base if "动量" in c or "反转" in c or "偏离" in c]
vol_cols = [c for c in fac_base if "波动" in c or "RSI" in c]
vol_cols = fac_base  # 暂跳，用全部

new_features = {}
n = 0

# 1. 动量乖离率: 动量和与均值的差距
print("\n[1] 动量乖离...", flush=True)
momo_list = ["短期反转","20日动量","60日动量","120日动量","EMA5偏离","EMA10偏离","EMA20偏离"]
momo_list = [c for c in momo_list if c in fac_base]
if len(momo_list) >= 3:
    arrs = np.column_stack([panel[c].values for c in momo_list])
    new_features["momo_mean"] = np.nanmean(arrs, axis=1).astype(np.float32)
    new_features["momo_std"] = np.nanstd(arrs, axis=1).astype(np.float32)
    new_features["momo_skew"] = np.nanmean(((arrs - np.nanmean(arrs, axis=1, keepdims=True))**3), axis=1)
    n += 3

# 2. 波动调整动量
print("[2] 波动调整...", flush=True)
if "20日动量" in fac_base and "波动率" in fac_base:
    m20 = panel["20日动量"].values
    vol = panel["波动率"].values + 0.001
    new_features["momo20_adj"] = (m20 / vol).astype(np.float32)
    n += 1

# 3. 因子极端值信号 (上下15%分位)
print("[3] 极端信号...", flush=True)
risk_cols = ["波动率","换手率","量比","高波反转","量价背离"]
risk_cols = [c for c in risk_cols if c in fac_base]
for rc in risk_cols:
    vals = panel[rc].values
    p15, p85 = np.nanpercentile(vals, 15), np.nanpercentile(vals, 85)
    new_features[f"{rc}_extreme"] = ((vals < p15) | (vals > p85)).astype(np.float32)
    n += 1

# 4. 双因子交叉信号（非线性: 高动量+低波动才有效之类）
print("[4] 交叉信号...", flush=True)
pairs = [("20日动量","波动率"), ("短期反转","超跌信号"), ("量比","量能趋势"),
         ("高波反转","量价背离"), ("BP","20日动量"), ("换手率","波动率"),
         ("量价背离","波动率"), ("RSI_共振净信号","20日动量")]
pairs = [(a,b) for a,b in pairs if a in fac_base and b in fac_base]
for a,b in pairs:
    va = panel[a].rank(pct=True).values
    vb = panel[b].rank(pct=True).values
    new_features[f"cross_{a}_{b}"] = (va * vb * np.abs(va - vb)).astype(np.float32)
    n += 1

# 5. 因子间的非线性对比（取绝对差值再平方）
print("[5] 因子对比...", flush=True)
comp_pairs = [("短期反转","20日动量"), ("波动率","换手率"), ("量比","量能趋势"),
              ("高波反转","量价背离"), ("20日动量","60日动量"), ("MACD","BOLL位置"),
              ("BP","EP"), ("EMA5偏离","EMA20偏离")]
comp_pairs = [(a,b) for a,b in comp_pairs if a in fac_base and b in fac_base]
for a,b in comp_pairs:
    va = panel[a].values
    vb = panel[b].values
    diff = va - vb
    new_features[f"sqdiff_{a}_{b}"] = (diff * diff).astype(np.float32)
    new_features[f"absdiff_{a}_{b}"] = np.abs(diff).astype(np.float32)
    n += 2

# 6. 排名聚合信号
print("[6] 排名聚合...", flush=True)
# 取所有因子的rank均值
all_ranks = np.column_stack([panel[c].rank(pct=True).values for c in fac_base])
new_features["rank_mean"] = np.nanmean(all_ranks, axis=1).astype(np.float32)
new_features["rank_top10_pct"] = (np.nanmean(all_ranks > 0.9, axis=1)).astype(np.float32)
new_features["rank_bottom10_pct"] = (np.nanmean(all_ranks < 0.1, axis=1)).astype(np.float32)
n += 3

print(f"  总非线性衍生: {n}", flush=True)

# ===== 构建完整训练集 =====
print(f"\nLGB训练 ({n}衍生+{len(fac_base)}基础={len(fac_base)+n})...", flush=True)

train_idx = panel[panel["trade_date"] >= "2024-01-01"].sample(50000, random_state=42).index

# 构建特征矩阵
X_parts = [panel.loc[train_idx, fac_base].fillna(0).values.astype(np.float32)]
for name in new_features:
    X_parts.append(new_features[name][train_idx].reshape(-1,1).astype(np.float32))
X = np.column_stack(X_parts)

y_all = panel.loc[train_idx, "fwd_20d_ret"].values
valid = ~np.isnan(y_all) & (np.abs(y_all) < 0.5)
X = X[np.where(valid)[0]]
y = np.clip(y_all[valid], -0.3, 0.3)
print(f"  有效样本: {len(X)}", flush=True)

nv = max(1, int(len(X)*0.15))
lgb_m = lgb.LGBMRegressor(n_estimators=300, max_depth=3, lr=0.05,
    subsample=0.8, colsample_bytree=0.5, reg_alpha=0.5, reg_lambda=1.0,
    min_child_weight=20, min_data_in_leaf=100, random_state=42, verbose=-1, n_jobs=8)
lgb_m.fit(X[:-nv], y[:-nv], eval_set=[(X[-nv:], y[-nv:])],
          callbacks=[lgb.early_stopping(20, verbose=False)], eval_metric="mse")

all_features = fac_base + list(new_features.keys())
imp = pd.DataFrame({"factor": all_features, "importance": lgb_m.feature_importances_})
imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
imp_base = imp[imp["factor"].isin(fac_base)]
imp_nl = imp[~imp["factor"].isin(fac_base)]

print(f"\n  Top20:", flush=True)
for i,(_,row) in enumerate(imp.head(20).iterrows()):
    tag = " 🆕" if row["factor"] in new_features else ""
    print(f"  {i+1}. {row['factor'][:60]:60s} imp={row['importance']:>4d}{tag}", flush=True)

# 筛选
n_base = max(1, int(len(fac_base)*0.3))
top_base = imp_base.head(n_base)["factor"].tolist()
top_nl = imp_nl[imp_nl["importance"] > 0].head(20)["factor"].tolist()

sel = list(dict.fromkeys(top_base + top_nl))
print(f"\n入选: {len(sel)} ({len(top_base)}基础+{len(top_nl)}非线性)", flush=True)

out = {"selected": sel, "base": top_base, "nl": top_nl, "n_base": len(top_base), "n_nl": len(top_nl)}
json.dump(out, open("models/auto_features_lgb.json","w"), indent=2)
print(f"✅ models/auto_features_lgb.json", flush=True)

# 保存全量非线性衍生
print(f"保存非线性衍生因子...", flush=True)
df_nl = pd.DataFrame({k: v.astype(np.float32) for k,v in new_features.items()})
df_nl.to_parquet("data/factors/auto_nl_features.parquet")
print(f"✅ data/factors/auto_nl_features.parquet ({df_nl.shape})", flush=True)

print(f"⏱ {(time.time()-tt)/60:.1f}分", flush=True)
