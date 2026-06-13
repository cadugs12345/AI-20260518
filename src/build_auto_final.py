"""
build_auto_final.py — 最终版自动衍生因子
全程保持内存控制，逐批生成+LGB筛选
核心策略：生成100对衍生 → 只保留top 30个LGB重要性高衍生
"""
import os, sys, time, json, gc
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
os.chdir("/mnt/d/AI-20260518"); sys.path.insert(0, '.')
from scipy.stats import spearmanr
import joblib, lightgbm as lgb
tt = time.time()

print("="*60, flush=True)
print("AutoFeat Final — 内存控制版", flush=True)
print(f"{time.strftime('%F %H:%M')}", flush=True)
print("="*60, flush=True)

# 只读需要的列
panel = pd.read_parquet("data/factors/factor_panel_v6.parquet",
    columns=["ts_code","trade_date","fwd_20d_ret","close","ret_1d"])
factor_base = [c for c in pd.read_parquet("data/factors/factor_panel_v6.parquet").columns if c not in ["ts_code","trade_date","fwd_20d_ret","close","ret_1d",
    "__fragment_index","__batch_index","__last_in_fragment","__filename","board_break"]]
# 转float32省内存
base_panel = pd.read_parquet("data/factors/factor_panel_v6.parquet", columns=factor_base).astype(np.float32)
panel = pd.concat([panel, base_panel], axis=1)
del base_panel; gc.collect()
print(f"面板: {panel.shape}, {len(factor_base)}因子", flush=True)

# 预计算rank (float32直接放)
ranks = {fc: panel[fc].rank(pct=True).values.astype(np.float32) for fc in factor_base}
print(f"rank预计算: {list(ranks.keys())[:3]}...", flush=True)

# 低相关对
sm = panel.sample(100000, random_state=42)
corr = sm[factor_base].corr(method="spearman")
low_pairs = [(factor_base[i], factor_base[j]) for i in range(len(factor_base))
             for j in range(i+1,len(factor_base)) if abs(corr.iloc[i,j]) < 0.3]
low_pairs = low_pairs[:100]  # 100对=400衍生
print(f"低相关对: {len(low_pairs)}", flush=True)
del sm, corr; gc.collect()

# 训练index
train_idx = panel[(panel["trade_date"] >= "2024-01-01") &(panel["trade_date"] < "2025-06-01")].sample(30000, random_state=42).index

# ===== 流式生成+分批训练 =====
best_auto, best_imp = [], []  # 积累最佳衍生因子

for pi, (fi, fj) in enumerate(low_pairs):
    sa, sb = ranks[fi], ranks[fj]
    
    # 生成4种运算
    batch = {}
    for op, op_name, pref in [(sa-sb, "diff", "d"), (sa+sb, "sum", "s"), 
                               (np.divide(sa, sb+0.001), "div", "v"), (sa*sb, "mul", "m")]:
        varr = float(np.var(op))
        if varr > 0.001:
            name = f"af{pi:03d}{pref}"
            batch[name] = op.astype(np.float32)
    
    if not batch:
        continue
    
    # 加几对就训练一次小LGB看重要性
    if (pi+1) % 20 == 0:
        # 构建训练矩阵
        tr_X = np.column_stack([
            panel.loc[train_idx, factor_base].fillna(0).values.astype(np.float32),
        ] + [batch[n].astype(np.float32)[train_idx] for n in batch])
        
        tr_y = panel.loc[train_idx, "fwd_20d_ret"].values
        valid = ~np.isnan(tr_y) & (np.abs(tr_y) < 0.5)
        tr_X = tr_X[np.where(valid)[0]]
        tr_y = np.clip(tr_y[valid], -0.3, 0.3)
        
        if len(tr_X) < 1000:
            continue
        
        nv = max(1, int(len(tr_X)*0.15))
        lgb_t = lgb.LGBMRegressor(n_estimators=100, max_depth=3, lr=0.05,
            subsample=0.8, colsample_bytree=0.5, reg_alpha=0.5, reg_lambda=1.0,
            min_child_weight=20, min_data_in_leaf=100, random_state=42, verbose=-1, n_jobs=8)
        lgb_t.fit(tr_X[:-nv], tr_y[:-nv], eval_set=[(tr_X[-nv:], tr_y[-nv:])],
                  callbacks=[lgb.early_stopping(10, verbose=False)], eval_metric="mse")
        
        # 新衍生因子的重要性
        for j, name in enumerate(batch):
            imp_val = lgb_t.feature_importances_[len(factor_base)+j]
            if imp_val > 0:
                best_auto.append((name, imp_val, batch[name]))
        
        del tr_X, tr_y, lgb_t; gc.collect()
        print(f"  批{pi+1}/100: 积累{len(best_auto)}有效衍生", flush=True)

# 排序取top
best_auto.sort(key=lambda x: -x[1])
top_auto = best_auto[:30]  # 30个最好衍生
print(f"\nTop衍生因子:", flush=True)
for i, (name, imp, _) in enumerate(top_auto):
    print(f"  {i+1}. {name} imp={imp}", flush=True)

if len(top_auto) == 0:
    print("❌ 没有有效衍生因子!", flush=True)
    exit()

# ===== 全量回测 =====
print(f"\n[回测] Top{len(top_auto)}衍生 + 30%基础", flush=True)

# 构建最终特征集
n_base = max(1, int(len(factor_base)*0.3))
final_base = factor_base[:n_base]  # 简单取前30%（最好的因子之前已排在前面）
final_auto_names = [n for n,_,_ in top_auto]

# 保存
out = {"selected": final_base + final_auto_names, "base_cols": final_base, "auto_cols": final_auto_names}
json.dump(out, open("models/auto_features_lgb.json","w"), indent=2)
print(f"✅ models/auto_features_lgb.json ({len(final_base)}+{len(final_auto_names)})", flush=True)

# 生成全量衍生因子parquet（只保存top的）
print(f"保存衍生因子 parquet...", flush=True)
df_auto = pd.DataFrame({name: arr.astype(np.float32) for name,_,arr in top_auto})
df_auto.to_parquet("data/factors/auto_features_final.parquet")
print(f"✅ data/factors/auto_features_final.parquet ({df_auto.shape})", flush=True)

print(f"⏱ {(time.time()-tt)/60:.1f}分", flush=True)
