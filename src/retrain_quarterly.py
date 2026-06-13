"""
📅 季度模型重训练 — 用最新730天数据重新训练 v38 模型
只在季度末执行（3/6/9/12月最后一个交易日）

产出：
  1. models/live_lgb_v38_final.joblib   ← 覆盖最新版
  2. 训练日志到 logs/retrain_YYYYMMDD.log
"""

import sys, os, json, time, numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")

PROJECT = "/mnt/d/AI-20260518"
os.chdir(PROJECT); sys.path.insert(0, '.')
import joblib, lightgbm as lgb

t0 = time.time()
print(f"📅 季度模型重训练 — {time.strftime('%F %H:%M')}")
print("="*50)

# 加载参考因子列
ref = joblib.load("models/ml_ensemble_v1.joblib")
factor_cols = ref["factor_cols"]
print(f"  因子: {len(factor_cols)}个")

# 读完整面板
panel = pd.read_parquet("data/factors/factor_panel_v5_final.parquet")
panel = panel.sort_values(["trade_date","ts_code"]).reset_index(drop=True)
latest_date = panel["trade_date"].max()
print(f"  面板: {len(panel):,}行, 最新 {latest_date}")

# 构建rank标签
print("  构建rank标签...", end=" ", flush=True)
mask = panel["fwd_20d_ret"].notna() & (panel["fwd_20d_ret"].abs() < 0.5)
panel["label_rank"] = np.nan
panel.loc[mask, "label_rank"] = (
    panel[mask].groupby("trade_date")["fwd_20d_ret"]
    .rank(pct=True, ascending=True)
)
print(f"ok {mask.sum():,.0f}条", flush=True)

# 训练: 最新730天, 留5天gap
train_end = latest_date - pd.Timedelta(days=5)
train_mask = (
    (panel["trade_date"] >= train_end - pd.Timedelta(days=730)) &
    (panel["trade_date"] <= train_end) &
    panel["label_rank"].notna()
)
train = panel[train_mask].copy()
print(f"  训练集: {len(train):,}行 ({train['trade_date'].min()} ~ {train['trade_date'].max()})")

if len(train) > 200000:
    train = train.sample(200000, random_state=42)
    print(f"  采样: 200,000行")

X_tr = train[factor_cols].fillna(0).values.astype(np.float32)
y_tr = train["label_rank"].values.astype(np.float32)
n_v = max(1, int(len(train) * 0.15))

# 训练
print("  训练中...", end=" ", flush=True)
train_t0 = time.time()
lgb_m = lgb.LGBMRegressor(
    n_estimators=500, max_depth=3, learning_rate=0.02,
    subsample=0.7, colsample_bytree=0.7,
    reg_alpha=0.2, reg_lambda=1.0,
    min_child_weight=20, min_data_in_leaf=100,
    random_state=42, verbose=-1, n_jobs=8,
)
lgb_m.fit(
    X_tr[:-n_v], y_tr[:-n_v],
    eval_set=[(X_tr[-n_v:], y_tr[-n_v:])],
    callbacks=[lgb.early_stopping(30, verbose=False)],
    eval_metric="mse",
)
print(f"完成 ({time.time()-train_t0:.0f}s)  best_iter={lgb_m.best_iteration_}")

# 保存
output = {
    "model": lgb_m,
    "factor_cols": factor_cols,
    "label": "fwd_20d_rank",
    "train_date": str(latest_date)[:10],
    "train_samples": len(train),
}
joblib.dump(output, "models/live_lgb_v38_final.joblib")
print(f"  ✅ 已保存: models/live_lgb_v38_final.joblib")

# 验证：预测最新日
idx = panel["trade_date"] == latest_date
latest = panel[idx].copy()
if len(latest) > 0:
    X_te = latest[factor_cols].fillna(0).values.astype(np.float32)
    latest["score"] = lgb_m.predict(X_te)
    latest = latest.sort_values("score", ascending=False)
    top5_avg = latest.head(5)["label_rank"].mean()
    print(f"  最新日Top5 rank均值: {top5_avg:.4f} (越高越好, max≈0.95)")
    
    # 特征重要性
    imp = pd.DataFrame({
        'feature': factor_cols,
        'gain': lgb_m.booster_.feature_importance(importance_type='gain'),
    }).sort_values('gain', ascending=False)
    print(f"\n  Top5特征:")
    for _, r in imp.head(5).iterrows():
        print(f"    {r['feature']:20s} gain={r['gain']:.1f}")

print(f"\n⏱ 总计: {time.time()-t0:.0f}s")
