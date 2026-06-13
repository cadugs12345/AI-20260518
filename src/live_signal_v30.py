"""
v30 实盘信号生成器 — LGB+RF风控集成版
LightGBM预测 + RF回退 + 风控剔除
"""
import sys, os, json, time
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")

PROJECT = "/mnt/d/AI-20260518"
os.chdir(PROJECT)
sys.path.insert(0, '.')
import joblib
import lightgbm as lgb

OUTPUT = "signals"
os.makedirs(OUTPUT, exist_ok=True)

t0 = time.time()
print(f"🚀 v30 LGB+RF集成实盘信号 — {time.strftime('%F %H:%M')}")
print("=" * 50)

rf_md = joblib.load("models/ml_ensemble_v1.joblib")
rf_model, factor_cols = rf_md["model"], rf_md["factor_cols"]
print(f"  RF模型: RandomForest {len(factor_cols)}因子")

# 读面板 — 只读需要的列
read_cols = ["ts_code","trade_date","fwd_20d_ret",
             "repair_force_10d","高波反转","量价背离"] + factor_cols
read_cols = list(dict.fromkeys(read_cols))  # 去重
panel = pd.read_parquet("data/factors/factor_panel_v6.parquet", columns=read_cols)
panel = panel.sort_values(["trade_date","ts_code"]).reset_index(drop=True)

latest_date = panel["trade_date"].max()
idx = panel["trade_date"] == latest_date
latest = panel[idx].copy()

# RF预测
X_rf = latest[factor_cols].fillna(0).values.astype(np.float32)
latest["pred_rf"] = rf_model.predict_proba(X_rf)[:, 1]

# LGB训练
print("  训练LightGBM...", end=" ", flush=True)
train_end = latest_date - pd.Timedelta(days=5)
train_start = train_end - pd.Timedelta(days=730)

train = panel[(panel["trade_date"] >= train_start) & (panel["trade_date"] <= train_end)]
train = train[train["fwd_20d_ret"].notna() & (train["fwd_20d_ret"].abs() < 0.5)]

if len(train) > 150000:
    train = train.sample(150000, random_state=42)

X_tr = train[factor_cols].fillna(0).values.astype(np.float32)
y_tr = np.clip(train["fwd_20d_ret"].values.astype(np.float32), -0.3, 0.3)

n_val = max(1, int(len(train) * 0.15))
lgb_m = lgb.LGBMRegressor(
    n_estimators=500, max_depth=3, learning_rate=0.02,
    subsample=0.7, colsample_bytree=0.7,
    reg_alpha=0.2, reg_lambda=1.0,
    min_child_weight=20, min_data_in_leaf=100,
    random_state=42, verbose=-1, n_jobs=8
)
lgb_m.fit(X_tr[:-n_val], y_tr[:-n_val],
          eval_set=[(X_tr[-n_val:], y_tr[-n_val:])],
          callbacks=[lgb.early_stopping(30, verbose=False)],
          eval_metric="mse")

# LGB预测
X_te = latest[factor_cols].fillna(0).values.astype(np.float32)
latest["pred_lgb"] = lgb_m.predict(X_te)

# 集成：LGB偏离中位数越远越自信
latest["lgb_pct"] = latest["pred_lgb"].rank(pct=True)
latest["pred_ensemble"] = 0.0
for j in range(len(latest)):
    lgb_v = latest.iloc[j]["pred_lgb"]
    rf_v = latest.iloc[j]["pred_rf"]
    lgb_p = latest.iloc[j]["lgb_pct"]
    w = np.clip(abs(lgb_p - 0.5) * 2, 0.3, 0.8)
    latest.iloc[j, latest.columns.get_loc("pred_ensemble")] = w * lgb_v + (1 - w) * rf_v

latest = latest.sort_values("pred_ensemble", ascending=False).reset_index(drop=True)

# 风控
n_hold = 30
c_mask = np.zeros(len(latest), dtype=bool)
for j, (_, row) in enumerate(latest.iterrows()):
    r10 = row.get("repair_force_10d", np.nan)
    hv = row.get("高波反转", np.nan)
    dv = row.get("量价背离", np.nan)
    if (not np.isnan(r10) and r10 < -0.05) or \
       (not np.isnan(hv) and hv < -0.03) or \
       (not np.isnan(dv) and dv > 0.03):
        c_mask[j] = True

risk_removed = int(c_mask.sum())
safe = latest[~c_mask]
if len(safe) < n_hold:
    safe = latest
    risk_removed = 0

top = safe.head(n_hold)
top.to_csv(f"{OUTPUT}/v30_positions.csv", index=False,
           columns=["ts_code","pred_ensemble","pred_rf","pred_lgb"])

signal = {
    "date": str(latest_date)[:10],
    "version": "v30",
    "model": "LGB+RF集成+风控",
    "n_hold": n_hold,
    "risk_removed": risk_removed,
    "lgb_best_iter": lgb_m.best_iteration_,
    "positions": [
        {"ts_code": row["ts_code"], "score": round(row["pred_ensemble"], 4)}
        for _, row in top.iterrows()
    ]
}
with open(f"{OUTPUT}/v30_signal.json", "w") as f:
    json.dump(signal, f, indent=2)

joblib.dump({"model_lgb": lgb_m, "factor_cols": factor_cols},
            "models/ml_ensemble_v2.joblib")

print(f"done LGB best={lgb_m.best_iteration_}", flush=True)
print(f"\n📊 {latest_date} v30 Top{n_hold} (风控剔除{risk_removed}只):")
for i, p in enumerate(signal["positions"][:10]):
    print(f"  {i+1}. {p['ts_code']}  score={p['score']:.4f}")
print(f"\n  ⏱ {time.time()-t0:.1f}s")
