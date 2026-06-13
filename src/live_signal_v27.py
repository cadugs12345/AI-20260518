"""
ML实盘信号生成器 v27 — RF+风控 (替代等权合成)
每天09:30自动运行
风控规则: 排除断板修复<-5% 或 高波反转<-3% 或 量价背离>3%
"""
import sys, os, json, time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

PROJECT = "/mnt/d/AI-20260518"
os.chdir(PROJECT)
sys.path.insert(0, '.')
import joblib

OUTPUT = "signals"
os.makedirs(OUTPUT, exist_ok=True)

t0 = time.time()
print(f"🤖 v27 ML实盘信号 — {time.strftime('%F %H:%M')}")
print("=" * 50)

# 加载模型+面板
md = joblib.load("models/ml_ensemble_v1.joblib")
model = md["model"]
factor_cols = md["factor_cols"]
print(f"  模型: RandomForest 79因子")

# factor_cols已包含风控因子，不要重复添加
need_cols = ["ts_code","trade_date"] + factor_cols
panel = pd.read_parquet("data/factors/factor_panel_v6.parquet", columns=need_cols)
panel = panel.sort_values(["trade_date","ts_code"]).reset_index(drop=True)

latest_date = panel["trade_date"].max()
idx = panel["trade_date"] == latest_date
latest = panel[idx].copy()

# ML打分 — numpy数组确保列顺序
X_arr = np.column_stack([latest[c].values for c in factor_cols])
X_arr = np.nan_to_num(X_arr.astype(np.float32), nan=0.0)
latest["ml_score"] = model.predict_proba(X_arr)[:, 1]
latest = latest.sort_values("ml_score", ascending=False).reset_index(drop=True)

# 风控
n_hold = 30
candidates = latest.copy()
c_mask = np.zeros(len(candidates), dtype=bool)
for j, (_, row) in enumerate(candidates.iterrows()):
    r10 = row.get("repair_force_10d", np.nan)
    hv = row.get("高波反转", np.nan)
    dv = row.get("量价背离", np.nan)
    if (not np.isnan(r10) and r10 < -0.05) or \
       (not np.isnan(hv) and hv < -0.03) or \
       (not np.isnan(dv) and dv > 0.03):
        c_mask[j] = True

risk_removed = int(c_mask.sum())
safe = candidates[~c_mask]

# 如果风控剔除后不够30只，回退到全量
if len(safe) < n_hold:
    safe = candidates
    risk_removed = 0

top = safe.head(n_hold)
top.to_csv(f"{OUTPUT}/v27_positions.csv", index=False, columns=["ts_code","ml_score"])
top.to_csv(f"{OUTPUT}/latest_positions.csv", index=False, columns=["ts_code","ml_score"])

# 完整JSON
signal = {
    "date": str(latest_date)[:10],
    "version": "v27",
    "model": "RandomForest_79factors + 风控",
    "n_hold": n_hold,
    "risk_removed": risk_removed,
    "positions": [
        {"ts_code": row["ts_code"], "score": round(row["ml_score"], 4)}
        for _, row in top.iterrows()
    ]
}
with open(f"{OUTPUT}/v27_signal.json", "w") as f:
    json.dump(signal, f, indent=2)

print(f"\n📊 {latest_date} v27 Top{n_hold} (风控剔除{risk_removed}只):")
for i, p in enumerate(signal["positions"][:10]):
    print(f"  {i+1}. {p['ts_code']}  score={p['score']:.4f}")

print(f"\n  保存: signals/v27_positions.csv")
if risk_removed > 0:
    print(f"  风控: 剔除{risk_removed}只高风险股")
print(f"  ⏱ {time.time()-t0:.1f}s")
