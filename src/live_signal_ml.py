"""
ML实盘信号生成器 - 替代等权合成实盘信号
每天09:30自动运行
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
print(f"🤖 ML实盘信号 — {time.strftime('%F %H:%M')}")
print("=" * 50)

# 加载模型
model_dict = joblib.load("models/ml_ensemble_v1.joblib")
model = model_dict["model"]
factor_cols = model_dict["factor_cols"]
print(f"  模型: RandomForest 79因子, 训练于{model_dict.get('train_date','?')}")

# 加载面板最新数据
panel = pd.read_parquet("data/factors/factor_panel_v6.parquet",
    columns=["ts_code","trade_date"] + factor_cols)
latest_date = panel["trade_date"].max()
idx = panel["trade_date"] == latest_date

latest = panel[idx].copy()
X = latest[factor_cols].fillna(0)

# ML打分
scores = model.predict_proba(X)[:, 1]
latest["ml_score"] = scores
latest = latest.sort_values("ml_score", ascending=False).reset_index(drop=True)

# 输出
n_hold = 30
top = latest.head(n_hold)
top.to_csv(f"{OUTPUT}/ml_latest_positions.csv", index=False, columns=["ts_code","ml_score"])

# 完整JSON
signal = {
    "date": str(latest_date)[:10],
    "model": "RandomForest_79factors",
    "model_ir": model_dict.get("ir", "?"),
    "n_hold": n_hold,
    "positions": [
        {"ts_code": row["ts_code"], "score": round(row["ml_score"], 4)}
        for _, row in top.iterrows()
    ]
}
with open(f"{OUTPUT}/latest_signal_ml.json", "w") as f:
    json.dump(signal, f, indent=2)

print(f"\n📊 {latest_date} ML Top{n_hold}:")
for i, p in enumerate(signal["positions"][:10]):
    print(f"  {i+1}. {p['ts_code']}  score={p['score']:.4f}")

print(f"\n  保存: signals/ml_latest_positions.csv")
print(f"  ⏱ {time.time()-t0:.1f}s")
