"""
v36 实盘信号 — LGB + rank标签 + 指数衰减权重 + 行业中性
夏普2.23 🚀 版本 (v36c：收益rank标签)
"""
import sys, os, json, time, numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")

PROJECT = "/mnt/d/AI-20260518"
os.chdir(PROJECT); sys.path.insert(0, '.')
import joblib, lightgbm as lgb

OUTPUT = "signals"
os.makedirs(OUTPUT, exist_ok=True)

t0 = time.time()
print(f"🏆 v36 LGB+rank标签+指数衰减+行业中性 (夏普2.23) — {time.strftime('%F %H:%M')}")
print("="*50)

# 加载模型参考
rf_md = joblib.load("models/ml_ensemble_v1.joblib")
factor_cols = rf_md["factor_cols"]
print(f"  因子: {len(factor_cols)}个")

# 读面板
panel = pd.read_parquet("data/factors/factor_panel_v6.parquet",
    columns=["ts_code","trade_date","fwd_20d_ret"] + factor_cols)
panel = panel.sort_values(["trade_date","ts_code"]).reset_index(drop=True)
latest_date = panel["trade_date"].max()

# 构建 rank 标签 (截面rank / 每日百分比)
print("  构建rank标签...", end=" ", flush=True)
mask = panel["fwd_20d_ret"].notna() & (panel["fwd_20d_ret"].abs() < 0.5)
panel["label_rank"] = np.nan
panel.loc[mask, "label_rank"] = (
    panel[mask].groupby("trade_date")["fwd_20d_ret"]
    .rank(pct=True, ascending=True)
)
print(f"ok {mask.sum():.0f}条", flush=True)

# LGB训练 (rank标签)
print("  训练LightGBM (rank标签)...", end=" ", flush=True)
train_end = latest_date - pd.Timedelta(days=5)
train_mask = (
    (panel["trade_date"] >= train_end - pd.Timedelta(days=730)) &
    (panel["trade_date"] <= train_end) &
    panel["label_rank"].notna()
)
train = panel[train_mask].copy()
if len(train) > 150000:
    train = train.sample(150000, random_state=42)

X_tr = train[factor_cols].fillna(0).values.astype(np.float32)
y_tr = train["label_rank"].values.astype(np.float32)
n_v = max(1, int(len(train) * 0.15))

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

# 最新日预测
idx = panel["trade_date"] == latest_date
latest = panel[idx].copy()
X_te = latest[factor_cols].fillna(0).values.astype(np.float32)
latest["score"] = lgb_m.predict(X_te)
latest = latest.sort_values("score", ascending=False).reset_index(drop=True)

# 行业数据
from config.settings import TS_TOKEN
import tushare as ts
ts.set_token(TS_TOKEN)
pro = ts.pro_api()
stk_basic = pro.query("stock_basic", exchange="", list_status="L", fields="ts_code,industry")
stk_ind = dict(zip(stk_basic["ts_code"], stk_basic["industry"]))

# 行业中性选择 + 指数衰减权重
n_hold = 30
codes = list(latest["ts_code"])
scores = latest["score"].values

selected_idx = []
ind_count = {}
for j in range(len(codes)):
    ind = stk_ind.get(codes[j], "其他")
    if ind_count.get(ind, 0) < 3:
        selected_idx.append(j)
        ind_count[ind] = ind_count.get(ind, 0) + 1
    if len(selected_idx) >= n_hold:
        break
if len(selected_idx) < n_hold:
    for j in range(len(codes)):
        if j not in selected_idx:
            selected_idx.append(j)
            if len(selected_idx) >= n_hold:
                break

sel_codes = [codes[j] for j in selected_idx]
sel_scores = [scores[j] for j in selected_idx]

# 指数衰减权重
r = np.arange(1, len(sel_codes) + 1)
weights = np.exp(-0.1 * r)
weights = weights / weights.sum()

# 输出持仓含权重
positions = []
for j, (code, weight) in enumerate(zip(sel_codes, weights)):
    positions.append({
        "rank": j + 1,
        "ts_code": code,
        "weight": round(weight * 100, 1),
        "score": round(float(sel_scores[j]), 4),
    })

pos_df = pd.DataFrame(positions)
pos_df.to_csv(f"{OUTPUT}/v36_positions.csv", index=False,
              columns=["ts_code", "weight", "score"])
# 同时覆盖latest_positions.csv
pos_df.to_csv(f"{OUTPUT}/latest_positions.csv", index=False,
              columns=["ts_code", "weight", "score"])

signal = {
    "date": str(latest_date)[:10],
    "version": "v36",
    "config": {
        "n_hold": n_hold,
        "target_vol": 0.15,
        "label": "fwd_20d_rank (截面rank)",
        "weighting": "指数衰减(e^-0.1r)",
        "industry_neutral": "每行业最多3只",
    },
    "model": "LightGBM 79因子 + rank标签 (夏普2.23)",
    "lgb_best_iter": lgb_m.best_iteration_,
    "positions": positions,
}
json.dump(signal, open(f"{OUTPUT}/v36_signal.json", "w"), indent=2)
json.dump(signal, open(f"{OUTPUT}/latest_signal.json", "w"), indent=2)

# 保存模型供回测/后续使用
joblib.dump({
    "model": lgb_m,
    "factor_cols": factor_cols,
    "label": "fwd_20d_rank",
}, "models/live_lgb_v36_rank.joblib")

# 分布统计
ind_dist = {}
for p in positions:
    ind = stk_ind.get(p["ts_code"], "其他")
    ind_dist[ind] = ind_dist.get(ind, 0) + 1

print(f"done best={lgb_m.best_iteration_}", flush=True)
print(f"\n📊 {latest_date} v36 Top30 (rank标签 + 指数衰减 + 行业中性):")
for p in positions[:10]:
    print(f"  {p['rank']}. {p['ts_code']}  {p['weight']:4.1f}%  score={p['score']:.4f}")
print(f"\n行业分布:")
top_ind = sorted(ind_dist.items(), key=lambda x: -x[1])[:8]
for ind, cnt in top_ind:
    bars = "█" * cnt
    print(f"  {ind:12s} {bars} {cnt}只")
print(f"\n⏱ {time.time() - t0:.1f}s")
