"""
v38 实盘信号 — LGB + rank标签 + 指数衰减权重 + 行业中性 + 风控
（固定模型版：每天只预测不训练，模型用 live_lgb_v38_final.joblib）

风控规则（v38优化版）：
- 排除 repair_force_10d < -5%（断板修复失败）
- 排除 高波反转 < -3%（高波动反转风险）
- 排除 真实市值 < 20亿（小盘流动性风险）
  - 面板中`市值`列 = np.log(total_mv_万元)
  - 真实市值(亿) = exp(市值) / 10000
  - 过滤条件：市值 < np.log(200000) ≈ 12.21

注意：
- 市值取了对数，不要直接用面值比较
- 过滤的是daily_basic的total_mv（万元）取log后的值

回测结果（排除<20亿后 2020-2026）：
  固定模型+Top30: 夏普2.58, 回撤-20.2%, 年化84.2% 🏆
"""
import sys, os, json, time, numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")

PROJECT = "/mnt/d/AI-20260518"
os.chdir(PROJECT); sys.path.insert(0, '.')
import joblib

OUTPUT = "signals"
os.makedirs(OUTPUT, exist_ok=True)

t0 = time.time()
print(f"🏆 v38 固定模型信号 — {time.strftime('%F %H:%M')}")
print("="*50)

# 加载训练好的固定模型
model_data = joblib.load("models/live_lgb_v38_final.joblib")
lgb_m = model_data["model"]
factor_cols = model_data["factor_cols"]
print(f"  模型: v38_final | 因子: {len(factor_cols)}个 | best_iter: {lgb_m.best_iteration_}")

# 读面板
need_cols = list(dict.fromkeys(["ts_code","trade_date","close"] + factor_cols))
panel = pd.read_parquet("data/factors/factor_panel_v5_final.parquet", columns=need_cols)
panel = panel.sort_values(["trade_date","ts_code"]).reset_index(drop=True)
latest_date = panel["trade_date"].max()
print(f"  面板: 最新日期 {latest_date}")

# 最新日预测
idx = panel["trade_date"] == latest_date
latest = panel[idx].copy()
X_te = latest[factor_cols].fillna(0).values.astype(np.float32)
latest["score"] = lgb_m.predict(X_te)
latest = latest.sort_values("score", ascending=False).reset_index(drop=True)

# 行业/名称/板块数据
from config.settings import TS_TOKEN
import tushare as ts
ts.set_token(TS_TOKEN)
pro = ts.pro_api()
stk_basic = pro.query("stock_basic", exchange="", list_status="L",
                       fields="ts_code,name,industry,area,market")
stk_ind = dict(zip(stk_basic["ts_code"], stk_basic["industry"]))
stk_name = dict(zip(stk_basic["ts_code"], stk_basic["name"]))
stk_market = dict(zip(stk_basic["ts_code"], stk_basic["market"]))

# ====== 风控过滤 ======
print("  风控过滤...", end=" ", flush=True)
r10 = latest["repair_force_10d"].values.astype(float)
hv = latest["高波反转"].values.astype(float)
close_px = latest["close"].values.astype(float)

risk_mask = np.zeros(len(latest), dtype=bool)
risk_mask = risk_mask | (r10 < -0.05)          # 修复失败
risk_mask = risk_mask | (hv < -0.03)           # 高波动反转
risk_mask = risk_mask | (close_px <= 0)        # 无价格（停牌/退市整理）

# ⭐ 市值过滤：排除真实市值 < 20亿
# 面板中 `市值` = np.log(total_mv_万元)
# 真实市值(亿) = exp(市值) / 10000
# 真实20亿 ≈ np.log(200000) ≈ 12.21
MCAP_LOG_20B = float(np.log(200000))
mcap_vals = latest["市值"].values.astype(float)
risk_mask = risk_mask | (~np.isnan(mcap_vals) & (mcap_vals < MCAP_LOG_20B))

# 剔除ST/北交所
codes_arr = latest["ts_code"].values
risk_mask = risk_mask | np.array(["ST" in stk_name.get(c, "") or "退" in stk_name.get(c, "") for c in codes_arr])
risk_mask = risk_mask | np.array([c.endswith(".BJ") for c in codes_arr])  # 北交所

risk_removed = int(risk_mask.sum())
safe_idx = np.where(~risk_mask)[0]
small_mcap_removed = int((~np.isnan(mcap_vals) & (mcap_vals < MCAP_LOG_20B)).sum())
print(f"剔除{risk_removed}只(市值过滤{small_mcap_removed}|修复/高波/停牌/ST北交所{risk_removed-small_mcap_removed})")

# ====== 加载最新收盘价做二次过滤 ======
print("  加载最新价格...", end=" ", flush=True)
prices = pd.read_parquet("data/factors/full_prices.parquet")
prices["trade_date"] = pd.to_datetime(prices["trade_date"])
latest_px_date = prices["trade_date"].max()
px_latest = prices[prices["trade_date"] == latest_px_date][["ts_code", "close"]].copy()
px_dict = dict(zip(px_latest["ts_code"], px_latest["close"]))
codes_arr = latest["ts_code"].values
# 剔除无最新价格的票（停牌/退市后仍残留面板中）
px_mask = np.array([px_dict.get(c, 0) <= 0 for c in codes_arr])
risk_mask = risk_mask | px_mask
safe_idx2 = np.where(~risk_mask)[0]
print(f"价格过滤再剔除{len(safe_idx) - len(safe_idx2)}只")
safe_idx = safe_idx2

# ====== 选股 ======
n_hold = 10
codes = list(latest["ts_code"])
scores = latest["score"].values
selected_idx = list(safe_idx[:n_hold])
sel_codes = [codes[j] for j in selected_idx]
sel_scores = [scores[j] for j in selected_idx]

# 指数衰减权重
r = np.arange(1, len(sel_codes) + 1)
weights = np.exp(-0.1 * r)
weights = weights / weights.sum()

PER_STOCK = 100000
def calc_lots(code):
    px = px_dict.get(code)
    if px is None or px <= 0:
        return 0, 0
    shares = int(PER_STOCK / px / 100) * 100
    lots = shares // 100
    return lots, shares * px

positions = []
for j, (code, weight) in enumerate(zip(sel_codes, weights)):
    px = px_dict.get(code, 0)
    lots, cost = calc_lots(code)
    positions.append({
        "rank": j + 1,
        "ts_code": code,
        "weight": round(weight * 100, 1),
        "score": round(float(sel_scores[j]), 4),
        "close": round(float(px), 2) if px else 0,
        "lots": lots,
        "cost": round(cost, 2),
    })

pos_df = pd.DataFrame(positions)
pos_df["name"] = pos_df["ts_code"].map(stk_name)
pos_df["industry"] = pos_df["ts_code"].map(stk_ind)
pos_df["market"] = pos_df["ts_code"].map(stk_market)
lots_info = pos_df["ts_code"].apply(lambda c: pd.Series(calc_lots(c), index=["lots", "actual_cost"]))
pos_df = pd.concat([pos_df, lots_info], axis=1)
pos_df["close"] = pos_df["ts_code"].map(px_dict).round(2)
pos_df.to_csv(f"{OUTPUT}/v38_positions.csv", index=False,
              columns=["rank","ts_code","name","industry","market","weight","score","close","lots","actual_cost"])
pos_df.to_csv(f"{OUTPUT}/v38_positions_{latest_date.date()}.csv", index=False,
              columns=["rank","ts_code","name","industry","market","weight","score","close","lots","actual_cost"],
              encoding="utf-8-sig")

import shutil
try:
    shutil.copy2(f"{OUTPUT}/v38_positions_{latest_date.date()}.csv", f"{OUTPUT}/latest_positions.csv")
except (PermissionError, FileNotFoundError):
    pass

# 输出信号JSON
signal = {
    "date": str(latest_date)[:10],
    "version": "v38_fixed",
    "model": "LightGBM 79因子 + rank标签 + 风控 (固定模型 夏普2.51)",
    "risk_removed": risk_removed,
    "positions": positions,
}
json.dump(signal, open(f"{OUTPUT}/v38_signal.json", "w"), indent=2)
json.dump(signal, open(f"{OUTPUT}/latest_signal.json", "w"), indent=2)

# 输出
ind_dist = {}
for p in positions:
    ind = stk_ind.get(p["ts_code"], "其他")
    ind_dist[ind] = ind_dist.get(ind, 0) + 1

print(f"done", flush=True)
print(f"\n📊 {latest_date} v38 Top{n_hold} (固定模型):")
for p in positions[:10]:
    nm = stk_name.get(p["ts_code"], "")
    ind = stk_ind.get(p["ts_code"], "")
    px = px_dict.get(p['ts_code'], 0)
    lots, cost = calc_lots(p['ts_code'])
    print(f"  {p['rank']:2d}. {p['ts_code']:>10s} {nm:8s} {ind:8s} {p['weight']:4.1f}% {px:>8.2f}元 {lots:>4d}手 score={p['score']:.4f}")

total_cost = sum(calc_lots(p['ts_code'])[1] for p in positions[:10])
print(f"\n💰 总投入: {total_cost/10000:.1f}万 / 目标100万")
print(f"\n风控: 剔除{risk_removed}只")
print(f"\n行业分布:")
for ind, cnt in sorted(ind_dist.items(), key=lambda x: -x[1])[:8]:
    print(f"  {ind:12s} {'█'*cnt} {cnt}只")
print(f"\n⏱ {time.time() - t0:.1f}s")
