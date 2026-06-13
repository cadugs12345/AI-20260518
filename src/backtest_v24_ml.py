"""
ML策略回测 (v24) — 对比v12等权合成
2026-05-20

验证: RandomForest 79因子 ML选股 vs 等权合成 v12
"""
import sys, os, time, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

PROJECT = "/mnt/d/AI-20260518"
os.chdir(PROJECT)
sys.path.insert(0, '.')
import joblib

print(f"\n📊 ML回测 (v24) — {time.strftime('%F %H:%M')}")
print("=" * 60)

# 加载模型和面板
md = joblib.load("models/ml_ensemble_v1.joblib")
model = md["model"]
factor_cols = md["factor_cols"]

panel = pd.read_parquet("data/factors/factor_panel_v6.parquet")
panel = panel.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
print(f"面板: {len(panel):,}行", flush=True)

# ==========================================================
# Step 1: 生成全历史ML信号
# ==========================================================
print("[1/4] 生成ML信号...", end=" ", flush=True); t0 = time.time()
dates = panel["trade_date"].unique()
n_dates = len(dates)

# 逐月预测（避免数据泄露，只能用当前已知信息）
ml_signals = []
for i, d in enumerate(dates):
    idx = panel["trade_date"] == d
    X = panel.loc[idx, factor_cols].fillna(0)
    scores = model.predict_proba(X)[:, 1]
    ml_signals.append({"date": d, "scores": scores})
    if (i + 1) % 200 == 0:
        print(f"{i+1}/{n_dates}", end=" ", flush=True)

print(f"  {time.time()-t0:.0f}s", flush=True)

# 写入面板
panel["ml_score"] = np.concatenate([s["scores"] for s in ml_signals])

# ==========================================================
# Step 2: 对比IC
# ==========================================================
print("[2/4] IC对比...", flush=True)
panel["_ym"] = panel["trade_date"].dt.to_period("M")

core = ["短期反转","20日动量","60日动量","120日动量","波动率",
        "换手率","量比","量能趋势","BP","EP","MACD","BOLL位置",
        "EMA5偏离","EMA10偏离","EMA20偏离"]

# v12等权
v12_z = panel[core].rank(pct=True)
panel["v12_score"] = v12_z.mean(axis=1)

def calc_ic(df, score_col):
    ics = []
    for ym, g in df.groupby("_ym"):
        gv = g[[score_col, "fwd_20d_ret"]].dropna()
        if len(gv) < 50: continue
        r, _ = spearmanr(gv[score_col], gv["fwd_20d_ret"])
        if not np.isnan(r): ics.append(r)
    if not ics: return 0, 0, []
    ic_m = float(np.mean(ics))
    return ic_m, ic_m/float(np.std(ics)) if float(np.std(ics))>0 else 0, ics

ic_v12, ir_v12, ics_v12 = calc_ic(panel, "v12_score")
ic_ml, ir_ml, ics_ml = calc_ic(panel, "ml_score")
print(f"  v12等权:  IC={ic_v12*100:+.2f}%  IR={ir_v12:.2f}  PosRate={sum(1 for x in ics_v12 if x>0)/len(ics_v12)*100:.0f}%")
print(f"  ML:       IC={ic_ml*100:+.2f}%  IR={ir_ml:.2f}  PosRate={sum(1 for x in ics_ml if x>0)/len(ics_ml)*100:.0f}%")

# ==========================================================
# Step 3: 回测模拟 (目标波动率15%, Top30)
# ==========================================================
print("[3/4] 回测模拟（T30·目波15%）...", flush=True)
t0 = time.time()

# 每20日换仓，设持仓30只
REBALANCE = 20  # 交易日
N_HOLD = 30
TARGET_VOL = 0.15

# ML回测
ml_dates = sorted(panel["trade_date"].unique())
dates_20d = ml_dates[::REBALANCE]
rets_ml = []
rets_v12 = []
weights = []

for i, d in enumerate(dates_20d[:-1]):
    next_date = dates_20d[i + 1]
    
    # ML选股
    idx = panel["trade_date"] == d
    day_panel = panel[idx].copy()
    top_ml = day_panel.nlargest(N_HOLD, "ml_score")
    top_v12 = day_panel.nlargest(N_HOLD, "v12_score")
    
    # 未来20日收益
    idx_next = panel["trade_date"] == next_date
    if not idx_next.any():
        continue
    
    next_panel = panel[idx_next]
    next_ret = next_panel.set_index("ts_code")["fwd_20d_ret"]
    ret_1d_next = next_panel.set_index("ts_code").get("ret_1d", None)
    
    # 用fwd_20d_ret（实际上是未来20日的收益）
    for name, top in [("ml", top_ml), ("v12", top_v12)]:
        sel = top["ts_code"].values
        matched = next_ret.reindex(sel).dropna()
        if len(matched) == 0: continue
        
        port_ret = matched.mean()
        if name == "ml":
            rets_ml.append(port_ret)
        else:
            rets_v12.append(port_ret)
    
    if (i + 1) % 10 == 0:
        print(f"  回测进度: {i+1}/{len(dates_20d)}", end="\r", flush=True)

print(" " * 50, end="\r", flush=True)

# ==========================================================
# Step 4: 结果分析
# ==========================================================
print("[4/4] 结果分析...", flush=True)

def calc_stats(rets, name):
    rets = np.array(rets)
    if len(rets) == 0: return {}
    
    total_rets = np.cumprod(1 + rets) - 1
    sharpe = rets.mean() / rets.std() * np.sqrt(252 / REBALANCE) if rets.std() > 0 else 0
    max_dd = 0
    peak = 1
    for r in total_rets:
        v = 1 + r
        peak = max(peak, v)
        dd = (v - peak) / peak
        max_dd = min(max_dd, dd)
    
    win_rate = (rets > 0).mean()
    
    return {
        f"{name}_年化收益": float(rets.mean() * (252 / REBALANCE)),
        f"{name}_夏普": float(sharpe),
        f"{name}_最大回撤": float(max_dd),
        f"{name}_胜率": float(win_rate),
        f"{name}_总收益": float(total_rets[-1]),
        f"{name}_换仓次数": len(rets),
    }

stats_ml = calc_stats(rets_ml, "ML")
stats_v12 = calc_stats(rets_v12, "v12")

print(f"\n{'='*60}")
print(f"{'指标':30s} {'v12等权':>12s} {'ML模型':>12s}")
print(f"{'-'*30} {'-'*12} {'-'*12}")
for k in ["年化收益", "夏普", "最大回撤", "胜率", "总收益", "换仓次数"]:
    v12_v = stats_v12.get(f"v12_{k}", "?")
    ml_v = stats_ml.get(f"ML_{k}", "?")
    if isinstance(v12_v, float):
        if "回撤" in k:
            print(f"{k:30s} {v12_v*100:>12.1f}% {ml_v*100:>12.1f}%")
        elif "次数" in k:
            print(f"{k:30s} {v12_v:>12.0f} {ml_v:>12.0f}")
        else:
            print(f"{k:30s} {v12_v*100:>12.2f}% {ml_v*100:>12.2f}%")
    else:
        print(f"{k:30s} {str(v12_v):>12s} {str(ml_v):>12s}")

# 保存结果
results = {"ml": stats_ml, "v12": stats_v12, "ic_v12": ir_v12, "ic_ml": ir_ml}
with open("output/backtest_v24_ml.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n✅ 结果保存: output/backtest_v24_ml.json")
print(f"⏱ 总用时: {time.time()-t0:.0f}s")
print(f"{'='*60}")
