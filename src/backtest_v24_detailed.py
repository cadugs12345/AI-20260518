"""
v24 ML精细回测 — 目标波动率框架 (与v12公平对比)
2026-05-20
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

print(f"📊 v24 ML精细回测 — {time.strftime('%F %H:%M')}")
print("=" * 60)
tt = time.time()

md = joblib.load("models/ml_ensemble_v1.joblib")
model = md["model"]
factor_cols = md["factor_cols"]

core15 = ["短期反转","20日动量","60日动量","120日动量","波动率",
          "换手率","量比","量能趋势","BP","EP","MACD","BOLL位置",
          "EMA5偏离","EMA10偏离","EMA20偏离"]

cols = ["ts_code","trade_date","fwd_20d_ret","ret_1d"] + factor_cols + core15
cols = list(dict.fromkeys(cols))

panel = pd.read_parquet("data/factors/factor_panel_v6.parquet", columns=cols)
panel = panel.sort_values(["trade_date","ts_code"]).reset_index(drop=True)
print(f"面板: {len(panel):,}行", flush=True)

# ML预测（只换仓日做预测）
dates = sorted(panel["trade_date"].unique())
T = len(dates)
pred_dates = dates  # 每日预测

panel["ml_score"] = np.nan
for d in pred_dates:
    idx = panel["trade_date"] == d
    X = panel.loc[idx, factor_cols].fillna(0)
    panel.loc[idx, "ml_score"] = model.predict_proba(X)[:, 1]
panel["ml_score"] = panel.groupby("ts_code")["ml_score"].ffill()

# v12等权
v12_z = panel[core15].rank(pct=True)
panel["v12_score"] = v12_z.mean(axis=1)

# ========================================================
# 回测引擎 — 目标波动率15%, Top30
# ========================================================
TARGET_VOL = 0.15
N_HOLD = 30

cagr = lambda r, f: (1 + r) ** f - 1

def run_backtest(panel, score_col, name):
    print(f"\n[{name}] 回测中...", flush=True)
    positions = {}  # date -> [ts_code, ...]
    rets = []  # 每日组合收益
    
    for i in range(1, T):
        d = dates[i]
        prev_d = dates[i-1]
        
        day_data = panel[panel["trade_date"] == d].set_index("ts_code")
        ret_1d = day_data["ret_1d"].reindex(positions.get(prev_d, [])).dropna()
        
        # 换仓日：每20日
        if i % 20 == 1 or i == 1:
            dp = panel[panel["trade_date"] == d]
            top = dp.nlargest(N_HOLD, score_col)
            positions[d] = list(top["ts_code"])
        
        positions[d] = positions.get(prev_d, [])
        
        if ret_1d.std() > 0:
            scale = TARGET_VOL / ret_1d.std() / np.sqrt(252)
            scaled_ret = ret_1d.mean() * min(scale, 2.0)
            rets.append(scaled_ret)
        else:
            rets.append(0)
        
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{T}", end="\r", flush=True)
    
    rets = np.array(rets)
    if len(rets) < 2:
        return {"ret": 0, "sharpe": 0, "mdd": 0, "wr": 0, "n": 0}
    
    ann = rets.mean() * 252
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    cf = np.cumprod(1 + rets)
    mdd = (np.maximum.accumulate(cf) - cf).max()
    wr = (rets > 0).mean()
    
    print(f"  {name}: 年化{ann*100:+.1f}% | 夏普{sharpe:.2f} | 回撤{mdd*100:.1f}% | 胜率{wr*100:.0f}%")
    return {"ret": float(ann), "sharpe": float(sharpe), "mdd": float(mdd),
            "wr": float(wr), "n": int(len(rets))}

# ML每日预测很快，但回测数据量大
ret_v12 = run_backtest(panel, "v12_score", "v12")
ret_ml = run_backtest(panel, "ml_score", "ML")

result = {"v12": ret_v12, "ml": ret_ml}
with open("output/backtest_v24_ml_detailed.json","w") as f:
    json.dump(result, f, indent=2, default=str)

print(f"\n{'='*60}")
print(f"{'指标':30s} {'v12等权':>12s} {'ML模型':>12s}")
print(f"{'-'*30} {'-'*12} {'-'*12}")
for k in ["ret", "sharpe", "mdd", "wr", "n"]:
    label = {"ret":"年化收益","sharpe":"夏普","mdd":"最大回撤","wr":"胜率","n":"交易日"}[k]
    v12_v = ret_v12.get(k, 0)
    ml_v = ret_ml.get(k, 0)
    if isinstance(v12_v, float):
        pct = k in ("ret", "mdd")
        print(f"  {label:30s} {v12_v*100 if pct else v12_v:>12.2f}{'%' if pct else '':>2s} {ml_v*100 if pct else ml_v:>12.2f}{'%' if pct else ''}")
    else:
        print(f"  {label:30s} {v12_v:>12} {ml_v:>12}")

print(f"⏱ {time.time()-tt:.0f}s")
print(f"✅ 结果: output/backtest_v24_ml_detailed.json")
PYEOF
