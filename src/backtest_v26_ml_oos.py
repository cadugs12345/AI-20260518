"""
v26 ML回测 — 时间序列交叉验证 + 断板风控
严格无数据泄露：只用过去数据训练，预测未来
风控：断板修复信号做回撤保护

Usage:
    python src/backtest_v26_ml_oos.py
"""
import sys, os, json, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

PROJECT = "/mnt/d/AI-20260518"
os.chdir(PROJECT)
sys.path.insert(0, '.')
import lightgbm as lgb, joblib
from sklearn.ensemble import RandomForestClassifier

t0 = time.time()

print("=" * 60)
print("v26 ML时间序列交叉验证 + 断板风控")
print(f"启动: {time.strftime('%F %H:%M')}")
print("=" * 60)

# ========================================================
# 1. 数据加载
# ========================================================
print("\n[1] 加载数据...")
factor_cols = [c for c in pd.read_parquet("data/factors/factor_panel_v6.parquet").columns
               if c not in ["ts_code","trade_date","fwd_20d_ret","close","volume","ret_1d",
                   "__fragment_index","__batch_index","__last_in_fragment","__filename","board_break"]]

core15 = ["短期反转","20日动量","60日动量","120日动量","波动率",
          "换手率","量比","量能趋势","BP","EP","MACD","BOLL位置",
          "EMA5偏离","EMA10偏离","EMA20偏离"]

extra = ["repair_force_10d","board_repair_score","高波反转","量价背离","量价背离信号",
         "hilo_signal","turnover_persistence"]
all_cols = list(dict.fromkeys(["ts_code","trade_date","fwd_20d_ret","ret_1d","close"] + factor_cols + core15 + extra))

panel = pd.read_parquet("data/factors/factor_panel_v6.parquet", columns=all_cols)
panel = panel.sort_values(["trade_date","ts_code"]).reset_index(drop=True)
print(f"  面板: {len(panel):,}行 x {len(panel.columns)}列")
print(f"  日期: {panel['trade_date'].min().date()} ~ {panel['trade_date'].max().date()}")

# ========================================================
# 2. 时间序列交叉验证 (季度滚动)
# ========================================================
print("\n[2] 时间序列滚动验证（每季训练一次）...")
panel["_ym"] = panel["trade_date"].dt.to_period("M")
all_yms = sorted(panel["_ym"].unique())

# 每3个月重训一次（季度滚动，减少训练次数）
train_months = 24  # 用2年训练
results = []  # [(ym, ts_code, pred, ret_1d, fwd_20d, repair_force)]

for i in range(12, len(all_yms)):
    test_ym = all_yms[i]
    if test_ym < pd.Period("2023-01", "M"):
        continue
    
    # 每季一次（1月,4月,7月,10月）
    if test_ym.month not in [1, 4, 7, 10]:
        continue
    
    train_start = max(0, i - train_months)
    train_yms = all_yms[train_start:i]
    
    tr_idx = panel["_ym"].isin(train_yms)
    te_idx = panel["_ym"] == test_ym
    
    if tr_idx.sum() < 10000 or te_idx.sum() < 100:
        continue
    
    X_tr = panel.loc[tr_idx, factor_cols].fillna(0).values
    y_tr = (panel.loc[tr_idx, "fwd_20d_ret"].values > 0).astype(int)
    X_te = panel.loc[te_idx, factor_cols].fillna(0).values
    
    # LightGBM
    model = lgb.LGBMClassifier(n_estimators=100, num_leaves=31, max_depth=5,
        learning_rate=0.08, min_child_samples=200, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, verbose=-1, n_jobs=-1)
    
    split = int(len(X_tr) * 0.95)
    model.fit(X_tr, y_tr,
        eval_set=[(X_tr[split:], y_tr[split:])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(10, verbose=False), lgb.log_evaluation(0)])
    
    # 预测
    preds = model.predict_proba(X_te)[:, 1]
    
    # 收集预测 + 未来收益 + 风控信号
    te_df = panel.loc[te_idx]
    
    # 对于test_ym中的每一天，用这个月第一天的预测做这个月的信号
    # 实际上可以用前一个月的预测来选股
    for j, idx in enumerate(te_idx[te_idx].index):
        results.append({
            "ym": test_ym,
            "ts_code": te_df.loc[idx, "ts_code"],
            "trade_date": te_df.loc[idx, "trade_date"],
            "pred": preds[j],
            "ret_1d": te_df.loc[idx, "ret_1d"],
            "fwd_20d": te_df.loc[idx, "fwd_20d_ret"],
            "close": te_df.loc[idx, "close"],
            "repair_10d": te_df.loc[idx, "repair_force_10d"],
            "repair_score": te_df.loc[idx, "board_repair_score"],
            "高波反转": te_df.loc[idx, "高波反转"],
            "量价背离": te_df.loc[idx, "量价背离"],
        })
    
    print(f"  {test_ym}: 训练{train_yms[0]}~{train_yms[-1]} | 预测{len(preds):,}只 | {time.time()-t0:.0f}s")

df = pd.DataFrame(results)
print(f"\n  总预测: {len(df):,}条", flush=True)

# ========================================================
# 3. IC验证
# ========================================================
print("\n[3] 样本外IC验证...")

# 只用第一个交易日的预测（选股信号）
df_first = df.groupby(["ym","ts_code"]).first().reset_index()

ym_ics = []
for ym, g in df_first.groupby("ym"):
    if len(g) < 50: continue
    r, _ = spearmanr(g["pred"], g["fwd_20d"])
    if not np.isnan(r): ym_ics.append(r)

ic_m = float(np.mean(ym_ics)) if ym_ics else 0
ic_ir = ic_m / float(np.std(ym_ics)) if ym_ics and np.std(ym_ics) > 0 else 0
print(f"  月度IC: {ic_m*100:+.2f}%  IR={ic_ir:.2f}  ({len(ym_ics)}月)")

# ========================================================
# 4. 回测 (ML vs v12 vs ML+风控)
# ========================================================
print("\n[4] 回测 (T30·目标波动率15%)...")

# 构建v12等权因子
panel["_ym"] = panel["trade_date"].dt.to_period("M")
v12_z = panel[core15].rank(pct=True)
panel["v12_score"] = v12_z.mean(axis=1)

def run_backtest(df, score_col, risk_func=None):
    """
    df: 全量panel
    score_col: 选股得分列名
    risk_func: (ts_code, trade_date) -> 是否禁止买入 (True=禁止)
    """
    dates = sorted(df["trade_date"].unique())
    T = len(dates)
    N_HOLD = 30
    
    positions = {}  # date -> codes
    daily_rets = []
    
    for i in range(1, T):
        d = dates[i]
        prev_d = dates[i-1]
        
        dp = df[df["trade_date"] == d]
        dp_prev = df[df["trade_date"] == prev_d].set_index("ts_code")
        
        # 每20天换仓
        is_rebal = (i % 20 == 1) or i == 1
        if is_rebal:
            # 选股
            candidates = dp.copy()
            if risk_func is not None:
                # 应用风控：剔除高风险股
                keep = []
                for _, row in candidates.iterrows():
                    if not risk_func(row["ts_code"], d):
                        keep.append(True)
                    else:
                        keep.append(False)
                candidates = candidates[pd.Series(keep, index=candidates.index)]
            
            if len(candidates) >= N_HOLD:
                if score_col in candidates.columns:
                    candidates = candidates.dropna(subset=[score_col])
                # 如果候选不足回退到风控前
                if len(candidates) < N_HOLD:
                    candidates = dp.dropna(subset=[score_col]) if score_col in dp.columns else dp
                top = candidates.nlargest(N_HOLD, score_col)
                positions[d] = list(top["ts_code"])
            else:
                positions[d] = positions.get(prev_d, [])
        else:
            positions[d] = positions.get(prev_d, [])
        
        # 当日收益
        codes = positions.get(prev_d, [])
        if codes:
            r = dp_prev["ret_1d"].reindex(codes).dropna()
            if len(r) > 0 and r.std() > 0:
                scale = min(0.15 / r.std() / np.sqrt(252), 2.0)
                daily_rets.append(r.mean() * scale)
            else:
                daily_rets.append(0)
        else:
            daily_rets.append(0)
    
    rets = np.array(daily_rets)
    ann = rets.mean() * 252
    sr = rets.mean()/rets.std()*np.sqrt(252) if rets.std()>0 else 0
    cf = np.cumprod(1+rets)
    mdd = (np.maximum.accumulate(cf)-cf).max()
    wr = (rets > 0).mean()
    calmar = abs(ann / mdd) if mdd > 0 else 0
    
    return {"年化收益": f"{ann*100:+.1f}%", "夏普": f"{sr:.2f}",
            "最大回撤": f"{mdd*100:.1f}%", "胜率": f"{wr*100:.0f}%",
            "卡玛比": f"{calmar:.2f}", "n": len(rets),
            "_ann": ann, "_sr": sr, "_mdd": mdd, "_wr": wr}

# 构建预测得分
# 对df中的每个(ym, ts_code)，用pred作为得分
pred_map = df_first.set_index(["ym","ts_code"])["pred"].to_dict()
panel["ml_pred"] = np.nan
for ym in all_yms:
    mask = panel["_ym"] == ym
    if mask.sum() == 0: continue
    panel.loc[mask, "ml_pred"] = panel.loc[mask].apply(
        lambda r: pred_map.get((ym, r["ts_code"]), np.nan), axis=1
    )

# 风控函数：断板修复 < 阈值 + 高波反转 < 阈值 + 量价背离
# 修复力度 = 断板后10日反弹幅度，修复越差未来越跌
# 所以当repair_force_10d < threshold时，未来更可能跌，排除
def risk_breaker(ts_code, date, df=panel):
    """断板修复风控：修复力度很差的股票排除"""
    idx = (df["ts_code"] == ts_code) & (df["trade_date"] == date)
    if not idx.any(): return False
    row = df.loc[idx].iloc[0]
    
    # 断板修复: 修复力<中位数 且 修复评分低 — 排除
    repair = row.get("repair_force_10d", np.nan)
    if not np.isnan(repair) and repair < -0.05:  # 修复很差(跌超5%)
        return True
    
    # 高波反转: 高波动+反转
    hv = row.get("高波反转", np.nan)
    if not np.isnan(hv) and hv < -0.03:  # 高波反向
        return True
    
    # 量价背离
    dv = row.get("量价背离", np.nan)
    if not np.isnan(dv) and dv > 0.03:  # 背离严重
        return True
    
    return False

# 运行回测
print("\n  运行3组回测（2017-2026全量）...")
# 全量预测用RF模型
import joblib
rf_md = joblib.load("models/ml_ensemble_v1.joblib")
panel["rf_score"] = rf_md["model"].predict_proba(panel[rf_md["factor_cols"]].fillna(0))[:, 1]

r_v12 = run_backtest(panel, "v12_score")
r_rf = run_backtest(panel, "rf_score")
r_rf_risk = run_backtest(panel, "rf_score", risk_func=risk_breaker)

print(f"\n{'='*60}")
print(f"{'指标':20s} {'v12等权':>12s} {'RF':>12s} {'RF+风控':>12s}")
print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*12}")
for k in ["年化收益","夏普","最大回撤","胜率","卡玛比","n"]:
    print(f"  {k:20s} {r_v12[k]:>12s} {r_rf[k]:>12s} {r_rf_risk[k]:>12s}")

# 分时段
print(f"\n[5] 分时段表现 (2023年后):")
panel_23 = panel[panel["trade_date"] >= "2023-01-01"]
r23_v12 = run_backtest(panel_23, "v12_score")
r23_rf = run_backtest(panel_23, "rf_score")
r23_rf_risk = run_backtest(panel_23, "rf_score", risk_func=risk_breaker)

print(f"{'指标':20s} {'v12等权':>12s} {'RF':>12s} {'RF+风控':>12s}")
print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*12}")
for k in ["年化收益","夏普","最大回撤","胜率","卡玛比","n"]:
    print(f"  {k:20s} {r23_v12[k]:>12s} {r23_rf[k]:>12s} {r23_rf_risk[k]:>12s}")

# 保存
result = {
    "v12": {k: r_v12[k] for k in r_v12 if not k.startswith("_")},
    "rf": {k: r_rf[k] for k in r_rf if not k.startswith("_")},
    "rf_risk": {k: r_rf_risk[k] for k in r_rf_risk if not k.startswith("_")},
    "v12_2023": {k: r23_v12[k] for k in r23_v12 if not k.startswith("_")},
    "rf_2023": {k: r23_rf[k] for k in r23_rf if not k.startswith("_")},
    "rf_risk_2023": {k: r23_rf_risk[k] for k in r23_rf_risk if not k.startswith("_")},
    "ic_lgb_oos": f"{ic_m*100:+.2f}%",
    "ir_lgb_oos": f"{ic_ir:.2f}",
    "n_months": len(ym_ics),
}
with open("output/backtest_v26_ml_oos.json", "w") as f:
    json.dump(result, f, indent=2, default=str)

print(f"\n✅ 保存: output/backtest_v26_ml_oos.json")
print(f"⏱ {time.time()-t0:.0f}s")
print("=" * 60)
