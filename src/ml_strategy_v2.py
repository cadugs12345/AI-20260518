"""
ML v2 - 滚动训练 + 月频调仓 + 多模型集成
"""
import os, sys, time, gc, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DATA_FACTORS

t0 = time.time()
print("=" * 60)
print("ML 多因子选股 v2 - 滚动训练版")
print("=" * 60)

panel = pd.read_parquet(os.path.join(DATA_FACTORS, "factor_panel_with_fwd.parquet"))
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
panel = panel[panel["trade_date"] >= "2018-01-01"].copy()
print(f"面板: {len(panel):,} 条")

drop_cols = ["短期反转", "毛利率", "量比", "BP", "SP", "EP", "股息率", "流通市值"]
factor_cols = [c for c in panel.columns 
               if c not in ("ts_code","trade_date","fwd_20d_ret") + tuple(drop_cols)
               and panel[c].dtype in ("float64","int64")]
n_f = len(factor_cols)

import xgboost as xgb
import lightgbm as lgb
print(f"因子: {n_f} 个 | XGBoost: {xgb.__version__} | LightGBM: {lgb.__version__}")

# 月频节点
dates = sorted(panel["trade_date"].unique())
monthly = sorted(set(
    panel[panel["trade_date"].dt.to_period("M").isin(
        set(pd.Series(dates).dt.to_period("M").unique())
    )].groupby(panel["trade_date"].dt.to_period("M"))["trade_date"].max().tolist()
))
monthly = [d for d in monthly if d >= pd.Timestamp("2021-01-01")]
print(f"月频节点: {len(monthly)} 个 ({monthly[0].date()} ~ {monthly[-1].date()})")

# 全局中位数填充
col_medians = np.nanmedian(panel[factor_cols].values, axis=0)
col_medians = np.nan_to_num(col_medians, 0)

# 滚动训练 + 预测
test_months = monthly
all_preds = []

for i, date in enumerate(test_months):
    train_start = date - pd.Timedelta(days=4*365)
    val_start = date - pd.Timedelta(days=365)
    
    train = panel[(panel["trade_date"] >= train_start) & (panel["trade_date"] < val_start)].dropna(subset=factor_cols + ["fwd_20d_ret"])
    val = panel[(panel["trade_date"] >= val_start) & (panel["trade_date"] < date)].dropna(subset=factor_cols + ["fwd_20d_ret"])
    
    train = train[train["fwd_20d_ret"].abs() < 0.5]
    val = val[val["fwd_20d_ret"].abs() < 0.5]
    
    if len(train) < 10000 or len(val) < 2000:
        continue
    
    X_tr = np.nan_to_num(train[factor_cols].values.astype(np.float32), nan=0)
    y_tr = train["fwd_20d_ret"].values.astype(np.float32)
    X_va = np.nan_to_num(val[factor_cols].values.astype(np.float32), nan=0)
    y_va = val["fwd_20d_ret"].values.astype(np.float32)
    
    xgb_m = xgb.XGBRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5,
        random_state=42, verbosity=0, n_jobs=8,
        early_stopping_rounds=50
    )
    xgb_m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    
    lgb_m = lgb.LGBMRegressor(
        n_estimators=800, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0, min_child_samples=50,
        random_state=42, verbose=-1, n_jobs=8
    )
    lgb_m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(50)], eval_metric="mse")
    
    X_te = np.nan_to_num(panel[panel["trade_date"] == date][factor_cols].values.astype(np.float32), nan=0)
    
    p_xgb = xgb_m.predict(X_te)
    p_lgb = lgb_m.predict(X_te)
    ensemble = (p_xgb + p_lgb) / 2
    
    codes = panel[panel["trade_date"] == date]["ts_code"].values
    for j, code in enumerate(codes):
        all_preds.append({"trade_date": date, "ts_code": code, "pred_ret": float(ensemble[j])})
    
    if (i + 1) % 10 == 0:
        print(f"  [{i+1}/{len(test_months)}] train={len(train):,} val={len(val):,} xgb_trees={xgb_m.best_iteration+1 if xgb_m.best_iteration else xgb_m.n_estimators} lgb_trees={lgb_m.best_iteration_}", flush=True)
    
    del train, val, xgb_m, lgb_m; gc.collect()

df_pred = pd.DataFrame(all_preds)
print(f"预测: {len(df_pred):,} 条")

# 回测
pred_dates = sorted(df_pred["trade_date"].unique())
print(f"\n[回测] 结果:")

for top_pct in [0.1, 0.2, 0.3]:
    pnl = []
    cum = [1.0]
    
    for i, date in enumerate(pred_dates):
        if i == len(pred_dates) - 1:
            continue
        
        day = df_pred[df_pred["trade_date"] == date].sort_values("pred_ret", ascending=False)
        if len(day) < 100:
            continue
        
        n_top = max(int(len(day) * top_pct), 20)
        top = day.head(n_top)
        
        ndate = pred_dates[i + 1]
        nday = panel[panel["trade_date"] == ndate]
        if nday.empty:
            continue
        
        rets = []
        for _, row in top.iterrows():
            nd = nday[nday["ts_code"] == row["ts_code"]]
            if not nd.empty and not np.isnan(nd["fwd_20d_ret"].iloc[0]):
                rets.append(nd["fwd_20d_ret"].iloc[0])
        
        if rets:
            r = np.mean(rets)
            pnl.append(r)
            cum.append(cum[-1] * (1 + r))
    
    if not pnl:
        continue
    
    pnl = np.array(pnl)
    cum = np.array(cum)
    tr = cum[-1] - 1
    ar = (cum[-1])**(12/len(pnl)) - 1
    vol = np.std(pnl) * np.sqrt(12)
    sr = np.mean(pnl) / np.std(pnl) * np.sqrt(12)
    dd = np.maximum.accumulate(cum) - cum
    mdd = dd.max()
    wr = np.mean(pnl > 0)
    calmar = ar / mdd if mdd > 0 else 0
    
    print(f"\n  Top {top_pct*100:.0f}%: 总收益 {tr*100:.1f}% | 年化 {ar*100:.1f}% | 波动 {vol*100:.1f}% | 夏普 {sr:.2f} | 回撤 {mdd*100:.1f}% | Calmar {calmar:.2f} | 胜率 {wr*100:.0f}% | {len(pnl)}月")

df_pred.to_parquet(os.path.join(DATA_FACTORS, "ml_predictions_v2.parquet"), index=False)
print(f"\n总用时: {(time.time()-t0)/60:.1f} 分钟")
