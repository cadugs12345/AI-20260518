"""
周频调仓回测 v6 - 正确周节点 + 多批滚动
"""
import os, sys, time, gc
import numpy as np
import pandas as pd

warnings = __import__('warnings')
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_FACTORS = "data/factors"
t0 = time.time()
print("=" * 60)
print("周频调仓回测 v6")
print("=" * 60)

panel = pd.read_parquet(os.path.join(DATA_FACTORS, "factor_panel_with_fwd.parquet"))
prices = pd.read_parquet(os.path.join(DATA_FACTORS, "full_prices.parquet"))
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
prices["trade_date"] = pd.to_datetime(prices["trade_date"])
panel = panel[panel["trade_date"] >= "2018-01-01"].copy()
panel = panel.merge(prices, on=["ts_code","trade_date"], how="left")
del prices; gc.collect()

drop_cols = ["短期反转", "毛利率", "量比", "BP", "SP", "EP", "股息率", "流通市值"]
factor_cols = [c for c in panel.columns 
               if c not in ("ts_code","trade_date","fwd_20d_ret","close") + tuple(drop_cols)
               and panel[c].dtype in ("float64","int64")]

import xgboost as xgb
import lightgbm as lgb

print("\n计算周节点...")
# 从2021-01-04开始（周一），每周取周一
all_dates = sorted(panel["trade_date"].unique())
# 取每周第一个交易日
weekly_dates = []
prev_week = None
for d in all_dates:
    iso = d.isocalendar()
    wk = (iso[0], iso[1])  # (year, week)
    if wk != prev_week:
        weekly_dates.append(d)
        prev_week = wk

weekly_dates = [d for d in weekly_dates if d >= pd.Timestamp("2021-01-01") and d <= pd.Timestamp("2026-05-15")]
print(f"  {len(all_dates)}个交易日 → {len(weekly_dates)}个周节点")

# 预构建价格表
price_map = {}
for d in weekly_dates:
    sub = panel[panel["trade_date"] == d][["ts_code","close"]].dropna()
    price_map[d] = dict(zip(sub["ts_code"], sub["close"]))

# ===== ML预测 =====
print(f"[ML] 滚动训练...")
pred_cache = os.path.join(DATA_FACTORS, "pred_weekly_v6.parquet")

if os.path.exists(pred_cache):
    df_pred = pd.read_parquet(pred_cache)
    pred_dates = sorted(df_pred["trade_date"].unique())
    print(f"  缓存加载: {len(df_pred):,} 条, {len(pred_dates)}周")
else:
    all_preds = []
    for i, date in enumerate(weekly_dates):
        train_start = date - pd.Timedelta(days=3*365)
        val_start = date - pd.Timedelta(days=180)
        
        train = panel[(panel["trade_date"] >= train_start) & (panel["trade_date"] < val_start)]
        val = panel[(panel["trade_date"] >= val_start) & (panel["trade_date"] < date)]
        
        train = train.dropna(subset=factor_cols + ["fwd_20d_ret"])
        val = val.dropna(subset=factor_cols + ["fwd_20d_ret"])
        train = train[train["fwd_20d_ret"].abs() < 0.5]
        val = val[val["fwd_20d_ret"].abs() < 0.5]
        
        if len(train) < 10000 or len(val) < 2000:
            continue
        
        X_tr = np.nan_to_num(train[factor_cols].values.astype(np.float32), nan=0)
        y_tr = train["fwd_20d_ret"].values.astype(np.float32)
        X_va = np.nan_to_num(val[factor_cols].values.astype(np.float32), nan=0)
        y_va = val["fwd_20d_ret"].values.astype(np.float32)
        
        xgb_m = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbosity=0, n_jobs=8, early_stopping_rounds=30)
        xgb_m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        
        lgb_m = lgb.LGBMRegressor(n_estimators=500, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbose=-1, n_jobs=8)
        lgb_m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(30)], eval_metric="mse")
        
        day = panel[panel["trade_date"] == date]
        X_te = np.nan_to_num(day[factor_cols].values.astype(np.float32), nan=0)
        p = (xgb_m.predict(X_te) + lgb_m.predict(X_te)) / 2
        
        for j, code in enumerate(day["ts_code"].values):
            all_preds.append({"trade_date": date, "ts_code": code, "pred_ret": float(p[j])})
        
        if (i+1) % 50 == 0:
            print(f"  [{i+1}/{len(weekly_dates)}] train={len(train):,}")
    
    df_pred = pd.DataFrame(all_preds)
    df_pred.to_parquet(pred_cache)
    pred_dates = sorted(df_pred["trade_date"].unique())
    print(f"  ML完成: {len(df_pred):,} 条, {len(pred_dates)}周")

# ===== 无成本验证（fwd_20d_ret）=====
print(f"\n[验证] 选股能力:")
for n in [30, 50, 100]:
    rets = []
    for date in pred_dates:
        day = df_pred[df_pred["trade_date"] == date].sort_values("pred_ret", ascending=False)
        top = set(day.head(n)["ts_code"].values)
        actual = panel[panel["trade_date"] == date]
        rr = actual[actual["ts_code"].isin(top)]["fwd_20d_ret"].mean()
        if not np.isnan(rr):
            rets.append(rr)
    if rets:
        pnl = np.array(rets)
        sr = np.mean(pnl) / np.std(pnl) * np.sqrt(13)
        print(f"  Top{n:3d}: 均值{np.mean(pnl)*100:+.2f}% 夏普{sr:.2f} 胜率{np.mean(pnl>0)*100:.0f}% {len(rets)}期")

# ===== 滚动回测 =====
print(f"\n[回测] 多批持有:")

def rolling_backtest(n_new_per_week, max_trade_days=20, label=""):
    """每周买入n只新票, 每只最多持有max_trade_days个交易日"""
    cash = 0.03
    # 持仓: list of {"date_bought": date, "codes": {code: shares}}
    batches = []
    navs = [1.0]
    
    for i, date in enumerate(pred_dates):
        if i >= len(pred_dates) - 1:
            break
        next_date = pred_dates[i+1]
        
        px_p = price_map.get(date, {})
        px_n = price_map.get(next_date, {})
        
        # 计算当前市值并清理过期持仓
        # 交易日计数 = i到bidx的周数（近似）
        port_val = 0
        new_batches = []
        for batch in batches:
            trade_days_held = sum(1 for d in pred_dates if batch["date"] <= d <= date) - 1
            if trade_days_held < max_trade_days:
                batch_val = 0
                batch_codes = {}
                for code, shares in batch["codes"].items():
                    px = px_p.get(code, 0)
                    if px > 0:
                        batch_val += shares * px
                        batch_codes[code] = shares
                if batch_codes:
                    new_batches.append({"date": batch["date"], "codes": batch_codes})
                    port_val += batch_val
        
        batches = new_batches
        total_val = port_val + cash
        
        # 买入新一批
        day_pred = df_pred[df_pred["trade_date"] == date].sort_values("pred_ret", ascending=False)
        top_codes = set(day_pred.head(n_new_per_week)["ts_code"].values)
        
        # 去重
        held = set()
        for batch in batches:
            held.update(batch["codes"].keys())
        new_codes = list(top_codes - held)[:n_new_per_week]
        
        if new_codes and cash > 0.001:
            available = cash * 0.97
            per = available / len(new_codes)
            bought = {}
            for code in new_codes:
                px = px_p.get(code, 0)
                if px > 0 and per > 0:
                    buy_cost = per * (0.0002 + 0.002)
                    shares = (per - buy_cost) / px
                    if shares > 0:
                        bought[code] = shares
            if bought:
                batches.append({"date": date, "codes": bought})
                cash -= per * len(bought)
        
        # 下期
        new_port = 0
        for batch in batches:
            for code, shares in batch["codes"].items():
                px = px_n.get(code, 0)
                if px > 0:
                    new_port += shares * px
        
        new_total = new_port + cash
        ret = new_total / total_val - 1 if total_val > 0 else 0
        navs.append(navs[-1] * (1 + ret))
    
    pnl = np.array(navs[1:]) / np.array(navs[:-1]) - 1
    nav_arr = np.array(navs)
    tr = nav_arr[-1] - 1
    n_years = len(pnl) / 52
    ar = nav_arr[-1] ** (1/n_years) - 1 if n_years > 0 else 0
    vol = np.std(pnl) * np.sqrt(52) if len(pnl) > 1 else 0
    sr = np.mean(pnl) / np.std(pnl) * np.sqrt(52) if np.std(pnl) > 0 else 0
    dd = np.maximum.accumulate(nav_arr) - nav_arr
    mdd = dd.max()
    wr = np.mean(pnl > 0)
    
    held_count = sum(len(b["codes"]) for b in batches)
    print(f"\n  {label} (新{n_new_per_week}/周 ≈ {n_new_per_week*5}票):")
    print(f"    总收益: {tr*100:.1f}% | 年化: {ar*100:.1f}%")
    print(f"    波动: {vol*100:.1f}% | 夏普: {sr:.2f} | 回撤: {mdd*100:.1f}%")
    print(f"    胜率: {wr*100:.0f}% | {len(pnl)}周 | 持仓: {held_count}票")
    return nav_arr

# 参数扫描: 每批30/50/100只, 持有20/40个交易日
for n in [30, 50]:
    for hold in [20, 40]:
        rolling_backtest(n, hold, f"持有{hold}日")

print(f"\n总用时: {(time.time()-t0)/60:.1f}分")
