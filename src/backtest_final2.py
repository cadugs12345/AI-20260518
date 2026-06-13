"""
高效回测 - 基于已有预测, 全部预索引好
"""
import os, sys, time, gc
import numpy as np
import pandas as pd

warnings = __import__('warnings')
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DATA_FACTORS

t0 = time.time()
print("=" * 60)
print("高效回测 - 预索引版")
print("=" * 60)

# 读预测 (之前已保存)
panel = pd.read_parquet(os.path.join(DATA_FACTORS, "factor_panel_with_fwd.parquet"))
prices = pd.read_parquet(os.path.join(DATA_FACTORS, "full_prices.parquet"))

panel["trade_date"] = pd.to_datetime(panel["trade_date"])
prices["trade_date"] = pd.to_datetime(prices["trade_date"])
panel = panel[panel["trade_date"] >= "2018-01-01"].copy()
panel = panel.merge(prices, on=["ts_code","trade_date"], how="left")
del prices
gc.collect()

drop_cols = ["短期反转", "毛利率", "量比", "BP", "SP", "EP", "股息率", "流通市值"]
factor_cols = [c for c in panel.columns 
               if c not in ("ts_code","trade_date","fwd_20d_ret","close") + tuple(drop_cols)
               and panel[c].dtype in ("float64","int64")]

import xgboost as xgb
import lightgbm as lgb

# 周频节点
dates = sorted(panel["trade_date"].unique())
weekly = dates[::5]
weekly = [d for d in weekly if d >= pd.Timestamp("2021-01-01")]

# 预构建价格索引
print("\n[索引] 预构建 trade_date -> DataFrame...")
price_index = {}
prepared_panel = {}
for d in weekly:
    sub = panel[panel["trade_date"] == d][["ts_code","close"]].set_index("ts_code")
    price_index[d] = sub.to_dict()["close"]
print(f"  价格索引: {len(price_index)} 个节点")

# ===== 滚动预测 =====
print("\n[ML] 滚动训练...")
all_preds = []

for i, date in enumerate(weekly):
    train_start = date - pd.Timedelta(days=3*365)
    val_start = date - pd.Timedelta(days=180)
    
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
    
    if (i+1) % 40 == 0:
        xgb_t = xgb_m.best_iteration + 1 if xgb_m.best_iteration else xgb_m.n_estimators
        lgb_t = getattr(lgb_m, 'best_iteration_', lgb_m.n_estimators)
        print(f"  [{i+1}/{len(weekly)}] train={len(train):,} val={len(val):,} xgb={xgb_t} lgb={lgb_t}")

df_pred = pd.DataFrame(all_preds)
pred_weeks = sorted(df_pred["trade_date"].unique())
nweeks = len(pred_weeks)
print(f"\n预测完成: {len(df_pred):,} 条, {nweeks}周")

# ===== 无成本验证 =====
print(f"\n[验证] 无成本:")
for n_top in [50, 100, 200, 300]:
    rets = []
    for i, date in enumerate(pred_weeks):
        if i == 0: continue
        day = df_pred[df_pred["trade_date"] == pred_weeks[i-1]].sort_values("pred_ret", ascending=False)
        top = set(day.head(n_top)["ts_code"].values)
        # 用 fwd_20d_ret 做收益
        nxt = panel[panel["trade_date"] == date]
        rr = nxt[nxt["ts_code"].isin(top)]["fwd_20d_ret"].mean()
        if not np.isnan(rr):
            rets.append(rr)
    
    if rets:
        pnl = np.array(rets)
        sr = np.mean(pnl) / np.std(pnl) * np.sqrt(52/4)
        print(f"  Top {n_top}: {len(rets)}笔 | 周均{np.mean(pnl)*100:.2f}% | 夏普{sr:.2f} | 胜率{np.mean(pnl>0)*100:.0f}%")

# ===== 含成本回测 (字典索引版) =====
print(f"\n[回测] 含交易成本:")
stamp_tax = 0.001; commission = 0.0002; slippage = 0.001

def backtest_fast(n_stocks):
    holdings = {}     # code -> shares
    nav = 1.0
    navs = [1.0]
    pnls = []
    
    for i, date in enumerate(pred_weeks):
        if i == len(pred_weeks) - 1: break
        next_date = pred_weeks[i+1]
        
        day = df_pred[df_pred["trade_date"] == date].sort_values("pred_ret", ascending=False)
        selected = set(day.head(n_stocks)["ts_code"].values)
        
        px_this = price_index.get(date, {})
        px_next = price_index.get(next_date, {})
        
        # 当前市值
        curr_val = sum(shares * px_this.get(code, 0) for code, shares in holdings.items())
        total_val = curr_val + 0.97  # 97% allocation
        
        # 卖出
        sell_set = set(holdings.keys()) - selected
        for code in sell_set:
            px = px_this.get(code, 0)
            if px > 0:
                proceeds = holdings[code] * px
                cost = proceeds * (commission + stamp_tax + slippage)
                holdings[code] = max(holdings[code] * 0, 0)
        holdings = {k:v for k,v in holdings.items() if v > 1e-10}
        
        # 买入
        buy_set = selected - set(holdings.keys())
        avail = total_val * 0.97 - sum(shares * px_this.get(code, 0) for code, shares in holdings.items())
        if buy_set and avail > 0:
            per = avail * 0.95 / len(buy_set)
            for code in buy_set:
                px = px_this.get(code, 0)
                if px > 0:
                    buy_cost = per * (commission + slippage)
                    shares = (per - buy_cost) / px
                    holdings[code] = holdings.get(code, 0) + shares
        
        # 下期
        next_val = sum(shares * px_next.get(code, 0) for code, shares in holdings.items())
        ret = next_val / total_val - 1
        pnls.append(ret)
        nav *= (1 + ret)
        navs.append(nav)
    
    if len(pnls) < 10: return
    
    pnl = np.array(pnls); navs_arr = np.array(navs)
    tr = navs_arr[-1] - 1
    ar = navs_arr[-1] ** (13/len(pnl)) - 1
    vol = np.std(pnl) * np.sqrt(13)
    sr = np.mean(pnl) / np.std(pnl) * np.sqrt(13)
    dd = np.maximum.accumulate(navs_arr) - navs_arr
    mdd = dd.max()
    wr = np.mean(pnl > 0)
    
    print(f"\n  Top {n_stocks} (含成本):")
    print(f"    总收益: {tr*100:.1f}% | 年化: {ar*100:.1f}%")
    print(f"    波动: {vol*100:.1f}% | 夏普: {sr:.2f} | 回撤: {mdd*100:.1f}%")
    print(f"    胜率: {wr*100:.0f}% | {len(pnl)}周")

for n in [50, 100, 200]:
    backtest_fast(n)

print(f"\n总用时: {(time.time()-t0)/60:.1f}分")
