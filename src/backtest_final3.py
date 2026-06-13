"""
最终回测 v3 - 修复正确的现金/持仓跟踪
"""
import os, sys, time, gc, json
import numpy as np
import pandas as pd

warnings = __import__('warnings')
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DATA_FACTORS

t0 = time.time()
print("=" * 60)
print("最终回测 v3 - 修复现金逻辑")
print("=" * 60)

# ===== 加载 =====
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

# 周频节点
dates = sorted(panel["trade_date"].unique())
weekly = dates[::5]
weekly = [d for d in weekly if d >= pd.Timestamp("2021-01-01")]

# 预构建价格表 (dict of dicts)
print("\n[索引] price_by_date...")
price_by_date = {}
for d in weekly:
    sub = panel[panel["trade_date"] == d][["ts_code","close"]].set_index("ts_code")["close"].to_dict()
    price_by_date[d] = sub
print(f"  {len(price_by_date)} 个节点")

# ===== 滚动预测 =====
print(f"\n[ML] 滚动训练...")

# 如果已有缓存就直接读
pred_cache = os.path.join(DATA_FACTORS, "pred_weekly_v3.parquet")
if os.path.exists(pred_cache):
    df_pred = pd.read_parquet(pred_cache)
    pred_weeks = sorted(df_pred["trade_date"].unique())
    print(f"  缓存加载: {len(df_pred):,} 条, {len(pred_weeks)}周")
else:
    all_preds = []
    for i, date in enumerate(weekly):
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
        
        if (i+1) % 40 == 0:
            print(f"  [{i+1}/{len(weekly)}] train={len(train):,}")
    
    df_pred = pd.DataFrame(all_preds)
    df_pred.to_parquet(pred_cache)
    pred_weeks = sorted(df_pred["trade_date"].unique())
    print(f"  ML完成: {len(df_pred):,} 条, {len(pred_weeks)}周")

# ===== 验证: 无成本 =====
print(f"\n[验证] 无成本:")
for n_top in [50, 100, 200, 300]:
    rets = []
    for i, date in enumerate(pred_weeks):
        if i == 0: continue
        day = df_pred[df_pred["trade_date"] == pred_weeks[i-1]].sort_values("pred_ret", ascending=False)
        top = set(day.head(n_top)["ts_code"].values)
        px_n = price_by_date.get(date, {})
        px_p = price_by_date.get(pred_weeks[i-1], {})
        rs = []
        for code in top:
            if code in px_p and code in px_n and px_p[code] > 0:
                rs.append(px_n[code] / px_p[code] - 1)
        if rs:
            rets.append(np.mean(rs))
    
    if rets:
        pnl = np.array(rets)
        ar = (1 + np.mean(pnl))**(52/len(pnl)*len(rets)) - 1  # 年化(周复利修正)
        sr = np.mean(pnl) / np.std(pnl) * np.sqrt(52)
        print(f"  Top {n_top}: {len(rets)}周 | 周均{np.mean(pnl)*100:.2f}% | 夏普{sr:.2f} | 胜率{np.mean(pnl>0)*100:.0f}%")

# ===== 含成本回测 (正确版) =====
print(f"\n[回测] 含交易成本:")
stamp_tax = 0.001; commission = 0.0002; slippage = 0.002

def backtest(n_stocks):
    # 初始化: 1元本金, 97%入市
    cash = 0.03
    holdings = {}  # code -> shares
    init_value = 0
    
    # 第一周买入
    first_date = pred_weeks[0]
    day = df_pred[df_pred["trade_date"] == first_date].sort_values("pred_ret", ascending=False)
    selected = set(day.head(n_stocks)["ts_code"].values)
    
    px_f = price_by_date.get(first_date, {})
    available = 0.97 / len(selected) if selected else 0
    for code in selected:
        px = px_f.get(code, 0)
        if px > 0:
            buy_cost = available * (commission + slippage)
            shares = (available - buy_cost) / px
            holdings[code] = shares
    
    navs = [1.0]
    pnls = []
    
    for i in range(1, len(pred_weeks)):
        date = pred_weeks[i]
        prev_date = pred_weeks[i-1]
        
        px_p = price_by_date.get(prev_date, {})
        px_n = price_by_date.get(date, {})
        
        # 上期收盘市值
        portfolio_val = sum(shares * px_p.get(code, 0) for code, shares in holdings.items())
        total_val = portfolio_val + cash
        
        # 本期收益 (持仓价值从 prev 价格到本期价格)
        # 注意: 这里是真正的市值变化
        port_val_now = sum(shares * px_n.get(code, 0) for code, shares in holdings.items())
        
        # 换仓
        day = df_pred[df_pred["trade_date"] == prev_date].sort_values("pred_ret", ascending=False)
        selected = set(day.head(n_stocks)["ts_code"].values)
        
        # 计算卖出
        total_sell_val = 0
        total_sell_cost = 0
        old_holdings = holdings.copy()
        holdings = {}
        
        for code, shares in old_holdings.items():
            px = px_p.get(code, 0)
            if px > 0:
                val = shares * px
                if code in selected:
                    # 保留
                    holdings[code] = shares
                else:
                    # 卖出
                    cost = val * (commission + stamp_tax + slippage)
                    total_sell_val += val
                    total_sell_cost += cost
        
        cash += total_sell_val - total_sell_cost
        
        # 买入新股
        current_codes = set(holdings.keys())
        buy_list = [c for c in selected if c not in current_codes]
        
        if buy_list and cash > 0.001:
            avail_per = (cash * 0.95) / len(buy_list)
            for code in buy_list:
                px = px_p.get(code, 0)
                if px > 0 and avail_per > 0:
                    buy_cost = avail_per * (commission + slippage)
                    shares = (avail_per - buy_cost) / px
                    holdings[code] = shares
                    cash -= avail_per
        
        # 本期净值
        new_port_val = sum(shares * px_n.get(code, 0) for code, shares in holdings.items())
        new_total = new_port_val + cash
        
        ret = new_total / total_val - 1
        pnls.append(ret)
        navs.append(navs[-1] * (1 + ret))
    
    pnl = np.array(pnls); navs_arr = np.array(navs)
    tr = navs_arr[-1] - 1
    n_years = len(pnl) / 52
    ar = navs_arr[-1] ** (1/n_years) - 1
    vol = np.std(pnl) * np.sqrt(52)
    sr = np.mean(pnl) / np.std(pnl) * np.sqrt(52)
    dd = np.maximum.accumulate(navs_arr) - navs_arr
    mdd = dd.max()
    wr = np.mean(pnl > 0)
    
    print(f"\n  Top {n_stocks}:")
    print(f"    总收益: {tr*100:.1f}% | 年化: {ar*100:.1f}%")
    print(f"    波动: {vol*100:.1f}% | 夏普: {sr:.2f} | 回撤: {mdd*100:.1f}%")
    print(f"    胜率: {wr*100:.0f}% | {len(pnl)}周")

for n in [50, 100, 200]:
    backtest(n)

print(f"\n总用时: {(time.time()-t0)/60:.1f}分")
