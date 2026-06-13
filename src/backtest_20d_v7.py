"""
最终回测 v7 - 精确20个交易日持有期
- 每20个交易日调仓一次（与fwd_20d_ret标签完全对齐）
- ML预测 → 选TopN买入 → 持20日卖出
- 验证（无成本）和回测（含成本）使用相同时间框架
"""
import os, sys, time, gc
import numpy as np
import pandas as pd

warnings = __import__('warnings')
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_FACTORS = "data/factors"

t0 = time.time()
print("="*60)
print("最终回测 v7 - 20日调仓（精确对齐标签）")
print("="*60)

panel = pd.read_parquet(os.path.join(DATA_FACTORS, "factor_panel_with_fwd_v2.parquet"))
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

# ===== 20个交易日节点 =====
all_dates = sorted(panel["trade_date"].unique())
# 每20个交易日取一个调仓节点
period_dates = [all_dates[i] for i in range(0, len(all_dates), 20) 
                if all_dates[i] >= pd.Timestamp("2021-01-01")]
print(f"交易日: {len(all_dates)}, 20日周期节点: {len(period_dates)}")

# 价格表
price_map = {}
for d in period_dates:
    sub = panel[panel["trade_date"] == d][["ts_code","close"]].dropna()
    price_map[d] = dict(zip(sub["ts_code"], sub["close"]))

# ===== ML预测 =====
pred_cache = os.path.join(DATA_FACTORS, "pred_20d_v7.parquet")
if os.path.exists(pred_cache):
    df_pred = pd.read_parquet(pred_cache)
    df_pred["trade_date"] = pd.to_datetime(df_pred["trade_date"])
    print(f"ML缓存: {len(df_pred):,}条, {len(df_pred['trade_date'].unique())}期")
else:
    print("[ML] 滚动训练...")
    all_preds = []
    for i, date in enumerate(period_dates):
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
        
        if (i+1) % 30 == 0:
            print(f"  [{i+1}/{len(period_dates)}] train={len(train):,}")
    
    df_pred = pd.DataFrame(all_preds)
    df_pred.to_parquet(pred_cache)
    print(f"ML完成: {len(df_pred):,}条, {len(df_pred['trade_date'].unique())}期")

# ===== 无成本验证 =====
print(f"\n{'='*60}")
print("无成本验证（直接用fwd_20d_ret列）")
print(f"{'='*60}")
pred_dates = sorted(df_pred["trade_date"].unique())

for n in [30, 50, 100, 200]:
    rets = []
    for d in pred_dates:
        day = df_pred[df_pred["trade_date"] == d].sort_values("pred_ret", ascending=False)
        top = set(day.head(n)["ts_code"].values)
        actual = panel[panel["trade_date"] == d]
        rr = actual[actual["ts_code"].isin(top)]["fwd_20d_ret"].mean()
        if not np.isnan(rr):
            rets.append(rr)
    if rets:
        pnl = np.array(rets)
        # 20日 ≈ 1/13年
        sr = np.mean(pnl) / np.std(pnl) * np.sqrt(13)
        ar = (1+np.mean(pnl))**13 - 1
        print(f"  Top{n:3d}: 平均{np.mean(pnl)*100:+.2f}% 年化{ar*100:.1f}% 夏普{sr:.2f} 胜率{np.mean(pnl>0)*100:.0f}%  {len(rets)}期")

# ===== 真实回测（含成本）=====
print(f"\n{'='*60}")
print("含成本回测（买入→持20日卖出；成本=0.322%/边）")
print(f"{'='*60}")

STAMP = 0.001
COMM = 0.0002
SLIP = 0.002

def backtest_20d(n_stocks, label=""):
    cash = 0.03
    holdings = {}
    navs = [1.0]
    
    for i, date in enumerate(pred_dates[:-1]):
        # 这次买入的卖出日是下一个20日周期
        sell_date = pred_dates[i+1]
        
        px_buy = price_map.get(date, {})
        px_sell = price_map.get(sell_date, {})
        
        # 当前市值
        port_val = 0
        for code, shares in holdings.items():
            px = px_sell.get(code, 0)  # 用卖出日价格估值
            if px > 0:
                port_val += shares * px
        # 实际应该用买入日价格估值来算ret
        # 修正: 持仓用买入日的价格
        hold_val = 0
        for code, shares in holdings.items():
            px = px_buy.get(code, 0)
            if px > 0:
                hold_val += shares * px
        total_val = hold_val + cash
        
        # ---- 卖出 ----
        sell_proceeds = 0
        for code, shares in holdings.items():
            px = px_sell.get(code, 0)
            if px > 0:
                val = shares * px
                cost = val * (STAMP + COMM + SLIP)
                sell_proceeds += val - cost
        cash = cash + sell_proceeds
        holdings = {}
        
        # ---- 买入 ----
        day_pred = df_pred[df_pred["trade_date"] == date].sort_values("pred_ret", ascending=False)
        selected = list(day_pred.head(n_stocks)["ts_code"].values)
        
        if selected and cash > 0.001:
            available = cash * 0.98
            per = available / len(selected)
            for code in selected:
                px = px_buy.get(code, 0)
                if px > 0 and per > 0:
                    buy_cost = per * (COMM + SLIP)
                    bought = (per - buy_cost) / px
                    if bought > 0:
                        holdings[code] = bought
            cash -= per * len(holdings)
        
        # ---- 收益计算 ----
        new_port = 0
        for code, shares in holdings.items():
            px = px_sell.get(code, 0)
            if px > 0:
                new_port += shares * px
        new_total = new_port + cash
        
        ret = new_total / total_val - 1 if total_val > 0 else 0
        navs.append(navs[-1] * (1 + ret))
        
        if i < 3 or (i+1) % 20 == 0:
            print(f"  p{i:3d} {date.date()}->{sell_date.date()} | "
                  f"持仓{len(holdings)} | "
                  f"总{total_val:.3f}->{new_total:.3f} "
                  f"ret={ret*100:+.2f}% cash={cash:.3f}")
    
    pnl = np.array(navs[1:]) / np.array(navs[:-1]) - 1
    nav_arr = np.array(navs)
    tr = nav_arr[-1] - 1
    n_years = len(pnl) / 13
    ar = nav_arr[-1] ** (1/n_years) - 1 if n_years > 0 else 0
    vol = np.std(pnl) * np.sqrt(13)
    sr = np.mean(pnl) / np.std(pnl) * np.sqrt(13) if np.std(pnl) > 0 else 0
    dd = np.maximum.accumulate(nav_arr) - nav_arr
    mdd = dd.max()
    wr = np.mean(pnl > 0)
    calmar = ar / mdd if mdd > 0 else 0
    
    print(f"\n  {label} Top {n_stocks}:")
    print(f"    总收益: {tr*100:.1f}% | 年化: {ar*100:.1f}%")
    print(f"    年化波动: {vol*100:.1f}% | 夏普: {sr:.2f} | 回撤: {mdd*100:.1f}%")
    print(f"    卡玛: {calmar:.2f} | 胜率: {wr*100:.0f}% | {len(pnl)}期")
    return nav_arr, pnl

for n in [30, 50, 100]:
    backtest_20d(n, "含成本")

print(f"\n总用时: {(time.time()-t0)/60:.1f}分")
