"""
最终回测 v8 - 20日调仓 + 目标波动率控制
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
print("最终回测 v8 - 20日调仓 + 目标波动率")
print("="*60)

panel = pd.read_parquet(os.path.join(DATA_FACTORS, "factor_panel_with_fwd_v2.parquet"))
prices = pd.read_parquet(os.path.join(DATA_FACTORS, "full_prices.parquet"))
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
prices["trade_date"] = pd.to_datetime(prices["trade_date"])

factor_cols = [c for c in panel.columns 
               if c not in ("ts_code","trade_date","fwd_20d_ret")
               and panel[c].dtype in ("float64","int64")]
factor_cols = [c for c in factor_cols if c not in ("短期反转","毛利率","量比","BP","SP","EP","股息率","流通市值")]

import xgboost as xgb, lightgbm as lgb

# ===== 20日节点 =====
all_dates = sorted(panel["trade_date"].unique())
period_dates = [all_dates[i] for i in range(0, len(all_dates), 20) 
                if all_dates[i] >= pd.Timestamp("2021-01-01")]

price_map = {}
for d in period_dates:
    sub = prices[prices["trade_date"] == d][["ts_code","close"]].dropna()
    price_map[d] = dict(zip(sub["ts_code"], sub["close"]))

# ===== ML预测（使用缓存）=====
import pickle
pred_cache = os.path.join(DATA_FACTORS, "pred_20d_v8.pkl")

if os.path.exists(pred_cache):
    # 缓存不存在预测文件，用v7的
    pred = pd.read_parquet(os.path.join(DATA_FACTORS, "pred_20d_v7.parquet") if os.path.exists(os.path.join(DATA_FACTORS, "pred_20d_v7.parquet")) else 
                           os.path.join(DATA_FACTORS, "pred_20d_v7.parquet"))
    pred["trade_date"] = pd.to_datetime(pred["trade_date"])
    print(f"ML缓存: {len(pred):,}条")
else:
    print("[ML] 新训练...")
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
        if len(train) < 10000 or len(val) < 2000: continue
        
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
        if (i+1) % 30 == 0: print(f"  [{i+1}/{len(period_dates)}] train={len(train):,}")
    
    pred = pd.DataFrame(all_preds)
    pred.to_parquet(os.path.join(DATA_FACTORS, "pred_20d_v8.parquet"))
    print(f"ML完成: {len(pred):,}条")

pred["trade_date"] = pd.to_datetime(pred["trade_date"])
pred_dates = sorted(pred["trade_date"].unique())
print(f"节点: {len(pred_dates)}")

# ===== 目标波动率回测 =====
print(f"\n{'='*60}")
print("含成本回测 + 目标波动率控制")
print(f"{'='*60}")

STAMP, COMM, SLIP = 0.001, 0.0002, 0.002

def backtest_vol_target(n_stocks, target_vol=0.20, label=""):
    """
    目标波动率控制:
    - 每期用过去4期(12个月)收益的波动率估计未来波动
    - 目标杠杆 = target_vol / 估计波动
    - 杠杆上限2.0（最多加一倍杠杆）
    """
    cash = 0.03
    holdings = {}
    navs = [1.0]
    hist_rets = []  # 用于波动率估计
    
    for i, date in enumerate(pred_dates[:-1]):
        sell_date = pred_dates[i+1]
        px_buy = price_map.get(date, {})
        px_sell = price_map.get(sell_date, {})
        
        # 当前市值
        hold_val = sum(shares * px_buy.get(c, 0) for c, shares in holdings.items())
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
        
        # ---- 计算目标杠杆 ----
        if len(hist_rets) >= 4:
            est_vol = np.std(hist_rets[-4:]) * np.sqrt(13)  # 过去4期年化波动
            if est_vol > 0.01:
                leverage = min(target_vol / est_vol, 2.0)
            else:
                leverage = 1.0
        else:
            leverage = 1.0
        
        # ---- 买入 ----
        day_pred = pred[pred["trade_date"] == date].sort_values("pred_ret", ascending=False)
        selected = list(day_pred.head(n_stocks)["ts_code"].values)
        
        if selected and cash > 0.001:
            # 用杠杆调整买入金额
            available = cash * 0.98 * leverage
            per = available / len(selected)
            for code in selected:
                px = px_buy.get(code, 0)
                if px > 0 and per > 0:
                    buy_cost = per * (COMM + SLIP)
                    bought = (per - buy_cost) / px
                    if bought > 0:
                        holdings[code] = bought
            cash -= per * len(holdings)
            cash += cash * (leverage - 1)  # 杠杆部分从"借入"获得
        
        # ---- 收益计算 ----
        new_port = sum(shares * px_sell.get(c, 0) for c, shares in holdings.items())
        new_total = new_port + cash
        
        ret = new_total / total_val - 1 if total_val > 0 else 0
        navs.append(navs[-1] * (1 + ret))
        hist_rets.append(ret)
        
        if i < 3 or (i+1) % 15 == 0:
            print(f"  p{i:3d} {str(date.date())}->{str(sell_date.date())} | "
                  f"持{len(holdings)} | "
                  f"总{total_val:.3f}->{new_total:.3f} "
                  f"ret={ret*100:+6.2f}% 杠杆={leverage:.2f} nav={navs[-1]:.3f}")
    
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
    
    print(f"\n  {label} Top{n_stocks} 目标波动{target_vol*100:.0f}%:")
    print(f"    总收益: {tr*100:.1f}% | 年化: {ar*100:.1f}% | 实际波动: {vol*100:.1f}%")
    print(f"    夏普: {sr:.2f} | 回撤: {mdd*100:.1f}% | 卡玛: {calmar:.2f} | 胜率: {wr*100:.0f}%")
    print(f"    平均杠杆: {np.mean([min(target_vol/(np.std(hist_rets[max(0,i-4):i])*np.sqrt(13)+0.01) if i>=4 and np.std(hist_rets[max(0,i-4):i])>0 else 1.0, 2.0) for i in range(len(hist_rets))]):.2f}")
    return nav_arr

# 参数扫描
for n in [30, 50]:
    for tv in [0.20, 0.25, 0.30]:
        backtest_vol_target(n, target_vol=tv, label="含成本")

print(f"\n总用时: {(time.time()-t0)/60:.1f}分")
