"""
v13 - 5日调仓 + 截面波动率 + daily_basic财务因子
"""
import os, sys, time, gc, pickle
import numpy as np
import pandas as pd

warnings = __import__('warnings')
warnings.filterwarnings("ignore")

DATA_FACTORS = "data/factors"
t0 = time.time()
print("="*60)
print("v13 - 5日调仓 + 截面波动率 + 财务因子")
print("="*60)

# ===== 加载v3面板 =====
panel = pd.read_parquet(os.path.join(DATA_FACTORS, "factor_panel_v3.parquet"))
prices = pd.read_parquet(os.path.join(DATA_FACTORS, "full_prices.parquet"))
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
prices["trade_date"] = pd.to_datetime(prices["trade_date"])
print(f"面板v3: {len(panel):,}行")

# ===== 预计算个股波动率 =====
print("预计算个股20日波动率（5日调仓用短窗口）...")
prices_sorted = prices.sort_values(['ts_code','trade_date']).copy()
prices_sorted['ret_1d'] = prices_sorted.groupby('ts_code')['close'].pct_change()
prices_sorted['vol_20d'] = prices_sorted.groupby('ts_code')['ret_1d'].transform(
    lambda x: x.rolling(20, min_periods=5).std())
prices_sorted['vol_20d_ann'] = prices_sorted['vol_20d'] * np.sqrt(244)

# ===== 因子列 =====
skip_cols = ("ts_code","trade_date","fwd_20d_ret","fwd_5d_ret","行业","行业_大类",
             "pe","pe_ttm","pb","ps","ps_ttm","dv_ratio","dv_ttm")  # 保留这些作为因子
factor_cols = [c for c in panel.columns if c not in skip_cols 
               and panel[c].dtype in ("float64","int64")]
# 手动排除不必要的
factor_cols = [c for c in factor_cols if c not in ("短期反转","毛利率","量比","BP","SP","EP","股息率","流通市值")]
# 加入pe/pb等财务因子
factor_cols += ["pe","pe_ttm","pb","ps","ps_ttm","dv_ratio","dv_ttm"]
print(f"因子数: {len(factor_cols)}")
print(f"因子: {factor_cols}")

# ===== 5日周期节点 =====
all_dates = sorted(panel["trade_date"].unique())
period_dates = [all_dates[i] for i in range(0, len(all_dates), 5) 
                if all_dates[i] >= pd.Timestamp("2021-01-01")]
print(f"5日周期节点: {len(period_dates)}")

# 价格+波动映射
price_map, vol_map = {}, {}
for d in period_dates:
    sub = prices[prices["trade_date"] == d][["ts_code","close"]].dropna()
    price_map[d] = dict(zip(sub["ts_code"], sub["close"]))
    v_sub = prices_sorted[prices_sorted["trade_date"] == d][["ts_code","vol_20d_ann"]].dropna()
    vol_map[d] = dict(zip(v_sub["ts_code"], v_sub["vol_20d_ann"]))

# ===== ML训练 =====
import xgboost as xgb, lightgbm as lgb

pred_path = os.path.join(DATA_FACTORS, "pred_5d_v13.pkl")
label_col = "fwd_5d_ret"

if os.path.exists(pred_path):
    with open(pred_path, 'rb') as f:
        pred = pickle.load(f)
    print(f"ML缓存: {len(pred):,}条, {pred['trade_date'].nunique()}期")
else:
    print("[ML] 训练预测（5日标签）...")
    all_preds = []
    for i, date in enumerate(period_dates):
        train_start = date - pd.Timedelta(days=3*365)
        val_start = date - pd.Timedelta(days=180)
        
        train = panel[(panel["trade_date"] >= train_start) & (panel["trade_date"] < val_start)]
        val = panel[(panel["trade_date"] >= val_start) & (panel["trade_date"] < date)]
        train = train.dropna(subset=factor_cols + [label_col])
        val = val.dropna(subset=factor_cols + [label_col])
        train = train[train[label_col].abs() < 0.5]
        val = val[val[label_col].abs() < 0.5]
        if len(train) < 10000 or len(val) < 2000: continue
        
        X_tr = np.nan_to_num(train[factor_cols].values.astype(np.float32), nan=0)
        y_tr = train[label_col].values.astype(np.float32)
        X_va = np.nan_to_num(val[factor_cols].values.astype(np.float32), nan=0)
        y_va = val[label_col].values.astype(np.float32)
        
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
        if (i+1) % 50 == 0: print(f"  [{i+1}/{len(period_dates)}] train={len(train):,}")
    
    pred = pd.DataFrame(all_preds)
    with open(pred_path, 'wb') as f:
        pickle.dump(pred, f)
    print(f"ML完成: {len(pred):,}条, {pred['trade_date'].nunique()}期")

pred["trade_date"] = pd.to_datetime(pred["trade_date"])
pred_dates = sorted(pred["trade_date"].unique())

# ===== 无成本验证 =====
print(f"\n无成本验证（5日标签）:")
for n in [30, 50, 100]:
    rets = []
    for d in pred_dates:
        day = pred[pred["trade_date"] == d].sort_values("pred_ret", ascending=False)
        top = set(day.head(n)["ts_code"].values)
        actual = panel[panel["trade_date"] == d]
        rr = actual[actual["ts_code"].isin(top)][label_col].mean()
        if not np.isnan(rr): rets.append(rr)
    if rets:
        pnl = np.array(rets)
        sr = np.mean(pnl)/np.std(pnl)*np.sqrt(49)  # 5日*49≈年化
        print(f"  Top{n:3d}: 均值{np.mean(pnl)*100:+.2f}% 夏普{sr:.2f} {len(rets)}期")

# ===== 含成本回测 =====
print(f"\n{'='*60}")
print("含成本 + 5日调仓 + 截面波动率")
print(f"{'='*60}")

STAMP, COMM, SLIP = 0.001, 0.0002, 0.002

def backtest_v13(n_stocks, target_vol=0.20, label=""):
    cash = 0.03
    holdings = {}
    navs = [1.0]
    
    for i, date in enumerate(pred_dates[:-1]):
        sell_date = pred_dates[i+1]
        px_buy = price_map.get(date, {})
        px_sell = price_map.get(sell_date, {})
        stock_vol = vol_map.get(date, {})
        
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
        
        # ---- 截面波动 ----
        day_pred = pred[pred["trade_date"] == date].sort_values("pred_ret", ascending=False)
        selected = list(day_pred.head(n_stocks)["ts_code"].values)
        
        selected_vols = [stock_vol.get(c, np.nan) for c in selected]
        selected_vols = [v for v in selected_vols if not np.isnan(v) and v > 0.01]
        
        if len(selected_vols) >= 5:
            median_vol = np.median(selected_vols)
            position_ratio = min(target_vol / median_vol, 1.0)
            position_ratio = max(position_ratio, 0.05)
        else:
            median_vol = np.nan
            position_ratio = 1.0
        
        # ---- 买入 ----
        if selected and cash > 0.001:
            available = cash * position_ratio * 0.98
            if available > 0.001:
                per = available / len(selected)
                for code in selected:
                    px = px_buy.get(code, 0)
                    if px > 0 and per > 0:
                        buy_cost = per * (COMM + SLIP)
                        bought = (per - buy_cost) / px
                        if bought > 0: holdings[code] = bought
                cash -= per * len(holdings)
        
        # ---- 收益 ----
        new_port = sum(shares * px_sell.get(c, 0) for c, shares in holdings.items())
        new_total = new_port + cash
        ret = new_total / total_val - 1 if total_val > 0 else 0
        navs.append(navs[-1] * (1 + ret))
        
        if i < 3 or (i+1) % 50 == 0:
            vol_str = f"{median_vol*100:.0f}%" if not np.isnan(median_vol) else "N/A"
            print(f"  p{i:4d} {str(date.date())}->{str(sell_date.date())} | "
                  f"持{len(holdings):3d} | 股波{vol_str:5s} | "
                  f"仓{position_ratio:.2f} | "
                  f"总{total_val:.3f}->{new_total:.3f} "
                  f"ret={ret*100:+6.2f}% nav={navs[-1]:.3f}")
    
    pnl = np.array(navs[1:]) / np.array(navs[:-1]) - 1
    nav_arr = np.array(navs)
    tr = nav_arr[-1] - 1
    n_years = len(pnl) / 49
    ar = nav_arr[-1] ** (1/n_years) - 1 if n_years > 0 and nav_arr[-1] > 0 else 0
    vol = np.std(pnl) * np.sqrt(49)
    sr = np.mean(pnl) / np.std(pnl) * np.sqrt(49) if np.std(pnl) > 0 else 0
    dd = np.maximum.accumulate(nav_arr) - nav_arr
    mdd = dd.max()
    wr = np.mean(pnl > 0)
    calmar = ar / mdd if mdd > 0 else 0
    
    print(f"\n  {label} Top{n_stocks} 目波{target_vol*100:.0f}%:")
    print(f"    总收益: {tr*100:.1f}% | 年化: {ar*100:.1f}%")
    print(f"    实际波动: {vol*100:.1f}% | 夏普: {sr:.2f} | 回撤: {mdd*100:.1f}%")
    print(f"    卡玛: {calmar:.2f} | 胜率: {wr*100:.0f}% | {len(pnl)}期")
    return nav_arr, pnl

# 参数扫描
for n in [30, 50]:
    for tv in [0.15, 0.20, 0.25, 0.30]:
        nav, pnl = backtest_v13(n, target_vol=tv, label=f"5d_Top{n}")
        print()

print(f"\n总用时: {(time.time()-t0)/60:.1f}分")
