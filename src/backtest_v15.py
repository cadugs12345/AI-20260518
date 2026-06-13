"""
v15 - 20日调仓 + 截面波动率 + 市场择时（系统性降仓）
用全市场指数判断当前市场状态，危险时降仓
"""
import os, sys, time, gc, pickle
import numpy as np
import pandas as pd

warnings = __import__('warnings')
warnings.filterwarnings("ignore")

DATA_FACTORS = "data/factors"
t0 = time.time()
print("="*60)
print("v15 - 20日调仓 + 截面波动率 + 市场择时")
print("="*60)

# ===== 加载数据 =====
panel = pd.read_parquet(os.path.join(DATA_FACTORS, "factor_panel_with_fwd_v2.parquet"))
prices = pd.read_parquet(os.path.join(DATA_FACTORS, "full_prices.parquet"))
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
prices["trade_date"] = pd.to_datetime(prices["trade_date"])
print(f"面板: {len(panel):,}行")

# ===== 预计算个股波动率 =====
print("预计算个股60日滚动波动率...")
prices_sorted = prices.sort_values(['ts_code','trade_date']).copy()
prices_sorted['ret_1d'] = prices_sorted.groupby('ts_code')['close'].pct_change()
prices_sorted['vol_60d'] = prices_sorted.groupby('ts_code')['ret_1d'].transform(
    lambda x: x.rolling(60, min_periods=20).std())
prices_sorted['vol_60d_ann'] = prices_sorted['vol_60d'] * np.sqrt(244)

# ===== 构建全市场指数（等权全A） =====
print("构建全市场指数...")
# 每日全市场等权收益率
daily_index = prices_sorted.groupby('trade_date').apply(
    lambda g: g['ret_1d'].mean(), include_groups=False).reset_index()
daily_index.columns = ['trade_date', 'market_ret']
daily_index = daily_index.sort_values('trade_date')

# 市场择时指标
# 1. 20日滚动波动率（全市场）
daily_index['market_vol_20d'] = daily_index['market_ret'].rolling(20, min_periods=5).std() * np.sqrt(244)

# 2. 全市场指数净值（从1开始）
daily_index['market_nav'] = (1 + daily_index['market_ret']).cumprod()

# 3. 120日收益率（趋势）
daily_index['trend_120d'] = daily_index['market_nav'] / daily_index['market_nav'].shift(120) - 1

# 4. 60日收益率
daily_index['trend_60d'] = daily_index['market_nav'] / daily_index['market_nav'].shift(60) - 1

# 5. 最大回撤
daily_index['mdd_60d'] = daily_index['market_nav'] / daily_index['market_nav'].rolling(60).max() - 1

# 6. 波动率突变（当前波动/过去60日中位数）
daily_index['vol_ratio'] = daily_index['market_vol_20d'] / daily_index['market_vol_20d'].rolling(60).median()

print(f"全市场指数: {len(daily_index)}行")
print(f"  最近: {daily_index.tail(1).iloc[0]['trade_date'].date()} "
      f"vol={daily_index.tail(1).iloc[0]['market_vol_20d']*100:.1f}% "
      f"trend={daily_index.tail(1).iloc[0]['trend_120d']*100:.1f}%")

# ===== 因子列 =====
factor_cols = [c for c in panel.columns 
               if c not in ("ts_code","trade_date","fwd_20d_ret","短期反转","毛利率","量比","BP","SP","EP","股息率","流通市值")
               and panel[c].dtype in ("float64","int64")]
print(f"因子数: {len(factor_cols)}")

# ===== 周期节点 =====
all_dates = sorted(panel["trade_date"].unique())
period_dates = [all_dates[i] for i in range(0, len(all_dates), 20) 
                if all_dates[i] >= pd.Timestamp("2021-01-01")]
print(f"20日周期节点: {len(period_dates)}")

# 价格+波动映射
price_map, vol_map = {}, {}
for d in period_dates:
    sub = prices[prices["trade_date"] == d][["ts_code","close"]].dropna()
    price_map[d] = dict(zip(sub["ts_code"], sub["close"]))
    v_sub = prices_sorted[prices_sorted["trade_date"] == d][["ts_code","vol_60d_ann"]].dropna()
    vol_map[d] = dict(zip(v_sub["ts_code"], v_sub["vol_60d_ann"]))

# 市场择时映射（每个调仓日获取市场状态）
market_cond = {}
for d in period_dates:
    m = daily_index[daily_index["trade_date"] == d]
    if len(m) > 0:
        m = m.iloc[0]
        market_cond[d] = {
            'vol': m['market_vol_20d'],
            'trend_60d': m['trend_60d'],
            'trend_120d': m['trend_120d'],
            'mdd_60d': m['mdd_60d'],
            'vol_ratio': m['vol_ratio'],
        }
    else:
        market_cond[d] = {'vol': 0.25, 'trend_60d': 0, 'trend_120d': 0, 'mdd_60d': 0, 'vol_ratio': 1}

# 打印市场状态概览
print("\n市场状态概览（每期末）:")
for i, d in enumerate(period_dates[::12]):
    mc = market_cond[d]
    print(f"  {d.date()}: vol={mc['vol']*100:.0f}% trend60={mc['trend_60d']*100:+.0f}% "
          f"mdd60={mc['mdd_60d']*100:.0f}% vr={mc['vol_ratio']:.1f}")

# ===== ML预测 =====
pred_path = os.path.join(DATA_FACTORS, "pred_20d_v15.pkl")

if os.path.exists(pred_path):
    with open(pred_path, 'rb') as f:
        pred = pickle.load(f)
    print(f"\nML缓存: {len(pred):,}条, {pred['trade_date'].nunique()}期")
else:
    print("\n[ML] 训练预测...")
    import xgboost as xgb, lightgbm as lgb
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
        if (i+1) % 15 == 0: print(f"  [{i+1}/{len(period_dates)}]")
    pred = pd.DataFrame(all_preds)
    with open(pred_path, 'wb') as f:
        pickle.dump(pred, f)
    print(f"ML完成: {len(pred):,}条")

pred["trade_date"] = pd.to_datetime(pred["trade_date"])
pred_dates = sorted(pred["trade_date"].unique())

# ===== 无成本 =====
print("\n无成本验证:")
for n in [30, 50]:
    rets = []
    for d in pred_dates:
        day = pred[pred["trade_date"] == d].sort_values("pred_ret", ascending=False)
        top = set(day.head(n)["ts_code"].values)
        actual = panel[panel["trade_date"] == d]
        rr = actual[actual["ts_code"].isin(top)]["fwd_20d_ret"].mean()
        if not np.isnan(rr): rets.append(rr)
    if rets:
        pnl = np.array(rets)
        sr = np.mean(pnl)/np.std(pnl)*np.sqrt(13)
        print(f"  Top{n:3d}: 均值{np.mean(pnl)*100:+.2f}% 夏普{sr:.2f}")

# ===== 含成本 + 市场择时 =====
print(f"\n{'='*60}")
print("含成本 + 市场择时")
print(f"{'='*60}")

STAMP, COMM, SLIP = 0.001, 0.0002, 0.002

def compute_market_weight(date):
    """根据市场状态计算权益仓位（0~1）"""
    mc = market_cond.get(date, None)
    if mc is None:
        return 1.0
    
    w = 1.0
    
    # 条件1：高波动降仓（全市场年化波动 > 30% → 减少仓位）
    vol = mc['vol']
    if vol > 0.35:
        w *= 0.3
    elif vol > 0.30:
        w *= 0.5
    elif vol > 0.25:
        w *= 0.7
    
    # 条件2：趋势下跌（60日收益 < -10% → 显著降仓）
    trend = mc['trend_60d']
    if trend < -0.20:
        w *= 0.2
    elif trend < -0.15:
        w *= 0.3
    elif trend < -0.10:
        w *= 0.5
    elif trend < -0.05:
        w *= 0.7
    
    # 条件3：近期最大回撤（60日回撤 > 15% → 降仓）
    mdd = mc['mdd_60d']
    if mdd < -0.25:
        w *= 0.2
    elif mdd < -0.20:
        w *= 0.4
    elif mdd < -0.15:
        w *= 0.6
    elif mdd < -0.10:
        w *= 0.8
    
    # 条件4：波动率突变（当前波动 > 过去中位数的2倍 → 异常市降仓）
    vr = mc['vol_ratio']
    if vr > 3.0:
        w *= 0.2
    elif vr > 2.0:
        w *= 0.5
    
    return max(w, 0.05)

def backtest_v15(n_stocks, target_vol=0.15, use_timing=True, label=""):
    cash = 0.03
    holdings = {}
    navs = [1.0]
    n_risk_reduced = 0
    
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
            pos_ratio = min(target_vol / median_vol, 1.0)
            pos_ratio = max(pos_ratio, 0.05)
        else:
            median_vol = np.nan
            pos_ratio = 1.0
        
        # ---- 市场择时 ----
        market_w = compute_market_weight(date) if use_timing else 1.0
        if market_w < 0.95:
            n_risk_reduced += 1
        
        final_ratio = pos_ratio * market_w
        final_ratio = min(final_ratio, 1.0)
        
        # ---- 买入 ----
        if selected and cash > 0.001:
            available = cash * final_ratio * 0.98
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
        
        if i < 3 or (i+1) % 15 == 0:
            v_str = f"{median_vol*100:.0f}%" if not np.isnan(median_vol) else "N/A"
            print(f"  p{i:3d} {str(date.date())} | "
                  f"持{len(holdings):3d} | 股波{v_str:5s} | "
                  f"仓{pos_ratio:.2f} | 市{market_w:.2f} | "
                  f"总{total_val:.3f}->{new_total:.3f} "
                  f"ret={ret*100:+6.2f}% nav={navs[-1]:.3f}")
    
    pnl = np.array(navs[1:]) / np.array(navs[:-1]) - 1
    nav_arr = np.array(navs)
    tr = nav_arr[-1] - 1
    n_years = len(pnl) / 13
    ar = nav_arr[-1] ** (1/n_years) - 1 if n_years > 0 and nav_arr[-1] > 0 else 0
    vol = np.std(pnl) * np.sqrt(13)
    sr = np.mean(pnl) / np.std(pnl) * np.sqrt(13) if np.std(pnl) > 0 else 0
    dd = np.maximum.accumulate(nav_arr) - nav_arr
    mdd = dd.max()
    wr = np.mean(pnl > 0)
    calmar = ar / mdd if mdd > 0 else 0
    
    print(f"\n  {label} Top{n_stocks} 目波{target_vol*100:.0f}%:")
    print(f"    总收益: {tr*100:.1f}% | 年化: {ar*100:.1f}%")
    print(f"    实际波动: {vol*100:.1f}% | 夏普: {sr:.2f} | 回撤: {mdd*100:.1f}%")
    print(f"    卡玛: {calmar:.2f} | 胜率: {wr*100:.0f}% | {len(pnl)}期")
    print(f"    择时降仓: {n_risk_reduced}/{len(pnl)}次")
    return nav_arr, pnl

# ===== 对比测试（有/无择时） =====
for n in [30]:
    for tv in [0.15, 0.20]:
        print(f"\n--- 有择时 ---")
        nav1, pnl1 = backtest_v15(n, target_vol=tv, use_timing=True, label=f"V15择时")
        print()
        
        print(f"--- 无择时（纯截面波动对照） ---")
        nav2, pnl2 = backtest_v15(n, target_vol=tv, use_timing=False, label=f"V15无择时")
        print()

print(f"\n总用时: {(time.time()-t0)/60:.1f}分")
