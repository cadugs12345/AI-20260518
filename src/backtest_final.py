"""
最终回测 - 周频调仓 + fwd_20d_ret标签 + 完整交易成本
"""
import os, sys, time, gc, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DATA_FACTORS

t0 = time.time()
print("=" * 60)
print("最终回测 - 周频 + fwd_20d_ret")
print("=" * 60)

# ===== 加载 =====
panel = pd.read_parquet(os.path.join(DATA_FACTORS, "factor_panel_with_fwd.parquet"))
prices = pd.read_parquet(os.path.join(DATA_FACTORS, "full_prices.parquet"))

panel["trade_date"] = pd.to_datetime(panel["trade_date"])
prices["trade_date"] = pd.to_datetime(prices["trade_date"])
panel = panel[panel["trade_date"] >= "2018-01-01"].copy()
panel = panel.merge(prices, on=["ts_code","trade_date"], how="left")
del prices
gc.collect()
print(f"面板: {len(panel):,} 条")

# 因子筛选
drop_cols = ["短期反转", "毛利率", "量比", "BP", "SP", "EP", "股息率", "流通市值"]
factor_cols = [c for c in panel.columns 
               if c not in ("ts_code","trade_date","fwd_20d_ret","close") + tuple(drop_cols)
               and panel[c].dtype in ("float64","int64")]
n_f = len(factor_cols)

import xgboost as xgb
import lightgbm as lgb
print(f"因子: {n_f} 个")

# ===== 周频节点 (周三附近) =====
dates = sorted(panel["trade_date"].unique())
# 每5个交易日取一个（~每周）
weekly = dates[::5]  # 简单取每5天一个节点
weekly = [d for d in weekly if d >= pd.Timestamp("2019-01-01")]
print(f"周频节点: {len(weekly)} 周 ({weekly[0].date()} ~ {weekly[-1].date()})")

# ===== 参数 =====
stamp_tax = 0.001; commission = 0.0002; slippage = 0.001

# ===== 滚动训练预测 =====
print("\n[ML] 滚动训练...")

col_medians = np.nan_to_num(np.nanmedian(panel[factor_cols].values, axis=0), 0)
all_predictions = []

test_weeks = [d for d in weekly if d >= pd.Timestamp("2021-01-01")]

for i, date in enumerate(test_weeks):
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
        all_predictions.append({"trade_date": date, "ts_code": code, "pred_ret": float(p[j])})
    
    if (i+1) % 30 == 0:
        print(f"  [{i+1}/{len(test_weeks)}] train={len(train):,} val={len(val):,}")

df_pred = pd.DataFrame(all_predictions)
pred_weeks = sorted(df_pred["trade_date"].unique())
print(f"预测: {len(df_pred):,} 条, {len(pred_weeks)}周")

# ===== 无成本验证 =====
print(f"\n[验证] 选股能力 (无成本):")
for n_top in [50, 100, 200, 300]:
    rets = []
    for i, date in enumerate(pred_weeks):
        if i == len(pred_weeks)-1: break
        day = df_pred[df_pred["trade_date"] == date].sort_values("pred_ret", ascending=False)
        nxt = panel[panel["trade_date"] == pred_weeks[i+1]]
        top = day.head(n_top)
        rs = nxt[nxt["ts_code"].isin(top["ts_code"])]["fwd_20d_ret"].mean()
        if not np.isnan(rs):
            rets.append(rs)
    
    if rets:
        pnl = np.array(rets)
        # fwd_20d_ret对应20个交易日 ≈ 约4周
        factor = 52/4  # 年化因子
        sr = np.mean(pnl) / np.std(pnl) * np.sqrt(52/4)
        print(f"  Top {n_top}: 共{len(rets)}周 | 周均{np.mean(pnl)*100:.2f}% | 夏普{sr:.2f} | 胜率{np.mean(pnl>0)*100:.0f}%")

# ===== 含成本回测 =====
print(f"\n[回测] 含交易成本:")

def backtest_weekly(pred_df, n_stocks, name):
    weeks = sorted(pred_df["trade_date"].unique())
    
    holdings = {}
    cash = 1.0
    nav = 1.0
    nav_hist = [1.0]
    pnls = []
    
    for i, date in enumerate(weeks):
        if i == len(weeks)-1: break
        next_date = weeks[i+1]
        
        day = pred_df[pred_df["trade_date"] == date].sort_values("pred_ret", ascending=False)
        selected = set(day.head(n_stocks)["ts_code"].values)
        
        day_px = panel[panel["trade_date"] == date]
        nxt_px = panel[panel["trade_date"] == next_date]
        
        # 当前市值
        curr_port = 0
        for code, shares in list(holdings.items()):
            px = day_px[day_px["ts_code"] == code]["close"].values
            if len(px) > 0 and px[0] > 0:
                curr_port += shares * px[0]
        
        total = curr_port + cash
        
        # 调仓
        current_set = set(holdings.keys())
        sell_set = current_set - selected
        buy_set = selected - current_set
        
        # 卖出
        for code in sell_set:
            px = day_px[day_px["ts_code"] == code]["close"].values
            if len(px) > 0 and px[0] > 0:
                proceeds = holdings[code] * px[0]
                cost = proceeds * (commission + stamp_tax + slippage)
                cash += proceeds - cost
            del holdings[code]
        
        # 买入
        if buy_set:
            avail = cash * 0.98
            per_stock = avail / len(buy_set)
            for code in buy_set:
                px = day_px[day_px["ts_code"] == code]["close"].values
                if len(px) > 0 and px[0] > 0:
                    buy_cost = per_stock * (commission + slippage)
                    holdings[code] = (per_stock - buy_cost) / px[0]
                    cash -= per_stock
        
        # 下一期收益
        next_port = 0
        for code, shares in holdings.items():
            px = nxt_px[nxt_px["ts_code"] == code]["close"].values
            if len(px) > 0 and px[0] > 0:
                next_port += shares * px[0]
        
        next_total = next_port + cash
        ret = next_total / total - 1
        pnls.append(ret)
        nav *= (1 + ret)
        nav_hist.append(nav)
        
        if (i+1) % 50 == 0:
            print(f"  [{i+1}/{len(weeks)}] nav={nav:.4f}")
    
    if len(pnls) < 10:
        return None
    
    pnl = np.array(pnls)
    navs = np.array(nav_hist)
    
    tr = navs[-1] - 1
    factor = 52/4
    ar = (navs[-1])**(factor/len(pnl)) - 1
    vol = np.std(pnl) * np.sqrt(factor)
    sr = np.mean(pnl) / np.std(pnl) * np.sqrt(factor)
    dd = np.maximum.accumulate(navs) - navs
    mdd = dd.max()
    wr = np.mean(pnl > 0)
    calmar = ar / mdd if mdd > 0 else 0
    
    print(f"\n  {name} ({n_stocks}只):")
    print(f"    总收益: {tr*100:.1f}% | 年化: {ar*100:.1f}%")
    print(f"    年化波动: {vol*100:.1f}% | 夏普: {sr:.2f}")
    print(f"    最大回撤: {mdd*100:.1f}% | Calmar: {calmar:.2f}")
    print(f"    周胜率: {wr*100:.0f}% | {len(pnl)}笔交易")

for n_top in [50, 100, 200]:
    backtest_weekly(df_pred, n_top, f"Top {n_top}")

print(f"\n总用时: {(time.time()-t0)/60:.1f} 分钟")
