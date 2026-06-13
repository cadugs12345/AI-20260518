"""
v17回测 — 基于预警调整的优化版本
调整:
1. 降低衰减因子权重（60日动量×0.5, 市值×0.5, 20日动量×0.8）
2. 保持增强因子权重（波动率+换手率不变）
3. 用调整后权重做多因子合成 → ML

Usage:
    python src/backtest_v17.py [--no-train] [--quick]
"""
import sys, os, json, time, pickle
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config.settings import DATA_FACTORS

t0 = time.time()

print("="*60)
print("v17 回测 — 预警因子权重调整")
print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("="*60)

# 加载面板
print("\n[1] 加载数据...")
panel = pd.read_parquet(os.path.join(DATA_FACTORS, "factor_panel_with_fwd_v2.parquet"))
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
prices = pd.read_parquet(os.path.join(DATA_FACTORS, "full_prices.parquet"))
prices["trade_date"] = pd.to_datetime(prices["trade_date"])
print(f"  面板: {len(panel):,}行, 股票数: {panel['ts_code'].nunique():,}")
print(f"  时间: {panel['trade_date'].min().date()} ~ {panel['trade_date'].max().date()}")

# 使用调整后的因子权重
ALERTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "alerts")
weights_path = os.path.join(ALERTS_DIR, "adjusted_weights.json")

if os.path.exists(weights_path):
    with open(weights_path) as f:
        w_data = json.load(f)
    factor_weights = w_data["weights"]
    print(f"\n[2] 加载调整权重 ({len(factor_weights)}个因子)")
else:
    # 降权版（手动）
    print("\n[2] ⚠️ 无调权文件，使用手动降权...")
    factor_weights = {
        "60日动量": 0.08, "20日动量": 0.13, "市值": 0.08,
        "EMA20偏离": 0.06, "120日动量": 0.06, "换手率": 0.07,
        "EMA5偏离": 0.05, "波动率": 0.06,
        "RSI_24": 0.05, "MACD": 0.05, "OBV": 0.05,
        "BOLL位置": 0.05, "RSI_12": 0.05, "RSI_6": 0.05,
        "量能趋势": 0.05, "EMA10偏离": 0.05,
    }

# 多因子合成
print("\n[3] 多因子合成...")
factor_cols = [c for c in factor_weights.keys() if c in panel.columns]
print(f"  参与合成的因子: {len(factor_cols)}")

# 用调整权重加权合成得分
for f in factor_cols:
    panel[f"rank_{f}"] = panel.groupby("trade_date")[f].rank(pct=True, method="dense")

WEIGHT_COLS = [f"rank_{f}" for f in factor_cols]
panel["composite"] = sum(
    panel[f"rank_{f}"] * factor_weights[f] for f in factor_cols
)

# 保存合成因子供ML
composite_path = os.path.join(DATA_FACTORS, "composite_v17.pkl")
panel[["trade_date", "ts_code", "composite"]].to_pickle(composite_path)
print(f"  合成得分已保存: {composite_path}")

# ========== ML训练 ==========
print("\n[4] ML训练...")
USE_PRED_PATH = True
pred_path_v16 = os.path.join(DATA_FACTORS, "pred_20d_v16.pkl")
pred_path_v17 = os.path.join(DATA_FACTORS, "pred_20d_v17.pkl")

if os.path.exists(pred_path_v16):
    # 用v16的predictions作为参考
    print("  加载v16预测结果作为基准...")
    pred = pd.read_pickle(pred_path_v16)
    pred["trade_date"] = pd.to_datetime(pred["trade_date"])
else:
    # 简易ML
    print("  ⚠️ 无v16预测文件，训练简易模型...")
    
    from xgboost import XGBRegressor
    from sklearn.model_selection import TimeSeriesSplit
    
    factor_cols_all = [c for c in panel.columns 
                      if c not in ["ts_code","trade_date","fwd_20d_ret","fwd_5d_ret","composite"]
                      and panel[c].dtype in ("float64","int64")
                      and c != "均值"]
    
    # 过滤掉衍生字段
    factor_cols_all = [c for c in factor_cols_all if not c.startswith("rank_")]
    
    panel_ml = panel[["trade_date","ts_code","composite","fwd_20d_ret"] + factor_cols_all].copy()
    dates = sorted(panel_ml["trade_date"].unique())
    train_end = dates[int(len(dates) * 0.85)]
    test_start = dates[int(len(dates) * 0.85)]
    
    train = panel_ml[panel_ml["trade_date"] < train_end].dropna(subset=["fwd_20d_ret"] + factor_cols_all)
    test = panel_ml[panel_ml["trade_date"] >= test_start].dropna(subset=["fwd_20d_ret"] + factor_cols_all)
    
    print(f"  Train: {len(train):,} rows, Test: {len(test):,} rows")
    
    X_train = train[factor_cols_all].values.astype(np.float32)
    y_train = train["fwd_20d_ret"].values.astype(np.float32)
    X_test = test[factor_cols_all].values.astype(np.float32)
    
    model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        random_state=42, n_jobs=-1, verbosity=0)
    model.fit(X_train, y_train)
    
    # 预测
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    
    # 加入合成得分做ensemble
    train_pred = train[["trade_date","ts_code"]].copy()
    train_pred["pred_ret"] = pred_train * 0.7 + train["composite"].values * 0.3
    test_pred = test[["trade_date","ts_code"]].copy()
    test_pred["pred_ret"] = pred_test * 0.7 + test["composite"].values * 0.3
    
    pred = pd.concat([train_pred, test_pred], ignore_index=True)
    pred.to_pickle(pred_path_v17)
    USE_PRED_PATH = pred_path_v17

print("\n[5] 回测...")

# 回测参数
STAMP, COMM, SLIP = 0.001, 0.0002, 0.002
N_STOCKS = 30
TARGET_VOL = 0.15

if isinstance(USE_PRED_PATH, str) and os.path.exists(USE_PRED_PATH):
    pred = pd.read_pickle(USE_PRED_PATH)
    pred["trade_date"] = pd.to_datetime(pred["trade_date"])

pred_dates = sorted(pred["trade_date"].unique())
print(f"  交易日: {len(pred_dates)}")

# 预计算波动率
prices_sorted = prices.sort_values(['ts_code','trade_date']).copy()
prices_sorted['ret_1d'] = prices_sorted.groupby('ts_code')['close'].pct_change()
prices_sorted['vol_60d'] = prices_sorted.groupby('ts_code')['ret_1d'].transform(
    lambda x: x.rolling(60, min_periods=20).std())
prices_sorted['vol_60d_ann'] = prices_sorted['vol_60d'] * np.sqrt(244)

vol_map, price_map = {}, {}
for d in pred_dates:
    sub = prices[prices["trade_date"] == d][["ts_code","close"]].dropna()
    price_map[d] = dict(zip(sub["ts_code"], sub["close"]))
    v_sub = prices_sorted[prices_sorted["trade_date"] == d][["ts_code","vol_60d_ann"]].dropna()
    vol_map[d] = dict(zip(v_sub["ts_code"], v_sub["vol_60d_ann"]))

results = {}
for n_stocks, target_vol in [(30, 0.15), (30, 0.20), (50, 0.15), (50, 0.20)]:
    cash = 0.03
    holdings = {}
    navs = [1.0]
    dates = []
    
    for i, date in enumerate(pred_dates[:-1]):
        sell_date = pred_dates[i+1]
        px_buy = price_map.get(date, {})
        px_sell = price_map.get(sell_date, {})
        stock_vol = vol_map.get(date, {})
        
        hold_val = sum(shares * px_buy.get(c, 0) for c, shares in holdings.items())
        total_val = hold_val + cash
        
        # 卖出
        sell_proceeds = 0
        for code, shares in list(holdings.items()):
            px = px_sell.get(code, 0)
            if px > 0:
                val = shares * px
                cost = val * (STAMP + COMM + SLIP)
                sell_proceeds += val - cost
        cash += sell_proceeds
        holdings = {}
        
        # 选股
        day_pred = pred[pred["trade_date"] == date].sort_values("pred_ret", ascending=False)
        selected = list(day_pred.head(n_stocks)["ts_code"].values)
        
        # 波动率控制
        selected_vols = [stock_vol.get(c, np.nan) for c in selected]
        selected_vols = [v for v in selected_vols if not np.isnan(v) and v > 0.01]
        
        if len(selected_vols) >= 5:
            median_vol = np.median(selected_vols)
            pos_ratio = min(target_vol / median_vol, 1.0)
            pos_ratio = max(pos_ratio, 0.05)
        else:
            pos_ratio = 1.0
        
        # 买入
        if selected and cash > 0.001:
            available = cash * pos_ratio * 0.98
            if available > 0.001:
                per = available / len(selected)
                for code in selected:
                    px = px_buy.get(code, 0)
                    if px > 0 and per > 0:
                        buy_cost = per * (COMM + SLIP)
                        bought = (per - buy_cost) / px
                        if bought > 0:
                            holdings[code] = bought
                cash -= per * len(holdings)
        
        new_port = sum(shares * px_sell.get(c, 0) for c, shares in holdings.items())
        new_total = new_port + cash
        ret = new_total / total_val - 1 if total_val > 0 else 0
        navs.append(navs[-1] * (1 + ret))
        dates.append(date)
    
    # 回测指标计算
    nav_array = np.array(navs)
    pnl = nav_array[1:] / nav_array[:-1] - 1
    sr = np.mean(pnl) / np.std(pnl) * np.sqrt(13) if np.std(pnl) > 0 else 0
    tr = nav_array[-1] - 1
    dd = np.maximum.accumulate(nav_array) - nav_array
    mdd = dd.max()
    calmar = tr / mdd if mdd > 0 else 0
    wr = np.mean(pnl > 0)
    
    version_name = f"T{n_stocks}_V{int(target_vol*100)}"
    results[version_name] = {
        "navs": nav_array,
        "dates": dates,
        "total_return": tr,
        "sharpe": sr,
        "max_dd": mdd,
        "calmar": calmar,
        "win_rate": wr,
        "mean_ret": np.mean(pnl),
        "std_ret": np.std(pnl),
    }
    
    print(f"\n  {version_name}")
    print(f"    总收益: {tr*100:.1f}%")
    print(f"    年化: {((1+tr)**(12/len(pnl))-1)*100:.1f}%")
    print(f"    夏普: {sr:.2f}")
    print(f"    最大回撤: {mdd*100:.1f}%")
    print(f"    卡玛: {calmar:.2f}")
    print(f"    胜率: {wr*100:.0f}%")

# 保存
out = {}
for k, v in results.items():
    out[k] = {kk: vv for kk, vv in v.items() if kk not in ("navs", "dates")}
    out[k]["date_range"] = f"{v['dates'][0].date()}~{v['dates'][-1].date()}"

out_path = os.path.join(DATA_FACTORS, "backtest_v17_results.json")
with open(out_path, "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print(f"\n✅ v17结果已保存: {out_path}")
print(f"总用时: {(time.time()-t0)/60:.1f}分钟")
