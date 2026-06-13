"""
完整回测框架 - ML选股 + 交易成本 + 波动率控制 + 风控
"""
import os, sys, time, gc, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DATA_FACTORS

t0 = time.time()
print("=" * 60)
print("完整回测框架 v1")
print("=" * 60)

# ===== 加载数据 =====
panel = pd.read_parquet(os.path.join(DATA_FACTORS, "factor_panel_with_fwd.parquet"))
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
panel = panel[panel["trade_date"] >= "2018-01-01"].copy()

# 合并价格
prices = pd.read_parquet(os.path.join(DATA_FACTORS, "full_prices.parquet"))
prices["trade_date"] = pd.to_datetime(prices["trade_date"])
panel = panel.merge(prices, on=["ts_code","trade_date"], how="left")
del prices
print(f"面板: {len(panel):,} 条")

drop_cols = ["短期反转", "毛利率", "量比", "BP", "SP", "EP", "股息率", "流通市值"]
factor_cols = [c for c in panel.columns 
               if c not in ("ts_code","trade_date","fwd_20d_ret") + tuple(drop_cols)
               and panel[c].dtype in ("float64","int64")]
n_f = len(factor_cols)

import xgboost as xgb
import lightgbm as lgb
print(f"因子: {n_f} 个 | XGBoost: {xgb.__version__} | LightGBM: {lgb.__version__}")

# ===== 月频节点 =====
dates = sorted(panel["trade_date"].unique())
monthly = sorted(set(
    panel[panel["trade_date"].dt.to_period("M").isin(
        set(pd.Series(dates).dt.to_period("M").unique())
    )].groupby(panel["trade_date"].dt.to_period("M"))["trade_date"].max().tolist()
))
monthly = [d for d in monthly if d >= pd.Timestamp("2021-01-01")]
print(f"月频调仓: {len(monthly)} 个月 ({monthly[0].date()} ~ {monthly[-1].date()})")

# ===== 参数 =====
class Config:
    # 交易成本
    stamp_tax = 0.001       # 印花税 千分之一 (卖出)
    commission = 0.0002     # 佣金 万二
    slippage = 0.001        # 冲击成本 千分之一
    min_commission = 5.0    # 最低佣金 5元
    
    # 风控
    target_vol = 0.20       # 目标年化波动 20%
    max_pos_pct = 0.10      # 单只股票最大仓位 10%
    max_turnover = 0.50     # 单次最大换手率 50%
    stop_loss = 0.20        # 个股止损线 20%
    n_stocks = 100          # 持仓数量
    
    # ML
    train_years = 4         # 训练窗口 4年
    val_months = 6          # 验证窗口 6个月

cfg = Config()
print(f"\n参数:")
print(f"  目标波动: {cfg.target_vol*100:.0f}% | 持仓: {cfg.n_stocks}只")
print(f"  印花税: {cfg.stamp_tax*100:.1f}‰ | 佣金: {cfg.commission*100:.1f}‰ | 滑点: {cfg.slippage*100:.1f}‰")
print(f"  最大换手: {cfg.max_turnover*100:.0f}% | 止损: {cfg.stop_loss*100:.0f}%")

# ===== 全局统计量 =====
col_medians = np.nan_to_num(np.nanmedian(panel[factor_cols].values, axis=0), 0)

# ===== 滚动训练 + 回测 =====
test_months = monthly
all_preds = []

print(f"\n[训练] 滚动训练...")
for i, date in enumerate(test_months):
    train_start = date - pd.Timedelta(days=cfg.train_years*365)
    val_start = date - pd.Timedelta(days=cfg.val_months*30)
    
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
    p = (xgb_m.predict(X_te) + lgb_m.predict(X_te)) / 2
    
    codes = panel[panel["trade_date"] == date]["ts_code"].values
    for j, code in enumerate(codes):
        all_preds.append({"trade_date": date, "ts_code": code, "pred_ret": float(p[j])})
    
    if (i + 1) % 15 == 0:
        print(f"  [{i+1}/{len(test_months)}] train={len(train):,} val={len(val):,}", flush=True)

df_pred = pd.DataFrame(all_preds)
print(f"预测: {len(df_pred):,} 条")

# ===== 回测（含交易成本 + 波动率控制）=====
def backtest_with_costs(pred_df, name, cfg):
    pred_dates = sorted(pred_df["trade_date"].unique())
    
    # 状态变量
    positions = {}          # {code: shares} (按投入1元等权建仓)
    cash = 1.0
    nav = [1.0]
    dates_log = [pred_dates[0]]
    turnovers = []
    trade_costs = []
    monthly_rets = []
    
    for i, date in enumerate(pred_dates):
        # 当前持仓市值
        last_nav = nav[-1]
        current_day = panel[panel["trade_date"] == date]
        if current_day.empty:
            continue
        
        # 获取预测
        day_pred = pred_df[pred_df["trade_date"] == date].sort_values("pred_ret", ascending=False)
        if len(day_pred) < cfg.n_stocks:
            continue
        
        # 选股
        selected = set(day_pred.head(cfg.n_stocks)["ts_code"].values)
        
        # 计算当前持仓市值
        port_value = 0
        for code in list(positions.keys()):
            nd = current_day[current_day["ts_code"] == code]
            if not nd.empty:
                px = nd["close"].iloc[0]
                if px and px > 0:
                    port_value += positions[code] * px
            else:
                # 停牌，按原值保留
                pass
        
        total_value = port_value + cash
        
        # === 调仓 ===
        target_pos = {}
        target_per_stock = total_value * (1 - cfg.stamp_tax * 0.3) / cfg.n_stocks  # 略留现金
        
        for code in selected:
            nd = current_day[current_day["ts_code"] == code]
            if not nd.empty:
                px = nd["close"].iloc[0]
                if px and px > 0:
                    target_pos[code] = target_per_stock / px
        
        # 计算换手
        old_set = set(positions.keys())
        new_set = set(target_pos.keys())
        
        sell_value = sum(positions.get(c, 0) * (current_day[current_day["ts_code"] == c]["close"].iloc[0] if c in current_day["ts_code"].values else 0) for c in old_set - new_set)
        buy_value = sum(target_pos.get(c, 0) * (current_day[current_day["ts_code"] == c]["close"].iloc[0] if c in current_day["ts_code"].values else 0) for c in new_set - old_set)
        
        turnover = (sell_value + buy_value) / (2 * total_value) if total_value > 0 else 0

        # 换手率限制
        if turnover > cfg.max_turnover:
            scale = cfg.max_turnover / turnover
            # 按比例缩减调仓量
            sell_value *= scale
            buy_value *= scale
        
        # 计算交易成本
        cost_sell = sell_value * (cfg.commission + cfg.stamp_tax + cfg.slippage)
        cost_buy = buy_value * (cfg.commission + cfg.slippage)
        total_cost = cost_sell + cost_buy
        trade_costs.append(total_cost)
        turnovers.append(turnover)
        
        # 更新持仓
        # 先卖再买
        for code in (old_set - new_set):
            if code in positions:
                del positions[code]
        
        # 补新仓 - 简化处理，假设全在调仓日完成
        positions = target_pos
        cash = total_value - sum(target_pos.get(c, 0) * (current_day[current_day["ts_code"] == c]["close"].iloc[0]) for c in target_pos if c in current_day["ts_code"].values) - total_cost
               
        # 下次调仓时的收益
        if i >= len(pred_dates) - 1:
            break
        
        next_date = pred_dates[i + 1]
        next_day = panel[panel["trade_date"] == next_date]
        if next_day.empty:
            continue
        
        next_nav = 0
        for code, shares in positions.items():
            nd = next_day[next_day["ts_code"] == code]
            if not nd.empty:
                px = nd["close"].iloc[0]
                if px and px > 0:
                    next_nav += shares * px
        
        next_nav += cash
        
        # 波动率控制: 实际波动率调整
        if len(monthly_rets) >= 6:
            realized_vol = np.std(monthly_rets[-6:]) * np.sqrt(12)
            if realized_vol > 0:
                leverage = min(cfg.target_vol / realized_vol, 1.5)
            else:
                leverage = 1.0
        else:
            leverage = 1.0
        
        month_ret = (next_nav / total_value - 1) * leverage
        monthly_rets.append(month_ret)
        nav.append(nav[-1] * (1 + month_ret))
        dates_log.append(next_date)
    
    # ===== 统计 =====
    if len(monthly_rets) < 6:
        return None
    
    pnl = np.array(monthly_rets)
    cum = np.array(nav)
    tr = cum[-1] - 1
    ar = (cum[-1])**(12/len(pnl)) - 1 if len(pnl) > 0 else 0
    vol = np.std(pnl) * np.sqrt(12)
    sr = np.mean(pnl) / np.std(pnl) * np.sqrt(12) if np.std(pnl) > 0 else 0
    dd = np.maximum.accumulate(cum) - cum
    mdd = dd.max()
    wr = np.mean(pnl > 0)
    calmar = ar / mdd if mdd > 0 else 0
    avg_turnover = np.mean(turnovers) if turnovers else 0
    avg_cost_rate = np.sum(trade_costs) / (cum[-1] * len(trade_costs)) * 100 if trade_costs else 0
    
    return {
        "name": name,
        "总收益": f"{tr*100:.1f}%",
        "年化收益": f"{ar*100:.1f}%",
        "年化波动": f"{vol*100:.1f}%",
        "夏普比率": f"{sr:.2f}",
        "最大回撤": f"{mdd*100:.1f}%",
        "Calmar": f"{calmar:.2f}",
        "月胜率": f"{wr*100:.0f}%",
        "月均换手": f"{avg_turnover*100:.0f}%",
        "交易月数": len(pnl),
    }

# ===== 三种选股比例对比 =====
print(f"\n[回测] 含交易成本 + 波动率控制:", flush=True)

results = []
for top_pct in [0.02, 0.05, 0.10]:
    n_stocks = max(int(len(panel[panel["trade_date"] == test_months[0]]) * top_pct), 20)
    cfg.n_stocks = n_stocks
    
    result = backtest_with_costs(df_pred, f"Top {top_pct*100:.0f}%", cfg)
    if result:
        results.append(result)
        print(f"\n  {result['name']}: 持仓{n_stocks}只")
        for k, v in result.items():
            if k != "name":
                print(f"    {k}: {v}")

# 汇总
if results:
    print(f"\n{'='*60}")
    print("回测汇总")
    print(f"{'='*60}")
    df_summary = pd.DataFrame(results).set_index("name")
    print(df_summary.to_string())

print(f"\n总用时: {(time.time()-t0)/60:.1f} 分钟")
