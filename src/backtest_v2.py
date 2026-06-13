"""
完整回测框架 v2 - 修复调仓bug + 精确标签
"""
import os, sys, time, gc, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DATA_FACTORS, DATA_RAW

t0 = time.time()
print("=" * 60)
print("完整回测框架 v2 - 修复调仓逻辑")
print("=" * 60)

# ===== 加载数据 =====
panel = pd.read_parquet(os.path.join(DATA_FACTORS, "factor_panel_with_fwd.parquet"))
prices = pd.read_parquet(os.path.join(DATA_FACTORS, "full_prices.parquet"))

panel["trade_date"] = pd.to_datetime(panel["trade_date"])
prices["trade_date"] = pd.to_datetime(prices["trade_date"])
panel = panel[panel["trade_date"] >= "2018-01-01"].copy()

# 合并价格
panel = panel.merge(prices, on=["ts_code","trade_date"], how="left")
del prices
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

# ===== 精确标签: 从当前月最后交易日 到 下个月最后交易日的实际收益 =====
print("\n[标签] 构建精确月频标签...")

dates = sorted(panel["trade_date"].unique())
# 取每月最后交易日
monthly_dates = sorted(set(
    panel[panel["trade_date"].dt.to_period("M").isin(
        set(pd.Series(dates).dt.to_period("M").unique())
    )].groupby(panel["trade_date"].dt.to_period("M"))["trade_date"].max().tolist()
))
# 过滤掉太早的
monthly_dates = [d for d in monthly_dates if d >= pd.Timestamp("2017-02-01")]

# 构建月频收益: 当月close / 前一个月close - 1
date_to_next = {d: monthly_dates[i+1] for i, d in enumerate(monthly_dates[:-1])}

# 构建上月收盘价映射
prev_close_map = {}
for i, date in enumerate(monthly_dates):
    if i == 0: continue
    prev_date = monthly_dates[i-1]
    prev_closes = panel[panel["trade_date"] == prev_date][["ts_code","close"]].copy()
    prev_closes.columns = ["ts_code", "prev_close"]
    prev_close_map[date] = prev_closes

# 给 panel 加月频收益列
panel["monthly_ret"] = np.nan
panel["next_month_date"] = None

for date in monthly_dates:
    if date not in date_to_next:
        continue
    next_date = date_to_next[date]
    
    # 本月价格
    curr = panel[panel["trade_date"] == date][["ts_code","close"]].copy()
    if curr.empty:
        continue
    
    # 下月价格
    nxt = panel[panel["trade_date"] == next_date][["ts_code","close"]].copy()
    if nxt.empty:
        continue
    
    merged = curr.merge(nxt, on="ts_code", suffixes=("", "_next"), how="inner")
    merged["monthly_ret"] = merged["close_next"] / merged["close"] - 1
    merged["next_month_date"] = next_date
    
    # 写回panel
    ret_dict = dict(zip(merged["ts_code"], merged["monthly_ret"]))
    next_dict = dict(zip(merged["ts_code"], merged["next_month_date"]))
    
    mask = panel["trade_date"] == date
    panel.loc[mask, "monthly_ret"] = panel.loc[mask, "ts_code"].map(ret_dict)
    panel.loc[mask, "next_month_date"] = panel.loc[mask, "ts_code"].map(next_dict)

# 筛选有标签的数据
labeled = panel.dropna(subset=["monthly_ret"]).copy()
labeled = labeled[labeled["monthly_ret"].abs() < 0.5].copy()

# 月频调仓节点 (从2021年开始测试)
test_dates = sorted(labeled[labeled["trade_date"] >= "2021-01-01"]["trade_date"].unique())
print(f"标签构建完成: 共{len(test_dates)}个调仓节点")

# ===== 参数 =====
stamp_tax = 0.001      # 印花税 千分之一
commission = 0.0002    # 佣金 万二
slippage = 0.001       # 滑点 千分之一

print(f"\n参数: 印花税{stamp_tax*100:.1f}‰ | 佣金{commission*100:.1f}‰ | 滑点{slippage*100:.1f}‰")

# ===== 全局统计量 =====
col_medians = np.nan_to_num(np.nanmedian(labeled[factor_cols].values, axis=0), 0)

# ===== 更精确的回测函数 =====
def precise_backtest(pred_df, n_stocks, name="策略"):
    """给定预测 df (trade_date, ts_code, pred_ret), 执行含成本的月频回测"""
    dates = sorted(pred_df["trade_date"].unique())
    
    # 状态
    holdings = {}  # code -> shares
    cash = 1.0     # 初始单位资金
    nav = 1.0
    nav_history = [1.0]
    pnl_history = []
    turnover_history = []
    
    for i, date in enumerate(dates):
        if date not in date_to_next:
            continue
        next_date = date_to_next[date]
        
        # 当前日期数据
        day_data = labeled[labeled["trade_date"] == date]
        next_day_data = labeled[labeled["trade_date"] == next_date]
        if day_data.empty or next_day_data.empty:
            continue
        
        # 当前预测
        day_pred = pred_df[pred_df["trade_date"] == date].sort_values("pred_ret", ascending=False)
        selected = day_pred.head(n_stocks)["ts_code"].values
        
        # 计算当前持仓市值
        port_val = 0
        valid_holdings = {}
        for code, shares in holdings.items():
            px_row = day_data[day_data["ts_code"] == code]
            if not px_row.empty and px_row["close"].iloc[0] > 0:
                px = px_row["close"].iloc[0]
                val = shares * px
                port_val += val
                valid_holdings[code] = shares
        
        total_val = port_val + cash
        
        # === 确定目标持仓 ===
        target_codes = set(selected)
        current_codes = set(valid_holdings.keys())
        
        sell_codes = current_codes - target_codes
        keep_codes = current_codes & target_codes
        buy_codes = target_codes - current_codes
        
        # 卖出
        sell_proceeds = 0
        for code in sell_codes:
            px_row = day_data[day_data["ts_code"] == code]
            if not px_row.empty:
                px = px_row["close"].iloc[0]
                sell_proceeds += valid_holdings[code] * px
            del valid_holdings[code]
        
        # 交易成本 (卖出)
        sell_cost = sell_proceeds * (commission + stamp_tax + slippage)
        cash += (sell_proceeds - sell_cost)
        
        # 可用资金
        available = cash * 0.98  # 留2%现金
        
        # 买入
        if buy_codes and available > 0:
            per_stock = available / len(buy_codes)
            for code in buy_codes:
                px_row = day_data[day_data["ts_code"] == code]
                if not px_row.empty and px_row["close"].iloc[0] > 0:
                    px = px_row["close"].iloc[0]
                    shares = per_stock / px * (1 - commission - slippage)  # 扣买入成本
                    valid_holdings[code] = shares
                    cash -= per_stock
        
        # 计算换手率
        total_traded = sell_proceeds + (available)
        avg_turnover = total_traded / (2 * max(total_val, 1e-6)) if total_val > 0 else 0
        turnover_history.append(min(avg_turnover, 1.0))
        
        # 卖出总交易成本
        buy_cost = available * (commission + slippage)
        total_cost = sell_cost + buy_cost
        
        # === 到下个月底的收益 ===
        next_port_val = 0
        for code, shares in valid_holdings.items():
            px_row = next_day_data[next_day_data["ts_code"] == code]
            if not px_row.empty:
                px = px_row["close"].iloc[0]
                if px > 0:
                    next_port_val += shares * px
        
        next_total = next_port_val + cash
        
        # 月度收益
        month_ret = next_total / total_val - 1
        pnl_history.append(month_ret)
        nav *= (1 + month_ret)
        nav_history.append(nav)
        
        # 更新持仓和现金
        holdings = valid_holdings
        cash = cash  # 已经更新
        
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(dates)}] nav={nav:.4f}")
    
    # 统计
    if len(pnl_history) < 6:
        return None
    
    pnl = np.array(pnl_history)
    navs = np.array(nav_history)
    tr = navs[-1] - 1
    ar = (navs[-1])**(12/len(pnl)) - 1
    vol = np.std(pnl) * np.sqrt(12)
    sr = np.mean(pnl) / np.std(pnl) * np.sqrt(12)
    dd = np.maximum.accumulate(navs) - navs
    mdd = dd.max()
    wr = np.mean(pnl > 0)
    calmar = ar / mdd if mdd > 0 else 0
    avg_to = np.mean(turnover_history)
    sharpe_annualized = np.mean(pnl) / np.std(pnl) * np.sqrt(12)
    
    print(f"\n  {name} ({n_stocks}只):")
    print(f"  总收益: {tr*100:.1f}% | 年化: {ar*100:.1f}%")
    print(f"  年化波动: {vol*100:.1f}% | 夏普: {sr:.2f}")
    print(f"  最大回撤: {mdd*100:.1f}% | Calmar: {calmar:.2f}")
    print(f"  月胜率: {wr*100:.0f}% | 月均换手: {avg_to*100:.0f}% | {len(pnl)}个月")
    
    return {
        "name": name, "n_stocks": n_stocks,
        "total_return": tr, "annual_return": ar,
        "volatility": vol, "sharpe": sr,
        "max_drawdown": mdd, "calmar": calmar,
        "win_rate": wr, "avg_turnover": avg_to,
        "n_months": len(pnl)
    }

# ===== 先用清洗后的快速方法验证选股能力 =====
print(f"\n[验证] 选股能力 (无成本, 精确月频标签):")

results = []

for n_top in [50, 100, 200, 300]:
    rets = []
    dates_for_test = labeled["trade_date"].unique()
    
    for date in test_dates:
        if date not in date_to_next:
            continue
        
        day = labeled[labeled["trade_date"] == date].dropna(subset=factor_cols)
        if len(day) < n_top * 2:
            continue
        
        # 用ML模型预测
        X = np.nan_to_num(day[factor_cols].values.astype(np.float32), nan=0)
        # 只预测一次: 使用训练集训练的模型
        
        top = day.sort_values("monthly_ret", ascending=False)  # 理想情况，实际应该用预测
        # 这里先用真实收益排序做上限验证
        top_ret = top.head(n_top)["monthly_ret"].mean()
        rets.append(top_ret)
    
    if rets:
        pnl = np.array(rets)
        sr = np.mean(pnl) / np.std(pnl) * np.sqrt(12)
        avg_ret = np.mean(pnl) * 100
        print(f"  理想选股 Top{n_top}: 月均{avg_ret:.2f}% | 夏普{sr:.2f} | {len(rets)}个月")

# ===== ML 模型滚动训练 + 回测 =====
print(f"\n[ML] XGBoost滚动训练预测...")

all_predictions = []

for i, date in enumerate(test_dates):
    # 训练: 过去4年, 验证: 过去6个月
    train_start = date - pd.Timedelta(days=4*365)
    val_start = date - pd.Timedelta(days=6*30)
    
    train = labeled[(labeled["trade_date"] >= train_start) & 
                    (labeled["trade_date"] < val_start)].dropna(subset=factor_cols + ["monthly_ret"])
    val = labeled[(labeled["trade_date"] >= val_start) & 
                  (labeled["trade_date"] < date)].dropna(subset=factor_cols + ["monthly_ret"])
    
    train = train[train["monthly_ret"].abs() < 0.3]
    val = val[val["monthly_ret"].abs() < 0.3]
    
    if len(train) < 5000 or len(val) < 1000:
        continue
    
    X_tr = np.nan_to_num(train[factor_cols].values.astype(np.float32), nan=0)
    y_tr = train["monthly_ret"].values.astype(np.float32)
    X_va = np.nan_to_num(val[factor_cols].values.astype(np.float32), nan=0)
    y_va = val["monthly_ret"].values.astype(np.float32)
    
    xgb_m = xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbosity=0, n_jobs=8,
        early_stopping_rounds=30
    )
    xgb_m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    
    lgb_m = lgb.LGBMRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbose=-1, n_jobs=8
    )
    lgb_m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(30)], eval_metric="mse")
    
    # 预测
    day = labeled[labeled["trade_date"] == date]
    X_te = np.nan_to_num(day[factor_cols].values.astype(np.float32), nan=0)
    p = (xgb_m.predict(X_te) + lgb_m.predict(X_te)) / 2
    
    for j, code in enumerate(day["ts_code"].values):
        all_predictions.append({
            "trade_date": date, "ts_code": code, 
            "pred_ret": float(p[j]),
            "monthly_ret": day["monthly_ret"].iloc[j]
        })
    
    if (i + 1) % 15 == 0:
        xgb_t = xgb_m.best_iteration + 1 if xgb_m.best_iteration else xgb_m.n_estimators
        lgb_t = getattr(lgb_m, 'best_iteration_', lgb_m.n_estimators)
        print(f"  [{i+1}/{len(test_dates)}] train={len(train):,} val={len(val):,} xgb={xgb_t} lgb={lgb_t}")

df_pred = pd.DataFrame(all_predictions)
print(f"预测: {len(df_pred):,} 条")

# ===== 回测 =====
print(f"\n[回测] 精确月频标签 + 交易成本:")

# 无成本回测 - 验证选股能力
for n_top in [50, 100, 200]:
    rets = []
    for date in test_dates:
        day = df_pred[df_pred["trade_date"] == date].sort_values("pred_ret", ascending=False)
        if len(day) < n_top:
            continue
        rets.append(day.head(n_top)["monthly_ret"].mean())
    
    if rets:
        pnl = np.array(rets)
        tr = np.prod(1 + pnl) - 1
        ar = (1+tr)**(12/len(pnl)) - 1
        vol = np.std(pnl) * np.sqrt(12)
        sr = np.mean(pnl) / np.std(pnl) * np.sqrt(12)
        dd = np.maximum.accumulate(np.cumprod(1+pnl)) - np.cumprod(1+pnl)
        mdd = dd.max()
        print(f"\n  Top {n_top} (无成本): 总{tr*100:.1f}% 年化{ar*100:.1f}% 波动{vol*100:.1f}% 夏普{sr:.2f} 回撤{mdd*100:.1f}%")

# 有成本回测
for n_top in [50, 100, 200]:
    result = precise_backtest(df_pred, n_top, f"Top {n_top}")
    if result:
        results.append(result)

print(f"\n总用时: {(time.time()-t0)/60:.1f} 分钟")
