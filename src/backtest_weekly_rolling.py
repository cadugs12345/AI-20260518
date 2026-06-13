"""
真实周频调仓回测
- 每周用ML预测选股，买入持有多周
- 每只股票最多持有20个交易日（约4周）
- 持仓由5批构成，每周轮换一批
"""
import os, sys, time, gc
import numpy as np
import pandas as pd

warnings = __import__('warnings')
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_FACTORS = "data/factors"
t0 = time.time()
print("=" * 60)
print("周频调仓回测 v5 - 多批持仓")
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

# 每周节点（周一到周五都行，每周第一个交易日）
dates = sorted(panel["trade_date"].unique())
# 按周分组取每周第一个交易日
weekly_dates = []
for i, d in enumerate(dates):
    if i == 0:
        weekly_dates.append(d)
    else:
        if (d - dates[i-1]).days > 3:  # 周末间隔（3天以上=新一周）
            weekly_dates.append(d)
    if d >= pd.Timestamp("2026-05-15"):
        break
weekly_dates = [d for d in weekly_dates if d >= pd.Timestamp("2021-01-01")]

# 预构建价格表
print(f"\n[索引] {len(weekly_dates)} 周节点")
price_map = {}
for d in weekly_dates:
    sub = panel[panel["trade_date"] == d][["ts_code","close"]].dropna()
    price_map[d] = dict(zip(sub["ts_code"], sub["close"]))

# ===== ML预测 =====
print("[ML] 滚动训练+预测（每期预测所有标的的fwd_20d_ret）")

all_preds = []
for i, date in enumerate(weekly_dates):
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
    
    if (i+1) % 50 == 0:
        xgb_t = xgb_m.best_iteration + 1 if xgb_m.best_iteration else xgb_m.n_estimators
        lgb_t = getattr(lgb_m, 'best_iteration_', lgb_m.n_estimators)
        print(f"  [{i+1}/{len(weekly_dates)}] train={len(train):,} xgb={xgb_t} lgb={lgb_t}")

df_pred = pd.DataFrame(all_preds)
pred_dates = sorted(df_pred["trade_date"].unique())
print(f"  ML完成: {len(df_pred):,} 条, {len(pred_dates)}周")

# ===== 验证：选股能力 =====
print(f"\n[验证] 选股能力（fwd_20d_ret作为真实收益）:")
for n_top in [50, 100]:
    rets = []
    for i, date in enumerate(pred_dates):
        day = df_pred[df_pred["trade_date"] == date].sort_values("pred_ret", ascending=False)
        top = set(day.head(n_top)["ts_code"].values)
        actual = panel[panel["trade_date"] == date]
        rr = actual[actual["ts_code"].isin(top)]["fwd_20d_ret"].mean()
        if not np.isnan(rr):
            rets.append(rr)
    if rets:
        pnl = np.array(rets)
        # fwd_20d_ret对应20个交易日≈4周, 年化 = (1+mean)^13 - 1 (因为20D=约1/13年)
        # 更准确的: 每年约 52周 / 4周批次 = 13次
        sr = np.mean(pnl) / np.std(pnl) * np.sqrt(13)
        print(f"  Top{n_top}: 均值{np.mean(pnl)*100:.2f}% 夏普{sr:.2f} 胜率{np.mean(pnl>0)*100:.0f}%  {len(rets)}期")

# ===== 真实周频回测 =====
print(f"\n[回测] 真实周频调仓（持仓至20日/被替换）:")

stamp_tax = 0.001
commission = 0.0002
slippage = 0.002

def backtest_rolling(n_stocks_per_week, label=""):
    """
    策略：每周买入Top n_stocks，每只股票最多持有20个交易日
    持仓 = 最近5批买入的股票，每周轮换一批
    实际持仓数量 = n_stocks_per_week × 5
    """
    cash = 0.03
    # 持仓批次: list of {"date":买入日期, "codes":{code:shares}}
    batches = []
    navs = [1.0]
    trade_log = []  # 用于debug
    
    for i, date in enumerate(pred_dates):
        if i >= len(pred_dates) - 1:
            break
        next_date = pred_dates[i+1]
        
        # === 计算当前总市值 ===
        px_p = price_map.get(date, {})
        px_n = price_map.get(next_date, {})
        
        # 当前所有持仓的市值
        port_val = 0
        new_batches = []
        for batch in batches:
            batch_val = 0
            new_codes = {}
            for code, shares in batch["codes"].items():
                px = px_p.get(code, 0)
                if px > 0:
                    val = shares * px
                    # 检查是否满20日
                    days_held = (date - batch["date"]).days
                    if days_held < 28:  # 约20个交易日
                        batch_val += val
                        new_codes[code] = shares
            if new_codes:
                new_batches.append({"date": batch["date"], "codes": new_codes})
                port_val += batch_val
        
        batches = new_batches
        total_val = port_val + cash
        
        # === 买入新一批 ===
        day_pred = df_pred[df_pred["trade_date"] == date].sort_values("pred_ret", ascending=False)
        new_codes = set(day_pred.head(n_stocks_per_week)["ts_code"].values)
        
        # 去重（避免已持有的股票重复买入）
        already_held = set()
        for batch in batches:
            already_held.update(batch["codes"].keys())
        new_codes = new_codes - already_held
        
        # 买入
        cash_spent = 0
        shares_bought = {}
        if new_codes and cash > 0.01:
            # 用97%当前现金买入（保留3%现金缓冲）
            available = cash * 0.97
            per = available / len(new_codes)
            for code in new_codes:
                px = px_p.get(code, 0)
                if px > 0 and per > 0:
                    buy_cost = per * (commission + slippage)
                    shares = (per - buy_cost) / px
                    if shares > 0:
                        shares_bought[code] = shares
                        cash_spent += per
        
        if shares_bought:
            batches.append({"date": date, "codes": shares_bought})
            cash -= cash_spent
        
        # === 下期市值 ===
        new_port_val = 0
        held_codes = set()
        for batch in batches:
            for code, shares in batch["codes"].items():
                px = px_n.get(code, 0)
                if px > 0:
                    new_port_val += shares * px
                    held_codes.add(code)
        
        new_total = new_port_val + cash
        ret = new_total / total_val - 1 if total_val > 0 else 0
        navs.append(navs[-1] * (1 + ret))
        
        if i < 5 or (i+1) % 50 == 0:
            trade_log.append(
                f"  w{i}: {date.date()} | "
                f"仓位={len(held_codes)}票 | "
                f"批次={len(batches)} | "
                f"净值={navs[-1]:.4f} | "
                f"ret={ret*100:+.2f}% | "
                f"cash={cash:.3f}"
            )
    
    # 统计
    pnl = np.array(navs[1:]) / np.array(navs[:-1]) - 1
    nav_arr = np.array(navs)
    tr = nav_arr[-1] - 1
    n_years = len(pnl) / 52
    ar = nav_arr[-1] ** (1 / n_years) - 1 if n_years > 0 else 0
    vol = np.std(pnl) * np.sqrt(52) if len(pnl) > 1 else 0
    sr = np.mean(pnl) / np.std(pnl) * np.sqrt(52) if np.std(pnl) > 0 else 0
    dd = np.maximum.accumulate(nav_arr) - nav_arr
    mdd = dd.max()
    wr = np.mean(pnl > 0)
    calmar = ar / mdd if mdd > 0 else 0
    
    # 最大持有数
    max_positions = n_stocks_per_week * 5
    
    print(f"\n  {label}Top {n_stocks_per_week}/周 (≈{max_positions}票):")
    print(f"    总收益: {tr*100:.1f}% | 年化: {ar*100:.1f}%")
    print(f"    波动: {vol*100:.1f}% | 夏普: {sr:.2f} | 回撤: {mdd*100:.1f}%")
    print(f"    卡玛: {calmar:.2f} | 胜率: {wr*100:.0f}% | {len(pnl)}周")
    print(f"    期末仓位: {len(held_codes)}票, 现金: {cash:.3f}")
    
    # 前几周日志
    for log in trade_log[:10]:
        print(log)
    
    return nav_arr

nav_50 = backtest_rolling(50, "（每周50只新票）")

print(f"\n总用时: {(time.time()-t0)/60:.1f}分")
