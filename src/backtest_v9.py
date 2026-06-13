"""
最终回测 v9 - 20日调仓 + 目标波动率
正确实现：不借钱，通过仓位调整控制波动
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
print("v9 - 20日调仓 + 目标波动率")
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

# 20日节点
all_dates = sorted(panel["trade_date"].unique())
period_dates = [all_dates[i] for i in range(0, len(all_dates), 20) 
                if all_dates[i] >= pd.Timestamp("2021-01-01")]

price_map = {}
for d in period_dates:
    sub = prices[prices["trade_date"] == d][["ts_code","close"]].dropna()
    price_map[d] = dict(zip(sub["ts_code"], sub["close"]))

# 使用v8缓存的预测（与period_dates完全对齐）
pred = pd.read_parquet(os.path.join(DATA_FACTORS, "pred_20d_v8.parquet"))
pred["trade_date"] = pd.to_datetime(pred["trade_date"])
pred_dates = sorted(pred["trade_date"].unique())
print(f"预测节点: {len(pred_dates)}")

# 检查 time period 对齐
print("日期对齐检查:")
for i in range(min(5, len(period_dates))):
    print(f"  周期{i}: {str(period_dates[i].date())}  {'' if period_dates[i] == pred_dates[i] else '≠'} 预测{str(pred_dates[i].date()) if i < len(pred_dates) else 'N/A'}")

# ===== 无成本验证 =====
print(f"\n无成本验证:")
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
        print(f"  Top{n:3d}: 均值{np.mean(pnl)*100:+.2f}% 夏普{sr:.2f} {len(rets)}期")

# ===== 含成本 + 目标波动控制 =====
print(f"\n{'='*60}")
print("含成本 + 目标波动率（仓位调整）")
print(f"{'='*60}")

STAMP, COMM, SLIP = 0.001, 0.0002, 0.002

def backtest_vol_target(n_stocks, target_vol=0.20, label=""):
    """
    通过仓位比例调整波动率
    - 波动率估计：过去4期收益的年化波动
    - 仓位比例 = target_vol / est_vol，最大1.0（不上杠杆）
    - 剩余现金留在账户不动
    """
    cash = 0.03
    holdings = {}
    navs = [1.0]
    hist_rets = []
    
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
        
        # ---- 计算仓位比例 ----
        position_ratio = 1.0  # 默认满仓
        if len(hist_rets) >= 4:
            est_vol = np.std(hist_rets[-4:]) * np.sqrt(13)
            if est_vol > 0.01:
                position_ratio = min(target_vol / est_vol, 1.0)
                position_ratio = max(position_ratio, 0.1)  # 最低10%
        
        # ---- 买入 ----
        day_pred = pred[pred["trade_date"] == date].sort_values("pred_ret", ascending=False)
        selected = list(day_pred.head(n_stocks)["ts_code"].values)
        
        if selected and cash > 0.001:
            # 仓位调整：只用 position_ratio 比例的资金买入
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
        hist_rets.append(ret)
        
        if i < 5 or (i+1) % 15 == 0 or len(holdings) == 0:
            print(f"  p{i:3d} {str(date.date())}->{str(sell_date.date())} | "
                  f"持{len(holdings):3d} | 仓{position_ratio:.2f} | "
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
    
    print(f"\n  {label} Top{n_stocks} 目标波动{target_vol*100:.0f}%:")
    print(f"    总收益: {tr*100:.1f}% | 年化: {ar*100:.1f}%")
    print(f"    实际波动: {vol*100:.1f}% | 夏普: {sr:.2f} | 回撤: {mdd*100:.1f}%")
    print(f"    卡玛: {calmar:.2f} | 胜率: {wr*100:.0f}% | {len(pnl)}期")
    return nav_arr

for n in [30, 50]:
    for tv in [0.20, 0.30]:
        backtest_vol_target(n, target_vol=tv, label="含成本")

print(f"\n总用时: {(time.time()-t0)/60:.1f}分")
