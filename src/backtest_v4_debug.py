"""
小型验证：1. 回测引擎测试(简单因子) 2. ML预测效度检查
"""
import os, sys, time, gc
import numpy as np
import pandas as pd

warnings = __import__('warnings')
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DATA_FACTORS

t0 = time.time()

# 加载
panel = pd.read_parquet(os.path.join(DATA_FACTORS, "factor_panel_with_fwd.parquet"))
prices = pd.read_parquet(os.path.join(DATA_FACTORS, "full_prices.parquet"))
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
prices["trade_date"] = pd.to_datetime(prices["trade_date"])
panel = panel[panel["trade_date"] >= "2018-01-01"].copy()
panel = panel.merge(prices, on=["ts_code","trade_date"], how="left")
del prices; gc.collect()

# 周频节点
dates = sorted(panel["trade_date"].unique())
weekly = dates[::5]
weekly = [d for d in weekly if d >= pd.Timestamp("2021-01-01")]

# 价格索引
px_index = {}
for d in weekly:
    sub = panel[panel["trade_date"] == d][["ts_code","close"]].set_index("ts_code")["close"].to_dict()
    px_index[d] = sub

print(f"[1] 价格索引: {len(px_index)} 节点")

# ===== 验证1: 简单动量因子回测 =====
# 用 fwd_20d_ret 选最近20日涨幅最大的股票 —— 验证回测引擎
print("\n[验证1] 简单动量因子回测 (上周涨幅前50)")
print("=" * 50)

# 先构建每期前20日涨跌幅 (作为简单因子)
print("  构建动量因子...")
momentum_panel = panel[["ts_code","trade_date","close"]].copy()
momentum_panel = momentum_panel.sort_values(["ts_code","trade_date"])
momentum_panel["ret_20d"] = momentum_panel.groupby("ts_code")["close"].pct_change(20)
# 保留周频节点
weekly_panel = momentum_panel[momentum_panel["trade_date"].isin(weekly)].copy()
del momentum_panel; gc.collect()
print(f"  周频因子表: {len(weekly_panel):,} 条")

def simple_backtest(n_stocks, use_ret_20d=True):
    """简单回测：用ret_20d选股"""
    cash = 0.03
    holdings = {}
    navs = [1.0]
    
    for i in range(len(weekly)):
        if i == len(weekly) - 1:
            break
        date = weekly[i]
        next_date = weekly[i+1]
        
        px_p = px_index.get(date, {})
        px_n = px_index.get(next_date, {})
        
        # 组合市值
        port_val = sum(shares * px_p.get(code, 0) for code, shares in holdings.items())
        total = port_val + cash
        
        # 当期选股 (用上一期的ret_20d)
        prev_row = weekly_panel[weekly_panel["trade_date"] == date].copy()
        prev_row = prev_row.dropna(subset=["ret_20d"])
        prev_row = prev_row.sort_values("ret_20d", ascending=False)
        selected = set(prev_row.head(n_stocks)["ts_code"].values)
        
        # 卖出不在selected的
        sell_amt = 0
        sell_cost = 0
        old_h = holdings.copy()
        holdings = {}
        for code, shares in old_h.items():
            px = px_p.get(code, 0)
            if px > 0:
                val = shares * px
                if code in selected:
                    holdings[code] = shares
                else:
                    cost = val * (0.0002 + 0.001 + 0.002)  # comm + stamp + slippage
                    sell_amt += val
                    sell_cost += cost
        
        cash += sell_amt - sell_cost
        
        # 买入新股
        buy_list = [c for c in selected if c not in holdings]
        if buy_list and cash > 0.001:
            per = (cash * 0.95) / len(buy_list)
            for code in buy_list:
                px = px_p.get(code, 0)
                if px > 0 and per > 0:
                    buy_cost = per * (0.0002 + 0.002)
                    shares = (per - buy_cost) / px
                    holdings[code] = shares
                    cash -= per
        
        # 下期市值
        new_port = sum(shares * px_n.get(code, 0) for code, shares in holdings.items())
        new_total = new_port + cash
        ret = new_total / total - 1
        navs.append(navs[-1] * (1 + ret))
    
    pnl = np.array(navs[1:]) / np.array(navs[:-1]) - 1
    navs_arr = np.array(navs)
    tr = navs_arr[-1] - 1
    ar = navs_arr[-1] ** (52/len(pnl)) - 1
    vol = np.std(pnl) * np.sqrt(52)
    sr = np.mean(pnl) / np.std(pnl) * np.sqrt(52) if np.std(pnl) > 0 else 0
    dd = np.maximum.accumulate(navs_arr) - navs_arr
    mdd = dd.max()
    wr = np.mean(pnl > 0)
    
    print(f"  Top {n_stocks}:")
    print(f"    总收益: {tr*100:.1f}% | 年化: {ar*100:.1f}%")
    print(f"    波动: {vol*100:.1f}% | 夏普: {sr:.2f} | 回撤: {mdd*100:.1f}%")
    print(f"    胜率: {wr*100:.0f}% | {len(pnl)}周")

simple_backtest(30)
simple_backtest(50)
simple_backtest(100)

print(f"\n用时: {(time.time()-t0)/60:.1f}分")
