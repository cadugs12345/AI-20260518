"""
调试回测引擎 - 逐笔跟踪
"""
import os, sys, numpy as np, pandas as pd

warnings = __import__('warnings')
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DATA_FACTORS

panel = pd.read_parquet(os.path.join(DATA_FACTORS, "factor_panel_with_fwd.parquet"))
prices = pd.read_parquet(os.path.join(DATA_FACTORS, "full_prices.parquet"))
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
prices["trade_date"] = pd.to_datetime(prices["trade_date"])
panel = panel[panel["trade_date"] >= "2018-01-01"].copy()
panel = panel.merge(prices, on=["ts_code","trade_date"], how="left")

dates = sorted(panel["trade_date"].unique())
weekly = dates[::5]
weekly = [d for d in weekly if d >= pd.Timestamp("2021-01-01")]

# 价格索引
px_index = {}
for d in weekly:
    sub = panel[panel["trade_date"] == d][["ts_code","close"]].set_index("ts_code")["close"].to_dict()
    px_index[d] = sub

# 极简回测 - 手工跟踪前10周
print("=== 手工跟踪回测 (Top 50动量) ===")
print()

# 先计算ret_20d
momentum = panel[["ts_code","trade_date","close"]].copy().sort_values(["ts_code","trade_date"])
momentum["ret_20d"] = momentum.groupby("ts_code")["close"].pct_change(20)

cash = 0.03  # 初始现金
holdings = {}  # code -> shares
navs = [1.0]
logs = []

for w in range(10):
    if w >= len(weekly) - 1:
        break
    date = weekly[w]
    next_date = weekly[w+1]
    
    px_p = px_index.get(date, {})
    px_n = px_index.get(next_date, {})
    
    # 当前市值
    port_val = sum(shares * px_p.get(code, 0) for code, shares in holdings.items())
    total = port_val + cash
    prev_total = total
    
    # 选股 (用截止到date的ret_20d)
    m_row = momentum[momentum["trade_date"] == date].dropna(subset=["ret_20d"])
    m_row = m_row.sort_values("ret_20d", ascending=False)
    selected = set(m_row.head(50)["ts_code"].values)
    
    # 卖出
    sell_amt, sell_cost = 0, 0
    old_h = holdings.copy()
    holdings = {}
    for code, shares in old_h.items():
        px = px_p.get(code, 0)
        if px > 0:
            val = shares * px
            if code in selected:
                holdings[code] = shares
            else:
                cost = val * (0.0002 + 0.001 + 0.002)
                sell_amt += val
                sell_cost += cost
    
    cash += sell_amt - sell_cost
    
    # 买入
    buy_list = [c for c in selected if c not in holdings]
    buy_amt = 0
    if buy_list and cash > 0.01:
        per = (cash * 0.95) / len(buy_list)
        for code in buy_list:
            px = px_p.get(code, 0)
            if px > 0 and per > 0:
                buy_cost = per * (0.0002 + 0.002)
                shares_bought = (per - buy_cost) / px
                holdings[code] = shares_bought
                cash -= per
                buy_amt += per
    
    # 新市值
    new_port = sum(shares * px_n.get(code, 0) for code, shares in holdings.items())
    new_total = new_port + cash
    ret = new_total / total - 1
    navs.append(navs[-1] * (1 + ret))
    
    logs.append({
        "week": w, "date": date, "n_hold": len(holdings), "port_val": port_val,
        "cash": cash, "total": total, "new_total": new_total,
        "sell_amt": sell_amt, "buy_amt": buy_amt, "ret": ret
    })

for log in logs:
    print(f"  w{log['week']} {str(log['date'].date()):12s}  holdings={log['n_hold']:3d}  "
          f"port={log['port_val']:.2f}  cash={log['cash']:.2f}  "
          f"total={log['total']:.3f}→{log['new_total']:.3f}  "
          f"ret={log['ret']*100:+.2f}%  "
          f"sell={log['sell_amt']:.2f}  buy={log['buy_amt']:.2f}")
    if any(v < 0 for v in [log['port_val'], log['cash'], log['total']]):
        print(f"  *** 负值! ***")

print(f"\n最终净值: {navs[-1]:.4f}, 10周累计: {(navs[-1]-1)*100:.1f}%")
