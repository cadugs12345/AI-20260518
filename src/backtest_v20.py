"""
v20 回测 — 使用v19的预测 + 轻量净值计算
避免OOM：不加载完整面板，只加载pred + 价格 + 波动率

衍生因子（高波反转IR 0.54 + 量价背离IR 0.78）已通过ML训练包含在pred中

Usage:
    time python src/backtest_v20.py
"""
import sys, os, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config.settings import DATA_FACTORS

t0 = time.time()

print("="*60)
print("v20 回测 — 含新衍生因子的ML预测")
print("="*60)

# 1. 加载v19预测（已含衍生因子）
print("\n[1] 加载预测...")
pred = pd.read_pickle(os.path.join(DATA_FACTORS, "pred_20d_v19.pkl"))
pred["trade_date"] = pd.to_datetime(pred["trade_date"])
pred_dates = sorted(pred["trade_date"].unique())
print(f"  预测: {len(pred):,}行, {len(pred_dates)}个交易日")
print(f"  时间: {pred_dates[0].date()} ~ {pred_dates[-1].date()}")

# 2. 加载价格（仅close）
print("\n[2] 加载价格...")
prices = pd.read_parquet(os.path.join(DATA_FACTORS, "full_prices.parquet"))
prices["trade_date"] = pd.to_datetime(prices["trade_date"])
print(f"  价格: {len(prices):,}行")

# 3. 预计算60日波动率（只对pred_dates中的日期）
print("\n[3] 预计算波动率...")
prices_sorted = prices.sort_values(["ts_code","trade_date"]).copy()
prices_sorted["ret_1d"] = prices_sorted.groupby("ts_code")["close"].pct_change()
prices_sorted["vol_60d"] = prices_sorted.groupby("ts_code")["ret_1d"].transform(
    lambda x: x.rolling(60, min_periods=20).std())
prices_sorted["vol_60d_ann"] = prices_sorted["vol_60d"] * np.sqrt(244)

# 构建dict缓存（高效查询）
print("  构建价格/波动率缓存...")
vol_map = {}
price_map = {}

# 逐日处理，避免大dict一次性内存峰值
for d in pred_dates:
    # 价格
    sub = prices[prices["trade_date"] == d][["ts_code","close"]].dropna()
    price_map[d] = dict(zip(sub["ts_code"], sub["close"]))
    
    # 波动率
    v_sub = prices_sorted[prices_sorted["trade_date"] == d][["ts_code","vol_60d_ann"]].dropna()
    vol_map[d] = dict(zip(v_sub["ts_code"], v_sub["vol_60d_ann"]))

del prices_sorted, prices, sub, v_sub
print(f"  价格缓存: {len(price_map)}日, 波动率缓存: {len(vol_map)}日")

# 4. 净值计算
print("\n[4] 净值计算...")

STAMP, COMM, SLIP = 0.001, 0.0002, 0.002

results = {}
for n_stocks, target_vol in [(30, 0.15), (30, 0.20), (50, 0.15), (50, 0.20)]:
    cash = 0.03
    holdings = {}
    navs = [1.0]
    dates_nav = []
    
    n_dates = len(pred_dates)
    for idx, date in enumerate(pred_dates[:-1]):
        sell_date = pred_dates[idx + 1]
        px_buy = price_map.get(date, {})
        px_sell = price_map.get(sell_date, {})
        stock_vol = vol_map.get(date, {})
        
        # 持仓市值 + 现金
        hold_val = 0
        for code, shares in list(holdings.items()):
            px = px_buy.get(code, 0)
            hold_val += shares * px
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
            median_vol = float(np.median(selected_vols))
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
        
        # 净值更新
        new_port = 0
        for code, shares in holdings.items():
            px = px_sell.get(code, 0)
            if px > 0:
                new_port += shares * px
        new_total = new_port + cash
        ret = new_total / total_val - 1 if total_val > 0 else 0
        navs.append(navs[-1] * (1 + ret))
        dates_nav.append(date)
        
        # 进度
        if (idx + 1) % 200 == 0:
            print(f"  {n_stocks}只V{int(target_vol*100)}: {idx+1}/{n_dates}日 ({(idx+1)/n_dates*100:.0f}%)")
    
    nav_array = np.array(navs)
    pnl = nav_array[1:] / nav_array[:-1] - 1
    sr = float(np.mean(pnl) / np.std(pnl) * np.sqrt(13)) if np.std(pnl) > 0 else 0
    tr = float(nav_array[-1] - 1)
    ann_ret = float((1 + tr) ** (12 / max(len(pnl), 1)) - 1) if len(pnl) > 0 else 0
    dd = (np.maximum.accumulate(nav_array) - nav_array).max()
    mdd = float(dd)
    wr = float(np.mean(pnl > 0))
    calmar = tr / mdd if mdd > 0 else 0
    
    ver = f"T{n_stocks}_V{int(target_vol*100)}"
    results[ver] = {
        "total_return": tr,
        "annualized_return": ann_ret,
        "sharpe": sr,
        "max_dd": mdd,
        "win_rate": wr,
        "calmar": calmar,
    }
    
    print(f"\n  {ver}")
    print(f"    总收益: {tr*100:.1f}%")
    print(f"    年化: {ann_ret*100:.1f}%")
    print(f"    夏普: {sr:.2f}")
    print(f"    最大回撤: {mdd*100:.1f}%")
    print(f"    卡玛: {calmar:.2f}")
    print(f"    胜率: {wr*100:.0f}%")

# 5. 保存
out_path = os.path.join(DATA_FACTORS, "backtest_v20_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

time_used = (time.time() - t0) / 60
print(f"\n✅ v20结果: {out_path}")
print(f"总用时: {time_used:.1f}分")
