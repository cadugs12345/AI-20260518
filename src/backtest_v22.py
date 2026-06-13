"""
v22 回测 — ML选股 + 衍生因子动态仓位

设计：
  1. 基础：v16 ML选股（T50_V15，夏普0.90基准）
  2. 动态仓位调整（调整target_vol）：
     - 量价背离信号 强烈 → 加仓（背离越强，反转概率越低，可更高仓位）
     - 高波反转 强烈 → 减仓（波动剧烈时降低风险暴露）
     - 多排强度 强烈 → 加仓（趋势明确，持仓更安全）

测试：
  用衍生因子的截面中位数判断市场状态 → 调整目标波动率

输出: backtest_v22_results.json
"""
import sys, os, json, time
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config.settings import DATA_FACTORS

t0 = time.time()

print("="*60)
print("v22 回测 — ML选股 + 衍生因子动态仓位")
print("="*60)

# 1. 加载
print("\n[1] 加载数据...")
pred = pd.read_pickle(os.path.join(DATA_FACTORS, "pred_20d_v16.pkl"))
pred["trade_date"] = pd.to_datetime(pred["trade_date"])

panel = pd.read_parquet(os.path.join(DATA_FACTORS, "factor_panel_v5.parquet"))
panel["trade_date"] = pd.to_datetime(panel["trade_date"])

prices = pd.read_parquet(os.path.join(DATA_FACTORS, "full_prices.parquet"))
prices["trade_date"] = pd.to_datetime(prices["trade_date"])

# 2. 合并衍生因子到pred
print("\n[2] 动态仓位信号...")
# 市场状态信号：取截面中位数 → 判断市场处于什么状态
derived_sigs = panel[["trade_date","ts_code","高波反转","量价背离信号","多排强度","波动收缩"]].copy()

# 每日市场状态
daily_market = derived_sigs.groupby("trade_date").agg({
    "高波反转": "median",
    "量价背离信号": "median",
    "多排强度": "median",
    "波动收缩": "median"
}).reset_index()

daily_market.columns = ["trade_date","mkt_高波反转","mkt_量价背离","mkt_多排强度","mkt_波动收缩"]

# 计算滚动百分位（过去60天的位置）
mkt = daily_market.copy()
for col in ["mkt_高波反转","mkt_量价背离","mkt_多排强度","mkt_波动收缩"]:
    mkt[f"{col}_pct"] = mkt[col].rolling(60, min_periods=20).apply(
        lambda x: rankdata(x)[-1] / len(x), raw=True)

# 动态波动率乘数
mkt["动态乘数"] = 1.0
mkt.loc[mkt["mkt_量价背离_pct"] > 0.7, "动态乘数"] *= 1.3    # 背离强 → 高仓位
mkt.loc[mkt["mkt_量价背离_pct"] < 0.2, "动态乘数"] *= 0.8    # 背离弱 → 低仓位
mkt.loc[mkt["mkt_高波反转_pct"] > 0.8, "动态乘数"] *= 0.7    # 高波反转极端 → 减仓
mkt.loc[mkt["mkt_高波反转_pct"] < 0.2, "动态乘数"] *= 1.1    # 低波稳定 → 加仓
mkt.loc[mkt["mkt_多排强度_pct"] > 0.7, "动态乘数"] *= 1.2    # 多头排列 → 加仓
mkt.loc[mkt["mkt_多排强度_pct"] < 0.3, "动态乘数"] *= 0.9    # 空头排列 → 减仓
mkt["动态乘数"] = mkt["动态乘数"].clip(0.4, 1.8)

print(f"  动态乘数统计:")
print(f"    min={mkt['动态乘数'].min():.2f}, max={mkt['动态乘数'].max():.2f}")
print(f"    mean={mkt['动态乘数'].mean():.2f}, <0.8={np.mean(mkt['动态乘数']<0.8)*100:.0f}%, >1.2={np.mean(mkt['动态乘数']>1.2)*100:.0f}%")

# 合并到pred
pred = pred.merge(mkt[["trade_date","动态乘数","mkt_量价背离","mkt_高波反转","mkt_多排强度"]], 
                  on="trade_date", how="left", suffixes=("","_mkt"))
pred["动态乘数"] = pred["动态乘数"].fillna(1.0)

# 3. 波动率缓存
print("\n[3] 波动率缓存...")
prices_sorted = prices.sort_values(["ts_code","trade_date"]).copy()
prices_sorted["ret_1d"] = prices_sorted.groupby("ts_code")["close"].pct_change()
prices_sorted["vol_60d"] = prices_sorted.groupby("ts_code")["ret_1d"].transform(
    lambda x: x.rolling(60, min_periods=20).std())
prices_sorted["vol_60d_ann"] = prices_sorted["vol_60d"] * np.sqrt(244)

pred_dates = sorted(pred["trade_date"].unique())
vol_map, price_map = {}, {}
for d in pred_dates:
    sub = prices[prices["trade_date"] == d][["ts_code","close"]].dropna()
    price_map[d] = dict(zip(sub["ts_code"], sub["close"]))
    v_sub = prices_sorted[prices_sorted["trade_date"] == d][["ts_code","vol_60d_ann"]].dropna()
    vol_map[d] = dict(zip(v_sub["ts_code"], v_sub["vol_60d_ann"]))

del prices_sorted, prices

# 4. 净值计算
print("\n[4] 净值计算...")

STAMP, COMM, SLIP = 0.001, 0.0002, 0.002
N_DATES = len(pred_dates)

# 测试两组参数
configs = [
    ("T50_V15_BASE", 50, 0.15),
    ("T50_V15_DYN", 50, 0.15, True),
    ("T50_V20_BASE", 50, 0.20),
    ("T50_V20_DYN", 50, 0.20, True),
]

results = {}
for cfg in configs:
    name, n_stocks, target_vol = cfg[:3]
    use_dynamic = cfg[3] if len(cfg) > 3 else False
    
    cash = 0.03
    holdings = {}
    navs = [1.0]
    
    for idx, date in enumerate(pred_dates[:-1]):
        sell_date = pred_dates[idx + 1]
        px_buy = price_map.get(date, {})
        px_sell = price_map.get(sell_date, {})
        stock_vol = vol_map.get(date, {})
        
        hold_val = sum(shares * px_buy.get(c, 0) for c, shares in holdings.items())
        total_val = hold_val + cash
        
        sell_proceeds = 0
        for code, shares in list(holdings.items()):
            px = px_sell.get(code, 0)
            if px > 0:
                val = shares * px
                cost = val * (STAMP + COMM + SLIP)
                sell_proceeds += val - cost
        cash += sell_proceeds
        holdings = {}
        
        day_pred = pred[pred["trade_date"] == date].sort_values("pred_ret", ascending=False)
        selected = list(day_pred.head(n_stocks)["ts_code"].values)
        
        selected_vols = [stock_vol.get(c, np.nan) for c in selected]
        selected_vols = [v for v in selected_vols if not np.isnan(v) and v > 0.01]
        
        if len(selected_vols) >= 5:
            median_vol = float(np.median(selected_vols))
            # 动态目标波动率
            if use_dynamic:
                mult = float(pred[pred["trade_date"] == date]["动态乘数"].iloc[0])
                vol_target = target_vol * mult
            else:
                vol_target = target_vol
            pos_ratio = min(vol_target / median_vol, 1.0) if median_vol > 0 else 1.0
            pos_ratio = max(pos_ratio, 0.05)
            pos_ratio = min(pos_ratio, 0.99)
        else:
            pos_ratio = 1.0
        
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
        
        if (idx + 1) % 500 == 0:
            print(f"  {name}: {idx+1}/{N_DATES}")
    
    nav_array = np.array(navs)
    pnl = nav_array[1:] / nav_array[:-1] - 1
    sr = float(np.mean(pnl) / np.std(pnl) * np.sqrt(13)) if np.std(pnl) > 0 else 0
    tr = float(nav_array[-1] - 1)
    ann_ret = float((1 + tr) ** (12 / max(len(pnl), 1)) - 1)
    dd = (np.maximum.accumulate(nav_array) - nav_array).max()
    wr = float(np.mean(pnl > 0))
    calmar = tr / dd if dd > 0 else 0
    
    results[name] = {"total_return": tr, "annualized": ann_ret, "sharpe": sr, 
                     "max_dd": float(dd), "win_rate": wr, "calmar": calmar}
    
    dyn_mark = "动态" if use_dynamic else "固定"
    print(f"\n  {name:20s} ({dyn_mark}仓位)")
    print(f"    总收益: {tr*100:.1f}%, 年化: {ann_ret*100:.1f}%")
    print(f"    夏普: {sr:.2f}, 回撤: {dd*100:.1f}%, 胜率: {wr*100:.0f}%, 卡玛: {calmar:.2f}")

# 5. 对比
print("\n[5] 对比汇总")
print(f"{'配置':20s} | {'年化':>6s} | {'夏普':>5s} | {'回撤':>6s} | {'卡玛':>5s} | {'胜率':>4s}")
print("-"*55)
for name in ["T50_V15_BASE","T50_V15_DYN","T50_V20_BASE","T50_V20_DYN"]:
    r = results.get(name, {})
    print(f"{name:20s} | {r.get('annualized',0)*100:5.1f}% | {r.get('sharpe',0):4.2f} | {r.get('max_dd',0)*100:5.1f}% | {r.get('calmar',0):4.2f} | {r.get('win_rate',0)*100:3.0f}%")

out_path = os.path.join(DATA_FACTORS, "backtest_v22_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n✅ v22结果: {out_path}")
print(f"总用时: {(time.time()-t0)/60:.1f}分")
