"""
最终版 - v12最优结果绘制净值曲线 + 回测结果汇总
"""
import os, sys, time, gc, pickle
import numpy as np
import pandas as pd

warnings = __import__('warnings')
warnings.filterwarnings("ignore")

DATA_FACTORS = "data/factors"
t0 = time.time()
print("="*60)
print("最终版净值曲线绘制")
print("="*60)

# ===== 加载v12预测缓存（panel_v2，20日）=====
panel = pd.read_parquet(os.path.join(DATA_FACTORS, "factor_panel_with_fwd_v2.parquet"))
prices = pd.read_parquet(os.path.join(DATA_FACTORS, "full_prices.parquet"))
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
prices["trade_date"] = pd.to_datetime(prices["trade_date"])

# 个股波动率
prices_sorted = prices.sort_values(['ts_code','trade_date']).copy()
prices_sorted['ret_1d'] = prices_sorted.groupby('ts_code')['close'].pct_change()
prices_sorted['vol_60d'] = prices_sorted.groupby('ts_code')['ret_1d'].transform(
    lambda x: x.rolling(60, min_periods=20).std())
prices_sorted['vol_60d_ann'] = prices_sorted['vol_60d'] * np.sqrt(244)

# 周期节点
all_dates = sorted(panel["trade_date"].unique())
period_dates = [all_dates[i] for i in range(0, len(all_dates), 20) 
                if all_dates[i] >= pd.Timestamp("2021-01-01")]

# 映射
price_map, vol_map = {}, {}
for d in period_dates:
    sub = prices[prices["trade_date"] == d][["ts_code","close"]].dropna()
    price_map[d] = dict(zip(sub["ts_code"], sub["close"]))
    v_sub = prices_sorted[prices_sorted["trade_date"] == d][["ts_code","vol_60d_ann"]].dropna()
    vol_map[d] = dict(zip(v_sub["ts_code"], v_sub["vol_60d_ann"]))

# 加载ML预测
pred_path = os.path.join(DATA_FACTORS, "pred_20d_v8.parquet")
pred = pd.read_parquet(pred_path)
pred["trade_date"] = pd.to_datetime(pred["trade_date"])
pred_dates = sorted(pred["trade_date"].unique())
print(f"pred: {len(pred):,}条, {len(pred_dates)}期")

STAMP, COMM, SLIP = 0.001, 0.0002, 0.002

def backtest_nav(n_stocks, target_vol=0.15, label=""):
    """返回当期仓位序列、nav序列、回撤序列"""
    cash = 0.03
    holdings = {}
    navs = [1.0]
    dates = [pred_dates[0]]
    positions = []
    
    for i, date in enumerate(pred_dates[:-1]):
        sell_date = pred_dates[i+1]
        px_buy = price_map.get(date, {})
        px_sell = price_map.get(sell_date, {})
        stock_vol = vol_map.get(date, {})
        
        hold_val = sum(shares * px_buy.get(c, 0) for c, shares in holdings.items())
        total_val = hold_val + cash
        
        # 卖出
        sell_proceeds = 0
        for code, shares in holdings.items():
            px = px_sell.get(code, 0)
            if px > 0:
                val = shares * px
                cost = val * (STAMP + COMM + SLIP)
                sell_proceeds += val - cost
        cash = cash + sell_proceeds
        holdings = {}
        
        # 截面波动
        day_pred = pred[pred["trade_date"] == date].sort_values("pred_ret", ascending=False)
        selected = list(day_pred.head(n_stocks)["ts_code"].values)
        
        selected_vols = [stock_vol.get(c, np.nan) for c in selected]
        selected_vols = [v for v in selected_vols if not np.isnan(v) and v > 0.01]
        
        if len(selected_vols) >= 5:
            median_vol = np.median(selected_vols)
            pos_ratio = min(target_vol / median_vol, 1.0)
            pos_ratio = max(pos_ratio, 0.05)
        else:
            median_vol = np.nan
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
                        if bought > 0: holdings[code] = bought
                cash -= per * len(holdings)
        
        # 收益
        new_port = sum(shares * px_sell.get(c, 0) for c, shares in holdings.items())
        new_total = new_port + cash
        ret = new_total / total_val - 1 if total_val > 0 else 0
        navs.append(navs[-1] * (1 + ret))
        dates.append(sell_date)
        positions.append(pos_ratio)
    
    nav_arr = np.array(navs)
    dd = np.maximum.accumulate(nav_arr) - nav_arr
    
    return dates, nav_arr, dd, positions

# ===== 运行最优配置 =====
print("\n运行最优配置...")
results = {}
for n, tv, name in [
    (30, 0.15, "Top30_目波15"),
    (30, 0.20, "Top30_目波20"),
    (50, 0.15, "Top50_目波15"),
    (50, 0.20, "Top50_目波20"),
]:
    dates, nav_arr, dd, pos = backtest_nav(n, target_vol=tv)
    pnl = nav_arr[1:] / nav_arr[:-1] - 1
    tr = nav_arr[-1] - 1
    n_years = len(pnl) / 13
    ar = nav_arr[-1] ** (1/n_years) - 1
    vol = np.std(pnl) * np.sqrt(13)
    sr = np.mean(pnl) / np.std(pnl) * np.sqrt(13)
    mdd = dd.max()
    wr = np.mean(pnl > 0)
    calmar = ar / mdd
    
    results[name] = {
        'dates': dates,
        'nav': nav_arr,
        'dd': dd,
        'positions': pos,
        'stats': {
            'tr': tr, 'ar': ar, 'vol': vol, 'sr': sr, 
            'mdd': mdd, 'wr': wr, 'calmar': calmar,
            'n_periods': len(pnl)
        }
    }
    
    print(f"  {name}: 年化{ar*100:.1f}% 波动{vol*100:.1f}% 夏普{sr:.2f} 回撤{mdd*100:.1f}%")

# ===== 绘图 =====
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(3, 1, figsize=(16, 14), gridspec_kw={'height_ratios': [3, 1, 1]})

# 颜色
colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']
best_colors = {'Top30_目波15': colors[0], 'Top30_目波20': colors[1], 
               'Top50_目波15': colors[2], 'Top50_目波20': colors[3]}

# ---- 净值曲线 ----
ax1 = axes[0]
for name, r in results.items():
    dates_dt = [pd.Timestamp(d) for d in r['dates']]
    ax1.plot(dates_dt, r['nav'], label=name, color=best_colors[name], linewidth=1.8)
    ax1.fill_between(dates_dt, r['nav'], alpha=0.08, color=best_colors[name])

# 标注最优夏普
best = results['Top30_目波15']['stats']
ax1.text(0.02, 0.97, f"最优: 夏普{best['sr']:.2f} 年化{best['ar']*100:.1f}% 回撤{best['mdd']*100:.1f}%",
         transform=ax1.transAxes, fontsize=12, va='top',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

ax1.set_title('v12 截面波动率控制策略 — 净值曲线 (含成本)', fontsize=14, fontweight='bold')
ax1.set_ylabel('净值', fontsize=11)
ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.2)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

# ---- 回撤图 ----
ax2 = axes[1]
for name, r in results.items():
    dates_dt = [pd.Timestamp(d) for d in r['dates']]
    ax2.fill_between(dates_dt, -r['dd'], 0, 
                     color=best_colors[name], alpha=0.4, step='post')
    ax2.plot(dates_dt, -r['dd'], color=best_colors[name], linewidth=0.8)

ax2.set_ylabel('回撤', fontsize=11)
ax2.set_ylim(-0.45, 0.02)
ax2.grid(True, alpha=0.2)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

# ---- 仓位图 ----
ax3 = axes[2]
for name, r in results.items():
    if 'Top30' in name:  # 只看Top30
        dates_dt = [pd.Timestamp(d) for d in r['dates'][:-1]]
        ax3.plot(dates_dt, r['positions'], label=name, 
                 color=best_colors[name], linewidth=1.5)

ax3.set_ylabel('仓位(权益占比)', fontsize=11)
ax3.set_ylim(0, 1.05)
ax3.set_xlabel('交易日期', fontsize=11)
ax3.legend(loc='upper left', fontsize=9)
ax3.grid(True, alpha=0.2)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

plt.tight_layout()
plt.savefig('v12_final_results.png', dpi=150, bbox_inches='tight')
print(f"\n保存: v12_final_results.png")

# ===== 表格 =====
print(f"\n{'='*60}")
print("v12 最终结果汇总")
print(f"{'='*60}")
print(f"{'策略':20s} | {'年化':>7s} | {'波动':>7s} | {'夏普':>6s} | {'回撤':>7s} | {'卡玛':>6s} | {'胜率':>5s} | {'期数':>4s}")
print("-"*80)
for name in ['Top30_目波15', 'Top30_目波20', 'Top50_目波15', 'Top50_目波20']:
    s = results[name]['stats']
    print(f"{name:20s} | {s['ar']*100:6.1f}% | {s['vol']*100:6.1f}% | {s['sr']:5.2f} | {s['mdd']*100:6.1f}% | "
          f"{s['calmar']:5.2f} | {s['wr']*100:4.0f}% | {s['n_periods']:4d}")

# ===== 阶段分析 =====
print(f"\n阶段收益分析（Top30_目波15%）:")
r = results['Top30_目波15']
nav_arr = r['nav']
dates_dt = [pd.Timestamp(d) for d in r['dates']]
pnl = nav_arr[1:] / nav_arr[:-1] - 1

years = [2021, 2022, 2023, 2024, 2025, 2026]
for yr in years:
    mask = [pd.Timestamp(d).year == yr for d in r['dates'][:-1]]
    if sum(mask) > 0:
        yr_pnl = pnl[mask]
        yr_nav = nav_arr[np.insert(mask, 0, False)]
        yr_tr = yr_nav[-1] / yr_nav[0] - 1 if len(yr_nav) > 1 else 0
        print(f"  {yr}: {len(yr_pnl):2d}期 绝对收益{yr_tr*100:+.1f}% "
              f"均值{np.mean(yr_pnl)*100:+.2f}% 波动{np.std(yr_pnl)*np.sqrt(13)*100:.1f}% "
              f"夏普{np.mean(yr_pnl)/np.std(yr_pnl)*np.sqrt(13):.2f}")

print(f"\n总用时: {(time.time()-t0)/60:.1f}分")
