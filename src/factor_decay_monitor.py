"""
因子衰减预警系统 + 可视化
"""
import os, sys, time
import numpy as np
import pandas as pd

warnings = __import__('warnings')
warnings.filterwarnings("ignore")

t0 = time.time()
print("="*60)
print("因子衰减预警系统")
print("="*60)

# ===== 加载 =====
panel = pd.read_parquet("data/factors/factor_panel_with_fwd_v2.parquet")
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
all_dates = sorted(panel["trade_date"].unique())

skip = {"ts_code","trade_date","fwd_20d_ret","fwd_5d_ret","均值","20日收益率",
        "短期反转","毛利率","量比","BP","SP","EP","股息率","流通市值"}
factors = [c for c in panel.columns if c not in skip and panel[c].dtype in ("float64","int64")]
print(f"因子: {len(factors)}个")

# ===== 计算所有因子的IC时间序列（20日采样）=====
from scipy import stats

ic_data = {}
sample_dates = all_dates[::20]
print(f"IC计算: {len(sample_dates)}个采样点")

for f in factors:
    ic_vals = []
    for date in sample_dates:
        day = panel[panel["trade_date"] == date]
        day = day[[f, "fwd_20d_ret"]].dropna()
        if len(day) < 50: continue
        v = day[f].values.astype(np.float64)
        r = day["fwd_20d_ret"].values.astype(np.float64)
        mask = np.abs(r) < 0.5
        if mask.sum() < 50: continue
        ic, _ = stats.spearmanr(v[mask], r[mask])
        ic_vals.append({"trade_date": date, "IC": ic})
    ic_data[f] = pd.DataFrame(ic_vals)
    if factors.index(f) % 5 == 4:
        print(f"  {factors.index(f)+1}/{len(factors)}")

# ===== 计算滚动IC_IR =====
def rolling_ic_ir(ic_series, window=20):
    """滚动IC_IR = 滚动均值 / 滚动标准差"""
    ic = ic_series["IC"].values
    if len(ic) < window:
        return ic_series.copy()
    roll_mean = pd.Series(ic).rolling(window, min_periods=10).mean()
    roll_std = pd.Series(ic).rolling(window, min_periods=10).std()
    roll_ir = roll_mean / roll_std
    result = ic_series.copy()
    result["IC_IR"] = roll_ir.values
    return result

# ===== 衰减分析 =====
print("\n衰减分析（按3个窗口比较）:")
alerts = []
analysis = []

for f in factors:
    ic_df = ic_data.get(f)
    if ic_df is None or len(ic_df) < 30:
        continue
    
    # 早期 (最早30个点) vs 近期 (最近30个点)
    early = ic_df["IC"].head(30)
    recent = ic_df["IC"].tail(30)
    
    early_ir = early.mean() / early.std() if early.std() > 0 else 0
    recent_ir = recent.mean() / recent.std() if recent.std() > 0 else 0
    decay = recent_ir - early_ir
    
    # IC绝对值
    ic_abs = abs(ic_df["IC"].mean())
    
    # 滚动IC_IR（最近窗口）
    rolling = rolling_ic_ir(ic_df, window=15)
    last_ir = rolling["IC_IR"].tail(10).mean() if len(rolling) > 10 else 0
    
    # 半衰期：IC下降到原来一半所需的时间
    # 简单版本：看最近20个点的IC趋势斜率的符号
    from scipy import stats as ss
    recent_20 = ic_df.tail(20).copy()
    if len(recent_20) > 10:
        recent_20["idx"] = range(len(recent_20))
        slope, _, _, pval, _ = ss.linregress(recent_20["idx"], recent_20["IC"])
        trend = "下降" if slope < -0.0005 else ("上升" if slope > 0.0005 else "平稳")
    else:
        slope, pval, trend = 0, 1, "未知"
    
    # 状态判断
    if abs(ic_abs) < 0.01:  # IC几乎为零 → 基本无效
        status = "无效❌"
    elif decay < -0.3 and abs(recent_ir) < 0.3:
        status = "衰减⚠️"
    elif decay < -0.3 and abs(recent_ir) >= 0.3:
        status = "衰减但仍有信号"
    elif trend == "下降" and pval < 0.1:
        status = "趋势性下降📉"
    else:
        status = "正常"
    
    analysis.append({
        "factor": f, "ic_mean": ic_df["IC"].mean(), "ic_std": ic_df["IC"].std(),
        "early_ir": early_ir, "recent_ir": recent_ir, "decay": decay,
        "last_ir": last_ir, "trend": trend, "slope": slope, "pval": pval,
        "status": status, "n_points": len(ic_df)
    })
    
    if "衰减" in status or "下降" in status:
        alerts.append(f)

# ===== 输出 =====
print(f"\n{'因子':18s} | {'IC均值':>8s} | {'早期IR':>7s} | {'近期IR':>7s} | {'衰减':>6s} | {'趋势':6s} | {'状态':16s}")
print("-"*85)

# 按重要性排序
core_order = ["60日动量","20日动量","市值","EMA20偏离","120日动量","换手率","EMA5偏离","波动率","ROE",
              "MACD","RSI_24","OBV","EMA10偏离","BOLL位置","量能趋势","RSI_12","RSI_6","净利率","杠杆"]
analysis.sort(key=lambda x: core_order.index(x["factor"]) if x["factor"] in core_order else 99)

for a in analysis:
    trend_arrow = "↘" if a["trend"] == "下降" else ("↗" if a["trend"] == "上升" else "→")
    print(f"{a['factor']:18s} | {a['ic_mean']*100:+7.3f}% | {a['early_ir']:+6.2f} | "
          f"{a['recent_ir']:+6.2f} | {a['decay']:+5.2f} | {trend_arrow}{' '*4} | {a['status']:16s}")

print(f"\n{'='*60}")
print(f"预警: {len(alerts)}个因子需关注")
print(f"="*60)
if alerts:
    for f in alerts:
        a = [x for x in analysis if x["factor"] == f][0]
        print(f"  ⚠️ {f:18s} | IC均值{a['ic_mean']*100:+.3f}% | "
              f"早期IR {a['early_ir']:+.2f} → 近期IR {a['recent_ir']:+.2f} | 衰减{a['decay']:+.2f}")

# ===== 绘图 =====
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 选前8重要因子的IC趋势
top8 = [a for a in analysis if a["factor"] in core_order[:8]]
top8_names = [a["factor"] for a in top8]

n_plots = len(top8_names)
fig = plt.figure(figsize=(18, 3 * n_plots))
gs = gridspec.GridSpec(n_plots, 1, hspace=0.5)

colors = plt.cm.RdYlBu(np.linspace(0.2, 0.8, n_plots))

for i, fname in enumerate(top8_names):
    ax = fig.add_subplot(gs[i])
    ic_df = ic_data.get(fname, pd.DataFrame())
    if len(ic_df) > 0:
        dates = pd.to_datetime(ic_df["trade_date"])
        ic = ic_df["IC"].values
        
        # 柱状图
        colors_ic = ['#4CAF50' if v > 0 else '#F44336' for v in ic]
        ax.bar(dates, ic, color=colors_ic, alpha=0.6, width=15)
        
        # 30日滚动均线
        if len(ic) > 15:
            ma = pd.Series(ic).rolling(15, min_periods=5).mean()
            ax.plot(dates, ma, color='#1565C0', linewidth=2, label='15期滚动均值')
        
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.2)
        ax.axhline(y=-0.1, color='gray', linestyle='--', alpha=0.2)
        
        # 衰减标注
        a_info = [x for x in analysis if x["factor"] == fname][0]
        status = a_info["status"]
        color_status = 'red' if '衰减' in status or '下降' in status else 'green'
        
        ax.set_title(f'{fname} (IC均值{a_info["ic_mean"]*100:+.3f}% | '
                     f'早期IR{a_info["early_ir"]:+.2f}→近期IR{a_info["recent_ir"]:+.2f} | {status})',
                     fontsize=11, color=color_status, fontweight='bold')
        ax.set_ylabel('Rank IC', fontsize=9)
        ax.grid(True, alpha=0.15)
        if i == n_plots - 1:
            ax.set_xlabel('交易日期', fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

plt.savefig('factor_decay_monitor.png', dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n保存: factor_decay_monitor.png")

# ===== 季度报告 =====
print(f"\n{'='*60}")
print("因子有效性季度摘要 (近1年)")
print(f"{'='*60}")
print(f"{'因子':18s} | {'近20期IC均值':>10s} | {'IC_IR':>6s} | {'方向':>6s} | {'有效':>4s}")
print("-"*55)
for a in analysis:
    f = a["factor"]
    ic_df = ic_data.get(f)
    if ic_df is None or len(ic_df) < 20: continue
    recent_20 = ic_df.tail(20)
    ic_m = recent_20["IC"].mean()
    ic_ir = ic_m / recent_20["IC"].std() if recent_20["IC"].std() > 0 else 0
    direction = "+TOP" if ic_m > 0 else "-BOT"
    effective = "✅" if abs(ic_ir) > 0.3 else ("◻️" if abs(ic_ir) > 0.15 else "❌")
    print(f"{a['factor']:18s} | {ic_m*100:+9.3f}% | {ic_ir:+5.2f} | {direction:6s} | {effective}")

print(f"\n总用时: {(time.time()-t0)/60:.1f}分")
