"""
策略Dashboard - 三合一可视化看板
1. 净值曲线 + 回撤曲线
2. 因子贡献度（堆叠柱状图）
3. IC衰减趋势（预警标注）
4. 输出独立HTML文件

Usage:
    python src/dashboard.py              # 使用缓存快速生成
    python src/dashboard.py --refresh    # 重新计算数据
"""
import os, sys, json, base64, io
from datetime import datetime
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config.settings import DATA_FACTORS
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from alert_system import FactorAlertSystem, ALERTS_DIR
from factor_lifecycle import FactorContribution

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

CACHE_DIR = os.path.join(DATA_FACTORS, "..", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fig_to_base64(fig, dpi=120):
    """matplotlib figure → base64 HTML img src"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_b64}"


class StrategyDashboard:
    """策略看板生成器"""
    
    def __init__(self, refresh=False):
        self.refresh = refresh
        self.report_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    def load_nav_data(self):
        """加载回测净值数据（v39可信回测）"""
        PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        nav_path = os.path.join(CACHE_DIR, "dashboard_nav.csv")
        if not self.refresh and os.path.exists(nav_path):
            return pd.read_csv(nav_path)
        
        # 加载 v39 可信回测净值（排除<20亿，固定模型，Top30）
        v39_path = os.path.join(PROJECT, "backtest_results", "v39_honest_nav.parquet")
        if os.path.exists(v39_path):
            v39 = pd.read_parquet(v39_path)
            v39["date"] = v39["entry_date"]
            nav_df = v39[["date", "nav"]].rename(columns={"nav": "v39可信回测"})
            nav_df.to_csv(nav_path, index=False)
            return nav_df
        
        # 兜底：没有v39就返回None
        print("⚠️ 未找到 v39 可信回测净值，跳过净值曲线")
        return None
    
    def _load_fallback_nav(self):
        """加载v12最终结果图上的数据（从PNG反推不现实，返回简单模拟）"""
        nav_path = os.path.join(CACHE_DIR, "dashboard_nav.csv")
        if os.path.exists(nav_path):
            return pd.read_csv(nav_path)
        return None
    
    def load_importance_data(self):
        """加载因子重要性数据"""
        imp_path = os.path.join(CACHE_DIR, "dashboard_importance.csv")
        if not self.refresh and os.path.exists(imp_path):
            return pd.read_csv(imp_path)
        
        panel_path = os.path.join(DATA_FACTORS, "factor_panel_with_fwd_v2.parquet")
        if not os.path.exists(panel_path):
            return None
        
        panel = pd.read_parquet(panel_path)
        panel["trade_date"] = pd.to_datetime(panel["trade_date"])
        
        skip = {"ts_code","trade_date","fwd_20d_ret","fwd_5d_ret","均值","20日收益率",
                "短期反转","毛利率","量比","BP","SP","EP","股息率","流通市值"}
        factor_cols = [c for c in panel.columns if c not in skip 
                       and panel[c].dtype in ("float64","int64")]
        
        contrib = FactorContribution(panel, factor_cols)
        importance, _ = contrib.train_and_analyze()
        importance.to_csv(imp_path, index=False)
        return importance
    
    def load_alert_data(self):
        """加载预警数据"""
        alert_path = os.path.join(ALERTS_DIR, "latest.json")
        if os.path.exists(alert_path):
            with open(alert_path) as f:
                latest = json.load(f)
            report_path = latest.get("path")
            if report_path and os.path.exists(report_path):
                with open(report_path) as f:
                    return json.load(f)
        
        # 重新运行
        system = FactorAlertSystem()
        report, text, df = system.build_report()
        system.save_report(report, text, df)
        return report
    
    def plot_nav_chart(self, nav_df):
        """绘制净值+回撤图"""
        if nav_df is None or len(nav_df) < 5:
            return None
        
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
        
        dates = pd.to_datetime(nav_df["date"])
        
        # 上：净值曲线
        ax1 = fig.add_subplot(gs[0])
        colors = ['#2196F3', '#FF9800', '#4CAF50', '#F44336']
        strategies = [c for c in nav_df.columns if c != "date"]
        
        for i, s in enumerate(strategies):
            nav_vals = nav_df[s].values
            ax1.plot(dates, nav_vals, color=colors[i % len(colors)], 
                    linewidth=1.8, alpha=0.85, label=s.replace("_", " "))
            
            # 标注最终夏普
            pnl = nav_vals / np.array([1] + list(nav_vals[:-1])) - 1
            sr = np.mean(pnl) / np.std(pnl) * np.sqrt(12) if np.std(pnl) > 0 else 0
            tr = nav_vals[-1] - 1
            label_text = f"{s.replace('_',' ')} | 总收益{tr*100:.1f}% | 夏普{sr:.2f}"
        
        ax1.axhline(y=1.0, color='gray', linestyle='-', alpha=0.3)
        ax1.set_ylabel('净值', fontsize=12)
        ax1.set_title('策略净值曲线', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.15)
        ax1.tick_params(labelbottom=False)
        
        # 下：回撤曲线
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        for i, s in enumerate(strategies):
            nav_vals = nav_df[s].values
            dd = np.maximum.accumulate(nav_vals) - nav_vals
            ax2.fill_between(dates, 0, dd, color=colors[i % len(colors)], 
                           alpha=0.3, label=f"{s.replace('_',' ')}")
            
            mdd = dd.max()
            if len(dd) > 0:
                ax2.annotate(f'MaxDD={mdd*100:.1f}%', 
                           xy=(dates.iloc[-1], dd[-1]),
                           fontsize=8, color=colors[i % len(colors)])
        
        ax2.set_ylabel('回撤', fontsize=12)
        ax2.set_xlabel('交易日期', fontsize=12)
        ax2.grid(True, alpha=0.15)
        ax2.legend(loc='lower left', fontsize=8)
        
        plt.tight_layout()
        return fig
    
    def plot_importance_chart(self, importance_df):
        """绘制因子贡献度图"""
        if importance_df is None or len(importance_df) < 3:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        top15 = importance_df.head(15).copy()
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(top15)))
        
        # 左：水平条形图
        bars = ax1.barh(range(len(top15)), top15["importance"].values * 100, 
                       color=colors, alpha=0.8)
        ax1.set_yticks(range(len(top15)))
        ax1.set_yticklabels(top15["factor"].values, fontsize=9)
        ax1.set_xlabel('贡献度 (%)', fontsize=12)
        ax1.set_title('因子贡献度 (Top15)', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.15, axis='x')
        
        # 添加标签
        for bar, val in zip(bars, top15["importance"].values * 100):
            ax1.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, 
                    f'{val:.1f}%', va='center', fontsize=8)
        
        # 右：累积贡献度饼图
        top5 = importance_df.head(5)
        other = importance_df.iloc[5:]
        labels = list(top5["factor"].values) + ["其他"]
        sizes = list(top5["importance"].values) + [other["importance"].sum()]
        colors_pie = plt.cm.Set2(np.linspace(0, 1, len(labels)))
        
        wedges, texts, autotexts = ax2.pie(sizes, labels=labels, autopct='%1.1f%%',
                                           colors=colors_pie, startangle=90,
                                           textprops={'fontsize': 8})
        ax2.set_title('因子贡献度分布', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_alert_chart(self, alert_report):
        """绘制IC衰减趋势图"""
        if not alert_report or "all_factors" not in alert_report:
            return None
        
        factors_data = alert_report["all_factors"]
        if not factors_data:
            return None
        
        # 选前8重要因子
        core = ["60日动量","20日动量","市值","EMA20偏离","120日动量","换手率","EMA5偏离","波动率"]
        selected = [f for f in core if any(d["factor"] == f for d in factors_data)]
        
        if not selected:
            selected = [d["factor"] for d in factors_data[:8]]
        
        n = len(selected)
        fig, axes = plt.subplots(n, 1, figsize=(14, 3*n))
        if n == 1:
            axes = [axes]
        
        # 尝试从alert_system获取IC序列
        system = FactorAlertSystem()
        system.calc_all_ic_series()
        
        for i, fname in enumerate(selected):
            ax = axes[i]
            ic_df = system.calc_ic_series(fname)
            
            if len(ic_df) > 0:
                dates = pd.to_datetime(ic_df["trade_date"])
                ic = ic_df["IC"].values
                
                colors_bar = ['#4CAF50' if v > 0 else '#F44336' for v in ic]
                ax.bar(dates, ic, color=colors_bar, alpha=0.5, width=15)
                
                if len(ic) > 10:
                    ma = pd.Series(ic).rolling(10, min_periods=5).mean()
                    ax.plot(dates, ma, color='#1565C0', linewidth=2, alpha=0.8)
                
                ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                ax.axhline(y=0.05, color='gray', linestyle='--', alpha=0.15)
                ax.axhline(y=-0.05, color='gray', linestyle='--', alpha=0.15)
            
            # 状态标注
            f_info = next((d for d in factors_data if d["factor"] == fname), None)
            if f_info:
                level = f_info.get("level", "未知")
                color = 'red' if "严重" in level or "关注" in level else (
                    'orange' if "轻微" in level else 'green')
                ax.set_title(f"{fname}  [{level}]  IC均值{f_info['ic_mean']*100:+.2f}%  "
                           f"衰减{f_info['decay']:+.2f}",
                           fontsize=11, color=color, fontweight='bold')
            else:
                ax.set_title(fname, fontsize=11)
            
            ax.set_ylabel('IC', fontsize=9)
            ax.grid(True, alpha=0.15)
            ax.axhline(y=0.01, color='red', linestyle=':', alpha=0.2)
            ax.axhline(y=-0.01, color='red', linestyle=':', alpha=0.2)
            
            if i < n - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel('交易日期', fontsize=9)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        
        plt.suptitle('因子IC衰减趋势', fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        return fig
    
    def generate_html(self, nav_fig, imp_fig, alert_fig, alert_report):
        """生成仪表盘HTML"""
        nav_img = fig_to_base64(nav_fig) if nav_fig else ""
        imp_img = fig_to_base64(imp_fig) if imp_fig else ""
        alert_img = fig_to_base64(alert_fig) if alert_fig else ""
        
        # 预警摘要
        alerts_html = ""
        if alert_report and alert_report.get("alerts"):
            alerts_html = '<div class="alert-section"><h3>⚠️ 因子预警</h3><ul>'
            for a in alert_report["alerts"][:8]:
                level_icon = "🔴" if "严重" in a["level"] else "🟡" if "关注" in a["level"] else "🟠"
                alerts_html += f'<li>{level_icon} <b>{a["factor"]}</b>: 衰减{a["decay"]:+.2f}, ' \
                              f'近期IR {a["recent_ic_ir"]:+.2f}</li>'
            alerts_html += '</ul></div>'
        
        # 概览指标
        if nav_fig is not None:
            ax = nav_fig.axes[0]
            # 解析净值数据
            nav_df = self.load_nav_data()
            if nav_df is not None:
                strategies = [c for c in nav_df.columns if c != "date"]
                metrics_rows = ""
                for s in strategies:
                    nav_vals = nav_df[s].values
                    pnl = nav_vals / np.array([1] + list(nav_vals[:-1])) - 1
                    sr = np.mean(pnl) / np.std(pnl) * np.sqrt(12) if np.std(pnl) > 0 else 0
                    tr = nav_vals[-1] - 1
                    dd = (np.maximum.accumulate(nav_vals) - nav_vals).max()
                    wr = np.mean(pnl > 0)
                    label = s.replace("_", " ")
                    metrics_rows += f"""
                    <tr>
                        <td>{label}</td>
                        <td>{tr*100:.1f}%</td>
                        <td>{sr:.2f}</td>
                        <td>{dd*100:.1f}%</td>
                        <td>{wr*100:.0f}%</td>
                    </tr>"""
                
                metrics_table = f"""
                <table>
                    <tr><th>策略</th><th>总收益</th><th>夏普</th><th>最大回撤</th><th>胜率</th></tr>
                    {metrics_rows}
                </table>"""
            else:
                metrics_table = ""
        else:
            metrics_table = ""
        
        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>量化策略Dashboard</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, 'Microsoft YaHei', sans-serif; background: #f5f5f5; color: #333; }}
.header {{ background: linear-gradient(135deg, #1a237e, #283593); color: white; padding: 20px 30px; }}
.header h1 {{ font-size: 24px; }}
.header .sub {{ font-size: 13px; opacity: 0.8; margin-top: 5px; }}
.grid {{ display: grid; grid-template-columns: 1fr; gap: 20px; padding: 20px; max-width: 1400px; margin: 0 auto; }}
.card {{ background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
.card-title {{ padding: 15px 20px; font-size: 16px; font-weight: bold; border-bottom: 1px solid #eee; }}
.card-body {{ padding: 15px; }}
.card-body img {{ width: 100%; height: auto; display: block; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ padding: 10px 15px; text-align: center; border-bottom: 1px solid #eee; }}
th {{ background: #f8f9fa; font-weight: 600; color: #555; }}
tr:hover {{ background: #f8f9fa; }}
.alert-section {{ background: #fff3e0; border: 1px solid #ffcc02; border-radius: 8px; padding: 15px 20px; margin-bottom: 15px; }}
.alert-section h3 {{ color: #e65100; margin-bottom: 10px; }}
.alert-section ul {{ list-style: none; }}
.alert-section li {{ padding: 4px 0; font-size: 13px; }}
.footer {{ text-align: center; padding: 20px; color: #999; font-size: 12px; }}
</style>
</head>
<body>

<div class="header">
    <h1>📊 A股量化策略 Dashboard</h1>
    <div class="sub">生成时间: {self.report_time} | 因子池: ~27个 | 策略: XGB+LightGBM 混合</div>
</div>

<div class="grid">
    {alerts_html}
    
    {f'<div class="card"><div class="card-title">📈 策略净值曲线</div><div class="card-body">{metrics_table}<br><img src="{nav_img}" alt="净值曲线"></div></div>' if nav_img else ''}
    
    {f'<div class="card"><div class="card-title">🔬 因子贡献度</div><div class="card-body"><img src="{imp_img}" alt="因子贡献度"></div></div>' if imp_img else ''}
    
    {f'<div class="card"><div class="card-title">📉 IC衰减趋势</div><div class="card-body"><img src="{alert_img}" alt="IC衰减"></div></div>' if alert_img else ''}
</div>

<div class="footer">
    Generated by AI-20260518 Alert System | 数据基于Tushare日线+财务+Panorama
</div>

</body>
</html>"""
        return html
    
    def build(self):
        """构建完整Dashboard"""
        print("="*60)
        print("构建策略Dashboard")
        print(f"时间: {self.report_time}")
        print("="*60)
        
        print("[1/4] 加载净值数据...")
        nav_df = self.load_nav_data()
        
        print("[2/4] 加载因子重要性...")
        imp_df = self.load_importance_data()
        
        print("[3/4] 加载预警数据...")
        alert_report = self.load_alert_data()
        
        print("[4/4] 生成图表...")
        nav_fig = self.plot_nav_chart(nav_df)
        imp_fig = self.plot_importance_chart(imp_df)
        alert_fig = self.plot_alert_chart(alert_report)
        
        print("生成HTML...")
        html = self.generate_html(nav_fig, imp_fig, alert_fig, alert_report)
        
        # 保存
        html_path = os.path.join(OUTPUT_DIR, "dashboard.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        
        print(f"Dashboard已保存: {html_path}")
        return html_path


if __name__ == "__main__":
    refresh = "--refresh" in sys.argv
    db = StrategyDashboard(refresh=refresh)
    db.build()
