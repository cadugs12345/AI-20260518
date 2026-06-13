"""
📊 每日 9:30 开盘报告 — 模型信号质量 + 关键因子状态

核心逻辑：
  - 报告的是 v38 模型过去N个信号日选出的 Top10 组合的 未来20日平均收益
  - 这不是交易收益，而是衡量模型「选股能力」的指标
  - fwd_20d_ret > 0 且越高 ⇨ 模型选出的股票未来20天涨得越好
  - 结合因子预警，判断当前市场风格是否有利于模型

产出：
  1. 模型近期信号质量（Top10在最近信号日的平均未来20日收益）
  2. 当前持仓信号（如有）
  3. 因子状态简表（仅显示红色/黄色预警因子）
  4. HTML版 + 终端文本版
"""

import sys, os, json, time
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

PROJECT = "/mnt/d/AI-20260518"
os.chdir(PROJECT)

OUTPUT_DIR = "alerts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TODAY = pd.Timestamp.now().strftime("%Y-%m-%d")
TODAY_OBJ = pd.Timestamp.now()


def load_model_signal():
    """加载最新v38信号"""
    signal_path = "signals/latest_signal.json"
    if not os.path.exists(signal_path):
        return None
    try:
        with open(signal_path) as f:
            return json.load(f)
    except:
        return None


def load_backtest_results():
    """加载滚动回测结果（样本外），优先用最新的"""
    rolling_path = "backtest_results/v38_rolling_top10_2026.parquet"
    retro_path = "backtest_results/v38_daily_top10_retro.parquet"
    
    if os.path.exists(rolling_path):
        df = pd.read_parquet(rolling_path)
        return df, "rolling"
    if os.path.exists(retro_path):
        df = pd.read_parquet(retro_path)
        return df, "retro"
    return None, None


def calc_signal_quality(df, date_col='date', ret_col='avg_fwd_ret', n_last=20):
    """
    计算过去n个信号日的信号质量。
    fwd_20d_ret = 信号日买入Top10并持有20日后的平均收益率
    """
    recent = df.tail(n_last).copy()
    if len(recent) == 0:
        return None
    
    rets = recent[ret_col].values
    
    # 信号质量统计
    mean_ret = float(np.mean(rets))
    median_ret = float(np.median(rets))
    std_ret = float(np.std(rets))
    win_rate = float(np.mean(rets > 0))
    best = float(np.max(rets))
    worst = float(np.min(rets))
    
    # Score与收益的相关性（衡量预测能力）
    if 'avg_score' in recent.columns:
        score_ret_corr = float(recent['avg_score'].corr(recent[ret_col]))
    else:
        score_ret_corr = 0
    
    # 各信号日详情
    details = []
    for _, row in recent.iterrows():
        d = {
            'date': str(row[date_col])[:10],
            'fwd_ret_pct': round(float(row[ret_col]) * 100, 2),
        }
        if 'avg_score' in recent.columns:
            d['score'] = round(float(row['avg_score']), 4)
        details.append(d)
    
    return {
        "n_signals": len(recent),
        "mean_fwd_ret": round(mean_ret * 100, 2),
        "median_fwd_ret": round(median_ret * 100, 2),
        "std_fwd_ret": round(std_ret * 100, 2),
        "win_rate": round(win_rate * 100, 1),
        "best_fwd_ret": round(best * 100, 2),
        "worst_fwd_ret": round(worst * 100, 2),
        "score_ret_corr": round(score_ret_corr, 4),
        "details": details[-10:],  # 只显示最近10条
    }


def load_alert_factors():
    """从因子预警缓存中提取红色/黄色的因子"""
    factors_path = "alerts/latest.json"
    if not os.path.exists(factors_path):
        return []
    
    try:
        with open(factors_path) as f:
            meta = json.load(f)
        
        report_path = meta.get("path", "")
        if not report_path or not os.path.exists(report_path):
            return []
        
        with open(report_path) as f:
            report = json.load(f)
        
        alerts = []
        for a in report.get("alerts", []):
            level = a.get("level", "")
            ir = round(a.get("recent_ic_ir", 0), 2)
            if "严重" in level:
                alerts.append({
                    "factor": a.get("factor", ""),
                    "level": level,
                    "ic_ir": ir,
                    "severity": "critical",
                    "description": f"因子IC严重衰减 (IR={ir:+.2f})，该因子近期选股能力显著下降"
                })
            elif "显著" in level:
                alerts.append({
                    "factor": a.get("factor", ""),
                    "level": level,
                    "ic_ir": ir,
                    "severity": "warning",
                    "description": f"因子IC在衰减 (IR={ir:+.2f})，选股能力在变弱"
                })
            elif "关注" in level:
                alerts.append({
                    "factor": a.get("factor", ""),
                    "level": level,
                    "ic_ir": ir,
                    "severity": "info",
                    "description": f"因子IC有下滑趋势 (IR={ir:+.2f})"
                })
        
        return alerts
    except:
        return []


def build_html_report(sig_20, sig_60, alerts, signal):
    """生成HTML报告"""
    
    # 信号质量看板
    quality_html = ""
    if sig_20:
        s = sig_20
        # 状态指示
        if s['mean_fwd_ret'] > 10:
            status = "🟢 优秀"
            status_color = "#27ae60"
        elif s['mean_fwd_ret'] > 5:
            status = "🟡 良好"
            status_color = "#e67e22"
        elif s['mean_fwd_ret'] > 0:
            status = "🟠 一般"
            status_color = "#e74c3c"
        else:
            status = "🔴 失效"
            status_color = "#c0392b"
        
        quality_html = f'''
        <div class="status-banner" style="background:{status_color}10; border-left:4px solid {status_color}; padding:12px 16px; margin-bottom:16px; border-radius:6px;">
            <div style="font-size:18px; font-weight:600;">信号质量: {status}</div>
            <div style="font-size:13px; color:#666; margin-top:4px;">过去{s["n_signals"]}个信号日 · Top10未来20日平均收益 {s["mean_fwd_ret"]:+.2f}%</div>
        </div>
        
        <div class="perf-grid">
            <div class="perf-item"><span class="label">信号日数</span><span class="value">{s["n_signals"]}</span></div>
            <div class="perf-item"><span class="label">平均fwd_ret</span><span class="value {'up' if s['mean_fwd_ret']>0 else 'down'}">{s["mean_fwd_ret"]:+.2f}%</span></div>
            <div class="perf-item"><span class="label">中位数fwd_ret</span><span class="value">{s["median_fwd_ret"]:+.2f}%</span></div>
            <div class="perf-item"><span class="label">标准差</span><span class="value">{s["std_fwd_ret"]:.2f}%</span></div>
            <div class="perf-item"><span class="label">胜率</span><span class="value">{s["win_rate"]}%</span></div>
            <div class="perf-item"><span class="label">最好/最差</span><span class="value">{s["best_fwd_ret"]:+.1f}% / {s["worst_fwd_ret"]:+.1f}%</span></div>
            <div class="perf-item"><span class="label">Score-收益相关性</span><span class="value">{s["score_ret_corr"]:.4f}</span></div>
        </div>
        '''
    
    if sig_60:
        quality_html += f'''
        <div class="perf-card" style="margin-top:12px;">
            <h3>📊 最近60日信号质量</h3>
            <div class="perf-grid">
                <div class="perf-item"><span class="label">信号日数</span><span class="value">{sig_60["n_signals"]}</span></div>
                <div class="perf-item"><span class="label">平均fwd_ret</span><span class="value {'up' if sig_60['mean_fwd_ret']>0 else 'down'}">{sig_60["mean_fwd_ret"]:+.2f}%</span></div>
                <div class="perf-item"><span class="label">中位数fwd_ret</span><span class="value">{sig_60["median_fwd_ret"]:+.2f}%</span></div>
                <div class="perf-item"><span class="label">胜率</span><span class="value">{sig_60["win_rate"]}%</span></div>
                <div class="perf-item"><span class="label">最好</span><span class="value up">{sig_60["best_fwd_ret"]:+.1f}%</span></div>
                <div class="perf-item"><span class="label">最差</span><span class="value down">{sig_60["worst_fwd_ret"]:+.1f}%</span></div>
            </div>
        </div>'''
    
    # 近10个信号日详情
    detail_rows = ""
    if sig_20 and sig_20.get("details"):
        detail_rows = '<table><thead><tr><th>信号日</th><th>Score均值</th><th>Top10未来20日收益</th></tr></thead><tbody>'
        for d in reversed(sig_20["details"]):
            rev = d['fwd_ret_pct']
            color = "#e74c3c" if rev > 0 else "#27ae60"
            score_str = f'{d["score"]:.4f}' if d.get('score') else '-'
            detail_rows += f'<tr><td>{d["date"]}</td><td>{score_str}</td><td style="color:{color}">{rev:+.2f}%</td></tr>'
        detail_rows += '</tbody></table>'
    
    # 预警
    alerts_html = '<div class="perf-card"><h3>⚡ 因子预警</h3>'
    if alerts:
        for a in alerts:
            icon = "🔴" if a["severity"] == "critical" else ("🟡" if a["severity"] == "warning" else "ℹ️")
            color = "#e74c3c" if a["severity"] == "critical" else "#e67e22"
            alerts_html += f'<div class="alert-row" style="border-left:3px solid {color}; padding:6px 10px; margin:6px 0; background:{color}08;"><strong>{icon} {a["factor"]}</strong><br><span style="font-size:12px;color:#888;">{a["description"]}</span></div>'
    else:
        alerts_html += '<div style="color:#27ae60; padding:6px 0;">🟢 所有核心因子正常</div>'
    alerts_html += '</div>'
    
    # 当前信号（如有）
    signal_html = ""
    if signal:
        rows = ""
        for i, p in enumerate(signal.get("positions", [])[:10]):
            color = "#e74c3c" if i < 3 else "#333"
            rows += f'<tr><td style="color:{color};font-weight:600">#{p["rank"]}</td><td>{p.get("ts_code","")}</td><td>{p.get("name","")}</td><td>{p.get("industry","")}</td><td>{p.get("weight","")}%</td><td>{p.get("score","")}</td></tr>'
        signal_html = f'''
        <div class="perf-card">
            <h3>📋 当前持仓信号 (v38) — {signal.get("date","?")}</h3>
            <table><thead><tr><th>#</th><th>代码</th><th>名称</th><th>行业</th><th>权重</th><th>Score</th></tr></thead><tbody>{rows}</tbody></table>
        </div>'''
    
    # ===== 整体HTML =====
    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><title>📊 开盘报告 — {TODAY}</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: -apple-system, 'Microsoft YaHei', 'PingFang SC', sans-serif; max-width: 920px; margin: 0 auto; padding: 20px 16px; background: #f5f5f5; color: #333; }}
h1 {{ font-size: 22px; margin-bottom: 2px; }}
h2 {{ font-size: 15px; color: #888; margin: 18px 0 10px; border-bottom: 1px solid #e0e0e0; padding-bottom: 6px; }}
.perf-card {{ background: white; border-radius: 10px; padding: 14px 16px; margin-bottom: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }}
.perf-card h3 {{ font-size: 14px; color: #666; margin: 0 0 10px; font-weight: 600; }}
.perf-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px 16px; }}
.perf-item {{ display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid #f3f3f3; font-size: 13px; }}
.label {{ color: #999; }}
.value {{ font-weight: 600; }}
.up {{ color: #e74c3c; }}
.down {{ color: #27ae60; }}
table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
th {{ background: #f8f8f8; padding: 7px 8px; text-align: left; font-weight: 600; color: #666; border-bottom: 1px solid #ddd; }}
td {{ padding: 6px 8px; border-bottom: 1px solid #eee; }}
.alert-row {{ font-size: 13px; border-radius: 4px; }}
.footer {{ text-align: center; color: #bbb; font-size: 11px; margin-top: 24px; }}
</style></head>
<body>
<h1>📊 开盘报告</h1>
<p style="color:#999;font-size:13px;margin-bottom:16px;">{TODAY} 9:30 · 基于滚动样本外回测</p>

<h2>📈 模型信号质量</h2>
{quality_html}

<h2>📋 近10个信号日详情</h2>
<div class="perf-card">
{detail_rows if detail_rows else '<p style="color:#999">暂无回测数据</p>'}
</div>

{signal_html}

{alerts_html}

<div class="footer">v38 LightGBM + rank标签 · 信号质量 = Top10组合的未来20日平均收益 · 自动生成 {time.strftime("%Y-%m-%d %H:%M")}</div>
</body></html>'''
    
    report_path = os.path.join(OUTPUT_DIR, f"morning_report_{TODAY}.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    latest_path = os.path.join(OUTPUT_DIR, "latest_morning_report.html")
    with open(latest_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    return report_path


def print_report_text(sig_20, sig_60, alerts, signal):
    """终端文本版"""
    print()
    print("=" * 60)
    print(f"📊 开盘报告 — {TODAY}")
    print("=" * 60)
    
    if sig_20:
        s = sig_20
        # 状态
        if s['mean_fwd_ret'] > 10:
            status = "🟢 信号优秀"
        elif s['mean_fwd_ret'] > 5:
            status = "🟡 信号良好"
        elif s['mean_fwd_ret'] > 0:
            status = "🟠 信号一般"
        else:
            status = "🔴 信号失效"
        
        print(f"\n{status} (最近{s['n_signals']}个信号日)")
        print(f"  Top10未来20日平均收益: {s['mean_fwd_ret']:+.2f}%")
        print(f"  中位数: {s['median_fwd_ret']:+.2f}%  胜率: {s['win_rate']}%")
        print(f"  Score-收益相关性: {s['score_ret_corr']:.4f}")
        print(f"  最好: {s['best_fwd_ret']:+.2f}%  |  最差: {s['worst_fwd_ret']:+.2f}%")
        
        # 最近5日
        if s.get("details"):
            print(f"\n  最近信号日详情:")
            for d in s["details"][-5:]:
                icon = "✅" if d['fwd_ret_pct'] > 0 else "❌"
                score_str = f' score={d["score"]:.4f}' if d.get('score') else ''
                print(f"    {icon} {d['date']}{score_str} → fwd_ret={d['fwd_ret_pct']:+.2f}%")
    
    if sig_60 and sig_60['n_signals'] > 20:
        s = sig_60
        print(f"\n📊 最近60日 ({s['n_signals']}个信号日):")
        print(f"  Top10未来20日均值: {s['mean_fwd_ret']:+.2f}%  胜率: {s['win_rate']}%")
    
    # 信号
    if signal:
        print(f"\n📋 当前持仓 (v38 {signal.get('date','?')}):")
        for p in signal.get("positions", [])[:5]:
            print(f"  {p['rank']}. {p['ts_code']:>10s} {p.get('name',''):8s} {p.get('industry',''):8s} {p['weight']}% score={p['score']:.4f}")
    
    # 预警
    if alerts:
        print(f"\n⚡ 因子预警:")
        for a in alerts:
            icon = "🔴" if a['severity'] == 'critical' else ("🟡" if a['severity'] == 'warning' else "ℹ️")
            print(f"  {icon} {a['factor']:18s} {a['level']:8s} (IR={a['ic_ir']:+.2f})")
            print(f"     {a.get('description','')}")
    else:
        print(f"\n🟢 因子状态: 全部正常")
    
    print(f"\n  💡 Top10 fwd_ret 含义: 信号日选出的Top10组合在未来20日的平均涨幅")
    print(f"  >10%=信号优秀   >5%=良好   >0%=一般   <0%=信号失效")
    print(f"\n{'='*60}")
    return True


def main():
    print(f"🔍 生成开盘报告...")
    
    # 1. 回测数据
    bt_df, bt_type = load_backtest_results()
    sig_20 = None
    sig_60 = None
    
    if bt_df is not None:
        date_col = 'date' if 'date' in bt_df.columns else 'trade_date'
        ret_col = 'avg_fwd_ret' if 'avg_fwd_ret' in bt_df.columns else 'top10_avg_fwd_ret'
        print(f"  回测: {bt_type}, {len(bt_df)}天 ({bt_df[date_col].min()}~{bt_df[date_col].max()})")
        sig_20 = calc_signal_quality(bt_df, date_col, ret_col, 20)
        sig_60 = calc_signal_quality(bt_df, date_col, ret_col, 60)
    
    # 2. 预警
    alerts = load_alert_factors()
    print(f"  预警因子: {len(alerts)}个")
    
    # 3. 最新信号
    signal = load_model_signal()
    if signal:
        print(f"  最新信号: {signal.get('date','?')}")
    
    # 4. 输出
    print_report_text(sig_20, sig_60, alerts, signal)
    
    # 5. HTML
    report_path = build_html_report(sig_20, sig_60, alerts, signal)
    print(f"  HTML: {report_path}")
    
    return 0


if __name__ == "__main__":
    main()
