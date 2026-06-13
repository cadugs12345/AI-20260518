#!/usr/bin/env python3
"""
生成涨停突破 v6 — HTML信号报告（自包含版本）
修复: 信号日显示、重复信号去重、止损止盈价正确显示
"""
import json, os, time, pandas as pd

PROJ_B = "/mnt/d/AI-20260604"
OUTPUT_DIR = os.path.join(PROJ_B, "signals")
REPORT_DIR = os.path.join(PROJ_B, "alerts")
os.makedirs(REPORT_DIR, exist_ok=True)


def generate_html():
    t0 = time.time()
    
    # 加载信号CSV
    csv_path = os.path.join(OUTPUT_DIR, "v6_screener_latest.csv")
    pos_path = os.path.join(OUTPUT_DIR, "v6_positions.csv")
    summary_path = os.path.join(OUTPUT_DIR, "v6_signals_summary.json")
    
    df_signals = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
    df_pos = pd.read_csv(pos_path) if os.path.exists(pos_path) else pd.DataFrame()
    
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
    
    exit_signals = summary.get('exit_signals', [])
    positions_active = summary.get('positions_active', 0)
    run_time = summary.get('run_time', '--')
    next_date = summary.get('next_trade_date', '--')
    now = time.strftime('%Y-%m-%d %H:%M')
    
    # 构建买入信号HTML行（仅最近3个信号日）
    buy_rows = ''
    if len(df_signals) > 0:
        df_sorted = df_signals.sort_values('signal_date_str', ascending=False)
        recent_dates = set()
        for d in df_sorted['signal_date_str'].unique():
            if len(recent_dates) >= 3:
                break
            recent_dates.add(d)
        df_filtered = df_sorted[df_sorted['signal_date_str'].isin(recent_dates)]
        for _, row in df_filtered.iterrows():
            code = str(row.get('ts_code', ''))
            name = str(row.get('name', ''))
            sig = str(row.get('signal_date_str', ''))[:10]
            
            ep = row.get('entry_price_reference', 0)
            sp = row.get('stop_reference', 0)
            tp = row.get('target_reference', 0)
            
            # 过滤无效
            if not code or not name or pd.isna(ep) or ep == 0:
                continue
            
            # 判断是否过期（信号日距今超过3个交易日）
            is_expired = False
            try:
                sig_date = pd.Timestamp(sig)
                latest_date = pd.Timestamp(summary.get('latest_trade_date', run_time[:10]))
                trade_days_diff = len(pd.bdate_range(sig_date, latest_date))
                if trade_days_diff > 3:
                    is_expired = True
            except:
                pass
            
            if is_expired:
                continue  # 过期信号不展示
            
            buy_rows += f'''
            <tr>
                <td><span class="stock-code">{code}</span></td>
                <td>{name}</td>
                <td>{sig}</td>
                <td class="price-up">≤{ep:.2f}</td>
                <td class="price-down">{sp:.2f}</td>
                <td class="price-up">{tp:.2f}</td>
            </tr>'''
    else:
        buy_rows = '<tr><td colspan="6" class="empty">暂无信号</td></tr>'
    
    # 卖出信号行
    sell_rows = ''
    if exit_signals:
        for es in exit_signals:
            ret = es.get('ret_pct', 0)
            ret_tag = 'price-up' if ret > 0 else 'price-down'
            sell_rows += f'''
            <tr>
                <td><span class="stock-code">{es.get("ts_code","")}</span></td>
                <td>{str(es.get("buy_date",""))[:10]}</td>
                <td>{es.get("buy_price",0):.2f}</td>
                <td>{es.get("exit_price",0):.2f}</td>
                <td class="{ret_tag}">{ret:+.2f}%</td>
                <td>{es.get("hold_days",0)}天</td>
                <td><span class="tag">{es.get("exit_reason","")}</span></td>
            </tr>'''
    else:
        sell_rows = '<tr><td colspan="7" class="empty">✅ 暂无卖出提示</td></tr>'
    
    # 持仓行
    pos_rows = ''
    if len(df_pos) > 0:
        active = df_pos[df_pos['status'].fillna('持有').isin(['持有', ''])]
        if len(active) > 0:
            for _, row in active.iterrows():
                pos_rows += f'''
            <tr>
                <td><span class="stock-code">{row.get("ts_code","")}</span></td>
                <td>{row.get("name","")}</td>
                <td>{str(row.get("buy_date",""))[:10]}</td>
                <td>{row.get("buy_price",0):.2f}</td>
                <td class="price-down">{row.get("stop_price",0):.2f}</td>
                <td class="price-up">{row.get("target_price",0):.2f}</td>
                <td><span class="status-badge status-active">持有中</span></td>
            </tr>'''
        else:
            pos_rows = '<tr><td colspan="7" class="empty">📭 当前无持仓</td></tr>'
    else:
        pos_rows = '<tr><td colspan="7" class="empty">📭 当前无持仓记录</td></tr>'
    
    total_signals = len(buy_rows.split('</tr>')) - 1  # 粗略估计
    # 统计各板块数
    n_buy = total_signals
    n_exit = len(exit_signals)
    n_pos = len(df_pos)
    n_active = positions_active
    
    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>涨停突破 v6 信号报告</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
               background: #f0f2f5; color: #333; padding: 20px; }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   border-radius: 16px; padding: 28px; color: white; margin-bottom: 20px; }}
        .header h1 {{ font-size: 22px; margin-bottom: 6px; }}
        .header .meta {{ font-size: 13px; opacity: 0.85; }}
        .badge {{ display: inline-block; background: rgba(255,255,255,0.2);
                  padding: 3px 10px; border-radius: 20px; font-size: 12px; margin-top: 6px; margin-right: 6px; }}
        .card {{ background: white; border-radius: 12px; padding: 18px; margin-bottom: 14px;
                 box-shadow: 0 1px 6px rgba(0,0,0,0.05); }}
        .card-title {{ font-size: 15px; font-weight: 600; margin-bottom: 12px; display: flex; align-items: center; gap: 6px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th {{ text-align: left; padding: 8px 6px; border-bottom: 2px solid #eee; font-weight: 600; color: #666; font-size: 12px; }}
        td {{ padding: 8px 6px; border-bottom: 1px solid #f0f0f0; }}
        tr:hover {{ background: #f8f9ff; }}
        .stock-code {{ font-weight: 600; color: #1a73e8; font-family: monospace; }}
        .tag {{ display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 11px; font-weight: 600; }}
        .tag-profit {{ background: #e8f5e9; color: #2e7d32; }}
        .tag-stop {{ background: #fce4ec; color: #c62828; }}
        .tag-expire {{ background: #fff3e0; color: #e65100; }}
        .price-up {{ color: #d32f2f; font-weight: 500; }}
        .price-down {{ color: #2e7d32; font-weight: 500; }}
        .empty {{ text-align: center; padding: 20px; color: #999; font-size: 13px; }}
        .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 14px; }}
        .stat-box {{ background: #f8f9ff; border-radius: 10px; padding: 12px; text-align: center; }}
        .stat-box .num {{ font-size: 20px; font-weight: 700; color: #1a73e8; }}
        .stat-box .label {{ font-size: 11px; color: #888; margin-top: 3px; }}
        .status-badge {{ display: inline-block; padding: 1px 8px; border-radius: 8px; font-size: 11px; font-weight: 500; }}
        .status-active {{ background: #e8f5e9; color: #2e7d32; }}
        .footer {{ text-align: center; padding: 16px; color: #bbb; font-size: 11px; }}
        .tip-box {{ margin-top: 10px; padding: 10px 14px; background: #fff8e1; border-radius: 8px; font-size: 12px; line-height: 1.8; }}
        .signal-row td {{ border-left: 3px solid #4caf50; }}
        @media (max-width: 600px) {{
            .container {{ padding: 0; }}
            .header {{ padding: 16px; }}
            .stats {{ grid-template-columns: repeat(2, 1fr); }}
            table {{ font-size: 11px; }}
            th, td {{ padding: 4px 3px; }}
        }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>📊 涨停突破 v6 信号报告</h1>
        <div class="meta">报告生成: {now} | 数据截止: {run_time}</div>
        <div><span class="badge">🟢 次日关注: {next_date}</span><span class="badge">📈 沪深300 MA20过滤</span></div>
    </div>
    
    <div class="stats">
        <div class="stat-box"><div class="num">{n_buy}</div><div class="label">🟢 买入信号</div></div>
        <div class="stat-box"><div class="num">{n_exit}</div><div class="label">🔴 卖出提示</div></div>
        <div class="stat-box"><div class="num">{n_pos}</div><div class="label">📦 持仓记录</div></div>
        <div class="stat-box"><div class="num">{n_active}</div><div class="label">🏃 持有中</div></div>
    </div>

    <div class="card">
        <div class="card-title"><span style="color:#d32f2f;">●</span> 卖出提示</div>
        <table>{sell_rows}</table>
    </div>

    <div class="card">
        <div class="card-title"><span style="color:#1565c0;">●</span> 当前持仓</div>
        <table><tr><th>股票</th><th>名称</th><th>买入日</th><th>买入价</th><th>止损价</th><th>止盈价</th><th>状态</th></tr>{pos_rows}</table>
    </div>

    <div class="card">
        <div class="card-title"><span style="color:#2e7d32;">●</span> 买入信号（次日关注）</div>
        <table><tr><th>股票</th><th>名称</th><th>信号日</th><th>参考入场≤</th><th>止损参考</th><th>止盈目标</th></tr>{buy_rows}</table>
    </div>

    <div class="card">
        <div class="card-title"><span style="color:#f57c00;">●</span> 操作提示</div>
        <table><tr><th>步骤</th><th>说明</th></tr>
        <tr><td>1️⃣</td><td>开盘价接近BOLL下轨时低吸买入</td></tr>
        <tr><td>2️⃣</td><td>买入后追加记录到 <code>signals/v6_positions.csv</code></td></tr>
        <tr><td>3️⃣</td><td>每日系统自动检查止损/止盈</td></tr>
        <tr><td>4️⃣</td><td>触发后标记 <code>status=已平仓</code></td></tr>
        </table>
        <div class="tip-box">
            💡 买入条件：沪深300 MA20向上 + 全市场信号≥3家 + BOLL下轨低吸<br>
            📋 仓位：每只~9.5%，每日总仓位≤60%<br>
            🛑 止损：max(涨停最低×0.95, ATR追踪, BOLL下轨) | 止盈：+20% | 最长30日
        </div>
    </div>
    <div class="footer">涨停突破 v6 策略 · 自动生成</div>
</div>
</body>
</html>'''
    
    out_path = os.path.join(REPORT_DIR, "v6_signals_report.html")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ HTML报告: {out_path} ({len(html)}B)")
    print(f"⏱ {time.time()-t0:.1f}秒")


if __name__ == "__main__":
    generate_html()
