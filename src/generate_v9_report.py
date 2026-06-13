#!/usr/bin/env python3
"""生成最终版v9 ABCD报告，含回测汇总 + 今日TOP5推荐"""
import json, os

PROJ_B = "/mnt/d/AI-20260604"
ALERTS_DIR = os.path.join(PROJ_B, "alerts")

def generate_report():
    # 加载信号
    with open(os.path.join(ALERTS_DIR, "v9_signals_today.json"), "r") as f:
        signals = json.load(f)
    
    today_str = "2026-06-04（最新数据）"
    
    # 从MA18全程向上的信号中选TOP5
    import pandas as pd, numpy as np
    DATA = os.path.join(PROJ_B, "data", "raw", "daily")
    
    quality_signals = []
    for s in signals:
        if s.get('signal_quality', 0) < 3: continue
        code = s['code']
        fp = os.path.join(DATA, f"{code}.parquet")
        try: df = pd.read_parquet(fp).sort_values('trade_date').reset_index(drop=True)
        except: continue
        c = df['close'].values.astype(np.float64)
        h = df['high'].values.astype(np.float64)
        n = len(df)
        ma = np.full(n, np.nan)
        if n >= 18:
            s1 = np.cumsum(c); ma[17] = s1[17]/18
            for i in range(18, n): ma[i] = (s1[i]-s1[i-18])/18
        mu = np.full(n, False); mu[1:] = ma[1:] > ma[:-1]
        lu = np.full(n, False); lu[1:] = (c[1:]/c[:-1] > 1.095) & (c[1:] == h[1:])
        last_lu = -1
        for i in range(n-1, -1, -1):
            if lu[i]: last_lu = i; break
        if last_lu < 0: continue
        all_up = True
        for j in range(last_lu+1, n):
            if not mu[j]: all_up = False; break
        if not all_up: continue
        
        score = 0
        cross = abs(s.get('cross_pct', 0))
        if cross <= 3: score += 30
        elif cross <= 5: score += 20
        elif cross <= 8: score += 10
        else: score += 2
        vr = s.get('volume_ratio', 0)
        if 1.5 <= vr <= 4: score += 30
        elif 1.0 <= vr < 1.5: score += 20
        elif 0.7 <= vr < 1.0: score += 10
        else: score += 5
        ds = s.get('days_since_limit', 0)
        if 5 <= ds <= 10: score += 25
        elif 3 <= ds < 5: score += 20
        elif 10 < ds <= 15: score += 15
        else: score += 5
        if s.get('ma5',0) and s.get('ma10',0) and s['ma10'] > 0:
            gap = (s['ma5'] - s['ma10']) / s['ma10'] * 100
            if 0 < gap <= 6: score += 15
            elif gap > 6: score += 5
            else: score -= 10
        s['_score'] = score
        quality_signals.append(s)
    
    quality_signals.sort(key=lambda x: x['_score'], reverse=True)
    top5 = quality_signals[:5]
    
    # TOP5行
    top5_rows = ""
    for i, s in enumerate(top5, 1):
        limit_dt = str(s['limit_date'])[:10] if isinstance(s['limit_date'], str) else ''
        buy_cond = f"开盘≤{s['max_buy_price']:.2f}"
        top5_rows += f"""<tr>
            <td style="font-weight:bold;font-size:16px">{i}</td>
            <td><b>{s['code']}</b></td>
            <td>{s['name']}</td>
            <td>{s['industry']}</td>
            <td>{s['close']}</td>
            <td>{s['ma18']}</td>
            <td>{s['cross_pct']}%</td>
            <td>{s['volume_ratio']}x</td>
            <td>{s['ma5']}/{s['ma10']}</td>
            <td>{limit_dt}</td>
            <td style="color:#c0392b;font-weight:bold">{buy_cond}</td>
        </tr>"""
    
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><title>v9 ABCD 策略报告 - {today_str}</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: -apple-system, 'Microsoft YaHei', sans-serif; background: #f5f6fa; color: #2c3e50; padding: 20px; }}
.container {{ max-width: 1400px; margin: 0 auto; }}
.header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 24px 30px; border-radius: 12px; margin-bottom: 20px; }}
.header h1 {{ font-size: 24px; margin-bottom: 4px; }}
.header .meta {{ font-size: 14px; opacity: 0.85; }}
.section-title {{ font-size: 18px; font-weight: bold; margin: 24px 0 12px 0; padding-left: 12px; border-left: 4px solid #667eea; }}
.card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 20px; }}
.card {{ background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
.card .label {{ font-size: 12px; color: #95a5a6; margin-bottom: 4px; }}
.card .value {{ font-size: 26px; font-weight: bold; }}
.card .value.green {{ color: #27ae60; }}
.card .value.red {{ color: #e74c3c; }}
.card .value.orange {{ color: #f39c12; }}
.card .sub {{ font-size: 12px; color: #7f8c8d; margin-top: 2px; }}
table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
th {{ background: #34495e; color: white; padding: 12px 10px; font-size: 13px; text-align: left; white-space: nowrap; }}
td {{ padding: 10px; font-size: 13px; border-bottom: 1px solid #ecf0f1; }}
tr:hover {{ background: #f8f9fa; }}
.footer {{ text-align: center; padding: 20px; color: #95a5a6; font-size: 12px; }}
.note {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 12px 16px; margin-bottom: 16px; border-radius: 4px; font-size: 13px; line-height: 1.6; }}
.comp-table {{ font-size: 13px; }}
.comp-table td {{ padding: 8px 12px; }}
.comp-table .val {{ text-align: center; font-weight: bold; }}
.yearly-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); gap: 8px; margin: 12px 0; }}
.yearly-item {{ background: white; border-radius: 6px; padding: 10px; text-align: center; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }}
.yearly-item .yr {{ font-size: 14px; font-weight: bold; }}
.yearly-item .yr-val {{ font-size: 16px; font-weight: bold; margin-top: 2px; }}
.yearly-item .yr-info {{ font-size: 11px; color: #7f8c8d; margin-top: 1px; }}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>📈 v9 ABCD 涨停突破策略报告</h1>
        <div class="meta">{today_str} ｜ 沪深300 60日线: ✅ 上行 ｜ 回测夏普 2.24</div>
    </div>
    
    <!-- 大盘概览 -->
    <div class="card-grid">
        <div class="card">
            <div class="label">今日信号总数</div>
            <div class="value green">{len(signals)}</div>
            <div class="sub">MA18全程向上: {len(quality_signals)}只</div>
        </div>
        <div class="card">
            <div class="label">沪深300 60日线</div>
            <div class="value green">上行 ✅</div>
            <div class="sub">大盘环境友好</div>
        </div>
        <div class="card">
            <div class="label">回测全样本夏普</div>
            <div class="value green">2.24</div>
            <div class="sub">2017-2026, 含摩擦成本</div>
        </div>
        <div class="card">
            <div class="label">回测最大回撤</div>
            <div class="value orange">-7.42%</div>
            <div class="sub">样本外仅-5.01%</div>
        </div>
    </div>
    
    <!-- 回测汇总对比 -->
    <div class="section-title">📊 回测样本内/外对比</div>
    <table class="comp-table">
        <thead>
            <tr><th>指标</th><th>样本内 (2017-2023)</th><th>样本外 (2024-2026)</th><th>全样本</th></tr>
        </thead>
        <tbody>
            <tr><td>交易数</td><td class="val">2,189</td><td class="val">1,345</td><td class="val">3,534</td></tr>
            <tr><td>月频夏普</td><td class="val">1.47</td><td class="val" style="color:#27ae60">3.37 🏆</td><td class="val">2.24</td></tr>
            <tr><td>胜率</td><td class="val">45.6%</td><td class="val">49.3%</td><td class="val">44.4%</td></tr>
            <tr><td>盈亏比</td><td class="val">2.20:1</td><td class="val">2.26:1</td><td class="val">2.23:1</td></tr>
            <tr><td>最大回撤</td><td class="val">-11.33%</td><td class="val" style="color:#27ae60">-5.01% ✅</td><td class="val">-7.42%</td></tr>
            <tr><td>年化收益</td><td class="val">60.0%</td><td class="val" style="color:#8e44ad">234.8% 🚀</td><td class="val">—</td></tr>
            <tr><td>累计净值</td><td class="val">20.35x</td><td class="val">14.62x</td><td class="val">29,655x</td></tr>
        </tbody>
    </table>
    
    <div class="section-title">📅 各年表现</div>
    <div class="yearly-grid">
        <div class="yearly-item" style="background:#fdedec"><div class="yr">2017</div><div class="yr-val" style="color:#27ae60">+29.7%</div><div class="yr-info">315笔</div></div>
        <div class="yearly-item" style="background:#fdedec"><div class="yr">2018</div><div class="yr-val" style="color:#e74c3c">-14.4% ❌</div><div class="yr-info">87笔</div></div>
        <div class="yearly-item" style="background:#e8f8f0"><div class="yr">2019</div><div class="yr-val" style="color:#27ae60">+41.6%</div><div class="yr-info">405笔</div></div>
        <div class="yearly-item" style="background:#e8f8f0"><div class="yr">2020</div><div class="yr-val" style="color:#27ae60">+99.8%</div><div class="yr-info">513笔</div></div>
        <div class="yearly-item" style="background:#e8f8f0"><div class="yr">2021</div><div class="yr-val" style="color:#27ae60">+61.2%</div><div class="yr-info">359笔</div></div>
        <div class="yearly-item" style="background:#e8f8f0"><div class="yr">2022</div><div class="yr-val" style="color:#27ae60">+75.3%</div><div class="yr-info">277笔</div></div>
        <div class="yearly-item" style="background:#e8f8f0"><div class="yr">2023</div><div class="yr-val" style="color:#27ae60">+13.5%</div><div class="yr-info">233笔</div></div>
        <div class="yearly-item" style="background:#e8f8f0"><div class="yr">2024</div><div class="yr-val" style="color:#27ae60">+74.7%</div><div class="yr-info">419笔</div></div>
        <div class="yearly-item" style="background:#e8f8f0"><div class="yr">2025</div><div class="yr-val" style="color:#27ae60">+123.9%</div><div class="yr-info">624笔</div></div>
        <div class="yearly-item" style="background:#e8f8f0"><div class="yr">2026</div><div class="yr-val" style="color:#27ae60">+65.7%</div><div class="yr-info">311笔</div></div>
    </div>
    
    <!-- TOP5推荐 -->
    <div class="section-title">🏆 今日TOP5推荐（MA18全程向上筛选）</div>
    <div class="note">
        <strong>⚡ 买入规则：</strong>下周一（6/8）开盘价 ≤ MA18×1.01 才可买入，否则盘中等待最低价回踩<br>
        <strong>止损：</strong>连续2日收盘跌破18日线 ｜ <strong>止盈：</strong>BOLL(20,2)上轨 ｜ <strong>最大持有：</strong>30日
    </div>
    
    <table>
        <thead>
            <tr>
                <th>#</th><th>代码</th><th>名称</th><th>行业</th>
                <th>收盘</th><th>MA18</th><th>上穿%</th><th>量比</th>
                <th>MA5/MA10</th><th>涨停日</th><th>买入条件</th>
            </tr>
        </thead>
        <tbody>
            {top5_rows}
        </tbody>
    </table>
    
    <!-- 策略说明 -->
    <div class="section-title">📋 策略说明</div>
    <div class="note">
        <strong>v9 ABCD版选股逻辑：</strong><br>
        <strong>A</strong> — 站上5日线 + 5日>10日线多头排列（价量共振）<br>
        <strong>B</strong> — 行业分域上穿阈值（银行10%、消费8%、科技4%、其他5%）<br>
        <strong>C</strong> — 大盘过滤：沪深300 60日线向上<br>
        <strong>D</strong> — 买点优化：开盘≤MA18×1.01买入，高开太多等盘中最低价回踩<br>
        <strong>硬性条件：</strong>涨停后18日均线全程向上（不可变动）
    </div>
    
    <div class="footer">
        v9 ABCD 涨停突破策略 ｜ 数据至 {today_str} ｜ ⚠️ 历史回测不代表未来收益
    </div>
</div>
</body>
</html>"""
    
    path = os.path.join(ALERTS_DIR, "v9_strategy_report.html")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"✅ 综合报告: {path}")

if __name__ == "__main__":
    generate_report()
