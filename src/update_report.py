#!/usr/bin/env python3
"""从头生成 daily_operation.html：清空历史，只跟踪今日V3信号"""
import json, os

PROJ = "/mnt/d/AI-20260604"
signals = json.load(open(os.path.join(PROJ, "signals", "v3_signals_latest.json")))
# 过滤：只保留距MA18 >= -3% 的可关注股票
signals = [s for s in signals if s['dist_pct'] >= -3]

# V3表格行
v3_rows = []
for i, s in enumerate(signals, 1):
    d = s['dist_pct']
    ref = round(s['ma18'] * 1.01, 2)
    v3_rows.append(
        '<tr class="v3-buy"><td>' + str(i) + '</td><td><b>' + s['code'] + '</b></td><td>' + s['name'] + '</td><td style="color:#666">' + s['industry'][:6] + '</td>'
        '<td style="color:#27ae60;font-weight:bold">' + str(d) + '%</td><td class="price">' + str(s['price']) + '</td>'
        '<td class="price">' + str(s['ma18']) + '</td><td class="price">不高于' + str(ref) + '</td><td>' + s['mark_date'] + '</td>'
        '<td><span class="tag tag-buy">可关注</span></td><td style="font-size:12px;color:#999">接近MA18</td></tr>')
v3_table = "\n".join(v3_rows)

# V3买入推荐（操作清单用）
v3_rec = []
for s in signals:
    ref = round(s['ma18']*1.01, 2)
    v3_rec.append(
        '<li class="track-li"><span class="step">  &#x1F50D;</span> <b>' + s['code'] + ' ' + s['name']
        + '</b> 现价' + str(s['price']) + ' 距MA18 ' + str(s['dist_pct'])
        + '% 参考买入不高于' + str(ref) + '</li>')
v3_rec_html = "\n".join(v3_rec)

html = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>涨停股低吸策略 - 每日操作报告 - 2026-06-08</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Microsoft YaHei','PingFang SC',sans-serif;background:#f0f2f5;color:#333;line-height:1.6}
.container{max-width:1200px;margin:0 auto;padding:20px}
.header{background:linear-gradient(135deg,#1a56db,#0d3b9e);color:#fff;padding:24px 32px;border-radius:12px;margin-bottom:20px}
.header h1{font-size:26px;margin-bottom:4px}
.header .meta{font-size:14px;opacity:0.85}
.header .scorecard{display:flex;gap:24px;margin-top:16px;flex-wrap:wrap}
.header .card{background:rgba(255,255,255,0.15);border-radius:10px;padding:14px 20px;text-align:center;min-width:100px}
.header .card .num{font-size:32px;font-weight:bold;display:block}
.header .card .lbl{font-size:12px;opacity:0.8}
.market{background:#fff;border-radius:10px;padding:16px 24px;margin-bottom:20px;display:flex;gap:24px;align-items:center;flex-wrap:wrap;box-shadow:0 1px 3px rgba(0,0,0,0.06)}
.market .item{font-size:15px}
.good{color:#dc2626} .bad{color:#16a34a} .warn{color:#e67e22}
.section{background:#fff;border-radius:10px;padding:24px;margin-bottom:20px;box-shadow:0 1px 3px rgba(0,0,0,0.06)}
.section h2{font-size:20px;margin-bottom:16px;padding-bottom:10px;border-bottom:2px solid #e5e7eb;display:flex;align-items:center;gap:8px}
.section h2 .icon{font-size:24px}
.section h2 .count{font-size:13px;color:#94a3b8;font-weight:normal;margin-left:8px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{background:#f8fafc;text-align:left;padding:10px 12px;font-weight:600;color:#64748b;border-bottom:2px solid #e2e8f0;white-space:nowrap}
td{padding:9px 12px;border-bottom:1px solid #f1f5f9}
tr:hover td{background:#f8fafc}
tr.v3-buy td{background:#f0fdf4!important}
.tag{display:inline-block;padding:2px 10px;border-radius:12px;font-size:11px;font-weight:600}
.tag-buy{background:#dcfce7;color:#16a34a}
.tag-sell{background:#fef2f2;color:#dc2626}
.tag-hold{background:#eff6ff;color:#2563eb}
.tag-done{background:#f1f5f9;color:#94a3b8}
.tag-warn{background:#fef9c3;color:#a16207}
.checklist{list-style:none}
.checklist li{padding:10px 16px;margin:6px 0;border-radius:8px;font-size:15px;display:flex;align-items:center;gap:10px}
.checklist .hold-li{background:#eff6ff;border-left:4px solid #2563eb}
.checklist .track-li{background:#fefce8;border-left:4px solid #eab308}
.checklist .sell-li{background:#fef2f2;border-left:4px solid #dc2626}
.checklist .step{min-width:80px;font-weight:bold}
.price{font-family:'Consolas','Monaco',monospace;font-weight:600}
.num-up{color:#dc2626;font-weight:600} .num-down{color:#16a34a;font-weight:600}
.ref{font-size:12px;color:#94a3b8;margin-top:4px}
.footer{text-align:center;color:#94a3b8;font-size:12px;padding:20px}
</style>
</head>
<body>
<div class="container">
<div class="header">
  <h1>&#x1F4CA; 涨停股低吸策略</h1>
  <div class="meta">报告生成: 2026-06-08 23:15 | 最新交易日: 2026-06-08 | 下一个交易日: 2026-06-09</div>
  <div class="scorecard">
    <div class="card"><span class="num">0</span><span class="lbl">&#x1F4CC; 持仓</span></div>
    <div class="card"><span class="num">''' + str(len(signals)) + '''</span><span class="lbl">&#x1F50D; V3信号</span></div>
    <div class="card"><span class="num">0</span><span class="lbl">&#x2705; 历史成交</span></div>
  </div>
</div>

<div class="market">
  <div class="item">&#x1F4C8; 沪深300 MA60: <span class="good">向上 &#x2705;</span></div>
  <div class="item">&#x1F4CA; MACD柱: <span class="good">&gt;0 &#x2705;</span></div>
  <div class="item">&#x1F4A1; 大盘状态: <span class="good">&#x1F7E2; 可以做多</span></div>
</div>

<!-- V3信号 -->
<div class="section">
<h2><span class="icon">&#x1F50D;</span> V3 低吸信号 <span class="count">共''' + str(len(signals)) + '''只 | 距MA18越近越优先</span></h2>
<p class="ref">策略：标志K线&#x2192;回调破MA18&#x2192;站上MA18&#x2192;MA18向上&#x2192;回踩MA18附近买入 | 回测胜率72% 盈亏比3.02</p>
<p class="ref">&#x26A0;&#xFE0F; 买入价不可预判。买入区仅供参考，需盘中观察分时走势在支撑位附近择机入场。</p>
<table>
<tr><th>#</th><th>代码</th><th>名称</th><th>行业</th><th>距MA18</th><th>现价</th><th>MA18</th><th>参考买入区</th><th>标志日</th><th>状态</th><th>说明</th></tr>
''' + v3_table + '''
</table>
</div>

<!-- 持仓 -->
<div class="section">
<h2><span class="icon">&#x1F4CC;</span> 持仓跟踪</h2>
<p class="ref">今日无持仓。V3信号将于盘中价格触发买入条件时记录为持仓。</p>
<div style="text-align:center;padding:40px;color:#94a3b8;font-size:14px">
暂无持仓数据
</div>
</div>

<!-- 历史成交 -->
<div class="section">
<h2><span class="icon">&#x2705;</span> 历史成交</h2>
<div style="text-align:center;padding:40px;color:#94a3b8;font-size:14px">
暂无历史成交记录
</div>
</div>

<!-- 操作清单 -->
<div class="section">
<h2><span class="icon">&#x1F4CB;</span> 操作清单（2026-06-09）</h2>
<ul class="checklist">
<li class="track-li"><span class="step">&#x1F50D; 关注</span> ''' + str(len(signals)) + '''只V3信号，明日盘中关注MA18附近买入机会</li>
''' + v3_rec_html + '''
</ul>
</div>

<div class="footer">
  涨停股低吸策略 | 每日操作报告 | 2026-06-08<br>
  &#x26A0; 信号仅供参考。买入价不可预判，请根据盘中实际价格决策。<br>
  V3: 标志K线&#x2192;回调破MA18&#x2192;站上MA18&#x2192;MA18向上&#x2192;回踩MA18附近买入 | 双向摩擦0.64%
</div>
</div>
</body>
</html>'''

out = os.path.join(PROJ, "alerts", "daily_operation.html")
with open(out, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"OK {out} ({os.path.getsize(out)/1024:.0f}KB)")
