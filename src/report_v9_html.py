#!/usr/bin/env python3
"""v9 (ABCD版) HTML信号报告"""
import json, os

PROJ_B = "/mnt/d/AI-20260604"
SIGNAL_DIR = os.path.join(PROJ_B, "signals")
ALERT_DIR = os.path.join(PROJ_B, "alerts")

with open(os.path.join(SIGNAL_DIR, 'v9_signals_summary.json')) as f:
    d = json.load(f)

hs300 = d.get('hs300_filter', {})
hs300_str = f"MA60:{'✅' if hs300.get('ma60_up')=='True' else '❌'} MACD:{'✅' if hs300.get('macd_up')=='True' else '❌'}"

buf = []
buf.append("""<!DOCTYPE html><html><head><meta charset='utf-8'>
<title>v9 涨停突破信号报告 (ABCD版)</title>
<style>
body{font-family:'Microsoft YaHei',sans-serif;background:#f5f5f5;margin:20px}
h2{color:#333}
.card{background:#fff;border-radius:8px;padding:15px;margin:15px 0;box-shadow:0 2px 4px rgba(0,0,0,.1)}
table{width:100%;border-collapse:collapse;margin:10px 0}
th{background:#1976d2;color:#fff;padding:8px;text-align:left}
td{padding:8px;border-bottom:1px solid #eee}
tr:hover{background:#f0f0f0}
.stats{display:flex;gap:15px;margin:15px 0;flex-wrap:wrap}
.stat-box{background:#fff;border-radius:8px;padding:15px;flex:1;min-width:120px;text-align:center;box-shadow:0 2px 4px rgba(0,0,0,.1)}
.stat-box .num{font-size:28px;font-weight:bold}
.stat-box .label{font-size:12px;color:#666;margin-top:5px}
.empty{text-align:center;color:#999;padding:20px!important}
.tip-box{background:#fff3e0;border-left:4px solid #ff9800;padding:12px;margin:12px 0;border-radius:4px}
.footer{text-align:center;color:#999;margin:20px;font-size:12px}
.sell{color:#d32f2f}
.buy{color:#2e7d32}
</style></head><body>
<h2>📊 涨停突破 v9 (ABCD版) 信号报告</h2>
<div style='color:#666;font-size:14px;margin-bottom:15px'>
  生成: """ + d['generated_at'] + """ | 最新交易日: """ + d['latest_trade_date'] + """ | 沪深300 """ + hs300_str + """
</div>
""")

# 统计
buf.append("""<div class="stats">
  <div class="stat-box"><div class="num">""" + str(len(d['signals'])) + """</div><div class="label">🟢 买入信号</div></div>
  <div class="stat-box"><div class="num">""" + str(len(d['exit_signals'])) + """</div><div class="label">🔴 卖出提示</div></div>
  <div class="stat-box"><div class="num">""" + str(d['holding_count']) + """</div><div class="label">📦 持仓记录</div></div>
</div>""")

# 卖出
buf.append("""<div class="card"><div class="card-title" style="font-size:16px;font-weight:bold;margin-bottom:10px"><span style="color:#d32f2f;">●</span> 卖出提示</div>""")
if d['exit_signals']:
    buf.append("<table><tr><th>股票</th><th>名称</th><th>信号</th><th>收益</th></tr>")
    for e in d['exit_signals']:
        buf.append(f"<tr><td>{e['code']}</td><td>{e['name']}</td><td class='sell'>{e['signal']}</td><td>{e['ret']:+.2f}%</td></tr>")
    buf.append("</table>")
else:
    buf.append("<table><tr><td class='empty'>✅ 暂无卖出提示</td></tr></table>")
buf.append("</div>")

# 买入信号
buf.append("""<div class="card"><div class="card-title" style="font-size:16px;font-weight:bold;margin-bottom:10px"><span style="color:#2e7d32;">●</span> 买入信号（次日开盘关注）</div>""")
if d['signals']:
    buf.append("<table><tr><th>股票</th><th>名称</th><th>行业</th><th>信号日</th><th>18日线</th><th>收盘价</th><th>止损</th></tr>")
    for s in d['signals']:
        buf.append(f"<tr><td>{s['ts_code']}</td><td>{s['name']}</td><td>{s.get('industry','')}</td><td>{s['signal_date']}</td><td>{s['ma_value']}</td><td>{s['close_price']}</td><td>{s['stop_price_ref']}</td></tr>")
    buf.append("</table>")
else:
    buf.append("<table><tr><td class='empty'>📭 暂无买入信号</td></tr></table>")
buf.append("</div>")

# 操作提示
buf.append("""<div class="card">
<div class="tip-box">
  💡 <b>买入规则：</b>涨停→5日>10日线→放量站上18日线+MACD>0<br>
  🛑 <b>止损：</b>连续2日收盘跌破18日线（均线动态上移）<br>
  📈 <b>止盈：</b>BOLL(20,2)上轨 | 最长持有30日<br>
  📋 <b>仓位：</b>每只~9%，每日总仓位≤60%<br>
  🔬 <b>回测：</b>月频夏普2.07，最大回撤-6.99%<br>
  📝 买入后追加到 <code>signals/v9_positions.csv</code>，卖出改 status=已平仓
</div>
</div>
<div class="footer">涨停突破 v9 ABCD版 · 自动生成</div>
</body></html>""")

with open(os.path.join(ALERT_DIR, 'v9_signals_report.html'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(buf))
print(f"✅ HTML报告: alerts/v9_signals_report.html ({len('\n'.join(buf))}B)")
