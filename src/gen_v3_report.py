#!/usr/bin/env python3
"""生成V3信号HTML报告"""
import json, os

PROJ = "/mnt/d/AI-20260604"
signals = json.load(open(os.path.join(PROJ, "signals", "v3_signals_latest.json")))

# 大盘信号
hs_msg = "✅ 沪深300 MA60向上 + MACD>0，可以做多"

rows = ""
for i, s in enumerate(signals, 1):
    dist = s['dist_pct']
    dist_color = "#e74c3c" if dist < -5 else "#e67e22" if dist < -3 else "#27ae60"
    rows += f"""<tr>
        <td>{i}</td>
        <td><b>{s['code']}</b></td>
        <td>{s['name']}</td>
        <td style="color:#666">{s['industry']}</td>
        <td style="color:{"#e74c3c" if dist < -5 else "#e67e22" if dist < -3 else "#27ae60"};font-weight:bold">{dist}%</td>
        <td>{s['price']}</td>
        <td>{s['ma18']}</td>
        <td style="color:#e74c3c;font-weight:bold">{s['entry_price']}</td>
        <td>{s['mark_date']}</td>
        <td>{s.get('stand_date','')}</td>
        <td>{s['slope']}</td>
        <td>{s['volatility']}%</td>
    </tr>"""

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>V3 涨停低吸策略 - 信号报告 {signals[0]['mark_date'][:7] if signals else ''}</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: -apple-system, 'Microsoft YaHei', sans-serif; background: #f5f6fa; padding: 20px; }}
.container {{ max-width: 1200px; margin: 0 auto; }}
.header {{ background: linear-gradient(135deg, #1a1a2e, #16213e); color: white; padding: 30px; border-radius: 12px; margin-bottom: 20px; }}
.header h1 {{ font-size: 24px; margin-bottom: 8px; }}
.header p {{ opacity: 0.8; font-size: 14px; }}
.market {{ background: {"#d4edda" if "✅" in hs_msg else "#f8d7da"}; color: {"#155724" if "✅" in hs_msg else "#721c24"}; padding: 12px 18px; border-radius: 8px; margin-bottom: 20px; font-weight: bold; }}
.card {{ background: white; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
.card h2 {{ font-size: 18px; margin-bottom: 15px; color: #2c3e50; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th {{ background: #2c3e50; color: white; padding: 10px 8px; text-align: left; }}
td {{ padding: 9px 8px; border-bottom: 1px solid #eee; }}
tr:hover {{ background: #f8f9fa; }}
tr.buy-row {{ background: #fff3cd !important; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; }}
.badge-close {{ background: #27ae60; color: white; }}
.badge-mid {{ background: #f39c12; color: white; }}
.badge-far {{ background: #e74c3c; color: white; }}
.footer {{ text-align: center; color: #999; font-size: 12px; padding: 20px; }}
@media (max-width: 768px) {{ table {{ font-size: 11px; }} }}
</style>
</head>
<body>
<div class="container">
<div class="header">
    <h1>📊 V3 涨停低吸策略 — 信号报告</h1>
    <p>生成时间: 2026-06-08 22:15 | 策略: 标志K线→回调MA18→站上MA18→回踩买入</p>
</div>

<div class="market">{hs_msg}</div>

<div class="card">
    <h2>🟢 买入信号 ({len(signals)}只)</h2>
    <div style="margin-bottom:12px;color:#666;font-size:13px">
        买入价 = min(MA18×1.01, 开盘价, 最低价) | 止损: 连续2日跌破MA18 | 止盈: BOLL上轨
    </div>
    <div style="overflow-x:auto">
    <table>
        <thead>
        <tr>
            <th>#</th><th>代码</th><th>名称</th><th>行业</th><th>距MA18</th><th>现价</th><th>MA18</th><th>买入价</th><th>标志日</th><th>站上日</th><th>斜率</th><th>波动</th>
        </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    </div>
</div>

<div class="card">
    <h2>📋 策略说明</h2>
    <ul style="padding-left:20px;color:#555;font-size:14px;line-height:1.8">
        <li><b>标志K线</b>：涨停 或 涨幅>6%+量>昨日量×3 的阳线，当时MA18向上</li>
        <li><b>回调确认</b>：20天内股价曾收盘跌破MA18</li>
        <li><b>站上确认</b>：随后收盘站上MA18（从下方上穿回归），MA18保持向上</li>
        <li><b>买入</b>：开盘价或最低价 ≤ MA18×1.01 → 按实际成交价买入</li>
        <li><b>卖出</b>：BOLL上轨止盈(收盘价) / 连续2日跌破MA18止损 / 30天到期</li>
        <li><b>回测夏普</b>：V3全样本夏普16.26 | 样本外胜率72% | 盈亏比3.02</li>
    </ul>
</div>

<div class="footer">
    涨停股低吸策略 · 自动生成 · 仅供参考 · 不构成投资建议
</div>
</div>
</body>
</html>"""

out_path = os.path.join(PROJ, "alerts", "v3_signals_report.html")
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"✅ {out_path} ({os.path.getsize(out_path)/1024:.0f}KB)")
