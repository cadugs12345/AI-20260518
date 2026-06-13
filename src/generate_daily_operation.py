#!/usr/bin/env python3
"""
涨停低吸策略 — 每日操作报告生成（全新开始，2026-06-11起）
只保留今日可买入信号，历史清空
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta, date

PROJ = "/mnt/d/AI-20260604"
SIGNAL_FILE = os.path.join(PROJ, "signals", "zt_pullback_v2_latest.csv")
DATA_DIR = os.path.join(PROJ, "data", "raw", "daily")
STOCK_LIST_FILE = os.path.join(PROJ, "data", "raw", "stock_list.parquet")
OUT_HTML = os.path.join(PROJ, "alerts", "daily_operation.html")

today = date.today()
today_str = today.strftime("%Y-%m-%d")
next_trade_day = today_str

# ====== 加载数据 ======
sl = pd.read_parquet(STOCK_LIST_FILE)
name_map = dict(zip(sl['ts_code'], sl.get('name', [''] * len(sl))))
ind_map = dict(zip(sl['ts_code'], sl.get('industry', [''] * len(sl))))

signals = pd.read_csv(SIGNAL_FILE)
signals['signal_date'] = pd.to_datetime(signals['signal_date'])
latest_sig_date = signals['signal_date'].max()

# 只取最近3个交易日的信号
latest_dates = sorted(signals['signal_date'].unique(), reverse=True)[:3]
recent_sigs = signals[signals['signal_date'].isin(latest_dates)]

print(f"最新信号日: {latest_sig_date.date()}")
print(f"最近信号数: {len(recent_sigs)}")

# ====== 获取最新行情 ======
def get_latest_data(ts_code):
    """返回 (close, high, last_date) 或 (None, None, None)"""
    fp = os.path.join(DATA_DIR, f"{ts_code}.parquet")
    if not os.path.exists(fp):
        return None, None, None
    try:
        df = pd.read_parquet(fp).sort_values('trade_date')
        last = df.iloc[-1]
        return float(last['close']), float(last['high']), str(last['trade_date'])[:10]
    except:
        return None, None, None

# ====== 已买入（当日最高价触及买入价） vs 观察中 ======
holdings = []   # 已买入，开始持仓跟踪
watch_signals = []  # 未到买点

for _, sig in recent_sigs.iterrows():
    code = sig['ts_code']
    entry_price = sig['entry_price']
    sig_date = sig['signal_date']

    close, high, last_date = get_latest_data(code)
    if close is None or last_date is None:
        continue

    last_dt = pd.Timestamp(last_date)
    if last_dt < sig_date:
        continue

    pct_diff = (close / entry_price - 1) * 100 if entry_price > 0 else 999

    # 只在买入价附近±5%才纳入跟踪
    if abs(pct_diff) > 5:
        continue

    item = {
        'code': code,
        'name': name_map.get(code, ''),
        'industry': ind_map.get(code, ''),
        'signal_date': str(sig_date.date()),
        'entry_price': entry_price,
        'latest_close': close,
        'latest_high': high,
        'pct_diff': round(pct_diff, 2),
    }

    # 需要读取MA18来计算条件
    fp = os.path.join(DATA_DIR, f"{code}.parquet")
    ma18_val = None
    try:
        df_full = pd.read_parquet(fp).sort_values('trade_date')
        cls_arr = df_full['close'].values.astype(np.float64)
        if len(cls_arr) >= 18:
            ma18_val = float(np.mean(cls_arr[-18:]))
    except:
        pass

    # 判断标准：
    # 1) 当天最高价 >= 买入价（盘中价格覆盖买入价）
    # 2) 收盘价站上MA18
    buy_cond = high >= entry_price
    ma_cond = (close >= ma18_val) if ma18_val is not None else False

    if buy_cond and ma_cond:
        from datetime import timedelta
        sig_dt = pd.Timestamp(sig_date)
        entry_day = (sig_dt + timedelta(days=1)).strftime('%Y-%m-%d')
        item['entry_day'] = entry_day
        item['hold_days'] = 1 if last_dt == sig_dt else (last_dt - sig_dt).days
        holdings.append(item)
    else:
        item['gap'] = round(entry_price - close, 2)
        item['gap_pct'] = round(abs(pct_diff), 2)
        item['ma18_val'] = round(ma18_val, 2) if ma18_val else 0
        watch_signals.append(item)

holdings.sort(key=lambda x: x['signal_date'], reverse=True)
watch_signals.sort(key=lambda x: x['gap_pct'])

print(f"已买入(持仓): {len(holdings)}只, 观察中: {len(watch_signals)}只")

# ====== 生成HTML ======
html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>涨停股低吸策略 — 每日操作报告 — {today_str}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Microsoft YaHei','PingFang SC',sans-serif;background:#f0f2f5;color:#333;line-height:1.6}}
.container{{max-width:1200px;margin:0 auto;padding:20px}}
.header{{background:linear-gradient(135deg,#1a56db,#0d3b9e);color:#fff;padding:24px 32px;border-radius:12px;margin-bottom:20px}}
.header h1{{font-size:26px;margin-bottom:4px}}
.header .meta{{font-size:14px;opacity:0.85}}
.header .scorecard{{display:flex;gap:24px;margin-top:16px;flex-wrap:wrap}}
.header .card{{background:rgba(255,255,255,0.15);border-radius:10px;padding:14px 20px;text-align:center;min-width:100px}}
.header .card .num{{font-size:32px;font-weight:bold;display:block}}
.header .card .lbl{{font-size:12px;opacity:0.8}}
.market{{background:#fff;border-radius:10px;padding:16px 24px;margin-bottom:20px;display:flex;gap:24px;align-items:center;flex-wrap:wrap;box-shadow:0 1px 3px rgba(0,0,0,0.06)}}
.market .item{{font-size:15px}}
.market .item span{{font-weight:bold}}
.good{{color:#dc2626}} .bad{{color:#16a34a}} .warn{{color:#e67e22}}
.section{{background:#fff;border-radius:10px;padding:24px;margin-bottom:20px;box-shadow:0 1px 3px rgba(0,0,0,0.06)}}
.section h2{{font-size:20px;margin-bottom:16px;padding-bottom:10px;border-bottom:2px solid #e5e7eb;display:flex;align-items:center;gap:8px}}
.section h2 .icon{{font-size:24px}}
table{{width:100%;border-collapse:collapse;font-size:14px}}
th{{background:#f8fafc;text-align:left;padding:10px 12px;font-weight:600;color:#64748b;border-bottom:2px solid #e2e8f0;white-space:nowrap}}
td{{padding:10px 12px;border-bottom:1px solid #f1f5f9}}
tr:hover td{{background:#f8fafc}}
.tag{{display:inline-block;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600}}
.tag-buy{{background:#dcfce7;color:#16a34a}}
.tag-sell{{background:#fef2f2;color:#dc2626}}
.tag-hold{{background:#eff6ff;color:#2563eb}}
.tag-done{{background:#f1f5f9;color:#94a3b8}}
.tag-warn{{background:#fef9c3;color:#a16207}}
.checklist{{list-style:none}}
.checklist li{{padding:10px 16px;margin:6px 0;border-radius:8px;font-size:15px;display:flex;align-items:center;gap:10px}}
.checklist .buy-li{{background:#f0fdf4;border-left:4px solid #16a34a}}
.checklist .sell-li{{background:#fef2f2;border-left:4px solid #dc2626}}
.checklist .hold-li{{background:#eff6ff;border-left:4px solid #2563eb}}
.checklist .track-li{{background:#fefce8;border-left:4px solid #eab308}}
.checklist .step{{min-width:80px;font-weight:bold}}
.price{{font-family:'Consolas','Monaco',monospace;font-weight:600}}
.note{{color:#94a3b8;font-size:13px;margin-top:4px}}
.footer{{text-align:center;color:#94a3b8;font-size:12px;padding:20px}}
.num-up{{color:#dc2626;font-weight:600}} .num-down{{color:#16a34a;font-weight:600}}
</style>
</head>
<body>
<div class="container">
<div class="header">
  <h1>📊 涨停股低吸策略</h1>
  <div class="meta">报告生成: {datetime.now().strftime('%Y-%m-%d %H:%M')} | 最新信号: {latest_sig_date.date()} | 下一个交易日: {next_trade_day}</div>
  <div class="scorecard">
    <div class="card"><span class="num">{len(holdings)}</span><span class="lbl">📌 持仓中</span></div>
    <div class="card"><span class="num">{len(watch_signals)}</span><span class="lbl">📡 观察中</span></div>
    <div class="card"><span class="num">0</span><span class="lbl">✅ 历史成交</span></div>
  </div>
</div>

<div class="section">
<h2><span class="icon">📌</span> 持仓跟踪（{len(holdings)}只）</h2>
<p class="note">当日最高价触及买入价即标记为已买入，持仓按天跟踪。止盈=BOLL上轨 | 止损=连续2日收破MA18 | 30天到期</p>
<table>
<tr><th>代码</th><th>名称</th><th>行业</th><th>信号日</th><th>入场日</th><th>买入价</th><th>最新价</th><th>当日最高</th><th>收益</th><th>天数</th><th>状态</th></tr>
"""

for h in holdings:
    pct = h['pct_diff']
    pct_str = f"+{pct:.2f}%" if pct >= 0 else f"{pct:.2f}%"
    pct_cls = "num-up" if pct >= 0 else "num-down"
    tag = "tag-hold" if pct >= -3 else "tag-warn"
    tag_text = "📌 持仓中" if pct >= -3 else "⚠️ 需关注"
    html += f"""<tr>
<td>{h['code']}</td><td>{h['name']}</td><td>{h['industry'][:8]}</td>
<td>{h['signal_date']}</td>
<td>{h['entry_day']}</td>
<td class="price">{h['entry_price']:.2f}</td>
<td class="price">{h['latest_close']:.2f}</td>
<td class="price">{h['latest_high']:.2f}</td>
<td class="{pct_cls}">{pct_str}</td>
<td>{h['hold_days']}天</td>
<td><span class="tag {tag}">{tag_text}</span></td></tr>
"""

html += """</table></div>

<div class="section">
<h2><span class="icon">📡</span> 观察中信号</h2>
<p class="note">未同时满足两个条件：1)盘中最高价触及买入价 2)收盘站上MA18。买入价=MA18×1.01</p>
<table>
<tr><th>代码</th><th>名称</th><th>行业</th><th>信号日</th><th>买入信号价</th><th>最新价</th><th>MA18</th><th>C≥MA18</th><th>当日最高</th><th>H≥买入价</th><th>状态</th></tr>
"""

for w in watch_signals:
    buy_ok = "✅" if w['latest_high'] >= w['entry_price'] else "❌"
    ma_ok = "✅" if w['latest_close'] >= w['ma18_val'] else "❌"
    html += f"""<tr>
<td>{w['code']}</td><td>{w['name']}</td><td>{w['industry'][:8]}</td>
<td>{w['signal_date']}</td>
<td class="price">{w['entry_price']:.2f}</td>
<td class="price">{w['latest_close']:.2f}</td>
<td class="price">{w['ma18_val']:.2f}</td>
<td>{ma_ok}</td>
<td class="price">{w['latest_high']:.2f}</td>
<td>{buy_ok}</td>
<td><span class="tag tag-warn">{'盘中达买点·收未站上' if buy_ok == '✅' else '⏳ 等待中'}</span></td></tr>
"""

html += """</table></div>

<div class="section">
<h2><span class="icon">📡</span> 观察中信号</h2>
<p class="note">未同时满足两个条件：1)盘中最高价触及买入价 2)收盘站上MA18。买入价=MA18×1.01</p>
<table>
<tr><th>代码</th><th>名称</th><th>行业</th><th>信号日</th><th>买入信号价</th><th>最新价</th><th>MA18</th><th>C≥MA18</th><th>当日最高</th><th>H≥买入价</th><th>状态</th></tr>
"""

for w in watch_signals:
    buy_ok = "✅" if w['latest_high'] >= w['entry_price'] else "❌"
    ma_ok = "✅" if w['latest_close'] >= w['ma18_val'] else "❌"
    html += f"""<tr>
<td>{w['code']}</td><td>{w['name']}</td><td>{w['industry'][:8]}</td>
<td>{w['signal_date']}</td>
<td class="price">{w['entry_price']:.2f}</td>
<td class="price">{w['latest_close']:.2f}</td>
<td class="price">{w['ma18_val']:.2f}</td>
<td>{ma_ok}</td>
<td class="price">{w['latest_high']:.2f}</td>
<td>{buy_ok}</td>
<td><span class="tag tag-warn">{'盘中达买点·收未站上' if buy_ok == '✅' else '⏳ 等待中'}</span></td></tr>
"""

html += """</table></div>

<div class="section">
<h2><span class="icon">✅</span> 历史成交</h2>
<table>
<tr><th>代码</th><th>名称</th><th>入场日</th><th>买入价</th><th>卖出日</th><th>卖出价</th><th>收益</th><th>天数</th><th>原因</th></tr>
<tr><td colspan="9" style="text-align:center;color:#94a3b8;padding:30px">暂无历史成交</td></tr>
</table></div>

<div class="section">
<h2><span class="icon">📋</span> 操作清单（{next_trade_day}）</h2>
<ul class="checklist">
"""

if holdings:
    html += f"""<li class="hold-li"><span class="step">📌 持仓</span> 共{len(holdings)}只，关注止盈止损信号</li>
"""
    for h in holdings:
        pct = h['pct_diff']
        icon = "🟢" if pct >= 0 else "🔴"
        html += f"""<li class="hold-li"><span class="step">  {icon}</span> <b>{h['code']} {h['name']}</b> 买入价<span class="price">{h['entry_price']:.2f}</span> 最新价{h['latest_close']:.2f} 收益{pct:+.2f}%</li>
"""

if watch_signals:
    html += f"""<li class="track-li"><span class="step">📡 观察</span> 共{len(watch_signals)}只等待买点</li>
"""
    for w in watch_signals[:5]:
        html += f"""<li class="track-li"><span class="step">  📡</span> <b>{w['code']} {w['name']}</b> 买入价{w['entry_price']:.2f} 最新价{w['latest_close']:.2f} 差距-{w['gap_pct']:.1f}%</li>
"""
    if len(watch_signals) > 5:
        html += f"""<li class="track-li"><span class="step">  📡</span> 还有{len(watch_signals)-5}只观察中...</li>
"""

if not holdings and not watch_signals:
    html += """<li class="track-li"><span class="step">📡</span> 暂无信号，等待新的买入条件满足</li>"""

html += f"""</ul>
</div>

<div class="footer">
  涨停低吸策略 | 每日操作报告 | 生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}<br>
  买入价=MA18×1.01 | 当日最高价触及买入价即标记持仓 | 止盈=BOLL上轨(收盘价) | 止损=连续2日跌破MA18 | 30天到期 | 含摩擦成本0.64% | v3版(MA18斜率>-2%)
</div>
</div>
</body>
</html>"""

with open(OUT_HTML, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\n✅ 报告已生成: {OUT_HTML}")
print(f"   持仓: {len(holdings)}只 | 观察中: {len(watch_signals)}只")
