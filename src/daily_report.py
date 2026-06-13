#!/usr/bin/env python3
"""
涨停回踩不破 v2 — 每日操作报告
==========================
基于 zt_pullback_v2.py 的信号生成操作报告
持仓跟踪：仅跟踪最新信号日当天选出的股票
"""
import pandas as pd
import numpy as np
import os, sys, json, warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

PROJ = "/mnt/d/AI-20260604"
DATA_DIR = os.path.join(PROJ, "data", "raw", "daily")
INDEX_PATH = os.path.join(PROJ, "data", "raw", "index_000300.parquet")
SIGNAL_DIR = os.path.join(PROJ, "signals")
ALERT_DIR = os.path.join(PROJ, "alerts")
OUTPUT_HTML = os.path.join(ALERT_DIR, "daily_operation.html")
os.makedirs(ALERT_DIR, exist_ok=True)

MA_PERIOD = 18
BOLL_PERIOD = 20
BOLL_STD = 2.0
MAX_HOLD = 30
COST = 0.0032


def load_market_status():
    """沪深300大盘状态"""
    df = pd.read_parquet(INDEX_PATH).sort_values('trade_date').reset_index(drop=True)
    c = df['close'].values.astype(np.float64); n = len(c)
    ma60 = np.full(n, np.nan)
    if n >= 60:
        s = np.cumsum(c); ma60[59] = s[59]/60
        for i in range(60, n): ma60[i] = (s[i]-s[i-60])/60
    mu60 = np.full(n, False); mu60[1:] = ma60[1:] > ma60[:-1]
    macd_up = np.full(n, False)
    if n >= 26:
        ema12=np.full(n,np.nan); ema26=np.full(n,np.nan)
        ema12[0]=c[0]; ema26[0]=c[0]
        k12=2/(12+1); k26=2/(26+1)
        for i in range(1,n):
            ema12[i]=c[i]*k12+ema12[i-1]*(1-k12)
            ema26[i]=c[i]*k26+ema26[i-1]*(1-k26)
        macd_up[25:] = (ema12[25:] - ema26[25:]) > 0
    r = {}
    for i in range(n):
        key = str(df.iloc[i]['trade_date'])[:10]
        r[key] = (bool(mu60[i]), bool(macd_up[i]))
    last = str(df.iloc[-1]['trade_date'])[:10]
    return r, last


def load_signals():
    """加载 zt_pullback_v2 的最新信号"""
    fp = os.path.join(SIGNAL_DIR, "zt_pullback_v2_latest.csv")
    if not os.path.exists(fp):
        return pd.DataFrame()
    df = pd.read_csv(fp)
    df['signal_date'] = pd.to_datetime(df['signal_date'])
    return df


def _find_latest_signal_date(all_signals):
    """找到最新信号日"""
    if len(all_signals) == 0:
        return None
    return all_signals['signal_date'].max()


def _get_holdings_from_signals(all_signals, latest_signal_date):
    """
    zt_pullback v2 策略: 最新信号日选出的所有股票即为持仓
    (策略逻辑: 买入价=MA18×1.01，当天信号当天可买)
    返回: [{code, name, industry, entry_price, entry_date, ...}]
    """
    if latest_signal_date is None:
        return []
    
    latest = all_signals[all_signals['signal_date'] == latest_signal_date]
    holdings = []
    for _, row in latest.iterrows():
        code = row['ts_code']
        # 检查卖出状态
        action, ret, exit_price, days, reason = _check_exit(code, latest_signal_date, row['entry_price'])
        
        # 获取最新价
        rma, rclose, rlow, rhigh, ropen, last_date = _calc_price(code)
        
        if action:
            # 已完成交易，放入历史
            holdings.append({
                'type': 'history',
                'code': code,
                'name': row.get('name',''),
                'industry': row.get('industry',''),
                'entry_date': str(latest_signal_date)[:10],
                'entry_price': round(row['entry_price'], 2),
                'exit_date': str(_find_exit_date(code, latest_signal_date, row['entry_price']))[:10],
                'exit_price': exit_price if exit_price else rclose,
                'return_pct': ret if ret else 0,
                'days': days if days else 0,
                'exit_reason': reason if reason else '',
            })
        else:
            # 持有中
            cp = rclose if rclose and not np.isnan(rclose) else row['close']
            cur_ret = round((cp/row['entry_price']-1)*100 - COST*2, 2)
            cur_ma = rma if rma and not np.isnan(rma) else row['ma18']
            holdings.append({
                'type': 'holding',
                'code': code,
                'name': row.get('name',''),
                'industry': row.get('industry',''),
                'signal_date': str(latest_signal_date)[:10],
                'entry_date': str(latest_signal_date)[:10],
                'entry_price': round(row['entry_price'], 2),
                'buy_type': '开盘买入',
                'current_price': cp,
                'return_pct': cur_ret,
                'days': days if days else 1,
                'current_ma': cur_ma,
            })
    return holdings


def _check_exit(code, entry_date, entry_price):
    """检查卖出条件，返回 (action, ret, exit_price, days, reason)"""
    fp = os.path.join(DATA_DIR, f"{code}.parquet")
    if not os.path.exists(fp):
        return None, None, None, None, None
    df = pd.read_parquet(fp).sort_values('trade_date').reset_index(drop=True)
    
    ed_str = str(entry_date)[:10]
    match = df['trade_date'].astype(str).str[:10] == ed_str
    if not match.any():
        return None, None, None, None, None
    ei = match.idxmax()
    
    for la in range(1, MAX_HOLD+1):
        if ei+la >= len(df):
            break
        row = df.iloc[ei+la]
        ci = ei+la
        
        # BOLL止盈
        tp_boll = np.inf
        if ci >= BOLL_PERIOD-1:
            w = df.iloc[ci-BOLL_PERIOD+1:ci+1]['close'].values.astype(np.float64)
            if len(w) == BOLL_PERIOD:
                tp_boll = np.mean(w) + BOLL_STD*np.std(w,ddof=1)
        if float(row['high']) >= tp_boll - 1e-8:
            exit_price = float(row['close'])
            return ('止盈', round((exit_price/entry_price-1)*100-COST*2,2), exit_price, la, f'止盈(T+{la})')
        
        # 连续2日跌破MA18止损
        cur_ma = np.nan
        if ci >= MA_PERIOD-1:
            w = df.iloc[ci-MA_PERIOD+1:ci+1]['close'].values.astype(np.float64)
            if len(w) == MA_PERIOD:
                cur_ma = np.mean(w)
        if ci >= 2:
            c1 = float(df.iloc[ci-1]['close'])
            c2 = float(row['close'])
            if not np.isnan(cur_ma) and c2 < cur_ma and c1 < cur_ma:
                return ('止损', round((c2/entry_price-1)*100-COST*2,2), c2, la, f'止损(T+{la})')
    
    # 仍持有中
    last_close = float(df.iloc[-1]['close'])
    ret = round((last_close/entry_price-1)*100-COST*2, 2)
    days = min(len(df) - ei, MAX_HOLD)
    return None, ret, last_close, days, None


def _calc_price(code):
    """返回 (ma18, close, low, high, open, trade_date)"""
    fp = os.path.join(DATA_DIR, f"{code}.parquet")
    if not os.path.exists(fp):
        return (np.nan, np.nan, np.nan, np.nan, np.nan, None)
    df = pd.read_parquet(fp).sort_values('trade_date').reset_index(drop=True)
    if len(df) == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, None)
    last = df.iloc[-1]
    ci = len(df) - 1
    cur_ma = np.nan
    if ci >= MA_PERIOD-1:
        w = df.iloc[ci-MA_PERIOD+1:ci+1]['close'].values.astype(np.float64)
        if len(w) == MA_PERIOD:
            cur_ma = np.mean(w)
    return (cur_ma, float(last['close']), float(last['low']),
            float(last['high']), float(last['open']), last['trade_date'])


def _find_exit_date(code, entry_date, entry_price):
    """找实际卖出日期"""
    fp = os.path.join(DATA_DIR, f"{code}.parquet")
    if not os.path.exists(fp):
        return ''
    df = pd.read_parquet(fp).sort_values('trade_date').reset_index(drop=True)
    ed_str = str(entry_date)[:10]
    match = df['trade_date'].astype(str).str[:10] == ed_str
    if not match.any():
        return ''
    ei = match.idxmax()
    for la in range(1, MAX_HOLD+1):
        if ei+la >= len(df):
            break
        ci = ei+la
        tp_boll = np.inf
        if ci >= BOLL_PERIOD-1:
            w = df.iloc[ci-BOLL_PERIOD+1:ci+1]['close'].values.astype(np.float64)
            if len(w) == BOLL_PERIOD:
                tp_boll = np.mean(w) + BOLL_STD*np.std(w,ddof=1)
        row = df.iloc[ci]
        if float(row['high']) >= tp_boll - 1e-8:
            return str(row['trade_date'])[:10]
        cur_ma = np.nan
        if ci >= MA_PERIOD-1:
            w = df.iloc[ci-MA_PERIOD+1:ci+1]['close'].values.astype(np.float64)
            if len(w) == MA_PERIOD:
                cur_ma = np.mean(w)
        if ci >= 2:
            c1 = float(df.iloc[ci-1]['close'])
            c2 = float(row['close'])
            if not np.isnan(cur_ma) and c2 < cur_ma and c1 < cur_ma:
                return str(row['trade_date'])[:10]
    return ''


def _next_trading_day(date_str):
    d = pd.Timestamp(date_str)
    d += timedelta(days=1)
    while d.dayofweek >= 5:
        d += timedelta(days=1)
    return d.strftime('%Y-%m-%d')


def main():
    print("="*50)
    print("📊 涨停回踩不破 v2 — 每日操作报告")
    print("="*50)
    
    # 大盘
    mf, latest_date = load_market_status()
    mn, hm = mf.get(latest_date, (False, False))
    print(f"   大盘: {latest_date} | MA60:{'↑' if mn else '↓'} MACD:{'>0' if hm else '≤0'}")
    
    # 加载信号
    all_signals = load_signals()
    if len(all_signals) == 0:
        print("   ❌ 无信号数据")
        return
    
    latest_signal_date = _find_latest_signal_date(all_signals)
    print(f"   最新信号日: {str(latest_signal_date)[:10]}")
    print(f"   历史信号总数: {len(all_signals)}")
    
    # 构建持仓/历史
    items = _get_holdings_from_signals(all_signals, latest_signal_date)
    
    holdings = [i for i in items if i['type'] == 'holding']
    history = [i for i in items if i['type'] == 'history']
    
    # 按日期分组统计
    print(f"\n📋 今日信号 ({str(latest_signal_date)[:10]}):")
    today_signals = all_signals[all_signals['signal_date'] == latest_signal_date]
    print(f"   选中 {len(today_signals)} 只")
    for _, row in today_signals.iterrows():
        print(f"   {row['ts_code']} {row['name']:<6}  "
              f"收盘{row['close']:.2f} MA18{row['ma18']:.2f} "
              f"买入价{row['entry_price']:.2f}")
    
    # ====== 生成HTML ======
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    next_day = _next_trading_day(latest_date)
    
    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>涨停回踩不破 v2 — 每日操作报告 — {latest_date}</title>
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
  <h1>📊 涨停回踩不破 v2</h1>
  <div class="meta">报告生成: {now} | 最新交易日: {latest_date} | 下一个交易日: {next_day}</div>
  <div class="scorecard">
    <div class="card"><span class="num">{len(today_signals)}</span><span class="lbl">🔥 今日信号</span></div>
    <div class="card"><span class="num">{len(holdings)}</span><span class="lbl">📌 持仓跟踪</span></div>
    <div class="card"><span class="num">{len(history)}</span><span class="lbl">✅ 历史成交</span></div>
  </div>
</div>

<div class="market">
  <div class="item">📈 沪深300 MA60: <span class="{'good' if mn else 'warn'}">{'向上 ✅' if mn else '向下 ⚠️'}</span></div>
  <div class="item">📊 MACD柱: <span class="{'good' if hm else 'warn'}">{'>0 ✅' if hm else '≤0 ⚠️'}</span></div>
  <div class="item">💡 大盘状态: <span class="{'good' if mn and hm else 'warn'}">{'🟢 可以做多' if mn and hm else '🟡 谨慎'}</span></div>
</div>
'''
    
    # ── ① 今日最新信号（第一栏）──
    html += f'''<div class="section">
<h2><span class="icon">🔥</span> 今日信号（{str(latest_signal_date)[:10]}）</h2>
<p class="note">策略: 涨停回踩+MA18向上+上穿MA18+MA18斜率>-2%+波动<100%+15日内涨停。买入价=MA18×1.01</p>
<table>
<tr><th>代码</th><th>名称</th><th>行业</th><th>信号日</th><th>买入信号价</th><th>MA18</th><th>收盘价</th><th>涨停日</th><th>跌穿日</th></tr>'''
    for _, row in today_signals.iterrows():
        ld = str(row['limit_bar_date'])[:10]
        bd = str(row['broke_date'])[:10]
        html += f'''<tr>
<td>{row['ts_code']}</td><td>{row['name']}</td><td>{row['industry']}</td>
<td>{str(row['signal_date'])[:10]}</td>
<td class="price">{row['entry_price']:.2f}</td>
<td class="price">{row['ma18']:.2f}</td>
<td class="price">{row['close']:.2f}</td>
<td>{ld}</td><td>{bd}</td></tr>'''
    html += '</table></div>'
    
    # ── ② 持仓跟踪 ──
    if holdings:
        html += f'''<div class="section">
<h2><span class="icon">📌</span> 持仓跟踪</h2>
<table>
<tr><th>代码</th><th>名称</th><th>行业</th><th>入场日</th><th>买入价</th><th>最新价</th><th>收益</th><th>天数</th><th>MA18</th><th>操作</th></tr>'''
        for s in holdings:
            code = s.get('code','')
            name = s.get('name','')
            ind = s.get('industry','')
            entry_date = s.get('entry_date','')
            ep = s.get('entry_price',0)
            cp = s.get('current_price',0)
            ret = s.get('return_pct',0)
            days = s.get('days',0)
            cur_ma = s.get('current_ma', np.nan)
            cur_ma_str = f'{cur_ma:.2f}' if not np.isnan(cur_ma) else '-'
            ret_cls = 'num-up' if ret >= 0 else 'num-down'
            html += f'''<tr>
<td>{code}</td><td>{name}</td><td>{ind}</td>
<td>{entry_date}</td>
<td class="price">{ep:.2f}</td>
<td class="price">{cp:.2f}</td>
<td class="{ret_cls}">{ret:+.2f}%</td>
<td>{days}天</td>
<td class="price">{cur_ma_str}</td>
<td><span class="tag tag-hold">持有中</span></td></tr>'''
        html += '</table></div>'
    else:
        html += f'''<div class="section">
<h2><span class="icon">📌</span> 持仓跟踪</h2>
<p style="color:#94a3b8">今日信号刚出，明日开盘买入</p>
</div>'''
    
    # ── ③ 历史成交 ──
    if history:
        html += f'''<div class="section">
<h2><span class="icon">✅</span> 历史成交</h2>
<table>
<tr><th>代码</th><th>名称</th><th>入场日</th><th>买入价</th><th>卖出日</th><th>卖出价</th><th>收益</th><th>天数</th><th>原因</th></tr>'''
        for s in history:
            code = s.get('code','')
            name = s.get('name','')
            entry_date = s.get('entry_date','')
            ep = s.get('entry_price',0)
            exit_date = s.get('exit_date','')
            exit_price = s.get('exit_price',0)
            ret = s.get('return_pct',0)
            days = s.get('days',0)
            reason = s.get('exit_reason','')
            ret_cls = 'num-up' if ret >= 0 else 'num-down'
            html += f'''<tr>
<td>{code}</td><td>{name}</td><td>{entry_date}</td>
<td class="price">{ep:.2f}</td><td>{exit_date}</td><td class="price">{exit_price:.2f}</td>
<td class="{ret_cls}">{ret:+.2f}%</td><td>{days}天</td><td>{reason}</td></tr>'''
        html += '</table></div>'
    
    # ── ④ 操作清单 ──
    html += f'''<div class="section">
<h2><span class="icon">📋</span> 操作清单（{next_day}）</h2>
<ul class="checklist">
<li class="buy-li"><span class="step">🔥 今日信号</span> {len(today_signals)}只，明日开盘买入</li>'''
    for _, row in today_signals.iterrows():
        name = row.get('name','')
        html += f'<li class="buy-li"><span class="step">  🔥</span> <b>{row["ts_code"]} {name}</b> 买入价{row["entry_price"]:.2f} MA18{row["ma18"]:.2f}</li>'
    
    html += f'''</ul>
</div>

<div class="footer">
  涨停回踩不破 v2 | 每日操作报告 | 生成于 {now}<br>
  买入价=MA18×1.01 | 止盈=BOLL上轨 | 止损=连续2日跌破MA18 | 30天到期<br>
  策略: 涨停+MA18向上+5天内跌破/接近+今天上穿+距MA18<5%+斜率>-2%+波动<100%+15日内涨停
</div>
</div>
</body>
</html>'''
    
    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n✅ 报告已生成: {OUTPUT_HTML}")
    print(f"   大小: {os.path.getsize(OUTPUT_HTML)/1024:.1f} KB")
    print(f"\n📋 汇总:")
    print(f"   🔥 今日信号: {len(today_signals)}只")
    print(f"   📌 持仓: {len(holdings)}只")
    print(f"   ✅ 历史: {len(history)}只")


if __name__ == '__main__':
    main()
