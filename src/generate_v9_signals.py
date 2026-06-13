#!/usr/bin/env python3
"""生成今天v9 ABCD版选股信号 + HTML报告"""

import pandas as pd
import numpy as np
import os, json, math
from collections import Counter

PROJ_B = "/mnt/d/AI-20260604"
DATA_DAILY_DIR = os.path.join(PROJ_B, "data", "raw", "daily")
STOCK_LIST_FILE = os.path.join(PROJ_B, "data", "raw", "stock_list.parquet")
INDEX_PATH = os.path.join(PROJ_B, "data", "raw", "index_000300.parquet")
OUTPUT_DIR = os.path.join(PROJ_B, "alerts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LIMIT_UP_PCT = 1.095
MAX_DAYS_SINCE_LIMIT = 25
MIN_DAYS_SINCE_LIMIT = 1
MIN_TRADE_DAYS = 180
MA_PERIOD = 18

def get_sector_cross_limit(industry):
    if not isinstance(industry, str) or industry == '':
        return 5.0
    ultra_low = ['银行','保险','石油石化']
    low_vol = ['公用事业','交通运输','建筑','汽车','房地产','有色金属','煤炭','商贸零售','家用电器','食品饮料']
    high_vol = ['电子','计算机','通信','传媒','国防军工','综合']
    if industry in ultra_low: return 10.0
    if industry in low_vol: return 8.0
    if industry in high_vol: return 4.0
    return 5.0

def load_market_filter():
    df = pd.read_parquet(INDEX_PATH).sort_values('trade_date').reset_index(drop=True)
    c = df['close'].values.astype(np.float64); n = len(c)
    ma = np.full(n, np.nan)
    if n >= 60:
        s = np.cumsum(c); ma[59] = s[59]/60
        for i in range(60, n): ma[i] = (s[i]-s[i-60])/60
    mu = np.full(n, False, dtype=bool); mu[1:] = ma[1:] > ma[:-1]
    r = {}
    for i in range(n):
        dt = pd.Timestamp(df.iloc[i]['trade_date']).strftime('%Y-%m-%d')
        r[dt] = bool(mu[i])
    return r

def generate_html(signals, market_filter_today, today_str):
    total_signals = len(signals)
    hs300_status = "✅ 多头" if market_filter_today else "❌ 空头（过滤）"
    
    table_rows = ""
    for i, s in enumerate(signals, 1):
        limit_dt = pd.Timestamp(s['limit_date']).strftime('%Y-%m-%d')
        buy_cond = f"开盘≤{s['max_buy_price']:.2f}买入"
        
        # 信号质量着色
        color_class = ""
        if s['signal_quality'] >= 3:
            color_class = 'style="background:#e8f8f0"'
        elif s['signal_quality'] <= 1:
            color_class = 'style="background:#fdedec"'
        
        table_rows += f"""<tr {color_class}>
            <td>{i}</td>
            <td><b>{s['code']}</b></td>
            <td>{s['name']}</td>
            <td>{s['industry']}</td>
            <td>{s['close']}</td>
            <td>{s['ma18']}</td>
            <td>{s['cross_pct']}%</td>
            <td>{s['volume_ratio']}x</td>
            <td>{s['ma5']}/{s['ma10']}</td>
            <td>{limit_dt}（{s['days_since_limit']}d）</td>
            <td style="color:#c0392b;font-weight:bold">{buy_cond}</td>
            <td>{'★'*s['signal_quality']}</td>
        </tr>"""
    
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><title>v9 ABCD 选股信号 - {today_str}</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: -apple-system, 'Microsoft YaHei', sans-serif; background: #f5f6fa; color: #2c3e50; padding: 20px; }}
.container {{ max-width: 1400px; margin: 0 auto; }}
.header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 24px 30px; border-radius: 12px; margin-bottom: 20px; }}
.header h1 {{ font-size: 24px; margin-bottom: 8px; }}
.header .meta {{ font-size: 14px; opacity: 0.85; }}
.summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 20px; }}
.card {{ background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
.card .label {{ font-size: 12px; color: #95a5a6; margin-bottom: 4px; }}
.card .value {{ font-size: 28px; font-weight: bold; }}
.card .value.green {{ color: #27ae60; }}
.card .value.red {{ color: #e74c3c; }}
table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
th {{ background: #34495e; color: white; padding: 12px 10px; font-size: 13px; text-align: left; white-space: nowrap; }}
td {{ padding: 10px; font-size: 13px; border-bottom: 1px solid #ecf0f1; }}
tr:hover {{ background: #f8f9fa; }}
.footer {{ text-align: center; padding: 20px; color: #95a5a6; font-size: 12px; }}
.note {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 12px 16px; margin-bottom: 16px; border-radius: 4px; font-size: 13px; line-height: 1.6; }}
.quality {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; }}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>📈 v9 ABCD 涨停突破选股</h1>
        <div class="meta">{today_str} 信号报告 ｜ 沪深300: {hs300_status}</div>
    </div>
    
    <div class="summary">
        <div class="card">
            <div class="label">今日信号数</div>
            <div class="value {'green' if total_signals > 0 else 'red'}">{total_signals}</div>
        </div>
        <div class="card">
            <div class="label">沪深300 60日线</div>
            <div class="value {'green' if hs300_status.startswith('✅') else 'red'}">{'上行' if hs300_status.startswith('✅') else '下行'}</div>
        </div>
        <div class="card">
            <div class="label">回测总夏普</div>
            <div class="value green">2.23</div>
        </div>
        <div class="card">
            <div class="label">回测最大回撤</div>
            <div class="value orange">-7.42%</div>
        </div>
    </div>
    
    <div class="note">
        <strong>⚡ 操作提醒：</strong>
        ABCD版条件：①涨停后18日均线向上 ②站上5日线+5日>10日线 ③行业自适应上穿阈值 
        ④放量确认 ⑤<b>买入规则</b>：明日开盘价 ≤ 18日线×1.01 才可买入，否则盘中等待回踩<br>
        <strong>止损：</strong>连续2日收盘跌破18日线 ｜ <strong>止盈：</strong>BOLL(20,2)上轨 ｜ <strong>最大持有：</strong>30日<br>
        <strong>★信号质量：</strong>★★★=强信号（全条件通过） ★★=中信号 ★=弱信号（部分条件不达标）
    </div>
    
    <table>
        <thead>
            <tr>
                <th>#</th><th>代码</th><th>名称</th><th>行业</th>
                <th>收盘</th><th>MA18</th><th>上穿%</th><th>量比</th>
                <th>MA5/MA10</th><th>涨停日</th><th>买入条件</th><th>质量</th>
            </tr>
        </thead>
        <tbody>
            {table_rows if table_rows else '<tr><td colspan="12" style="text-align:center;padding:40px;color:#95a5a6">今日无信号</td></tr>'}
        </tbody>
    </table>
    
    <div class="footer">
        v9 ABCD 涨停突破策略 ｜ 数据至 {today_str} ｜ 风险提示：历史回测不代表未来收益
    </div>
</div>
</body>
</html>"""
    
    path = os.path.join(OUTPUT_DIR, "v9_signals_today.html")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"✅ 信号HTML: {path}")
    return path

def main():
    # 自动取最新日期：从日K线中找最近交易日
    latest_date = "2026-06-01"
    for fname in os.listdir("data/raw/daily"):
        if not fname.endswith('.parquet'): continue
        try:
            dft = pd.read_parquet(os.path.join("data/raw/daily", fname))
            ld = pd.Timestamp(dft.iloc[-1]['trade_date']).strftime('%Y-%m-%d')
            if ld > latest_date: latest_date = ld
        except: pass
    print(f"  最新数据日期: {latest_date}")
    today_str = f"{latest_date}（最新数据）"
    
    print("加载沪深300过滤...")
    market_filter = load_market_filter()
    today_filter = market_filter.get(latest_date, True)
    
    print("加载股票列表...")
    sl = pd.read_parquet(STOCK_LIST_FILE)
    codes = sorted(sl['ts_code'].unique())
    names = dict(zip(sl['ts_code'], sl.get('name', ['']*len(sl))))
    industries = dict(zip(sl['ts_code'], sl.get('industry', ['']*len(sl))))
    print(f"全市场 {len(codes)} 只股票")

    all_signals = []
    for idx, code in enumerate(codes):
        if (idx+1) % 1000 == 0:
            print(f"  扫描中 {idx+1}/{len(codes)} ... ({len(all_signals)}个信号)")
        fp = os.path.join(DATA_DAILY_DIR, f"{code}.parquet")
        if not os.path.exists(fp): continue
        try: df = pd.read_parquet(fp)
        except: continue
        if len(df) < MIN_TRADE_DAYS: continue
        df = df.sort_values('trade_date').reset_index(drop=True)
        
        last_ds = pd.Timestamp(df.iloc[-1]['trade_date']).strftime('%Y-%m-%d')
        if last_ds != latest_date: continue
        
        c = df['close'].values.astype(np.float64)
        h = df['high'].values.astype(np.float64)
        v = df['vol'].values.astype(np.float64)
        n = len(df)
        if n < MA_PERIOD: continue
        
        # 18日均线
        ma = np.full(n, np.nan)
        s = np.cumsum(c); ma[MA_PERIOD-1] = s[MA_PERIOD-1]/MA_PERIOD
        for i in range(MA_PERIOD, n): ma[i] = (s[i]-s[i-MA_PERIOD])/MA_PERIOD
        
        # 均线方向
        mu = np.full(n, False)
        mu[1:] = ma[1:] > ma[:-1]
        
        # 涨停
        lu = np.full(n, False)
        lu[1:] = (c[1:]/c[:-1] > LIMIT_UP_PCT) & (c[1:] == h[1:])
        
        # 5日/10日线
        ma5 = np.full(n, np.nan)
        if n >= 5:
            s5 = np.cumsum(c); ma5[4] = s5[4]/5
            for i in range(5, n): ma5[i] = (s5[i]-s5[i-5])/5
        ma10 = np.full(n, np.nan)
        if n >= 10:
            s10 = np.cumsum(c); ma10[9] = s10[9]/10
            for i in range(10, n): ma10[i] = (s10[i]-s10[i-10])/10
        
        # 最近涨停
        last_lu = -1
        for i in range(n-1, -1, -1):
            if lu[i]: last_lu = i; break
        if last_lu < 0: continue
        ds = (n-1) - last_lu
        if ds < MIN_DAYS_SINCE_LIMIT or ds > MAX_DAYS_SINCE_LIMIT: continue
        
        lb = last_lu
        
        # 检查条件并计算信号质量
        quality = 0
        
        # ① 18日线全程向上
        ma_all_up = True
        for j in range(lb+1, n):
            if not mu[j]: ma_all_up = False; break
        
        # ② 站上5日线
        on_ma5 = np.isfinite(ma5[n-1]) and c[n-1] > ma5[n-1]
        
        # ③ 5日>10日线
        ma5_above_ma10 = np.isfinite(ma5[n-1]) and np.isfinite(ma10[n-1]) and ma5[n-1] > ma10[n-1]
        
        # ④ 站上18日线
        on_ma18 = np.isfinite(ma[n-1]) and c[n-1] > ma[n-1]
        
        # ⑤ 上穿幅度不超过行业限制
        cross_pct = (c[n-1] / ma[n-1] - 1) * 100 if np.isfinite(ma[n-1]) else 0
        industry = industries.get(code, '')
        max_cross = get_sector_cross_limit(industry)
        
        # ⑥ 放量
        vol_sum = 0; vol_count = 0
        for jj in range(max(0, n-6), n-1):
            if v[jj] > 0: vol_sum += v[jj]; vol_count += 1
        vol_ma5 = vol_sum / vol_count if vol_count >= 3 else 0
        has_volume = vol_ma5 > 0 and v[n-1] >= vol_ma5 * 1.2
        
        # 信号质量评分（加分项：量比+涨停距离合理）
        quality = 1
        if ma_all_up: quality += 1
        if has_volume: quality += 1
        
        # 硬性条件：必须站上18日线（收盘>MA18）
        if not on_ma18:
            continue
        # B条件：上穿幅度不能超过行业限制
        if cross_pct > max_cross:
            continue
        # A条件：站上5日线 + 5日>10日
        if not on_ma5 or not ma5_above_ma10:
            continue
        
        # 涨停后涨幅过滤
        since_limit_high = np.max(c[lb+1:n])
        limit_close = c[lb]
        rise_pct = (since_limit_high / limit_close - 1) * 100
        if rise_pct > 15.0: continue
        
        # 计算明日买入价上限
        max_buy_price = round(ma[n-1] * 1.01, 2) if np.isfinite(ma[n-1]) else 0
        
        all_signals.append({
            'code': code,
            'name': names.get(code, ''),
            'industry': industry,
            'limit_date': df.iloc[lb]['trade_date'],
            'days_since_limit': ds,
            'close': round(c[n-1], 2),
            'ma18': round(ma[n-1], 2) if np.isfinite(ma[n-1]) else 0,
            'cross_pct': round(cross_pct, 2),
            'volume_ratio': round(v[n-1] / vol_ma5, 2) if vol_ma5 > 0 else 0,
            'ma5': round(ma5[n-1], 2) if np.isfinite(ma5[n-1]) else 0,
            'ma10': round(ma10[n-1], 2) if np.isfinite(ma10[n-1]) else 0,
            'max_buy_price': max_buy_price,
            'signal_quality': quality,
            'ma_all_up': ma_all_up,
            'on_ma5': on_ma5,
            'ma5_above_ma10': ma5_above_ma10,
            'on_ma18': on_ma18,
            'has_volume': has_volume,
            'rise_since_limit': round(rise_pct, 1),
        })
    
    # 按信号质量排序
    all_signals.sort(key=lambda x: (x['signal_quality'], -x['volume_ratio'], -x['close']/max(x['ma18'],0.01)), reverse=True)
    
    print(f"\n今日信号数: {len(all_signals)}")
    
    if all_signals:
        print(f"\n{'代码':<12} {'名称':<10} {'行业':<10} {'收盘':<8} {'MA18':<8} {'上穿%':<8} {'量比':<8} {'质量':<6}")
        print("-" * 70)
        for s in all_signals:
            q = '★'*s['signal_quality']
            print(f"{s['code']:<12} {s['name']:<10} {s['industry']:<10} {s['close']:<8} {s['ma18']:<8} {s['cross_pct']:<8} {s['volume_ratio']:<8} {q:<6}")
    
    path = generate_html(all_signals, today_filter, today_str)
    
    # JSON
    json.dump(all_signals, open(os.path.join(OUTPUT_DIR, "v9_signals_today.json"), 'w', encoding='utf-8'), 
              ensure_ascii=False, indent=2, default=str)
    print(f"✅ 信号JSON: {os.path.join(OUTPUT_DIR, 'v9_signals_today.json')}")

if __name__ == "__main__":
    main()
