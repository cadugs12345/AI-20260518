#!/usr/bin/env python3
"""
涨停股低吸策略 — 每日选股+持仓管理
===========================================
选股逻辑:
  1. 最近25天内有涨停，涨停当天18日均线向上
  2. 涨停后1~25天，18日均线始终向上
  3. 价量共振: 收盘站上5日线 + 5日线>10日线
  4. 收盘站上18日线，上穿幅度行业自适应
  5. 排除涨停后涨超15%的
  6. 放量: 量>前5日均量×1.2
  7. 大盘过滤: 沪深300 MA60向上 + MACD柱>0
  8. 市场热度: 当天信号≥2

买入: 次日开盘不高于18日线×1.01才买，否则等盘中回落，最长3天
卖出: BOLL(20,2)上轨止盈 | 连续2日跌破18日线止损 | 30天到期
仓位: 每只~9%，每日总仓位≤60%
"""
import pandas as pd
import numpy as np
import os, sys, json, warnings
from collections import Counter
warnings.filterwarnings('ignore')

PROJ_B = "/mnt/d/AI-20260604"
DATA_DIR = os.path.join(PROJ_B, "data", "raw", "daily")
SL_FILE = os.path.join(PROJ_B, "data", "raw", "stock_list.parquet")
INDEX_PATH = os.path.join(PROJ_B, "data", "raw", "index_000300.parquet")
SIGNAL_DIR = os.path.join(PROJ_B, "signals")
ALERT_DIR = os.path.join(PROJ_B, "alerts")
os.makedirs(SIGNAL_DIR, exist_ok=True)
os.makedirs(ALERT_DIR, exist_ok=True)

# ====== 参数 ======
LIMIT_UP_PCT = 1.095
MAX_DAYS_SINCE_LIMIT = 25
MIN_DAYS_SINCE_LIMIT = 1
MIN_TRADE_DAYS = 180
COST_PER_TRADE = 0.0032
MAX_HOLD = 30
MIN_MARKET_SIGNALS = 2
MAX_DAILY_TOTAL_POSITION = 0.60
MA_PERIOD = 18
MA_BREAK_CANDLES = 2
BOLL_PERIOD = 20
BOLL_STD = 2.0

# B. 行业分域阈值（上穿幅度上限）
# 超低波动: 银行/保险/石油石化 → 10%
# 低波动: 公用/交运/建筑/汽车/地产/有色/煤炭/商贸/家电/食品饮料 → 8%
# 中波动: 机械/化工/建材/医药/农业/纺织/电设/轻工 → 5%
# 高波动: 电子/计算机/通信/传媒/军工/非银/综合 → 4%
CROSS_LIMITS = {
    '银行':10.0,'保险':10.0,'石油':10.0,'石化':10.0,
    '公用':8.0,'交通':8.0,'运输':8.0,'建筑':8.0,'汽车':8.0,
    '地产':8.0,'有色':8.0,'煤炭':8.0,'商贸':8.0,'家电':8.0,'食品':8.0,'饮料':8.0,
    '电子':4.0,'计算机':4.0,'通信':4.0,'传媒':4.0,'军工':4.0,'国防':4.0,'非银':4.0,'综合':4.0,
}
DEFAULT_CROSS_LIMIT = 5.0
def get_cross_limit(ind):
    for k, v in CROSS_LIMITS.items():
        if k in str(ind): return v
    return DEFAULT_CROSS_LIMIT

POSITION_FILE = os.path.join(SIGNAL_DIR, "v9_positions.csv")


def load_market_filter():
    """返回 {date: (ma60_up, macd_up)}"""
    print("  [沪深300] ", end="", flush=True)
    df = pd.read_parquet(INDEX_PATH).sort_values('trade_date').reset_index(drop=True)
    c = df['close'].values.astype(np.float64); n = len(c)
    # MA60
    ma60 = np.full(n, np.nan)
    if n >= 60:
        s = np.cumsum(c); ma60[59] = s[59]/60
        for i in range(60, n): ma60[i] = (s[i]-s[i-60])/60
    mu60 = np.full(n, False, dtype=bool); mu60[1:] = ma60[1:] > ma60[:-1]
    # MACD柱>0
    macd_up = np.full(n, False, dtype=bool)
    if n >= 26:
        ema12 = np.full(n, np.nan); ema26 = np.full(n, np.nan)
        ema12[0] = c[0]; ema26[0] = c[0]
        k12 = 2/(12+1); k26 = 2/(26+1)
        for i in range(1, n):
            ema12[i] = c[i]*k12 + ema12[i-1]*(1-k12)
            ema26[i] = c[i]*k26 + ema26[i-1]*(1-k26)
        macd_up[25:] = (ema12[25:] - ema26[25:]) > 0
    r = {}
    for i in range(n):
        dt = pd.Timestamp(df.iloc[i]['trade_date']).strftime('%Y-%m-%d')
        r[dt] = (bool(mu60[i]), bool(macd_up[i]))
    return r


def detect_signals(df, industry_map=None, code=None):
    """ABCD版选股"""
    c = df['close'].values.astype(np.float64)
    h = df['high'].values.astype(np.float64)
    v = df['vol'].values.astype(np.float64)
    n = len(df)
    
    # MA18
    ma = np.full(n, np.nan)
    if n >= MA_PERIOD:
        s = np.cumsum(c); ma[MA_PERIOD-1] = s[MA_PERIOD-1]/MA_PERIOD
        for i in range(MA_PERIOD, n): ma[i] = (s[i]-s[i-MA_PERIOD])/MA_PERIOD
    mu = np.full(n, False, dtype=bool); mu[1:] = ma[1:] > ma[:-1]
    # MA5, MA10
    ma5 = np.full(n, np.nan)
    if n >= 5:
        s5 = np.cumsum(c); ma5[4] = s5[4]/5
        for i in range(5, n): ma5[i] = (s5[i]-s5[i-5])/5
    ma10 = np.full(n, np.nan)
    if n >= 10:
        s10 = np.cumsum(c); ma10[9] = s10[9]/10
        for i in range(10, n): ma10[i] = (s10[i]-s10[i-10])/10
    
    # 涨停
    lu = np.full(n, False, dtype=bool)
    lu[1:] = (c[1:]/c[:-1] > LIMIT_UP_PCT) & (c[1:] == h[1:])
    
    lli = np.full(n, -1, dtype=np.int32); ls = -1
    for i in range(n):
        if lu[i]: ls = i
        lli[i] = i - ls if ls >= 0 else -1
    
    sigs = []
    for i in range(n):
        ds = lli[i]
        if ds <= 0 or ds < MIN_DAYS_SINCE_LIMIT or ds > MAX_DAYS_SINCE_LIMIT: continue
        lb = i - ds
        if np.isnan(ma[i]) or np.isnan(ma[lb]): continue
        if not mu[i]: continue
        ma_all_up = True
        for j in range(lb+1, i+1):
            if not mu[j]: ma_all_up = False; break
        if not ma_all_up: continue
        
        # A. 价量共振: 站上5日线 + 5>10日线
        if np.isnan(ma5[i]) or c[i] <= ma5[i]: continue
        if np.isfinite(ma10[i]) and ma5[i] <= ma10[i]: continue
        
        # B. 行业自适应上穿幅度
        max_cross = get_cross_limit(industry_map.get(code, '')) if industry_map and code else DEFAULT_CROSS_LIMIT
        if np.isnan(ma[i]) or c[i] <= ma[i]: continue
        cross_pct = (c[i] / ma[i] - 1) * 100
        if cross_pct > max_cross: continue
        
        # 排除涨停后涨超15%
        since_high = np.max(c[lb+1:i+1])
        rise_pct = (since_high / c[lb] - 1) * 100
        if rise_pct > 15.0: continue
        
        # 放量确认
        vol_sum = 0; vol_count = 0
        for jj in range(max(0,i-5), i):
            if v[jj] > 0: vol_sum += v[jj]; vol_count += 1
        vol_ma5 = vol_sum / vol_count if vol_count >= 3 else 0
        if vol_ma5 > 0 and v[i] <= vol_ma5 * 1.2: continue
        
        sigs.append({
            'idx': i, 'lb': lb,
            'ma_value': ma[i],
            'cross_pct': cross_pct,
            'limit_open': float(df.iloc[lb]['open']),
            'limit_close': float(df.iloc[lb]['close']),
            'close_price': float(c[i]),
            'ma5': float(ma5[i]) if np.isfinite(ma5[i]) else 0,
        })
    return sigs


def check_exit(code, df, entry_date_str, entry_price, stop_price, entry_idx):
    """检查持仓"""
    for la in range(1, MAX_HOLD+1):
        if entry_idx + la >= len(df): break
        row = df.iloc[entry_idx + la]
        ddh, ddl, ddc = row['high'], row['low'], row['close']
        ci = entry_idx + la
        
        # BOLL上轨止盈
        tp_boll = np.inf
        if ci >= BOLL_PERIOD-1:
            w = df.iloc[ci-BOLL_PERIOD+1:ci+1]['close'].values.astype(np.float64)
            if len(w) == BOLL_PERIOD:
                tp_boll = np.mean(w) + BOLL_STD * np.std(w, ddof=1)
        if ddh >= tp_boll - 1e-8:
            return f'止盈@{ddh:.2f}', round((ddh/entry_price-1)*100 - COST_PER_TRADE*2, 2)
        
        # 18日线动态止损
        current_ma = np.nan
        if ci >= MA_PERIOD-1:
            w = df.iloc[ci-MA_PERIOD+1:ci+1]['close'].values.astype(np.float64)
            if len(w) == MA_PERIOD: current_ma = np.mean(w)
        if not np.isnan(current_ma):
            stop_price = max(stop_price, current_ma)
        
        if ci >= 2:
            ddc1 = float(df.iloc[ci-1]['close'])
            ddc2 = float(ddc)
            if ddc2 < stop_price and ddc1 < stop_price:
                return f'止损@{ddc2:.2f}', round((ddc2/entry_price-1)*100 - COST_PER_TRADE*2, 2)
    
    return None, None


def main():
    print(f"{'='*50}")
    print(f"📊 涨停股低吸策略")
    print(f"{'='*50}")
    print(f"   系统时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    
    # 沪深300大盘过滤
    mf = load_market_filter()
    latest_dates = sorted(mf.keys(), reverse=True)
    latest_trade_date = latest_dates[0] if latest_dates else 'N/A'
    hs300_d = mf.get(latest_trade_date, (False, False))
    print(f"   {latest_trade_date} 沪深300 MA60:{'向上✅' if hs300_d[0] else '向下❌'} MACD:{'柱>0✅' if hs300_d[1] else '柱≤0❌'}")
    
    # 加载持仓
    positions = []
    if os.path.exists(POSITION_FILE):
        dfp = pd.read_csv(POSITION_FILE)
        positions = dfp[dfp['status'] == '持有中'].to_dict('records') if 'status' in dfp.columns else []
    print(f"   持仓: {len(positions)}只")
    
    # 检查卖出
    exit_signals = []
    t0 = pd.Timestamp.now()
    for pos in positions:
        code = pos.get('ts_code', '')
        fp = os.path.join(DATA_DIR, f"{code}.parquet")
        if not os.path.exists(fp): continue
        df = pd.read_parquet(fp).sort_values('trade_date').reset_index(drop=True)
        ed = pd.Timestamp(pos['entry_date']).strftime('%Y-%m-%d')
        match = (df['trade_date'].astype(str) == ed)
        if not match.any(): continue
        eidx = match.idxmax()
        sig, ret = check_exit(code, df, ed, float(pos['entry_price']), float(pos['stop_price']), eidx)
        if sig:
            exit_signals.append({'code': code, 'name': pos.get('name', ''), 'signal': sig, 'ret': ret})
    print(f"   卖出信号: {len(exit_signals)}")
    
    # 加载股票
    print("\n[扫描信号]")
    sl = pd.read_parquet(SL_FILE)
    codes = sorted(sl['ts_code'].unique())
    nm = dict(zip(sl['ts_code'], sl.get('name', ['']*len(sl))))
    industry_map = dict(zip(sl['ts_code'], sl.get('industry', ['']*len(sl)))) if 'industry' in sl else {}
    
    all_sigs = []
    for idx, code in enumerate(codes):
        if (idx+1) % 1000 == 0: print(f"   {idx+1}/{len(codes)}", end=" ", flush=True)
        fp = os.path.join(DATA_DIR, f"{code}.parquet")
        if not os.path.exists(fp): continue
        try: df = pd.read_parquet(fp).sort_values('trade_date').reset_index(drop=True)
        except: continue
        if len(df) < MIN_TRADE_DAYS: continue
        
        for sig in detect_signals(df, industry_map, code):
            gi = sig['idx']
            all_sigs.append({
                'code': code, 'name': nm.get(code, ''),
                'date': df.iloc[gi]['trade_date'],
                'idx': gi,
                'ma_value': sig['ma_value'],
                'cross_pct': sig['cross_pct'],
                'limit_open': sig['limit_open'],
                'close_price': sig['close_price'],
                'ma5': sig['ma5'],
            })
    
    # 信号计数
    dt_count = Counter()
    for s in all_sigs:
        d = pd.Timestamp(s['date']).strftime('%Y-%m-%d')
        dt_count[d] += 1
    print(f"\n   总信号: {len(all_sigs)}")
    
    # 过滤: 大盘 + 市场热度 + 排除持仓
    filtered = []
    for s in all_sigs:
        d = pd.Timestamp(s['date']).strftime('%Y-%m-%d')
        fd = mf.get(d, (False, False))
        if not fd[0] or not fd[1]: continue  # MACD+MA60
        if dt_count.get(d, 0) < MIN_MARKET_SIGNALS: continue
        if any(p.get('ts_code') == s['code'] for p in positions): continue
        filtered.append(s)
    print(f"   过滤后: {len(filtered)}")
    
    # 分组输出
    groups = {}
    for s in filtered:
        d = pd.Timestamp(s['date']).strftime('%Y-%m-%d')
        if d not in groups: groups[d] = []
        groups[d].append(s)
    
    # JSON输出
    summary = []
    for dt in sorted(groups.keys(), reverse=True)[:10]:
        for s in groups[dt]:
            sig_date = pd.Timestamp(s['date'])
            ind = industry_map.get(s['code'], '')
            max_cross = get_cross_limit(ind)
            summary.append({
                'ts_code': s['code'], 'name': s['name'],
                'signal_date': sig_date.strftime('%Y-%m-%d'),
                'ma_value': round(s['ma_value'], 2),
                'close_price': round(s['close_price'], 2),
                'ma5': round(s['ma5'], 2),
                'stop_price_ref': round(s['ma_value'], 2),
                'industry': ind,
                'max_cross': max_cross,
                'is_holding': False,
            })
    
    out = {
        'generated_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
        'version': '涨停股低吸策略',
        'latest_trade_date': latest_trade_date,
        'hs300_filter': {'ma60_up':str(hs300_d[0]), 'macd_up':str(hs300_d[1])},
        'total_signals': len(filtered),
        'holding_count': len(positions),
        'exit_signals': [{'code': e['code'], 'name': e['name'], 'signal': e['signal'], 'ret': e['ret']} for e in exit_signals],
        'signals': summary,
    }
    with open(os.path.join(SIGNAL_DIR, 'v9_signals_summary.json'), 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    
    # CSV输出
    if filtered:
        import csv
        with open(os.path.join(SIGNAL_DIR, 'v9_screener_latest.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['股票','名称','信号日','18日线','5日线','收盘价','止损参考','行业','上穿上限'])
            for s in filtered:
                d = pd.Timestamp(s['date']).strftime('%Y-%m-%d')
                ind = industry_map.get(s['code'], '')
                w.writerow([s['code'], s['name'], d,
                           round(s['ma_value'],2), round(s['ma5'],2),
                           round(s['close_price'],2),
                           round(s['ma_value'],2), ind,
                           get_cross_limit(ind)])
    
    # 终端输出
    print(f"\n{'='*50}")
    print(f"📋 信号摘要")
    print(f"{'='*50}")
    print(f"   最新交易日: {latest_trade_date}")
    print(f"   总信号: {len(filtered)} | 市场过滤: {len(dt_count)}个信号日")
    
    for dt in sorted(groups.keys(), reverse=True)[:5]:
        g = groups[dt]
        n_sig = dt_count.get(dt, 0)
        print(f"\n   {dt} — {len(g)}只 (全市场{n_sig}信号)")
        for s in g[:5]:
            ind = industry_map.get(s['code'], '')
            max_c = get_cross_limit(ind)
            print(f"      {s['code']} {s['name']}  {ind}  MA18={s['ma_value']:.1f} 收盘{s['close_price']:.1f} 上限{max_c}%")
    
    if exit_signals:
        print(f"\n🔴 卖出提示:")
        for e in exit_signals:
            print(f"   {e['code']} {e['name']}: {e['signal']} ({e['ret']:+.2f}%)")
    
    print(f"\n   持仓文件: {POSITION_FILE} ({len(positions)}只持有中)")
    print(f"   JSON: signals/v9_signals_summary.json")
    print(f"   CSV: signals/v9_screener_latest.csv")
    print(f"\n⏱ {int((pd.Timestamp.now()-t0).total_seconds())}秒")


if __name__ == '__main__':
    main()
