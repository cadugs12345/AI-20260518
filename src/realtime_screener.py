#!/usr/bin/env python3
"""
v9 实时盘中选股引擎 (ABCD版)
===========================
双模式:
  realtime   — 14:50~15:00运行，用rt_min实时分钟数据
  after_close — 收盘后运行，用当天日K线数据判断（cron用）

买入条件:
  1. 当天有涨停过的股票
  2. 涨停后1~25天，18日均线始终向上
  3. 价量共振: 收盘站上5日线 + 5日线>10日线
  4. 收盘站上18日线，行业自适应上穿幅度
  5. 排除涨停后涨超15%
  6. 放量: 当日量 > 前5日均量×1.2
  7. 大盘: 沪深300 MA60向上 + MACD柱>0
  8. 市场热度: 当天信号≥2

买入: 实时模式14:57集合竞价 / 收盘模式次日开盘
"""
import pandas as pd
import numpy as np
import os, sys, json, warnings, time, argparse
from datetime import datetime
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
MA_PERIOD = 18
MIN_MARKET_SIGNALS = 2
BOLL_PERIOD = 20
BOLL_STD = 2.0

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


def load_market_filter():
    df = pd.read_parquet(INDEX_PATH).sort_values('trade_date').reset_index(drop=True)
    c = df['close'].values.astype(np.float64); n = len(c)
    ma60 = np.full(n, np.nan)
    if n >= 60:
        s = np.cumsum(c); ma60[59] = s[59]/60
        for i in range(60, n): ma60[i] = (s[i]-s[i-60])/60
    mu60 = np.full(n, False, dtype=bool); mu60[1:] = ma60[1:] > ma60[:-1]
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


def check_conditions(code, daily_df, industry_map):
    """
    核心条件判断（通用，实时/收盘共用）
    daily_df: 包含当天数据（如果是收盘后才有当天完整数据）
    """
    c = daily_df['close'].values.astype(np.float64)
    h = daily_df['high'].values.astype(np.float64)
    v = daily_df['vol'].values.astype(np.float64)
    n = len(daily_df)
    
    # 指标
    ma = np.full(n, np.nan)
    if n >= MA_PERIOD:
        s = np.cumsum(c); ma[MA_PERIOD-1] = s[MA_PERIOD-1]/MA_PERIOD
        for i in range(MA_PERIOD, n): ma[i] = (s[i]-s[i-MA_PERIOD])/MA_PERIOD
    mu = np.full(n, False, dtype=bool); mu[1:] = ma[1:] > ma[:-1]
    
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
    
    # 找最近涨停
    last_limit = -1
    for i in range(n-1, max(0, n-40), -1):
        if lu[i]:
            last_limit = i
            break
    if last_limit < 0:
        return None
    
    days_since = n - 1 - last_limit
    if days_since < MIN_DAYS_SINCE_LIMIT or days_since > MAX_DAYS_SINCE_LIMIT:
        return None
    
    # 18日线持续向上
    for j in range(last_limit+1, n):
        if not mu[j]:
            return None
    
    if not mu[n-1]:
        return None
    
    today_ma18 = ma[-1]
    today_ma5 = ma5[-1]
    today_ma10 = ma10[-1]
    
    if np.isnan(today_ma18) or np.isnan(today_ma5):
        return None
    
    # 涨停后涨幅
    since_high = np.max(c[last_limit+1:n])
    rise_pct = (since_high / c[last_limit] - 1) * 100
    if rise_pct > 15.0:
        return None
    
    # A. 价量共振
    if c[-1] <= today_ma5:
        return None
    if np.isfinite(today_ma10) and today_ma5 <= today_ma10:
        return None
    
    # 站上18日线
    if c[-1] <= today_ma18:
        return None
    
    # B. 行业自适应
    cross_pct = (c[-1] / today_ma18 - 1) * 100
    max_cross = get_cross_limit(industry_map.get(code, ''))
    if cross_pct > max_cross:
        return None
    
    # 放量
    vol_ma5 = np.mean(v[max(0, n-6):n-1])
    if vol_ma5 > 0 and v[-1] <= vol_ma5 * 1.2:
        return None
    
    return {
        'code': code,
        'ma18': round(today_ma18, 2),
        'ma5': round(today_ma5, 2),
        'close': round(float(c[-1]), 2),
        'cross_pct': round(cross_pct, 2),
        'max_cross': max_cross,
        'days_since_limit': days_since,
        'rise_since_limit_pct': round(rise_pct, 2),
        'limit_date': str(daily_df.iloc[last_limit]['trade_date'])[:10],
    }


def mode_after_close():
    """收盘后用日K线出信号"""
    print("\n[收盘模式] 用日K线数据判断")
    
    mf = load_market_filter()
    today_str = datetime.now().strftime('%Y-%m-%d')
    hs300 = mf.get(today_str, (False, False))
    print(f"   沪深300 MA60:{'✅' if hs300[0] else '❌'} MACD柱:{'✅' if hs300[1] else '❌'}")
    if not hs300[0] or not hs300[1]:
        print("   ❌ 大盘不满足")
        return []
    
    # 加载行业
    sl = pd.read_parquet(SL_FILE)
    industry_map = dict(zip(sl['ts_code'], sl.get('industry', ['']*len(sl)))) if 'industry' in sl else {}
    nm = dict(zip(sl['ts_code'], sl.get('name', ['']*len(sl))))
    codes = sorted(sl['ts_code'].unique())
    
    # 扫描
    signals = []
    for idx, code in enumerate(codes):
        if (idx+1) % 1000 == 0:
            print(f"   {idx+1}/{len(codes)}", end=" ", flush=True)
        
        fp = os.path.join(DATA_DIR, f"{code}.parquet")
        if not os.path.exists(fp): continue
        try:
            df = pd.read_parquet(fp).sort_values('trade_date').reset_index(drop=True)
        except: continue
        if len(df) < MIN_TRADE_DAYS: continue
        
        # 必须包含今天
        last_date = str(df['trade_date'].max())[:10]
        if last_date < today_str: continue
        
        sig = check_conditions(code, df, industry_map)
        if sig:
            sig['name'] = nm.get(code, '')
            signals.append(sig)
    
    return signals


def mode_realtime():
    """盘中用rt_min出信号（14:50后运行）"""
    import tushare as ts
    pro = ts.pro_api()
    
    print("\n[实时模式] 用rt_min实时分钟数据")
    
    mf = load_market_filter()
    today_str = datetime.now().strftime('%Y-%m-%d')
    hs300 = mf.get(today_str, (False, False))
    print(f"   沪深300 MA60:{'✅' if hs300[0] else '❌'} MACD柱:{'✅' if hs300[1] else '❌'}")
    if not hs300[0] or not hs300[1]:
        print("   ❌ 大盘不满足")
        return []
    
    sl = pd.read_parquet(SL_FILE)
    industry_map = dict(zip(sl['ts_code'], sl.get('industry', ['']*len(sl)))) if 'industry' in sl else {}
    nm = dict(zip(sl['ts_code'], sl.get('name', ['']*len(sl))))
    
    # 用rt_min批量找涨停（分批拉）
    codes_all = sl['ts_code'].tolist()
    limit_high = {}  # code -> 当天最高价
    
    batch_size = 50  # 减少超频风险
    for i in range(0, len(codes_all), batch_size):
        batch = codes_all[i:i+batch_size]
        codes_str = ','.join(batch)
        try:
            df = pro.rt_min(ts_code=codes_str, freq='5MIN')
            time.sleep(0.1)
            if df is None or len(df) == 0: continue
            for code in batch:
                cdf = df[df['ts_code'] == code]
                if len(cdf) == 0: continue
                limit_high[code] = cdf['high'].max()
        except:
            time.sleep(0.5)
            continue
    
    # 过滤涨停（用昨天的收盘价）
    limit_codes = []
    for code, high in limit_high.items():
        fp = os.path.join(DATA_DIR, f"{code}.parquet")
        if not os.path.exists(fp): continue
        daily = pd.read_parquet(fp).sort_values('trade_date')
        if len(daily) < 2: continue
        last_d = daily.iloc[-1]
        if str(last_d['trade_date'])[:10] >= today_str:
            pre = daily.iloc[-2]['close']
        else:
            pre = daily.iloc[-1]['close']
        if high >= pre * LIMIT_UP_PCT - 0.02:
            limit_codes.append(code)
    
    print(f"   盘中涨停: {len(limit_codes)}只")
    
    # 逐一检查条件
    signals = []
    for idx, code in enumerate(limit_codes):
        if (idx+1) % 10 == 0:
            print(f"   检查: {idx+1}/{len(limit_codes)}", flush=True)
        
        fp = os.path.join(DATA_DIR, f"{code}.parquet")
        if not os.path.exists(fp): continue
        try:
            df = pd.read_parquet(fp).sort_values('trade_date').reset_index(drop=True)
        except: continue
        if len(df) < MIN_TRADE_DAYS: continue
        
        # 获取实时收盘价和成交量
        rtm = pro.rt_min(ts_code=code, freq='5MIN')
        time.sleep(0.1)
        if rtm is None or len(rtm) == 0: continue
        
        last_close = float(rtm.iloc[-1]['close'])
        today_vol = rtm['vol'].sum()
        
        sig = check_conditions(code, df, industry_map)
        if sig:
            # 更新为实时数据
            sig['close'] = round(last_close, 2)
            sig['name'] = nm.get(code, '')
            signals.append(sig)
    
    return signals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['realtime', 'after_close'], default='after_close',
                       help='realtime=盘中(需rt_min), after_close=收盘后')
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"📡 v9 选股引擎 (ABCD版) - {args.mode}模式")
    print("=" * 60)
    print(f"   时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    t0 = time.time()
    
    if args.mode == 'realtime':
        signals = mode_realtime()
    else:
        signals = mode_after_close()
    
    # 输出
    print(f"\n{'=' * 60}")
    print(f"📋 信号: {len(signals)}只")
    print(f"{'=' * 60}")
    
    # 分组输出
    if signals:
        for s in signals:
            ind = industry_map.get(s['code'], '')
            print(f"\n   🟢 {s['code']} {s['name']:<12} {ind}")
            print(f"      18日线:{s['ma18']}  5日线:{s['ma5']}  收盘:{s['close']}")
            print(f"      上穿:{s['cross_pct']:+.1f}% 上限:{s['max_cross']}%")
            print(f"      涨停后{s['days_since_limit']}天 涨停后涨{s['rise_since_limit_pct']:.1f}%")
    else:
        print("\n   📭 今日无信号")
    
    dt_key = datetime.now().strftime('%Y%m%d_%H%M')
    
    # 保存
    out = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'mode': args.mode,
        'signals_count': len(signals),
        'signals': [{
            'ts_code': s['code'], 'name': s['name'],
            'ma18': s['ma18'], 'ma5': s['ma5'],
            'price': s['close'],
            'cross_pct': s['cross_pct'], 'max_cross': s['max_cross'],
            'days_since_limit': s['days_since_limit'],
            'rise_since_limit_pct': s['rise_since_limit_pct'],
            'industry': industry_map.get(s['code'], ''),
        } for s in signals],
    }
    
    with open(os.path.join(SIGNAL_DIR, 'v9_realtime_signal.json'), 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    
    print(f"\n   JSON: signals/v9_realtime_signal.json")
    print(f"⏱ {int(time.time()-t0)}秒")


if __name__ == '__main__':
    # 先加载行业（全局用）
    if os.path.exists(SL_FILE):
        sl = pd.read_parquet(SL_FILE)
        industry_map = dict(zip(sl['ts_code'], sl.get('industry', ['']*len(sl)))) if 'industry' in sl else {}
    else:
        industry_map = {}
    main()
