#!/usr/bin/env python3
"""
14:50 下载实时分钟数据，更新v9 ABCD选股信号
流程：
  1. 从日K数据筛选候选池（满足大部分条件的股票）
  2. 对候选池下载rt_min 1MIN数据
  3. 合成当日K线，验证今日条件
  4. 输出信号报告
"""
import pandas as pd, numpy as np, os, json, time, sys

PROJ_B = "/mnt/d/AI-20260604"
DATA_DAILY_DIR = os.path.join(PROJ_B, "data", "raw", "daily")
STOCK_LIST_FILE = os.path.join(PROJ_B, "data", "raw", "stock_list.parquet")
INDEX_PATH = os.path.join(PROJ_B, "data", "raw", "index_000300.parquet")
OUTPUT_DIR = os.path.join(PROJ_B, "alerts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LIMIT_UP_PCT = 1.095
MAX_DAYS_SINCE_LIMIT = 25
MIN_DAYS_SINCE_LIMIT = 1
MA_PERIOD = 18

def get_sector_cross_limit(industry):
    if not isinstance(industry, str) or industry == '': return 5.0
    ultra = ['银行','保险','石油石化']
    low = ['公用事业','交通运输','建筑','汽车','房地产','有色金属','煤炭','商贸零售','家用电器','食品饮料']
    high = ['电子','计算机','通信','传媒','国防军工','综合']
    if industry in ultra: return 10.0
    if industry in low: return 8.0
    if industry in high: return 4.0
    return 5.0

def load_market_filter():
    df = pd.read_parquet(INDEX_PATH).sort_values('trade_date').reset_index(drop=True)
    c = df['close'].values.astype(np.float64); n = len(c)
    ma60 = np.full(n, np.nan)
    if n >= 60:
        s = np.cumsum(c); ma60[59] = s[59]/60
        for i in range(60, n): ma60[i] = (s[i]-s[i-60])/60
    mu = np.full(n, False); mu[1:] = ma60[1:] > ma60[:-1]
    r = {}
    for i in range(n):
        dt = pd.Timestamp(df.iloc[i]['trade_date']).strftime('%Y-%m-%d')
        r[dt] = bool(mu[i])
    return r

def prepare_candidates():
    """从日K数据中筛选今日候选池（排除不满足条件的）"""
    latest_date = "2026-06-04"
    
    print("加载股票列表...")
    sl = pd.read_parquet(STOCK_LIST_FILE)
    codes = sorted(sl['ts_code'].unique())
    names = dict(zip(sl['ts_code'], sl.get('name', ['']*len(sl))))
    industries = dict(zip(sl['ts_code'], sl.get('industry', ['']*len(sl))))
    
    print(f"全市场 {len(codes)} 只，扫描候选池...")
    
    candidates = []
    for idx, code in enumerate(codes):
        if (idx+1) % 1000 == 0:
            print(f"  {idx+1}/{len(codes)} ... {len(candidates)}候选")
        fp = os.path.join(DATA_DAILY_DIR, f"{code}.parquet")
        if not os.path.exists(fp): continue
        try: df = pd.read_parquet(fp)
        except: continue
        if len(df) < 180: continue
        df = df.sort_values('trade_date').reset_index(drop=True)
        
        last_ds = pd.Timestamp(df.iloc[-1]['trade_date']).strftime('%Y-%m-%d')
        if last_ds != latest_date: continue
        
        c = df['close'].values.astype(np.float64)
        h = df['high'].values.astype(np.float64)
        v = df['vol'].values.astype(np.float64)
        n = len(df)
        if n < MA_PERIOD: continue
        
        ma = np.full(n, np.nan)
        s = np.cumsum(c); ma[MA_PERIOD-1] = s[MA_PERIOD-1]/MA_PERIOD
        for i in range(MA_PERIOD, n): ma[i] = (s[i]-s[i-MA_PERIOD])/MA_PERIOD
        mu = np.full(n, False); mu[1:] = ma[1:] > ma[:-1]
        lu = np.full(n, False); lu[1:] = (c[1:]/c[:-1] > LIMIT_UP_PCT) & (c[1:] == h[1:])
        
        last_lu = -1
        for i in range(n-1, -1, -1):
            if lu[i]: last_lu = i; break
        if last_lu < 0: continue
        ds = (n-1) - last_lu
        if ds < MIN_DAYS_SINCE_LIMIT or ds > MAX_DAYS_SINCE_LIMIT: continue
        
        lb = last_lu
        ma_all_up = True
        for j in range(lb+1, n):
            if not mu[j]: ma_all_up = False; break
        
        # 涨停后涨幅
        rise = (np.max(c[lb+1:n]) / c[lb] - 1) * 100
        if rise > 15.0: continue
        
        # 日K基础条件（A部分+放量）
        ma5 = np.full(n, np.nan)
        if n >= 5:
            s5 = np.cumsum(c); ma5[4] = s5[4]/5
            for i in range(5, n): ma5[i] = (s5[i]-s5[i-5])/5
        ma10 = np.full(n, np.nan)
        if n >= 10:
            s10 = np.cumsum(c); ma10[9] = s10[9]/10
            for i in range(10, n): ma10[i] = (s10[i]-s10[i-10])/10
        
        on_ma5 = np.isfinite(ma5[n-1]) and c[n-1] > ma5[n-1]
        ma5_gt_ma10 = np.isfinite(ma5[n-1]) and np.isfinite(ma10[n-1]) and ma5[n-1] > ma10[n-1]
        on_ma18 = np.isfinite(ma[n-1]) and c[n-1] > ma[n-1]
        cross_pct = (c[n-1] / ma[n-1] - 1) * 100 if np.isfinite(ma[n-1]) else 0
        industry = industries.get(code, '')
        max_cross = get_sector_cross_limit(industry)
        
        vol_sum = 0; vol_count = 0
        for jj in range(max(0, n-6), n-1):
            if v[jj] > 0: vol_sum += v[jj]; vol_count += 1
        vol_ma5_l = vol_sum / vol_count if vol_count >= 3 else 0
        has_vol = vol_ma5_l > 0 and v[n-1] >= vol_ma5_l * 1.2
        
        # 评估质量
        quality = 0
        if ma_all_up: quality += 1
        if on_ma5: quality += 1
        if ma5_gt_ma10: quality += 1
        if on_ma18 and cross_pct <= max_cross: quality += 1
        if has_vol: quality += 1
        
        candidates.append({
            'code': code,
            'name': names.get(code, ''),
            'industry': industry,
            'close_prev': round(c[n-1], 2),
            'ma18': round(ma[n-1], 2) if np.isfinite(ma[n-1]) else 0,
            'ma5': round(ma5[n-1], 2) if np.isfinite(ma5[n-1]) else 0,
            'ma10': round(ma10[n-1], 2) if np.isfinite(ma10[n-1]) else 0,
            'cross_pct': round(cross_pct, 2),
            'volume_ratio': round(v[n-1] / vol_ma5_l, 2) if vol_ma5_l > 0 else 0,
            'signal_quality': quality,
            'ma_all_up': ma_all_up,
            'days_since_limit': ds,
            'limit_date': str(df.iloc[lb]['trade_date'])[:10],
        })
    
    print(f"候选池: {len(candidates)} 只")
    return candidates

def fetch_minute_data(codes_batch):
    """下载分钟数据"""
    import tushare as ts
    pro = ts.pro_api()
    
    today_str = "20260605"
    all_data = {}
    
    # 分批下载，每次最多20只
    batch_size = 20
    for i in range(0, len(codes_batch), batch_size):
        batch = codes_batch[i:i+batch_size]
        codes_str = ','.join(batch)
        
        for retry in range(3):
            try:
                df = pro.rt_min(ts_code=codes_str, freq='1MIN')
                if df is not None and len(df) > 0:
                    for code in batch:
                        code_df = df[df['code'] == code]
                        if len(code_df) > 0:
                            all_data[code] = code_df.sort_values('time')
                    break
            except Exception as e:
                if retry < 2:
                    time.sleep(2)
                else:
                    print(f"  ⚠️ {batch[0]}等下载失败: {e}")
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"    下载 {min(i+batch_size, len(codes_batch))}/{len(codes_batch)}")
    
    return all_data

def main():
    print("=" * 60)
    print("📊 14:50 实时分钟信号更新")
    print(f"   {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    # 1. 沪深300过滤
    print("\n[1/4] 沪深300过滤...")
    market_filter = load_market_filter()
    hs300_up = market_filter.get('2026-06-05', market_filter.get('2026-06-04', True))
    print(f"   {'✅ 多头' if hs300_up else '❌ 空头'}")
    
    # 2. 候选池
    print("\n[2/4] 从日K筛选候选池...")
    candidates = prepare_candidates()
    
    # 3. 下载分钟数据
    print(f"\n[3/4] 下载候选池分钟数据 ({len(candidates)}只)...")
    candidate_codes = [c['code'] for c in candidates]
    minute_data = fetch_minute_data(candidate_codes)
    print(f"   成功下载: {len(minute_data)} 只")
    
    # 4. 合成当日K线并出信号
    print("\n[4/4] 合成信号...")
    today_signals = []
    for c in candidates:
        code = c['code']
        if code not in minute_data or len(minute_data[code]) < 10:
            continue
        
        md = minute_data[code]
        # 合成当日K线
        today_open = float(md.iloc[0]['open'])
        today_close = float(md.iloc[-1]['close'])
        today_high = float(md['high'].max())
        today_low = float(md['low'].min())
        today_vol = float(md['vol'].sum())
        
        # 检查今日条件：(日K条件基础上) 收盘站上MA18、站上MA5
        ma18 = c['ma18']
        ma5 = c['ma5']
        ma10 = c['ma10']
        
        on_ma18_today = today_close > ma18
        on_ma5_today = today_close > ma5
        cross_pct_today = (today_close / ma18 - 1) * 100
        
        industry = c['industry']
        max_cross = get_sector_cross_limit(industry)
        
        # 评分
        quality = c['signal_quality']
        extra_today = 0
        if on_ma18_today and cross_pct_today <= max_cross: extra_today += 1
        if on_ma5_today: extra_today += 1
        
        total_quality = quality + extra_today
        
        c.update({
            'today_open': round(today_open, 2),
            'today_close': round(today_close, 2),
            'today_high': round(today_high, 2),
            'today_low': round(today_low, 2),
            'today_vol': int(today_vol),
            'today_cross_pct': round(cross_pct_today, 2),
            'today_on_ma18': on_ma18_today,
            'today_on_ma5': on_ma5_today,
            'total_quality': total_quality,
            'max_buy_price': round(ma18 * 1.01, 2),
        })
        
        today_signals.append(c)
    
    # 排序
    today_signals.sort(key=lambda x: (x['total_quality'], x.get('today_vol', 0)), reverse=True)
    
    print(f"   今日信号: {len(today_signals)} 只")
    print(f"   高质量(≥5): {sum(1 for s in today_signals if s['total_quality'] >= 5)} 只")
    
    # TOP5
    top5 = [s for s in today_signals if s['total_quality'] >= 5][:5]
    print(f"\n🏆 TOP5 推荐:")
    for i, s in enumerate(top5, 1):
        print(f"  {i}. {s['code']} {s['name']} | 今日{s['today_close']} MA18={s['ma18']} | 明日开盘≤{s['max_buy_price']}买入 | 质量{'★'*s['total_quality']}")
    
    # 保存JSON
    json_path = os.path.join(OUTPUT_DIR, "v9_signals_today.json")
    json.dump(today_signals, open(json_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2, default=str)
    print(f"✅ JSON: {json_path}")
    
    # 生成HTML
    from src.generate_v9_signals import generate_html
    try:
        generate_html(today_signals, hs300_up, "2026-06-05（实时分钟数据）")
    except:
        print("  ⚠️ HTML生成略过")
    
    print(f"\n✅ 完成！")

if __name__ == "__main__":
    main()
