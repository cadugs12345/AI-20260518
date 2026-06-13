#!/usr/bin/env python3
"""
涨停突破 v6 — 每日选股 + 持仓管理 + 卖出提示
==============================================
每日运行，输出:
  1. 买入信号 — 次日可关注的股票
  2. 卖出提示 — 已持仓股票是否触发止损/止盈/到期
  3. 持仓管理 — 自动更新止损追踪线

使用方法:
  python src/screener_v6.py

持仓文件:
  signals/v6_positions.csv  — 手动记录买入（首次运行自动创建模板）

输出:
  signals/v6_screener_latest.csv   — 买入信号
  signals/v6_signals_summary.json  — 当日完整报告
"""
import pandas as pd
import numpy as np
import os, sys, warnings, json, time
from datetime import datetime, timedelta
from collections import Counter
warnings.filterwarnings('ignore')

PROJ_B = "/mnt/d/AI-20260604"
DATA_DAILY_DIR = os.path.join(PROJ_B, "data", "raw", "daily")
STOCK_LIST_FILE = os.path.join(PROJ_B, "data", "raw", "stock_list.parquet")
INDEX_PATH = os.path.join(PROJ_B, "data", "raw", "index_000300.parquet")
OUTPUT_DIR = os.path.join(PROJ_B, "signals")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====== 策略参数 ======
LIMIT_UP_PCT = 1.095
MAX_DAYS_SINCE_LIMIT = 20
MIN_DAYS_SINCE_LIMIT = 3
MIN_TRADE_DAYS = 120
SHRINK_VOL_RATIO = 0.5
BREAKOUT_VOL_RATIO = 1.2
MIN_MARKET_SIGNALS = 3
BOLL_PERIOD = 20
BOLL_STD = 2.0
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0
TAKE_PROFIT_PCT = 0.20
STOP_LOSS_PCT = 0.95
MAX_HOLD = 30

HISTORY_FILE = os.path.join(OUTPUT_DIR, "v6_screener_history.csv")
POSITIONS_FILE = os.path.join(OUTPUT_DIR, "v6_positions.csv")

POSITIONS_COLUMNS = [
    'ts_code', 'name', 'buy_date', 'buy_price', 'shares',
    'stop_price', 'target_price', 'limit_low', 'entry_date_idx',
    'status', 'exit_date', 'exit_price', 'exit_reason',
]


def boll_lower_at_idx(close, period=20, n_std=2.0):
    if len(close) < period: return np.nan
    w = close[-period:]
    return np.mean(w) - n_std * np.std(w, ddof=1)


def calc_atr(high, low, close, period=14):
    n = len(high)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i-1]*(period-1)+tr[i])/period
    return atr


def load_market_filter():
    if not os.path.exists(INDEX_PATH):
        return None
    df = pd.read_parquet(INDEX_PATH).sort_values('trade_date').reset_index(drop=True)
    c = df['close'].values.astype(np.float64); n = len(c)
    ma20 = np.full(n, np.nan)
    if n >= 20:
        s = np.cumsum(c); ma20[19] = s[19]/20
        for i in range(20, n): ma20[i] = (s[i]-s[i-20])/20
    mu = np.full(n, False, dtype=bool); mu[1:] = ma20[1:] > ma20[:-1]
    return {pd.Timestamp(df.iloc[i]['trade_date']).strftime('%Y-%m-%d'): bool(mu[i]) for i in range(n)}


def get_latest_trade_date():
    daily_files = [f for f in os.listdir(DATA_DAILY_DIR) if f.endswith('.parquet')]
    if not daily_files: return None
    df = pd.read_parquet(os.path.join(DATA_DAILY_DIR, daily_files[0]))
    dates = sorted(df['trade_date'].dropna().unique(), reverse=True)
    return pd.Timestamp(dates[0]) if len(dates) > 0 else None


def init_positions_file():
    """创建持仓文件模板"""
    pd.DataFrame(columns=POSITIONS_COLUMNS).to_csv(POSITIONS_FILE, index=False)
    print(f"   📋 已创建持仓文件模板: {POSITIONS_FILE}")
    print(f"     请手动记录买入: ts_code, name, buy_date, buy_price, shares")
    print(f"     止损/止盈价会自动计算剩余字段")


def check_positions():
    """
    检查持仓股票是否触发卖出条件
    返回: (active_positions, exit_signals)
    """
    if not os.path.exists(POSITIONS_FILE) or os.path.getsize(POSITIONS_FILE) == 0:
        init_positions_file()
        return [], []
    
    try:
        pf = pd.read_csv(POSITIONS_FILE)
    except:
        return [], []
    
    if len(pf) == 0:
        return [], []
    
    # 只检查状态为"持有"的
    active = pf[pf['status'].isna() | (pf['status'] == '持有')].copy()
    if len(active) == 0:
        return [], []
    
    latest_date = get_latest_trade_date()
    if latest_date is None:
        return active.to_dict('records'), []
    
    exit_signals = []
    remaining = []
    
    for idx, row in active.iterrows():
        code = row['ts_code']
        fp = os.path.join(DATA_DAILY_DIR, f"{code}.parquet")
        if not os.path.exists(fp):
            remaining.append(row)
            continue
        try:
            df = pd.read_parquet(fp).sort_values('trade_date').reset_index(drop=True)
        except:
            remaining.append(row)
            continue
        
        # 找到买入后到现在的数据
        buy_dt = pd.Timestamp(row['buy_date'])
        buy_mask = df['trade_date'] >= buy_dt
        if not buy_mask.any():
            remaining.append(row)
            continue
        
        buy_idx = buy_mask.argmax()
        recent = df.iloc[buy_idx:]
        
        if len(recent) == 0:
            remaining.append(row)
            continue
        
        hold_days = len(recent) - 1  # 买入后的交易日数
        buy_price = row['buy_price']
        stop_price = row['stop_price']
        target_price = row['target_price']
        limit_low = row.get('limit_low', stop_price / STOP_LOSS_PCT if stop_price > 0 else 0)
        
        # ATR追踪止损更新
        c = recent['close'].values.astype(np.float64)
        h = recent['high'].values.astype(np.float64)
        l = recent['low'].values.astype(np.float64)
        
        lookback = max(0, buy_idx - ATR_PERIOD - 5) if buy_idx > ATR_PERIOD else 0
        atr_vals = calc_atr(
            df.iloc[lookback:buy_idx+1]['high'].values.astype(np.float64),
            df.iloc[lookback:buy_idx+1]['low'].values.astype(np.float64),
            df.iloc[lookback:buy_idx+1]['close'].values.astype(np.float64),
            ATR_PERIOD
        )
        atr_v = atr_vals[-1] if len(atr_vals) > 0 and not np.isnan(atr_vals[-1]) else 0
        
        # 逐日追踪止损
        highest = buy_price
        current_stop = stop_price
        exit_signal = None
        
        for j in range(1, len(recent)):
            ddo, ddh, ddl = (
                float(recent.iloc[j]['open']),
                float(recent.iloc[j]['high']),
                float(recent.iloc[j]['low']),
            )
            
            if ddh > highest:
                highest = ddh
                if atr_v > 0:
                    new_stop = highest - atr_v * ATR_MULTIPLIER
                    cb = boll_lower_at_idx(c[:j+1], BOLL_PERIOD, BOLL_STD)
                    if not np.isnan(cb):
                        new_stop = max(new_stop, cb)
                    current_stop = max(current_stop, new_stop)
            
            trade_date = pd.Timestamp(recent.iloc[j]['trade_date'])
            
            # 止盈
            if ddh >= target_price - 1e-8:
                exit_signal = {
                    'ts_code': code, 'name': row.get('name', ''),
                    'buy_date': str(buy_dt.date()), 'buy_price': buy_price,
                    'exit_price': target_price if ddo < target_price else ddo,
                    'exit_date': str(trade_date.date()),
                    'exit_reason': '止盈 ✅',
                    'ret_pct': round((target_price/buy_price - COST)*100, 2) if 'COST' in dir() else 0,
                    'hold_days': j,
                    'current_stop': round(current_stop, 2),
                    'detail': f"最高{ddh:.2f}触及止盈{target_price:.2f}",
                }
                # 估算收益
                exit_px = target_price if ddo < target_price else ddo
                ret = exit_px / buy_price - 1 - 0.0064
                exit_signal['ret_pct'] = round(ret * 100, 2)
                break
            
            # 止损
            if ddl <= current_stop - 1e-8:
                exit_px = ddo if ddo <= current_stop - 1e-8 else current_stop
                exit_signal = {
                    'ts_code': code, 'name': row.get('name', ''),
                    'buy_date': str(buy_dt.date()), 'buy_price': buy_price,
                    'exit_price': exit_px,
                    'exit_date': str(trade_date.date()),
                    'exit_reason': '止损 ⚠️',
                    'ret_pct': round((exit_px/buy_price - 1 - 0.0064)*100, 2),
                    'hold_days': j,
                    'current_stop': round(current_stop, 2),
                    'detail': f"最低{ddl:.2f}跌破止损{current_stop:.2f}",
                }
                break
            
            # 到期
            if j >= MAX_HOLD:
                exit_px = float(recent.iloc[j]['close'])
                exit_signal = {
                    'ts_code': code, 'name': row.get('name', ''),
                    'buy_date': str(buy_dt.date()), 'buy_price': buy_price,
                    'exit_price': exit_px,
                    'exit_date': str(trade_date.date()),
                    'exit_reason': '到期 ⏰',
                    'ret_pct': round((exit_px/buy_price - 1 - 0.0064)*100, 2),
                    'hold_days': j,
                    'current_stop': round(current_stop, 2),
                    'detail': f"持有{j}日未触发，收盘{exit_px:.2f}",
                }
                break
        
        if exit_signal:
            # 更新止损到最新
            row['stop_price'] = round(current_stop, 2)
            exit_signals.append(exit_signal)
            # 标记该持仓已平仓（稍后在输出中提示用户手动更新持仓文件）
        else:
            # 更新止损追踪线
            row['stop_price'] = round(current_stop, 2)
            remaining.append(row)
    
    return remaining, exit_signals


def run_screener():
    print("=" * 60)
    print("📊 涨停突破 v6 — 每日选股+持仓管理")
    print("=" * 60)
    t0 = time.time()
    
    today = pd.Timestamp.now()
    latest_date = get_latest_trade_date()
    if latest_date is None:
        print("   ❌ 无法获取交易日")
        return
    next_date = latest_date + timedelta(days=1)
    while next_date.weekday() >= 5:
        next_date += timedelta(days=1)
    
    print(f"   系统时间: {today.strftime('%Y-%m-%d %H:%M')}")
    print(f"   最新交易日: {latest_date.strftime('%Y-%m-%d')}")
    print(f"   预计次日: {next_date.strftime('%Y-%m-%d')} 开盘关注")
    print()
    
    # ====== 0. 检查持仓和卖出信号 ======
    print("[0/5] 检查持仓...")
    active_positions, exit_signals = check_positions()
    print(f"   持仓: {len(active_positions)}只 | 卖出信号: {len(exit_signals)}")
    print()
    
    # ====== 1. 加载 ======
    print("[1/5] 加载股票...")
    sl = pd.read_parquet(STOCK_LIST_FILE)
    codes = sorted(sl['ts_code'].unique())
    name_map = dict(zip(sl['ts_code'], sl.get('name', ['']*len(sl))))
    print(f"   共{len(codes)}只")
    
    print("[2/5] 加载沪深300...")
    market_filter = load_market_filter()
    
    # ====== 2. 扫描买入信号 ======
    print("[3/5] 扫描买入信号...")
    all_signals = []
    
    for idx, code in enumerate(codes):
        if (idx+1) % 500 == 0:
            print(f"   进度: {idx+1}/{len(codes)} ({100*(idx+1)//len(codes)}%)")
        fp = os.path.join(DATA_DAILY_DIR, f"{code}.parquet")
        if not os.path.exists(fp): continue
        try: df = pd.read_parquet(fp)
        except: continue
        if len(df) < MIN_TRADE_DAYS: continue
        df = df.sort_values('trade_date').reset_index(drop=True)
        cutoff = latest_date - timedelta(days=90)
        recent = df[df['trade_date'] >= pd.Timestamp(cutoff)]
        if len(recent) < 20: continue
        
        c, h, l, v = [recent[k].values.astype(np.float64) for k in ['close','high','low','vol']]
        n = len(c)
        
        ma20 = np.full(n, np.nan)
        if n >= 20:
            s = np.cumsum(c); ma20[19] = s[19]/20
            for i in range(20, n): ma20[i] = (s[i]-s[i-20])/20
        mu = np.full(n, False, dtype=bool); mu[1:] = ma20[1:] > ma20[:-1]
        
        lu = np.full(n, False, dtype=bool)
        lu[1:] = (c[1:]/c[:-1] > LIMIT_UP_PCT) & (c[1:] == h[1:])
        
        lli = np.full(n, -1, dtype=np.int32); ls = -1
        for i in range(n):
            if lu[i]: ls = i
            lli[i] = i - ls if ls >= 0 else -1
        
        for i in range(n):
            ds = lli[i]
            if ds <= 0 or ds < MIN_DAYS_SINCE_LIMIT or ds > MAX_DAYS_SINCE_LIMIT: continue
            lb = i - ds
            if not mu[lb]: continue
            lh, ll, lv = h[lb], l[lb], v[lb]
            if lv <= 0: continue
            shrink = any(v[j] <= lv * SHRINK_VOL_RATIO for j in range(lb+1, i))
            if not shrink: continue
            if not (c[i] > lh and v[i] > lv * BREAKOUT_VOL_RATIO): continue
            
            bl = boll_lower_at_idx(c[:i+1], BOLL_PERIOD, BOLL_STD)
            signal_low = float(recent.iloc[i]['low'])
            day_low = signal_low  # 信号日的当日最低价
            entry_price = None
            if bl is not None and not np.isnan(bl) and bl > 0:
                ref = min(float(recent.iloc[i]['open']), max(bl, signal_low))
                # 参考入场价不能低于当日最低价
                entry_price = max(ref, day_low)
            else:
                entry_price = float(recent.iloc[i]['open'])
            
            # 计算止损参考价
            limit_low = ll
            atr_window = c[:i+1]
            bl_stop = bl if (bl is not None and not np.isnan(bl)) else 0
            
            all_signals.append({
                'ts_code': code, 'name': name_map.get(code, ''),
                'signal_date': recent.iloc[i]['trade_date'],
                'signal_date_str': pd.Timestamp(recent.iloc[i]['trade_date']).strftime('%Y-%m-%d'),
                'limit_date': recent.iloc[lb]['trade_date'],
                'days_since_limit': ds,
                'limit_high': round(lh, 2), 'limit_low': round(ll, 2),
                'limit_vol_ratio': round(v[i]/lv, 2) if lv > 0 else 0,
                'entry_price_reference': round(entry_price, 2) if entry_price else 0,
                'boll_lower': round(bl, 2) if (bl is not None and not np.isnan(bl)) else 0,
                'close_price': round(c[i], 2),
                'stop_reference': round(limit_low * STOP_LOSS_PCT, 2),
                'target_reference': round(entry_price * (1+TAKE_PROFIT_PCT), 2) if entry_price else 0,
                'boll_stop': round(bl_stop, 2) if bl_stop > 0 else 0,
            })
    
    print(f"   总信号: {len(all_signals):,}")
    
    # ====== 3. 过滤 ======
    print("[4/5] 过滤...")
    signal_count_map = Counter()
    valid_sigs = [s for s in all_signals
                  if (market_filter is None or
                      s['signal_date_str'] in market_filter and market_filter[s['signal_date_str']])]
    for s in valid_sigs:
        signal_count_map[s['signal_date_str']] += 1
    
    filtered = []
    for s in all_signals:
        if market_filter is not None and s['signal_date_str'] not in market_filter: continue
        if market_filter is not None and not market_filter[s['signal_date_str']]: continue
        if signal_count_map.get(s['signal_date_str'], 0) < MIN_MARKET_SIGNALS: continue
        filtered.append(s)
    
    signals_by_date = {}
    for s in filtered:
        d = s['signal_date_str']
        if d not in signals_by_date: signals_by_date[d] = []
        signals_by_date[d].append(s)
    
    sorted_dates = sorted(signals_by_date.keys(), reverse=True)
    print(f"   过滤后: {len(filtered):,} | 天数: {len(sorted_dates)}")
    
    # ====== 4. 保存 ======
    print("[5/5] 输出...")
    
    df_sigs = pd.DataFrame(filtered).sort_values(['signal_date_str', 'ts_code'])
    df_sigs.to_csv(os.path.join(OUTPUT_DIR, "v6_screener_latest.csv"), index=False)
    
    hist_exists = os.path.exists(HISTORY_FILE) and os.path.getsize(HISTORY_FILE) > 0
    if hist_exists:
        hist = pd.read_csv(HISTORY_FILE)
        combined = pd.concat([hist, df_sigs], ignore_index=True)
    else:
        combined = df_sigs
    combined.to_csv(HISTORY_FILE, index=False)
    
    # ====== 5. 输出报告 ======
    print()
    print("=" * 70)
    print("📋 每日信号报告")
    print("=" * 70)
    
    # --- 卖出信号 ---
    if exit_signals:
        print(f"\n🔴 卖出提示（共{len(exit_signals)}只，建议今日操作）")
        print("─" * 70)
        print(f"  {'股票':<12} {'买入日':<12} {'买入价':<10} {'卖出价':<10} {'收益':<10} {'持有天':<8} {'原因'}")
        print(f"  {'─'*10} {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*10}")
        for es in exit_signals:
            print(f"  {es['ts_code']:<12} {es['buy_date']:<12} {es['buy_price']:<10.2f} "
                  f"{es['exit_price']:<10.2f} {es['ret_pct']:>+7.2f}%  {es['hold_days']:<6} {es['exit_reason']}")
        print(f"\n  💡 操作: 请在 {POSITIONS_FILE} 中标记该持仓 status=已平仓")
    
    # --- 持仓状态 ---
    if active_positions:
        print(f"\n📦 当前持仓（{len(active_positions)}只，最新止损追踪线）")
        print("─" * 60)
        print(f"  {'股票':<12} {'买入日':<12} {'买入价':<10} {'止损价':<10} {'止盈价':<10}")
        for p in active_positions:
            print(f"  {p['ts_code']:<12} {str(p['buy_date'])[:10]:<12} {p['buy_price']:<10.2f} "
                  f"{p['stop_price']:<10.2f} {p['target_price']:<10.2f}")
    
    # --- 买入信号 ---
    print(f"\n🟢 买入信号（次日{next_date.strftime('%m-%d')}开盘关注）")
    print("─" * 70)
    
    any_tradeable = False
    for d in sorted_dates[:3]:
        sigs_today = signals_by_date[d]
        count_today = signal_count_map.get(d, 0)
        hs300 = market_filter.get(d, False) if market_filter else False
        
        can_trade = count_today >= MIN_MARKET_SIGNALS and hs300
        if can_trade:
            any_tradeable = True
        
        status = "🟢 可开仓" if can_trade else "🔴 不开仓"
        print(f"\n{d} — {len(sigs_today)}只 {status}")
        print(f"   全市场信号:{count_today} 沪深300:{'向上✅' if hs300 else '向下❌'}")
        
        if can_trade:
            for s in sigs_today[:10]:
                stop_ref = s.get('stop_reference', 0)
                target_ref = s.get('target_reference', 0)
                print(f"   {s['ts_code']} {s['name']:<8} "
                      f"涨停后{s['days_since_limit']}天 "
                      f"量{s['limit_vol_ratio']}× "
                      f"参考入场≤{s['entry_price_reference']} "
                      f"止损{stop_ref} 止盈{target_ref}")
            if len(sigs_today) > 10:
                print(f"   ... 还有{len(sigs_today)-10}只")
    
    if not any_tradeable:
        print("\n   ⚠️ 今日无满足条件的买入信号")
    
    # --- 持仓文件提示 ---
    if not os.path.exists(POSITIONS_FILE) or os.path.getsize(POSITIONS_FILE) == 0:
        print(f"\n📋 首次运行: {POSITIONS_FILE}")
        print(f"   记录买入后，次日运行本程序即可自动检查卖出条件")
        print(f"   格式: ts_code,name,buy_date,buy_price,shares,stop_price,target_price,limit_low,status")
    else:
        print(f"\n📋 持仓文件: {POSITIONS_FILE}")
        print(f"   买入后请手动追加记录，卖出后标记 status=已平仓")
    
    # --- 保存JSON摘要 ---
    report = {
        'run_time': today.strftime('%Y-%m-%d %H:%M:%S'),
        'latest_trade_date': latest_date.strftime('%Y-%m-%d'),
        'next_trade_date': next_date.strftime('%Y-%m-%d'),
        'positions_active': len(active_positions),
        'exit_signals': [
            {'ts_code': es['ts_code'], 'exit_reason': es['exit_reason'],
             'exit_price': es['exit_price'], 'ret_pct': es['ret_pct'],
             'hold_days': es['hold_days']} for es in exit_signals
        ],
        'buy_signals_today': len(filtered),
        'top_signals': [
            {'ts_code': s['ts_code'], 'name': s['name'],
             'entry_price_reference': s['entry_price_reference'],
             'stop_reference': s.get('stop_reference', 0),
             'target_reference': s.get('target_reference', 0)}
            for s in filtered[:20]
        ],
    }
    with open(os.path.join(OUTPUT_DIR, "v6_signals_summary.json"), 'w') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n⏱ 用时: {time.time()-t0:.0f}秒")
    print(f"\n💡 操作流程:")
    print(f"   1️⃣ 早盘查看买入信号 → 关注BOLL低吸位置")
    print(f"   2️⃣ 查看卖出提示 → 到达止损/止盈价的持仓及时卖出")
    print(f"   3️⃣ 记录新买入 → 追加到 {POSITIONS_FILE}")
    print(f"   4️⃣ 标记已平仓 → 在 {POSITIONS_FILE} 中改 status=已平仓")


if __name__ == "__main__":
    run_screener()
