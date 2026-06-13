#!/usr/bin/env python3
"""
涨停缩量回调 → 放量突破 第二波策略 v4（优化版）
=================================================

相对v3的优化:
  1. 大盘过滤: 沪深300的MA20向下时不开仓
  2. 放量条件收紧: 突破日量 > 涨停日量 × 1.2
  3. ATR动态止损 + BOLL下轨 + 固定止损 三重防线
  4. 风险控制: 单月亏损超过20%暂停交易

选股逻辑:
  1. 最近20天内有涨停，涨停当天MA20向上
  2. 涨停后3~20天
  3. 涨停后有缩量（量≤涨停日×0.5）
  4. 缩量后放量突破（收盘>涨停最高 + 量>涨停量×1.2）
  5. 买入: BOLL(20,2)下轨低吸 (min(开盘价, 下轨), 不下于信号日最低价)

出场:
  止损: max(涨停日最低价×0.95, ATR(14)×2动态追踪, BOLL下轨)
  止盈: +20%
  持有上限: 30日
"""
import pandas as pd
import numpy as np
import os, sys, time, warnings, math
warnings.filterwarnings('ignore')

PROJ_B = "/mnt/d/AI-20260604"
DATA_DAILY_DIR = os.path.join(PROJ_B, "data", "raw", "daily")
STOCK_LIST_FILE = os.path.join(PROJ_B, "data", "raw", "stock_list.parquet")
OUTPUT_DIR = os.path.join(PROJ_B, "backtest_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====== 参数 ======
LIMIT_UP_PCT = 1.095
MAX_DAYS_SINCE_LIMIT = 20
MIN_DAYS_SINCE_LIMIT = 3
MIN_TRADE_DAYS = 120
COST_PER_TRADE = 0.0032
SHRINK_VOL_RATIO = 0.5
BREAKOUT_VOL_RATIO = 1.2
TAKE_PROFIT_PCT = 0.20
STOP_LOSS_PCT = 0.95
MAX_HOLD = 30

# BOLL参数
BOLL_PERIOD = 20
BOLL_STD = 2.0

# ATR参数
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0

# 风控
MAX_MONTHLY_LOSS_PCT = -20

START_DATE = "2017-01-01"
END_DATE = "2026-06-03"

INDEX_PATH = os.path.join(PROJ_B, "data", "raw", "index_000300.parquet")


def calc_atr(high, low, close, period=14):
    n = len(high)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr


def boll_lower_at_idx(close, target_idx, period=20, n_std=2.0):
    if target_idx < period - 1:
        return np.nan
    window = close[target_idx - period + 1 : target_idx + 1]
    ma = np.mean(window)
    std = np.std(window, ddof=1)
    return ma - n_std * std


def load_market_filter():
    """
    加载沪深300指数，计算MA20
    返回: dict {trade_date_str: is_ma20_up}
    """
    print("   加载沪深300(MA20过滤)...", end=" ", flush=True)
    df = pd.read_parquet(INDEX_PATH)
    df = df.sort_values('trade_date').reset_index(drop=True)
    close = df['close'].values.astype(np.float64)
    n = len(df)
    
    ma20 = np.full(n, np.nan)
    if n >= 20:
        cumsum = np.cumsum(close)
        ma20[19] = cumsum[19] / 20
        for i in range(20, n):
            ma20[i] = (cumsum[i] - cumsum[i-20]) / 20
    
    ma20_up = np.full(n, False, dtype=bool)
    ma20_up[1:] = ma20[1:] > ma20[:-1]
    
    result = {}
    for i in range(n):
        dt = pd.Timestamp(df.iloc[i]['trade_date'])
        result[dt.strftime('%Y-%m-%d')] = bool(ma20_up[i])
    
    print(f"done ({len(result)}日)")
    return result


def detect_breakout_signals(df):
    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    vol = df['vol'].values.astype(np.float64)
    n = len(df)
    
    ma20_seq = np.full(n, np.nan)
    if n >= 20:
        cumsum = np.cumsum(close)
        ma20_seq[19] = cumsum[19] / 20
        for i in range(20, n):
            ma20_seq[i] = (cumsum[i] - cumsum[i-20]) / 20
    ma20_up = np.full(n, False, dtype=bool)
    ma20_up[1:] = ma20_seq[1:] > ma20_seq[:-1]
    
    limit_up = np.full(n, False, dtype=bool)
    limit_up[1:] = (close[1:] / close[:-1] > LIMIT_UP_PCT) & (close[1:] == high[1:])
    
    last_limit_idx = np.full(n, -1, dtype=np.int32)
    last_seen = -1
    for i in range(n):
        if limit_up[i]:
            last_seen = i
        last_limit_idx[i] = i - last_seen if last_seen >= 0 else -1
    
    signals = []
    for i in range(n):
        days_since = last_limit_idx[i]
        if days_since <= 0 or days_since < MIN_DAYS_SINCE_LIMIT or days_since > MAX_DAYS_SINCE_LIMIT:
            continue
        limit_bar = i - days_since
        if not ma20_up[limit_bar]:
            continue
        limit_high = high[limit_bar]
        limit_low = low[limit_bar]
        limit_vol = vol[limit_bar]
        if limit_vol <= 0 or np.isnan(limit_vol):
            continue
        has_shrink = False
        for j in range(limit_bar + 1, i):
            if vol[j] <= limit_vol * SHRINK_VOL_RATIO:
                has_shrink = True
                break
        if not has_shrink:
            continue
        if not (close[i] > limit_high and vol[i] > limit_vol * BREAKOUT_VOL_RATIO):
            continue
        
        bl = boll_lower_at_idx(close, i, BOLL_PERIOD, BOLL_STD)
        signals.append({
            'idx': i, 'limit_bar': limit_bar,
            'limit_high': limit_high, 'limit_low': limit_low,
            'limit_vol': limit_vol, 'boll_lower': bl if not np.isnan(bl) else None,
        })
    return signals


def run_backtest():
    print("=" * 70)
    print("📊 涨停缩量回调→放量突破 v4 (沪深300过滤)")
    print("=" * 70)
    print(f"   区间: {START_DATE} ~ {END_DATE}")
    print(f"   双过滤(沪深300 MA20+市场热度≥3) + 涨停后≥{MIN_DAYS_SINCE_LIMIT}天")
    print(f"   放量条件: 量>涨停量×{BREAKOUT_VOL_RATIO}")
    print(f"   买入: BOLL({BOLL_PERIOD},{BOLL_STD})下轨低吸")
    print(f"   止损: ATR({ATR_PERIOD})×{ATR_MULTIPLIER} + BOLL下轨 + 涨停最低×{STOP_LOSS_PCT}")
    print(f"   止盈: +{TAKE_PROFIT_PCT*100:.0f}% | 风控: 月亏>{abs(MAX_MONTHLY_LOSS_PCT)}%暂停")
    print(f"   成本: {COST_PER_TRADE*100:.2f}%/边")
    print()

    # ========== 0. 大盘过滤 ==========
    print("[0/5] 加载沪深300...")
    market_filter = load_market_filter()
    print()

    # ========== 1. 加载股票 ==========
    print("[1/5] 加载股票列表...")
    stock_list = pd.read_parquet(STOCK_LIST_FILE)
    ts_codes = sorted(stock_list['ts_code'].unique())
    name_map = dict(zip(stock_list['ts_code'], stock_list.get('name', [''] * len(stock_list))))
    print(f"   共 {len(ts_codes)} 只股票")

    # ========== 2. 扫描信号 ==========
    print("[2/5] 逐股扫描信号...")
    all_signals = []
    total_stocks = len(ts_codes)

    for idx, ts_code in enumerate(ts_codes):
        if (idx + 1) % 1000 == 0:
            print(f"   进度: {idx+1}/{total_stocks} ({100*(idx+1)/total_stocks:.0f}%)")
        fpath = os.path.join(DATA_DAILY_DIR, f"{ts_code}.parquet")
        if not os.path.exists(fpath):
            continue
        try:
            df = pd.read_parquet(fpath)
        except Exception:
            continue
        if len(df) < MIN_TRADE_DAYS:
            continue
        df = df.sort_values('trade_date').reset_index(drop=True)
        mask = (df['trade_date'] >= np.datetime64(START_DATE)) & (df['trade_date'] <= np.datetime64(END_DATE))
        if not mask.any():
            continue
        start_idx = max(0, mask.argmax() - 60)
        sigs = detect_breakout_signals(df.iloc[start_idx:].reset_index(drop=True))
        for s in sigs:
            global_idx = start_idx + s['idx']
            if global_idx >= len(df):
                continue
            all_signals.append({
                'ts_code': ts_code, 'name': name_map.get(ts_code, ''),
                'signal_date': df.iloc[global_idx]['trade_date'],
                'signal_idx': global_idx,
                'limit_bar_global': start_idx + s['limit_bar'],
                'limit_high': s['limit_high'], 'limit_low': s['limit_low'],
                'limit_vol': s['limit_vol'], 'boll_lower': s['boll_lower'],
            })

    print(f"   总信号数: {len(all_signals):,}")
    
    # 计算每日信号数（市场热度过滤用）
    from collections import Counter
    signal_count_map = Counter()
    for s in all_signals:
        dt = pd.Timestamp(s['signal_date']).strftime('%Y-%m-%d')
        signal_count_map[dt] += 1
    signal_count_map = dict(signal_count_map)
    print(f"   市场热度: {len(signal_count_map)}日")
    print()

    # ========== 3. 模拟交易 ==========
    print("[3/5] 模拟交易...")
    print("   构建数据索引...", end=" ", flush=True)
    code_dfs = {}
    for ts_code in ts_codes:
        fpath = os.path.join(DATA_DAILY_DIR, f"{ts_code}.parquet")
        if os.path.exists(fpath):
            try:
                df = pd.read_parquet(fpath).sort_values('trade_date').reset_index(drop=True)
                code_dfs[ts_code] = df
            except Exception:
                pass
    print(f"done ({len(code_dfs)}只)")

    monthly_pnl = {}
    trades = []
    skipped_market_total = 0
    skipped_riskctrl = 0

    for tidx, sig in enumerate(all_signals):
        if (tidx + 1) % 1000 == 0:
            print(f"   进度: {tidx+1}/{len(all_signals)} ({100*(tidx+1)/len(all_signals):.1f}%)")
        
        ts_code = sig['ts_code']
        signal_idx = sig['signal_idx']
        signal_date = pd.Timestamp(sig['signal_date'])
        
        # 双过滤: 沪深300 MA20 + 市场热度
        dt_key = signal_date.strftime('%Y-%m-%d')
        if dt_key in market_filter and not market_filter[dt_key]:
            skipped_market_total += 1
            continue
        
        # 市场热度: 当日全市场信号<3家不开仓
        dt_key2 = signal_date.strftime('%Y-%m-%d')
        signal_count_today = signal_count_map.get(dt_key2, 0)
        if signal_count_today < 3:
            skipped_market_total += 1
            continue
        
        # 月风控
        ym = signal_date.strftime('%Y-%m')
        if ym in monthly_pnl and monthly_pnl[ym] <= MAX_MONTHLY_LOSS_PCT:
            skipped_riskctrl += 1
            continue
        
        df = code_dfs.get(ts_code)
        if df is None or signal_idx + 1 >= len(df):
            continue
        
        entry_idx = signal_idx + 1
        day_open = float(df.iloc[entry_idx]['open'])
        if day_open <= 0:
            continue
        
        # 买入价
        boll_lower = sig.get('boll_lower')
        signal_low = float(df.iloc[signal_idx]['low'])
        if boll_lower is not None and not np.isnan(boll_lower) and boll_lower > 0:
            entry_price = min(day_open, max(boll_lower, signal_low))
        else:
            entry_price = day_open
        if entry_price <= 0:
            continue
        
        # 止损: 三重防线
        stop_fixed = sig['limit_low'] * STOP_LOSS_PCT
        
        atr_lookback = max(0, entry_idx - ATR_PERIOD - 5)
        atr_vals = calc_atr(
            df.iloc[atr_lookback:entry_idx+1]['high'].values.astype(np.float64),
            df.iloc[atr_lookback:entry_idx+1]['low'].values.astype(np.float64),
            df.iloc[atr_lookback:entry_idx+1]['close'].values.astype(np.float64),
            ATR_PERIOD,
        )
        atr_val = atr_vals[-1] if len(atr_vals) > 0 and not np.isnan(atr_vals[-1]) else 0
        stop_atr = entry_price - atr_val * ATR_MULTIPLIER if atr_val > 0 else 0
        
        stop_boll = boll_lower if (boll_lower is not None and not np.isnan(boll_lower)) else 0
        if isinstance(stop_boll, (np.floating, float)) and np.isnan(stop_boll):
            stop_boll = 0
        
        stop_price = max(p for p in [stop_fixed, stop_atr, stop_boll] if p > 0)
        target_price = entry_price * (1 + TAKE_PROFIT_PCT)
        highest_since_entry = entry_price
        
        exit_idx, exit_price_val, exit_reason = None, None, None
        
        for la in range(1, MAX_HOLD + 1):
            if entry_idx + la >= len(df):
                break
            row = df.iloc[entry_idx + la]
            do, dh, dl = row['open'], row['high'], row['low']
            
            if dh > highest_since_entry:
                highest_since_entry = dh
                new_stop = highest_since_entry - atr_val * ATR_MULTIPLIER
                cb = boll_lower_at_idx(df['close'].values.astype(np.float64), entry_idx + la, BOLL_PERIOD, BOLL_STD)
                if not np.isnan(cb):
                    stop_boll = cb
                stop_price = max(stop_price, new_stop, stop_boll)
            
            if dh >= target_price - 1e-8:
                exit_price_val = target_price if do < target_price else do
                exit_idx, exit_reason = entry_idx + la, 'take_profit'
                break
            
            if dl <= stop_price - 1e-8:
                exit_price_val = do if do <= stop_price - 1e-8 else stop_price
                exit_idx, exit_reason = entry_idx + la, 'stop_loss'
                break
        
        if exit_idx is None:
            last_idx = min(entry_idx + MAX_HOLD, len(df) - 1)
            exit_idx, exit_price_val, exit_reason = last_idx, float(df.iloc[last_idx]['close']), 'timeout'
        
        ret = exit_price_val / entry_price - 1
        ret_ac = ret - COST_PER_TRADE * 2
        
        exit_ym = pd.Timestamp(df.iloc[exit_idx]['trade_date']).strftime('%Y-%m')
        monthly_pnl[exit_ym] = monthly_pnl.get(exit_ym, 0) + ret_ac
        
        trades.append({
            'ts_code': ts_code, 'name': sig.get('name', ''),
            'signal_date': signal_date,
            'entry_date': df.iloc[entry_idx]['trade_date'],
            'entry_price': round(entry_price, 3), 'day_open': round(day_open, 3),
            'exit_date': df.iloc[exit_idx]['trade_date'],
            'exit_price': round(exit_price_val, 3),
            'exit_reason': exit_reason, 'hold_days': exit_idx - entry_idx,
            'stop_price': round(stop_price, 3),
            'ret': round(ret * 100, 2), 'ret_after_cost': round(ret_ac * 100, 2),
        })

    trades_df = pd.DataFrame(trades)
    print(f"   总交易: {len(trades_df):,} 笔")
    print(f"   沪深300过滤跳过: {skipped_market_total:,}")
    print(f"   风控过滤跳过: {skipped_riskctrl:,}")
    print()

    # ========== 4. 分析 ==========
    print("[4/5] 绩效分析...")
    if len(trades_df) == 0:
        print("   ⚠️ 无交易")
        return

    n_trades = len(trades_df)
    win_mask = trades_df['ret_after_cost'] > 0
    n_wins, win_rate = win_mask.sum(), win_mask.mean() * 100
    avg_ret, avg_win, avg_loss = (
        trades_df['ret_after_cost'].mean(),
        trades_df.loc[win_mask, 'ret_after_cost'].mean(),
        trades_df.loc[~win_mask, 'ret_after_cost'].mean(),
    )
    total_ret = trades_df['ret_after_cost'].sum()

    monthly_rets = trades_df.groupby(trades_df['entry_date'].dt.to_period('M'))['ret_after_cost'].sum()
    sharpe = monthly_rets.mean() / monthly_rets.std() * math.sqrt(12) if len(monthly_rets) > 1 and monthly_rets.std() > 0 else 0

    trades_df = trades_df.sort_values('exit_date').reset_index(drop=True)
    nav = (1 + trades_df['ret_after_cost'] / 100).cumprod()
    peak, dd = nav.expanding().max(), (nav / nav.expanding().max() - 1)
    max_dd = dd.min()

    trades_df['year'] = trades_df['entry_date'].dt.year
    yearly = trades_df.groupby('year').agg(交易次数=('ret_after_cost', 'count'), 胜率=('ret_after_cost', lambda x: (x > 0).mean() * 100), 平均收益=('ret_after_cost', 'mean'), 总收益=('ret_after_cost', 'sum')).round(2)

    reason_stats = trades_df.groupby('exit_reason').agg(次数=('ret_after_cost', 'count'), 胜率=('ret_after_cost', lambda x: (x > 0).mean() * 100), 平均收益=('ret_after_cost', 'mean'), 平均持有天数=('hold_days', 'mean')).round(2)

    trades_df['hold_bucket'] = pd.cut(trades_df['hold_days'], bins=[0, 1, 3, 5, 10, 20, 100], labels=['1天', '2-3天', '4-5天', '6-10天', '11-20天', '21天+'])
    hold_stats = trades_df.groupby('hold_bucket', observed=True).agg(次数=('ret_after_cost', 'count'), 胜率=('ret_after_cost', lambda x: (x > 0).mean() * 100), 平均收益=('ret_after_cost', 'mean')).round(2)

    trades_df['boll_lower_used'] = trades_df['entry_price'] < trades_df['day_open'] - 0.001
    boll_used, boll_stats = trades_df['boll_lower_used'].sum(), trades_df.groupby('boll_lower_used').agg(次数=('ret_after_cost', 'count'), 胜率=('ret_after_cost', lambda x: (x > 0).mean() * 100), 平均收益=('ret_after_cost', 'mean')).round(2)
    
    # ========== 5. 输出 ==========
    print("\n[5/5] 输出结果...\n")

    print("=" * 70)
    print("📊 回测结果 — 涨停缩量回调→放量突破 v4")
    print("=" * 70)
    print(f"  {'总交易数':<22} {n_trades:>8,}")
    print(f"  {'胜率':<22} {win_rate:>7.2f}%")
    print(f"  {'平均单笔(扣费后)':<22} {avg_ret:>+8.2f}%")
    print(f"  {'平均盈利/亏损':<22} {avg_win:>+7.2f}% / {avg_loss:>+6.2f}%")
    print(f"  {'累计总收益(扣费后)':<22} {total_ret:>+8.2f}%")
    print(f"  {'月频夏普比率':<22} {sharpe:>8.2f}")
    print(f"  {'最大回撤':<22} {max_dd*100:>7.2f}%")
    print(f"  {'月胜率':<22} {len(monthly_rets[monthly_rets>0])/len(monthly_rets)*100:>7.1f}%")
    print(f"  {'沪深300过滤':<22} {skipped_market_total:>8,}")
    print(f"  {'BOLL低吸比例':<22} {100*boll_used/n_trades:>7.1f}%")
    print()

    print("─" * 70)
    print("📅 各年统计")
    print("─" * 70)
    print(f"  {'年份':>6} {'次数':>8} {'胜率':>8} {'平均收益':>10} {'总收益':>10}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
    for yr, row in yearly.iterrows():
        print(f"  {int(yr):>6} {row['交易次数']:>8} {row['胜率']:>7.1f}% {row['平均收益']:>+9.2f}% {row['总收益']:>+9.2f}%")
    print()

    print("─" * 70)
    print("🏁 退出原因分析")
    print("─" * 70)
    print(f"  {'原因':<15} {'次数':>8} {'胜率':>8} {'平均收益':>10} {'平均持有天':>10}")
    for reason, row in reason_stats.iterrows():
        label = {'stop_loss':'止损', 'take_profit':'止盈', 'timeout':'到期'}.get(reason, reason)
        print(f"  {label:<15} {row['次数']:>8} {row['胜率']:>7.1f}% {row['平均收益']:>+9.2f}% {row['平均持有天数']:>9.1f}")
    print()

    print("─" * 70)
    print("📆 持有天数分布")
    print("─" * 70)
    print(f"  {'持有天数':<12} {'次数':>8} {'胜率':>8} {'平均收益':>10}")
    for bucket, row in hold_stats.iterrows():
        print(f"  {str(bucket):<12} {row['次数']:>8} {row['胜率']:>7.1f}% {row['平均收益']:>+9.2f}%")

    print()
    print("─" * 70)
    print("🏆 最佳3 / 💀 最差3")
    print("─" * 70)
    for _, row in trades_df.nlargest(3, 'ret_after_cost').iterrows():
        print(f"  🏆 {row['ts_code']} {row.get('name',''):<8} "
              f"{row['ret_after_cost']:+.2f}% ({row['exit_reason']}) 持{row['hold_days']}天")
    for _, row in trades_df.nsmallest(3, 'ret_after_cost').iterrows():
        print(f"  💀 {row['ts_code']} {row.get('name',''):<8} "
              f"{row['ret_after_cost']:+.2f}% ({row['exit_reason']}) 持{row['hold_days']}天")

    # ========== 保存 ==========
    print()
    print("─" * 70)
    print("💾 保存结果...")
    
    trades_df.to_csv(os.path.join(OUTPUT_DIR, "breakout_v4_backtest.csv"), index=False)
    summary = pd.DataFrame([{
        'start_date': START_DATE, 'end_date': END_DATE,
        'total_trades': n_trades, 'win_rate_pct': round(win_rate, 2),
        'avg_return_pct': round(avg_ret, 2),
        'avg_win_pct': round(avg_win if not pd.isna(avg_win) else 0, 2),
        'avg_loss_pct': round(avg_loss if not pd.isna(avg_loss) else 0, 2),
        'total_return_pct': round(total_ret, 2),
        'monthly_sharpe': round(sharpe, 2),
        'max_drawdown_pct': round(max_dd * 100, 2),
        'month_win_rate': round(len(monthly_rets[monthly_rets>0])/len(monthly_rets)*100, 1),
        'boll_lower_used_pct': round(100 * boll_used / n_trades, 1) if n_trades > 0 else 0,
        'cost_per_trade': COST_PER_TRADE,
    }])
    summary.to_csv(os.path.join(OUTPUT_DIR, "breakout_v4_summary.csv"), index=False)
    print(f"   ✅ breakouts_v4_backtest.csv / summary.csv")
    print(f"\n⏱ {time.time()-t0:.0f}秒 ({(time.time()-t0)/60:.1f}分)")


if __name__ == "__main__":
    t0 = time.time()
    run_backtest()
