#!/usr/bin/env python3
"""
涨停缩量回调 → 放量突破 第二波策略 v3 — 10年回测

改动:
  1. 涨停当天MA20必须向上（选股收紧）
  2. 买入价用BOLL(20,2)下轨（低吸，不追高）
     - BOLL下轨不得低于信号当日最低价（防止前复权/重组异常）
  
选股逻辑:
  1. 最近20天内有涨停 (C/REF(C,1)>1.095 AND C=H)
  2. 涨停当天MA20 > REF(MA20,1)
  3. 涨停后至少3天
  4. 涨停后某天缩量: 成交量 <= 涨停日成交量 × 0.5
  5. 缩量之后某天放量突破: 收盘价 > 涨停日最高价 + 量 > 涨停日成交量
  6. 信号次日在BOLL(20,2)下轨附近买入（取min(开盘价, BOLL下轨)，但不下于信号日最低价）

出场:
  止损: 涨停日最低价 × 0.97（追踪，最高回落15%）
  止盈: +20%
  持有上限: 30个交易日
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
BREAKOUT_VOL_RATIO = 1.0
TAKE_PROFIT_PCT = 0.20
STOP_LOSS_PCT = 0.97
MAX_HOLD = 30

# BOLL参数
BOLL_PERIOD = 20
BOLL_STD = 2.0

START_DATE = "2017-01-01"
END_DATE = "2026-06-03"


def calc_ma20_and_boll_for_idx(close, high, low, target_idx, period=20, n_std=2.0):
    """
    计算 target_idx 位置的 MA20 和 BOLL 上下轨
    只用 target_idx 前 period 天的数据，避免前复权跨时段问题
    返回: (ma20, boll_upper, boll_mid, boll_lower)
    """
    if target_idx < period - 1:
        return np.nan, np.nan, np.nan, np.nan
    
    # 只用最近period天的close算MA20和标准差
    window_close = close[target_idx - period + 1 : target_idx + 1]
    ma = np.mean(window_close)
    std = np.std(window_close, ddof=1)
    
    boll_upper = ma + n_std * std
    boll_mid = ma
    boll_lower = ma - n_std * std
    
    return ma, boll_upper, boll_mid, boll_lower


def detect_breakout_signals(df):
    """
    检测放量突破信号
    返回: list of dict
    """
    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    vol = df['vol'].values.astype(np.float64)
    n = len(df)
    
    # 计算MA20序列（用于ma20向上判断）
    ma20_seq = np.full(n, np.nan)
    if n >= 20:
        cumsum = np.cumsum(close)
        ma20_seq[19] = cumsum[19] / 20
        for i in range(20, n):
            ma20_seq[i] = (cumsum[i] - cumsum[i-20]) / 20
    
    ma20_up = np.full(n, False, dtype=bool)
    ma20_up[1:] = ma20_seq[1:] > ma20_seq[:-1]
    
    # 涨停
    limit_up = np.full(n, False, dtype=bool)
    limit_up[1:] = (close[1:] / close[:-1] > LIMIT_UP_PCT) & (close[1:] == high[1:])
    
    # 距上次涨停
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
        
        # 条件1: 涨停当天MA20必须向上
        if not ma20_up[limit_bar]:
            continue
        
        limit_high = high[limit_bar]
        limit_low = low[limit_bar]
        limit_vol = vol[limit_bar]
        
        if limit_vol <= 0 or np.isnan(limit_vol):
            continue
        
        # 条件2: 涨停日到昨天之间有缩量
        has_shrink = False
        for j in range(limit_bar + 1, i):
            if vol[j] <= limit_vol * SHRINK_VOL_RATIO:
                has_shrink = True
                break
        
        if not has_shrink:
            continue
        
        # 条件3: 今天放量突破
        if not (close[i] > limit_high and vol[i] > limit_vol * BREAKOUT_VOL_RATIO):
            continue
        
        # 计算信号日的BOLL下轨（只用在买入时）
        _, _, _, boll_lower = calc_ma20_and_boll_for_idx(close, high, low, i, BOLL_PERIOD, BOLL_STD)
        
        signals.append({
            'idx': i,
            'limit_bar': limit_bar,
            'limit_high': limit_high,
            'limit_low': limit_low,
            'limit_vol': limit_vol,
            'boll_lower': boll_lower if not np.isnan(boll_lower) else None,
        })
    
    return signals


def run_backtest():
    print("=" * 70)
    print("📊 涨停缩量回调→放量突破 v3 (MA20向上+BOLL低吸)")
    print("=" * 70)
    print(f"   区间: {START_DATE} ~ {END_DATE}")
    print(f"   涨停MA20向上 + 涨停后≥{MIN_DAYS_SINCE_LIMIT}天")
    print(f"   缩量: 量≤涨停日×{SHRINK_VOL_RATIO}")
    print(f"   突破: 收盘>涨停最高 + 量>涨停量×{BREAKOUT_VOL_RATIO}")
    print(f"   买入: BOLL({BOLL_PERIOD},{BOLL_STD})下轨低吸")
    print(f"   止损: 涨停日最低价×{STOP_LOSS_PCT} + 高点回落15%追踪")
    print(f"   止盈: +{TAKE_PROFIT_PCT*100:.0f}%")
    print(f"   交易成本: {COST_PER_TRADE*100:.2f}%/边")
    print()

    # ========== 1. 加载 ==========
    print("[1/5] 加载股票列表...")
    stock_list = pd.read_parquet(STOCK_LIST_FILE)
    ts_codes = sorted(stock_list['ts_code'].unique())
    name_map = dict(zip(stock_list['ts_code'], stock_list.get('name', [''] * len(stock_list))))
    industry_map = dict(zip(stock_list['ts_code'], stock_list.get('industry', [''] * len(stock_list))))
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
        df_window = df.iloc[start_idx:].reset_index(drop=True)

        sigs = detect_breakout_signals(df_window)
        for s in sigs:
            global_idx = start_idx + s['idx']
            if global_idx >= len(df):
                continue
            all_signals.append({
                'ts_code': ts_code,
                'name': name_map.get(ts_code, ''),
                'industry': industry_map.get(ts_code, ''),
                'signal_date': df.iloc[global_idx]['trade_date'],
                'signal_idx': global_idx,
                'limit_bar_global': start_idx + s['limit_bar'],
                'limit_high': s['limit_high'],
                'limit_low': s['limit_low'],
                'limit_vol': s['limit_vol'],
                'boll_lower': s['boll_lower'],
            })

    print(f"   总信号数: {len(all_signals):,}")
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

    trades = []

    for tidx, sig in enumerate(all_signals):
        if (tidx + 1) % 1000 == 0:
            print(f"   交易进度: {tidx+1}/{len(all_signals)} ({100*(tidx+1)/len(all_signals):.1f}%)")

        ts_code = sig['ts_code']
        signal_idx = sig['signal_idx']

        df = code_dfs.get(ts_code)
        if df is None:
            continue

        if signal_idx + 1 >= len(df):
            continue

        entry_idx = signal_idx + 1
        entry_row = df.iloc[entry_idx]
        day_open = entry_row['open']

        if day_open <= 0 or np.isnan(day_open):
            continue

        # 买入价: BOLL下轨低吸
        boll_lower = sig.get('boll_lower')
        signal_low = float(df.iloc[signal_idx]['low'])

        if boll_lower is not None and not np.isnan(boll_lower) and boll_lower > 0:
            # BOLL下轨不得低于信号当日最低价（防止前复权/重组导致的异常低价）
            boll_lower_clamped = max(boll_lower, signal_low)
            entry_price = min(day_open, boll_lower_clamped)
        else:
            entry_price = day_open
            boll_lower_clamped = day_open

        if entry_price <= 0 or np.isnan(entry_price):
            continue

        # 止损价 = 涨停日最低价 × 0.97
        stop_price = sig['limit_low'] * STOP_LOSS_PCT
        target_price = entry_price * (1 + TAKE_PROFIT_PCT)
        highest_since_entry = entry_price

        exit_idx = None
        exit_price_val = None
        exit_reason = None

        for lookahead in range(1, MAX_HOLD + 1):
            if entry_idx + lookahead >= len(df):
                break

            row = df.iloc[entry_idx + lookahead]
            day_open = row['open']
            day_high = row['high']
            day_low = row['low']

            if day_high > highest_since_entry:
                highest_since_entry = day_high

            # 止盈
            if day_high >= target_price - 1e-8:
                exit_price_val = target_price if day_open < target_price else day_open
                exit_idx = entry_idx + lookahead
                exit_reason = 'take_profit'
                break

            # 止损: 固定止损 + 追踪止损
            current_stop = max(stop_price, highest_since_entry * 0.85)
            if day_low <= current_stop - 1e-8:
                if day_open <= current_stop - 1e-8:
                    exit_price_val = day_open
                else:
                    exit_price_val = current_stop
                exit_idx = entry_idx + lookahead
                exit_reason = 'stop_loss'
                break

        if exit_idx is None:
            last_idx = min(entry_idx + MAX_HOLD, len(df) - 1)
            exit_idx = last_idx
            exit_price_val = df.iloc[last_idx]['close']
            exit_reason = 'timeout'

        ret = exit_price_val / entry_price - 1
        ret_after_cost = ret - COST_PER_TRADE * 2

        trades.append({
            'ts_code': ts_code,
            'name': sig.get('name', ''),
            'industry': sig.get('industry', ''),
            'signal_date': sig['signal_date'],
            'entry_date': df.iloc[entry_idx]['trade_date'],
            'entry_price': round(entry_price, 3),
            'day_open': round(day_open, 3),
            'exit_date': df.iloc[exit_idx]['trade_date'],
            'exit_price': round(exit_price_val, 3),
            'exit_reason': exit_reason,
            'hold_days': exit_idx - entry_idx,
            'limit_high': sig['limit_high'],
            'limit_low': sig['limit_low'],
            'boll_lower': round(boll_lower, 3) if boll_lower is not None else None,
            'signal_low': round(signal_low, 3),
            'boll_lower_clamped': round(boll_lower_clamped, 3),
            'ret': round(ret * 100, 2),
            'ret_after_cost': round(ret_after_cost * 100, 2),
        })

    trades_df = pd.DataFrame(trades)
    print(f"   总交易: {len(trades_df):,} 笔")
    print()

    # ========== 4. 分析 ==========
    print("[4/5] 绩效分析...")

    if len(trades_df) == 0:
        print("   ⚠️ 无交易")
        return

    n_trades = len(trades_df)
    win_mask = trades_df['ret_after_cost'] > 0
    n_wins = win_mask.sum()
    win_rate = n_wins / n_trades * 100
    avg_ret = trades_df['ret_after_cost'].mean()
    avg_win = trades_df.loc[win_mask, 'ret_after_cost'].mean()
    avg_loss = trades_df.loc[~win_mask, 'ret_after_cost'].mean()
    total_ret = trades_df['ret_after_cost'].sum()

    monthly_rets = trades_df.groupby(trades_df['entry_date'].dt.to_period('M'))['ret_after_cost'].sum()
    sharpe = monthly_rets.mean() / monthly_rets.std() * math.sqrt(12) if len(monthly_rets) > 1 and monthly_rets.std() > 0 else 0

    trades_df = trades_df.sort_values('exit_date').reset_index(drop=True)
    nav = (1 + trades_df['ret_after_cost'] / 100).cumprod()
    peak = nav.expanding().max()
    drawdown = (nav - peak) / peak
    max_dd = drawdown.min()

    trades_df['year'] = trades_df['entry_date'].dt.year
    yearly = trades_df.groupby('year').agg(
        交易次数=('ret_after_cost', 'count'),
        胜率=('ret_after_cost', lambda x: (x > 0).mean() * 100),
        平均收益=('ret_after_cost', 'mean'),
        总收益=('ret_after_cost', 'sum'),
    ).round(2)

    reason_stats = trades_df.groupby('exit_reason').agg(
        次数=('ret_after_cost', 'count'),
        胜率=('ret_after_cost', lambda x: (x > 0).mean() * 100),
        平均收益=('ret_after_cost', 'mean'),
        平均持有天数=('hold_days', 'mean'),
    ).round(2)

    trades_df['hold_bucket'] = pd.cut(trades_df['hold_days'], bins=[0, 1, 3, 5, 10, 20, 100],
                                       labels=['1天', '2-3天', '4-5天', '6-10天', '11-20天', '21天+'])
    hold_stats = trades_df.groupby('hold_bucket', observed=True).agg(
        次数=('ret_after_cost', 'count'),
        胜率=('ret_after_cost', lambda x: (x > 0).mean() * 100),
        平均收益=('ret_after_cost', 'mean'),
    ).round(2)

    # BOLL低吸 vs 开盘买入对比
    trades_df['boll_lower_used'] = trades_df['entry_price'] < trades_df['day_open'] - 0.001
    boll_used = trades_df['boll_lower_used'].sum()
    boll_stats = trades_df.groupby('boll_lower_used').agg(
        次数=('ret_after_cost', 'count'),
        胜率=('ret_after_cost', lambda x: (x > 0).mean() * 100),
        平均收益=('ret_after_cost', 'mean'),
    ).round(2)
    
    # BOLL折扣统计
    trades_df['boll_discount_pct'] = (trades_df['entry_price'] / trades_df['day_open'] - 1) * 100

    # ========== 5. 输出 ==========
    print("[5/5] 输出结果...\n")

    print("=" * 70)
    print("📊 回测结果 — 涨停缩量回调→放量突破 v3")
    print("=" * 70)
    print()
    print(f"  {'总交易数':<22} {n_trades:>8,}")
    print(f"  {'胜率':<22} {win_rate:>7.2f}%")
    print(f"  {'平均单笔收益(扣费后)':<22} {avg_ret:>+8.2f}%")
    print(f"  {'平均盈利':<22} {avg_win:>+8.2f}%")
    print(f"  {'平均亏损':<22} {avg_loss:>+8.2f}%")
    print(f"  {'累计总收益(扣费后)':<22} {total_ret:>+8.2f}%")
    print(f"  {'月频夏普比率':<22} {sharpe:>8.2f}")
    print(f"  {'最大回撤':<22} {max_dd*100:>7.2f}%")
    print()

    print("─" * 70)
    print("📊 BOLL低吸使用情况")
    print("─" * 70)
    print(f"  BOLL下轨低吸次数: {boll_used}/{n_trades} ({100*boll_used/n_trades:.1f}%)")
    if boll_used > 0:
        boll_discounts = trades_df[trades_df['boll_lower_used']]['boll_discount_pct']
        print(f"  BOLL平均折扣: {boll_discounts.mean():.1f}%")
        print(f"  BOLL折扣中位数: {boll_discounts.median():.1f}%")
        print(f"  BOLL折扣范围: {boll_discounts.min():.1f}% ~ {boll_discounts.max():.1f}%")
    print(f"  {'买入方式':<12} {'次数':>8} {'胜率':>8} {'平均收益':>10}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*10}")
    for used, row in boll_stats.iterrows():
        label = 'BOLL低吸' if used else '开盘买入'
        print(f"  {label:<12} {row['次数']:>8} {row['胜率']:>7.1f}% {row['平均收益']:>+9.2f}%")

    print()
    print("─" * 70)
    print("📅 各年统计")
    print("─" * 70)
    print(f"  {'年份':>6} {'次数':>8} {'胜率':>8} {'平均收益':>10} {'总收益':>10}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
    for year, row in yearly.iterrows():
        print(f"  {int(year):>6} {row['交易次数']:>8} {row['胜率']:>7.1f}% {row['平均收益']:>+9.2f}% {row['总收益']:>+9.2f}%")
    print()

    print("─" * 70)
    print("🏁 退出原因分析")
    print("─" * 70)
    print(f"  {'原因':<15} {'次数':>8} {'胜率':>8} {'平均收益':>10} {'平均持有天':>10}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
    for reason, row in reason_stats.iterrows():
        label = {'stop_loss':'止损', 'take_profit':'止盈', 'timeout':'到期'}.get(reason, reason)
        print(f"  {label:<15} {row['次数']:>8} {row['胜率']:>7.1f}% {row['平均收益']:>+9.2f}% {row['平均持有天数']:>9.1f}")
    print()

    print("─" * 70)
    print("📆 持有天数分布")
    print("─" * 70)
    print(f"  {'持有天数':<12} {'次数':>8} {'胜率':>8} {'平均收益':>10}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*10}")
    for bucket, row in hold_stats.iterrows():
        print(f"  {str(bucket):<12} {row['次数']:>8} {row['胜率']:>7.1f}% {row['平均收益']:>+9.2f}%")
    print()

    print("─" * 70)
    print("🏆 最佳5笔")
    print("─" * 70)
    best = trades_df.nlargest(5, 'ret_after_cost')
    for _, row in best.iterrows():
        boll_tag = '✅' if row['boll_lower_used'] else '❌'
        print(f"  {row['ts_code']} {row.get('name',''):<8} {boll_tag} "
              f"入{row['entry_date'].date()} 出{row['exit_date'].date()} "
              f"收益{row['ret_after_cost']:+.2f}% ({row['exit_reason']})")
    print()

    print("─" * 70)
    print("💀 最差5笔")
    print("─" * 70)
    worst = trades_df.nsmallest(5, 'ret_after_cost')
    for _, row in worst.iterrows():
        boll_tag = '✅' if row['boll_lower_used'] else '❌'
        print(f"  {row['ts_code']} {row.get('name',''):<8} {boll_tag} "
              f"入{row['entry_date'].date()} 出{row['exit_date'].date()} "
              f"收益{row['ret_after_cost']:+.2f}% ({row['exit_reason']})")

    # ========== 保存 ==========
    print()
    print("─" * 70)
    print("💾 保存结果...")

    out_file = os.path.join(OUTPUT_DIR, "breakout_v3_backtest.csv")
    trades_df.to_csv(out_file, index=False)
    print(f"   ✅ {out_file}")

    summary = pd.DataFrame([{
        'start_date': START_DATE,
        'end_date': END_DATE,
        'total_trades': n_trades,
        'win_rate_pct': round(win_rate, 2),
        'avg_return_pct': round(avg_ret, 2),
        'avg_win_pct': round(avg_win if not pd.isna(avg_win) else 0, 2),
        'avg_loss_pct': round(avg_loss if not pd.isna(avg_loss) else 0, 2),
        'total_return_pct': round(total_ret, 2),
        'monthly_sharpe': round(sharpe, 2),
        'max_drawdown_pct': round(max_dd * 100, 2),
        'take_profit_pct': TAKE_PROFIT_PCT * 100,
        'boll_lower_used_pct': round(100 * boll_used / n_trades, 1) if n_trades > 0 else 0,
        'cost_per_trade': COST_PER_TRADE,
    }])
    summary_file = os.path.join(OUTPUT_DIR, "breakout_v3_summary.csv")
    summary.to_csv(summary_file, index=False)
    print(f"   ✅ {summary_file}")

    total_time = time.time() - t0
    print(f"\n⏱ 总耗时: {total_time:.0f}秒 ({total_time/60:.1f}分钟)")


if __name__ == "__main__":
    t0 = time.time()
    run_backtest()
