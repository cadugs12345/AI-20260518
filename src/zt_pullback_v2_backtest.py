#!/usr/bin/env python3
"""
涨停回踩不破 升级版 — 10年回测

交易规则:
  1. 信号日当天满足策略条件 → 次日开盘买入
  2. 止损: 跌破涨停开盘价时卖出（盘中破即卖）
  3. 止盈: 涨幅达到10%时卖出
  4. 持有上限: 20个交易日（超过自动平仓）
  5. 每只股票独立管理，不设总持仓限制

回测范围: 2017-01-01 ~ 2026-06-03
"""
import pandas as pd
import numpy as np
import os, sys, time, warnings, math
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

PROJ_B = "/mnt/d/AI-20260604"
DATA_DAILY_DIR = os.path.join(PROJ_B, "data", "raw", "daily")
STOCK_LIST_FILE = os.path.join(PROJ_B, "data", "raw", "stock_list.parquet")
OUTPUT_DIR = os.path.join(PROJ_B, "backtest_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====== 参数 ======
LIMIT_UP_PCT = 1.095       # 涨停阈值
MIN_DAYS_SINCE_LIMIT = 5  # 涨停后至少5天再考虑（让筹码换手）
MAX_DAYS_SINCE_LIMIT = 20  # 距上次涨停不超过20日
MIN_TRADE_DAYS = 120       # 最低上市交易日
COST_PER_TRADE = 0.0032    # 单边交易成本 (印花税0.1%+佣金0.02%+滑点0.2%)
STOP_LOSS_PCT = 0.97      # 止损价 = 涨停开盘价 × 0.97（给3%缓冲）
TAKE_PROFIT_PCT = 0.20     # 止盈涨幅

START_DATE = "2017-01-01"
END_DATE = "2026-06-03"

# ====== 辅助函数 ======
def detect_signals(df):
    """
    对一只股票检测所有满足条件的信号日
    返回 signal_dates: list of (signal_idx in df) 当天即满足条件
    """
    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    open_ = df['open'].values.astype(np.float64)
    n = len(df)

    # MA20
    ma20 = np.full(n, np.nan)
    if n >= 20:
        cumsum = np.cumsum(close)
        ma20[19] = cumsum[19] / 20
        for i in range(20, n):
            ma20[i] = (cumsum[i] - cumsum[i-20]) / 20

    # MA20向上
    ma20_up = np.full(n, False, dtype=bool)
    ma20_up[1:] = ma20[1:] > ma20[:-1]

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

        # 涨停那根K线必须是 收盘=最高（涨停），且是当天买入能确认的
        # 条件已由 limit_up[limit_bar] 保证

        limit_open = open_[limit_bar]

        # 全程未破: 涨停后每天收盘 >= 涨停开盘
        all_above = True
        for j in range(limit_bar + 1, i + 1):
            if close[j] < limit_open - 1e-8:
                all_above = False
                break
        if not all_above:
            continue

        # 当天盘中破开盘但收盘收回
        if not (low[i] < limit_open - 1e-8 and close[i] > limit_open - 1e-8):
            continue

        # MA20向上
        if not ma20_up[i]:
            continue

        signals.append(i)

    return signals


def run_backtest():
    print("=" * 70)
    print("📊 涨停回踩不破 升级版 — 10年回测")
    print("=" * 70)
    print(f"   区间: {START_DATE} ~ {END_DATE}")
    print(f"   止损: 跌破涨停开盘价-3%")
    print(f"   止盈: +{TAKE_PROFIT_PCT*100:.0f}%")
    print(f"   交易成本: {COST_PER_TRADE*100:.2f}%/边")
    print()

    # ========== 1. 加载股票列表 ==========
    print("[1/5] 加载股票列表...")
    stock_list = pd.read_parquet(STOCK_LIST_FILE)
    ts_codes = sorted(stock_list['ts_code'].unique())
    print(f"   共 {len(ts_codes)} 只股票")
    
    # 提取行业信息
    industry_map = dict(zip(stock_list['ts_code'], stock_list.get('industry', [''] * len(stock_list))))
    name_map = dict(zip(stock_list['ts_code'], stock_list.get('name', [''] * len(stock_list))))

    # ========== 2. 逐股扫描信号 ==========
    print("[2/5] 逐股扫描信号...")
    
    all_signals = []  # list of (ts_code, signal_idx, signal_date)

    for idx, ts_code in enumerate(ts_codes):
        if (idx + 1) % 1000 == 0:
            print(f"   进度: {idx+1}/{len(ts_codes)} ({100*(idx+1)/len(ts_codes):.0f}%)")

        fpath = os.path.join(DATA_DAILY_DIR, f"{ts_code}.parquet")
        if not os.path.exists(fpath):
            continue

        try:
            df = pd.read_parquet(fpath)
        except Exception:
            continue

        if len(df) < MIN_TRADE_DAYS:
            continue

        # 过滤日期范围
        df = df.sort_values('trade_date').reset_index(drop=True)
        mask = (df['trade_date'] >= np.datetime64(START_DATE)) & (df['trade_date'] <= np.datetime64(END_DATE))
        if not mask.any():
            continue
        
        # 截取从START_DATE往前至少120天以计算MA20等指标
        start_idx = max(0, mask.argmax() - 120)
        df_window = df.iloc[start_idx:].reset_index(drop=True)
        
        sig_idxs = detect_signals(df_window)
        for si in sig_idxs:
            # 全局索引
            global_idx = start_idx + si
            if global_idx < len(df):
                sig_date = df.iloc[global_idx]['trade_date']
                limit_bar = None
                # 找到涨停日
                for k in range(global_idx - 1, max(0, global_idx-21), -1):
                    c = df.iloc[k]['close']
                    pc = df.iloc[k-1]['close'] if k > 0 else c
                    if c / pc > LIMIT_UP_PCT and c == df.iloc[k]['high']:
                        limit_bar = k
                        break
                
                if limit_bar is not None:
                    all_signals.append({
                        'ts_code': ts_code,
                        'name': name_map.get(ts_code, ''),
                        'industry': industry_map.get(ts_code, ''),
                        'signal_date': sig_date,
                        'signal_idx': global_idx,
                        'limit_open': df.iloc[limit_bar]['open'],
                        'entry_date': None,  # 后续填充
                        'limit_bar_date': df.iloc[limit_bar]['trade_date'],
                    })

    print(f"   总信号数: {len(all_signals):,}")
    print()

    # ========== 3. 模拟交易 ==========
    print("[3/5] 模拟交易...")
    
    # 构建快速访问: ts_code -> df
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

    trades = []  # 每笔交易记录
    
    for tidx, sig in enumerate(all_signals):
        if (tidx + 1) % 1000 == 0:
            print(f"   交易进度: {tidx+1}/{len(all_signals)} ({100*(tidx+1)/len(all_signals):.1f}%)")
        
        ts_code = sig['ts_code']
        signal_idx = sig['signal_idx']
        limit_open = sig['limit_open']
        stop_price = limit_open * STOP_LOSS_PCT  # 止损线 = 涨停开盘价下3%
        
        df = code_dfs.get(ts_code)
        if df is None:
            continue
        
        if signal_idx + 1 >= len(df):
            # 没有后续交易日，无法买入
            continue
        
        # 买入: 信号日次日开盘
        entry_idx = signal_idx + 1
        entry_row = df.iloc[entry_idx]
        entry_date = entry_row['trade_date']
        entry_price = entry_row['open']
        sig['entry_date'] = entry_date
        
        if entry_price <= 0 or np.isnan(entry_price):
            continue
        
        # 寻找卖出日
        exit_idx = None
        exit_price = None
        exit_reason = None
        max_hold = 20  # 最长持有20个交易日
        
        for lookahead in range(1, max_hold + 1):
            if entry_idx + lookahead >= len(df):
                break
            
            row = df.iloc[entry_idx + lookahead]
            day_open = row['open']
            day_high = row['high']
            day_low = row['low']
            day_close = row['close']
            
            # 止损检查: 最低价跌破止损线（涨停开盘价 × 0.97）
            if day_low <= stop_price - 1e-8:
                # 以开盘价或止损价成交
                if day_open <= stop_price - 1e-8:
                    exit_price = day_open
                else:
                    # 盘中跌破，以止损价成交
                    exit_price = stop_price
                exit_idx = entry_idx + lookahead
                exit_reason = 'stop_loss'
                break
            
            # 止盈检查: 最高价达到10%
            target_price = entry_price * (1 + TAKE_PROFIT_PCT)
            if day_high >= target_price - 1e-8:
                if day_open >= target_price - 1e-8:
                    exit_price = day_open
                else:
                    exit_price = target_price
                exit_idx = entry_idx + lookahead
                exit_reason = 'take_profit'
                break
        
        # 如果没有触发，到期以收盘价卖出
        if exit_idx is None:
            # 找到最后一个可用的交易日
            last_idx = min(entry_idx + max_hold, len(df) - 1)
            exit_idx = last_idx
            exit_price = df.iloc[last_idx]['close']
            exit_reason = 'timeout'
        
        exit_date = df.iloc[exit_idx]['trade_date']
        hold_days = exit_idx - entry_idx
        
        # 收益计算
        ret = exit_price / entry_price - 1
        ret_after_cost = (exit_price / entry_price - 1) - COST_PER_TRADE * 2  # 双边成本
        
        # 交易净值
        trades.append({
            'ts_code': ts_code,
            'name': sig.get('name', ''),
            'industry': sig.get('industry', ''),
            'signal_date': sig['signal_date'],
            'entry_date': entry_date,
            'entry_price': round(entry_price, 3),
            'exit_date': exit_date,
            'exit_price': round(exit_price, 3),
            'exit_reason': exit_reason,
            'hold_days': hold_days,
            'limit_open': limit_open,
            'ret': round(ret * 100, 2),
            'ret_after_cost': round(ret_after_cost * 100, 2),
        })

    trades_df = pd.DataFrame(trades)
    print(f"   总交易: {len(trades_df):,} 笔")
    print()

    # ========== 4. 绩效分析 ==========
    print("[4/5] 绩效分析...")
    
    if len(trades_df) == 0:
        print("   ⚠️ 无交易，跳过分析")
        return
    
    # 基础统计
    n_trades = len(trades_df)
    win_mask = trades_df['ret_after_cost'] > 0
    n_wins = win_mask.sum()
    win_rate = n_wins / n_trades * 100
    avg_ret = trades_df['ret_after_cost'].mean()
    avg_win = trades_df.loc[win_mask, 'ret_after_cost'].mean()
    avg_loss = trades_df.loc[~win_mask, 'ret_after_cost'].mean()
    total_ret = trades_df['ret_after_cost'].sum()
    
    # 夏普比率 (假设月频)
    monthly_rets = trades_df.groupby(trades_df['entry_date'].dt.to_period('M'))['ret_after_cost'].sum()
    sharpe = monthly_rets.mean() / monthly_rets.std() * math.sqrt(12) if len(monthly_rets) > 1 and monthly_rets.std() > 0 else 0
    
    # 最大回撤 (按净值曲线)
    trades_df = trades_df.sort_values('exit_date').reset_index(drop=True)
    nav = (1 + trades_df['ret_after_cost'] / 100).cumprod()
    peak = nav.expanding().max()
    drawdown = (nav - peak) / peak
    max_dd = drawdown.min()
    
    # 各年统计
    trades_df['year'] = trades_df['entry_date'].dt.year
    yearly = trades_df.groupby('year').agg(
        交易次数=('ret_after_cost', 'count'),
        胜率=('ret_after_cost', lambda x: (x > 0).mean() * 100),
        平均收益=('ret_after_cost', 'mean'),
        总收益=('ret_after_cost', 'sum'),
    ).round(2)
    
    # 按退出原因统计
    reason_stats = trades_df.groupby('exit_reason').agg(
        次数=('ret_after_cost', 'count'),
        胜率=('ret_after_cost', lambda x: (x > 0).mean() * 100),
        平均收益=('ret_after_cost', 'mean'),
        平均持有天数=('hold_days', 'mean'),
    ).round(2)
    
    # 按持有天数分布
    trades_df['hold_bucket'] = pd.cut(trades_df['hold_days'], bins=[0, 1, 3, 5, 10, 20], 
                                       labels=['1天', '2-3天', '4-5天', '6-10天', '11-20天'])
    hold_stats = trades_df.groupby('hold_bucket', observed=True).agg(
        次数=('ret_after_cost', 'count'),
        胜率=('ret_after_cost', lambda x: (x > 0).mean() * 100),
        平均收益=('ret_after_cost', 'mean'),
    ).round(2)
    
    # 按月分布
    monthly_stats = trades_df.groupby(trades_df['entry_date'].dt.month).agg(
        次数=('ret_after_cost', 'count'),
        胜率=('ret_after_cost', lambda x: (x > 0).mean() * 100),
        平均收益=('ret_after_cost', 'mean'),
    ).round(2)
    monthly_stats.index = ['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月']

    # ========== 5. 输出结果 ==========
    print("[5/5] 输出结果...")
    
    print()
    print("=" * 70)
    print("📊 回测结果")
    print("=" * 70)
    print()
    print(f"  {'总交易数':<20} {n_trades:>8,}")
    print(f"  {'胜率':<20} {win_rate:>7.2f}%")
    print(f"  {'平均单笔收益(扣费后)':<20} {avg_ret:>+8.2f}%")
    print(f"  {'平均盈利':<20} {avg_win:>+8.2f}%")
    print(f"  {'平均亏损':<20} {avg_loss:>+8.2f}%")
    print(f"  {'累计总收益(扣费后)':<20} {total_ret:>+8.2f}%")
    print(f"  {'月频夏普比率':<20} {sharpe:>8.2f}")
    print(f"  {'最大回撤':<20} {max_dd*100:>7.2f}%")
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
        reason_label = {'stop_loss':'止损', 'take_profit':'止盈', 'timeout':'到期'}.get(reason, reason)
        print(f"  {reason_label:<15} {row['次数']:>8} {row['胜率']:>7.1f}% {row['平均收益']:>+9.2f}% {row['平均持有天数']:>9.1f}")
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
    print("📆 月份效应")
    print("─" * 70)
    print(f"  {'月份':<8} {'次数':>8} {'胜率':>8} {'平均收益':>10}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for month, row in monthly_stats.iterrows():
        print(f"  {month:<8} {row['次数']:>8} {row['胜率']:>7.1f}% {row['平均收益']:>+9.2f}%")
    print()
    
    # 最佳/最差交易
    print("─" * 70)
    print("🏆 最佳5笔交易")
    print("─" * 70)
    best = trades_df.nlargest(5, 'ret_after_cost')
    for _, row in best.iterrows():
        print(f"  {row['ts_code']} {row.get('name',''):<8} "
              f"入{row['entry_date'].date()} 出{row['exit_date'].date()} "
              f"收益{row['ret_after_cost']:+.2f}% "
              f"({row['exit_reason']})")
    
    print()
    print("─" * 70)
    print("💀 最差5笔交易")
    print("─" * 70)
    worst = trades_df.nsmallest(5, 'ret_after_cost')
    for _, row in worst.iterrows():
        print(f"  {row['ts_code']} {row.get('name',''):<8} "
              f"入{row['entry_date'].date()} 出{row['exit_date'].date()} "
              f"收益{row['ret_after_cost']:+.2f}% "
              f"({row['exit_reason']})")

    # ========== 保存结果 ==========
    print()
    print("─" * 70)
    print("💾 保存结果...")
    
    # 主结果CSV
    out_file = os.path.join(OUTPUT_DIR, "zt_pullback_v2_backtest.csv")
    trades_df.to_csv(out_file, index=False)
    print(f"   ✅ {out_file}")
    
    # 绩效摘要
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
        'stop_loss': 'limit_open_97pct',
        'cost_per_trade': COST_PER_TRADE,
    }])
    summary_file = os.path.join(OUTPUT_DIR, "zt_pullback_v2_summary.csv")
    summary.to_csv(summary_file, index=False)
    print(f"   ✅ {summary_file}")
    
    # 按年统计
    yearly_file = os.path.join(OUTPUT_DIR, "zt_pullback_v2_yearly.csv")
    yearly.to_csv(yearly_file)
    print(f"   ✅ {yearly_file}")
    
    total_time = time.time() - t0
    print(f"\n⏱ 总耗时: {total_time:.0f}秒 ({total_time/60:.1f}分钟)")


if __name__ == "__main__":
    t0 = time.time()
    run_backtest()
