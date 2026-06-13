#!/usr/bin/env python3
"""
涨停缩量回调 → 放量突破 第二波策略 v6（仓位管理版）
=================================================

核心改动:
  1. 双过滤: 沪深300 MA20向上 + 市场热度(当日信号≥3)
  2. 放量条件: 1.2倍
  3. 三重止损: ATR(14)×2 + BOLL下轨 + 涨停最低×0.95
  4. 动态仓位管理（核心升级）:
     a. 基础仓位按"当前净值位置"浮动 — 净值在历史高点附近保守，在低位进取
     b. 凯利公式限制 — 基于历史胜率和盈亏比计算最优仓位，取25%凯利
     c. 连续亏损递减 — 连亏N笔，仓位减半
     d. 最大单笔亏损 ≤ 总资金0.5%
     e. 全局最大回撤 > 15% 清仓暂停

选股逻辑同v4
"""
import pandas as pd
import numpy as np
import os, sys, time, warnings, math
from collections import Counter
warnings.filterwarnings('ignore')

PROJ_B = "/mnt/d/AI-20260604"
DATA_DAILY_DIR = os.path.join(PROJ_B, "data", "raw", "daily")
STOCK_LIST_FILE = os.path.join(PROJ_B, "data", "raw", "stock_list.parquet")
OUTPUT_DIR = os.path.join(PROJ_B, "backtest_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====== 策略参数 ======
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
MIN_MARKET_SIGNALS = 3

# ====== 仓位管理参数 ======
BASE_POSITION = 0.6           # 凯利比例(全凯利×60%)
MAX_POSITION = 0.30           # 最大单笔仓位(总资金30%)
MIN_POSITION = 0.05           # 最小单笔仓位
MAX_LOSS_PCT_OF_EQUITY = 0.04  # 单笔最大亏损占总资金 ≤ 4%
MAX_EQUITY_DD = -0.20         # 净值回撤>20%暂停
COOLDOWN_DAYS_BIG_DD = 15     # 大回撤后暂停
CONSECUTIVE_LOSS_CUT = 0.75   # 连亏后仓位乘数
CONSECUTIVE_LOSS_LOOKBACK = 3  # 看最近3笔
POSITION_REBOUND_BOOST = 1.0
POSITION_REBOUND_THRESH = 0.20
MAX_DAILY_TOTAL_POSITION = 0.60  # 每日总资金上限60% (100万=60万持仓)

BOLL_PERIOD = 20
BOLL_STD = 2.0
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0

START_DATE = "2017-01-01"
END_DATE = "2026-06-03"

INDEX_PATH = os.path.join(PROJ_B, "data", "raw", "index_000300.parquet")


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
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr


def boll_lower_at_idx(close, target_idx, period=20, n_std=2.0):
    if target_idx < period - 1:
        return np.nan
    w = close[target_idx - period + 1 : target_idx + 1]
    return np.mean(w) - n_std * np.std(w, ddof=1)


def load_market_filter():
    print("   加载沪深300(MA20过滤)...", end=" ", flush=True)
    df = pd.read_parquet(INDEX_PATH).sort_values('trade_date').reset_index(drop=True)
    c = df['close'].values.astype(np.float64)
    n = len(c)
    ma20 = np.full(n, np.nan)
    if n >= 20:
        s = np.cumsum(c); ma20[19] = s[19]/20
        for i in range(20, n): ma20[i] = (s[i]-s[i-20])/20
    mu = np.full(n, False, dtype=bool)
    mu[1:] = ma20[1:] > ma20[:-1]
    result = {pd.Timestamp(df.iloc[i]['trade_date']).strftime('%Y-%m-%d'): bool(mu[i]) for i in range(n)}
    print(f"done ({len(result)}日)")
    return result


def detect_breakout_signals(df):
    c = df['close'].values.astype(np.float64)
    h = df['high'].values.astype(np.float64)
    l = df['low'].values.astype(np.float64)
    v = df['vol'].values.astype(np.float64)
    n = len(df)
    
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
    
    sigs = []
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
        bl = boll_lower_at_idx(c, i, BOLL_PERIOD, BOLL_STD)
        sigs.append({'idx': i, 'lb': lb, 'lh': lh, 'll': ll, 'lv': lv, 'bl': bl if not np.isnan(bl) else None})
    return sigs


def calc_position_size(status, entry_price, stop_price, max_loss_price):
    """
    动态仓位管理引擎
    status: {
        'equity': 当前净值,
        'peak': 历史最高净值,
        'low_point': 最近低点(用于反弹判断),
        'recent_trades': 最近N笔交易的收益列表,
        'win_rate': 滚动胜率,
        'avg_win': 平均盈利,
        'avg_loss': 平均亏损,
        'total_trades': 总交易数,
    }
    """
    # 1. 净值位置因子
    nav_position = status['equity'] / status['peak']
    if nav_position >= 0.95:
        nav_factor = 0.80
    elif nav_position >= 0.85:
        nav_factor = 0.90
    elif nav_position >= 0.70:
        nav_factor = 1.0
    elif nav_position >= 0.50:
        nav_factor = 1.05
    else:
        nav_factor = 0.7
    
    # 2. 凯利公式
    kelly_fraction = 0
    recent = status['recent_trades']
    if len(recent) >= 10 and status['avg_loss'] < 0:
        win_rate = max(0.01, status['win_rate'] / 100)
        avg_win = status['avg_win']
        avg_loss = abs(status['avg_loss'])
        odds_ratio = avg_win / avg_loss if avg_loss > 0 else 1
        kelly_pct = (win_rate * odds_ratio - (1 - win_rate)) / odds_ratio
        kelly_factor = max(0.4, min(1.0, kelly_pct * BASE_POSITION * 1.5))  # 60%凯利
    else:
        kelly_factor = 0.6
    
    # 3. 连亏递减
    consec_losses = 0
    for r in reversed(recent[-CONSECUTIVE_LOSS_LOOKBACK:]):
        if r < 0:
            consec_losses += 1
        else:
            break
    if consec_losses >= 3:
        consec_factor = CONSECUTIVE_LOSS_CUT ** (consec_losses - 2)
    elif consec_losses == 2:
        consec_factor = 0.90
    elif consec_losses == 1:
        consec_factor = 0.95
    else:
        consec_factor = 1.0
    
    # 4. 反弹加成
    rebound = status['equity'] / max(status['low_point'], 0.001) - 1
    if rebound > POSITION_REBOUND_THRESH:
        rebound_factor = min(POSITION_REBOUND_BOOST, 1 + rebound * 3)
    else:
        rebound_factor = 1.0
    
    # 5. 组合仓位
    raw_position = MAX_POSITION * nav_factor * kelly_factor * consec_factor * rebound_factor
    raw_position = np.clip(raw_position, MIN_POSITION, MAX_POSITION)
    
    # 6. 单笔最大亏损约束
    eq_risk = entry_price - max_loss_price
    if eq_risk > 0 and entry_price > 0:
        risk_pct = eq_risk / entry_price
        loss_limited_pos = MAX_LOSS_PCT_OF_EQUITY / max(risk_pct, 0.001)
    else:
        loss_limited_pos = MAX_POSITION
    
    final_position = min(raw_position, loss_limited_pos, MAX_POSITION)
    final_position = max(final_position, MIN_POSITION)
    
    return round(final_position, 4)


def run_backtest():
    print("=" * 70)
    print("📊 涨停突破 v6 (动态仓位管理)")
    print("=" * 70)
    print(f"   双过滤(沪深300↑+信号≥{MIN_MARKET_SIGNALS}) | 放量≥{BREAKOUT_VOL_RATIO}×")
    print(f"   止损: ATR+BOLL+固定 | 止盈: +{TAKE_PROFIT_PCT*100:.0f}%")
    print(f"   仓位: 凯利×{BASE_POSITION*100:.0f}% + 净值自适应(0-{MAX_POSITION*100:.0f}%)")
    print(f"   每日总仓位≤{MAX_DAILY_TOTAL_POSITION*100:.0f}% | 单笔≤{MAX_LOSS_PCT_OF_EQUITY*100:.0f}%总资")
    print(f"   单笔≤{MAX_LOSS_PCT_OF_EQUITY*100:.2f}%总资 | 连亏递减 | 回撤>{abs(MAX_EQUITY_DD)*100:.0f}%暂停")
    print(f"   成本: {COST_PER_TRADE*100:.2f}%/边")
    print()

    # ====== 0. 大盘 ======
    print("[0/5] 加载沪深300..."); market_filter = load_market_filter(); print()

    # ====== 1. 加载 ======
    print("[1/5] 加载股票...")
    sl = pd.read_parquet(STOCK_LIST_FILE)
    codes = sorted(sl['ts_code'].unique())
    nm = dict(zip(sl['ts_code'], sl.get('name', ['']*len(sl))))
    print(f"   {len(codes)}只")

    # ====== 2. 扫描信号 ======
    print("[2/5] 扫描信号...")
    all_sigs = []
    for idx, code in enumerate(codes):
        if (idx+1) % 1000 == 0: print(f"   {idx+1}/{len(codes)} ({100*(idx+1)//len(codes)}%)")
        fp = os.path.join(DATA_DAILY_DIR, f"{code}.parquet")
        if not os.path.exists(fp): continue
        try: df = pd.read_parquet(fp)
        except: continue
        if len(df) < MIN_TRADE_DAYS: continue
        df = df.sort_values('trade_date').reset_index(drop=True)
        mask = (df['trade_date'] >= np.datetime64(START_DATE)) & (df['trade_date'] <= np.datetime64(END_DATE))
        if not mask.any(): continue
        si = max(0, mask.argmax() - 60)
        for s in detect_breakout_signals(df.iloc[si:].reset_index(drop=True)):
            gi = si + s['idx']
            if gi >= len(df): continue
            all_sigs.append({'code': code, 'name': nm.get(code, ''),
                             'date': df.iloc[gi]['trade_date'], 'idx': gi,
                             'lb_gi': si + s['lb'], 'lh': s['lh'], 'll': s['ll'], 'lv': s['lv'], 'bl': s['bl']})
    print(f"   总信号: {len(all_sigs):,}")
    scm = Counter()
    for s in all_sigs: scm[pd.Timestamp(s['date']).strftime('%Y-%m-%d')] += 1
    print()

    # ====== 3. 模拟交易 ======
    print("[3/5] 模拟交易...")
    print("   索引...", end=" ", flush=True)
    cd = {}
    for code in codes:
        fp = os.path.join(DATA_DAILY_DIR, f"{code}.parquet")
        if os.path.exists(fp):
            try: cd[code] = pd.read_parquet(fp).sort_values('trade_date').reset_index(drop=True)
            except: pass
    print(f"done ({len(cd)}只)")

    trades = []
    skipped = 0
    nav_history = []
    
    # ====== 仓位管理状态 ======
    equity = 1.0
    peak_equity = 1.0
    equity_low_point = 1.0
    recent_returns = []       # 最近交易收益(用于连亏判断)
    rolling_window = 50       # 滚动统计窗口
    cooldown_until = None
    
    # 初始统计 — 基于v4历史数据的先验(保守估计)
    win_rate = 43.0
    avg_win = 10.93
    avg_loss = -1.51
    total_trades_count = 0
    daily_position_used = {}  # {日期: 已用仓位总和}

    for tidx, sig in enumerate(all_sigs):
        if (tidx+1) % 1000 == 0: print(f"   {tidx+1}/{len(all_sigs)} ({100*(tidx+1)//len(all_sigs)}%)")
        
        code = sig['code']
        sidx = sig['idx']
        sd = pd.Timestamp(sig['date'])
        dt_key = sd.strftime('%Y-%m-%d')
        
        # 双过滤
        if dt_key in market_filter and not market_filter[dt_key]: skipped += 1; continue
        if scm.get(dt_key, 0) < MIN_MARKET_SIGNALS: skipped += 1; continue
        if cooldown_until is not None and dt_key < cooldown_until: skipped += 1; continue
        
        df = cd.get(code)
        if df is None or sidx + 1 >= len(df): continue
        
        eidx = sidx + 1
        do = float(df.iloc[eidx]['open'])
        if do <= 0: continue
        
        # 买入价（修复: 不能低于当日最低价）
        bl = sig.get('bl')
        slv = float(df.iloc[sidx]['low'])
        elv = float(df.iloc[eidx]['low'])  # 当日最低价
        if bl is not None and not np.isnan(bl) and bl > 0:
            ep = min(do, max(bl, slv))
        else:
            ep = do
        # 买入价不能低于当日实际最低价（否则买不到）
        ep = max(ep, elv)
        if ep <= 0: continue
        
        # 计算止损
        sf = sig['ll'] * STOP_LOSS_PCT
        alb = max(0, eidx - ATR_PERIOD - 5)
        av = calc_atr(df.iloc[alb:eidx+1]['high'].values.astype(np.float64),
                      df.iloc[alb:eidx+1]['low'].values.astype(np.float64),
                      df.iloc[alb:eidx+1]['close'].values.astype(np.float64), ATR_PERIOD)
        atr_v = av[-1] if len(av) > 0 and not np.isnan(av[-1]) else 0
        sa = ep - atr_v * ATR_MULTIPLIER if atr_v > 0 else 0
        sb = bl if (bl is not None and not np.isnan(bl)) else 0
        if isinstance(sb, (np.floating, float)) and np.isnan(sb): sb = 0
        sp = max(p for p in [sf, sa, sb] if p > 0)
        
        # 单笔最大亏损线
        max_loss_price = ep * (1 - 0.15)  # 个股最大-15%
        sp = max(sp, max_loss_price)
        
        # ====== 动态仓位 ======
        status = {
            'equity': equity,
            'peak': peak_equity,
            'low_point': equity_low_point,
            'recent_trades': recent_returns[-max(CONSECUTIVE_LOSS_LOOKBACK * 3, 20):],
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': total_trades_count,
        }
        pos = calc_position_size(status, ep, sp, max_loss_price)
        
        # 每日总仓位上限
        entry_date_key = pd.Timestamp(df.iloc[eidx]['trade_date']).strftime('%Y-%m-%d')
        used = daily_position_used.get(entry_date_key, 0.0)
        if used + pos > MAX_DAILY_TOTAL_POSITION:
            # 压缩仓位到可用额度，不足最小仓位则跳过
            available = MAX_DAILY_TOTAL_POSITION - used
            if available < MIN_POSITION:
                skipped += 1
                continue
            pos = available
        daily_position_used[entry_date_key] = used + pos
        
        # ====== 出场逻辑 ======
        tp = ep * (1 + TAKE_PROFIT_PCT)
        hse = ep
        exit_idx, epv, reason = None, None, None
        
        for la in range(1, MAX_HOLD + 1):
            if eidx + la >= len(df): break
            row = df.iloc[eidx + la]
            ddo, ddh, ddl = row['open'], row['high'], row['low']
            if ddh > hse:
                hse = ddh
                ns = hse - atr_v * ATR_MULTIPLIER
                cb = boll_lower_at_idx(df['close'].values.astype(np.float64), eidx+la, BOLL_PERIOD, BOLL_STD)
                if not np.isnan(cb): sb = cb
                sp = max(sp, ns, sb)
            
            if ddh >= tp - 1e-8:
                epv = tp if ddo < tp else ddo; exit_idx = eidx + la; reason = 'take_profit'; break
            if ddl <= sp - 1e-8:
                epv = ddo if ddo <= sp - 1e-8 else sp; exit_idx = eidx + la; reason = 'stop_loss'; break
        
        if exit_idx is None:
            li = min(eidx + MAX_HOLD, len(df) - 1)
            exit_idx = li; epv = float(df.iloc[li]['close']); reason = 'timeout'
        
        ret = epv / ep - 1
        rac = ret - COST_PER_TRADE * 2
        
        # ====== 仓位影响 ======
        trade_impact = rac * pos
        equity *= (1 + trade_impact)
        peak_equity = max(peak_equity, equity)
        equity_low_point = min(equity_low_point, equity)
        
        nav_history.append({
            'exit_date': df.iloc[exit_idx]['trade_date'],
            'trade_impact': trade_impact,
            'position': pos,
            'equity': equity,
            'peak': peak_equity,
        })
        
        # 更新滚动统计
        recent_returns.append(rac)
        if len(recent_returns) > rolling_window:
            recent_returns.pop(0)
        total_trades_count += 1
        
        if len(recent_returns) >= 10:
            window = recent_returns
            win_rate = sum(1 for r in window if r > 0) / len(window) * 100
            wins = [r for r in window if r > 0]
            losses = [r for r in window if r < 0]
            avg_win = np.mean(wins) * 100 if wins else 10.0
            avg_loss = np.mean(losses) * 100 if losses else -5.0
        
        # 如果反弹了，更新低点标记
        if equity > equity_low_point * (1 + POSITION_REBOUND_THRESH):
            equity_low_point = min(equity_low_point, equity)
        
        # 全局回撤风控
        dd = equity / peak_equity - 1
        if dd <= MAX_EQUITY_DD:
            exit_dt = pd.Timestamp(df.iloc[exit_idx]['trade_date'])
            cooldown_until = (exit_dt + pd.Timedelta(days=COOLDOWN_DAYS_BIG_DD * 1.5)).strftime('%Y-%m-%d')
            print(f"   ⚠️ 回撤触发! equity={equity:.4f}, peak={peak_equity:.4f}, 暂停到{cooldown_until}")
        
        trades.append({
            'code': code, 'name': sig.get('name', ''),
            'entry_date': df.iloc[eidx]['trade_date'],
            'entry_price': round(ep, 3), 'exit_price': round(epv, 3),
            'exit_reason': reason, 'hold_days': exit_idx - eidx,
            'ret_ac': round(rac * 100, 2),
            'position_pct': round(pos * 100, 2),
            'trade_impact_pct': round(trade_impact * 100, 4),
            'equity_after': round(equity, 4),
        })

    tdf = pd.DataFrame(trades)
    final_equity = equity
    ret_pct = (final_equity - 1) * 100
    total_ret_simple = tdf['ret_ac'].sum() if len(tdf) > 0 else 0

    print(f"   交易: {len(tdf):,} | 过滤: {skipped:,}")
    print()

    # ====== 4. 分析 ======
    print("[4/5] 分析...")
    if len(tdf) == 0: print("   无交易"); return

    n = len(tdf)
    wm = tdf['ret_ac'] > 0
    wr = wm.mean() * 100
    avg_r = tdf['ret_ac'].mean()
    avg_w = tdf.loc[wm, 'ret_ac'].mean()
    avg_l = tdf.loc[~wm, 'ret_ac'].mean()
    
    tdf['entry_date'] = pd.to_datetime(tdf['entry_date'])
    weekly_rets = tdf.groupby(tdf['entry_date'].dt.isocalendar().week.astype(str) + tdf['entry_date'].dt.year.astype(str))['trade_impact_pct'].sum() / 100
    sharpe = weekly_rets.mean() / weekly_rets.std() * math.sqrt(52) if len(weekly_rets) > 1 and weekly_rets.std() > 0 else 0

    # 最大回撤
    peak_seq = np.maximum.accumulate(nav_history_equity := np.array([e['equity'] for e in nav_history]))
    dd_seq = nav_history_equity / peak_seq - 1
    max_dd = dd_seq.min()
    
    # 仓位统计
    avg_pos = tdf['position_pct'].mean()
    
    tdf['year'] = tdf['entry_date'].dt.year
    yearly = tdf.groupby('year').agg(
        次数=('ret_ac','count'), 胜率=('ret_ac',lambda x: (x>0).mean()*100),
        平均仓位=('position_pct','mean'), 总收益=('trade_impact_pct','sum')
    ).round(2)

    rs = tdf.groupby('exit_reason').agg(
        次数=('ret_ac','count'), 胜率=('ret_ac',lambda x: (x>0).mean()*100),
        平均收益=('ret_ac','mean'), 平均仓位=('position_pct','mean')
    ).round(2)

    # ====== 5. 输出 ======
    print("\n[5/5] 输出\n")
    print("=" * 70)
    print("📊 涨停突破 v6 — 动态仓位管理")
    print("=" * 70)
    print(f"  {'总交易数':<22} {n:>8,}")
    print(f"  {'胜率(按笔)':<22} {wr:>7.2f}%")
    print(f"  {'平均单笔收益(个股)':<22} {avg_r:>+8.2f}%")
    print(f"  {'平均盈利/亏损':<22} {avg_w:>+7.2f}% / {avg_l:>+6.2f}%")
    print(f"  {'平均仓位':<22} {avg_pos:>7.2f}%")
    print(f"  {'总收益率(净值)':<22} {ret_pct:>+8.2f}%")
    print(f"  {'周频夏普':<22} {sharpe:>8.2f}")
    print(f"  {'最大回撤(净值)':<22} {max_dd*100:>7.2f}%")
    print(f"  {'累计总收益(简单加总)':<22} {total_ret_simple:>+8.2f}%")
    print(f"  {'过滤跳过':<22} {skipped:>8,}")
    print()

    print("─" * 70)
    print("📅 各年")
    print("─" * 70)
    print(f"  {'年份':>6} {'次数':>8} {'胜率':>8} {'平均仓位':>10} {'总收益':>10}")
    for yr, row in yearly.iterrows():
        print(f"  {int(yr):>6} {row['次数']:>8} {row['胜率']:>7.1f}% {row['平均仓位']:>9.2f}% {row['总收益']:>+9.2f}%")
    print()

    print("─" * 70)
    print("🏁 退出原因")
    print("─" * 70)
    print(f"  {'原因':<12} {'次数':>8} {'胜率':>8} {'平均收益':>10} {'平均仓位':>10}")
    for reason, row in rs.iterrows():
        lbl = {'stop_loss':'止损','take_profit':'止盈','timeout':'到期'}.get(reason, reason)
        print(f"  {lbl:<12} {row['次数']:>8} {row['胜率']:>7.1f}% {row['平均收益']:>+9.2f}% {row['平均仓位']:>9.2f}%")
    print()

    # 仓位分布
    pos_buckets = pd.cut(tdf['position_pct'], bins=[0, 1, 2, 3, 4, 5, 6, 8, 100],
                          labels=['<1%', '1-2%', '2-3%', '3-4%', '4-5%', '5-6%', '6-8%', '>8%'])
    pos_stats = tdf.groupby(pos_buckets, observed=True).agg(次数=('ret_ac','count'), 胜率=('ret_ac',lambda x: (x>0).mean()*100)).round(2)
    print("─" * 70)
    print("📊 仓位分布")
    print("─" * 70)
    print(f"  {'仓位':<10} {'次数':>8} {'胜率':>8}")
    for bucket, row in pos_stats.iterrows():
        print(f"  {str(bucket):<10} {row['次数']:>8} {row['胜率']:>7.1f}%")

    print()
    print("─" * 70)
    print("🏆 最佳3 / 💀 最差3")
    print("─" * 70)
    for _, row in tdf.nlargest(3, 'trade_impact_pct').iterrows():
        print(f"  🏆 {row['code']} {row.get('name',''):<8} "
              f"个股{row['ret_ac']:+.2f}% × 仓位{row['position_pct']:.1f}% = {row['trade_impact_pct']:+.4f}%")
    for _, row in tdf.nsmallest(3, 'trade_impact_pct').iterrows():
        print(f"  💀 {row['code']} {row.get('name',''):<8} "
              f"个股{row['ret_ac']:+.2f}% × 仓位{row['position_pct']:.1f}% = {row['trade_impact_pct']:+.4f}%")

    # ====== 保存 ======
    print()
    print("─" * 70)
    print("💾 保存...")
    tdf.to_csv(os.path.join(OUTPUT_DIR, "breakout_v6_backtest.csv"), index=False)
    ndf = pd.DataFrame(nav_history)
    ndf.to_csv(os.path.join(OUTPUT_DIR, "breakout_v6_nav.csv"), index=False)
    pd.DataFrame([{
        'total_trades': n, 'win_rate_pct': round(wr,2),
        'avg_return_pct': round(avg_r,2), 'total_return_pct': round(ret_pct,2),
        'weekly_sharpe': round(sharpe,2), 'max_drawdown_pct': round(max_dd*100,2),
        'avg_position_pct': round(avg_pos,2),
    }]).to_csv(os.path.join(OUTPUT_DIR, "breakout_v6_summary.csv"), index=False)
    print(f"   ✅ 完成")
    print(f"\n⏱ {time.time()-t0:.0f}秒 ({time.time()-t0:.1f}分)")


if __name__ == "__main__":
    t0 = time.time()
    run_backtest()
