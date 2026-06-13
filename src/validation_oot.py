#!/usr/bin/env python3
"""
样本外验证 — 涨停突破 v6 策略
==============================
训练期: 2017-01 ~ 2021-12（固定参数）
验证期: 2022-01 ~ 2026-06（样本外）

注意：仓位管理的滚动统计在验证期内使用训练期末的先验统计，
不使用验证期的未来信息。
"""
import pandas as pd
import numpy as np
import os, warnings, math
from collections import Counter
warnings.filterwarnings('ignore')

PROJ_B = "/mnt/d/AI-20260604"
DATA_DAILY_DIR = os.path.join(PROJ_B, "data", "raw", "daily")
STOCK_LIST_FILE = os.path.join(PROJ_B, "data", "raw", "stock_list.parquet")
OUTPUT_DIR = os.path.join(PROJ_B, "backtest_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====== 策略参数（冻结，训练期确定）======
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
MAX_DAILY_TOTAL_POSITION = 0.60

# 仓位管理参数（固定）
BASE_POSITION = 0.6
MAX_POSITION = 0.30
MIN_POSITION = 0.05
MAX_LOSS_PCT_OF_EQUITY = 0.04
MAX_EQUITY_DD = -0.20
COOLDOWN_DAYS_BIG_DD = 15
CONSECUTIVE_LOSS_CUT = 0.75
CONSECUTIVE_LOSS_LOOKBACK = 3
POSITION_REBOUND_BOOST = 1.0
POSITION_REBOUND_THRESH = 0.20

BOLL_PERIOD = 20
BOLL_STD = 2.0
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0

# 训练期/验证期划分
TRAIN_START, TRAIN_END = "2017-01-01", "2021-12-31"
TEST_START, TEST_END = "2022-01-01", "2026-06-03"

INDEX_PATH = os.path.join(PROJ_B, "data", "raw", "index_000300.parquet")


def calc_atr(high, low, close, period=14):
    n = len(high); tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, n): atr[i] = (atr[i-1]*(period-1)+tr[i])/period
    return atr


def boll_lower_at_idx(close, ti, period=20, ns=2.0):
    if ti < period-1: return np.nan
    w = close[ti-period+1:ti+1]
    return np.mean(w) - ns * np.std(w, ddof=1)


def load_market_filter():
    print("   沪深300...", end=" ", flush=True)
    df = pd.read_parquet(INDEX_PATH).sort_values('trade_date').reset_index(drop=True)
    c = df['close'].values.astype(np.float64); n = len(c)
    ma20 = np.full(n, np.nan)
    if n >= 20:
        s = np.cumsum(c); ma20[19] = s[19]/20
        for i in range(20, n): ma20[i] = (s[i]-s[i-20])/20
    mu = np.full(n, False, dtype=bool); mu[1:] = ma20[1:] > ma20[:-1]
    r = {pd.Timestamp(df.iloc[i]['trade_date']).strftime('%Y-%m-%d'): bool(mu[i]) for i in range(n)}
    print(f"{len(r)}日"); return r


def detect_breakout_signals(df):
    c, h, l, v = [df[k].values.astype(np.float64) for k in ['close','high','low','vol']]
    n = len(c)
    ma20 = np.full(n, np.nan)
    if n >= 20:
        s = np.cumsum(c); ma20[19] = s[19]/20
        for i in range(20, n): ma20[i] = (s[i]-s[i-20])/20
    mu = np.full(n, False, dtype=bool); mu[1:] = ma20[1:] > ma20[:-1]
    lu = np.full(n, False, dtype=bool); lu[1:] = (c[1:]/c[:-1] > LIMIT_UP_PCT) & (c[1:] == h[1:])
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
        sigs.append({'idx': i, 'lb': lb, 'lh': lh, 'll': ll, 'lv': lv, 'bl': bl if not np.isnan(bl) else 0})
    return sigs


def calc_position_size(status, ep, max_loss_price):
    nav_pos = status['equity'] / max(status['peak'], 0.001)
    if nav_pos >= 0.95: nf = 0.80
    elif nav_pos >= 0.85: nf = 0.90
    elif nav_pos >= 0.70: nf = 1.0
    elif nav_pos >= 0.50: nf = 1.05
    else: nf = 0.7
    
    kelly_factor = 0.6
    recent = status['recent_trades']
    if len(recent) >= 10 and status['avg_loss'] < 0:
        wr = max(0.01, status['win_rate']/100)
        aw = status['avg_win']; al = abs(status['avg_loss'])
        odds = aw/al if al > 0 else 1
        kp = (wr*odds - (1-wr))/odds
        kelly_factor = max(0.4, min(1.0, kp * BASE_POSITION * 1.5))
    
    consec_losses = 0
    for r in reversed(recent[-CONSECUTIVE_LOSS_LOOKBACK:]):
        if r < 0: consec_losses += 1
        else: break
    if consec_losses >= 3: cf = CONSECUTIVE_LOSS_CUT ** (consec_losses-2)
    elif consec_losses == 2: cf = 0.90
    elif consec_losses == 1: cf = 0.95
    else: cf = 1.0
    
    rp = MAX_POSITION * nf * kelly_factor * cf
    rp = np.clip(rp, MIN_POSITION, MAX_POSITION)
    ek = ep - max_loss_price
    if ek > 0 and ep > 0:
        lfp = MAX_LOSS_PCT_OF_EQUITY / max(ek/ep, 0.001)
    else:
        lfp = MAX_POSITION
    return max(min(rp, lfp, MAX_POSITION), MIN_POSITION)


def run_one_period(period_name, start_date, end_date, priors=None):
    """运行一个区间的回测"""
    print(f"\n{'='*60}")
    print(f"📊 {period_name} ({start_date} ~ {end_date})")
    print(f"{'='*60}")
    
    market_filter = load_market_filter()
    
    print("   加载股票...", end=" ", flush=True)
    sl = pd.read_parquet(STOCK_LIST_FILE)
    codes = sorted(sl['ts_code'].unique())
    nm = dict(zip(sl['ts_code'], sl.get('name', ['']*len(sl))))
    print(f"{len(codes)}只")
    
    print("   扫描信号...")
    all_sigs = []
    for idx, code in enumerate(codes):
        if (idx+1) % 1000 == 0: print(f"     {idx+1}/{len(codes)} ({100*(idx+1)//len(codes)}%)")
        fp = os.path.join(DATA_DAILY_DIR, f"{code}.parquet")
        if not os.path.exists(fp): continue
        try: df = pd.read_parquet(fp)
        except: continue
        if len(df) < MIN_TRADE_DAYS: continue
        df = df.sort_values('trade_date').reset_index(drop=True)
        mask = (df['trade_date'] >= np.datetime64(start_date)) & (df['trade_date'] <= np.datetime64(end_date))
        if not mask.any(): continue
        si = max(0, mask.argmax() - 60)
        for s in detect_breakout_signals(df.iloc[si:].reset_index(drop=True)):
            gi = si + s['idx']
            if gi >= len(df): continue
            all_sigs.append({'code': code, 'name': nm.get(code,''),
                             'date': df.iloc[gi]['trade_date'], 'idx': gi,
                             'lb_gi': si+s['lb'], 'lh': s['lh'], 'll': s['ll'],
                             'lv': s['lv'], 'bl': s['bl']})
    print(f"    信号: {len(all_sigs):,}")
    scm = Counter()
    for s in all_sigs: scm[pd.Timestamp(s['date']).strftime('%Y-%m-%d')] += 1
    
    print("   索引...", end=" ", flush=True)
    cd = {}
    for code in codes:
        fp = os.path.join(DATA_DAILY_DIR, f"{code}.parquet")
        if os.path.exists(fp):
            try: cd[code] = pd.read_parquet(fp).sort_values('trade_date').reset_index(drop=True)
            except: pass
    print(f"{len(cd)}只")
    
    # 仓位初始状态
    if priors:
        equity = priors['equity']
        peak_equity = priors['peak']
        equity_low_point = priors['low_point']
        recent_returns = priors['recent_returns']
        win_rate = priors['win_rate']
        avg_win = priors['avg_win']
        avg_loss = priors['avg_loss']
        total_trades_count = priors['total_trades']
        print(f"    继承前段: 净值{equity:.4f}, 峰值{peak_equity:.4f}, {total_trades_count}笔经验")
    else:
        equity = 1.0; peak_equity = 1.0; equity_low_point = 1.0
        recent_returns = []; win_rate = 43.0; avg_win = 10.93; avg_loss = -1.51
        total_trades_count = 0
    
    daily_position_used = {}
    trades = []; nav_history = []; skipped = 0; cooldown_until = None
    
    for tidx, sig in enumerate(all_sigs):
        if (tidx+1) % 1000 == 0: print(f"   交易: {tidx+1}/{len(all_sigs)} ({100*(tidx+1)//len(all_sigs)}%)")
        code = sig['code']; sidx = sig['idx']; sd = pd.Timestamp(sig['date'])
        dt_key = sd.strftime('%Y-%m-%d')
        
        if dt_key in market_filter and not market_filter[dt_key]: skipped += 1; continue
        if scm.get(dt_key, 0) < MIN_MARKET_SIGNALS: skipped += 1; continue
        if cooldown_until is not None and dt_key < cooldown_until: skipped += 1; continue
        
        df = cd.get(code)
        if df is None or sidx+1 >= len(df): continue
        eidx = sidx+1; do = float(df.iloc[eidx]['open'])
        if do <= 0: continue
        
        bl = sig.get('bl', 0); slv = float(df.iloc[sidx]['low'])
        if bl is not None and not isinstance(bl, str) and bl > 0:
            ep = min(do, max(bl, slv))
        else: ep = do
        if ep <= 0: continue
        
        sf = sig['ll'] * STOP_LOSS_PCT
        alb = max(0, eidx-ATR_PERIOD-5)
        av = calc_atr(df.iloc[alb:eidx+1]['high'].values.astype(np.float64),
                      df.iloc[alb:eidx+1]['low'].values.astype(np.float64),
                      df.iloc[alb:eidx+1]['close'].values.astype(np.float64), ATR_PERIOD)
        atr_v = av[-1] if len(av)>0 and not np.isnan(av[-1]) else 0
        sa = ep - atr_v * ATR_MULTIPLIER if atr_v > 0 else 0
        sb = bl if (bl is not None and not isinstance(bl, str) and not np.isnan(bl)) else 0
        if isinstance(sb, (np.floating, float)) and np.isnan(sb): sb = 0
        sp = max(p for p in [sf, sa, sb] if p > 0)
        mlp = ep * 0.85; sp = max(sp, mlp)
        
        st = {'equity': equity, 'peak': peak_equity, 'low_point': equity_low_point,
              'recent_trades': recent_returns[-max(CONSECUTIVE_LOSS_LOOKBACK*3,20):],
              'win_rate': win_rate, 'avg_win': avg_win, 'avg_loss': avg_loss,
              'total_trades': total_trades_count}
        pos = calc_position_size(st, ep, mlp)
        
        entry_key = pd.Timestamp(df.iloc[eidx]['trade_date']).strftime('%Y-%m-%d')
        used = daily_position_used.get(entry_key, 0.0)
        if used + pos > MAX_DAILY_TOTAL_POSITION:
            avail = MAX_DAILY_TOTAL_POSITION - used
            if avail < MIN_POSITION: skipped += 1; continue
            pos = avail
        daily_position_used[entry_key] = used + pos
        
        tp = ep * (1+TAKE_PROFIT_PCT); hse = ep; exit_idx, epv, reason = None, None, None
        for la in range(1, MAX_HOLD+1):
            if eidx+la >= len(df): break
            row = df.iloc[eidx+la]; ddo, ddh, ddl = row['open'], row['high'], row['low']
            if ddh > hse:
                hse = ddh; ns = hse - atr_v * ATR_MULTIPLIER
                cb = boll_lower_at_idx(df['close'].values.astype(np.float64), eidx+la, BOLL_PERIOD, BOLL_STD)
                if not np.isnan(cb): sb = cb
                sp = max(sp, ns, sb)
            if ddh >= tp-1e-8: epv = tp if ddo < tp else ddo; exit_idx=eidx+la; reason='take_profit'; break
            if ddl <= sp-1e-8: epv = ddo if ddo <= sp-1e-8 else sp; exit_idx=eidx+la; reason='stop_loss'; break
        if exit_idx is None:
            li = min(eidx+MAX_HOLD, len(df)-1); exit_idx=li; epv=float(df.iloc[li]['close']); reason='timeout'
        
        ret = epv/ep - 1; rac = ret - COST_PER_TRADE*2
        impact = rac * pos
        equity *= (1+impact); peak_equity = max(peak_equity, equity)
        equity_low_point = min(equity_low_point, equity)
        nav_history.append({'exit_date': df.iloc[exit_idx]['trade_date'], 'impact': impact, 'pos': pos, 'equity': equity, 'peak': peak_equity})
        
        recent_returns.append(rac)
        if len(recent_returns) > 50: recent_returns.pop(0)
        total_trades_count += 1
        if len(recent_returns) >= 10:
            wr = sum(1 for r in recent_returns if r>0)/len(recent_returns)*100
            ws = [r for r in recent_returns if r>0]; ls = [r for r in recent_returns if r<0]
            win_rate = wr; avg_win = np.mean(ws)*100 if ws else 10.0; avg_loss = np.mean(ls)*100 if ls else -5.0
        
        dd = equity/peak_equity - 1
        if dd <= MAX_EQUITY_DD:
            cd_end = (pd.Timestamp(df.iloc[exit_idx]['trade_date']) + pd.Timedelta(days=COOLDOWN_DAYS_BIG_DD*1.5)).strftime('%Y-%m-%d')
            cooldown_until = cd_end
            print(f"     ⚠️ 回撤{dd*100:.1f}%触发暂停→{cd_end}")
        
        trades.append({'code':code, 'name':sig.get('name',''),
                       'entry_date':df.iloc[eidx]['trade_date'],
                       'ret_ac': round(rac*100,2), 'pos': round(pos*100,2),
                       'impact': round(impact*100,4), 'reason': reason})
    
    tdf = pd.DataFrame(trades)
    f_eq = equity
    ret_pct = (f_eq-1)*100
    nav_seq = np.array([e['equity'] for e in nav_history])
    peak_seq = np.maximum.accumulate(nav_seq) if len(nav_seq) > 0 else nav_seq
    dd_seq = nav_seq/peak_seq - 1 if len(nav_seq) > 0 else np.array([])
    max_dd = dd_seq.min() if len(dd_seq) > 0 else 0
    
    monthly = tdf.groupby(pd.to_datetime(tdf['entry_date']).dt.to_period('M'))['impact'].sum()
    sharpe = monthly.mean()/monthly.std()*math.sqrt(12) if len(monthly)>1 and monthly.std()>0 else 0
    
    wr = tdf['ret_ac'].gt(0).mean()*100 if len(tdf)>0 else 0
    
    print(f"\n结果:")
    print(f"  交易: {len(tdf):,} | 过滤: {skipped:,}")
    print(f"  胜率: {wr:.1f}%")
    print(f"  平均个股收益: {tdf['ret_ac'].mean():+.2f}%")
    print(f"  平均仓位: {tdf['pos'].mean():.2f}%")
    print(f"  净值收益: {ret_pct:.2f}%")
    print(f"  夏普(月): {sharpe:.2f}")
    print(f"  最大回撤: {max_dd*100:.2f}%")
    
    summary = pd.DataFrame([{
        'period': period_name, 'trades': len(tdf), 'win_rate': round(wr,1),
        'avg_ret': round(tdf['ret_ac'].mean(),2), 'avg_pos': round(tdf['pos'].mean(),2),
        'total_return': round(ret_pct,2), 'sharpe': round(sharpe,2),
        'max_dd': round(max_dd*100,2), 'final_equity': round(f_eq,4),
    }])
    
    state = {
        'equity': equity, 'peak': peak_equity, 'low_point': equity_low_point,
        'recent_returns': recent_returns, 'win_rate': win_rate,
        'avg_win': avg_win, 'avg_loss': avg_loss, 'total_trades': total_trades_count,
    }
    
    return tdf, summary, state


if __name__ == "__main__":
    print("=" * 60)
    print("样本外验证 — 涨停突破 v6")
    print("=" * 60)
    print(f"训练期: {TRAIN_START} ~ {TRAIN_END}")
    print(f"验证期: {TEST_START} ~ {TEST_END}")
    print()
    
    # 训练期
    train_tdf, train_summary, state = run_one_period("训练期", TRAIN_START, TRAIN_END)
    train_summary.to_csv(os.path.join(OUTPUT_DIR, "validation_train_summary.csv"), index=False)
    train_tdf.to_csv(os.path.join(OUTPUT_DIR, "validation_train_trades.csv"), index=False)
    
    # 验证期—继承训练期的仓位状态
    test_tdf, test_summary, _ = run_one_period("验证期(样本外)", TEST_START, TEST_END, priors=state)
    test_summary.to_csv(os.path.join(OUTPUT_DIR, "validation_test_summary.csv"), index=False)
    test_tdf.to_csv(os.path.join(OUTPUT_DIR, "validation_test_trades.csv"), index=False)
    
    # 合并
    final = pd.concat([train_summary, test_summary], ignore_index=True)
    final.to_csv(os.path.join(OUTPUT_DIR, "validation_summary.csv"), index=False)
    
    print("\n" + "=" * 60)
    print("📋 样本外验证总结")
    print("=" * 60)
    print()
    for _, row in final.iterrows():
        print(f"  {row['period']}:")
        print(f"    交易: {row['trades']:,} | 胜率: {row['win_rate']:.1f}%")
        print(f"    平均仓位: {row['avg_pos']:.2f}% | 净值收益: {row['total_return']:.2f}%")
        print(f"    夏普: {row['sharpe']:.2f} | 最大回撤: {row['max_dd']:.2f}%")
        print()
    
    # 验证期vs训练期
    if len(test_summary) > 0:
        t_sharpe = test_summary.iloc[0]['sharpe']
        tr_sharpe = train_summary.iloc[0]['sharpe']
        t_dd = test_summary.iloc[0]['max_dd']
        tr_dd = train_summary.iloc[0]['max_dd']
        t_ret = test_summary.iloc[0]['total_return']
        
        print(f"  {'进步':>10} {'夏普':>8} {'回撤':>8} {'收益':>10}")
        print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*10}")
        print(f"  {'训练期':>10} {tr_sharpe:>8.2f} {tr_dd:>7.2f}% {tr_ret:>+9.2f}%")
        print(f"  {'验证期':>10} {t_sharpe:>8.2f} {t_dd:>7.2f}% {t_ret:>+9.2f}%")
        
        if t_sharpe >= tr_sharpe * 0.7:
            print(f"\n  ✅ 验证通过! 样本外夏普为训练期的{t_sharpe/tr_sharpe*100:.0f}%")
        else:
            print(f"\n  ⚠️ 验证期表现明显弱于训练期，存在一定过拟合")
