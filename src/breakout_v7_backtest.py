#!/usr/bin/env python3
"""
涨停回调 → BOLL下轨低吸 v7（去掉放量突破条件）
==============================================
核心逻辑变更:
  v6: 涨停→回调缩量→放量突破涨停最高价→BOLL下轨买入
  v7: 涨停→回调→BOLL下轨附近低吸买入（去掉放量突破条件）
  
选股逻辑:
  1. 最近20天内有涨停，涨停当天MA20向上
  2. 涨停后3~20天
  3. 当前价格在BOLL(20,2)下轨附近（收盘价 ≤ BOLL中轨）
  4. 买入: BOLL下轨低吸（次日开盘价接近BOLL下轨时买入）

出场: 止损止盈同v6
"""
import pandas as pd
import numpy as np
import os, sys, time, warnings, math
from collections import Counter
warnings.filterwarnings('ignore')

PROJ_B = "/mnt/d/AI-20260604"
DATA_DAILY_DIR = os.path.join(PROJ_B, "data", "raw", "daily")
STOCK_LIST_FILE = os.path.join(PROJ_B, "data", "raw", "stock_list.parquet")
INDEX_PATH = os.path.join(PROJ_B, "data", "raw", "index_000300.parquet")
OUTPUT_DIR = os.path.join(PROJ_B, "backtest_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====== 策略参数 ======
LIMIT_UP_PCT = 1.095
MAX_DAYS_SINCE_LIMIT = 20
MIN_DAYS_SINCE_LIMIT = 3
MIN_TRADE_DAYS = 120
COST_PER_TRADE = 0.0032
TAKE_PROFIT_PCT = 0.20
STOP_LOSS_PCT = 0.95
MAX_HOLD = 30
MIN_MARKET_SIGNALS = 3
MAX_DAILY_TOTAL_POSITION = 0.60
BOLL_PERIOD = 20
BOLL_STD = 2.0

# BOLL买入条件: 收盘价在BOLL中轨以下的才考虑
BOLL_NEAR_LOWER = 0.03  # 收盘价在BOLL下轨的3%以内

# 仓位管理
BASE_POSITION = 0.6
MAX_POSITION = 0.30
MIN_POSITION = 0.05
MAX_LOSS_PCT_OF_EQUITY = 0.04
MAX_EQUITY_DD = -0.20
COOLDOWN_DAYS_BIG_DD = 15
CONSECUTIVE_LOSS_CUT = 0.75
CONSECUTIVE_LOSS_LOOKBACK = 3

ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0

START_DATE = "2017-01-01"
END_DATE = "2026-06-03"


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


def boll_band(close, period=20, ns=2.0):
    """返回 (中轨, 下轨)"""
    n = len(close)
    if n < period: return np.nan, np.nan
    w = close[-period:]
    ma = np.mean(w)
    std = np.std(w, ddof=1)
    return ma, ma - ns * std


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


def detect_signals_v7(df):
    """v7选股: 涨停→回调→BOLL下轨附近"""
    c, h, l, v = [df[k].values.astype(np.float64) for k in ['close','high','low','vol']]
    n = len(c)
    
    # MA20
    ma20 = np.full(n, np.nan)
    if n >= 20:
        s = np.cumsum(c); ma20[19] = s[19]/20
        for i in range(20, n): ma20[i] = (s[i]-s[i-20])/20
    mu = np.full(n, False, dtype=bool); mu[1:] = ma20[1:] > ma20[:-1]
    
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
        if not mu[lb]: continue  # 涨停日MA20向上
        
        # BOLL计算
        if i < BOLL_PERIOD - 1: continue
        mid, lower = boll_band(c[:i+1], BOLL_PERIOD, BOLL_STD)
        if np.isnan(lower) or np.isnan(mid): continue
        
        # 核心条件: 收盘价在BOLL下轨附近（接近下轨）
        if c[i] > lower * (1 + BOLL_NEAR_LOWER):
            continue
        # 同时价格不能跌破下轨太多
        if c[i] < lower * 0.95:
            continue
        
        # 缩量条件: 量比涨停日缩量
        lv = v[lb]
        if lv <= 0: continue
        if v[i] > lv * 0.8:  # 量能不超过涨停日的80%
            continue
        
        # 回调确认: 从涨停后的高点回落
        # 涨停后最高价至少比当前高3%（确认是在回调不是横盘）
        post_limit_high = np.max(c[lb+1:i+1])
        if post_limit_high < c[i] * 1.03:
            continue
        
        # 计算买入参考价: BOLL下轨低吸
        entry_ref = max(lower, l[i], c[i] * 0.98)  # 下轨附近，不低于当日最低
        
        sigs.append({
            'idx': i, 'lb': lb,
            'lh': h[lb], 'll': l[lb],
            'bl': lower, 'bm': mid,
            'entry_ref': entry_ref,
        })
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
    else: lfp = MAX_POSITION
    return max(min(rp, lfp, MAX_POSITION), MIN_POSITION)


def run_backtest():
    print("=" * 70)
    print("📊 涨停回调→BOLL下轨低吸 v7")
    print("=" * 70)
    print(f"   去掉放量突破条件 | 涨停后3~{MAX_DAYS_SINCE_LIMIT}天")
    print(f"   条件: 收盘价在BOLL中轨以下 + MA20向上")
    print(f"   买入: BOLL({BOLL_PERIOD},{BOLL_STD})下轨低吸")
    print(f"   止损: ATR×{ATR_MULTIPLIER}+BOLL+固定 | 止盈: +{TAKE_PROFIT_PCT*100:.0f}%")
    print(f"   每日总仓位≤{MAX_DAILY_TOTAL_POSITION*100:.0f}% | 单笔≤{MAX_LOSS_PCT_OF_EQUITY*100:.0f}%总资")
    print(f"   成本: {COST_PER_TRADE*100:.2f}%/边")
    print()

    # ====== 0. 大盘 ======
    print("[0/5] 加载沪深300..."); market_filter = load_market_filter(); print()

    # ====== 1. 加载股票 ======
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
        for s in detect_signals_v7(df.iloc[si:].reset_index(drop=True)):
            gi = si + s['idx']
            if gi >= len(df): continue
            all_sigs.append({'code': code, 'name': nm.get(code, ''),
                             'date': df.iloc[gi]['trade_date'], 'idx': gi,
                             'lb_gi': si + s['lb'], 'lh': s['lh'], 'll': s['ll'],
                             'bl': s['bl'], 'bm': s['bm'], 'entry_ref': s['entry_ref']})

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
    print(f"{len(cd)}只")

    trades = []; skipped = 0; nav_history = []
    equity = 1.0; peak_equity = 1.0; equity_low_point = 1.0
    recent_returns = []; win_rate = 43.0; avg_win = 10.93; avg_loss = -1.51
    total_trades_count = 0; daily_position_used = {}; cooldown_until = None

    for tidx, sig in enumerate(all_sigs):
        if (tidx+1) % 1000 == 0: print(f"   {tidx+1}/{len(all_sigs)} ({100*(tidx+1)//len(all_sigs)}%)")
        code = sig['code']; sidx = sig['idx']; sd = pd.Timestamp(sig['date'])
        dt_key = sd.strftime('%Y-%m-%d')
        
        if dt_key in market_filter and not market_filter[dt_key]: skipped += 1; continue
        if scm.get(dt_key, 0) < MIN_MARKET_SIGNALS: skipped += 1; continue
        if cooldown_until is not None and dt_key < cooldown_until: skipped += 1; continue
        
        df = cd.get(code)
        if df is None or sidx+1 >= len(df): continue
        eidx = sidx+1; do = float(df.iloc[eidx]['open'])
        elv = float(df.iloc[eidx]['low'])
        if do <= 0: continue
        
        # 买入价: BOLL下轨低吸，不低于当日最低价
        bl = sig['bl']; entry_ref = sig['entry_ref']
        ep = max(min(do, entry_ref), elv)
        if ep <= 0: continue
        
        # 止损
        sf = sig['ll'] * STOP_LOSS_PCT
        alb = max(0, eidx-ATR_PERIOD-5)
        av = calc_atr(df.iloc[alb:eidx+1]['high'].values.astype(np.float64),
                      df.iloc[alb:eidx+1]['low'].values.astype(np.float64),
                      df.iloc[alb:eidx+1]['close'].values.astype(np.float64), ATR_PERIOD)
        atr_v = av[-1] if len(av)>0 and not np.isnan(av[-1]) else 0
        sa = ep - atr_v * ATR_MULTIPLIER if atr_v > 0 else 0
        sb = bl if (bl is not None and not np.isnan(bl)) else 0
        if isinstance(sb, (np.floating, float)) and np.isnan(sb): sb = 0
        sp = max(p for p in [sf, sa, sb] if p > 0)
        mlp = ep * 0.85; sp = max(sp, mlp)
        
        # 仓位
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
        
        # 出场
        tp = ep * (1+TAKE_PROFIT_PCT); hse = ep; exit_idx, epv, reason = None, None, None
        for la in range(1, MAX_HOLD+1):
            if eidx+la >= len(df): break
            row = df.iloc[eidx+la]; ddo, ddh, ddl = row['open'], row['high'], row['low']
            if ddh > hse:
                hse = ddh; ns = hse - atr_v * ATR_MULTIPLIER
                cb = boll_band(df['close'].values.astype(np.float64)[:eidx+la+1], BOLL_PERIOD, BOLL_STD)[1]
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
            print(f"     ⚠️ 回撤{dd*100:.1f}%暂停→{cd_end}")
        
        trades.append({'code':code, 'name':sig.get('name',''), 'entry_date':df.iloc[eidx]['trade_date'],
                       'entry_price': round(ep,3), 'exit_price': round(epv,3),
                       'exit_reason': reason, 'hold_days': exit_idx-eidx,
                       'ret_ac': round(rac*100,2), 'position_pct': round(pos*100,2),
                       'trade_impact_pct': round(impact*100,4)})

    tdf = pd.DataFrame(trades)
    f_eq = equity; ret_pct = (f_eq-1)*100
    nav_seq = np.array([e['equity'] for e in nav_history]) if nav_history else np.array([])
    peak_seq = np.maximum.accumulate(nav_seq) if len(nav_seq) > 0 else nav_seq
    dd_seq = nav_seq/peak_seq - 1 if len(nav_seq) > 0 else np.array([])
    max_dd = dd_seq.min() if len(dd_seq) > 0 else 0
    
    tdf['entry_date'] = pd.to_datetime(tdf['entry_date'])
    monthly = tdf.groupby(tdf['entry_date'].dt.to_period('M'))['trade_impact_pct'].sum()/100
    sharpe = monthly.mean()/monthly.std()*math.sqrt(12) if len(monthly)>1 and monthly.std()>0 else 0
    
    wr = tdf['ret_ac'].gt(0).mean()*100 if len(tdf)>0 else 0
    avg_r = tdf['ret_ac'].mean() if len(tdf)>0 else 0
    avg_w = tdf[tdf['ret_ac']>0]['ret_ac'].mean() if len(tdf)>0 and tdf['ret_ac'].gt(0).any() else 0
    avg_l = tdf[tdf['ret_ac']<=0]['ret_ac'].mean() if len(tdf)>0 and tdf['ret_ac'].le(0).any() else 0
    
    print()
    print("=" * 70)
    print("📊 涨停回调→BOLL下轨低吸 v7")
    print("=" * 70)
    print(f"  {'总交易数':<22} {len(tdf):>8,}")
    print(f"  {'胜率':<22} {wr:>7.2f}%")
    print(f"  {'平均收益':<22} {avg_r:>+8.2f}%")
    print(f"  {'平均盈利/亏损':<22} {avg_w:>+7.2f}% / {avg_l:>+6.2f}%")
    print(f"  {'平均仓位':<22} {tdf['position_pct'].mean():>7.2f}%")
    print(f"  {'总净值收益':<22} {ret_pct:>+8.2f}%")
    print(f"  {'月频夏普':<22} {sharpe:>8.2f}")
    print(f"  {'最大回撤':<22} {max_dd*100:>7.2f}%")
    print(f"  {'过滤跳过':<22} {skipped:>8,}")
    
    tdf['year'] = tdf['entry_date'].dt.year
    yearly = tdf.groupby('year').agg(次数=('ret_ac','count'), 胜率=('ret_ac',lambda x: (x>0).mean()*100),
                                       平均=('ret_ac','mean'), 总收益=('trade_impact_pct','sum')).round(2)
    print("\n📅 各年")
    for yr, row in yearly.iterrows():
        print(f"  {int(yr):>6} {row['次数']:>8} {row['胜率']:>7.1f}% {row['平均']:>+9.2f}% {row['总收益']:>+9.2f}%")
    
    rs = tdf.groupby('exit_reason').agg(次数=('ret_ac','count'), 胜率=('ret_ac',lambda x: (x>0).mean()*100),
                                          平均=('ret_ac','mean'), 仓位=('position_pct','mean')).round(2)
    print("\n🏁 退出原因")
    for reason, row in rs.iterrows():
        lbl = {'stop_loss':'止损','take_profit':'止盈','timeout':'到期'}.get(reason, reason)
        print(f"  {lbl:<12} {row['次数']:>8} {row['胜率']:>7.1f}% {row['平均']:>+9.2f}% 仓位{row['仓位']:.1f}%")
    
    tdf.to_csv(os.path.join(OUTPUT_DIR, "breakout_v7_backtest.csv"), index=False)
    pd.DataFrame(nav_history).to_csv(os.path.join(OUTPUT_DIR, "breakout_v7_nav.csv"), index=False)
    pd.DataFrame([{'trades':len(tdf), 'win_rate':round(wr,2), 'avg_ret':round(avg_r,2),
                    'total_return':round(ret_pct,2), 'sharpe':round(sharpe,2),
                    'max_dd':round(max_dd*100,2)}]).to_csv(os.path.join(OUTPUT_DIR, "breakout_v7_summary.csv"), index=False)
    print(f"\n✅ done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    t0 = time.time()
    run_backtest()
