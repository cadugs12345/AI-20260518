#!/usr/bin/env python3
"""
v9 ABCD 样本内/样本外回测
样本内: 2017-01-01 ~ 2023-12-31
样本外: 2024-01-01 ~ 2026-06-04
"""
import pandas as pd, numpy as np, os, sys, math, time
from collections import Counter

PROJ_B = "/mnt/d/AI-20260604"
sys.path.insert(0, os.path.join(PROJ_B, "src"))
DATA_DAILY_DIR = os.path.join(PROJ_B, "data", "raw", "daily")
STOCK_LIST_FILE = os.path.join(PROJ_B, "data", "raw", "stock_list.parquet")
INDEX_PATH = os.path.join(PROJ_B, "data", "raw", "index_000300.parquet")
OUTPUT_DIR = os.path.join(PROJ_B, "backtest_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 参数
LIMIT_UP_PCT = 1.095
MAX_DAYS_SINCE_LIMIT = 25
MIN_DAYS_SINCE_LIMIT = 1
MIN_TRADE_DAYS = 180
MA_PERIOD = 18
MIN_MARKET_SIGNALS = 2
MAX_HOLD = 30
BOLL_PERIOD = 20
BOLL_STD = 2.0
COST_PER_TRADE = 0.0032
MAX_DAILY_TOTAL_POSITION = 0.60
BASE_POSITION = 0.6
MAX_POSITION = 0.30
MIN_POSITION = 0.05
MAX_LOSS_PCT_OF_EQUITY = 0.04
MAX_EQUITY_DD = -0.20
COOLDOWN_DAYS_BIG_DD = 15
MA_BREAK_CANDLES = 2
CONSECUTIVE_LOSS_CUT = 0.75
CONSECUTIVE_LOSS_LOOKBACK = 3

PERIODS = {
    "样本内(2017-2023)": ("2017-01-01", "2023-12-31"),
    "样本外(2024-2026)": ("2024-01-01", "2026-06-04"),
    "全样本(2017-2026)": ("2017-01-01", "2026-06-04"),
}

def get_sector_cross_limit(industry):
    if not isinstance(industry, str): return 5.0
    ultra = ['银行','保险','石油石化']
    low = ['公用事业','交通运输','建筑','汽车','房地产','有色金属','煤炭','商贸零售','家用电器','食品饮料']
    high = ['电子','计算机','通信','传媒','国防军工','综合']
    if industry in ultra: return 10.0
    if industry in low: return 8.0
    if industry in high: return 4.0
    return 5.0

def detect_signals_v9(df, industry_map, code):
    o = df['open'].values.astype(np.float64)
    c = df['close'].values.astype(np.float64)
    h = df['high'].values.astype(np.float64)
    v = df['vol'].values.astype(np.float64)
    n = len(df)
    ma = np.full(n, np.nan)
    if n >= MA_PERIOD:
        s = np.cumsum(c); ma[MA_PERIOD-1] = s[MA_PERIOD-1]/MA_PERIOD
        for i in range(MA_PERIOD, n): ma[i] = (s[i]-s[i-MA_PERIOD])/MA_PERIOD
    mu = np.full(n, False); mu[1:] = ma[1:] > ma[:-1]
    lu = np.full(n, False); lu[1:] = (c[1:]/c[:-1] > LIMIT_UP_PCT) & (c[1:] == h[1:])
    lli = np.full(n, -1, dtype=np.int32); ls = -1
    for i in range(n):
        if lu[i]: ls = i
        lli[i] = i - ls if ls >= 0 else -1
    ma5 = np.full(n, np.nan); ma10 = np.full(n, np.nan)
    if n >= 5:
        s5 = np.cumsum(c); ma5[4] = s5[4]/5
        for i in range(5, n): ma5[i] = (s5[i]-s5[i-5])/5
    if n >= 10:
        s10 = np.cumsum(c); ma10[9] = s10[9]/10
        for i in range(10, n): ma10[i] = (s10[i]-s10[i-10])/10
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
        since_limit_high = np.max(c[lb+1:i+1])
        if (since_limit_high / c[lb] - 1) * 100 > 15.0: continue
        if not np.isfinite(ma5[i]) or c[i] <= ma5[i]: continue
        if np.isfinite(ma10[i]) and ma5[i] <= ma10[i]: continue
        if not np.isfinite(ma[i]) or c[i] <= ma[i]: continue
        cross_pct = (c[i] / ma[i] - 1) * 100
        industry = industry_map.get(code, '')
        max_cross = get_sector_cross_limit(industry)
        if cross_pct > max_cross: continue
        vol_sum = 0; vol_count = 0
        for jj in range(max(0,i-5), i):
            if v[jj] > 0: vol_sum += v[jj]; vol_count += 1
        vol_ma5 = vol_sum / vol_count if vol_count >= 3 else 0
        if vol_ma5 > 0 and v[i] <= vol_ma5 * 1.2: continue
        sigs.append({'idx': i, 'lb': lb, 'ma_value': ma[i], 'cross_pct': cross_pct,
                     'date': df.iloc[i]['trade_date']})
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
    else: cf = 1.0
    max_risk_per_trade = MAX_LOSS_PCT_OF_EQUITY * kelly_factor * cf
    price_risk = (ep - max_loss_price) / ep if ep > max_loss_price else 0.03
    price_risk = max(0.02, min(0.10, price_risk))
    pos = max_risk_per_trade / price_risk
    return max(MIN_POSITION, min(MAX_POSITION, pos))

def run_backtest(start_date, end_date):
    """跑一个时间段回测"""
    print(f"\n{'='*60}")
    print(f"📊 回测区间: {start_date} ~ {end_date}")
    print(f"{'='*60}")
    
    # 加载沪深300
    print("[0/5] 加载沪深300...")
    hs = pd.read_parquet(INDEX_PATH).sort_values('trade_date').reset_index(drop=True)
    hc = hs['close'].values.astype(np.float64); hn = len(hc)
    hma60 = np.full(hn, np.nan)
    if hn >= 60:
        s = np.cumsum(hc); hma60[59] = s[59]/60
        for i in range(60, hn): hma60[i] = (s[i]-s[i-60])/60
    hmu = np.full(hn, False); hmu[1:] = hma60[1:] > hma60[:-1]
    market_filter = {}
    for i in range(hn):
        dt = pd.Timestamp(hs.iloc[i]['trade_date']).strftime('%Y-%m-%d')
        market_filter[dt] = bool(hmu[i])
    print(f"   {hn}日")
    
    # 加载股票
    print("[1/5] 加载股票...")
    sl = pd.read_parquet(STOCK_LIST_FILE)
    codes = sorted(sl['ts_code'].unique())
    names = dict(zip(sl['ts_code'], sl.get('name', ['']*len(sl))))
    industries = dict(zip(sl['ts_code'], sl.get('industry', ['']*len(sl))))
    print(f"   {len(codes)}只")
    
    # 扫描信号
    print("[2/5] 扫描信号...")
    all_sigs = []
    for idx, code in enumerate(codes):
        if (idx+1) % 1000 == 0:
            print(f"   {idx+1}/{len(codes)} ({100*(idx+1)//len(codes)}%)")
        fp = os.path.join(DATA_DAILY_DIR, f"{code}.parquet")
        if not os.path.exists(fp): continue
        try: df = pd.read_parquet(fp)
        except: continue
        if len(df) < MIN_TRADE_DAYS: continue
        df = df.sort_values('trade_date').reset_index(drop=True)
        
        # 只看时间范围内的数据
        sd = pd.Timestamp(start_date)
        ed = pd.Timestamp(end_date)
        end_idx = df[df['trade_date'] <= ed].index.max()
        start_idx = df[df['trade_date'] >= sd].index.min()
        if pd.isna(end_idx) or pd.isna(start_idx): continue
        if end_idx < MA_PERIOD: continue
        
        sigs = detect_signals_v9(df, industries, code)
        for s in sigs:
            s_dt = pd.Timestamp(s['date'])
            if s_dt < sd or s_dt > ed: continue
            if s['idx'] > end_idx: continue
            s['code'] = code
            s['name'] = names.get(code, '')
            all_sigs.append(s)
    
    print(f"   总信号: {len(all_sigs):,}")
    
    # 按日期统计信号数量
    scm = Counter()
    for s in all_sigs:
        scm[pd.Timestamp(s['date']).strftime('%Y-%m-%d')] += 1
    
    # 模拟交易
    print("[3/5] 模拟交易...")
    cd = {}
    for code in codes:
        fp = os.path.join(DATA_DAILY_DIR, f"{code}.parquet")
        if os.path.exists(fp):
            try: cd[code] = pd.read_parquet(fp).sort_values('trade_date').reset_index(drop=True)
            except: pass
    print(f"   {len(cd)}只")
    
    trades = []; skipped = 0; nav_history = []
    equity = 1.0; peak_equity = 1.0; equity_low_point = 1.0
    recent_returns = []; win_rate = 43.0; avg_win = 10.93; avg_loss = -1.51
    total_trades_count = 0; daily_position_used = {}; cooldown_until = None
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    
    for tidx, sig in enumerate(all_sigs):
        if (tidx+1) % 1000 == 0:
            print(f"   {tidx+1}/{len(all_sigs)} ({100*(tidx+1)//len(all_sigs)}%)")
        code = sig['code']; sidx = sig['idx']; sd = pd.Timestamp(sig['date'])
        dt_key = sd.strftime('%Y-%m-%d')
        
        if dt_key in market_filter and not market_filter[dt_key]:
            skipped += 1; continue
        if scm.get(dt_key, 0) < MIN_MARKET_SIGNALS:
            skipped += 1; continue
        if cooldown_until is not None and dt_key < cooldown_until:
            skipped += 1; continue
        
        df = cd.get(code)
        if df is None or sidx+1 >= len(df): continue
        eidx = sidx + 1
        
        # D条件：多日等待买入
        ema = sig['ma_value']
        ep = None; actual_entry_idx = None
        for wait in range(1, min(4, len(df)-eidx)):
            cur_idx = eidx + wait - 1
            do = float(df.iloc[cur_idx]['open'])
            elv = float(df.iloc[cur_idx]['low'])
            cur_ma = np.nan
            if cur_idx >= MA_PERIOD-1:
                w = df.iloc[cur_idx-MA_PERIOD+1:cur_idx+1]['close'].values.astype(np.float64)
                if len(w) == MA_PERIOD: cur_ma = np.mean(w)
            if not np.isfinite(cur_ma): cur_ma = ema
            max_open = cur_ma * 1.01
            if do <= max_open:
                ep = do; actual_entry_idx = cur_idx; break
            elif elv <= max_open:
                ep = elv; actual_entry_idx = cur_idx; break
        if ep is None:
            skipped += 1; continue
        
        # 检查入场日在区间内
        entry_dt = pd.Timestamp(df.iloc[actual_entry_idx]['trade_date'])
        if entry_dt > end_ts or entry_dt < start_ts:
            continue
        
        stop_price = ema
        max_loss_price = stop_price * 0.95
        st = {'equity': equity, 'peak': peak_equity, 'low_point': equity_low_point,
              'recent_trades': recent_returns[-max(CONSECUTIVE_LOSS_LOOKBACK*3,20):],
              'win_rate': win_rate, 'avg_win': avg_win, 'avg_loss': avg_loss,
              'total_trades': total_trades_count}
        pos = calc_position_size(st, ep, max_loss_price)
        entry_key = entry_dt.strftime('%Y-%m-%d')
        used = daily_position_used.get(entry_key, 0.0)
        if used + pos > MAX_DAILY_TOTAL_POSITION:
            avail = MAX_DAILY_TOTAL_POSITION - used
            if avail < MIN_POSITION: skipped += 1; continue
            pos = avail
        daily_position_used[entry_key] = used + pos
        
        exit_idx, epv, reason = None, None, None
        below_count = 0
        
        for la in range(1, MAX_HOLD+1):
            if actual_entry_idx+la >= len(df): break
            row = df.iloc[actual_entry_idx+la]
            ci = actual_entry_idx + la
            tp_boll = np.inf
            if ci >= BOLL_PERIOD-1:
                w = df.iloc[ci-BOLL_PERIOD+1:ci+1]['close'].values.astype(np.float64)
                if len(w) == BOLL_PERIOD:
                    boll_upper = np.mean(w) + BOLL_STD * np.std(w, ddof=1)
                    tp_boll = boll_upper
            current_ma = np.nan
            if ci >= MA_PERIOD-1:
                w = df.iloc[ci-MA_PERIOD+1:ci+1]['close'].values.astype(np.float64)
                if len(w) == MA_PERIOD: current_ma = np.mean(w)
            if not np.isnan(current_ma): stop_price = max(stop_price, current_ma)
            
            if float(row['high']) >= tp_boll-1e-8:
                epv = float(row['high']); exit_idx=ci; reason='take_profit'; break
            if float(row['close']) < stop_price:
                below_count += 1
                if below_count >= MA_BREAK_CANDLES:
                    epv = float(row['close']); exit_idx=ci; reason='stop_loss'; break
            else:
                below_count = 0
        
        if exit_idx is None:
            li = min(actual_entry_idx+MAX_HOLD, len(df)-1)
            exit_idx=li; epv=float(df.iloc[li]['close']); reason='timeout'
        
        ret = epv/ep - 1; rac = ret - COST_PER_TRADE*2
        impact = rac * pos
        equity *= (1+impact); peak_equity = max(peak_equity, equity)
        equity_low_point = min(equity_low_point, equity)
        
        recent_returns.append(rac)
        if len(recent_returns) > 50: recent_returns.pop(0)
        total_trades_count += 1
        if len(recent_returns) >= 10:
            wr = sum(1 for r in recent_returns if r>0)/len(recent_returns)*100
            ws = [r for r in recent_returns if r>0]; ls = [r for r in recent_returns if r<0]
            win_rate = wr; avg_win = np.mean(ws)*100 if ws else 10.0; avg_loss = np.mean(ls)*100 if ls else -5.0
        
        dd = equity/peak_equity - 1
        if dd <= MAX_EQUITY_DD:
            cd_end = (entry_dt + pd.Timedelta(days=COOLDOWN_DAYS_BIG_DD*1.5)).strftime('%Y-%m-%d')
            cooldown_until = cd_end
        
        trades.append({'code':code, 'name':sig.get('name',''),
                       'entry_date':entry_dt.strftime('%Y-%m-%d'),
                       'entry_price': round(ep,3), 'exit_price': round(epv,3),
                       'exit_reason': reason, 'hold_days': exit_idx-actual_entry_idx,
                       'ret_ac': round(rac*100,2), 'position_pct': round(pos*100,2),
                       'trade_impact_pct': round(impact*100,4)})
    
    print()
    return trades

def analyze_trades(trades, period_name):
    """分析交易结果"""
    if not trades:
        print(f"\n{period_name}: 无交易")
        return
    
    df = pd.DataFrame(trades)
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    nt = len(df)
    wr = (df['ret_ac'] > -0.5).sum() / nt * 100
    avg_ret = df['ret_ac'].mean()
    wins = df[df['ret_ac'] > 0]
    losses = df[df['ret_ac'] <= 0]
    avg_win = wins['ret_ac'].mean() if len(wins) > 0 else 0
    avg_loss = losses['ret_ac'].mean() if len(losses) > 0 else 0
    pl_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else 0
    avg_hold = df['hold_days'].mean()
    avg_pos = df['position_pct'].mean()
    
    # 夏普
    monthly = df.set_index('entry_date').resample('ME')['ret_ac'].sum()
    sharpe = monthly.mean() / monthly.std() * np.sqrt(12) if monthly.std() > 0 and len(monthly) >= 12 else 0
    rsharp = monthly.mean() / monthly.std() * np.sqrt(12) if monthly.std() > 0 and len(monthly) >= 12 else 0
    
    # 最大回撤
    equity = 1.0
    dd_list = []
    peak = 1.0
    for _, row in df.iterrows():
        equity *= (1 + row['trade_impact_pct']/100)
        peak = max(peak, equity)
        dd = (equity/peak - 1) * 100
        dd_list.append(dd)
    max_dd = min(dd_list)
    
    # 累计净值
    final_nav = equity
    
    # 年化
    if len(df) > 0:
        years = (df['entry_date'].max() - df['entry_date'].min()).days / 365.25
        if years > 0.5:
            ann_ret = (final_nav ** (1/years) - 1) * 100
        else:
            ann_ret = (final_nav - 1) * 100
    else:
        ann_ret = 0
    
    print(f"""
📊 {period_name}
{'='*50}
  总交易数:                {nt:,}
  胜率:                    {wr:.1f}%
  平均收益/笔:              {avg_ret:.2f}%
  平均盈利 / 平均亏损:      {avg_win:.2f}% / {avg_loss:.2f}%
  盈亏比:                   {pl_ratio:.2f}:1
  平均持有:                  {avg_hold:.1f}日
  平均仓位:                  {avg_pos:.2f}%
  累计净值:                  {final_nav:.4f}
  年化收益:                  {ann_ret:.2f}%
  月频夏普:                  {sharpe:.2f}
  最大回撤:                  {max_dd:.2f}%
  {'='*50}""")
    
    # 各年
    df['year'] = df['entry_date'].dt.year
    print("  📅 各年")
    for yr in sorted(df['year'].unique()):
        yt = df[df['year'] == yr]
        ywr = (yt['ret_ac'] > -0.5).sum() / len(yt) * 100
        yavg = yt['ret_ac'].mean()
        # 净值
        y_equity = 1.0
        for _, row in yt.iterrows():
            y_equity *= (1 + row['trade_impact_pct']/100)
        yret = (y_equity - 1) * 100
        print(f"    {yr:8} {len(yt):5}笔 胜率{ywr:.1f}% 平均{yavg:.2f}% 净值{yret:+.2f}%")
    
    return {
        'trades': nt,
        'win_rate': wr,
        'avg_ret': avg_ret,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'pl_ratio': pl_ratio,
        'avg_hold': avg_hold,
        'avg_pos': avg_pos,
        'final_nav': final_nav,
        'ann_ret': ann_ret,
        'sharpe': sharpe,
        'max_dd': max_dd,
    }

def main():
    results = {}
    for pname, (sd, ed) in PERIODS.items():
        t0 = time.time()
        trades = run_backtest(sd, ed)
        results[pname] = analyze_trades(trades, pname)
        save_path = os.path.join(OUTPUT_DIR, f"v9_{sd[:4]}-{ed[:4]}_trades.csv")
        if trades:
            pd.DataFrame(trades).to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"   保存: {save_path}")
        
        # 更新END_DATE到实际数据最新
        elapsed = time.time() - t0
        print(f"   耗时: {elapsed:.0f}s")
    
    # 汇总
    print(f"\n{'='*60}")
    print("📊 汇总对比")
    print(f"{'='*60}")
    print(f"{'指标':<20} {'样本内(2017-23)':<18} {'样本外(2024-26)':<18} {'全样本':<18}")
    print("-" * 74)
    for key in ['trades', 'sharpe', 'win_rate', 'pl_ratio', 'ann_ret', 'max_dd', 'final_nav']:
        labels = {'trades':'交易数','sharpe':'月频夏普','win_rate':'胜率%','pl_ratio':'盈亏比','ann_ret':'年化%','max_dd':'最大回撤%','final_nav':'累计净值'}
        vals = []
        for p in ['样本内(2017-2023)', '样本外(2024-2026)', '全样本(2017-2026)']:
            r = results.get(p, {})
            v = r.get(key, '-')
            vals.append(f"{v:.2f}" if isinstance(v, float) else str(v))
        print(f"{labels.get(key, key):<20} {vals[0]:<18} {vals[1]:<18} {vals[2]:<18}")

if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n✅ 总耗时: {time.time()-t0:.0f}s")
