#!/usr/bin/env python3
"""
涨停→18日均线回踩再上穿 v9
=================================
核心逻辑:
  1. 最近25天内有涨停
  2. 涨停后≥5天
  3. 整个过程18日均线始终向上
  4. 股价跌破18日均线后，再次上穿18日均线 → 买入信号
  5. 买入: 上穿日次日开盘

止损: 连续2日收盘跌破18日均线
止盈: BOLL(20,2)上轨
最长持有: 30日
"""
import pandas as pd
import numpy as np
import os, sys, time, warnings, math
from collections import Counter, defaultdict
warnings.filterwarnings('ignore')

PROJ_B = "/mnt/d/AI-20260604"
DATA_DAILY_DIR = os.path.join(PROJ_B, "data", "raw", "daily")
STOCK_LIST_FILE = os.path.join(PROJ_B, "data", "raw", "stock_list.parquet")
INDEX_PATH = os.path.join(PROJ_B, "data", "raw", "index_000300.parquet")
OUTPUT_DIR = os.path.join(PROJ_B, "backtest_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====== 策略参数 ======
LIMIT_UP_PCT = 1.095
MAX_DAYS_SINCE_LIMIT = 25
MIN_DAYS_SINCE_LIMIT = 1
MIN_TRADE_DAYS = 180
COST_PER_TRADE = 0.0032
TAKE_PROFIT_PCT = 0.20  # 备用
MAX_HOLD = 30
BOLL_PERIOD = 20
BOLL_STD = 2.0
MIN_MARKET_SIGNALS = 2
MAX_DAILY_TOTAL_POSITION = 0.60

# 均线参数
MA_PERIOD = 18
MA_BREAK_CANDLES = 2  # 连续2日跌破才止损

# 仓位管理
BASE_POSITION = 0.6
MAX_POSITION = 0.30
MIN_POSITION = 0.05
MAX_LOSS_PCT_OF_EQUITY = 0.04
MAX_EQUITY_DD = -0.20
COOLDOWN_DAYS_BIG_DD = 15
CONSECUTIVE_LOSS_CUT = 0.75
CONSECUTIVE_LOSS_LOOKBACK = 3

START_DATE = "2017-01-01"
END_DATE = "2026-06-03"


def build_industry_ma_filter(all_daily_data, stock_list_df, all_trade_dates):
    """
    为每个行业计算每日等权均线方向
    返回: {industry_name: {date_str: is_ma_up}}
    """
    print(f"   行业...", end=" ", flush=True)
    
    # 股票到行业的映射
    code_to_industry = dict(zip(stock_list_df['ts_code'], stock_list_df['industry']))
    
    # 按行业分组股票代码
    industry_codes = defaultdict(set)
    for code, ind in code_to_industry.items():
        if isinstance(ind, str):
            industry_codes[ind].add(code)
    
    print(f"{len(industry_codes)}个行业, 计算18日线方向...")
    
    # 日期索引
    date_to_idx = {d: i for i, d in enumerate(all_trade_dates)}
    n_dates = len(all_trade_dates)
    
    # 找每个行业每日等权价格
    industry_result = {}
    count = 0
    for ind_name, codes in sorted(industry_codes.items()):
        count += 1
        if count % 20 == 0:
            print(f"      {count}/{len(industry_codes)}", flush=True)
        
        # 构建该行业每日均价（等权）
        ind_close = np.full(n_dates, np.nan)
        ind_count = np.zeros(n_dates, dtype=np.int32)
        
        for code in codes:
            df = all_daily_data.get(code)
            if df is None: continue
            for j in range(len(df)):
                dt = pd.Timestamp(df.iloc[j]['trade_date'])
                if dt not in df_processed:
                    dt_key = dt.strftime('%Y-%m-%d')
                    if dt_key in date_to_idx:
                        didx = date_to_idx[dt_key]
                        if np.isnan(ind_close[didx]):
                            ind_close[didx] = float(df.iloc[j]['close'])
                            ind_count[didx] += 1
                        else:
                            # 等权累加
                            total = ind_close[didx] * ind_count[didx] + float(df.iloc[j]['close'])
                            ind_count[didx] += 1
                            ind_close[didx] = total / ind_count[didx]
        
        # 计算行业MA18方向
        ma18_dir = {}
        for j in range(n_dates):
            if j < 18 or np.isnan(ind_close[j]) or np.isnan(ind_close[j-1]):
                ma18_dir[all_trade_dates[j]] = True  # 数据不足时默认允许交易
                continue
            # 计算该日MA18
            start = j - 17
            vals = ind_close[start:j+1]
            if np.any(np.isnan(vals)):
                ma18_dir[all_trade_dates[j]] = True
                continue
            # 用前一天的MA18来判断方向
            if j >= 19:
                prev_vals = ind_close[j-18:j]
                if not np.any(np.isnan(prev_vals)):
                    ma18_dir[all_trade_dates[j]] = (np.mean(vals) > np.mean(prev_vals))
                else:
                    ma18_dir[all_trade_dates[j]] = True
            else:
                ma18_dir[all_trade_dates[j]] = True
        
        industry_result[ind_name] = ma18_dir
    
    print(f"      ✓")
    return industry_result, code_to_industry


# 缓存处理状态
df_processed = set()

def load_market_filter():
    print("   沪深300...", end=" ", flush=True)
    df = pd.read_parquet(INDEX_PATH).sort_values('trade_date').reset_index(drop=True)
    c = df['close'].values.astype(np.float64); n = len(c)
    # 沪深300 60日均线方向过滤
    ma = np.full(n, np.nan)
    if n >= 60:
        s = np.cumsum(c); ma[59] = s[59]/60
        for i in range(60, n): ma[i] = (s[i]-s[i-60])/60
    mu = np.full(n, False, dtype=bool); mu[1:] = ma[1:] > ma[:-1]
    r = {}
    for i in range(n):
        dt = pd.Timestamp(df.iloc[i]['trade_date']).strftime('%Y-%m-%d')
        r[dt] = bool(mu[i])
    print(f"{len(r)}日"); return r


def detect_signals_v9(df, industry_map=None, code=None):
    """
    v9 ABCD版: 涨停→18日线始终向上→价量共振+行业自适应+大盘MACD过滤+买点优化
    A. 连续站上5日线 + 5日>10日线多头排列
    B. 行业分域上穿阈值
    C. MACD大盘过滤（外部）
    D. 买点优化（外部）
    """
    o = df['open'].values.astype(np.float64)
    c = df['close'].values.astype(np.float64)
    h = df['high'].values.astype(np.float64)
    l = df['low'].values.astype(np.float64)
    v = df['vol'].values.astype(np.float64)
    n = len(df)
    
    # 18日均线
    ma = np.full(n, np.nan)
    if n >= MA_PERIOD:
        s = np.cumsum(c); ma[MA_PERIOD-1] = s[MA_PERIOD-1]/MA_PERIOD
        for i in range(MA_PERIOD, n): ma[i] = (s[i]-s[i-MA_PERIOD])/MA_PERIOD
    
    # 均线方向
    mu = np.full(n, False, dtype=bool)
    mu[1:] = ma[1:] > ma[:-1]
    
    # 涨停
    lu = np.full(n, False, dtype=bool)
    lu[1:] = (c[1:]/c[:-1] > LIMIT_UP_PCT) & (c[1:] == h[1:])
    
    # 记录距最近涨停的天数
    lli = np.full(n, -1, dtype=np.int32); ls = -1
    for i in range(n):
        if lu[i]: ls = i
        lli[i] = i - ls if ls >= 0 else -1
    
    # 5日/10日均线
    ma5 = np.full(n, np.nan); ma10 = np.full(n, np.nan)
    if n >= 5:
        s5 = np.cumsum(c); ma5[4] = s5[4]/5
        for i in range(5, n): ma5[i] = (s5[i]-s5[i-5])/5
    if n >= 10:
        s10 = np.cumsum(c); ma10[9] = s10[9]/10
        for i in range(10, n): ma10[i] = (s10[i]-s10[i-10])/10
    
    # 行业分类（用于B. 上穿幅度阈值）
    def get_sector_cross_limit(industry):
        if not isinstance(industry, str) or industry == '':
            return 5.0
        ultra_low = ['银行','保险','石油石化']
        low_vol = ['公用事业','交通运输','建筑','汽车','房地产','有色金属','煤炭','商贸零售','家用电器','食品饮料']
        high_vol = ['电子','计算机','通信','传媒','国防军工','综合']
        if industry in ultra_low: return 10.0
        if industry in low_vol: return 8.0
        if industry in high_vol: return 4.0
        return 5.0
    
    sigs = []
    for i in range(n):
        ds = lli[i]
        if ds <= 0 or ds < MIN_DAYS_SINCE_LIMIT or ds > MAX_DAYS_SINCE_LIMIT: continue
        lb = i - ds
        if np.isnan(ma[i]) or np.isnan(ma[lb]): continue
        
        # 整个过程中18日均线必须始终向上
        if not mu[i]: continue
        ma_all_up = True
        for j in range(lb+1, i+1):
            if not mu[j]:
                ma_all_up = False
                break
        if not ma_all_up: continue
        
        # 排除涨停后已经大涨过的
        since_limit_high = np.max(c[lb+1:i+1])
        limit_close = c[lb]
        rise_since_limit = (since_limit_high / limit_close - 1) * 100
        if rise_since_limit > 15.0: continue
        
        # A. 价量共振: 站上5日线 + 5日>10日线多头
        if not np.isfinite(ma5[i]) or c[i] <= ma5[i]: continue
        if np.isfinite(ma10[i]) and ma5[i] <= ma10[i]: continue
        
        # 当前收盘站上18日线
        if not np.isfinite(ma[i]) or c[i] <= ma[i]: continue
        
        # B. 上穿幅度行业自适应
        cross_pct = (c[i] / ma[i] - 1) * 100
        industry = industry_map.get(code, '') if industry_map else ''
        max_cross = get_sector_cross_limit(industry)
        if cross_pct > max_cross: continue
        
        # 放量确认: 上穿当日量 > 前5日均量 × 1.2
        vol_sum = 0; vol_count = 0
        for jj in range(max(0,i-5), i):
            if v[jj] > 0: vol_sum += v[jj]; vol_count += 1
        vol_ma5 = vol_sum / vol_count if vol_count >= 3 else 0
        if vol_ma5 > 0 and v[i] <= vol_ma5 * 1.2:
            continue
        
        sigs.append({
            'idx': i, 'lb': lb,
            'limit_date_idx': lb,
            'ma_value': ma[i],
            'cross_pct': cross_pct,
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
    print("📊 涨停→18日线回踩再上穿 v9")
    print("=" * 70)
    print(f"   涨停后≥{MIN_DAYS_SINCE_LIMIT}天, 18日均线始终向上")
    print(f"   必须回踩18日线→再上穿为买入信号")
    print(f"   止损: 连续{MA_BREAK_CANDLES}日收盘跌破18日线 | 止盈: BOLL({BOLL_PERIOD},{BOLL_STD})上轨")
    print(f"   大盘过滤: 沪深300 60日线向下时空仓")
    print(f"   每日总仓位≤{MAX_DAILY_TOTAL_POSITION*100:.0f}% | 单笔≤{MAX_LOSS_PCT_OF_EQUITY*100:.0f}%总资")
    print(f"   成本: {COST_PER_TRADE*100:.2f}%/边")
    print()

    t0 = time.time()
    
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
        sigs_raw = detect_signals_v9(df.iloc[si:].reset_index(drop=True), {}, code)
        for s in sigs_raw:
            gi = si + s['idx']
            if gi >= len(df): continue
            all_sigs.append({
                'code': code, 'name': nm.get(code, ''),
                'date': df.iloc[gi]['trade_date'], 'idx': gi,
                'lb': si + s['lb'],
                'ma_value': s['ma_value'],
            })

    print(f"   总信号: {len(all_sigs):,}")
    scm = Counter()
    for s in all_sigs: scm[pd.Timestamp(s['date']).strftime('%Y-%m-%d')] += 1
    print()

    # ====== 3. 模拟交易 ======
    print("[3/5] 模拟交易... 索引...", end=" ", flush=True)
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
        eidx = sidx+1
        
        # D. 买点优化: 开盘价不高于18日线×1.01才买入, 否则等盘中最低价
        ema = sig['ma_value']
        ep = None; actual_entry_idx = None
        for wait in range(1, min(4, len(df)-eidx)):
            cur_idx = eidx + wait - 1
            do = float(df.iloc[cur_idx]['open'])
            elv = float(df.iloc[cur_idx]['low'])
            cur_ma = np.nan
            if cur_idx >= MA_PERIOD-1:
                w = df.iloc[cur_idx-MA_PERIOD+1:cur_idx+1]['close'].values.astype(np.float64)
                if len(w) == MA_PERIOD:
                    cur_ma = np.mean(w)
            if not np.isfinite(cur_ma):
                cur_ma = ema
            max_open = cur_ma * 1.01
            if do <= max_open:
                ep = do
                actual_entry_idx = cur_idx
                break
            elif elv <= max_open:
                ep = elv
                actual_entry_idx = cur_idx
                break
        if ep is None:
            skipped += 1
            continue
        
        # 止损价: 18日均线（连续2日跌破才执行）
        stop_price = ema
        
        # 动态仓位
        max_loss_price = stop_price * 0.95
        st = {'equity': equity, 'peak': peak_equity, 'low_point': equity_low_point,
              'recent_trades': recent_returns[-max(CONSECUTIVE_LOSS_LOOKBACK*3,20):],
              'win_rate': win_rate, 'avg_win': avg_win, 'avg_loss': avg_loss,
              'total_trades': total_trades_count}
        pos = calc_position_size(st, ep, max_loss_price)
        entry_key = pd.Timestamp(df.iloc[actual_entry_idx]['trade_date']).strftime('%Y-%m-%d')
        used = daily_position_used.get(entry_key, 0.0)
        if used + pos > MAX_DAILY_TOTAL_POSITION:
            avail = MAX_DAILY_TOTAL_POSITION - used
            if avail < MIN_POSITION: skipped += 1; continue
            pos = avail
        daily_position_used[entry_key] = used + pos
        
        # 出场逻辑
        exit_idx, epv, reason = None, None, None
        below_count = 0
        
        for la in range(1, MAX_HOLD+1):
            if actual_entry_idx+la >= len(df): break
            row = df.iloc[actual_entry_idx+la]; ddo, ddh, ddl, ddc = row['open'], row['high'], row['low'], row['close']
            ci = actual_entry_idx + la
            
            # 更新止盈: BOLL上轨
            tp_boll = np.inf
            if ci >= BOLL_PERIOD-1:
                w = df.iloc[ci-BOLL_PERIOD+1:ci+1]['close'].values.astype(np.float64)
                if len(w) == BOLL_PERIOD:
                    boll_upper = np.mean(w) + BOLL_STD * np.std(w, ddof=1)
                    tp_boll = boll_upper
            
            # 更新止损: 18日均线上移
            current_ma = np.nan
            if ci >= MA_PERIOD-1:
                w = df.iloc[ci-MA_PERIOD+1:ci+1]['close'].values.astype(np.float64)
                if len(w) == MA_PERIOD:
                    current_ma = np.mean(w)
            if not np.isnan(current_ma):
                stop_price = max(stop_price, current_ma)
            
            # 止盈检查: 触及BOLL上轨
            if ddh >= tp_boll-1e-8:
                epv = ddh; exit_idx=actual_entry_idx+la; reason='take_profit'; break
            
            # 连续2日跌破18日线止损
            if ddc < stop_price:
                below_count += 1
                if below_count >= MA_BREAK_CANDLES:
                    epv = ddc; exit_idx=actual_entry_idx+la; reason='stop_loss'; break
            else:
                below_count = 0  # 回到线上，计数器清零
            
        if exit_idx is None:
            li = min(actual_entry_idx+MAX_HOLD, len(df)-1); exit_idx=li; epv=float(df.iloc[li]['close']); reason='timeout'
        
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
        
        trades.append({'code':code, 'name':sig.get('name',''), 'entry_date':df.iloc[actual_entry_idx]['trade_date'],
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
    avg_hold = tdf['hold_days'].mean() if len(tdf)>0 else 0
    avg_pos = tdf['position_pct'].mean() if len(tdf)>0 else 0
    
    # 盈亏比
    bc_ratio = abs(avg_w / avg_l) if avg_l != 0 else 0
    
    # 止损明细
    sl = tdf[tdf['exit_reason']=='stop_loss']
    sl_count = len(sl); sl_wr = sl['ret_ac'].gt(0).mean()*100 if sl_count>0 else 0
    sl_avg_r = sl['ret_ac'].mean() if sl_count>0 else 0
    sl_avg_w = sl[sl['ret_ac']>0]['ret_ac'].mean() if sl_count>0 and sl['ret_ac'].gt(0).any() else 0
    sl_avg_l = sl[sl['ret_ac']<=0]['ret_ac'].mean() if sl_count>0 and sl['ret_ac'].le(0).any() else 0
    sl_avg_hold = sl['hold_days'].mean() if sl_count>0 else 0
    
    # 止盈明细
    tp = tdf[tdf['exit_reason']=='take_profit']
    tp_count = len(tp); tp_avg_r = tp['ret_ac'].mean() if tp_count>0 else 0
    tp_avg_hold = tp['hold_days'].mean() if tp_count>0 else 0
    
    print()
    print("=" * 70)
    print("📊 涨停→18日线回踩再上穿 v9")
    print("=" * 70)
    print(f"  {'总交易数':<22} {len(tdf):>8,}")
    print(f"  {'胜率':<22} {wr:>7.2f}%")
    print(f"  {'平均收益/笔':<22} {avg_r:>+8.2f}%")
    print(f"  {'平均盈利 / 平均亏损':<22} {avg_w:>+7.2f}% / {avg_l:>+6.2f}%")
    print(f"  {'盈亏比':<22} {bc_ratio:>8.2f}:1")
    print(f"  {'平均持有':<22} {avg_hold:>7.1f}日")
    print(f"  {'平均仓位':<22} {avg_pos:>7.2f}%")
    print(f"  {'总净值收益':<22} {ret_pct:>+8.2f}%")
    print(f"  {'月频夏普':<22} {sharpe:>8.2f}")
    print(f"  {'最大回撤':<22} {max_dd*100:>7.2f}%")
    print(f"  {'过滤跳过':<22} {skipped:>8,}")
    
    # 退出详情
    print(f"\n  {'──止损明细──':<22}")
    print(f"  {'止损次数':<22} {sl_count:>8,} ({sl_count/len(tdf)*100:.1f}%)")
    print(f"  {'止损胜率':<22} {sl_wr:>7.2f}%")
    print(f"  {'止损平均收益':<22} {sl_avg_r:>+8.2f}%")
    print(f"  {'止损中盈利的':<22} {sl_avg_w:>+8.2f}%（追踪止损小赚出局）")
    print(f"  {'止损中亏损的':<22} {sl_avg_l:>+8.2f}%（真正亏钱）")
    print(f"  {'止损平均持有':<22} {sl_avg_hold:>7.1f}日")
    print(f"\n  {'──止盈明细──':<22}")
    print(f"  {'止盈次数':<22} {tp_count:>8,} ({tp_count/len(tdf)*100:.1f}%)")
    print(f"  {'止盈平均收益':<22} {tp_avg_r:>+8.2f}%")
    print(f"  {'止盈平均持有':<22} {tp_avg_hold:>7.1f}日")
    
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
    
    tdf.to_csv(os.path.join(OUTPUT_DIR, "breakout_v9_backtest.csv"), index=False)
    pd.DataFrame(nav_history).to_csv(os.path.join(OUTPUT_DIR, "breakout_v9_nav.csv"), index=False)
    pd.DataFrame([{'trades':len(tdf), 'win_rate':round(wr,2), 'avg_ret':round(avg_r,2),
                    'total_return':round(ret_pct,2), 'sharpe':round(sharpe,2),
                    'max_dd':round(max_dd*100,2)}]).to_csv(os.path.join(OUTPUT_DIR, "breakout_v9_summary.csv"), index=False)
    print(f"\n✅ done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    run_backtest()
