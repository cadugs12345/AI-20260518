#!/usr/bin/env python3
"""
分析: 涨停后到信号日的天数 vs 前向收益
=====================================
测试不同 MAX_DAYS_SINCE_LIMIT 的胜率和平均收益
"""
import pandas as pd, numpy as np, os, sys, warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

PROJ = "/mnt/d/AI-20260604"
DATA_DIR = os.path.join(PROJ, "data", "raw", "daily")
SL_FILE = os.path.join(PROJ, "data", "raw", "stock_list.parquet")

LIMIT_UP = 1.095; MA_PERIOD = 18
MIN_TRADE_DAYS = 180
BOLL_PERIOD = 20; BOLL_STD = 2.0

def get_cross_limit(ind):
    limits = {'银行':10,'保险':10,'石油':10,'石化':10,
              '公用':8,'交通':8,'运输':8,'建筑':8,'汽车':8,
              '地产':8,'有色':8,'煤炭':8,'商贸':8,'家电':8,'食品':8,'饮料':8,
              '电子':4,'计算机':4,'通信':4,'传媒':4,'军工':4,'国防':4,'非银':4,'综合':4}
    for k,v in limits.items():
        if k in str(ind): return v
    return 5.0

def detect_with_days(df, code, ind):
    """检测信号，同时记录 days_since_limit"""
    c = df['close'].values.astype(np.float64)
    h = df['high'].values.astype(np.float64)
    v = df['vol'].values.astype(np.float64)
    n = len(df)
    ma = pd.Series(c).rolling(MA_PERIOD).mean().values
    mu = np.full(n, False); mu[1:] = ma[1:] > ma[:-1]
    lu = np.full(n, False); lu[1:] = (c[1:]/c[:-1] > LIMIT_UP) & (c[1:] == h[1:])
    
    # 距最近涨停天数
    lli = np.full(n, -1, dtype=np.int32); ls = -1
    for i in range(n):
        if lu[i]: ls = i
        lli[i] = i-ls if ls>=0 else -1

    ma5 = pd.Series(c).rolling(5).mean().values
    ma10 = pd.Series(c).rolling(10).mean().values

    sigs = []
    for i in range(n):
        ds = lli[i]
        if ds <= 0 or ds > 25: continue
        lb = i-ds
        if np.isnan(ma[i]) or np.isnan(ma[lb]): continue
        # MA18全程向上
        if not mu[i]: continue
        ma_all_up = all(mu[lb+1:i+1])
        if not ma_all_up: continue
        # 排除涨超15%
        if (np.max(c[lb+1:i+1])/c[lb]-1)*100 > 15: continue
        # 价量共振
        if np.isnan(ma5[i]) or c[i] <= ma5[i]: continue
        if np.isfinite(ma10[i]) and ma5[i] <= ma10[i]: continue
        if not np.isfinite(ma[i]) or c[i] <= ma[i]: continue
        # 上穿幅度
        cross_pct = (c[i]/ma[i]-1)*100
        max_cross = get_cross_limit(ind)
        if cross_pct > max_cross: continue
        # 放量
        vol_sum=0; vol_count=0
        for jj in range(max(0,i-5),i):
            if v[jj]>0: vol_sum+=v[jj]; vol_count+=1
        vol_ma5 = vol_sum/max(vol_count,1)
        if vol_count>=3 and v[i]<=vol_ma5*1.2: continue
        
        sigs.append({
            'idx': i, 'code': code,
            'date': df.iloc[i]['trade_date'],
            'days_since_limit': int(ds),
            'limit_date': df.iloc[lb]['trade_date'],
            'close': float(c[i]),
            'ma18': float(ma[i]),
        })
    return sigs


def main():
    print("📊 分析: 涨停后距离天数 vs 信号质量")
    print("="*60)
    
    # 加载股票列表
    sl = pd.read_parquet(SL_FILE)
    codes = sorted(sl['ts_code'].unique())
    ind_map = dict(zip(sl['ts_code'], sl.get('industry', ['']*len(sl))))
    print(f"   股票: {len(codes)}只")
    
    # 只分析2023-2026年的信号（样本更近更有参考性）
    start_date = pd.Timestamp('2023-01-01')
    
    all_sigs = []
    for idx, code in enumerate(codes):
        if (idx+1) % 1000 == 0:
            print(f"   {idx+1}/{len(codes)}", flush=True)
        fp = os.path.join(DATA_DIR, f"{code}.parquet")
        if not os.path.exists(fp): continue
        try: df = pd.read_parquet(fp).sort_values('trade_date').reset_index(drop=True)
        except: continue
        if len(df) < MIN_TRADE_DAYS: continue
        sigs = detect_with_days(df, code, ind_map.get(code,''))
        for s in sigs:
            if pd.Timestamp(s['date']) >= start_date:
                all_sigs.append(s)
    
    print(f"\n   2023-2026信号总数: {len(all_sigs):,}")
    
    # 计算每个信号的5/10/20日前向收益
    # 加载所有股票的close
    print("   加载股价...", end=" ", flush=True)
    close_map = {}
    for code, d in [(s['code'], s['date']) for s in all_sigs[:1]]:
        pass
    # 批量加载
    needed_codes = set(s['code'] for s in all_sigs)
    for code in needed_codes:
        fp = os.path.join(DATA_DIR, f"{code}.parquet")
        if os.path.exists(fp):
            df = pd.read_parquet(fp, columns=['trade_date','close']).sort_values('trade_date')
            close_map[code] = df
    print("✓")
    
    print("   计算前向收益...", end=" ", flush=True)
    for s in all_sigs:
        code = s['code']; si_date = pd.Timestamp(s['date'])
        df = close_map.get(code)
        if df is None: continue
        mask = df['trade_date'] > si_date
        future = df[mask]
        for horizon, label in [(5,'fwd5'),(10,'fwd10'),(15,'fwd15'),(20,'fwd20')]:
            if len(future) > horizon:
                fwd_close = float(future.iloc[horizon-1]['close'])
                s[label] = (fwd_close / s['close'] - 1) * 100
    # 有前向收益的信号
    valid = [s for s in all_sigs if 'fwd5' in s]
    print(f"{len(valid):,}个有效")
    
    df = pd.DataFrame(valid)
    
    # 按days_since_limit分桶
    buckets = [(1,3),(4,5),(6,7),(8,10),(11,15),(16,20),(21,25)]
    
    print(f"\n{'距涨停天数':<12} {'信号数':>8} {'fwd5胜率':>8} {'fwd5均收':>8} {'fwd10胜率':>8} {'fwd10均收':>8} {'fwd15胜率':>8} {'fwd15均收':>8} {'fwd20胜率':>8} {'fwd20均收':>8}")
    print("-"*100)
    
    for lo, hi in buckets:
        sub = df[(df['days_since_limit']>=lo)&(df['days_since_limit']<=hi)]
        n = len(sub)
        vals = []
        for h in ['fwd5','fwd10','fwd15','fwd20']:
            vals.append(f"{(sub[h]>0).mean()*100:.1f}%")
            vals.append(f"{sub[h].mean():+.1f}%")
        print(f"  {lo:2d}-{hi:2d}天     {n:>8,}   {vals[0]:>8} {vals[1]:>8} {vals[2]:>8} {vals[3]:>8} {vals[4]:>8} {vals[5]:>8} {vals[6]:>8} {vals[7]:>8}")
    
    # 累积视角: 如果MAX_DAYS设为N，效果如何
    print(f"\n{'MAX_DAYS':>8} {'信号数':>8} {'fwd5胜率':>8} {'fwd5均值':>8} {'fwd10胜率':>8} {'fwd10均值':>8} {'fwd20胜率':>8} {'fwd20均值':>8}")
    print("-"*80)
    for max_days in [3,5,7,10,15,20,25]:
        sub = df[df['days_since_limit'] <= max_days]
        n = len(sub)
        v5 = (sub['fwd5']>0).mean()*100
        m5 = sub['fwd5'].mean()
        v10 = (sub['fwd10']>0).mean()*100
        m10 = sub['fwd10'].mean()
        v20 = (sub['fwd20']>0).mean()*100
        m20 = sub['fwd20'].mean()
        marker = " ← 当前" if max_days == 25 else ""
        print(f"  {max_days:>8} {n:>8,} {v5:>7.1f}% {m5:>+7.1f}% {v10:>7.1f}% {m10:>+7.1f}% {v20:>7.1f}% {m20:>+7.1f}%{marker}")

    # 天数分布图
    print(f"\n📊 天数分布:")
    dist = df['days_since_limit'].value_counts().sort_index()
    for d, cnt in dist.items():
        bar = '█' * max(1, cnt // 200)
        print(f"   {d:2d}天: {cnt:>5,} {bar}")
    
    # 最优窗口建议
    print(f"\n📌 结论:")
    best = 0; best_win = 0
    for max_days in [3,5,7,10,15,20,25]:
        sub = df[df['days_since_limit'] <= max_days]
        wr = (sub['fwd10']>0).mean()
        if wr > best_win: best = max_days; best_win = wr
    print(f"   基于fwd10胜率, 最优MAX_DAYS={best}, 胜率={best_win*100:.1f}%")


if __name__ == '__main__':
    main()
