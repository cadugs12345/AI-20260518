"""
因子构建 (WSL本地IO, 快速版)
"""
import os, sys, time, gc
import numpy as np
import pandas as pd

from config.settings import DATA_RAW, DATA_FACTORS

daily_dir = os.path.join(DATA_RAW, "daily")
factor_out = DATA_FACTORS

stock_list = pd.read_parquet(os.path.join(DATA_RAW, "stock_list.parquet"))
all_codes = stock_list['ts_code'].tolist()
print(f"股票池: {len(all_codes)} 只", flush=True)

sample = pd.read_parquet(os.path.join(daily_dir, f"{all_codes[0]}.parquet"))
all_dates = sorted(sample['trade_date'].dt.strftime('%Y%m%d').tolist())
years = sorted(set(d[:4] for d in all_dates))
print(f"交易日: {len(all_dates)} 天, 年份: {years}", flush=True)

def calc_factors(daily):
    """轻量时序因子计算"""
    daily = daily.set_index('trade_date') if 'trade_date' in daily.columns else daily
    c, h, l, v, p, a = daily['close'], daily['high'], daily['low'], daily['vol'], daily['pct_chg'], daily['amount']
    f = pd.DataFrame(index=daily.index)
    f['短期反转'] = -c.pct_change(5)
    f['20日动量'] = c.pct_change(20)
    f['60日动量'] = c.pct_change(60)
    f['120日动量'] = c.pct_change(120)
    f['波动率'] = p.rolling(20, min_periods=5).std() * np.sqrt(252) / 100.0
    for per in [6,12,24]:
        d = c.diff(); g = d.where(d>0,0); ls = (-d).where(d<0,0)
        f[f'RSI_{per}'] = 100 - 100/(1 + g.rolling(per).mean() / ls.rolling(per).mean().replace(0, np.nan))
    for per in [5,10,20]:
        f[f'EMA{per}偏离'] = (c - c.ewm(span=per).mean()) / c * 100
    ma20 = c.rolling(20).mean()
    f['BOLL位置'] = ((c - ma20) / (2*c.rolling(20).std()+1e-10)).clip(-3,3)
    ema12 = c.ewm(span=12).mean(); ema26 = c.ewm(span=26).mean()
    d12, d26 = ema12-ema26, (ema12-ema26).ewm(span=9).mean()
    f['MACD'] = ((d12-d26)*2/c*100).clip(-10,10)
    f['量能趋势'] = (v.ewm(span=20).mean()/c).pct_change(20).clip(-50,50)*100
    obv = (v * np.sign(p)).cumsum()
    f['OBV'] = ((obv - obv.expanding().mean()) / (obv.expanding().std()+1e-10)).clip(-5,5)
    return f

total = len(all_codes)
for year in years:
    t0 = time.time()
    ys = pd.Timestamp(f'{year}-01-01'); ye = pd.Timestamp(f'{int(year)+1}-01-01') - pd.Timedelta(days=1)
    all_rows = []
    for i, code in enumerate(all_codes):
        fp = os.path.join(daily_dir, f'{code}.parquet')
        if not os.path.exists(fp): continue
        d = pd.read_parquet(fp)
        if len(d) < 60: continue
        d['trade_date'] = pd.to_datetime(d['trade_date'])
        d = d.sort_values('trade_date').set_index('trade_date')
        fx = calc_factors(d)
        mask = (fx.index >= ys) & (fx.index <= ye)
        fy = fx[mask].copy()
        if fy.empty: continue
        fy['ts_code'] = code
        fy = fy.reset_index().rename(columns={'index':'trade_date'})
        all_rows.append(fy)
        if (i+1) % 1000 == 0:
            print(f"  [{year}] {i+1}/{total} stocks", flush=True)
    if all_rows:
        df = pd.concat(all_rows, ignore_index=True)
        df.to_parquet(os.path.join(factor_out, f'timeseries_{year}.parquet'), index=False)
        print(f"[{year}] saved: {len(df)} rows, {time.time()-t0:.0f}s", flush=True)
    del all_rows; gc.collect()

print("Done!", flush=True)
