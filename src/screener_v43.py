#!/usr/bin/env python3
"""V4.3 实时信号 — 今日收盘满足买入条件的股票"""
import pandas as pd, numpy as np, os, json

PROJ = "/mnt/d/AI-20260604"
DATA_DIR = os.path.join(PROJ, "data", "raw", "daily")
SL_FILE = os.path.join(PROJ, "data", "raw", "stock_list.parquet")
MA=18; W=20; LOOKBACK=60

def ma(arr,p):
    n=len(arr);r=np.full(n,np.nan)
    if n<p:return r
    s=np.cumsum(arr);r[p-1]=s[p-1]/p
    for i in range(p,n):r[i]=(s[i]-s[i-p])/p
    return r

sl=pd.read_parquet(SL_FILE)
sl['ld']=pd.to_datetime(sl['list_date'],errors='coerce')
sl=sl[sl['ld']<'2017-01-01'].head(2000)
codes=sorted(sl['ts_code'].unique())
nm=dict(zip(sl['ts_code'],sl['name']))
ind=dict(zip(sl['ts_code'],sl.get('industry',['']*len(sl))))

sd={}
for idx,code in enumerate(codes):
    fp=os.path.join(DATA_DIR,f"{code}.parquet")
    if not os.path.exists(fp):continue
    try:df=pd.read_parquet(fp).sort_values('trade_date').reset_index(drop=True)
    except:continue
    if len(df)<250:continue
    c=df['close'].values.astype(np.float64);h=df['high'].values.astype(np.float64)
    v=df['vol'].values.astype(np.float64);o=df['open'].values
    n=len(c);dt=[str(d)[:10] for d in df['trade_date'].values]
    ma18=ma(c,MA);mu=np.full(n,False);mu[1:]=ma18[1:]>ma18[:-1]
    limit=np.zeros(n,dtype=bool);limit[1:]=(c[1:]/c[:-1]>1.095)&(np.abs(c[1:]-h[1:])<1e-6)
    sd[code]={'d':dt,'c':c,'o':o,'h':h,'v':v,'ma18':ma18,'mu':mu,'limit':limit,'nm':nm.get(code,'')}

today='2026-06-08'
signals=[]

for code,s in sd.items():
    dt=s['d'];c=s['c'];v=s['v'];o=s['o'];h=s['h']
    ma18=s['ma18'];mu=s['mu'];limit=s['limit']
    try:ti=dt.index(today)
    except ValueError:continue
    
    # 今天收盘买入条件检查
    j=ti  # 今天
    if np.isnan(ma18[j]):continue
    
    # 今天的MA18必须走平或微向上/向下（斜率-0.1%~1%）
    if j<4 or np.isnan(ma18[j-4]):continue
    slp=(ma18[j]/ma18[j-4]-1)*100
    if slp<-0.1 or slp>1:continue
    
    # 今天必须是阴线
    if c[j]>=o[j]:continue
    
    # 今天量=5日最低
    if v[j]>np.min(v[j-4:j+1]):continue
    
    # 今天量>120日均量
    v120=np.mean(v[max(0,j-119):j+1])
    if np.isnan(v120) or v[j]<=v120:continue
    
    # MA18向上要确认，但这里slp已经-0.1%-1%之间，可能是微向下
    
    # 往前找标志K线（60天内）
    mk_found=False;mk_date=0
    for i in range(max(MA,ti-LOOKBACK),ti+1):
        if not mu[i]:continue
        mk=False
        if limit[i]:mk=True
        else:
            if i<1:continue
            pv=v[i-1];pct=(c[i]/c[i-1]-1)*100
            if pv>0 and v[i]>=pv*3 and pct>6 and c[i]>o[i]:mk=True
        if mk:
            # 标志K线后股价曾跌破MA18
            for k in range(i+1,min(len(c),i+W+1)):
                if c[k]<ma18[k]:
                    mk_found=True;mk_date=k;break
            if mk_found:break
    
    if not mk_found:continue
    
    # 今日收盘 < MA18（在MA18下方才符合买入条件）
    if c[j]>=ma18[j]:continue
    
    ep=round(ma18[j]*1.01,2)
    pct=round((c[j]/ma18[j]-1)*100,2)
    signals.append({
        'code':code,'name':s['nm'],'industry':ind.get(code,''),
        'price':round(c[j],2),'ma18':round(ma18[j],2),
        'dist_pct':pct,'entry_price':ep,
        'mark_date':dt[mk_date],'slope':round(slp,2),
        'vol_ratio':round(v[j]/v120,2) if not np.isnan(v120) else 0
    })

signals.sort(key=lambda x:x['dist_pct'],reverse=True)

print(f"V4.3 今日信号: {len(signals)}只\n")
print(f"{'代码':<12}{'名称':<10}{'行业':<10}{'现价':<8}{'MA18':<8}{'距MA18%':<10}{'斜率%':<8}{'量比':<8}")
print("-"*80)
for s in signals:
    c=s['dist_pct']
    mark='🟢' if c>=-3 else '⚠️' if c>=-5 else '❌'
    print(f"{mark} {s['code']:<10}{s['name']:<10}{s['industry'][:6]:<10}{s['price']:<8}{s['ma18']:<8}{s['dist_pct']:<10}{s['slope']:<8}{s['vol_ratio']:<8}")

near=[s for s in signals if s['dist_pct']>=-3]
print(f"\n可关注(距MA18>=-3%): {len(near)}只")
for s in near:
    print(f"  {s['code']} {s['name']} {s['dist_pct']}% 参考买入不高于{s['entry_price']}")

out=os.path.join(PROJ,"signals","v43_signals_latest.json")
json.dump(signals,open(out,'w'),ensure_ascii=False,indent=2)
if near:
    json.dump(near,open(os.path.join(PROJ,"signals","v43_signals_near.json"),'w'),ensure_ascii=False,indent=2)
print(f"\n已保存: {out}")
