#!/usr/bin/env python3
"""
v9 样本外验证
=============
训练期: 2017-01-01 ~ 2021-12-31（前5年）
验证期: 2022-01-01 ~ 2026-06-03（后4.5年）

验证: 用训练期跑出的参数，在验证期测试效果
"""
import pandas as pd
import numpy as np
import os, math

PROJ_B = "/mnt/d/AI-20260604"
BASE_CSV = os.path.join(PROJ_B, "backtest_results", "breakout_v9_backtest.csv")
OUT = os.path.join(PROJ_B, "backtest_results")

tdf = pd.read_csv(BASE_CSV)
tdf['entry_date'] = pd.to_datetime(tdf['entry_date'])

train = tdf[tdf['entry_date'] < '2022-01-01']
valid = tdf[tdf['entry_date'] >= '2022-01-01']

print("=" * 60)
print("📊 样本外验证 v9 (1+2+3)")
print("=" * 60)

for name, dfp in [('📚 训练期 (2017-2021)', train), ('🔬 验证期 (2022-2026)', valid)]:
    wr = dfp['ret_ac'].gt(0).mean() * 100
    avg_r = dfp['ret_ac'].mean()
    avg_w = dfp[dfp['ret_ac']>0]['ret_ac'].mean() if dfp['ret_ac'].gt(0).any() else 0
    avg_l = dfp[dfp['ret_ac']<=0]['ret_ac'].mean() if dfp['ret_ac'].le(0).any() else 0
    bc_ratio = abs(avg_w / avg_l) if avg_l != 0 else 0
    
    # 月频夏普
    monthly = dfp.groupby(dfp['entry_date'].dt.to_period('M'))['trade_impact_pct'].sum() / 100
    sr = monthly.mean() / monthly.std() * math.sqrt(12) if len(monthly) > 1 and monthly.std() > 0 else 0
    
    print(f"\n{name}")
    print(f"  {'交易数':<15} {len(dfp):>6}")
    print(f"  {'胜率':<15} {wr:>6.2f}%")
    print(f"  {'平均收益/笔':<15} {avg_r:>+6.2f}%")
    print(f"  {'平均盈利':<15} {avg_w:>+6.2f}%")
    print(f"  {'平均亏损':<15} {avg_l:>+6.2f}%")
    print(f"  {'盈亏比':<15} {bc_ratio:>6.2f}:1")
    print(f"  {'月频夏普':<15} {sr:>6.2f}")
    
    sl = dfp[dfp['exit_reason']=='stop_loss']
    tp = dfp[dfp['exit_reason']=='take_profit']
    sl_wr = sl['ret_ac'].gt(0).mean() * 100 if len(sl) > 0 else 0
    print(f"  {'止损/止盈':<15} {len(sl)}/{len(tp)}")
    print(f"  {'止损胜率':<15} {sl_wr:>6.2f}%")
    print(f"  {'止损均值':<15} {sl['ret_ac'].mean():>+6.2f}%" if len(sl) > 0 else "")
    
    # 最差月和最好月
    if len(monthly) > 0:
        print(f"  {'最差月':<15} {monthly.min()*100:>+6.2f}%")
        print(f"  {'最好月':<15} {monthly.max()*100:>+6.2f}%")

# 过拟合检验
print("\n" + "=" * 60)
print("📐 过拟合检验")
print("=" * 60)
train_wr = train['ret_ac'].gt(0).mean() * 100
valid_wr = valid['ret_ac'].gt(0).mean() * 100
train_sr = train.groupby(train['entry_date'].dt.to_period('M'))['trade_impact_pct'].sum()/100
valid_sr = valid.groupby(valid['entry_date'].dt.to_period('M'))['trade_impact_pct'].sum()/100
train_sr_v = train_sr.mean()/train_sr.std()*math.sqrt(12) if len(train_sr)>1 and train_sr.std()>0 else 0
valid_sr_v = valid_sr.mean()/valid_sr.std()*math.sqrt(12) if len(valid_sr)>1 and valid_sr.std()>0 else 0

print(f"  训练期夏普: {train_sr_v:.2f}  →  验证期夏普: {valid_sr_v:.2f}")
print(f"  训练期胜率: {train_wr:.1f}%  →  验证期胜率: {valid_wr:.1f}%")

diff = abs(train_sr_v - valid_sr_v)
if diff < 0.3:
    print(f"  ✅ 夏普差异 {diff:.2f}，无过拟合")
elif diff < 0.8:
    print(f"  ⚠️ 夏普差异 {diff:.2f}，轻度过拟合")
else:
    print(f"  ❌ 夏普差异 {diff:.2f}，严重过拟合")
