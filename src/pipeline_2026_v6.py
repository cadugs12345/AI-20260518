"""
项目B v6 — Debug版：确认回测逻辑
=================================
- 跳过中性化，直接训练+回测
- 确认 ret_1d 回测收益计算正确
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("项目B v6 — 回测debug")
print("=" * 60)

# 只读少量列加速
pf = pq.ParquetFile('data/factors/factor_panel_v5_final.parquet')
schema = pf.schema_arrow.names

use_cols = ['ts_code','trade_date','close','市值'] + \
           [c for c in schema if c not in ('ts_code','trade_date','close','ret_1d',
              'overnight_ret','moneyflow_raw','moneyflow_strength','idvol','turnover_bias',
              'revise_up_proxy','margin_proxy','seat_premium','big_order_ratio')]

panel = pf.read(columns=use_cols).to_pandas()
panel = panel[panel['trade_date'].astype(str) >= '2026-01-01'].copy()
panel = panel.sort_values(['ts_code','trade_date']).reset_index(drop=True)

si = pd.read_parquet('data/raw/stock_list.parquet')[['ts_code','name','industry']]
panel = panel.merge(si, on='ts_code', how='left')

# 标签
panel['ret_1d'] = panel.groupby('ts_code')['close'].transform(lambda x: x.pct_change())
panel['fwd_20d_ret'] = panel.groupby('ts_code')['close'].transform(lambda x: x.shift(-20)/x - 1)
panel['fwd_20d_ret'] = panel['fwd_20d_ret'].replace([np.inf,-np.inf], np.nan)
panel['label_rank'] = panel.groupby('trade_date')['fwd_20d_ret'].rank(pct=True)
panel['valid'] = panel['fwd_20d_ret'].notna() & (panel['close'] > 0)

# 特征
feat_exclude = {'ts_code','trade_date','name','industry','close','fwd_20d_ret',
                'label_rank','valid','ret_1d','市值'}
feature_cols = [c for c in panel.columns if c not in feat_exclude and panel[c].dtype in ('float64','int64')]
print(f"面板: {panel.shape}, 特征: {len(feature_cols)}")

# 训练
td = panel[panel['valid']].copy()
for c in feature_cols:
    td[c] = td[c].fillna(td[c].median())

ds = sorted(td['trade_date'].unique())
sp = int(len(ds)*0.8)
model = lgb.train({
    'objective':'regression','metric':'mse','num_leaves':31,'learning_rate':0.05,
    'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,
    'verbosity':-1,'seed':42,'n_jobs':4,
}, lgb.Dataset(
    td[td['trade_date'].isin(set(ds[:sp]))][feature_cols].values,
    td[td['trade_date'].isin(set(ds[:sp]))]['label_rank'].values),
   num_boost_round=500,
   valid_sets=[lgb.Dataset(
       td[td['trade_date'].isin(set(ds[sp:]))][feature_cols].values,
       td[td['trade_date'].isin(set(ds[sp:]))]['label_rank'].values)],
   callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
print(f"最佳: {model.best_iteration}")

# 预测
for c in feature_cols:
    panel[c] = panel[c].fillna(panel[c].median())
panel['pred'] = model.predict(panel[feature_cols].values)
panel['pred_rk'] = panel.groupby('trade_date')['pred'].rank(pct=True)

# 回测
panel['ym'] = panel['trade_date'].dt.to_period('M')
panel['day_rk'] = panel.groupby('ym')['trade_date'].transform('rank')
panel['is_sig'] = panel['day_rk'] <= 3

all_dates = sorted(panel['trade_date'].unique())
pf_h = {}
rets = []

for dt in all_dates:
    dd = panel[panel['trade_date'] == dt]
    if len(dd) == 0:
        continue

    if dd['is_sig'].iloc[0]:
        t10 = dd.nlargest(10, 'pred_rk')
        pf_h = {r['ts_code']: 1.0/10 for _, r in t10.iterrows()}

    if pf_h:
        dm = dd.set_index('ts_code')
        pr = 0.0
        vw = 0
        for ts, w in pf_h.items():
            if ts in dm.index:
                r_v = dm.loc[ts, 'ret_1d']
                if pd.notna(r_v):
                    pr += w * r_v
                    vw += w
        rets.append(pr if vw > 0 else 0.0)
    else:
        rets.append(0.0)

rets = np.array(rets)
sr = np.sqrt(252) * rets.mean() / (rets.std() + 1e-10)
cum = np.cumprod(1 + rets)
mdd = (np.maximum.accumulate(cum) - cum).max()

print(f"\n  📊 回测（未中性化，debug版）")
print(f"  日均收益: {rets.mean()*100:+.6f}%")
print(f"  累计收益: {(cum[-1]-1)*100:.2f}%")
print(f"  年化夏普: {sr:.4f}")
print(f"  最大回撤: {mdd*100:.2f}%")
print(f"  非零收益天数: {(rets!=0).sum()}/{len(rets)}")

# 手动验证几个交易日的持仓收益
print("\n  持仓验证:")
print(f"  第一个调仓日: {all_dates[0]}, is_sig? {panel[panel['trade_date']==all_dates[0]]['is_sig'].iloc[0]}")
first_signal = None
for dt in all_dates:
    if panel[panel['trade_date']==dt]['is_sig'].iloc[0]:
        first_signal = dt
        break
print(f"  首个信号日: {first_signal}")

if first_signal:
    sig_day = panel[panel['trade_date']==first_signal]
    t10 = sig_day.nlargest(10, 'pred_rk')
    print(f"  选股:")
    for _, r in t10.iterrows():
        print(f"    {r['ts_code']} {r['name']:10s} sc={r['pred_rk']:.4f}")

    next_dt = all_dates[all_dates.index(first_signal) + 1]
    nd = panel[panel['trade_date']==next_dt]
    dm = nd.set_index('ts_code')
    for ts, w in pf_h.items():
        if ts in dm.index:
            print(f"    {ts} ret_1d={dm.loc[ts,'ret_1d']:.6f}" if pd.notna(dm.loc[ts,'ret_1d']) else f"    {ts} ret_1d=nan")

print("\n✅ debug完成")
