"""
项目B v7 — 2026年 中性化 LGBM + 精确回测（最终版）
=====================================================
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("项目B v7 — 中性化LGBM + 精确回测")
print("=" * 60)

# ============================================================
# 1. 加载
# ============================================================
print("\n[1/5] 加载...")
pf = pq.ParquetFile('data/factors/factor_panel_v5_final.parquet')
schema = pf.schema_arrow.names

use_cols = ['ts_code','trade_date','close','市值'] + \
           [c for c in schema if c not in ('ts_code','trade_date','close','ret_1d')]
panel = pf.read(columns=use_cols).to_pandas()
panel = panel[panel['trade_date'].astype(str) >= '2026-01-01'].copy()
panel = panel.sort_values(['ts_code','trade_date']).reset_index(drop=True)
print(f"  面板: {panel.shape}, 交易日: {panel['trade_date'].nunique()}")

si = pd.read_parquet('data/raw/stock_list.parquet')[['ts_code','name','industry']]
panel = panel.merge(si, on='ts_code', how='left')

# 标签+ret_1d
panel['ret_1d'] = panel.groupby('ts_code')['close'].transform(lambda x: x.pct_change())
panel['fwd_20d_ret'] = panel.groupby('ts_code')['close'].transform(lambda x: x.shift(-20)/x - 1)
panel['fwd_20d_ret'] = panel['fwd_20d_ret'].replace([np.inf,-np.inf], np.nan)
panel['label_rank'] = panel.groupby('trade_date')['fwd_20d_ret'].rank(pct=True)
panel['valid'] = panel['fwd_20d_ret'].notna() & (panel['close'] > 0)

# 特征
feat_exclude = {'ts_code','trade_date','name','industry','close','fwd_20d_ret',
                'label_rank','valid','ret_1d','市值',
                'overnight_ret','moneyflow_raw','moneyflow_strength',
                'idvol','turnover_bias','revise_up_proxy','margin_proxy',
                'seat_premium','big_order_ratio'}
feature_cols = [c for c in panel.columns if c not in feat_exclude and panel[c].dtype in ('float64','int64')]
print(f"  特征: {len(feature_cols)}")

# ============================================================
# 2. 中性化
# ============================================================
print("\n[2/5] 中性化...")
for c in feature_cols:
    panel[c] = panel[c].fillna(panel[c].median())

feat_neut = {c: np.full(len(panel), np.nan) for c in feature_cols}
dates = sorted(panel['trade_date'].unique())

for i, dt in enumerate(dates):
    if (i+1)%20 == 0:
        print(f"  {i+1}/{len(dates)}")

    mask = panel['trade_date'] == dt
    idx = np.where(mask)[0]
    sub = panel.iloc[idx]
    n = len(sub)
    if n < 100:
        for c in feature_cols:
            feat_neut[c][idx] = sub[c].values
        continue

    # 设计矩阵
    cap = np.log(np.maximum(sub['市值'].values, 1e-6))
    cap_z = (cap - cap.mean()) / (cap.std() + 1e-10)
    inds = sub['industry'].values
    ind_vals = sorted(set(v for v in inds if isinstance(v, str)))
    ind_arr = np.zeros((n, len(ind_vals)))
    for j, iv in enumerate(ind_vals):
        ind_arr[:, j] = (inds == iv).astype(float)
    X = np.column_stack([np.ones(n), cap_z, ind_arr])

    y_mat = np.column_stack([sub[c].values.astype(float) for c in feature_cols])
    good_cols = ~np.any(np.isnan(y_mat) | np.isinf(y_mat), axis=0)

    try:
        beta = np.linalg.lstsq(X, y_mat[:, good_cols], rcond=None)[0]
        res = y_mat[:, good_cols] - X @ beta
        for j, ci in enumerate(np.where(good_cols)[0]):
            feat_neut[feature_cols[ci]][idx] = res[:, j]
    except:
        for c in feature_cols:
            feat_neut[c][idx] = sub[c].values

for c in feature_cols:
    panel[c] = feat_neut[c]

print(f"  {len(dates)}/{len(dates)} 完成")

# ============================================================
# 3. 训练
# ============================================================
print("\n[3/5] 训练...")
td = panel[panel['valid']].copy()
for c in feature_cols:
    td[c] = td[c].fillna(0.0)

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
print(f"  最佳: {model.best_iteration}, MSE: {model.best_score['valid_0']['l2']:.6f}")

imp = pd.DataFrame({'f': feature_cols, 'g': model.feature_importance(importance_type='gain')})
imp = imp.sort_values('g', ascending=False)
print("\n  Top15:")
print(imp.head(15).to_string(index=False))
model.save_model('models/lgb_2026_v7.txt')

# ============================================================
# 4. 回测
# ============================================================
print("\n[4/5] 回测...")

for c in feature_cols:
    panel[c] = panel[c].fillna(0.0)
panel['pred'] = model.predict(panel[feature_cols].values)
panel['pred_rk'] = panel.groupby('trade_date')['pred'].rank(pct=True)

# 正确的月频信号判定
# 先构建唯一的交易日列表
unique_dates = pd.DataFrame({'trade_date': sorted(panel['trade_date'].unique())})
unique_dates['ym'] = unique_dates['trade_date'].dt.to_period('M')
unique_dates['day_rk'] = unique_dates.groupby('ym')['trade_date'].rank(method='dense')
unique_dates['is_sig'] = unique_dates['day_rk'] <= 3
sig_dates = set(unique_dates[unique_dates['is_sig']]['trade_date'])
panel['is_sig'] = panel['trade_date'].isin(sig_dates)

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
                rv = dm.loc[ts, 'ret_1d']
                if pd.notna(rv):
                    pr += w * rv
                    vw += w
        rets.append(pr if vw > 0 else 0.0)
    else:
        rets.append(0.0)

rets = np.array(rets)
sr = np.sqrt(252) * rets.mean() / (rets.std() + 1e-10)
cum = np.cumprod(1 + rets)
mdd = (np.maximum.accumulate(cum) - cum).max()

print(f"\n  📊 2026年 回测（中性化 | 月频Top10等权 | 精确ret_1d）")
print(f"  {'日均收益':>12s}: {rets.mean()*100:+.4f}%")
print(f"  {'日波动率':>12s}: {rets.std()*100:.4f}%")
print(f"  {'年化夏普':>12s}: {sr:.4f}")
print(f"  {'最大回撤':>12s}: {mdd*100:.2f}%")
print(f"  {'累计收益':>12s}: {(cum[-1]-1)*100:.2f}%")
print(f"  {'非零天数':>12s}: {(rets!=0).sum()}/{len(rets)}")
print(f"  {'正收益天数':>12s}: {(rets>0).sum()}/{len(rets)}")
print(f"  {'胜率':>12s}: {(rets>0).sum()/max(len(rets),1)*100:.1f}%")

# 重新生成 ym（中性化可能覆盖）
panel['ym'] = panel['trade_date'].dt.to_period('M')
print("\n  逐月表现:")
pfh_month = pf_h
for ym in sorted(panel['ym'].unique()):
    pm = panel[panel['ym'] == ym]
    mrets = []
    for dt in sorted(pm['trade_date'].unique()):
        dd = pm[pm['trade_date'] == dt]
        if dd['is_sig'].iloc[0]:
            t10 = dd.nlargest(10, 'pred_rk')
            pfh_month = {r['ts_code']: 1.0/10 for _, r in t10.iterrows()}
        if pfh_month:
            dm = dd.set_index('ts_code')
            pr = sum(w * dm.loc[ts,'ret_1d'] for ts, w in pfh_month.items() if ts in dm.index and pd.notna(dm.loc[ts,'ret_1d']))
            mrets.append(pr)
    if mrets:
        ma = np.array(mrets)
        print(f"    {ym}: 日均{ma.mean()*100:+.4f}%, 夏普{np.sqrt(252)*ma.mean()/(ma.std()+1e-10):.2f}")

# ============================================================
# 5. 最新信号 + 市值对比
# ============================================================
print("\n[5/5] 最新信号...")

ld = panel['trade_date'].max()
lt = panel[panel['trade_date'] == ld].sort_values('pred', ascending=False)

# 从原始面板读市值（避免中性化影响展示）
pf_orig = pq.ParquetFile('data/factors/factor_panel_v5_final.parquet')
orig = pf_orig.read(columns=['ts_code','trade_date','市值']).to_pandas()
orig_2026 = orig[orig['trade_date'].astype(str) >= '2026-01-01']
orig_latest = orig_2026[orig_2026['trade_date'] == ld]
orig_map = orig_latest.set_index('ts_code')['市值'].to_dict()

print(f"\n  📋 {ld} Top20:")
tcaps = []
all_caps = []
for i, (_, r) in enumerate(lt.head(20).iterrows()):
    raw_cap = orig_map.get(r['ts_code'], np.nan)
    if pd.notna(raw_cap):
        cap_v = np.exp(raw_cap) / 1e8
        tcaps.append(cap_v)
    else:
        cap_v = 0
    print(f"  {i+1:2d}. {r['ts_code']} {r['name']:<10s} | {r['industry']:<8s} | {cap_v:.0f}亿 | sc={r['pred_rk']:.4f}")

for _, r in lt.iterrows():
    raw_cap = orig_map.get(r['ts_code'], np.nan)
    if pd.notna(raw_cap):
        all_caps.append(np.exp(raw_cap)/1e8)

print(f"\n  市值对比:")
print(f"    Top20中位数: {np.median(tcaps):.0f}亿  |  全市场中位数: {np.median(all_caps):.0f}亿")
print(f"    Top20均值:   {np.mean(tcaps):.0f}亿  |  全市场均值:   {np.mean(all_caps):.0f}亿")

print("\n" + "=" * 60)
print("✅ 项目B v7 完成")
print("=" * 60)
