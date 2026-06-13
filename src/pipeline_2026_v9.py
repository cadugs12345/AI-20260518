"""
项目B v9 — 15因子 + 不等权 + 止损
==================================
改进:
1. 15个特征（保留更多有效信息）
2. score加权（不等权）
3. 简易止损（单日跌超阈值减仓）
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("项目B v9 — 15因子 + 不等权 + 止损")
print("=" * 60)

# ============================================================
# 1. 加载
# ============================================================
print("\n[1/6] 加载2025+2026数据...")
pf = pq.ParquetFile('data/factors/factor_panel_v5_final.parquet')
schema = pf.schema_arrow.names
use_cols = ['ts_code','trade_date','close','市值'] + \
           [c for c in schema if c not in ('ts_code','trade_date','close','ret_1d')]
panel = pf.read(columns=use_cols).to_pandas()
panel = panel[panel['trade_date'].astype(str) >= '2025-01-01'].copy()
panel = panel.sort_values(['ts_code','trade_date']).reset_index(drop=True)
print(f"  面板: {panel.shape}, 交易日: {panel['trade_date'].nunique()}")

si = pd.read_parquet('data/raw/stock_list.parquet')[['ts_code','name','industry']]
panel = panel.merge(si, on='ts_code', how='left')
panel['year'] = panel['trade_date'].dt.year

# ============================================================
# 2. 标签
# ============================================================
panel['ret_1d'] = panel.groupby('ts_code')['close'].transform(lambda x: x.pct_change())
panel['fwd_20d_ret'] = panel.groupby('ts_code')['close'].transform(lambda x: x.shift(-20)/x - 1)
panel['fwd_20d_ret'] = panel['fwd_20d_ret'].replace([np.inf,-np.inf], np.nan)
panel['label_rank'] = panel.groupby('trade_date')['fwd_20d_ret'].rank(pct=True)
panel['valid'] = panel['fwd_20d_ret'].notna() & (panel['close'] > 0)

feat_exclude = {'ts_code','trade_date','name','industry','close','fwd_20d_ret',
                'label_rank','valid','ret_1d','year','市值',
                'overnight_ret','moneyflow_raw','moneyflow_strength',
                'idvol','turnover_bias','revise_up_proxy','margin_proxy',
                'seat_premium','big_order_ratio'}
feature_cols_all = [c for c in panel.columns if c not in feat_exclude and panel[c].dtype in ('float64','int64')]
print(f"  全部特征: {len(feature_cols_all)}")

# ============================================================
# 3. 特征预筛选 -> Top15
# ============================================================
print("\n[2/6] 特征预筛选 (找Top15)...")

for c in feature_cols_all:
    panel[c] = panel[c].fillna(panel[c].median())

dates = sorted(panel['trade_date'].unique())
print(f"  中性化 {len(dates)} 截面...")
for i, dt in enumerate(dates):
    if (i+1)%50 == 0: print(f"    {i+1}/{len(dates)}")
    mask = panel['trade_date'] == dt
    idx = np.where(mask)[0]
    sub = panel.iloc[idx]
    if len(sub) < 100: continue
    cap = np.log(np.maximum(sub['市值'].values, 1e-6))
    cap_z = (cap - cap.mean()) / (cap.std() + 1e-10)
    inds = sub['industry'].values
    ind_vals = sorted(set(v for v in inds if isinstance(v, str)))
    ind_arr = np.zeros((len(sub), len(ind_vals)))
    for j, iv in enumerate(ind_vals):
        ind_arr[:, j] = (inds == iv).astype(float)
    X = np.column_stack([np.ones(len(sub)), cap_z, ind_arr])
    y_mat = np.column_stack([sub[c].values.astype(float) for c in feature_cols_all])
    good_cols = ~np.any(np.isnan(y_mat) | np.isinf(y_mat), axis=0)
    try:
        beta = np.linalg.lstsq(X, y_mat[:, good_cols], rcond=None)[0]
        res = y_mat[:, good_cols] - X @ beta
        for j, ci in enumerate(np.where(good_cols)[0]):
            panel.iloc[idx, panel.columns.get_loc(feature_cols_all[ci])] = res[:, j]
    except:
        pass

td = panel[panel['valid']].copy()
for c in feature_cols_all:
    td[c] = td[c].fillna(0.0)
ds = sorted(td['trade_date'].unique())
sp = int(len(ds)*0.7)

model_screen = lgb.train({
    'objective':'regression','metric':'mse','num_leaves':31,'learning_rate':0.05,
    'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,
    'verbosity':-1,'seed':42,'n_jobs':4,
}, lgb.Dataset(
    td[td['trade_date'].isin(set(ds[:sp]))][feature_cols_all].values,
    td[td['trade_date'].isin(set(ds[:sp]))]['label_rank'].values),
   num_boost_round=300,
   valid_sets=[lgb.Dataset(
       td[td['trade_date'].isin(set(ds[sp:]))][feature_cols_all].values,
       td[td['trade_date'].isin(set(ds[sp:]))]['label_rank'].values)],
   callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])

imp_all = pd.DataFrame({'f': feature_cols_all, 'g': model_screen.feature_importance(importance_type='gain')})
imp_all = imp_all.sort_values('g', ascending=False)
top15_features = imp_all.head(15)['f'].tolist()
print(f"\n  Top15特征:")
for _, r in imp_all.head(15).iterrows():
    print(f"    {r['f']:25s}: {r['g']:.1f}")

# ============================================================
# 4. 用15特征重新训练
# ============================================================
print("\n[3/6] 使用Top15特征训练...")

panel2 = pf.read(columns=use_cols).to_pandas()
panel2 = panel2[panel2['trade_date'].astype(str) >= '2025-01-01'].copy()
panel2 = panel2.sort_values(['ts_code','trade_date']).reset_index(drop=True)
panel2 = panel2.merge(si, on='ts_code', how='left')
panel2['year'] = panel2['trade_date'].dt.year
panel2['ret_1d'] = panel2.groupby('ts_code')['close'].transform(lambda x: x.pct_change())
panel2['fwd_20d_ret'] = panel2.groupby('ts_code')['close'].transform(lambda x: x.shift(-20)/x - 1)
panel2['fwd_20d_ret'] = panel2['fwd_20d_ret'].replace([np.inf,-np.inf], np.nan)
panel2['label_rank'] = panel2.groupby('trade_date')['fwd_20d_ret'].rank(pct=True)
panel2['valid'] = panel2['fwd_20d_ret'].notna() & (panel2['close'] > 0)

for c in top15_features:
    panel2[c] = panel2[c].fillna(panel2[c].median())

dates2 = sorted(panel2['trade_date'].unique())
for i, dt in enumerate(dates2):
    if (i+1)%50 == 0: print(f"  中性化: {i+1}/{len(dates2)}")
    mask = panel2['trade_date'] == dt
    idx = np.where(mask)[0]
    sub = panel2.iloc[idx]
    if len(sub) < 100: continue
    cap = np.log(np.maximum(sub['市值'].values, 1e-6))
    cap_z = (cap - cap.mean()) / (cap.std() + 1e-10)
    inds = sub['industry'].values
    ind_vals = sorted(set(v for v in inds if isinstance(v, str)))
    ind_arr = np.zeros((len(sub), len(ind_vals)))
    for j, iv in enumerate(ind_vals):
        ind_arr[:, j] = (inds == iv).astype(float)
    X = np.column_stack([np.ones(len(sub)), cap_z, ind_arr])
    y_mat = np.column_stack([sub[c].values.astype(float) for c in top15_features])
    good_cols = ~np.any(np.isnan(y_mat) | np.isinf(y_mat), axis=0)
    try:
        beta = np.linalg.lstsq(X, y_mat[:, good_cols], rcond=None)[0]
        res = y_mat[:, good_cols] - X @ beta
        for j, ci in enumerate(np.where(good_cols)[0]):
            panel2.iloc[idx, panel2.columns.get_loc(top15_features[ci])] = res[:, j]
    except:
        pass

td2 = panel2[panel2['valid']].copy()
for c in top15_features:
    td2[c] = td2[c].fillna(0.0)
ds2 = sorted(td2['trade_date'].unique())
n_train = int(len(ds2)*0.7)
train_dates = set(ds2[:n_train])
val_dates = set(ds2[n_train:])

model_final = lgb.train({
    'objective':'regression','metric':'mse','num_leaves':31,'learning_rate':0.05,
    'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,
    'verbosity':-1,'seed':42,'n_jobs':4,
}, lgb.Dataset(
    td2[td2['trade_date'].isin(train_dates)][top15_features].values,
    td2[td2['trade_date'].isin(train_dates)]['label_rank'].values),
   num_boost_round=500,
   valid_sets=[lgb.Dataset(
       td2[td2['trade_date'].isin(val_dates)][top15_features].values,
       td2[td2['trade_date'].isin(val_dates)]['label_rank'].values)],
   callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
print(f"\n  最佳: {model_final.best_iteration}, MSE: {model_final.best_score['valid_0']['l2']:.6f}")

imp_final = pd.DataFrame({'f': top15_features, 'g': model_final.feature_importance(importance_type='gain')})
imp_final = imp_final.sort_values('g', ascending=False)
print("\n  最终特征重要性:")
print(imp_final.to_string(index=False))
model_final.save_model('models/lgb_2026_v9.txt')

# ============================================================
# 5. 回测
# ============================================================
print("\n[4/6] 回测...")

for c in top15_features:
    panel2[c] = panel2[c].fillna(0.0)
panel2['pred'] = model_final.predict(panel2[top15_features].values)
panel2['pred_rk'] = panel2.groupby('trade_date')['pred'].rank(pct=True)

ud = pd.DataFrame({'trade_date': sorted(panel2['trade_date'].unique())})
ud['ym'] = ud['trade_date'].dt.to_period('M')
ud['day_rk'] = ud.groupby('ym')['trade_date'].rank(method='dense')
ud['is_sig'] = ud['day_rk'] == 1
sig_dates = set(ud[ud['is_sig']]['trade_date'])
panel2['is_sig'] = panel2['trade_date'].isin(sig_dates)

STOP_LOSS = 0.07

def run_backtest(df, use_weighted=True, stop_loss=None):
    """通用回测函数"""
    all_dates = sorted(df['trade_date'].unique())
    pf = {}
    rets = []
    for dt in all_dates:
        dd = df[df['trade_date'] == dt]
        if len(dd) == 0: continue
        if dd['is_sig'].iloc[0]:
            top10 = dd.nlargest(10, 'pred')
            if use_weighted:
                sc = top10['pred'].values
                w = sc / sc.sum()
            else:
                w = np.ones(10) / 10
            pf = {r['ts_code']: w[j] for j, (_, r) in enumerate(top10.iterrows())}
        # 止损
        if stop_loss and pf:
            dm = dd.set_index('ts_code')
            pf = {ts: w for ts, w in pf.items()
                  if not (ts in dm.index and pd.notna(dm.loc[ts,'ret_1d']) and dm.loc[ts,'ret_1d'] < -stop_loss)}
        if pf:
            dm = dd.set_index('ts_code')
            tw = sum(pf.values())
            pr = sum((w/tw) * dm.loc[ts,'ret_1d'] for ts, w in pf.items()
                     if ts in dm.index and pd.notna(dm.loc[ts,'ret_1d']))
            rets.append(pr)
        else:
            rets.append(0)
    return np.array(rets)

# 三种方式对比
configs = [
    ('v8等权', False, None),
    ('v9不等权', True, None),
    ('v9不等权+止损7%', True, STOP_LOSS),
]

print("\n  --- 全样本对比 ---")
for name, w, sl in configs:
    r = run_backtest(panel2, use_weighted=w, stop_loss=sl)
    sr = np.sqrt(252) * r.mean() / (r.std() + 1e-10)
    cum = np.cumprod(1 + r)
    mdd = (np.maximum.accumulate(cum) - cum).max()
    print(f"  {name:25s}: 夏普={sr:.4f} | 累计={(cum[-1]-1)*100:8.2f}% | 回撤={mdd*100:5.2f}% | 胜率={(r>0).sum()/len(r)*100:.0f}%")

# 样本外
print("\n  --- 严格样本外 (2025训练->2026测试) ---")
td25 = td2[td2['trade_date'].astype(str) < '2026-01-01']
td26_test = panel2[panel2['year'] == 2026].copy()
ds25 = sorted(td25[td25['valid']]['trade_date'].unique())
sp25 = int(len(ds25)*0.8)

model_25only = lgb.train({
    'objective':'regression','metric':'mse','num_leaves':31,'learning_rate':0.05,
    'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,
    'verbosity':-1,'seed':42,'n_jobs':4,
}, lgb.Dataset(
    td25[td25['trade_date'].isin(set(ds25[:sp25]))][top15_features].values,
    td25[td25['trade_date'].isin(set(ds25[:sp25]))]['label_rank'].values),
   num_boost_round=500,
   valid_sets=[lgb.Dataset(
       td25[td25['trade_date'].isin(set(ds25[sp25:]))][top15_features].values,
       td25[td25['trade_date'].isin(set(ds25[sp25:]))]['label_rank'].values)],
   callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

for c in top15_features:
    td26_test[c] = td26_test[c].fillna(td26_test[c].median())
td26_test['pred'] = model_25only.predict(td26_test[top15_features].values)
td26_test['pred_rk'] = td26_test.groupby('trade_date')['pred'].rank(pct=True)
td26_test['is_sig'] = td26_test['trade_date'].isin(sig_dates)

for name, w, sl in configs:
    r = run_backtest(td26_test, use_weighted=w, stop_loss=sl)
    sr = np.sqrt(252) * r.mean() / (r.std() + 1e-10)
    cum = np.cumprod(1 + r)
    print(f"  {name:25s}: 夏普={sr:.4f} | 累计={(cum[-1]-1)*100:7.2f}%")

# ============================================================
# 6. 最新信号
# ============================================================
print("\n[5/6] 最新信号...")

ld = panel2['trade_date'].max()
lt = panel2[panel2['trade_date'] == ld].sort_values('pred', ascending=False)

pf_orig = pq.ParquetFile('data/factors/factor_panel_v5_final.parquet')
orig = pf_orig.read(columns=['ts_code','trade_date','市值']).to_pandas()
orig_latest = orig[orig['trade_date'] == ld]
orig_map = orig_latest.set_index('ts_code')['市值'].to_dict()

top10 = lt.head(10).copy()
scores = top10['pred'].values
weights = scores / scores.sum()

print(f"\n  {ld.date()} Top10（不等权）:")
for i, (_, r) in enumerate(top10.iterrows()):
    raw_cap = orig_map.get(r['ts_code'], np.nan)
    cap_v = np.exp(raw_cap)/1e8 if pd.notna(raw_cap) else 0
    print(f"  {i+1:2d}. {r['ts_code']} {r['name']:<10s} | {r['industry']:<8s} | {weights[i]*100:5.1f}% | {cap_v:.0f}亿")

out = top10[['ts_code','trade_date','pred']].copy()
out['weight'] = weights
out['name'] = [si[si['ts_code']==c]['name'].values[0] if len(si[si['ts_code']==c])>0 else '' for c in out['ts_code']]
out['industry'] = [si[si['ts_code']==c]['industry'].values[0] if len(si[si['ts_code']==c])>0 else '' for c in out['ts_code']]
out.to_csv('signals/v9_latest_signal.csv', index=False)
print(f"  信号已保存: signals/v9_latest_signal.csv")

print("\n" + "=" * 60)
print("项目B v9 完成")
print("=" * 60)
