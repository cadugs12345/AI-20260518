"""
项目B v8 — 改进版：去噪 + 月单信号 + 两年训练
================================================
改进点：
1. 因子精选（只用Top10重要特征）
2. 每月仅1个信号日（首日）
3. 2025+2026联合训练
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("项目B v8 — 改进版")
print("=" * 60)

# ============================================================
# 1. 加载2025+2026数据
# ============================================================
print("\n[1/6] 加载2025+2026数据...")
pf = pq.ParquetFile('data/factors/factor_panel_v5_final.parquet')
schema = pf.schema_arrow.names

use_cols = ['ts_code','trade_date','close','市值'] + \
           [c for c in schema if c not in ('ts_code','trade_date','close','ret_1d')]
panel = pf.read(columns=use_cols).to_pandas()
panel = panel[panel['trade_date'].astype(str) >= '2025-01-01'].copy()
panel = panel.sort_values(['ts_code','trade_date']).reset_index(drop=True)
print(f"  面板: {panel.shape}, 交易日: {panel['trade_date'].nunique()} ({panel['trade_date'].min()}~{panel['trade_date'].max()})")

si = pd.read_parquet('data/raw/stock_list.parquet')[['ts_code','name','industry']]
panel = panel.merge(si, on='ts_code', how='left')
panel['year'] = panel['trade_date'].dt.year

# ============================================================
# 2. 标签
# ============================================================
print("\n[2/6] 构建标签...")
panel['ret_1d'] = panel.groupby('ts_code')['close'].transform(lambda x: x.pct_change())
panel['fwd_20d_ret'] = panel.groupby('ts_code')['close'].transform(lambda x: x.shift(-20)/x - 1)
panel['fwd_20d_ret'] = panel['fwd_20d_ret'].replace([np.inf,-np.inf], np.nan)
panel['label_rank'] = panel.groupby('trade_date')['fwd_20d_ret'].rank(pct=True)
panel['valid'] = panel['fwd_20d_ret'].notna() & (panel['close'] > 0)

# 全部特征
feat_exclude = {'ts_code','trade_date','name','industry','close','fwd_20d_ret',
                'label_rank','valid','ret_1d','year','市值',
                'overnight_ret','moneyflow_raw','moneyflow_strength',
                'idvol','turnover_bias','revise_up_proxy','margin_proxy',
                'seat_premium','big_order_ratio'}
feature_cols_all = [c for c in panel.columns if c not in feat_exclude and panel[c].dtype in ('float64','int64')]
print(f"  全部特征: {len(feature_cols_all)}")

# ============================================================
# 3. 第一步：先用全部特征训练找Top10特征
# ============================================================
print("\n[3/6] 特征预筛选（用2025+2026全特征训练确定Top10）...")

for c in feature_cols_all:
    panel[c] = panel[c].fillna(panel[c].median())

# 中性化
dates = sorted(panel['trade_date'].unique())
print(f"  中性化 {len(dates)} 个截面...")
for i, dt in enumerate(dates):
    if (i+1)%50 == 0:
        print(f"    {i+1}/{len(dates)}")
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

# 时间分割: 2025=训练, 2026前半=验证
ds = sorted(td['trade_date'].unique())
sp = int(len(ds)*0.7)  # 前70%训练, 后30%验证

model_full = lgb.train({
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

# 取Top10特征
imp_all = pd.DataFrame({'f': feature_cols_all, 'g': model_full.feature_importance(importance_type='gain')})
imp_all = imp_all.sort_values('g', ascending=False)
top10_features = imp_all.head(10)['f'].tolist()

print(f"\n  Top10特征:")
for i, r in imp_all.head(10).iterrows():
    print(f"    {r['f']:25s}: gain={r['g']:.1f}")
print(f"  (保留 {len(top10_features)} 个特征)")

# ============================================================
# 4. 只用Top10特征重新训练
# ============================================================
print("\n[4/6] 只用Top10特征训练...")

# 重新加载（避免中性化残留影响）
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

# 只保留top10特征
for c in top10_features:
    panel2[c] = panel2[c].fillna(panel2[c].median())

# 中性化（仅Top10特征）
dates2 = sorted(panel2['trade_date'].unique())
for i, dt in enumerate(dates2):
    if (i+1)%50 == 0:
        print(f"  中性化: {i+1}/{len(dates2)}")
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
    y_mat = np.column_stack([sub[c].values.astype(float) for c in top10_features])
    good_cols = ~np.any(np.isnan(y_mat) | np.isinf(y_mat), axis=0)
    try:
        beta = np.linalg.lstsq(X, y_mat[:, good_cols], rcond=None)[0]
        res = y_mat[:, good_cols] - X @ beta
        for j, ci in enumerate(np.where(good_cols)[0]):
            panel2.iloc[idx, panel2.columns.get_loc(top10_features[ci])] = res[:, j]
    except:
        pass

# 训练
td2 = panel2[panel2['valid']].copy()
for c in top10_features:
    td2[c] = td2[c].fillna(0.0)

# 时间分割
ds2 = sorted(td2['trade_date'].unique())
n_train = int(len(ds2)*0.7)
train_dates = set(ds2[:n_train])
val_dates = set(ds2[n_train:])

model_final = lgb.train({
    'objective':'regression','metric':'mse','num_leaves':31,'learning_rate':0.05,
    'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,
    'verbosity':-1,'seed':42,'n_jobs':4,
}, lgb.Dataset(
    td2[td2['trade_date'].isin(train_dates)][top10_features].values,
    td2[td2['trade_date'].isin(train_dates)]['label_rank'].values),
   num_boost_round=500,
   valid_sets=[lgb.Dataset(
       td2[td2['trade_date'].isin(val_dates)][top10_features].values,
       td2[td2['trade_date'].isin(val_dates)]['label_rank'].values)],
   callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
print(f"  最佳: {model_final.best_iteration}, MSE: {model_final.best_score['valid_0']['l2']:.6f}")

imp_final = pd.DataFrame({'f': top10_features, 'g': model_final.feature_importance(importance_type='gain')})
imp_final = imp_final.sort_values('g', ascending=False)
print("\n  最终模型特征重要性:")
print(imp_final.to_string(index=False))

model_final.save_model('models/lgb_2026_v8.txt')
print("  ✅ 模型已保存")

# ============================================================
# 5. 回测（每月1个信号日）
# ============================================================
print("\n[5/6] 回测（每月首个交易日调仓）...")

for c in top10_features:
    panel2[c] = panel2[c].fillna(0.0)
panel2['pred'] = model_final.predict(panel2[top10_features].values)
panel2['pred_rk'] = panel2.groupby('trade_date')['pred'].rank(pct=True)

# 每月仅第1个交易日为信号日
unique_dates = pd.DataFrame({'trade_date': sorted(panel2['trade_date'].unique())})
unique_dates['ym'] = unique_dates['trade_date'].dt.to_period('M')
unique_dates['day_rk'] = unique_dates.groupby('ym')['trade_date'].rank(method='dense')
unique_dates['is_sig'] = unique_dates['day_rk'] == 1  # 只有第1天!
sig_dates = set(unique_dates[unique_dates['is_sig']]['trade_date'])
panel2['is_sig'] = panel2['trade_date'].isin(sig_dates)

# 全样本回测
all_dates = sorted(panel2['trade_date'].unique())
pf_h = {}
rets_all = []

for dt in all_dates:
    dd = panel2[panel2['trade_date'] == dt]
    if len(dd) == 0: continue
    if dd['is_sig'].iloc[0]:
        t10 = dd.nlargest(10, 'pred_rk')
        pf_h = {r['ts_code']: 1.0/10 for _, r in t10.iterrows()}
    if pf_h:
        dm = dd.set_index('ts_code')
        pr = sum(w * dm.loc[ts,'ret_1d'] for ts, w in pf_h.items() if ts in dm.index and pd.notna(dm.loc[ts,'ret_1d']))
        rets_all.append(pr)
    else:
        rets_all.append(0)

# ====== 全样本统计 ======
r_all = np.array(rets_all)
sr_all = np.sqrt(252) * r_all.mean() / (r_all.std() + 1e-10)
cum_all = np.cumprod(1 + r_all)
mdd_all = (np.maximum.accumulate(cum_all) - cum_all).max()

print(f"\n  📊 全样本回测 (2025+2026, 月首日调仓)")
print(f"  {'日均收益':>12s}: {r_all.mean()*100:+.4f}%")
print(f"  {'日波动率':>12s}: {r_all.std()*100:.4f}%")
print(f"  {'年化夏普':>12s}: {sr_all:.4f}")
print(f"  {'累计收益':>12s}: {(cum_all[-1]-1)*100:.2f}%")
print(f"  {'最大回撤':>12s}: {mdd_all*100:.2f}%")
print(f"  {'胜率':>12s}: {(r_all>0).sum()/len(r_all)*100:.1f}%")

# ====== 逐年统计 ======
for year in sorted(panel2['year'].unique()):
    yr_mask = panel2['year'] == year
    yr_dates = sorted(panel2[yr_mask]['trade_date'].unique())
    yr_rets = []
    pf_hy = {}
    for dt in yr_dates:
        dd = panel2[panel2['trade_date'] == dt]
        if dd['is_sig'].iloc[0]:
            t10 = dd.nlargest(10, 'pred_rk')
            pf_hy = {r['ts_code']: 1.0/10 for _, r in t10.iterrows()}
        if pf_hy:
            dm = dd.set_index('ts_code')
            pr = sum(w * dm.loc[ts,'ret_1d'] for ts, w in pf_hy.items() if ts in dm.index and pd.notna(dm.loc[ts,'ret_1d']))
            yr_rets.append(pr)
        else:
            yr_rets.append(0)
    if yr_rets:
        yr = np.array(yr_rets)
        yr_sr = np.sqrt(252) * yr.mean() / (yr.std() + 1e-10)
        yr_cum = np.cumprod(1 + yr)
        print(f"  {year}: 日均{yr.mean()*100:+.4f}% | 夏普{yr_sr:.4f} | 累计{(yr_cum[-1]-1)*100:+.2f}% | 胜率{(yr>0).sum()/len(yr)*100:.0f}%")

# ====== 2026单独（与v7对比） ======
panel_26test = panel2[panel2['year'] == 2026]
yr_dates = sorted(panel_26test['trade_date'].unique())
yr_rets = []
pf_hy = {}
for dt in yr_dates:
    dd = panel_26test[panel_26test['trade_date'] == dt]
    if dd['is_sig'].iloc[0]:
        t10 = dd.nlargest(10, 'pred_rk')
        pf_hy = {r['ts_code']: 1.0/10 for _, r in t10.iterrows()}
    if pf_hy:
        dm = dd.set_index('ts_code')
        pr = sum(w * dm.loc[ts,'ret_1d'] for ts, w in pf_hy.items() if ts in dm.index and pd.notna(dm.loc[ts,'ret_1d']))
        yr_rets.append(pr)
    else:
        yr_rets.append(0)
yr = np.array(yr_rets)
yr_sr = np.sqrt(252) * yr.mean() / (yr.std() + 1e-10)
yr_cum = np.cumprod(1 + yr)
print(f"\n  v7对比 (仅2026): 夏普 {yr_sr:.4f} (v7=8.39)")

# ====== 样本外测试（2025训练→2026测试） ======
print("\n  --- 严格样本外 (2025训练→2026测试) ---")

# 用2025前80%数据训练的模型测2026
td25 = td2[td2['trade_date'].astype(str) < '2026-01-01']
td26_test = panel2[panel2['year'] == 2026].copy()

# 训练
ds25 = sorted(td25[td25['valid']]['trade_date'].unique())
sp25 = int(len(ds25)*0.8)
td25_train = td25[td25['trade_date'].isin(set(ds25[:sp25]))]
td25_val = td25[td25['trade_date'].isin(set(ds25[sp25:]))]

model_25only = lgb.train({
    'objective':'regression','metric':'mse','num_leaves':31,'learning_rate':0.05,
    'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,
    'verbosity':-1,'seed':42,'n_jobs':4,
}, lgb.Dataset(td25_train[top10_features].values, td25_train['label_rank'].values),
   num_boost_round=500,
   valid_sets=[lgb.Dataset(td25_val[top10_features].values, td25_val['label_rank'].values)],
   callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

# 预测2026
for c in top10_features:
    td26_test[c] = td26_test[c].fillna(td26_test[c].median())
td26_test['pred'] = model_25only.predict(td26_test[top10_features].values)
td26_test['pred_rk'] = td26_test.groupby('trade_date')['pred'].rank(pct=True)
td26_test['is_sig'] = td26_test['trade_date'].isin(sig_dates)  # 同一个月首日

yr_dates26 = sorted(td26_test['trade_date'].unique())
pf_h26 = {}
rets26 = []
for dt in yr_dates26:
    dd = td26_test[td26_test['trade_date'] == dt]
    if dd['is_sig'].iloc[0]:
        t10 = dd.nlargest(10, 'pred_rk')
        pf_h26 = {r['ts_code']: 1.0/10 for _, r in t10.iterrows()}
    if pf_h26:
        dm = dd.set_index('ts_code')
        pr = sum(w * dm.loc[ts,'ret_1d'] for ts, w in pf_h26.items() if ts in dm.index and pd.notna(dm.loc[ts,'ret_1d']))
        rets26.append(pr)
    else:
        rets26.append(0)

rets26 = np.array(rets26)
sr_26 = np.sqrt(252) * rets26.mean() / (rets26.std() + 1e-10)
cum26 = np.cumprod(1 + rets26)
print(f"  日均收益: {rets26.mean()*100:+.4f}%")
print(f"  年化夏普: {sr_26:.4f} (v7诊断版=4.38)")
print(f"  累计收益: {(cum26[-1]-1)*100:.2f}%")
print(f"  v7诊断(75因子+3天信号, 2025→2026): 夏普=4.38")
print(f"  v8改进(10因子+1天信号, 2025→2026): 夏普={sr_26:.4f}")

# ============================================================
# 6. 最新信号
# ============================================================
print("\n[6/6] 最新信号...")

ld = panel2['trade_date'].max()
lt = panel2[panel2['trade_date'] == ld].sort_values('pred', ascending=False)

pf_orig = pq.ParquetFile('data/factors/factor_panel_v5_final.parquet')
orig = pf_orig.read(columns=['ts_code','trade_date','市值']).to_pandas()
orig_latest = orig[orig['trade_date'] == ld]
orig_map = orig_latest.set_index('ts_code')['市值'].to_dict()

print(f"\n  📋 {ld} Top20:")
tcaps = []
all_caps = []
for i, (_, r) in enumerate(lt.head(20).iterrows()):
    raw_cap = orig_map.get(r['ts_code'], np.nan)
    cap_v = np.exp(raw_cap)/1e8 if pd.notna(raw_cap) else 0
    if pd.notna(raw_cap):
        tcaps.append(cap_v)
    print(f"  {i+1:2d}. {r['ts_code']} {r['name']:<10s} | {r['industry']:<8s} | {cap_v:.0f}亿 | sc={r['pred_rk']:.4f}")

for _, r in lt.iterrows():
    raw_cap = orig_map.get(r['ts_code'], np.nan)
    if pd.notna(raw_cap):
        all_caps.append(np.exp(raw_cap)/1e8)

print(f"\n  市值: Top20中位数={np.median(tcaps):.0f}亿 vs 全市场中位数={np.median(all_caps):.0f}亿")

print("\n" + "=" * 60)
print("✅ 项目B v8 完成")
print("=" * 60)
