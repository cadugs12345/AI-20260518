"""
项目B — 过拟合诊断
==================
检查 v7 模型的各种过拟合迹象
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("项目B — 过拟合诊断")
print("=" * 60)

# ============================================================
# 0. 加载数据（一次性）
# ============================================================
pf = pq.ParquetFile('data/factors/factor_panel_v5_final.parquet')
schema = pf.schema_arrow.names
use_cols = ['ts_code','trade_date','close','市值'] + \
           [c for c in schema if c not in ('ts_code','trade_date','close','ret_1d')]
panel = pf.read(columns=use_cols).to_pandas()
panel = panel[panel['trade_date'].astype(str) >= '2025-01-01'].copy()  # 需要2025数据
panel = panel.sort_values(['ts_code','trade_date']).reset_index(drop=True)
print(f"面板: {panel.shape}")

si = pd.read_parquet('data/raw/stock_list.parquet')[['ts_code','name','industry']]
panel = panel.merge(si, on='ts_code', how='left')

panel['ret_1d'] = panel.groupby('ts_code')['close'].transform(lambda x: x.pct_change())
panel['fwd_20d_ret'] = panel.groupby('ts_code')['close'].transform(lambda x: x.shift(-20)/x - 1)
panel['fwd_20d_ret'] = panel['fwd_20d_ret'].replace([np.inf,-np.inf], np.nan)
panel['label_rank'] = panel.groupby('trade_date')['fwd_20d_ret'].rank(pct=True)
panel['valid'] = panel['fwd_20d_ret'].notna() & (panel['close'] > 0)
panel['year'] = panel['trade_date'].dt.year

feat_exclude = {'ts_code','trade_date','name','industry','close','fwd_20d_ret',
                'label_rank','valid','ret_1d','year','市值',
                'overnight_ret','moneyflow_raw','moneyflow_strength',
                'idvol','turnover_bias','revise_up_proxy','margin_proxy',
                'seat_premium','big_order_ratio'}
feature_cols = [c for c in panel.columns if c not in feat_exclude and panel[c].dtype in ('float64','int64')]
print(f"特征: {len(feature_cols)}")

# ============================================================
# 1. 检查1: 不同 random_seed 的稳定性
# ============================================================
print("\n" + "=" * 50)
print("检查1: Random Seed 稳定性")
print("=" * 50)

# 简单快速：只用2026年数据，不同seed
panel_2026 = panel[panel['year'] == 2026].copy()
for c in feature_cols:
    panel_2026[c] = panel_2026[c].fillna(panel_2026[c].median())

# 中性化
dates = sorted(panel_2026['trade_date'].unique())
for c in feature_cols:
    feat_neut = np.full(len(panel_2026), np.nan)

for i, dt in enumerate(dates):
    mask = panel_2026['trade_date'] == dt
    idx = np.where(mask)[0]
    sub = panel_2026.iloc[idx]
    if len(sub) < 100:
        for c in feature_cols:
            pass  # simplified
        continue
    cap = np.log(np.maximum(sub['市值'].values, 1e-6))
    cap_z = (cap - cap.mean()) / (cap.std() + 1e-10)
    inds = sub['industry'].values
    ind_vals = sorted(set(v for v in inds if isinstance(v, str)))
    ind_arr = np.zeros((len(sub), len(ind_vals)))
    for j, iv in enumerate(ind_vals):
        ind_arr[:, j] = (inds == iv).astype(float)
    X = np.column_stack([np.ones(len(sub)), cap_z, ind_arr])
    y_mat = np.column_stack([sub[c].values.astype(float) for c in feature_cols])
    good_cols = ~np.any(np.isnan(y_mat) | np.isinf(y_mat), axis=0)
    try:
        beta = np.linalg.lstsq(X, y_mat[:, good_cols], rcond=None)[0]
        res = y_mat[:, good_cols] - X @ beta
        for j, ci in enumerate(np.where(good_cols)[0]):
            panel_2026.iloc[idx, panel_2026.columns.get_loc(feature_cols[ci])] = res[:, j]
    except:
        pass

td = panel_2026[panel_2026['valid']].copy()
for c in feature_cols:
    td[c] = td[c].fillna(0.0)
ds = sorted(td['trade_date'].unique())
sp = int(len(ds)*0.8)

# 不同seed训练
results_seed = {}
for seed in [42, 123, 456, 789, 999]:
    model = lgb.train({
        'objective':'regression','metric':'mse','num_leaves':31,'learning_rate':0.05,
        'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,
        'verbosity':-1,'seed':seed,'n_jobs':4,
    }, lgb.Dataset(
        td[td['trade_date'].isin(set(ds[:sp]))][feature_cols].values,
        td[td['trade_date'].isin(set(ds[:sp]))]['label_rank'].values),
       num_boost_round=500,
       valid_sets=[lgb.Dataset(
           td[td['trade_date'].isin(set(ds[sp:]))][feature_cols].values,
           td[td['trade_date'].isin(set(ds[sp:]))]['label_rank'].values)],
       callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

    # 回测
    for c in feature_cols:
        panel_2026[c] = panel_2026[c].fillna(0.0)
    panel_2026['pred'] = model.predict(panel_2026[feature_cols].values)
    panel_2026['pred_rk'] = panel_2026.groupby('trade_date')['pred'].rank(pct=True)

    ud = pd.DataFrame({'trade_date': sorted(panel_2026['trade_date'].unique())})
    ud['ym'] = ud['trade_date'].dt.to_period('M')
    ud['day_rk'] = ud.groupby('ym')['trade_date'].rank(method='dense')
    ud['is_sig'] = ud['day_rk'] <= 3
    sig_dates = set(ud[ud['is_sig']]['trade_date'])
    panel_2026['is_sig'] = panel_2026['trade_date'].isin(sig_dates)

    pf_h = {}
    rets = []
    for dt in sorted(panel_2026['trade_date'].unique()):
        dd = panel_2026[panel_2026['trade_date'] == dt]
        if len(dd) == 0: continue
        if dd['is_sig'].iloc[0]:
            t10 = dd.nlargest(10, 'pred_rk')
            pf_h = {r['ts_code']: 1.0/10 for _, r in t10.iterrows()}
        if pf_h:
            dm = dd.set_index('ts_code')
            pr = sum(w * dm.loc[ts,'ret_1d'] for ts, w in pf_h.items() if ts in dm.index and pd.notna(dm.loc[ts,'ret_1d']))
            rets.append(pr)
        else:
            rets.append(0)
    rets = np.array(rets)
    sr = np.sqrt(252) * rets.mean() / (rets.std() + 1e-10)
    n_best = model.best_iteration
    train_mse = model.best_score['training']['l2'] if 'training' in model.best_score else None
    val_mse = model.best_score['valid_0']['l2']
    overlap = []
    if seed == 42:
        base_top10 = set(t10['ts_code'].values) if 't10' in dir() else set()
    results_seed[seed] = {'sharpe': sr, 'best_iter': n_best, 'val_mse': val_mse}

for seed, res in results_seed.items():
    print(f"  seed={seed:3d}: 夏普={res['sharpe']:.4f}, 最佳轮次={res['best_iter']}, 验证MSE={res['val_mse']:.6f}")

sharpe_list = [res['sharpe'] for res in results_seed.values()]
print(f"\n  夏普均值: {np.mean(sharpe_list):.4f}, 标准差: {np.std(sharpe_list):.4f}")
print(f"  夏普CV: {np.std(sharpe_list)/np.mean(sharpe_list)*100:.1f}%")

# ============================================================
# 2. 检查2: 2025年训练, 2026年测试（严格样本外）
# ============================================================
print("\n" + "=" * 50)
print("检查2: 2025训练 → 2026测试（严格样本外）")
print("=" * 50)

panel_2025 = panel[panel['year'] == 2025].copy()
for c in feature_cols:
    panel_2025[c] = panel_2025[c].fillna(panel_2025[c].median())

# 中性化
dates_2025 = sorted(panel_2025['trade_date'].unique())
for i, dt in enumerate(dates_2025):
    mask = panel_2025['trade_date'] == dt
    idx = np.where(mask)[0]
    sub = panel_2025.iloc[idx]
    if len(sub) < 100: continue
    cap = np.log(np.maximum(sub['市值'].values, 1e-6))
    cap_z = (cap - cap.mean()) / (cap.std() + 1e-10)
    inds = sub['industry'].values
    ind_vals = sorted(set(v for v in inds if isinstance(v, str)))
    ind_arr = np.zeros((len(sub), len(ind_vals)))
    for j, iv in enumerate(ind_vals):
        ind_arr[:, j] = (inds == iv).astype(float)
    X = np.column_stack([np.ones(len(sub)), cap_z, ind_arr])
    y_mat = np.column_stack([sub[c].values.astype(float) for c in feature_cols])
    good_cols = ~np.any(np.isnan(y_mat) | np.isinf(y_mat), axis=0)
    try:
        beta = np.linalg.lstsq(X, y_mat[:, good_cols], rcond=None)[0]
        res = y_mat[:, good_cols] - X @ beta
        for j, ci in enumerate(np.where(good_cols)[0]):
            panel_2025.iloc[idx, panel_2025.columns.get_loc(feature_cols[ci])] = res[:, j]
    except:
        pass

td25 = panel_2025[panel_2025['valid']].copy()
for c in feature_cols:
    td25[c] = td25[c].fillna(0.0)

print(f"  2025训练数据: {td25['trade_date'].nunique()}天, {len(td25)}行")

# 用2025训练
model_25 = lgb.train({
    'objective':'regression','metric':'mse','num_leaves':31,'learning_rate':0.05,
    'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,
    'verbosity':-1,'seed':42,'n_jobs':4,
}, lgb.Dataset(td25[feature_cols].values, td25['label_rank'].values),
   num_boost_round=200, callbacks=[lgb.log_evaluation(0)])
print(f"  2025模型训练完成, 轮次=200")

# 对2026年预测+回测
panel_26_test = panel[panel['year'] == 2026].copy()
for c in feature_cols:
    panel_26_test[c] = panel_26_test[c].fillna(panel_26_test[c].median())

panel_26_test['pred'] = model_25.predict(panel_26_test[feature_cols].values)
panel_26_test['pred_rk'] = panel_26_test.groupby('trade_date')['pred'].rank(pct=True)

ud = pd.DataFrame({'trade_date': sorted(panel_26_test['trade_date'].unique())})
ud['ym'] = ud['trade_date'].dt.to_period('M')
ud['day_rk'] = ud.groupby('ym')['trade_date'].rank(method='dense')
ud['is_sig'] = ud['day_rk'] <= 3
sig_dates = set(ud[ud['is_sig']]['trade_date'])
panel_26_test['is_sig'] = panel_26_test['trade_date'].isin(sig_dates)

pf_h = {}
rets = []
for dt in sorted(panel_26_test['trade_date'].unique()):
    dd = panel_26_test[panel_26_test['trade_date'] == dt]
    if len(dd) == 0: continue
    if dd['is_sig'].iloc[0]:
        t10 = dd.nlargest(10, 'pred_rk')
        pf_h = {r['ts_code']: 1.0/10 for _, r in t10.iterrows()}
    if pf_h:
        dm = dd.set_index('ts_code')
        pr = sum(w * dm.loc[ts,'ret_1d'] for ts, w in pf_h.items() if ts in dm.index and pd.notna(dm.loc[ts,'ret_1d']))
        rets.append(pr)
    else:
        rets.append(0)
rets = np.array(rets)
sr_25_26 = np.sqrt(252) * rets.mean() / (rets.std() + 1e-10)
cum = np.cumprod(1 + rets)
mdd = (np.maximum.accumulate(cum) - cum).max()

print(f"\n  📊 2025年训练→2026年回测（严格样本外）")
print(f"  {'日均收益':>12s}: {rets.mean()*100:+.4f}%")
print(f"  {'年化夏普':>12s}: {sr_25_26:.4f}")
print(f"  {'累计收益':>12s}: {(cum[-1]-1)*100:.2f}%")
print(f"  {'最大回撤':>12s}: {mdd*100:.2f}%")
print(f"  {'胜率':>12s}: {(rets>0).sum()/len(rets)*100:.1f}%")

# ============================================================
# 3. 检查3: 持仓换手率
# ============================================================
print("\n" + "=" * 50)
print("检查3: 持仓换手率（月度调仓的重叠度）")
print("=" * 50)

# 用panel_26_test（有预测）或重新跑
for c in feature_cols:
    panel_2026[c] = panel_2026[c].fillna(panel_2026[c].median())
panel_2026['pred'] = model.predict(panel_2026[feature_cols].values)
panel_2026['pred_rk'] = panel_2026.groupby('trade_date')['pred'].rank(pct=True)

monthly_holdings = {}
ud2 = pd.DataFrame({'trade_date': sorted(panel_2026['trade_date'].unique())})
ud2['ym'] = ud2['trade_date'].dt.to_period('M')
ud2['day_rk'] = ud2.groupby('ym')['trade_date'].rank(method='dense')
ud2['is_sig'] = ud2['day_rk'] <= 3
sig_dates2 = set(ud2[ud2['is_sig']]['trade_date'])

for dt in sorted(sig_dates2):
    dd = panel_2026[panel_2026['trade_date'] == dt]
    t10 = dd.nlargest(10, 'pred_rk')
    monthly_holdings[dt] = set(t10['ts_code'].values)
    print(f"  {dt}: {list(t10['ts_code'].values)}")

print(f"\n  信号日: {len(monthly_holdings)}")
dates_list = sorted(monthly_holdings.keys())
overlaps = []
for i in range(1, len(dates_list)):
    prev = monthly_holdings[dates_list[i-1]]
    curr = monthly_holdings[dates_list[i]]
    jaccard = len(prev & curr) / len(prev | curr)
    overlaps.append(jaccard)
if overlaps:
    print(f"  月度Jaccard相似度: 均值={np.mean(overlaps):.3f}, 范围=[{min(overlaps):.3f},{max(overlaps):.3f}]")
    print(f"  解释: 0=完全不同, >0.3=有重叠, >0.5=高度重叠")

# ============================================================
# 4. 检查4: 简单基准对比
# ============================================================
print("\n" + "=" * 50)
print("检查4: 基准对比")
print("=" * 50)

# 等权市场平均收益
all_rets = panel_2026.groupby('trade_date')['ret_1d'].mean()
print(f"  全市场等权日均收益: {all_rets.mean()*100:+.4f}%")
print(f"  全市场夏普: {np.sqrt(252) * all_rets.mean() / (all_rets.std() + 1e-10):.4f}")

# 简单因子Top10对比（只用换手率或波动率）
for simple_fac in ['换手率', 'BP', '超跌信号']:
    if simple_fac in panel_2026.columns:
        panel_2026['_simple_pred'] = panel_2026.groupby('trade_date')[simple_fac].rank(pct=True)
        pf_h = {}
        simple_rets = []
        for dt in sorted(panel_2026['trade_date'].unique()):
            dd = panel_2026[panel_2026['trade_date'] == dt]
            if dd['is_sig'].iloc[0]:
                t10 = dd.nlargest(10, '_simple_pred')
                pf_h = {r['ts_code']: 1.0/10 for _, r in t10.iterrows()}
            if pf_h:
                dm = dd.set_index('ts_code')
                pr = sum(w * dm.loc[ts,'ret_1d'] for ts, w in pf_h.items() if ts in dm.index and pd.notna(dm.loc[ts,'ret_1d']))
                simple_rets.append(pr)
            else:
                simple_rets.append(0)
        if simple_rets:
            sr_s = np.sqrt(252) * np.mean(simple_rets) / (np.std(simple_rets) + 1e-10)
            cum_s = np.cumprod(1 + np.array(simple_rets))
            print(f"  单因子[{simple_fac}]: 夏普={sr_s:.4f}, 累计={(cum_s[-1]-1)*100:.2f}%")

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)
