"""
项目B v4 — 2026年 快速中性化 LGBM
==================================
- 只中性化最可能被市值污染的因子
- 用 numpy lstsq 批量处理
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("项目B v4 — 2026年 快速中性化 LGBM")
print("=" * 60)

# ============================================================
# 1. 加载数据
# ============================================================
print("\n[1/5] 加载数据...")
pf = pq.ParquetFile('data/factors/factor_panel_v5_final.parquet')
schema = pf.schema_arrow.names

use_cols = ['ts_code','trade_date','close','市值'] + \
           [c for c in schema if c not in ('ts_code','trade_date','close','ret_1d','overnight_ret','moneyflow_raw','moneyflow_strength','idvol','turnover_bias','revise_up_proxy','margin_proxy','seat_premium','big_order_ratio')]

panel = pf.read(columns=use_cols).to_pandas()
panel = panel[panel['trade_date'].astype(str) >= '2026-01-01'].copy()
panel = panel.sort_values(['ts_code','trade_date']).reset_index(drop=True)
print(f"  {panel.shape}, {panel['trade_date'].nunique()} 交易日")

si = pd.read_parquet('data/raw/stock_list.parquet')[['ts_code','name','industry']]
panel = panel.merge(si, on='ts_code', how='left')

# ============================================================
# 2. 标签
# ============================================================
print("\n[2/5] 标签构建...")
panel['fwd_20d_ret'] = panel.groupby('ts_code')['close'].transform(lambda x: x.shift(-20)/x - 1)
panel['fwd_20d_ret'] = panel['fwd_20d_ret'].replace([np.inf,-np.inf], np.nan)
panel['label_rank'] = panel.groupby('trade_date')['fwd_20d_ret'].rank(pct=True)
panel['valid'] = panel['fwd_20d_ret'].notna() & (panel['close'] > 0)
panel['ret_1d'] = panel.groupby('ts_code')['close'].transform(lambda x: x.pct_change())

# 特征列（仅数值型因子）
feat_exclude = {'ts_code','trade_date','name','industry','close','fwd_20d_ret',
                'label_rank','valid','ret_1d'}
feature_cols = [c for c in panel.columns if c not in feat_exclude and panel[c].dtype in ('float64','int64')]
print(f"  特征: {len(feature_cols)}")

# ============================================================
# 3. 市值+行业中性化（批量numpy）
# ============================================================
print("\n[3/5] 中性化...")

# 先填nan为每列中位数
for c in feature_cols + ['市值']:
    panel[c] = panel[c].fillna(panel[c].median())

dates = sorted(panel['trade_date'].unique())
n_dates = len(dates)

for i, dt in enumerate(dates):
    if (i+1) % 20 == 0:
        print(f"  {i+1}/{n_dates}")

    mask = panel['trade_date'] == dt
    idx = np.where(mask)[0]

    sub = panel.iloc[idx]
    if len(sub) < 100:
        continue

    # 构建哑变量: 行业 + 市值
    inds = sub['industry'].values
    ind_cols = list(set(inds))
    n_ind = len(ind_cols)
    n_row = len(sub)

    cap = np.log(np.maximum(sub['市值'].values, 1e-6))
    cap_z = (cap - cap.mean()) / (cap.std() + 1e-10)

    # 设计矩阵 X: [1, cap_z, ind_dummies]
    ind_dummies = np.zeros((n_row, n_ind))
    for j, ic in enumerate(ind_cols):
        ind_dummies[:, j] = (inds == ic).astype(float)
    X = np.column_stack([np.ones(n_row), cap_z, ind_dummies])

    # 批量对所有因子回归取残差
    y_matrix = np.column_stack([sub[c].values.astype(float) for c in feature_cols])
    # 剔出全nan列
    good_cols = ~np.all(np.isnan(y_matrix), axis=0)

    try:
        beta = np.linalg.lstsq(X, y_matrix[:, good_cols], rcond=None)[0]
        residual = y_matrix[:, good_cols] - X @ beta
        for j, col_idx in enumerate(np.where(good_cols)[0]):
            fc = feature_cols[col_idx]
            panel.iloc[idx, panel.columns.get_loc(fc)] = residual[:, j]
    except:
        pass

print(f"  {n_dates}/{n_dates} 完成")

# ============================================================
# 4. 训练+回测
# ============================================================
print("\n[4/5] 训练...")

td = panel[panel['valid']].copy()
for c in feature_cols:
    td[c] = td[c].fillna(0.0)

ds = sorted(td['trade_date'].unique())
sp = int(len(ds)*0.8)
Xd = td[td['trade_date'].isin(set(ds[:sp]))][feature_cols].values
yd = td[td['trade_date'].isin(set(ds[:sp]))]['label_rank'].values
Xv = td[td['trade_date'].isin(set(ds[sp:]))][feature_cols].values
yv = td[td['trade_date'].isin(set(ds[sp:]))]['label_rank'].values

model = lgb.train({
    'objective':'regression','metric':'mse','num_leaves':31,'learning_rate':0.05,
    'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,
    'verbosity':-1,'seed':42,'n_jobs':4,
}, lgb.Dataset(Xd, yd), num_boost_round=500,
   valid_sets=[lgb.Dataset(Xv, yv)],
   callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
print(f"  最佳: {model.best_iteration}, MSE: {model.best_score['valid_0']['l2']:.6f}")

imp = pd.DataFrame({'f': feature_cols, 'g': model.feature_importance(importance_type='gain')})
imp = imp.sort_values('g', ascending=False)
print("\n  Top15:")
print(imp.head(15).to_string(index=False))
model.save_model('models/lgb_2026_v4.txt')

# 回测
for c in feature_cols:
    panel[c] = panel[c].fillna(0.0)
panel['pred'] = model.predict(panel[feature_cols].values)
panel['pred_rk'] = panel.groupby('trade_date')['pred'].rank(pct=True)
panel['ym'] = panel['trade_date'].dt.to_period('M')
panel['day_rk'] = panel.groupby('ym')['trade_date'].transform('rank')
panel['is_sig'] = panel['day_rk'] <= 3

all_dates = sorted(panel['trade_date'].unique())
pf_h = {}
rets = []

for dt in all_dates:
    dd = panel[panel['trade_date'] == dt]
    if len(dd) == 0: continue
    if dd['is_sig'].iloc[0]:
        t10 = dd.nlargest(10, 'pred_rk')
        pf_h = {r['ts_code']: 1/10 for _, r in t10.iterrows()}
    if pf_h:
        dm = dd.set_index('ts_code')
        pr = sum(w * dm.loc[ts, 'ret_1d'] for ts, w in pf_h.items() if ts in dm.index and pd.notna(dm.loc[ts,'ret_1d']))
        rets.append(pr)
    else:
        rets.append(0)

rets = np.array(rets)
sr = np.sqrt(252) * rets.mean() / (rets.std() + 1e-10)
cum = np.cumprod(1 + rets)
mdd = (np.maximum.accumulate(cum) - cum).max()

print(f"\n  📊 回测:")
print(f"   日均收益: {rets.mean()*100:+.4f}%")
print(f"   日波动率: {rets.std()*100:.4f}%")
print(f"   累计收益: {(cum[-1]-1)*100:.2f}%")
print(f"   年化夏普: {sr:.4f}")
print(f"   最大回撤: {mdd*100:.2f}%")

# 最新信号
ld = panel['trade_date'].max()
lt = panel[panel['trade_date'] == ld].sort_values('pred', ascending=False)
print(f"\n  📋 {ld} Top20:")
ac = panel[panel['trade_date']==ld]['市值']
for i, (_, r) in enumerate(lt.head(20).iterrows()):
    cap_v = np.exp(r['市值'])/1e8
    print(f"  {i+1:2d}. {r['ts_code']} {r['name']:<10s} | {r['industry']:<6s} | {cap_v:.0f}亿 | sc={r['pred_rk']:.4f}")

tc = lt.head(20)['市值']
print(f"\n  市值: Top20={np.median(np.exp(tc)/1e8):.0f}亿 vs 全市场={np.median(np.exp(ac)/1e8):.0f}亿")

print("\n" + "=" * 60)
print("✅")
print("=" * 60)
