"""
项目B v5 — 2026年 中性化 LGBM + 精确回测
=========================================
- 中性化不污染原始数据（copy模式）
- 精确回测用 ret_1d
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("项目B v5 — 2026年 中性化 LGBM + 精确回测")
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
print(f"  面板: {panel.shape}, {panel['trade_date'].nunique()}日")

si = pd.read_parquet('data/raw/stock_list.parquet')[['ts_code','name','industry']]
panel = panel.merge(si, on='ts_code', how='left')

# ret_1d（用于回测，不参与中性化/训练）
panel['ret_1d'] = panel.groupby('ts_code')['close'].transform(lambda x: x.pct_change())
panel['fwd_20d_ret'] = panel.groupby('ts_code')['close'].transform(lambda x: x.shift(-20)/x - 1)
panel['fwd_20d_ret'] = panel['fwd_20d_ret'].replace([np.inf,-np.inf], np.nan)
panel['label_rank'] = panel.groupby('trade_date')['fwd_20d_ret'].rank(pct=True)
panel['valid'] = panel['fwd_20d_ret'].notna() & (panel['close'] > 0)

# 特征列（所有数值型因子）
feat_exclude = {'ts_code','trade_date','name','industry','close','fwd_20d_ret',
                'label_rank','valid','ret_1d','市值',
                'overnight_ret','moneyflow_raw','moneyflow_strength',
                'idvol','turnover_bias','revise_up_proxy','margin_proxy',
                'seat_premium','big_order_ratio'}
feature_cols = [c for c in panel.columns if c not in feat_exclude and panel[c].dtype in ('float64','int64')]
print(f"  特征: {len(feature_cols)}个")

# ============================================================
# 2. 中性化（copy模式，不污染panel）
# ============================================================
print("\n[2/5] 市值+行业中性化...")

# 先填nan
for c in feature_cols:
    panel[c] = panel[c].fillna(panel[c].median())

# 创建中性化后的特征副本
feat_neutral = {}
for c in feature_cols:
    feat_neutral[c] = np.full(len(panel), np.nan)

dates = sorted(panel['trade_date'].unique())
n_dates = len(dates)

for i, dt in enumerate(dates):
    if (i+1) % 20 == 0:
        print(f"  {i+1}/{n_dates}")

    mask = panel['trade_date'] == dt
    idx = np.where(mask)[0]
    sub = panel.iloc[idx]

    if len(sub) < 100:
        for c in feature_cols:
            feat_neutral[c][idx] = sub[c].values
        continue

    # 设计矩阵
    cap = np.log(np.maximum(sub['市值'].values, 1e-6))
    cap_z = (cap - cap.mean()) / (cap.std() + 1e-10)

    inds = sub['industry'].values
    ind_vals = list(set(inds))
    ind_arr = np.zeros((len(sub), len(ind_vals)))
    for j, iv in enumerate(ind_vals):
        ind_arr[:, j] = (inds == iv).astype(float)

    X = np.column_stack([np.ones(len(sub)), cap_z, ind_arr])

    # 对所有特征做批量回归
    y_mat = np.column_stack([sub[c].values.astype(float) for c in feature_cols])
    good_cols = ~np.any(np.isnan(y_mat) | np.isinf(y_mat), axis=0)

    try:
        beta = np.linalg.lstsq(X, y_mat[:, good_cols], rcond=None)[0]
        residual = y_mat[:, good_cols] - X @ beta
        col_idx = np.where(good_cols)[0]
        for j, ci in enumerate(col_idx):
            c = feature_cols[ci]
            feat_neutral[c][idx] = residual[:, j]
    except:
        for c in feature_cols:
            feat_neutral[c][idx] = sub[c].values

# 把中性化后的特征放回panel（替换原始值，但ret_1d/close/市值不受影响）
for c in feature_cols:
    panel[c] = feat_neutral[c]

print(f"  {n_dates}/{n_dates} 完成")

# ============================================================
# 3. 训练
# ============================================================
print("\n[3/5] 训练...")
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
model.save_model('models/lgb_2026_v5.txt')

# ============================================================
# 4. 回测（精确版）
# ============================================================
print("\n[4/5] 回测...")

# 全量预测
for c in feature_cols:
    panel[c] = panel[c].fillna(0.0)
panel['pred'] = model.predict(panel[feature_cols].values)
panel['pred_rk'] = panel.groupby('trade_date')['pred'].rank(pct=True)

# 月频信号
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

    # 调仓
    if dd['is_sig'].iloc[0]:
        t10 = dd.nlargest(10, 'pred_rk')
        pf_h = {r['ts_code']: 1.0/10 for _, r in t10.iterrows()}

    # 持仓收益
    if pf_h:
        dm = dd.set_index('ts_code')
        pr = 0.0
        n_valid = 0
        for ts, w in pf_h.items():
            if ts in dm.index:
                r = dm.loc[ts, 'ret_1d']
                if pd.notna(r):
                    pr += w * r
                    n_valid += 1
        rets.append(pr if n_valid > 0 else 0.0)
    else:
        rets.append(0.0)

rets = np.array(rets)
sr = np.sqrt(252) * rets.mean() / (rets.std() + 1e-10)
cum = np.cumprod(1 + rets)
mdd = (np.maximum.accumulate(cum) - cum).max()

print(f"\n  📊 2026年 回测结果（精确ret_1d）")
print(f"  {'日均收益':>12s}: {rets.mean()*100:+.6f}%")
print(f"  {'日波动率':>12s}: {rets.std()*100:.6f}%")
print(f"  {'累计收益':>12s}: {(cum[-1]-1)*100:.2f}%")
print(f"  {'年化夏普':>12s}: {sr:.4f}")
print(f"  {'最大回撤':>12s}: {mdd*100:.2f}%")
print(f"  {'交易天数':>12s}: {len(rets)}")

# 分段收益
print("\n  分段收益:")
panel['ym_str'] = panel['ym'].astype(str)
for ym in sorted(panel['ym_str'].unique()):
    pm = panel[panel['ym_str'] == ym]
    pfm = pf_h
    mrets = []
    for dt in sorted(pm['trade_date'].unique()):
        dd = pm[pm['trade_date'] == dt]
        if dd['is_sig'].iloc[0]:
            t10 = dd.nlargest(10, 'pred_rk')
            pfm = {r['ts_code']: 1.0/10 for _, r in t10.iterrows()}
        if pfm:
            dm = dd.set_index('ts_code')
            pr = sum(w * dm.loc[ts, 'ret_1d'] for ts, w in pfm.items() if ts in dm.index and pd.notna(dm.loc[ts,'ret_1d']))
            mrets.append(pr)
    if mrets:
        mr = np.array(mrets)
        m_cum = np.cumprod(1 + mr)
        print(f"    {ym}: {mr.mean()*100:+.4f}%/d, 累计{(m_cum[-1]-1)*100:+.2f}%")

# ============================================================
# 5. 最新信号 + 市值对比
# ============================================================
print("\n[5/5] 最新信号...")

ld = panel['trade_date'].max()
lt = panel[panel['trade_date'] == ld].sort_values('pred', ascending=False)

# 从原始面板读真实市值
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
print(f"    Top20: 中位数={np.median(tcaps):.0f}亿, 均值={np.mean(tcaps):.0f}亿")
print(f"    全市场: 中位数={np.median(all_caps):.0f}亿, 均值={np.mean(all_caps):.0f}亿")

print("\n" + "=" * 60)
print("✅ 项目B v5 完成")
print("=" * 60)
