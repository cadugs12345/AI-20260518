"""
项目B v3 — 2026年 LGBM + 市值/行业中性化
=============================================
- 从项目A直接拷贝数据
- 对77个因子做市值+行业中性化后再训练
- 避免小市值偏见
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression

print("=" * 60)
print("项目B v3 — 2026年 中性化 LGBM")
print("=" * 60)

# ============================================================
# 1. 加载2026年数据
# ============================================================
print("\n[1/5] 加载2026年数据...")
pf = pq.ParquetFile('data/factors/factor_panel_v5_final.parquet')
schema = pf.schema_arrow.names
use_cols = ['ts_code', 'trade_date', 'close', '市值'] + \
           [c for c in schema if c not in ('ts_code','trade_date','close','ret_1d')]
panel = pf.read(columns=use_cols).to_pandas()
panel = panel[panel['trade_date'].astype(str) >= '2026-01-01'].copy()
panel = panel.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
print(f"  {panel.shape}, {panel['trade_date'].nunique()} 交易日, {panel['ts_code'].nunique()} 只股票")

si = pd.read_parquet('data/raw/stock_list.parquet')[['ts_code', 'name', 'industry']]
panel = panel.merge(si, on='ts_code', how='left')
print(f"  行业: {panel['industry'].nunique()}")

# ============================================================
# 2. 构建标签
# ============================================================
print("\n[2/5] 构建标签...")
panel['fwd_20d_ret'] = panel.groupby('ts_code')['close'].transform(lambda x: x.shift(-20)/x - 1)
panel['fwd_20d_ret'] = panel['fwd_20d_ret'].replace([np.inf, -np.inf], np.nan)
panel['label_rank'] = panel.groupby('trade_date')['fwd_20d_ret'].rank(pct=True)
panel['valid'] = panel['fwd_20d_ret'].notna() & (panel['close'] > 0)
panel['ret_1d'] = panel.groupby('ts_code')['close'].transform(lambda x: x.pct_change())

# 特征列
feat_exclude = {'ts_code','trade_date','name','industry','close','fwd_20d_ret',
                'label_rank','valid','ret_1d','overnight_ret','moneyflow_raw',
                'moneyflow_strength','idvol','turnover_bias','revise_up_proxy',
                'margin_proxy','seat_premium','big_order_ratio'}
feature_cols = [c for c in panel.columns if c not in feat_exclude and panel[c].dtype in ('float64','int64')]
print(f"  原始特征: {len(feature_cols)}")

# ============================================================
# 3. 市值+行业中性化
# ============================================================
print("\n[3/5] 市值+行业中性化（逐截面回归取残差）...")

# 填充nan
for c in feature_cols + ['市值']:
    panel[c] = panel[c].fillna(panel[c].median())

dates = sorted(panel['trade_date'].unique())
n = len(dates)

for i, dt in enumerate(dates):
    if (i+1) % 30 == 0:
        print(f"  进度: {i+1}/{n}")

    mask = panel['trade_date'] == dt
    sub = panel.loc[mask].copy()
    if len(sub) < 100:
        continue

    # 行业dummy
    ind_dummies = pd.get_dummies(sub['industry'])
    ind_dummies = ind_dummies.loc[:, ind_dummies.sum() > 0]

    # 市值（对数）
    cap = np.log(np.maximum(sub['市值'].values, 1e-6)).reshape(-1, 1)

    for fac in feature_cols:
        y = sub[fac].values.astype(float)
        good = ~np.isnan(y)
        if good.sum() < 100:
            continue

        # 只对有效样本回归
        X = np.column_stack([np.ones(good.sum()), cap[good, 0], ind_dummies.values[good]])
        yg = y[good]
        try:
            lr = LinearRegression(fit_intercept=False)
            lr.fit(X, yg)
            residual = yg - lr.predict(X)
            # 回填残差
            panel.loc[mask, fac] = np.nan
            panel.loc[mask.values.nonzero()[0][good], fac] = residual
        except:
            continue

print(f"  中性化完成")

# ============================================================
# 4. 训练 + 回测
# ============================================================
print("\n[4/5] 训练 LightGBM...")

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
print(f"  最佳轮次: {model.best_iteration}, MSE: {model.best_score['valid_0']['l2']:.6f}")

imp = pd.DataFrame({'f': feature_cols, 'g': model.feature_importance(importance_type='gain')})
imp = imp.sort_values('g', ascending=False)
print("\n  Top15 特征:")
print(imp.head(15).to_string(index=False))
model.save_model('models/lgb_2026_v3_neutral.txt')

# ----- 回测 -----
print("\n  精确回测...")
for c in feature_cols:
    panel[c] = panel[c].fillna(0.0)
panel['pred'] = model.predict(panel[feature_cols].values)
panel['pred_rk'] = panel.groupby('trade_date')['pred'].rank(pct=True)
panel['ym'] = panel['trade_date'].dt.to_period('M')
panel['day_rank'] = panel.groupby('ym')['trade_date'].transform('rank')
panel['is_signal'] = panel['day_rank'] <= 3

all_dates = sorted(panel['trade_date'].unique())
pf_hold = {}
rets = []

for dt in all_dates:
    dd = panel[panel['trade_date'] == dt]
    if len(dd) == 0:
        if pf_hold: rets.append(0)
        continue
    if dd['is_signal'].iloc[0]:
        top10 = dd.nlargest(10, 'pred_rk')
        pf_hold = {r['ts_code']: 1/10 for _, r in top10.iterrows()}
    if pf_hold:
        dm = dd.set_index('ts_code')
        pr = sum(w * dm.loc[ts].get('ret_1d', 0) for ts, w in pf_hold.items() if ts in dm.index and pd.notna(dm.loc[ts].get('ret_1d')))
        rets.append(pr)
    else:
        rets.append(0)

rets = np.array(rets)
sr = np.sqrt(252) * rets.mean() / (rets.std() + 1e-10)
cum = np.cumprod(1 + rets)
mdd = (np.maximum.accumulate(cum) - cum).max()

print(f"\n  📊 2026年回测 (中性化 + 月频Top10)")
print(f"  {'日均收益':>12s}: {rets.mean()*100:+.4f}%")
print(f"  {'日波动率':>12s}: {rets.std()*100:.4f}%")
print(f"  {'累计收益':>12s}: {(cum[-1]-1)*100:.2f}%")
print(f"  {'年化夏普':>12s}: {sr:.4f}")
print(f"  {'最大回撤':>12s}: {mdd*100:.2f}%")

# 最新信号
ld = panel['trade_date'].max()
lt = panel[panel['trade_date'] == ld].sort_values('pred', ascending=False)
print(f"\n  📋 {ld} Top20:")
for i, (_, r) in enumerate(lt.head(20).iterrows()):
    cap = np.exp(r['市值']) / 1e8 if pd.notna(r['市值']) else 0
    print(f"  {i+1:2d}. {r['ts_code']} {r['name']:<10s} | {r['industry']:<6s} | {cap:.0f}亿 | sc={r['pred_rk']:.4f}")

# 市值对比
tc = lt.head(20)['市值'].dropna()
ac = panel[panel['trade_date']==ld]['市值'].dropna()
tc_exp = np.exp(tc)/1e8
ac_exp = np.exp(ac)/1e8
print(f"\n  市值对比:")
print(f"    Top20: 中位数={tc_exp.median():.0f}亿, 均值={tc_exp.mean():.0f}亿")
print(f"    全市场: 中位数={ac_exp.median():.0f}亿, 均值={ac_exp.mean():.0f}亿")

print("\n" + "=" * 60)
print("✅ 完成")
print("=" * 60)
