"""
项目B v2 — 2026年 LGBM 更精确回测
"""
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("项目B v2 — 2026年精确回测")
print("=" * 60)

# ============================================================
# 1. 加载2026年数据
# ============================================================
print("\n[1/4] 加载数据...")
pf = pq.ParquetFile('data/factors/factor_panel_v5_final.parquet')
schema = pf.schema_arrow.names

# 额外需要原始日K线算精确收益
use_cols = ['ts_code', 'trade_date', 'close', '市值'] + [c for c in schema if c not in ('ts_code','trade_date','close','ret_1d')]
panel = pf.read(columns=use_cols).to_pandas()
panel = panel[panel['trade_date'].astype(str) >= '2026-01-01'].copy()
panel = panel.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
print(f"  {panel.shape}, 交易日: {panel['trade_date'].nunique()}")

si = pd.read_parquet('data/raw/stock_list.parquet')[['ts_code', 'symbol', 'name', 'industry']]
panel = panel.merge(si[['ts_code', 'name', 'industry']], on='ts_code', how='left')
print(f"  行业: {panel['industry'].nunique()}")

# ============================================================
# 2. 构建标签+特征
# ============================================================
print("\n[2/4] 构建标签...")
panel['fwd_20d_ret'] = panel.groupby('ts_code')['close'].transform(lambda x: x.shift(-20) / x - 1)
panel['fwd_20d_ret'] = panel['fwd_20d_ret'].replace([np.inf, -np.inf], np.nan)
panel['label_rank'] = panel.groupby('trade_date')['fwd_20d_ret'].rank(pct=True)
panel['valid'] = panel['fwd_20d_ret'].notna() & (panel['close'] > 0)

# 计算真正每日收益率（用于回测）
panel['ret_1d'] = panel.groupby('ts_code')['close'].transform(lambda x: x.pct_change())

feat_exclude = {'ts_code', 'trade_date', 'name', 'industry', 'close', 'fwd_20d_ret',
                'label_rank', 'valid', 'ret_1d', 'overnight_ret', 'moneyflow_raw',
                'moneyflow_strength', 'idvol', 'turnover_bias', 'revise_up_proxy',
                'margin_proxy', 'seat_premium', 'big_order_ratio'}
feature_cols = [c for c in panel.columns if c not in feat_exclude and panel[c].dtype in ('float64','int64')]
print(f"  特征: {len(feature_cols)}")

# ============================================================
# 3. 训练
# ============================================================
print("\n[3/4] 训练...")
td = panel[panel['valid']].copy()
for c in feature_cols:
    td[c] = td[c].fillna(td[c].median())

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
print("\n  Top15:")
print(imp.sort_values('g', ascending=False).head(15).to_string(index=False))
model.save_model('models/lgb_2026_v2.txt')
print("  ✅ 模型已保存")

# ============================================================
# 4. 精确回测（用每日pct_change）
# ============================================================
print("\n[4/4] 精确回测...")

# 全量预测
for c in feature_cols:
    panel[c] = panel[c].fillna(panel[c].median())
panel['pred'] = model.predict(panel[feature_cols].values)
panel['pred_rank'] = panel.groupby('trade_date')['pred'].rank(pct=True)

# 月频前3天调仓
panel['ym'] = panel['trade_date'].dt.to_period('M')
panel['day_of_month'] = panel.groupby('ym')['trade_date'].transform('rank')
panel['is_signal'] = panel['day_of_month'] <= 3

# 回测
all_dates = sorted(panel['trade_date'].unique())
pf_portfolio = {}
rets = []

for dt in all_dates:
    dd = panel[panel['trade_date'] == dt]
    if len(dd) == 0:
        if pf_portfolio:
            rets.append(0)
        continue

    # 调仓
    if dd['is_signal'].iloc[0]:
        top10 = dd.nlargest(10, 'pred_rank')
        pf_portfolio = {r['ts_code']: 1/10 for _, r in top10.iterrows()}

    # 计算当日实际收益
    if pf_portfolio:
        day_map = dd.set_index('ts_code')
        pr = 0
        vw = 0
        for ts, w in pf_portfolio.items():
            if ts in day_map.index:
                r = day_map.loc[ts].get('ret_1d', np.nan)
                if pd.notna(r):
                    pr += w * r
                    vw += w
        rets.append(pr / vw if vw > 0 else 0)
    else:
        rets.append(0)

rets = np.array(rets)
sr = np.sqrt(252) * rets.mean() / (rets.std() + 1e-10)
cum = np.cumprod(1 + rets)
max_dd = (np.maximum.accumulate(cum) - cum).max()

print(f"\n  📊 精确回测 (月频Top10等权)")
print(f"  {'日均收益':>12s}: {rets.mean()*100:+.4f}%")
print(f"  {'日波动率':>12s}: {rets.std()*100:.4f}%")
print(f"  {'累计收益':>12s}: {(cum[-1]-1)*100:.2f}%")
print(f"  {'年化夏普':>12s}: {sr:.4f}")
print(f"  {'最大回撤':>12s}: {max_dd*100:.2f}%")
print(f"  {'交易天数':>12s}: {len(rets)}")

# 最新信号
ld = panel['trade_date'].max()
latest = panel[panel['trade_date'] == ld].sort_values('pred', ascending=False)
print(f"\n  📋 {ld} Top20:")
for i, (_, r) in enumerate(latest.head(20).iterrows()):
    cap = np.exp(r.get('市值', 0)) / 1e8 if pd.notna(r.get('市值')) else 0
    print(f"  {i+1:2d}. {r['ts_code']} {r['name']:<10s} | {r['industry']:<6s} | {cap:.0f}亿 | sc={r['pred_rank']:.4f}")

tc = latest.head(20)['市值'].dropna()
ac = latest['市值'].dropna()
print(f"\n  市值: Top20 median={np.median(tc):.2f} vs 全市场 median={np.median(ac):.2f}")
print(f"       Top20 mean={np.mean(tc):.2f} vs 全市场 mean={np.mean(ac):.2f}")

print("\n" + "=" * 60)
print("✅ 完成")
print("=" * 60)
