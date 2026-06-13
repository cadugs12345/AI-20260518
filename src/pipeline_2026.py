"""
项目B — 2026年全量因子 LGBM Pipeline
======================================
- 只取2026年数据（截至6/2，97个交易日）
- 使用项目A已有的全部79因子
- 训练 LightGBM rank模型
- 月频调仓回测
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("项目B — 2026年 LGBM Pipeline")
print("=" * 60)

# ============================================================
# 1. 加载2026年数据
# ============================================================
print("\n[1/5] 加载2026年数据...")
panel_path = 'data/factors/factor_panel_v5_final.parquet'

pf = pq.ParquetFile(panel_path)
schema = pf.schema_arrow.names

# 排除无用列
exclude_cols = {'ret_1d'}
use_cols = ['ts_code', 'trade_date'] + [c for c in schema if c not in exclude_cols]

panel = pf.read(columns=use_cols).to_pandas()
panel = panel[panel['trade_date'].astype(str) >= '2026-01-01'].copy()
print(f"  2026年: {panel.shape}, 交易日: {panel['trade_date'].nunique()}, 日均: {len(panel)/panel['trade_date'].nunique():.0f}")

si = pd.read_parquet('data/raw/stock_list.parquet')[['ts_code', 'symbol', 'name', 'industry']]
panel = panel.merge(si[['ts_code', 'industry']], on='ts_code', how='left')
print(f"  行业: {panel['industry'].nunique()}")

# ============================================================
# 2. 构建标签+特征
# ============================================================
print("\n[2/5] 构建标签与特征...")
panel = panel.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
panel['fwd_20d_ret'] = panel.groupby('ts_code')['close'].transform(lambda x: x.shift(-20) / x - 1)
panel['fwd_20d_ret'] = panel['fwd_20d_ret'].replace([np.inf, -np.inf], np.nan)
panel['label_rank'] = panel.groupby('trade_date')['fwd_20d_ret'].rank(pct=True)
panel['valid'] = panel['fwd_20d_ret'].notna() & (panel['close'] > 0)

feat_exclude = {'ts_code', 'trade_date', 'industry', 'close', 'fwd_20d_ret',
                'label_rank', 'valid', 'overnight_ret', 'moneyflow_raw',
                'moneyflow_strength', 'idvol', 'turnover_bias', 'revise_up_proxy',
                'margin_proxy', 'seat_premium'}
feature_cols = [c for c in panel.columns
                if c not in feat_exclude and panel[c].dtype in ('float64', 'int64')]
print(f"  特征: {len(feature_cols)} 个")

# ============================================================
# 3. 训练
# ============================================================
print("\n[3/5] 训练 LightGBM...")
train_data = panel[panel['valid']].copy()
for c in feature_cols:
    train_data[c] = train_data[c].fillna(train_data[c].median())

dates_sorted = sorted(train_data['trade_date'].unique())
split_idx = int(len(dates_sorted) * 0.8)
train_dates = set(dates_sorted[:split_idx])
val_dates = set(dates_sorted[split_idx:])

X_train = train_data[train_data['trade_date'].isin(train_dates)][feature_cols].values
y_train = train_data[train_data['trade_date'].isin(train_dates)]['label_rank'].values
X_val = train_data[train_data['trade_date'].isin(val_dates)][feature_cols].values
y_val = train_data[train_data['trade_date'].isin(val_dates)]['label_rank'].values
print(f"  训练: {X_train.shape}, 验证: {X_val.shape}")

model = lgb.train({
    'objective': 'regression', 'metric': 'mse',
    'num_leaves': 31, 'learning_rate': 0.05,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
    'verbosity': -1, 'seed': 42, 'n_jobs': 4,
}, lgb.Dataset(X_train, y_train), num_boost_round=500,
   valid_sets=[lgb.Dataset(X_val, y_val)],
   callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
print(f"  最佳轮次: {model.best_iteration}, 验证MSE: {model.best_score['valid_0']['l2']:.6f}")

imp = pd.DataFrame({'feature': feature_cols, 'gain': model.feature_importance(importance_type='gain')})
imp = imp.sort_values('gain', ascending=False)
print("\n  Top15 特征:")
print(imp.head(15).to_string(index=False))
model.save_model('models/lgb_2026_v1.txt')
print("  ✅ 已保存")

# ============================================================
# 4. 回测（准确版：用实际 fwd_20d_ret）
# ============================================================
print("\n[4/5] 回测...")

# 全量预测
for c in feature_cols:
    panel[c] = panel[c].fillna(panel[c].median())
panel['pred'] = model.predict(panel[feature_cols].values)
panel['pred_rank'] = panel.groupby('trade_date')['pred'].rank(pct=True)

# 月频调仓
panel['ym'] = panel['trade_date'].dt.to_period('M')
monthly_signal_dates = panel.groupby('ym')['trade_date'].min().reset_index()
monthly_signal_dates.columns = ['ym', 'signal_date']
# 如果ym最小日期不是前3个交易日，用前3个
trade_day_rank = panel.groupby('trade_date').cumcount()
panel['is_signal_day'] = panel.groupby('ym')['trade_date'].transform('rank') <= 3

# 回测（每交易日检查持仓）
dates_bt = sorted(panel[panel['is_signal_day']]['trade_date'].unique())
# 实际用所有日期
all_dates = sorted(panel['trade_date'].unique())

portfolios = {}  # signal_date -> {ts_code: weight}
returns = []
trade_log = []

for dt in all_dates:
    day_data = panel[panel['trade_date'] == dt]

    # 检查是否是信号日
    is_signal = day_data['is_signal_day'].iloc[0] if len(day_data) > 0 else False

    if is_signal:
        # 选Top10
        top10 = day_data.nlargest(10, 'pred_rank')
        total_w = 0
        pw = {}
        for _, r in top10.iterrows():
            w = 1.0/10
            pw[r['ts_code']] = w
            total_w += w
        portfolios = {k: v/total_w for k, v in pw.items()}
        trade_log.append({'date': dt, 'action': 'rebalance', 'holdings': len(portfolios)})

    # 计算当日收益（等调仓日后用实际 close 计算的 1日收益）
    if len(portfolios) > 0:
        day_map = day_data.set_index('ts_code')
        ret = 0
        valid_w = 0
        for ts, w in portfolios.items():
            if ts in day_map.index:
                r = day_map.loc[ts]
                if pd.notna(r.get('fwd_20d_ret')) and r['close'] > 0:
                    ret += w * r['fwd_20d_ret'] / 20
                    valid_w += w
        if valid_w > 0:
            returns.append(ret / valid_w)
        else:
            returns.append(0)
    else:
        returns.append(0)

rets = np.array(returns)
sharpe = np.sqrt(252) * rets.mean() / (rets.std() + 1e-10)
cum = np.cumprod(1 + rets) - 1
max_dd = (np.maximum.accumulate(cum) - cum).max()

print(f"\n  📊 2026年回测 (月频Top10等权)")
print(f"  {'日均收益':>10s}: {rets.mean()*100:.4f}%")
print(f"  {'日波动率':>10s}: {rets.std()*100:.4f}%")
print(f"  {'累计收益':>10s}: {cum[-1]*100:.2f}%")
print(f"  {'年化夏普':>10s}: {sharpe:.4f}")
print(f"  {'最大回撤':>10s}: {max_dd*100:.2f}%")
print(f"  {'交易天数':>10s}: {len(rets)}")

# ============================================================
# 5. 最新信号
# ============================================================
print("\n[5/5] 最新信号...")
latest_date = panel['trade_date'].max()
latest = panel[panel['trade_date'] == latest_date].sort_values('pred', ascending=False)

name_map = si.set_index('ts_code')['name'].to_dict()
ind_map = si.set_index('ts_code')['industry'].to_dict()

print(f"\n  📋 {latest_date} Top20:")
for i, (_, row) in enumerate(latest.head(20).iterrows()):
    n = name_map.get(row['ts_code'], '')
    ind = ind_map.get(row['ts_code'], '')
    cap = row.get('市值', 0)
    print(f"  {i+1:2d}. {row['ts_code']} {n:<10s} | {ind:<8s} | 市值{cap/1e8:.0f}亿 | score={row['pred_rank']:.4f}")

# 统计Top20的市值特征
top_20_cap = latest.head(20)['市值'].dropna()
all_cap = latest['市值'].dropna()
print(f"\n  市值对比:")
print(f"    Top20 中位数: {top_20_cap.median()/1e8:.0f}亿")
print(f"    全市场 中位数: {all_cap.median()/1e8:.0f}亿")
print(f"    Top20 平均: {top_20_cap.mean()/1e8:.0f}亿")
print(f"    全市场 平均: {all_cap.mean()/1e8:.0f}亿")

print("\n" + "=" * 60)
print("✅ 完成")
print("=" * 60)
