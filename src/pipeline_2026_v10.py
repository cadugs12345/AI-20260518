"""
项目B v10 — 加入摩擦成本
=========================
费用:
- 手续费: 万3 (0.03%) 买入+卖出各一次
- 滑点: 千1 (0.1%) 买入卖出各一次
- 调仓时统一扣除: 买入0.13% + 卖出0.13% = 0.26% 双边
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("项目B v10 — 含摩擦成本回测")
print("=" * 60)

# 成本参数
COMMISSION = 0.0003      # 万3
SLIPPAGE = 0.001         # 千1
TRADE_COST = COMMISSION + SLIPPAGE  # 单边成本
ROUND_TRIP = TRADE_COST * 2         # 双边成本 0.26%

# ============================================================
# 1. 加载数据（用v9的Top15特征）
# ============================================================
print("\n[1/4] 加载数据...")
pf = pq.ParquetFile('data/factors/factor_panel_v5_final.parquet')
schema = pf.schema_arrow.names
use_cols = ['ts_code','trade_date','close','市值'] + \
           [c for c in schema if c not in ('ts_code','trade_date','close','ret_1d')]
panel = pf.read(columns=use_cols).to_pandas()
panel = panel[panel['trade_date'].astype(str) >= '2025-01-01'].copy()
panel = panel.sort_values(['ts_code','trade_date']).reset_index(drop=True)

si = pd.read_parquet('data/raw/stock_list.parquet')[['ts_code','name','industry']]
panel = panel.merge(si, on='ts_code', how='left')
panel['year'] = panel['trade_date'].dt.year
panel['ret_1d'] = panel.groupby('ts_code')['close'].transform(lambda x: x.pct_change())
panel['fwd_20d_ret'] = panel.groupby('ts_code')['close'].transform(lambda x: x.shift(-20)/x - 1)
panel['fwd_20d_ret'] = panel['fwd_20d_ret'].replace([np.inf,-np.inf], np.nan)
panel['label_rank'] = panel.groupby('trade_date')['fwd_20d_ret'].rank(pct=True)
panel['valid'] = panel['fwd_20d_ret'].notna() & (panel['close'] > 0)

# 取v9 Top15特征
top15_features = ['换手率','repair_force_10d','超跌信号','高波反转','120日动量',
                  'signal_量比3','board_break_pct','limit_up_quality','流通市值',
                  'board_break_depth','ATR比率','CCI','Amihud非流动性','EMA20偏离','净利率']

print(f"  面板: {panel.shape}, 特征: {len(top15_features)}")

# ============================================================
# 2. 中性化
# ============================================================
print("\n[2/4] 中性化...")
for c in top15_features:
    panel[c] = panel[c].fillna(panel[c].median())

dates = sorted(panel['trade_date'].unique())
for i, dt in enumerate(dates):
    if (i+1)%50 == 0: print(f"  {i+1}/{len(dates)}")
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
    y_mat = np.column_stack([sub[c].values.astype(float) for c in top15_features])
    good_cols = ~np.any(np.isnan(y_mat) | np.isinf(y_mat), axis=0)
    try:
        beta = np.linalg.lstsq(X, y_mat[:, good_cols], rcond=None)[0]
        res = y_mat[:, good_cols] - X @ beta
        for j, ci in enumerate(np.where(good_cols)[0]):
            panel.iloc[idx, panel.columns.get_loc(top15_features[ci])] = res[:, j]
    except:
        pass

# ============================================================
# 3. 训练
# ============================================================
print("\n[3/4] 训练模型...")
td = panel[panel['valid']].copy()
for c in top15_features:
    td[c] = td[c].fillna(0.0)
ds = sorted(td['trade_date'].unique())
n_train = int(len(ds)*0.7)
train_dates = set(ds[:n_train])
val_dates = set(ds[n_train:])

model = lgb.train({
    'objective':'regression','metric':'mse','num_leaves':31,'learning_rate':0.05,
    'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,
    'verbosity':-1,'seed':42,'n_jobs':4,
}, lgb.Dataset(
    td[td['trade_date'].isin(train_dates)][top15_features].values,
    td[td['trade_date'].isin(train_dates)]['label_rank'].values),
   num_boost_round=500,
   valid_sets=[lgb.Dataset(
       td[td['trade_date'].isin(val_dates)][top15_features].values,
       td[td['trade_date'].isin(val_dates)]['label_rank'].values)],
   callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
print(f"  最佳: {model.best_iteration}, MSE: {model.best_score['valid_0']['l2']:.6f}")
model.save_model('models/lgb_2026_v10.txt')

# 预测
for c in top15_features:
    panel[c] = panel[c].fillna(0.0)
panel['pred'] = model.predict(panel[top15_features].values)
panel['pred_rk'] = panel.groupby('trade_date')['pred'].rank(pct=True)

# 信号日
ud = pd.DataFrame({'trade_date': sorted(panel['trade_date'].unique())})
ud['ym'] = ud['trade_date'].dt.to_period('M')
ud['day_rk'] = ud.groupby('ym')['trade_date'].rank(method='dense')
ud['is_sig'] = ud['day_rk'] == 1
sig_dates = set(ud[ud['is_sig']]['trade_date'])
panel['is_sig'] = panel['trade_date'].isin(sig_dates)

# ============================================================
# 4. 回测（含摩擦成本）
# ============================================================
print("\n[4/4] 含摩擦成本回测...")

STOP_LOSS = 0.07

def run_backtest_with_cost(df, use_weighted=True, stop_loss=None, cost_per_trade=0):
    """
    含摩擦成本回测
    cost_per_trade: 单边成本比例 (如0.0013 = 万3手续费+千1滑点)
    """
    all_dates = sorted(df['trade_date'].unique())
    pf = {}        # ts_code -> weight
    prev_holdings = set()
    port_value = 1.0  # 初始1元
    daily_values = [1.0]
    rets = []

    for dt in all_dates:
        dd = df[df['trade_date'] == dt]
        if len(dd) == 0:
            daily_values.append(port_value)
            if prev_holdings:
                rets.append(0)
            continue

        new_holdings = None
        if dd['is_sig'].iloc[0]:
            top10 = dd.nlargest(10, 'pred')
            if use_weighted:
                sc = top10['pred'].values
                w = sc / sc.sum()
            else:
                w = np.ones(10) / 10
            pf = {r['ts_code']: w[j] for j, (_, r) in enumerate(top10.iterrows())}
            new_holdings = set(pf.keys())

        # 止损
        if stop_loss and pf:
            dm = dd.set_index('ts_code')
            pf = {ts: w for ts, w in pf.items()
                  if not (ts in dm.index and pd.notna(dm.loc[ts,'ret_1d']) and dm.loc[ts,'ret_1d'] < -stop_loss)}
            new_holdings = set(pf.keys())

        # 计算调仓成本
        trade_cost = 0
        if cost_per_trade > 0 and new_holdings is not None:
            # 计算换手率: 卖出的+买入的
            to_sell = prev_holdings - new_holdings
            to_buy = new_holdings - prev_holdings
            # 调仓比例 = 卖出的权重之和 + 买入的权重之和
            turnover = 0
            for ts in to_sell:
                if ts in prev_holdings:
                    turnover += 1.0 / max(len(prev_holdings), 1)
            for ts in to_buy:
                if ts in pf:
                    turnover += pf[ts]
            # 最多2倍（全部换仓）
            turnover = min(turnover, 2.0)
            trade_cost = turnover * cost_per_trade
            prev_holdings = new_holdings

        # 计算持仓收益
        if pf:
            dm = dd.set_index('ts_code')
            tw = sum(pf.values())
            pr = 0.0
            for ts, w in pf.items():
                if ts in dm.index:
                    rv = dm.loc[ts, 'ret_1d']
                    if pd.notna(rv):
                        pr += (w / tw) * rv
            # 扣除交易成本
            net_ret = pr - trade_cost
            port_value *= (1 + net_ret)
            rets.append(net_ret)
        else:
            daily_values.append(port_value)
            if prev_holdings:
                rets.append(-trade_cost)

    return np.array(rets), port_value

# 全样本对比（4种版本）
configs = [
    ('等权 无成本',   False, None,  0),
    ('等权 含成本',   False, None,  TRADE_COST),
    ('不等权 含成本',  True,  None,  TRADE_COST),
    ('不等权+止损 含成本', True, STOP_LOSS, TRADE_COST),
]

print("\n  --- 全样本 (2025+2026) 含成本对比 ---")
for name, w, sl, cost in configs:
    r, final_v = run_backtest_with_cost(panel, use_weighted=w, stop_loss=sl, cost_per_trade=cost)
    sr = np.sqrt(252) * r.mean() / (r.std() + 1e-10)
    cum = np.cumprod(1 + r)
    mdd = (np.maximum.accumulate(cum) - cum).max()
    print(f"  {name:25s}: 夏普={sr:6.4f} | 累计={final_v*100-100:8.2f}% | 回撤={mdd*100:5.2f}%")

# ====== 样本外 (2025训练->2026测试,含成本) ======
print("\n  --- 严格样本外 (2025训练->2026测试) ---")

td25 = td[td['trade_date'].astype(str) < '2026-01-01']
td26_test = panel[panel['year'] == 2026].copy()
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

for name, w, sl, cost in configs:
    r, final_v = run_backtest_with_cost(td26_test, use_weighted=w, stop_loss=sl, cost_per_trade=cost)
    sr = np.sqrt(252) * r.mean() / (r.std() + 1e-10)
    print(f"  {name:25s}: 夏普={sr:6.4f} | 累计={final_v*100-100:7.2f}%")

# ====== 2026年样本内对比（看v8/v9/v10差异） ======
print("\n  --- 2026样本内（仅供对比参考）---")
panel_2026 = panel[panel['year'] == 2026]
for name, w, sl, cost in configs:
    r, final_v = run_backtest_with_cost(panel_2026, use_weighted=w, stop_loss=sl, cost_per_trade=cost)
    sr = np.sqrt(252) * r.mean() / (r.std() + 1e-10)
    cum = np.cumprod(1 + r)
    mdd = (np.maximum.accumulate(cum) - cum).max()
    print(f"  {name:25s}: 夏普={sr:6.4f} | 累计={final_v*100-100:7.2f}% | 回撤={mdd*100:5.2f}%")

# ====== 最新信号 ======
print("\n  📋 最新信号 (2026-06-02):")
ld = panel['trade_date'].max()
lt = panel[panel['trade_date'] == ld].sort_values('pred', ascending=False)

pf_orig = pq.ParquetFile('data/factors/factor_panel_v5_final.parquet')
orig = pf_orig.read(columns=['ts_code','trade_date','市值']).to_pandas()
orig_latest = orig[orig['trade_date'] == ld]
orig_map = orig_latest.set_index('ts_code')['市值'].to_dict()

top10 = lt.head(10).copy()
scores = top10['pred'].values
weights = scores / scores.sum()

for i, (_, r) in enumerate(top10.iterrows()):
    raw_cap = orig_map.get(r['ts_code'], np.nan)
    cap_v = np.exp(raw_cap)/1e8 if pd.notna(raw_cap) else 0
    print(f"  {i+1:2d}. {r['ts_code']} {r['name']:<10s} | {r['industry']:<8s} | {weights[i]*100:5.1f}%")

out = top10[['ts_code','trade_date','pred']].copy()
out['weight'] = weights
out['name'] = [si[si['ts_code']==c]['name'].values[0] if len(si[si['ts_code']==c])>0 else '' for c in out['ts_code']]
out['industry'] = [si[si['ts_code']==c]['industry'].values[0] if len(si[si['ts_code']==c])>0 else '' for c in out['ts_code']]
out.to_csv('signals/v10_latest_signal.csv', index=False)
print("  ✅ signals/v10_latest_signal.csv")

print("\n" + "=" * 60)
print("项目B v10 完成")
print("=" * 60)
