"""
项目B — 10因子构建 Pipeline v3 (内存优化)
"""
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("项目B — 10因子构建 Pipeline v3")
print("=" * 60)

# ============================================================
# 1. 只读需要的列（大幅减少内存）
# ============================================================
print("\n[1/5] 加载数据（只读必要列）...")
panel_path = 'data/factors/factor_panel_v5_final.parquet'

need_cols = ['ts_code', 'trade_date', 'close', '市值',
             'overnight_ret', 'moneyflow_strength', 'idvol', 'turnover_bias',
             'ROE', '毛利率', 'revise_up_proxy', 'BP', 'fwd_20d_ret']
panel = pq.read_table(panel_path, columns=need_cols).to_pandas()
print(f"  面板: {panel.shape}, 内存: ~{panel.memory_usage(deep=True).sum() / 1024**3:.1f}GB")

si = pd.read_parquet('data/raw/stock_list.parquet')[['ts_code', 'industry']]
print(f"  行业: {si['industry'].nunique()} 个")

# ============================================================
# 2. 构建10因子
# ============================================================
print("\n[2/5] 构建因子...")
df = panel[['ts_code', 'trade_date', '市值']].copy()

# 2.1 直接复制
for new_n, old_n in [('mmt_overnight','overnight_ret'),('money_flow','moneyflow_strength'),
                      ('idvol','idvol'),('turnover_bias','turnover_bias'),
                      ('roe_ttm','ROE'),('gross_margin','毛利率'),('revise_up','revise_up_proxy')]:
    df[new_n] = panel[old_n].values
    print(f'  ✅ {new_n}')

# 2.2 营收增速: ROE同比变化
print('  ⚠️ revenue_growth: ROE同比变化 proxy')
df['revenue_growth'] = (panel.sort_values(['ts_code','trade_date'])
                         .groupby('ts_code')['ROE']
                         .transform(lambda x: x.pct_change(244)).clip(-0.5, 0.5)).values

# 2.3 北向净买入: 行业资金流排名
df = df.merge(si, on='ts_code', how='left')
df['northbound'] = df.groupby(['trade_date','industry'], group_keys=False)['money_flow'].apply(
    lambda g: g.rank(pct=True)
)
print('  ✅ northbound (行业资金流排名)')

# 2.4 PB = 1/BP
bp_v = panel['BP'].values
df['pb_raw'] = np.where(bp_v > 0, 1.0 / bp_v, np.nan)
print('  ✅ pb_raw')

# ============================================================
# 3. 截面处理: 去极值 → 行业+市值中性 → Z-score
# ============================================================
print("\n[3/5] 截面处理（逐日: 去极值→中性化→标准化）...")

all_factors = ['mmt_overnight','money_flow','idvol','turnover_bias',
               'roe_ttm','gross_margin','revenue_growth','northbound',
               'revise_up','pb_raw']
dates = sorted(df['trade_date'].unique())
n_dates = len(dates)
print(f"  共 {n_dates} 个交易日")

# 为每个因子准备输出列
for fac in all_factors:
    df[f'{fac}_z'] = np.nan

for i, dt in enumerate(dates):
    if (i+1) % 300 == 0:
        print(f"  进度: {i+1}/{n_dates} ({(i+1)/n_dates*100:.0f}%)")

    mask = df['trade_date'] == dt
    sub = df.loc[mask]
    n = len(sub)
    if n < 50:
        continue

    # 行业dummy + 市值
    ind_dummies = pd.get_dummies(sub['industry'])
    ind_dummies = ind_dummies.loc[:, ind_dummies.sum() > 0]
    log_cap = np.log(np.maximum(sub['市值'].values, 1e6))
    cap_reg = (log_cap - log_cap.mean()) / (log_cap.std() + 1e-10)
    X = np.column_stack([np.ones(n), cap_reg, ind_dummies.values])

    for fac in all_factors:
        vals = sub[fac].values.astype(float)

        # 去极值 MAD
        med = np.nanmedian(vals)
        mad = np.nanmedian(np.abs(vals - med))
        if mad > 1e-10:
            upper = med + 3 * 1.4826 * mad
            lower = med - 3 * 1.4826 * mad
            vals = np.clip(vals, lower, upper)

        # 中性化: OLS残差
        y = np.nan_to_num(vals, nan=0.0)
        try:
            beta = np.linalg.lstsq(X.T @ X, X.T @ y, rcond=None)[0]
            residual = y - X @ beta
        except:
            residual = y.copy()

        # 标准化 Z-score
        mu = np.nanmean(residual)
        std = np.nanstd(residual)
        if std > 1e-10:
            df.loc[mask, f'{fac}_z'] = (residual - mu) / std
        else:
            df.loc[mask, f'{fac}_z'] = 0.0

print(f"  全部 {n_dates} 个交易日处理完成")

# ============================================================
# 4. 统计 + IC
# ============================================================
print("\n[4/5] 因子统计...")
fwd_ret = panel['fwd_20d_ret'].values if 'fwd_20d_ret' in panel.columns else None

print(f"\n{'因子名称':>20s} | {'IC均值':>8s} | {'均值':>8s} | {'标准差':>8s} | {'缺失率':>8s}")
print("-" * 65)
for fac in all_factors:
    zcol = f'{fac}_z'
    vals = df[zcol].values
    good = ~np.isnan(vals)
    if good.sum() > 100 and fwd_ret is not None:
        ic = np.corrcoef(vals[good], fwd_ret[good])[0,1]
    else:
        ic = np.nan
    print(f"{fac:>20s} | {ic:>8.4f} | {vals[good].mean():>8.4f} | {vals[good].std():>8.4f} | {1-good.sum()/len(vals):>8.4f}")

# ============================================================
# 5. 保存
# ============================================================
print("\n[5/5] 保存结果...")
out_cols = ['ts_code', 'trade_date', 'industry', '市值'] + [f'{fac}_z' for fac in all_factors]
result_df = df[out_cols].copy()
result_df.to_parquet('data/factors/factor_panel_b.parquet', index=False)
pf2 = pq.ParquetFile('data/factors/factor_panel_b.parquet')
print(f'  ✅ factor_panel_b.parquet: {result_df.shape}, {round(pf2.metadata.total_byte_size/1024**3, 1)}GB')
print("  因子列:", [f'{fac}_z' for fac in all_factors])

print("\n" + "=" * 60)
print("✅ 项目B 10因子构建完成")
print("=" * 60)
