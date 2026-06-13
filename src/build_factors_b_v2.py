"""
项目B — 10因子构建 Pipeline v2
==============================
性能优化: 用 groupby.apply + OLS 一次性完成中性化
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 60)
print("项目B — 10因子构建 Pipeline v2")
print("=" * 60)

print("\n[1/5] 加载数据...")
panel_path = 'data/factors/factor_panel_v5_final.parquet'
si_path = 'data/raw/stock_list.parquet'

pf = pq.ParquetFile(panel_path)
panel = pf.read().to_pandas()
print(f"  面板: {panel.shape}, 日期: {panel['trade_date'].min()}~{panel['trade_date'].max()}")

si = pd.read_parquet(si_path)[['ts_code', 'industry']]
print(f"  股票列表: {len(si)} 只, 行业: {si['industry'].nunique()} 个")

# ============================================================
# 2. 提取/构建因子
# ============================================================
print("\n[2/5] 提取/构建因子...")

df = panel[['ts_code', 'trade_date']].copy()
df['市值'] = panel['市值'].values

# 2.1 直接提取的因子
raw_factors = {
    'mmt_overnight': 'overnight_ret',
    'money_flow': 'moneyflow_strength',
    'idvol': 'idvol',
    'turnover_bias': 'turnover_bias',
    'roe_ttm': 'ROE',
    'gross_margin': '毛利率',
    'revise_up': 'revise_up_proxy',
}
for new_n, old_n in raw_factors.items():
    df[new_n] = panel[old_n].values
    print(f'  ✅ {new_n} ← {old_n}')

# 2.2 营收增速 proxy
print('  ⚠️ 营收增速: ROE同比变化 proxy')
panel_sorted = panel.sort_values(['ts_code', 'trade_date'])
df['revenue_growth'] = (
    panel_sorted.groupby('ts_code')['ROE']
    .transform(lambda x: x.pct_change(periods=244))
    .clip(-0.5, 0.5)
).values
print('  ✅ revenue_growth')

# 2.3 北向 proxy
print('  ⚠️ 北向净买入: 行业资金流排名 proxy')
df = df.merge(si, on='ts_code', how='left')
# 在合入行业后做排名（每个日期-行业内对 money_flow 排名）
def rank_within_group(g):
    return g['money_flow'].rank(pct=True)
df['northbound'] = df.groupby(['trade_date', 'industry'], group_keys=False).apply(
    lambda g: g['money_flow'].rank(pct=True)
).values
print('  ✅ northbound')

# 2.4 PB行业中性
bp = panel['BP'].values
bp_safe = np.where(bp > 0, bp, np.nan)
pb_raw = np.where(~np.isnan(bp_safe), 1.0 / bp_safe, np.nan)
df['pb_raw'] = pb_raw
print('  ✅ pb_raw')

# ============================================================
# 3. 中性化 + 去极值 + 标准化（一次搞定）
# ============================================================
print("\n[3/5] 中性化 + 去极值 + 标准化（截面处理）...")

all_factors = ['mmt_overnight', 'money_flow', 'idvol', 'turnover_bias',
               'roe_ttm', 'gross_margin', 'revenue_growth', 'northbound',
               'revise_up', 'pb_raw']

date_groups = df.groupby('trade_date')
dates = sorted(df['trade_date'].unique())
print(f"  处理 {len(dates)} 个交易日...")

results = {}
for fac in all_factors:
    results[fac] = {'raw': [], 'neutral': [], 'z': []}

for i, dt in enumerate(dates):
    if (i + 1) % 200 == 0:
        print(f"  进度: {i+1}/{len(dates)}")

    mask = df['trade_date'] == dt
    idx = np.where(mask)[0]
    grp = df.loc[mask].copy()
    n = len(grp)
    if n < 50:
        for fac in all_factors:
            results[fac]['raw'].extend([np.nan] * n)
            results[fac]['neutral'].extend([np.nan] * n)
            results[fac]['z'].extend([np.nan] * n)
        continue

    # --- 行业中性化（对每个因子做行业dummy回归取残差）---
    ind_dummies = pd.get_dummies(grp['industry'])
    ind_dummies = ind_dummies.loc[:, ind_dummies.sum() > 0]

    log_cap = np.log(np.maximum(grp['市值'].values, 1e6))
    cap_reg = (log_cap - log_cap.mean()) / log_cap.std()
    X = np.column_stack([np.ones(n), cap_reg, ind_dummies.values])

    for fac in all_factors:
        raw_vals = grp[fac].values.astype(float)

        # 记原始值
        results[fac]['raw'].extend(raw_vals.tolist())

        # --- 去极值 ---
        good = ~np.isnan(raw_vals)
        if good.sum() >= 30:
            med = np.nanmedian(raw_vals)
            mad = np.nanmedian(np.abs(raw_vals - med))
            if mad > 1e-10:
                upper = med + 3 * 1.4826 * mad
                lower = med - 3 * 1.4826 * mad
                vals_clipped = np.clip(raw_vals, lower, upper)
            else:
                vals_clipped = raw_vals.copy()
        else:
            vals_clipped = raw_vals.copy()

        # --- 中性化（回归取残差）---
        y = vals_clipped.copy()
        y[np.isnan(y)] = 0.0  # nan 补0
        XtX = X.T @ X
        try:
            beta = np.linalg.lstsq(XtX, X.T @ y, rcond=None)[0]
            residual = y - X @ beta
        except:
            residual = y.copy()

        # 残差非nan的才有效
        results[fac]['neutral'].extend(residual.tolist())

        # --- 截面标准化 ---
        mu = np.nanmean(residual)
        std = np.nanstd(residual)
        if std > 1e-10:
            z = (residual - mu) / std
        else:
            z = residual * 0.0
        results[fac]['z'].extend(z.tolist())

# 把结果写回df
for fac in all_factors:
    df[f'{fac}_raw'] = results[fac]['raw']
    df[f'{fac}_neutral'] = results[fac]['neutral']
    df[f'{fac}_z'] = results[fac]['z']

print(f"  全部 {len(dates)} 个交易日处理完成")

# ============================================================
# 4. 因子最终统计
# ============================================================
print("\n[4/5] 因子统计摘要...")

print(f"\n{'因子名称':>20s} | {'IC均值':>8s} | {'均值':>8s} | {'标准差':>8s} | {'缺失率':>8s}")
print("-" * 65)
for fac in all_factors:
    zcol = f'{fac}_z'
    vals = df[zcol].dropna()
    ic = vals.corr(panel.loc[df.index, 'fwd_20d_ret']) if 'fwd_20d_ret' in panel.columns else 0
    print(f"{fac:>20s} | {ic:>8.4f} | {vals.mean():>8.4f} | {vals.std():>8.4f} | {1-len(vals)/len(df):>8.4f}")

# ============================================================
# 5. 保存
# ============================================================
print("\n[5/5] 保存结果...")

out_cols = ['ts_code', 'trade_date', 'industry', '市值']
for fac in all_factors:
    out_cols.append(f'{fac}_z')  # 最终标准化值

result_df = df[out_cols].copy()
output_path = 'data/factors/factor_panel_b.parquet'
result_df.to_parquet(output_path, index=False)
print(f'  ✅ 已保存: {output_path}')
print(f'    形状: {result_df.shape}, 列数: {len(result_df.columns)}')
print(f'    文件大小: {round(pq.ParquetFile(output_path).metadata.total_byte_size / 1024**3, 1)} GB')
