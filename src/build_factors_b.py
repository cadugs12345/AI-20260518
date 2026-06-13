"""
项目B — 10因子构建 Pipeline
============================
目标因子:
  1. 隔夜动量 (mmt_overnight)     — 已有 overnight_ret
  2. 资金流强度 (money_flow)      — 已有 moneyflow_strength
  3. 残差波动率 (idvol)           — 已有 idvol
  4. 换手率乖离 (turnover_bias)   — 已有 turnover_bias
  5. ROE(TTM)                     — 已有 ROE
  6. 毛利率 (gross_margin)        — 已有 毛利率
  7. 营收增速 (revenue_growth)    — 需从财务原始数据构建（无营收明细，用净利增速 proxy）
  8. 北向净买入 (northbound)      — 仅有总量数据，个股级别需外部数据
  9. 分析师预期上调 (revise_up)   — 已有 revise_up_proxy
  10. PB行业中性                  — 用 BP 反推 PB + 行业中性化

所有因子经过: 行业中性 + 市值中性 → 去极值(MAD) → 截面标准化(Z-score)
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import mstats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 加载基础数据
# ============================================================
print("=" * 60)
print("项目B — 10因子构建 Pipeline")
print("=" * 60)

print("\n[1/5] 加载数据...")
panel_path = 'data/factors/factor_panel_v5_final.parquet'
si_path = 'data/raw/stock_list.parquet'

# 读取最终面板
pf = pq.ParquetFile(panel_path)
panel = pf.read().to_pandas()
print(f"  面板: {panel.shape}, 日期: {panel['trade_date'].min()}~{panel['trade_date'].max()}")

# 行业信息
si = pd.read_parquet(si_path)[['ts_code', 'industry']]
print(f"  股票列表: {len(si)} 只, 行业: {si['industry'].nunique()} 个")

# ============================================================
# 2. 提取/构建因子
# ============================================================
print("\n[2/5] 提取/构建因子...")

# ----- 2.1 直接可用的因子（已有） -----
factor_map = {
    'mmt_overnight': 'overnight_ret',
    'money_flow': 'moneyflow_strength',
    'idvol': 'idvol',
    'turnover_bias': 'turnover_bias',
    'roe_ttm': 'ROE',
    'gross_margin': '毛利率',
    'revise_up': 'revise_up_proxy'
}

factors_df = panel[['ts_code', 'trade_date']].copy()
for new_name, old_name in factor_map.items():
    factors_df[new_name] = panel[old_name].values
    print(f'  ✅ {new_name} ← {old_name}')

# ----- 2.2 营收增速 (revenue_growth) -----
# 财务数据没有营收/利润明细，但可以用净利增速 proxied by ROE 变化
# 但更好的proxy: 用面板已有的 ROE 同比变化
# 实际上已知: 面板有 ROE，按季度计算同比变化 = TTM ROE 一年之差
print('  ⚠️ 营收增速: 无营收明细数据，使用 ROE_TTM 同比变化 proxy')
print('    (实际使用时建议补充利润表数据后替换)')

# 按股票+日期排序，计算年同比
panel_sorted = panel.sort_values(['ts_code', 'trade_date'])
# 用一年约244交易日近似
factors_df['revenue_growth'] = panel_sorted.groupby('ts_code')['ROE'].transform(
    lambda x: x.pct_change(periods=244)
).values
# 处理极端值
factors_df['revenue_growth'] = factors_df['revenue_growth'].clip(-0.5, 0.5)
print('  ✅ revenue_growth (ROE同比变化 proxy)')

# ----- 2.3 北向净买入 (northbound) -----
# 个股级别北向数据项目中没有（只有总量），用资金流强度 proxied
print('  ⚠️ 北向净买入: 仅有总量数据，用 moneyflow_strength 做行业排名替代')
print('    (如需精确数据需补充个股北向持仓数据)')
# 先求 moneyflow_strength 的行业排名
temp = factors_df[['trade_date', 'ts_code']].copy()
temp['money_flow'] = factors_df['money_flow'].values
# 加入行业
temp = temp.merge(si, on='ts_code', how='left')
temp['northbound'] = temp.groupby(['trade_date', 'industry'])['money_flow'].rank(pct=True)
factors_df['northbound'] = temp['northbound'].values
print('  ✅ northbound (行业资金流排名 proxy)')

# ----- 2.4 PB行业中性 -----
# 面板有 BP，PB = 1/BP（前提有净资产为正）
# BP可能很小或为负，避免除零
bp = panel['BP'].values
bp_safe = np.where(bp > 0, bp, np.nan)  # BP<=0的设为nan
pb_raw = np.where(~np.isnan(bp_safe), 1.0 / bp_safe, np.nan)
factors_df['pb_raw'] = pb_raw

# 加入行业和市值做中性化
temp2 = factors_df[['trade_date', 'ts_code']].copy()
temp2['pb_raw'] = pb_raw
temp2['市值'] = panel['市值'].values
temp2 = temp2.merge(si, on='ts_code', how='left')

print('  ✅ pb_raw 构建完成')

# ============================================================
# 3. 中性化处理（行业 + 市值）
# ============================================================
print("\n[3/5] 中性化处理...")

def neutralize_factor(df, factor_name, date_col='trade_date', ind_col='industry',
                       cap_col='市值', log_cap=True):
    """行业中性 + 市值中性（截面回归取残差）"""
    result = df[factor_name].copy().values
    for dt in df[date_col].unique():
        mask = df[date_col] == dt
        idx = np.where(mask)[0]
        if len(idx) < 50:
            continue

        y = df.loc[mask, factor_name].values
        cap = df.loc[mask, cap_col].values
        ind_dummies = pd.get_dummies(df.loc[mask, ind_col])

        # 去掉全零列
        ind_dummies = ind_dummies.loc[:, ind_dummies.sum() > 0]

        # 构建回归矩阵
        if log_cap:
            log_cap_vals = np.log(np.maximum(cap, 1e6))
        else:
            log_cap_vals = cap
        cap_std = (log_cap_vals - log_cap_vals.mean()) / log_cap_vals.std()

        X = np.column_stack([np.ones(len(idx)), cap_std, ind_dummies.values])
        XtX = X.T @ X
        try:
            beta = np.linalg.lstsq(XtX, X.T @ y, rcond=None)[0]
            residual = y - X @ beta
            result[idx] = residual
        except:
            pass
    return result

all_factors = ['mmt_overnight', 'money_flow', 'idvol', 'turnover_bias',
               'roe_ttm', 'gross_margin', 'revenue_growth', 'northbound',
               'revise_up', 'pb_raw']

# 合并行业和市值到 factors_df
factors_df = factors_df.merge(si, on='ts_code', how='left')
factors_df['市值'] = panel['市值'].values

for fac in all_factors:
    neutral = neutralize_factor(factors_df, fac)
    factors_df[f'{fac}_neutral'] = neutral
    print(f'  ✅ {fac}_neutral')

# ============================================================
# 4. 去极值 (MAD 3σ) + 截面标准化 (Z-score)
# ============================================================
print("\n[4/5] 去极值 + 标准化...")

for fac in all_factors:
    col = f'{fac}_neutral'

    # MAD 去极值
    for dt in factors_df['trade_date'].unique():
        mask = factors_df['trade_date'] == dt
        vals = factors_df.loc[mask, col].values
        if len(vals) < 10:
            continue
        med = np.nanmedian(vals)
        mad = np.nanmedian(np.abs(vals - med))
        upper = med + 3 * 1.4826 * mad
        lower = med - 3 * 1.4826 * mad
        factors_df.loc[mask, col] = np.clip(vals, lower, upper)

    # 截面 Z-score 标准化
    for dt in factors_df['trade_date'].unique():
        mask = factors_df['trade_date'] == dt
        vals = factors_df.loc[mask, col].values
        mu = np.nanmean(vals)
        std = np.nanstd(vals)
        if std > 1e-10:
            factors_df.loc[mask, f'{fac}_z'] = (vals - mu) / std
        else:
            factors_df.loc[mask, f'{fac}_z'] = 0.0

    print(f'  ✅ {fac}_z (标准化完成)')

# ============================================================
# 5. 保存结果
# ============================================================
print("\n[5/5] 保存结果...")

# 整理输出列
out_cols = ['ts_code', 'trade_date']
for fac in all_factors:
    out_cols.extend([fac, f'{fac}_neutral'])

# 加上最终标准化值（简洁版用于直接使用）
z_cols = [f'{fac}_z' for fac in all_factors]
out_cols += z_cols

# 同时保存行业信息用于后续
out_cols += ['industry', '市值']

result_df = factors_df[out_cols].copy()

output_path = 'data/factors/factor_panel_b.parquet'
result_df.to_parquet(output_path, index=False)
print(f'  ✅ 已保存: {output_path}')
print(f'    形状: {result_df.shape}')
print(f'    列数: {len(result_df.columns)}')

# 打印因子统计摘要
print("\n" + "=" * 60)
print("📊 因子最终统计摘要 (z-score)")
print("=" * 60)
stats = []
for fac in all_factors:
    zcol = f'{fac}_z'
    vals = result_df[zcol].dropna()
    stats.append({
        '因子': fac,
        '均值': vals.mean(),
        '标准差': vals.std(),
        '偏度': vals.skew(),
        '缺失率': 1 - len(vals) / len(result_df)
    })
stats_df = pd.DataFrame(stats)
print(stats_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
