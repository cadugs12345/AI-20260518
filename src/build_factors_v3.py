"""
快速构建面板v3 - 整合daioy_basic财务因子 + 5日标签
"""
import os, sys, time, gc
import numpy as np
import pandas as pd

t0 = time.time()
print("="*60)
print("面板v3 - 快速构建")
print("="*60)

# 加载面板（已有技术因子）
panel = pd.read_parquet("data/factors/factor_panel_with_fwd_v2.parquet")
prices = pd.read_parquet("data/factors/full_prices.parquet")
daily = pd.read_parquet("data/factors/daily_basic_all.parquet")

panel["trade_date"] = pd.to_datetime(panel["trade_date"])
prices["trade_date"] = pd.to_datetime(prices["trade_date"])
daily["trade_date"] = pd.to_datetime(daily["trade_date"])

print(f"原有面板: {len(panel):,}行")

# ===== 1. 合并daily_basic因子 =====
# 每天都有pe/pb/ps/总市值/流通市值
# 流通市值已经在panel中了，跳过
basic_cols = ['ts_code','trade_date','pe','pe_ttm','pb','ps','ps_ttm','dv_ratio','dv_ttm','total_mv','circ_mv']
daily = daily[basic_cols].drop_duplicates(subset=['ts_code','trade_date'])

panel = panel.merge(daily, on=['ts_code','trade_date'], how='left')
print(f"合并daily_basic后: {len(panel):,}行")

# ===== 2. 加几个关键财务指标（用fina_indicator按年全市场下载）=====
print("\n下载财务指标...")
import tushare as ts
pro = ts.pro_api('3e8953587c4c717c26e5cb99d028a66e044d184f2d464cab0950000e')

# 按年全量下载（无ts_code参数，一次拿全年所有股票）
all_indicators = []
for year in range(2016, 2027):
    try:
        # finaIndicator 可以用 start_date 和 end_date 不指定ts_code
        # 但有些版本必须指定ts_code，试试不行就按月分批
        df = pro.fina_indicator(start_date=f"{year}0101", end_date=f"{year}1231",
            fields="ts_code,ann_date,end_date,roe,grossprofit_margin,netprofit_margin,basic_eps_yoy,"
                   "netprofit_yoy,or_yoy,roe_yoy,eps,q_roe,q_sales_yoy,q_op_qoq,debt_to_assets,ocfps")
        if df is not None and len(df) > 0:
            df['ann_date'] = pd.to_datetime(df['ann_date'])
            df['end_date'] = pd.to_datetime(df['end_date'])
            all_indicators.append(df)
            print(f"  {year}: {len(df):,}条")
        else:
            print(f"  {year}: 空")
    except Exception as e:
        print(f"  {year}: 报错 {e}")

if all_indicators:
    fina = pd.concat(all_indicators, ignore_index=True)
    fina = fina.drop_duplicates(subset=['ts_code','ann_date','end_date'])
    fina = fina.sort_values(['ts_code','end_date'])
    print(f"财务指标汇总: {len(fina):,}条")
    
    # 对每个股票的每条记录，保持到下一个报告期之前都可用
    # 用ann_date对齐交易日：公告日后可用
    # 简单的做法：直接merge到最近的交易日
    # 先将panel排序，做asof merge
    panel_with_date = panel[['ts_code','trade_date']].copy()
    
    # 对每个股票做asof merge
    print("合并财务指标（逐股票asof merge）...")
    fin_cols = [c for c in fina.columns if c not in ('ts_code','ann_date','end_date')]
    
    result_parts = []
    codes = panel['ts_code'].unique()
    n_codes = len(codes)
    
    for i, code in enumerate(codes):
        p_sub = panel_with_date[panel_with_date['ts_code'] == code].sort_values('trade_date')
        f_sub = fina[fina['ts_code'] == code][['ann_date'] + fin_cols].sort_values('ann_date')
        
        if len(f_sub) == 0:
            for col in fin_cols:
                p_sub[col] = np.nan
        else:
            merged = pd.merge_asof(
                p_sub[['trade_date']],
                f_sub,
                left_on='trade_date', right_on='ann_date',
                direction='backward'
            )
            # 过期>365天的清除
            days_diff = (merged['trade_date'] - merged['ann_date']).dt.total_seconds() / 86400
            for col in fin_cols:
                p_sub[col] = merged[col].values
                if col in merged.columns:
                    p_sub.loc[days_diff.values > 365, col] = np.nan
        
        result_parts.append(p_sub)
        if (i+1) % 500 == 0:
            print(f"  {i+1}/{n_codes}")
    
    panel_fina = pd.concat(result_parts, ignore_index=True)
    
    # 合并回panel
    for col in fin_cols:
        if col in panel_fina.columns:
            panel[col] = panel_fina[col].values
    
    print(f"财务因子合并完成: {len(panel):,}行")
else:
    print("⚠️ 财务指标下载失败，仅使用daily_basic")

# ===== 3. 5日收益标签 =====
print("\n计算fwd_5d_ret...")
prices_sorted = prices.sort_values(['ts_code','trade_date']).copy()
prices_sorted['close_next5'] = prices_sorted.groupby('ts_code')['close'].shift(-5)
prices_sorted['fwd_5d_ret'] = prices_sorted['close_next5'] / prices_sorted['close'] - 1
fwd5 = prices_sorted[['trade_date','ts_code','fwd_5d_ret']].copy()

panel = panel.merge(fwd5, on=['trade_date','ts_code'], how='left')
print(f"  fwd_5d_ret NaN: {panel['fwd_5d_ret'].isna().sum()}/{len(panel)}")

# 验证
v = panel[(panel['ts_code'] == '000001.SZ') & (panel['trade_date'] == '2021-01-04')]
print(f"  000001.SZ 2021-01-04: fwd_5d_ret={v['fwd_5d_ret'].values[0]*100:.2f}%")

# ===== 4. 清理和保存 =====
panel.to_parquet("data/factors/factor_panel_v3.parquet", index=False)
print(f"\n保存: data/factors/factor_panel_v3.parquet")
print(f"  行数: {len(panel):,}")
print(f"  列: {[c for c in panel.columns if c not in ('ts_code','trade_date')]}")
print(f"总用时: {(time.time()-t0)/60:.1f}分")
