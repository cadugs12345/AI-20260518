#!/usr/bin/env python3
"""
涨停回踩不破 v3 — 独立选股程序

策略逻辑:
  1. 标志K线: 涨停
  2. 当时MA18向上
  3. 后续5天内，股价曾收盘跌破MA18，或未跌破但收盘距MA18不超3%
  4. 今天盘中曾低于MA18
  5. 今天收盘上穿MA18（昨天<MA18, 今天>=MA18）
  6. 买入价 = MA18 × 1.01
  7. 今天收盘价 < MA18 × 1.05（距均线不超过5%）
  8. MA18 4日斜率 > -2%（不能加速下行）
  9. 近6个月波动 < 100%（排除妖股）
  10. 涨停日在最近15个交易日内（排除陈年涨停）

买入信号: 当天即信号日，次日开盘买入
          买入价为 min(次日开盘价, MA18×1.01)

使用说明:
  运行: python src/zt_pullback_v2.py
  输出: signals/zt_pullback_v2_{YYYYMMDD}.csv
        signals/zt_pullback_v2_latest.csv
"""
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 路径配置
# ============================================================
PROJ_B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DAILY_DIR = os.path.join(PROJ_B, "data", "raw", "daily")
SIGNALS_DIR = os.path.join(PROJ_B, "signals")
STOCK_LIST_FILE = os.path.join(PROJ_B, "data", "raw", "stock_list.parquet")

os.makedirs(SIGNALS_DIR, exist_ok=True)

# ============================================================
# 参数
# ============================================================
LIMIT_UP_PCT = 1.095       # 涨停阈值
MA_PERIOD = 18             # 18日均线
LOOKBACK_LIMIT = 5         # 涨停后5天内必须跌破或接近MA18
CLOSE_TO_MA_THRESHOLD = 0.03  # 未跌破时距MA18不超过3%
MAX_LOOKBACK_LIMIT = 60    # 往回找涨停最多60天
MIN_TRADE_DAYS = 60        # 最低上市交易日
MIN_LIST_YEARS = 5         # 上市至少5年
DISTANCE_CAP_PCT = 5       # 收盘价距MA18不超过5%
MA_SLOPE_MIN = -2.0         # MA18 4日斜率下限（如-2%，即不能比-2%更陡）
MA_SLOPE_LOOKBACK = 4       # 斜率计算回看天数
VOLATILITY_LOOKBACK = 120    # 波动率回看天数（约6个月）
VOLATILITY_MAX_PCT = 100    # 波动率上限（最高价比最低价涨幅不超过100%）
LIMIT_BAR_RECENCY = 15      # 涨停日必须在最近15个交易日内

# ============================================================
# 辅助函数
# ============================================================
def compute_ma(arr, period):
    """计算移动平均"""
    n = len(arr)
    result = np.full(n, np.nan)
    if n < period:
        return result
    cumsum = np.cumsum(arr)
    result[period - 1] = cumsum[period - 1] / period
    for i in range(period, n):
        result[i] = (cumsum[i] - cumsum[i - period]) / period
    return result


def compute_bollinger(close, period=20, n_std=2):
    """计算布林带: mid, upper, lower"""
    n = len(close)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    mid = compute_ma(close, period)

    for i in range(period - 1, n):
        std = np.std(close[i - period + 1:i + 1])
        upper[i] = mid[i] + n_std * std
        lower[i] = mid[i] - n_std * std

    return mid, upper, lower


# ============================================================
# 1. 加载股票列表
# ============================================================
print("=" * 60)
print("涨停回踩不破 v3 — 选股 (MA18 + BOLL止盈)")
print("=" * 60)

print("\n[1/5] 加载股票列表...")
try:
    stock_list = pd.read_parquet(STOCK_LIST_FILE)
    # 上市至少MIN_LIST_YEARS年
    cutoff_date = pd.Timestamp.now() - pd.DateOffset(years=MIN_LIST_YEARS)
    stock_list = stock_list[pd.to_datetime(stock_list['list_date'], errors='coerce') <= cutoff_date]
    ts_codes = sorted(stock_list['ts_code'].unique())
    print(f"  共 {len(ts_codes)} 只股票（上市≥{MIN_LIST_YEARS}年）")
except Exception as e:
    print(f"  ⚠️ 无法加载股票列表: {e}")
    print(f"  → 尝试扫描 daily 目录...")
    ts_codes = sorted([
        f.replace('.parquet', '')
        for f in os.listdir(DATA_DAILY_DIR)
        if f.endswith('.parquet')
    ])
    print(f"  共 {len(ts_codes)} 只股票")

# 名称/行业映射
name_map = dict(zip(stock_list['ts_code'], stock_list.get('name', [''] * len(stock_list))))
industry_map = dict(zip(stock_list['ts_code'], stock_list.get('industry', [''] * len(stock_list))))

# ============================================================
# 2. 逐股计算
# ============================================================
print("\n[2/5] 逐股扫描计算...")

results = []
n_stocks = len(ts_codes)

for idx, ts_code in enumerate(ts_codes):
    if (idx + 1) % 500 == 0:
        print(f"  进度: {idx+1}/{n_stocks} ({100*(idx+1)/n_stocks:.0f}%)")

    fpath = os.path.join(DATA_DAILY_DIR, f"{ts_code}.parquet")
    if not os.path.exists(fpath):
        continue

    try:
        df = pd.read_parquet(fpath)
    except Exception:
        continue

    if len(df) < MIN_TRADE_DAYS:
        continue

    # 确保按日期排序
    df = df.sort_values('trade_date').reset_index(drop=True)

    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    open_ = df['open'].values.astype(np.float64)
    dates = df['trade_date'].values

    n = len(df)

    # ---- MA18 ----
    ma18 = compute_ma(close, MA_PERIOD)

    # ---- MA18向上: ma18[i] > ma18[i-1] ----
    ma18_up = np.full(n, False, dtype=bool)
    ma18_up[1:] = ma18[1:] > ma18[:-1]

    # ---- 布林带 ----
    _, boll_upper, _ = compute_bollinger(close, period=20, n_std=2)

    # ---- 涨停: C/REF(C,1) > 1.095 AND C=H ----
    limit_up = np.full(n, False, dtype=bool)
    pct_chg = np.full(n, np.nan)
    pct_chg[1:] = close[1:] / close[:-1]
    limit_up[1:] = (pct_chg[1:] > LIMIT_UP_PCT) & (np.abs(close[1:] - high[1:]) < 1e-6)

    # ---- 遍历每一天，检测信号 ----
    for today in range(MA_PERIOD, n):
        if np.isnan(ma18[today]):
            continue

        # ----- 条件5: 今天收盘价上穿MA18（今天>=MA18, 昨天<MA18）-----
        if not (close[today] >= ma18[today] and close[today - 1] < ma18[today - 1]):
            continue

        # ----- 条件4: 今天盘中曾低于MA18 -----
        if not (low[today] < ma18[today]):
            continue

        # ----- 条件7: 今天收盘价 < MA18 × 1.05（距均线不超过5%）-----
        if not (close[today] < ma18[today] * 1.05):
            continue

        # ----- 条件8: MA18 4日斜率 > -2%（不能加速下行）-----
        if today >= MA_SLOPE_LOOKBACK:
            prev_ma = ma18[today - MA_SLOPE_LOOKBACK]
            if not np.isnan(prev_ma) and prev_ma > 0:
                slope = (ma18[today] / prev_ma - 1) * 100
                if slope < MA_SLOPE_MIN:
                    continue

        # ----- 条件9: 近6个月波动 < 100%（最高/最低比）-----
        lookback_start = max(0, today - VOLATILITY_LOOKBACK)
        window_high = np.max(high[lookback_start:today + 1])
        window_low = np.min(low[lookback_start:today + 1])
        if window_low > 0:
            volatility = (window_high / window_low - 1) * 100
            if volatility >= VOLATILITY_MAX_PCT:
                continue

        # ----- 往回找涨停日（标志K线），60天内 -----
        limit_idx = -1
        for k in range(today - 1, max(MA_PERIOD, today - 60) - 1, -1):
            if limit_up[k]:
                limit_idx = k
                break

        if limit_idx == -1:
            continue  # 60天内无涨停，跳过

        # ----- 条件10: 涨停日在最近20个交易日内 -----
        if today - limit_idx > LIMIT_BAR_RECENCY:
            continue

        # ----- 条件1: 涨停(已在上面确认) -----

        # ----- 条件2: 涨停当天MA18向上 -----
        if not ma18_up[limit_idx]:
            continue

        # ----- 条件3: 涨停后5天内收盘跌破MA18，或收盘未跌破但距MA18不超3% -----
        condition3_ok = False
        broke_idx = -1
        for j in range(limit_idx + 1, min(n, limit_idx + 1 + LOOKBACK_LIMIT)):
            if np.isnan(ma18[j]):
                break
            if close[j] < ma18[j]:
                # 子条件A: 收盘跌破MA18
                condition3_ok = True
                broke_idx = j
                break
            # 子条件B: 未跌破，但收盘距MA18不超3%
            dist = (close[j] - ma18[j]) / ma18[j]
            if 0 <= dist <= CLOSE_TO_MA_THRESHOLD:
                condition3_ok = True
                broke_idx = j  # 标记为接近日（用于日志）
                # 不break，继续找是否后续有跌破

        if not condition3_ok:
            continue  # 5天内既没跌破也没接近均线

        # ----- ✅ 全部条件满足，这是一个有效信号! -----
        entry_price = round(ma18[today] * 1.01, 3)
        limit_open = open_[limit_idx]

        trade_date = pd.Timestamp(dates[today])
        results.append({
            'ts_code': ts_code,
            'name': name_map.get(ts_code, ''),
            'industry': industry_map.get(ts_code, ''),
            'signal_date': trade_date,
            'close': close[today],
            'ma18': round(ma18[today], 3),
            'entry_price': entry_price,
            'high': high[today],
            'low': low[today],
            'open': open_[today],
            'limit_bar_date': pd.Timestamp(dates[limit_idx]),
            'limit_open': round(limit_open, 3),
            'broke_date': pd.Timestamp(dates[broke_idx]),
            'pct_chg': round((pct_chg[today] - 1) * 100, 2),
        })

print(f"\n  扫描完成，共 {len(results)} 条命中")

# ============================================================
# 3. 输出结果
# ============================================================
print("\n[3/5] 输出结果...")

if len(results) == 0:
    print("  ⚠️ 未选中任何股票")
    empty_df = pd.DataFrame(columns=[
        'ts_code', 'name', 'industry', 'signal_date', 'close', 'ma18',
        'entry_price', 'high', 'low', 'open', 'limit_bar_date',
        'broke_date', 'boll_upper', 'pct_chg'
    ])
    today_str = datetime.now().strftime('%Y%m%d')
    empty_df.to_csv(os.path.join(SIGNALS_DIR, f'zt_pullback_v2_{today_str}.csv'), index=False)
    empty_df.to_csv(os.path.join(SIGNALS_DIR, 'zt_pullback_v2_latest.csv'), index=False)
    sys.exit(0)

result_df = pd.DataFrame(results)
result_df = result_df.sort_values(['signal_date', 'ts_code']).reset_index(drop=True)

# 按日期分组统计
print(f"\n  信号明细（按日期）:")
date_groups = result_df.groupby(result_df['signal_date'].dt.date)
for date, group in date_groups:
    print(f"  {date}: {len(group)} 只")
    for _, row in group.iterrows():
        print(f"    {row['ts_code']}  {row['name']:<8}  "
              f"收盘{row['close']:.2f}  MA18{row['ma18']:.2f}  "
              f"买入价{row['entry_price']:.2f}  "
              f"涨停日{row['limit_bar_date'].date()}  "
              f"跌穿日{row['broke_date'].date()}  "
              f"涨跌{row['pct_chg']:+.2f}%")

# ============================================================
# 4. 保存文件
# ============================================================
print("\n[4/5] 保存信号文件...")

today_str = datetime.now().strftime('%Y%m%d')
out_path = os.path.join(SIGNALS_DIR, f'zt_pullback_v2_{today_str}.csv')
latest_path = os.path.join(SIGNALS_DIR, 'zt_pullback_v2_latest.csv')

result_df.to_csv(out_path, index=False)
result_df.to_csv(latest_path, index=False)

print(f"  ✅ 信号已保存: {out_path}")
print(f"  ✅ 最新信号:   {latest_path}")

# ============================================================
# 5. 最新交易日信号摘要
# ============================================================
print("\n[5/5] 最新信号摘要...")

latest_date = result_df['signal_date'].max()
latest = result_df[result_df['signal_date'] == latest_date]

print(f"\n  最新信号日: {latest_date.date()}")
print(f"  选中股票数: {len(latest)}")

if len(latest) > 0:
    print(f"\n  {'代码':<12} {'名称':<10} {'行业':<10} {'收盘':>8} {'MA18':>8} {'买入价':>8} {'涨停日':<12} {'涨停开盘':>8} {'涨跌%':>8}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*12} {'-'*8} {'-'*8}")
    for _, row in latest.iterrows():
        name = row.get('name', '') or ''
        industry = row.get('industry', '') or ''
        ld = str(row['limit_bar_date'].date()) if hasattr(row['limit_bar_date'], 'date') else str(row['limit_bar_date'])[:10]
        print(f"  {row['ts_code']:<12} {name:<10} {industry[:6]:<10} "
              f"{row['close']:>8.2f} {row['ma18']:>8.2f} {row['entry_price']:>8.2f} "
              f"{ld:<12} {row['limit_open']:>8.2f} {row['pct_chg']:>+7.2f}%")

print(f"\n{'='*60}")
print(f"完成! {datetime.now().strftime('%H:%M:%S')}")
