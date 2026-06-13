#!/usr/bin/env python3
"""
分时数据采集 + 特征提取 + 模型训练三合一脚本
===============================================
流程:
  1. 从API下载历史1分钟分时数据（未复权）
  2. 特征提取：日内形态特征、量价特征、时序特征
  3. 构建标签：次日涨跌/日内预期收益
  4. 训练LightGBM模型预测日内/隔日走势

数据来源: https://data.diemeng.chat/api/stock/history (蝶梦API)
"""
import os, sys, json, time, warnings, gc
from datetime import datetime, timedelta
from typing import List, Tuple, Optional

import requests
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ============================================================
# 配置
# ============================================================
PROJ = "/mnt/d/AI-20260604"
API_URL = "https://data.diemeng.chat/api/stock/history"
API_KEY = "4b4d5c2093ec2260967007116f09a5732e5cbab7f8a17d00da"

DATA_DIR = os.path.join(PROJ, "data", "minute")        # 原始分时数据
FEATURE_DIR = os.path.join(PROJ, "data", "minute_features")  # 特征数据
MODEL_DIR = os.path.join(PROJ, "models")
SIGNALS_DIR = os.path.join(PROJ, "signals")
DAILY_DIR = os.path.join(PROJ, "data", "raw", "daily")
STOCK_LIST_FILE = os.path.join(PROJ, "data", "raw", "stock_list.parquet")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# 参数
MINUTE_LEVEL = "1min"
PAGE_SIZE = 10000          # 每页最多10000条
MAX_STOCKS = 500           # 最多采集/训练股票数（防止API配额耗尽）
FETCH_DAYS_BACK = 60       # 每次采集最近60个交易日
TRAIN_DAYS_BACK = 240      # 训练用最近240个交易日（约1年）

# ============================================================
# 工具函数
# ============================================================
def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def get_tushare_dates() -> List[str]:
    """从日K线文件推断实际交易日列表"""
    import glob
    files = glob.glob(os.path.join(DAILY_DIR, "*.parquet"))
    if not files:
        return []
    # 取第一只股票，读全部的trade_date
    df = pd.read_parquet(files[0])
    dates = sorted(df['trade_date'].astype(str).str[:10].unique().tolist())
    return dates


def fetch_minute_data(
    stock_code: str,
    start_time: str,
    end_time: str,
    level: str = MINUTE_LEVEL,
    max_pages: int = 5,
) -> pd.DataFrame:
    """
    从API获取历史分时数据
    max_pages: 最多拉取页数（避免无限循环）
    """
    headers = {
        "apiKey": API_KEY,
        "Content-Type": "application/json",
    }
    
    all_records = []
    page = 0
    
    while page < max_pages:
        payload = {
            "stock_code": stock_code,
            "level": level,
            "start_time": start_time,
            "end_time": end_time,
            "page": page,
            "page_size": PAGE_SIZE,
        }
        
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            data = resp.json()
        except Exception as e:
            log(f"  ⚠️ {stock_code} 请求失败(page={page}): {e}")
            break
        
        if data.get("code") != 200:
            log(f"  ⚠️ {stock_code} API返回错误: {data.get('msg', 'unknown')}")
            break
        
        records = data.get("data", {}).get("list", [])
        if not records:
            break  # 无更多数据
        
        all_records.extend(records)
        
        total = data.get("data", {}).get("total", 0)
        if len(all_records) >= total:
            break  # 已拉完
        
        page += 1
        time.sleep(0.3)  # 请求间隔
    
    if not all_records:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_records)
    
    # 统一列名 (API返回vol而非volume)
    col_map = {
        "trade_time": "trade_time",
        "stock_code": "ts_code",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "vol": "volume",
        "amount": "amount",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    # 有些缓存文件已经是volume，兼容
    if 'volume' not in df.columns and 'vol' in df.columns:
        df['volume'] = df['vol']
    
    # 确保数据类型
    df['trade_time'] = pd.to_datetime(df['trade_time'])
    df['open'] = pd.to_numeric(df['open'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    
    # 补stock_code字段
    if 'ts_code' not in df.columns:
        df['ts_code'] = stock_code
    df['ts_code'] = df['ts_code'].fillna(stock_code)
    
    df = df.sort_values('trade_time').reset_index(drop=True)
    return df


def load_daily_data(stock_code: str) -> pd.DataFrame:
    """从日K parquet加载股票数据"""
    fpath = os.path.join(DAILY_DIR, f"{stock_code}.parquet")
    if not os.path.exists(fpath):
        return pd.DataFrame()
    df = pd.read_parquet(fpath)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    return df.sort_values('trade_date').reset_index(drop=True)


# ============================================================
# 特征工程
# ============================================================
def extract_minute_features(df_min: pd.DataFrame, df_daily: pd.DataFrame) -> pd.Series:
    """
    从1分钟分时数据提取特征
    返回一个包含当天特征的Series
    """
    if len(df_min) < 30:
        return pd.Series(dtype=float)
    
    # 基础价格
    open_p = df_min['open'].iloc[0]
    close_p = df_min['close'].iloc[-1]
    high_p = df_min['high'].max()
    low_p = df_min['low'].min()
    mid_p = (high_p + low_p) / 2
    
    # 日内涨跌幅
    day_ret = close_p / open_p - 1
    
    # ---- 量价特征 ----
    avg_volume = df_min['volume'].mean()
    max_volume = df_min['volume'].max()
    last_30m_vol = df_min.tail(30)['volume'].mean() if len(df_min) >= 30 else avg_volume
    
    # 量能分布: 前30分钟 vs 最后30分钟
    first_30m = df_min.head(30)
    last_30m = df_min.tail(30)
    vol_ratio = last_30m_vol / (avg_volume + 1)
    first_last_vol_ratio = last_30m['volume'].sum() / (first_30m['volume'].sum() + 1)
    
    # ---- 分时形态特征 ----
    # 均价线
    avg_price = df_min['amount'].sum() / (df_min['volume'].sum() * 100 + 1)  # 股→手
    
    # 日内波动率
    volatility = (high_p - low_p) / mid_p
    
    # 价格位置: 收盘在日内区间的位置
    if high_p > low_p:
        price_position = (close_p - low_p) / (high_p - low_p)
    else:
        price_position = 0.5
    
    # 低点时间比: 最低价出现在什么位置(0~1)
    low_idx = df_min['low'].idxmin()
    low_time_ratio = low_idx / len(df_min) if len(df_min) > 0 else 0.5
    
    # 高点时间比
    high_idx = df_min['high'].idxmax()
    high_time_ratio = high_idx / len(df_min) if len(df_min) > 0 else 0.5
    
    # ---- 时序特征 ----
    prices = df_min['close'].values
    n = len(prices)
    
    # 移动平均乖离
    if n >= 10:
        ma10 = np.mean(prices[-10:])
        close_ma10_ratio = close_p / ma10 - 1
    else:
        close_ma10_ratio = 0
    
    if n >= 30:
        ma30 = np.mean(prices[-30:])
        close_ma30_ratio = close_p / ma30 - 1
    else:
        close_ma30_ratio = 0
    
    # 尾盘走势斜率 (最后20根K线)
    if n >= 20:
        tail_prices = prices[-20:]
        x = np.arange(20)
        slope, _ = np.polyfit(x, tail_prices, 1)
        tail_slope = slope / (np.mean(tail_prices) + 1)
    else:
        tail_slope = 0
    
    # 最大回撤 (从高点回落)
    cummax = np.maximum.accumulate(prices)
    drawdown = (cummax - prices) / (cummax + 1e-8)
    max_drawdown = drawdown.max()
    
    # ---- 量价背离 ----
    # 价格涨但量缩 (顶背离): 后30分钟涨跌幅 vs 量比
    if n >= 60:
        early_prices = prices[:n//2]
        late_prices = prices[n//2:]
        early_vol = df_min['volume'].values[:n//2]
        late_vol = df_min['volume'].values[n//2:]
        
        early_return = early_prices[-1] / early_prices[0] - 1
        late_return = late_prices[-1] / late_prices[0] - 1
        
        early_avg_vol = np.mean(early_vol)
        late_avg_vol = np.mean(late_vol)
        
        vol_divergence = (late_avg_vol / (early_avg_vol + 1)) / (late_return - early_return + 1)
    else:
        early_return = 0
        late_return = day_ret
        vol_divergence = 0
    
    # ---- V型反转特征 ----
    # 深跌后反弹幅度
    if max_drawdown > 0.01:
        rebound = (close_p - low_p) / (low_p + 1e-8) / max_drawdown
    else:
        rebound = 0
    
    # 低点后涨幅
    if low_idx < len(df_min) - 1:
        post_low_return = close_p / low_p - 1
    else:
        post_low_return = 0
    
    # ---- 分时横盘特征 ----
    # 将日内分成6段(约30分钟一段)，计算每段波动均值
    periods = 6
    period_vols = []
    period_size = max(n // periods, 1)
    for p in range(periods):
        start = p * period_size
        end = min((p + 1) * period_size, n)
        if end - start >= 2:
            seg_prices = prices[start:end]
            seg_vol = (seg_prices.max() - seg_prices.min()) / (np.mean(seg_prices) + 1e-8)
            period_vols.append(seg_vol)
    
    avg_period_vol = np.mean(period_vols) if period_vols else 0
    
    # 横盘比例: 波动低于0.5%的时段占比
    lull_ratio = sum(1 for v in period_vols if v < 0.005) / len(period_vols) if period_vols else 0
    
    # ---- 日K级别特征 ----
    daily_features = {}
    if len(df_daily) > 0:
        prev_close = df_daily['close'].iloc[-2] if len(df_daily) >= 2 else df_daily['close'].iloc[-1]
        daily_ma18 = df_daily['close'].rolling(18).mean().iloc[-1] if len(df_daily) >= 18 else prev_close
        daily_ma60 = df_daily['close'].rolling(60).mean().iloc[-1] if len(df_daily) >= 60 else prev_close
        
        daily_features = {
            'prev_close': prev_close,
            'daily_ma18': daily_ma18,
            'daily_ma60': daily_ma60,
            'close_ma18_ratio': close_p / daily_ma18 - 1 if daily_ma18 > 0 else 0,
            'close_ma60_ratio': close_p / daily_ma60 - 1 if daily_ma60 > 0 else 0,
        }
    
    # 打包特征
    features = {
        # 基础
        'open': open_p,
        'high': high_p,
        'low': low_p,
        'close': close_p,
        'avg_price': avg_price,
        'day_return': day_ret,
        'volatility': volatility,
        'price_position': price_position,
        'low_time_ratio': low_time_ratio,
        'high_time_ratio': high_time_ratio,
        
        # 量能
        'avg_volume': avg_volume,
        'max_volume': max_volume,
        'last_30m_vol': last_30m_vol,
        'vol_ratio': vol_ratio,
        'first_last_vol_ratio': first_last_vol_ratio,
        
        # 均线乖离
        'close_ma10_ratio': close_ma10_ratio,
        'close_ma30_ratio': close_ma30_ratio,
        
        # 走势形态
        'tail_slope': tail_slope,
        'max_drawdown': max_drawdown,
        'rebound_ratio': rebound,
        'post_low_return': post_low_return,
        'avg_period_vol': avg_period_vol,
        'lull_ratio': lull_ratio,
        
        # 量价背离
        'vol_divergence': vol_divergence,
        
        # 日K联动
        **daily_features,
    }
    
    return pd.Series(features)


# ============================================================
# 标签构建
# ============================================================
def build_labels(df_min: pd.DataFrame, df_daily: pd.DataFrame) -> pd.Series:
    """
    构建标签
    标签1: 明日涨跌 (次日close/今日close - 1)
    标签2: 尾盘买入次日早盘收益 (次日10:30均价比今日收盘)
    标签3: 日内买入信号 (今日收盘是否在日内高低点中间偏上)
    """
    labels = {}
    
    # 标签1: 明日涨跌 (需要次日日K线)
    if len(df_daily) >= 3:
        today_close = df_daily['close'].iloc[-2] if len(df_daily) >= 2 else df_daily['close'].iloc[-1]
        tmw_close = df_daily['close'].iloc[-1]  # 实际上是明天的
        labels['next_day_ret'] = tmw_close / today_close - 1
    else:
        labels['next_day_ret'] = np.nan
    
    # 标签2: 日内买入点质量
    if len(df_min) >= 30:
        close_p = df_min['close'].iloc[-1]
        high_p = df_min['high'].max()
        low_p = df_min['low'].min()
        # 收盘在区间中上部 -> 好买入信号
        labels['close_quality'] = (close_p - low_p) / (high_p - low_p + 1e-8)
        # 日内收益 (开盘买入收盘卖出)
        open_p = df_min['open'].iloc[0]
        labels['intraday_ret'] = close_p / open_p - 1
    else:
        labels['close_quality'] = 0.5
        labels['intraday_ret'] = 0
    
    # 标签3: 尾盘动量 (最后30分钟走势)
    if len(df_min) >= 60:
        mid_prices = df_min.iloc[len(df_min)//2:len(df_min)//2+30]['close'].mean()
        tail_prices = df_min.tail(30)['close'].mean()
        labels['tail_momentum'] = tail_prices / mid_prices - 1
    else:
        labels['tail_momentum'] = 0
    
    return pd.Series(labels)


# ============================================================
# 单只股票全流程
# ============================================================
def process_stock(stock_code: str, trade_dates: List[str]) -> Optional[pd.DataFrame]:
    """
    对单只股票: 下载分时 → 特征提取 → 返回特征DataFrame
    """
    if len(trade_dates) < 5:
        return None
    
    # 加载日K数据
    df_daily = load_daily_data(stock_code)
    if len(df_daily) < 60:
        return None  # 上市不足60天
    
    lookback = min(FETCH_DAYS_BACK, len(trade_dates))
    start_date = trade_dates[-lookback]  # 最近N天
    end_date = trade_dates[-1]
    
    start_str = f"{start_date} 09:20:00"
    end_str = f"{end_date} 15:05:00"
    
    log(f"  📥 下载 {stock_code} 分时: {start_str} ~ {end_str}")
    
    df_min = fetch_minute_data(stock_code, start_str, end_str)
    if len(df_min) < 60:
        log(f"  ⚠️ {stock_code} 分时数据不足 ({len(df_min)}行)")
        return None
    
    # 保存原始分时数据 (分日期)
    df_min['date'] = df_min['trade_time'].dt.date
    
    all_features = []
    for trade_date_str in trade_dates[-lookback:]:
        trade_date = pd.Timestamp(trade_date_str).date()
        day_min = df_min[df_min['date'] == trade_date]
        if len(day_min) < 30:
            continue
        
        # 当日分钟数据
        day_min = day_min.copy().reset_index(drop=True)
        
        # 当日之前的日K数据 (不含当天)
        today_daily_idx = df_daily[df_daily['trade_date'].dt.date == trade_date]
        if len(today_daily_idx) == 0:
            continue
        
        daily_up_to = df_daily[df_daily['trade_date'].dt.date <= trade_date]
        
        # 特征提取
        features = extract_minute_features(day_min, daily_up_to)
        if features.empty:
            continue
        
        features['ts_code'] = stock_code
        features['date'] = trade_date_str
        
        # 标签 (需要次日数据)
        daily_including_tmw = df_daily[df_daily['trade_date'].dt.date >= trade_date]
        labels = build_labels(day_min, daily_including_tmw)
        features = pd.concat([features, labels])
        
        all_features.append(features)
    
    if not all_features:
        return None
    
    result = pd.DataFrame(all_features)
    log(f"  ✅ {stock_code} 提取 {len(result)} 天特征")
    return result


# ============================================================
# 主流程
# ============================================================
def collect_all_stocks(trade_dates: List[str]) -> pd.DataFrame:
    """采集所有股票的分时数据并提取特征"""
    # 获取股票列表
    try:
        stock_df = pd.read_parquet(STOCK_LIST_FILE)
        stocks = stock_df['ts_code'].unique().tolist()
    except Exception:
        import glob
        stocks = [f.replace('.parquet','') for f in glob.glob(os.path.join(DAILY_DIR, "*.parquet"))]
    
    # 按市值/成交额排序取前N只 (优选活跃股)
    log(f"📋 股票池: {len(stocks)}只, 取前{MAX_STOCKS}只")
    
    # 优先选近期有涨停的活跃股
    active_stocks = []
    for code in stocks[:MAX_STOCKS]:
        df = load_daily_data(code)
        if len(df) < 60:
            continue
        # 最近60日内涨停次数
        recent = df.tail(60)
        if len(recent) >= 10:
            limit_cnt = ((recent['close'] / recent['close'].shift(1) > 1.095) & 
                         (recent['high'] == recent['close'])).sum()
            if limit_cnt >= 1:
                active_stocks.append(code)
    
    log(f"  → 筛选出 {len(active_stocks)} 只近期有涨停的活跃股")
    if len(active_stocks) > MAX_STOCKS:
        active_stocks = active_stocks[:MAX_STOCKS]
    
    all_feat_dfs = []
    for i, code in enumerate(active_stocks):
        log(f"[{i+1}/{len(active_stocks)}] {code}")
        feat_df = process_stock(code, trade_dates)
        if feat_df is not None:
            all_feat_dfs.append(feat_df)
        
        # 保存中间结果 (每50只保存一次)
        if (i + 1) % 50 == 0:
            interim = pd.concat(all_feat_dfs, ignore_index=True) if all_feat_dfs else pd.DataFrame()
            if len(interim) > 0:
                interim.to_parquet(os.path.join(FEATURE_DIR, f"features_interim_{i+1}.parquet"))
                log(f"  💾 中间保存: {i+1}只, {len(interim)}行特征")
            gc.collect()
        
        # API限速
        time.sleep(0.5)
    
    if not all_feat_dfs:
        log("❌ 未获取到任何有效特征数据")
        return pd.DataFrame()
    
    final_df = pd.concat(all_feat_dfs, ignore_index=True)
    log(f"\n📊 总特征量: {len(final_df)}行 × {len(final_df.columns)}列")
    return final_df


def train_model(feature_df: pd.DataFrame):
    """训练LightGBM模型"""
    try:
        import lightgbm as lgb
    except ImportError:
        log("⚠️ lightgbm未安装，跳过训练")
        log("安装: pip install lightgbm")
        return None, None
    
    # 特征列 (排除非特征列)
    exclude_cols = {'ts_code', 'date', 'next_day_ret', 'close_quality', 'intraday_ret', 'tail_momentum'}
    feature_cols = [c for c in feature_df.columns if c not in exclude_cols and feature_df[c].dtype in (np.float64, np.float32, np.int64)]
    
    log(f"\n📊 特征列数: {len(feature_cols)}")
    
    # 标签选择：预测明日涨跌
    target = 'next_day_ret'
    df = feature_df.dropna(subset=[target]).copy()
    log(f"   有效样本(有次日收益): {len(df)}")
    
    if len(df) < 100:
        log("❌ 样本太少，无法训练")
        return None, None
    
    X = df[feature_cols].fillna(0).values
    y = df[target].values
    
    # 时间序列切分 (按时间先后, 前80%训练, 后20%测试)
    time_order = df['date'] if 'date' in df.columns else df.index
    split_idx = int(len(df) * 0.8)
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    log(f"   训练集: {len(X_train)} | 测试集: {len(X_test)}")
    log(f"   标签分布: mean={y.mean():+.4f}, std={y.std():.4f}")
    
    # 训练
    params = {
        'objective': 'regression',
        'metric': 'mse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1,
    }
    
    train_data = lgb.Dataset(X_train, y_train)
    valid_data = lgb.Dataset(X_test, y_test, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, valid_data],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100),
        ],
    )
    
    # 评估
    y_pred = model.predict(X_test)
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # 方向准确率
    direction_acc = ((y_pred > 0) == (y_test > 0)).mean()
    
    log(f"\n📈 模型评估:")
    log(f"   MSE: {mse:.6f}")
    log(f"   MAE: {mae:.4f}")
    log(f"   方向准确率: {direction_acc:.2%}")
    log(f"   预测均值: {y_pred.mean():+.4f} | 实际均值: {y_test.mean():+.4f}")
    
    # 特征重要性
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance('gain'),
    }).sort_values('importance', ascending=False)
    
    log(f"\n📊 Top 10重要特征:")
    for _, row in importance.head(10).iterrows():
        log(f"   {row['feature']:<25s} {row['importance']:.4f}")
    
    # 保存模型
    model_path = os.path.join(MODEL_DIR, "minute_lgb_model.txt")
    model.save_model(model_path)
    log(f"\n✅ 模型已保存: {model_path}")
    
    # 保存特征重要性
    imp_path = os.path.join(MODEL_DIR, "minute_feature_importance.csv")
    importance.to_csv(imp_path, index=False)
    log(f"✅ 特征重要性已保存: {imp_path}")
    
    return model, importance


# ============================================================
# 入口
# ============================================================
def main():
    print("=" * 60)
    print("📊 分时数据采集 + 特征提取 + 模型训练")
    print("=" * 60)
    start_time = datetime.now()
    
    # 1. 获取交易日历
    log("\n[1/4] 获取交易日列表...")
    trade_dates = get_tushare_dates()
    log(f"   共 {len(trade_dates)} 个交易日, 最近: {trade_dates[-1] if trade_dates else 'N/A'}")
    
    if not trade_dates:
        log("❌ 无法获取交易日历")
        return
    
    # 2. 采集+特征提取
    log("\n[2/4] 采集分时数据 + 特征提取...")
    feature_df = collect_all_stocks(trade_dates)
    
    if len(feature_df) == 0:
        log("❌ 未采集到有效数据")
        return
    
    # 保存全部特征
    feat_path = os.path.join(FEATURE_DIR, "minute_features_all.parquet")
    feature_df.to_parquet(feat_path, index=False)
    log(f"\n💾 全部特征已保存: {feat_path}")
    
    # 3. 训练
    log("\n[3/4] 训练模型...")
    model, importance = train_model(feature_df)
    
    # 4. 统计摘要
    log("\n[4/4] 完成")
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "=" * 60)
    print(f"📊 完成! 耗时 {elapsed:.0f}s")
    print(f"   采集股票数: 待统计")
    print(f"   特征总数: {len(feature_df)}行 × {len(feature_df.columns)}列")
    print(f"   模型: {os.path.join(MODEL_DIR, 'minute_lgb_model.txt') if model else '未训练'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
