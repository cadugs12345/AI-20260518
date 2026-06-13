"""
新因子数据源下载
数据源:
1. 北向资金日频 (moneyflow_hsgt)
2. 个股资金流日频 (moneyflow) — 主力/散户净流入
3. 龙虎榜 (top_list, top_inst)
4. 业绩预告 (forecast)

Usage:
    python src/download_new_sources.py --all        # 下载全部
    python src/download_new_sources.py --northbound # 仅北向
    python src/download_new_sources.py --moneyflow  # 仅资金流
    python src/download_new_sources.py --toplist    # 仅龙虎榜
    python src/download_new_sources.py --forecast   # 仅业绩预告
"""
import os, sys, time
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import TS_TOKEN, START_DATE, END_DATE, DATA_RAW, DATA_FACTORS, TS_SLEEP

import tushare as ts
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

NEW_DATA_DIR = os.path.join(DATA_RAW, "new_sources")
os.makedirs(NEW_DATA_DIR, exist_ok=True)

FACTOR_DIR = os.path.join(DATA_FACTORS, "new_factors")
os.makedirs(FACTOR_DIR, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def download_stock_list():
    """获取股票代码列表（含ts_code）"""
    cache_path = os.path.join(DATA_RAW, "stock_list.parquet")
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    df = pro.stock_basic(exchange='', list_status='L', 
                        fields='ts_code,symbol,name,area,industry,list_date,market,exchange,is_hs')
    df.to_parquet(cache_path)
    log(f"股票名单: {len(df)}只")
    return df


# ==================== 1. 北向资金 ====================

def download_northbound_moneyflow(start_date=None, end_date=None):
    """
    沪深港通资金流向
    字段: trade_date, ggt_ss, ggt_sz, sgt, sgt_ss, sgt_sz, north_money, south_money
    """
    start = start_date or START_DATE
    end = end_date or END_DATE
    save_path = os.path.join(NEW_DATA_DIR, "northbound_moneyflow.parquet")
    
    if os.path.exists(save_path):
        df = pd.read_parquet(save_path)
        log(f"[北向资金] 已缓存: {len(df)}行, {df['trade_date'].min()}~{df['trade_date'].max()}")
        return df
    
    log("[北向资金] 下载沪深港通资金流...")
    df = pro.moneyflow_hsgt(start_date=start, end_date=end)
    if df is not None and not df.empty:
        df.to_parquet(save_path)
        log(f"[北向资金] 完成: {len(df)}行")
    else:
        log("[北向资金] 无数据")
        df = pd.DataFrame()
    return df


def build_northbound_factor(nb_df):
    """将北向资金流转化为因子"""
    if nb_df.empty:
        return None
    
    nb = nb_df.copy()
    nb["trade_date"] = pd.to_datetime(nb["trade_date"])
    nb = nb.sort_values("trade_date")
    
    for col in ['north_money', 'south_money', 'ggt_ss', 'ggt_sz']:
        if col in nb.columns:
            nb[col] = pd.to_numeric(nb[col], errors='coerce')
            nb[f"{col}_ma5"] = nb[col].rolling(5, min_periods=3).mean()
            nb[f"{col}_ma20"] = nb[col].rolling(20, min_periods=10).mean()
    
    # 净流入强度 = 北向净流入 / 均值
    if 'north_money' in nb.columns and 'south_money' in nb.columns:
        nb['north_net'] = nb['north_money'] - nb['south_money']
        nb['north_net_ma5'] = nb['north_net'].rolling(5, min_periods=3).mean()
        nb['north_net_ratio'] = nb['north_net'] / nb['north_net'].rolling(60, min_periods=20).mean()
        nb['north_net_ratio'] = nb['north_net_ratio'].replace([np.inf, -np.inf], np.nan)
    
    save_path = os.path.join(FACTOR_DIR, "northbound_factors.parquet")
    nb.to_parquet(save_path)
    log(f"[北向因子] 保存: {save_path} ({len(nb)}行)")
    return nb


# ==================== 2. 个股资金流 ====================

STOCK_LIST_CACHE = None

def get_stock_batch_list():
    """获取分批股票代码"""
    global STOCK_LIST_CACHE
    if STOCK_LIST_CACHE is None:
        stocks = download_stock_list()
        # 排除北交所
        stocks = stocks[~stocks['ts_code'].str.startswith('8')].copy()
        STOCK_LIST_CACHE = stocks['ts_code'].tolist()
    return STOCK_LIST_CACHE


def download_individual_moneyflow(start_date=None, end_date=None, max_stocks=None):
    """
    个股资金流 - 分批下载（Tushare限流）
    字段: ts_code, trade_date, buy_sm, buy_md, buy_lg, buy_elg, sell_sm, sell_md, sell_lg, sell_elg
    """
    start = start_date or "20250101"  # 近几个月够用
    end = end_date or END_DATE
    save_path = os.path.join(NEW_DATA_DIR, "individual_moneyflow.parquet")
    
    if os.path.exists(save_path):
        df = pd.read_parquet(save_path)
        log(f"[个股资金流] 已缓存: {len(df):,}行, {df['trade_date'].min()}~{df['trade_date'].max()}")
        return df
    
    codes = get_stock_batch_list()
    if max_stocks:
        codes = codes[:max_stocks]
    
    log(f"[个股资金流] 分批下载 {len(codes)}只股票...")
    
    all_data = []
    batch_size = 100
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i+batch_size]
        for code in batch:
            try:
                df = pro.moneyflow(ts_code=code, start_date=start, end_date=end)
                if df is not None and not df.empty:
                    all_data.append(df)
                time.sleep(TS_SLEEP)
            except Exception as e:
                log(f"  {code} 失败: {e}")
            time.sleep(TS_SLEEP)
        
        if (i+batch_size) % 500 == 0:
            log(f"  [{i+batch_size}/{len(codes)}]")
    
    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        result.to_parquet(save_path)
        log(f"[个股资金流] 完成: {len(result):,}行")
        return result
    return pd.DataFrame()


def build_moneyflow_factor(mf_df):
    """将个股资金流转化为因子"""
    if mf_df.empty:
        return None
    
    mf = mf_df.copy()
    mf["trade_date"] = pd.to_datetime(mf["trade_date"])
    
    # 先检查列是否存在
    cols = {"buy_sm", "buy_md", "buy_lg", "buy_elg", "sell_sm", "sell_md", "sell_lg", "sell_elg"}
    present = [c for c in cols if c in mf.columns]
    missing = cols - set(present)
    for c in missing:
        mf[c] = 0
    
    # 净主动买入
    mf["净主动买入"] = (mf["buy_sm"] + mf["buy_md"] + mf["buy_lg"] + mf["buy_elg"] -
                       mf["sell_sm"] - mf["sell_md"] - mf["sell_lg"] - mf["sell_elg"])
    
    # 主力净流入 = 大单+超大单
    mf["主力净流入"] = (mf["buy_lg"] + mf["buy_elg"] - mf["sell_lg"] - mf["sell_elg"])
    
    # 散户净流入 = 小单
    mf["散户净流入"] = mf["buy_sm"] - mf["sell_sm"]
    
    # 总交易额
    mf["总交易额"] = (mf["buy_sm"] + mf["buy_md"] + mf["buy_lg"] + mf["buy_elg"] +
                    mf["sell_sm"] + mf["sell_md"] + mf["sell_lg"] + mf["sell_elg"])
    
    # 主力净流入率
    mf["主力净流入率"] = mf["主力净流入"] / mf["总交易额"].replace(0, np.nan)
    mf["散户净流入率"] = mf["散户净流入"] / mf["总交易额"].replace(0, np.nan)
    
    # 主力-散户背离
    mf["主力散户背离"] = mf["主力净流入率"] - mf["散户净流入率"]
    
    # 滚动指标
    for code in mf["ts_code"].unique():
        mask = mf["ts_code"] == code
        for col in ["主力净流入率", "散户净流入率", "主力散户背离"]:
            mf.loc[mask, f"{col}_ma5"] = mf.loc[mask, col].rolling(5, min_periods=3).mean()
    
    # 保存
    save_path = os.path.join(FACTOR_DIR, "moneyflow_factors.parquet")
    mf.to_parquet(save_path)
    log(f"[资金流因子] 保存: {save_path} ({len(mf):,}行)")
    return mf


# ==================== 3. 龙虎榜 ====================

def download_top_list(start_date=None, end_date=None):
    """龙虎榜每日列表"""
    start = start_date or "20230101"
    end = end_date or END_DATE
    save_path = os.path.join(NEW_DATA_DIR, "top_list.parquet")
    
    if os.path.exists(save_path):
        df = pd.read_parquet(save_path)
        log(f"[龙虎榜] 已缓存: {len(df):,}行, {df['trade_date'].min()}~{df['trade_date'].max()}")
        return df
    
    log("[龙虎榜] 下载榜单...")
    dates = pd.date_range(start=start, end=end, freq='D')
    all_data = []
    
    for dt in dates:
        date_str = dt.strftime('%Y%m%d')
        try:
            df = pro.top_list(trade_date=date_str)
            if df is not None and not df.empty:
                all_data.append(df)
            time.sleep(TS_SLEEP * 0.5)
        except:
            pass
        if len(all_data) % 100 == 0 and all_data:
            log(f"  [{len(all_data)}/{len(dates)}] {date_str}")
    
    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        result.to_parquet(save_path)
        log(f"[龙虎榜] 完成: {len(result):,}行")
        return result
    return pd.DataFrame()


def build_toplist_factor(tl_df, panel):
    """将龙虎榜转化为因子（标记上榜次数、净买入等）"""
    if tl_df.empty:
        return None
    
    tl = tl_df.copy()
    tl["trade_date"] = pd.to_datetime(tl["trade_date"])
    
    # 每日每个股票的龙虎榜累计
    agg_dict = {"上榜次数": ("ts_code", "count"), "总净买入": ("net_amount", "sum")}
    if "buy_amount" in tl.columns:
        agg_dict["总买入"] = ("buy_amount", "sum")
    daily = tl.groupby(["trade_date", "ts_code"], as_index=False).agg(**agg_dict)
    
    # 是否为首次上榜
    daily = daily.sort_values(["ts_code", "trade_date"])
    daily["上次上榜间隔"] = daily.groupby("ts_code")["trade_date"].diff().dt.days
    
    # 最近N天上榜强度
    all_codes = panel[["trade_date", "ts_code"]].copy()
    merged = all_codes.merge(daily, on=["trade_date", "ts_code"], how="left")
    merged[["上榜次数", "总净买入"]] = merged[["上榜次数", "总净买入"]].fillna(0)
    
    # 近20日累计
    merged = merged.sort_values(["ts_code", "trade_date"])
    for col in ["上榜次数", "总净买入"]:
        merged[f"{col}_20d"] = merged.groupby("ts_code")[col].transform(
            lambda x: x.rolling(20, min_periods=5).sum())
    
    merged["是否上榜"] = (merged["上榜次数"] > 0).astype(int)
    
    save_path = os.path.join(FACTOR_DIR, "toplist_factors.parquet")
    merged.to_parquet(save_path)
    log(f"[龙虎榜因子] 保存: {save_path} ({len(merged):,}行)")
    return merged


# ==================== 4. 业绩预告 ====================

def download_forecast(start_date=None, end_date=None):
    """业绩预告"""
    start = start_date or "20180101"
    end = end_date or END_DATE
    save_path = os.path.join(NEW_DATA_DIR, "forecast.parquet")
    
    if os.path.exists(save_path):
        df = pd.read_parquet(save_path)
        log(f"[业绩预告] 已缓存: {len(df):,}行, {df['end_date'].min()}~{df['end_date'].max()}")
        return df
    
    log("[业绩预告] 按年下载...")
    stocks = download_stock_list()
    codes = stocks['ts_code'].tolist()
    
    all_data = []
    # 按ts_code+end_date逐年查询（业绩预告量很少，每个股票每年1-2条）
    for i, code in enumerate(codes):
        try:
            df = pro.forecast(ts_code=code, start_date=start, end_date=end)
            if df is not None and not df.empty:
                all_data.append(df)
        except Exception as e:
            pass
        time.sleep(TS_SLEEP * 0.2)  # 约0.1-0.2秒/只
        if (i+1) % 500 == 0:
            log(f"  [{i+1}/{len(codes)}] 已获{sum(len(d) for d in all_data):,}条")
    
    if all_data:
        result = pd.concat(all_data, ignore_index=True).drop_duplicates()
        result.to_parquet(save_path)
        log(f"[业绩预告] 完成: {len(result):,}行")
        return result
    return pd.DataFrame()


def build_forecast_factor(fc_df):
    """业绩预告因子"""
    if fc_df.empty:
        return None
    
    fc = fc_df.copy()
    fc["ann_date"] = pd.to_datetime(fc["ann_date"])
    fc["end_date"] = pd.to_datetime(fc["end_date"])
    
    # 预告类型编码
    type_map = {
        '预增': 2, '略增': 1, '扭亏': 3, '续盈': 1,
        '预减': -2, '略减': -1, '首亏': -3, '续亏': -3, '减亏': 1,
    }
    fc["type_code"] = fc["type"].map(type_map).fillna(0)
    
    # 变动幅度归一化
    if "p_change_min" in fc.columns and "p_change_max" in fc.columns:
        fc["p_change_mid"] = (fc["p_change_min"] + fc["p_change_max"]) / 2
    else:
        fc["p_change_mid"] = 0
    
    # 综合预告得分
    fc["预告得分"] = fc["type_code"] * 0.5 + fc["p_change_mid"] / 100 * 0.5
    
    # 取最新预告
    fc = fc.sort_values(["ts_code", "ann_date"])
    fc["交易日"] = fc["ann_date"]
    
    save_path = os.path.join(FACTOR_DIR, "forecast_factors.parquet")
    fc.to_parquet(save_path)
    log(f"[业绩预告因子] 保存: {save_path} ({len(fc):,}行)")
    return fc


# ==================== 主入口 ====================

if __name__ == "__main__":
    args = set(sys.argv[1:])
    
    log("="*60)
    log("新因子数据源下载")
    log(f"时间范围: {START_DATE}~{END_DATE}")
    log("="*60)
    
    t0 = time.time()
    
    if "--all" in args or not args - {"--all"}:
        args = {"--northbound", "--moneyflow", "--toplist", "--forecast"}
    
    # 加载现有面板用于对齐日期
    panel_path = os.path.join(DATA_FACTORS, "factor_panel_with_fwd_v2.parquet")
    panel = None
    if "--toplist" in args and os.path.exists(panel_path):
        panel = pd.read_parquet(panel_path)
        panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    
    if "--northbound" in args:
        log("\n--- 北向资金 ---")
        nb = download_northbound_moneyflow()
        build_northbound_factor(nb)
    
    if "--moneyflow" in args:
        log("\n--- 个股资金流 ---")
        mf = download_individual_moneyflow(max_stocks=500)  # 先测500只
        build_moneyflow_factor(mf)
    
    if "--toplist" in args:
        log("\n--- 龙虎榜 ---")
        tl = download_top_list()
        if panel is not None:
            build_toplist_factor(tl, panel)
    
    if "--forecast" in args:
        log("\n--- 业绩预告 ---")
        fc = download_forecast()
        build_forecast_factor(fc)
    
    log(f"\n总用时: {(time.time()-t0)/60:.1f}分")
