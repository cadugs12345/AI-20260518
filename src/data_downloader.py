"""
第一步：A股数据下载
- 日线行情（前复权）
- 财务数据（季报/年报）
- 股票基本信息（行业、上市日期等）
"""
import os, sys, time, logging
import pandas as pd
import numpy as np
import tushare as ts

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    TS_TOKEN, START_DATE, END_DATE, DATA_RAW, LOGS_DIR,
    EXCLUDE_BOARD, MIN_LIST_DAYS, TS_SLEEP, TS_BATCH_SIZE,
)

# ====== 日志 ======
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "data_download.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ====== 初始化 Tushare ======
ts.set_token(TS_TOKEN)
pro = ts.pro_api()


# ==================== 1. 获取股票名单 ====================
def get_stock_list() -> pd.DataFrame:
    """
    获取全A股股票名单
    数据下载阶段全量拉取（仅剔除ST/北交所/退市），
    上市天数过滤在因子计算阶段动态处理
    """
    logger.info("正在获取A股股票名单...")
    df = pro.stock_basic(exchange="", list_status="L",
                         fields="ts_code,symbol,name,area,industry,market,list_date,is_hs")
    logger.info(f"原始股票数量: {len(df)}")

    # 剔除ST/*ST/退市整理
    df = df[~df["name"].str.contains("ST|退", na=False)].copy()
    logger.info(f"剔除ST后: {len(df)}")

    # 剔除北交所
    df = df[~df["market"].isin(EXCLUDE_BOARD)].copy()
    logger.info(f"剔除北交所后: {len(df)}")

    # 保存上市日期为datetime供后续过滤用
    df["list_date"] = pd.to_datetime(df["list_date"])
    
    # 数据下载阶段不做上市天数截断，留到因子计算时动态判断
    logger.info(f"最终待下载股票池: {len(df)} 只 (上市日期范围: {df['list_date'].min().date()} ~ {df['list_date'].max().date()})")
    logger.info(f"板块分布: {df['market'].value_counts().to_dict()}")

    df = df.reset_index(drop=True)
    logger.info(f"最终有效股票池: {len(df)} 只")
    return df


# ==================== 2. 下载日线行情（前复权） ====================
def download_daily(stock_list: pd.DataFrame, batch_size: int = TS_BATCH_SIZE):
    """
    批量下载日线数据（前复权）
    保存格式: data/raw/daily/{ts_code}.parquet
    """
    codes = stock_list["ts_code"].tolist()
    save_dir = os.path.join(DATA_RAW, "daily")
    os.makedirs(save_dir, exist_ok=True)

    total = len(codes)
    existing = 0
    downloaded = 0
    failed = []

    for i in range(0, total, batch_size):
        batch = codes[i:i + batch_size]
        for code in batch:
            save_path = os.path.join(save_dir, f"{code}.parquet")
            if os.path.exists(save_path):
                existing += 1
                continue

            try:
                df = pro.daily(
                    ts_code=code,
                    start_date=START_DATE,
                    end_date=END_DATE,
                    fields="trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount",
                )
                if df is not None and len(df) > 0:
                    df["trade_date"] = pd.to_datetime(df["trade_date"])
                    df = df.sort_values("trade_date").reset_index(drop=True)
                    df.to_parquet(save_path, index=False)
                    downloaded += 1
                time.sleep(TS_SLEEP)
            except Exception as e:
                failed.append(code)
                logger.warning(f"下载失败 {code}: {e}")
                time.sleep(1)

        progress = min(i + batch_size, total)
        logger.info(f"日线进度: {progress}/{total} | 已有={existing} | 新增={downloaded} | 失败={len(failed)}")

    logger.info(f"日线下载完成: 总计={total}, 已有={existing}, 新增={downloaded}, 失败={len(failed)}")
    if failed:
        logger.warning(f"失败股票列表: {failed}")
    return downloaded


# ==================== 3. 下载财务数据 ====================
def download_financial(stock_list: pd.DataFrame):
    """
    下载财务指标数据
    使用 Tushare fina_indicator 接口（含ROE/毛利率/净利率/利润增速等）
    """
    codes = stock_list["ts_code"].tolist()
    save_dir = os.path.join(DATA_RAW, "financial")
    os.makedirs(save_dir, exist_ok=True)

    fields = (
        "ts_code,ann_date,end_date,eps,roe,roe_dt,roa,grossprofit_margin,"
        "profit_margin,operating_profit_margin,netprofit_margin,"
        "yoy_gr_yoy,gr_yoy,grossprofit_yoy,"
        "debt_to_assets,current_ratio,"
        "yoy_sales_gr_yoy,"
        "eps_yoy,"
    )

    total = len(codes)
    downloaded = 0
    failed = []

    for i, code in enumerate(codes):
        save_path = os.path.join(save_dir, f"{code}.parquet")
        if os.path.exists(save_path):
            downloaded += 1
            continue

        try:
            df = pro.fina_indicator(
                ts_code=code,
                start_date=str(int(START_DATE[:4]) - 1) + "0101",
                end_date=END_DATE,
                fields=fields,
            )
            if df is not None and len(df) > 0:
                df["end_date"] = pd.to_datetime(df["end_date"])
                df = df.sort_values("end_date").reset_index(drop=True)
                df.to_parquet(save_path, index=False)
            time.sleep(TS_SLEEP)
        except Exception as e:
            failed.append(code)
            logger.warning(f"财务数据下载失败 {code}: {e}")
            time.sleep(1)

        if (i + 1) % 200 == 0:
            logger.info(f"财务数据进度: {i + 1}/{total} | 成功={downloaded} | 失败={len(failed)}")

    logger.info(f"财务数据下载完成: 总计={total}, 成功={downloaded}, 失败={len(failed)}")
    return downloaded


# ==================== 4. 每日行情全景快照 ====================
def download_daily_basic(stock_list: pd.DataFrame):
    """
    下载每日全景行情（含PE/PB/换手率/量比等）
    用于截面因子计算（市值、BP、EP、SP等）
    每交易日一个文件: data/raw/daily_basic/{trade_date}.parquet
    """
    # 生成交易日列表
    trade_dates = pd.date_range(start=START_DATE, end=END_DATE, freq="B")
    save_dir = os.path.join(DATA_RAW, "daily_basic")
    os.makedirs(save_dir, exist_ok=True)

    # 用前复权行情中的实际交易日
    sample_code = stock_list["ts_code"].iloc[0]
    sample_df = pd.read_parquet(os.path.join(DATA_RAW, "daily", f"{sample_code}.parquet"))
    actual_trade_dates = sample_df["trade_date"].dt.strftime("%Y%m%d").tolist()

    total = len(actual_trade_dates)
    downloaded = 0

    for i, trade_date in enumerate(actual_trade_dates):
        save_path = os.path.join(save_dir, f"{trade_date}.parquet")
        if os.path.exists(save_path):
            downloaded += 1
            continue

        try:
            # 一次拉取所有股票的daily_basic
            df = pro.daily_basic(
                ts_code="",
                trade_date=trade_date,
                fields="ts_code,trade_date,turnover_rate,turnover_rate_f,"
                       "volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,"
                       "dv_ratio,dv_ttm,total_mv,circ_mv",
            )
            if df is not None and len(df) > 0:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df.to_parquet(save_path, index=False)
            time.sleep(TS_SLEEP * 2)  # 这个接口慢一些
        except Exception as e:
            logger.warning(f"daily_basic 失败 {trade_date}: {e}")
            time.sleep(1)

        if (i + 1) % 60 == 0:
            logger.info(f"daily_basic 进度: {i + 1}/{total} | 已有={downloaded}")

    logger.info(f"daily_basic 完成: 共计{total}个交易日")


# ==================== 主流程 ====================
def run():
    """运行完整数据下载流程"""
    logger.info("=" * 60)
    logger.info("A股量化因子系统 - 数据下载开始")
    logger.info(f"时间范围: {START_DATE} ~ {END_DATE}")
    logger.info("=" * 60)

    # 1. 获取股票名单
    stocks = get_stock_list()
    stocks.to_parquet(os.path.join(DATA_RAW, "stock_list.parquet"), index=False)
    stocks.to_csv(os.path.join(DATA_RAW, "stock_list.csv"), index=False, encoding="utf-8-sig")
    logger.info(f"股票名单已保存: {len(stocks)} 只")

    # 2. 下载日线
    logger.info("\n--- 开始下载日线行情 ---")
    download_daily(stocks)

    # 3. 下载财务数据
    logger.info("\n--- 开始下载财务数据 ---")
    download_financial(stocks)

    # 4. 下载每日全景
    logger.info("\n--- 开始下载每日全景数据 ---")
    download_daily_basic(stocks)

    logger.info("\n" + "=" * 60)
    logger.info("数据下载全部完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    run()
