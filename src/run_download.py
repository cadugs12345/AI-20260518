"""
启动全量数据下载（后台运行）
- 日线行情
- 财务数据
- 每日全景
"""
import os, sys, time, logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import tushare as ts

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    TS_TOKEN, START_DATE, END_DATE, DATA_RAW, LOGS_DIR,
    TS_SLEEP,
)

os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "full_download.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

ts.set_token(TS_TOKEN)

# ==================== 加载股票名单 ====================
def load_stocks():
    path = os.path.join(DATA_RAW, "stock_list.parquet")
    df = pd.read_parquet(path)
    logger.info(f"加载股票名单: {len(df)} 只")
    return df

# ==================== 日线下载（多线程加速） ====================
def download_one_daily(code, pro_api, save_dir):
    """单只股票日线下载"""
    save_path = os.path.join(save_dir, f"{code}.parquet")
    if os.path.exists(save_path):
        return code, "skip"
    try:
        df = pro_api.daily(
            ts_code=code,
            start_date=START_DATE,
            end_date=END_DATE,
            fields="trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount",
        )
        if df is not None and len(df) > 0:
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.sort_values("trade_date").reset_index(drop=True)
            df.to_parquet(save_path, index=False)
            return code, "ok"
        return code, "empty"
    except Exception as e:
        logger.warning(f"日线下载失败 {code}: {e}")
        return code, f"fail:{e}"

def download_daily_parallel(stocks, max_workers=3):
    """
    多线程下载日线
    Tushare 限频约 3次/秒，max_workers=3 配合 sleep=0.35 刚好
    """
    codes = stocks["ts_code"].tolist()
    save_dir = os.path.join(DATA_RAW, "daily")
    os.makedirs(save_dir, exist_ok=True)

    # 统计已有
    existing = sum(1 for c in codes if os.path.exists(os.path.join(save_dir, f"{c}.parquet")))
    logger.info(f"日线已有: {existing}/{len(codes)}")

    if existing >= len(codes):
        logger.info("日线已全部下载完成，跳过")
        return

    results = {"ok": 0, "skip": existing, "fail": 0, "empty": 0}
    batch_size = 300
    total = len(codes)

    for batch_start in range(0, total, batch_size):
        batch = codes[batch_start:batch_start + batch_size]
        # 过滤已有
        to_download = [c for c in batch if not os.path.exists(os.path.join(save_dir, f"{c}.parquet"))]
        if not to_download:
            results["skip"] += len(batch)
            continue

        pro = ts.pro_api()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_one_daily, code, pro, save_dir): code for code in to_download}
            for future in as_completed(futures):
                code, status = future.result()
                results[status] = results.get(status, 0) + 1
                if results[status] % 100 == 0:
                    logger.info(f"日线进度: ~{results['ok'] + results['skip']}/{total}")

        progress = min(batch_start + batch_size, total)
        logger.info(f"日线批量进度: {progress}/{total} | OK={results['ok']} | SKIP={results['skip']} | FAIL={results['fail']}")
        time.sleep(1)  # 每批次间隔

    logger.info(f"日线完成: OK={results['ok']} | SKIP={results['skip']} | FAIL={results['fail']} | EMPTY={results['empty']}")

# ==================== 财务数据下载 ====================
def download_one_financial(code, pro_api, save_dir):
    save_path = os.path.join(save_dir, f"{code}.parquet")
    if os.path.exists(save_path):
        return code, "skip"
    try:
        df = pro_api.fina_indicator(
            ts_code=code,
            start_date=str(int(START_DATE[:4]) - 1) + "0101",
            end_date=END_DATE,
            fields=(
                "ts_code,ann_date,end_date,eps,roe,roe_dt,roa,grossprofit_margin,"
                "profit_margin,operating_profit_margin,netprofit_margin,"
                "yoy_gr_yoy,yoy_sales_gr_yoy,"
                "debt_to_assets,current_ratio,"
            ),
        )
        if df is not None and len(df) > 0:
            df["end_date"] = pd.to_datetime(df["end_date"])
            df = df.sort_values("end_date").reset_index(drop=True)
            df.to_parquet(save_path, index=False)
            return code, "ok"
        return code, "empty"
    except Exception as e:
        logger.warning(f"财务下载失败 {code}: {e}")
        return code, f"fail:{e}"

def download_financial_parallel(stocks, max_workers=3):
    codes = stocks["ts_code"].tolist()
    save_dir = os.path.join(DATA_RAW, "financial")
    os.makedirs(save_dir, exist_ok=True)

    existing = sum(1 for c in codes if os.path.exists(os.path.join(save_dir, f"{c}.parquet")))
    logger.info(f"财务数据已有: {existing}/{len(codes)}")

    if existing >= len(codes):
        logger.info("财务数据已全部下载完成，跳过")
        return

    results = {"ok": 0, "skip": existing, "fail": 0, "empty": 0}
    batch_size = 300
    total = len(codes)

    for batch_start in range(0, total, batch_size):
        batch = codes[batch_start:batch_start + batch_size]
        to_download = [c for c in batch if not os.path.exists(os.path.join(save_dir, f"{c}.parquet"))]
        if not to_download:
            results["skip"] += len(batch)
            continue

        pro = ts.pro_api()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_one_financial, code, pro, save_dir): code for code in to_download}
            for future in as_completed(futures):
                code, status = future.result()
                results[status] = results.get(status, 0) + 1

        progress = min(batch_start + batch_size, total)
        logger.info(f"财务进度: {progress}/{total} | OK={results['ok']} | SKIP={results['skip']} | FAIL={results['fail']}")

    logger.info(f"财务完成: OK={results['ok']} | SKIP={results['skip']} | FAIL={results['fail']}")

# ==================== 每日全景 ====================
def download_daily_basic(stocks):
    """每交易日一文件，拉全部股票的全景数据"""
    save_dir = os.path.join(DATA_RAW, "daily_basic")
    os.makedirs(save_dir, exist_ok=True)

    # 从已下载的日线数据中获取实际交易日列表
    sample_code = stocks["ts_code"].iloc[0]
    sample_path = os.path.join(DATA_RAW, "daily", f"{sample_code}.parquet")
    sample_df = pd.read_parquet(sample_path)
    trade_dates = sample_df["trade_date"].dt.strftime("%Y%m%d").tolist()
    logger.info(f"交易日数量: {len(trade_dates)}")

    pro = ts.pro_api()
    total = len(trade_dates)
    ok_count = 0
    skip_count = 0

    for i, trade_date in enumerate(trade_dates):
        save_path = os.path.join(save_dir, f"{trade_date}.parquet")
        if os.path.exists(save_path):
            skip_count += 1
            continue

        try:
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
                ok_count += 1
            time.sleep(TS_SLEEP * 2)
        except Exception as e:
            logger.warning(f"daily_basic {trade_date} 失败: {e}")
            time.sleep(2)

        if (i + 1) % 50 == 0:
            logger.info(f"daily_basic: {i+1}/{total} | OK={ok_count} | SKIP={skip_count}")

    logger.info(f"daily_basic 完成: OK={ok_count} | SKIP={skip_count}")

# ==================== 主流程 ====================
def run():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("全量数据下载启动")
    logger.info(f"时间范围: {START_DATE} ~ {END_DATE}")
    logger.info("=" * 60)

    stocks = load_stocks()

    # 1. 日线
    logger.info("\n=== 1/3 日线行情下载 ===")
    download_daily_parallel(stocks, max_workers=4)
    elapsed = time.time() - t0
    logger.info(f"日线完成，已耗时: {elapsed/60:.1f} 分钟")

    # 2. 财务数据
    logger.info("\n=== 2/3 财务数据下载 ===")
    download_financial_parallel(stocks, max_workers=4)
    elapsed = time.time() - t0
    logger.info(f"财务完成，已耗时: {elapsed/60:.1f} 分钟")

    # 3. 每日全景
    logger.info("\n=== 3/3 每日全景下载 ===")
    download_daily_basic(stocks)
    elapsed = time.time() - t0
    logger.info(f"全景完成，总耗时: {elapsed/60:.1f} 分钟")

    elapsed_total = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"全部数据下载完成！总耗时: {elapsed_total/60:.1f} 分钟")
    logger.info(f"数据存储: {DATA_RAW}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run()
