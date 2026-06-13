"""
补充下载缺失的财务数据
"""
import os, sys, time, logging
import pandas as pd
import tushare as ts

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import TS_TOKEN, START_DATE, END_DATE, DATA_RAW, LOGS_DIR, TS_SLEEP

ts.set_token(TS_TOKEN)
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

stock = pd.read_parquet(os.path.join(DATA_RAW, "stock_list.parquet"))
all_codes = set(stock["ts_code"].tolist())
fin_dir = os.path.join(DATA_RAW, "financial")
os.makedirs(fin_dir, exist_ok=True)

existing = set(f.replace(".parquet", "") for f in os.listdir(fin_dir) if f.endswith(".parquet"))
missing = all_codes - existing
logger.info(f"已有: {len(existing)}, 缺失: {len(missing)}")

if not missing:
    logger.info("财务数据已完整")
    exit(0)

pro = ts.pro_api()
fields = (
    "ts_code,ann_date,end_date,eps,roe,roe_dt,roa,grossprofit_margin,"
    "profit_margin,operating_profit_margin,netprofit_margin,"
    "yoy_gr_yoy,yoy_sales_gr_yoy,"
    "debt_to_assets,current_ratio,"
)

ok = 0
fail = 0
for code in sorted(missing):
    save_path = os.path.join(fin_dir, f"{code}.parquet")
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
            ok += 1
        else:
            # 空数据也存一个标记文件, 避免重复尝试
            pd.DataFrame().to_parquet(save_path)
            ok += 1
        time.sleep(TS_SLEEP)
    except Exception as e:
        fail += 1
        logger.warning(f"失败 {code}: {e}")
        time.sleep(2)

    if (ok + fail) % 200 == 0:
        logger.info(f"进度: {ok+fail}/{len(missing)} | OK={ok} | FAIL={fail}")

logger.info(f"完成: OK={ok} | FAIL={fail}")
