"""
配置文件 - A股量化因子系统
"""
import os

# ====== 项目路径 ======
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJ_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJ_ROOT, "data", "processed")
DATA_FACTORS = os.path.join(PROJ_ROOT, "data", "factors")
LOGS_DIR = os.path.join(PROJ_ROOT, "logs")

# ====== Tushare Token ======
TS_TOKEN = "3e8953587c4c717c26e5cb99d028a66e044d184f2d464cab0950000e"

# ====== 回测时间范围 ======
START_DATE = "20170101"       # 数据拉取起始日（回测前多一年供因子计算前推）
END_DATE = "20260611"         # 最新完整交易日
BACKTEST_START = "20180101"   # 正式回测起始日

# ====== 股票池参数 ======
# 剔除: ST、*ST、退市整理、北交所、上市不足N天
MIN_LIST_DAYS = 60           # 上市满60天才纳入
EXCLUDE_BOARD = ["北交所"]   # 北交所排除，主板/创业板/科创板保留

# ====== 因子列表 ======
# 金工标准截面因子 (12个)
FACTOR_SECTIONAL = [
    "市值",          # 总市值
    "BP",            # 市净率倒数
    "EP",            # 市盈率倒数
    "SP",            # 市销率倒数
    "股息率",        # 
    "ROE",           # 净资产收益率
    "毛利率",
    "净利率",
    "利润增速",      # 归母净利润同比增长
    "营收增速",      # 营业收入同比增长
    "20日动量",      # 过去20日收益率
    "60日动量",
    "120日动量",
    "短期反转",      # 过去5日收益率
    "换手率",
    "流动性",        # Amihud 非流动性指标
    "量比",
    "杠杆",          # 资产负债率
    "波动率",        # 20日年化波动率
]

# 时序技术因子 (10个)
FACTOR_TIMESERIES = [
    "RSI_6",
    "RSI_12",
    "RSI_24",
    "BOLL",           # 布林带位置
    "MACD",
    "EMA5",
    "EMA10",
    "EMA20",
    "量能趋势",
]

ALL_FACTORS = FACTOR_SECTIONAL + FACTOR_TIMESERIES

# ====== Tushare 请求控制 ======
TS_SLEEP = 0.3        # 每次请求间隔(秒)，避免限频
TS_BATCH_SIZE = 500   # 每次拉取股票数量
