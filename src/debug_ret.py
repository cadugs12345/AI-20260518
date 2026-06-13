"""
极度简化的回测调试 - 只做第一周
"""
import os, sys, numpy as np, pandas as pd

warnings = __import__('warnings')
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DATA_FACTORS

panel = pd.read_parquet(os.path.join(DATA_FACTORS, "factor_panel_with_fwd.parquet"))
prices = pd.read_parquet(os.path.join(DATA_FACTORS, "full_prices.parquet"))
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
prices["trade_date"] = pd.to_datetime(prices["trade_date"])
panel = panel[panel["trade_date"] >= "2018-01-01"].copy()
panel = panel.merge(prices, on=["ts_code","trade_date"], how="left")

dates = sorted(panel["trade_date"].unique())
weekly = dates[::5]
weekly = [d for d in weekly if d >= pd.Timestamp("2021-01-01")]

px_index = {}
for d in weekly:
    sub = panel[panel["trade_date"] == d][["ts_code","close"]].set_index("ts_code")["close"].to_dict()
    px_index[d] = sub

print(f"每周节点数: {len(weekly)}")
print(f"第一期: {weekly[0].date()}, 第二期: {weekly[1].date()}, 间隔: {(weekly[1]-weekly[0]).days}天")

# 第一周: 选Top 100动量
d0 = weekly[0]
d1 = weekly[1]

# 看d0有多少股票有价格
sub0 = panel[panel["trade_date"] == d0]
print(f"\n{d0.date()} 有 {len(sub0):,} 只股票, close缺失: {sub0['close'].isna().sum():,}")

# 构建ret_20d 
momentum = panel[["ts_code","trade_date","close"]].copy().sort_values(["ts_code","trade_date"])
# 先看一个股票验证
msft_data = momentum[momentum["ts_code"] == "000001.SZ"]
print(f"\n000001.SZ 在{d0.date().isoformat()}前后的数据:")
print(msft_data[msft_data["trade_date"].between(d0-pd.Timedelta(days=30), d1)].tail(10))

# 看当天有多少价格=0的
p0 = sub0["close"].values
print(f"close=0或NaN: {np.sum(p0==0) + np.sum(np.isnan(p0))} / {len(p0)}")

# 简洁验证: 随机选50只, 计算下期收益分布
import random
random.seed(42)
codes_d0 = list(sub0.dropna(subset=["close"])["ts_code"].values)
codes_sample = random.sample(codes_d0, min(100, len(codes_d0)))

px0 = px_index[d0]
px1 = px_index[d1]

rets = []
for code in codes_sample:
    p0 = px0.get(code, 0)
    p1 = px1.get(code, 0)
    if p0 > 0 and p1 > 0:
        rets.append(p1/p0 - 1)

rets = np.array(rets)
print(f"\n{len(rets)}只随机股票 d0→d1 收益率:")
print(f"  平均: {np.mean(rets)*100:.2f}%")
print(f"  中位: {np.median(rets)*100:.2f}%")
print(f"  >0: {np.mean(rets>0)*100:.0f}%")
print(f"  >10%: {np.mean(rets>0.10)*100:.1f}%")
print(f"  max: {np.max(rets)*100:.1f}%, min: {np.min(rets)*100:.1f}%")
