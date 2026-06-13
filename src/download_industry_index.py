#!/usr/bin/env python3
"""
下载申万行业指数K线数据
申万一级行业指数代码: 801XXX.SI (XXX=010~210)
"""
import tushare as ts
import pandas as pd
import os, time

ts.set_token('3e8953587c4c717c26e5cb99d028a66e044d184f2d464cab0950000e')
pro = ts.pro_api()

OUT = "/mnt/d/AI-20260604/data/index_industry"
os.makedirs(OUT, exist_ok=True)

# 申万一级行业代码（28/31个）
codes = [
    '801010.SI', # 农林牧渔
    '801020.SI', # 采掘
    '801030.SI', # 化工
    '801040.SI', # 钢铁
    '801050.SI', # 有色金属
    '801080.SI', # 电子
    '801110.SI', # 家用电器
    '801120.SI', # 食品饮料
    '801130.SI', # 纺织服装
    '801140.SI', # 轻工制造
    '801150.SI', # 医药生物
    '801160.SI', # 公用事业
    '801170.SI', # 交通运输
    '801180.SI', # 房地产
    '801200.SI', # 商业贸易
    '801210.SI', # 休闲服务
    '801230.SI', # 综合
    '801710.SI', # 建筑材料
    '801720.SI', # 建筑装饰
    '801730.SI', # 电气设备
    '801740.SI', # 国防军工
    '801750.SI', # 计算机
    '801760.SI', # 传媒
    '801770.SI', # 通信
    '801780.SI', # 银行
    '801790.SI', # 非银金融
    '801880.SI', # 汽车
    '801890.SI', # 机械设备
    '801200.SI', # 商业贸易
    '801210.SI', # 休闲服务
    '801230.SI', # 综合
]

codes = sorted(set(codes))
print(f"下载 {len(codes)} 个申万行业指数...")

for idx, code in enumerate(codes):
    try:
        df = pro.index_daily(ts_code=code, start_date='20150101', end_date='20260603')
        if df is not None and len(df) > 0:
            df.to_parquet(f"{OUT}/{code}.parquet")
            print(f"  [{idx+1}/{len(codes)}] {code}: {len(df)}条 ({df.trade_date.min()}~{df.trade_date.max()})")
        else:
            print(f"  [{idx+1}/{len(codes)}] {code}: 无数据")
    except Exception as e:
        print(f"  [{idx+1}/{len(codes)}] {code}: {e}")
    time.sleep(0.2)

print("完成!")
