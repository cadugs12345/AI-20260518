"""
机构行为代理因子 (机构调研替代方案)
用现有tushare数据构建:

1. 股东户数因子:
   - holder_concentration: 股东户数变化率（负=筹码集中）
   - holder_momentum: 连续户数减少强度

2. 十大股东机构因子:
   - inst_holding_change: 机构类股东持仓变化
   - top10_quality: 十大股东中机构占比变化

3. 综合:
   - inst_behavior_score: 机构行为综合得分
"""
import sys, os, json, time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import TS_TOKEN

import tushare as ts
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT)

NOW = datetime.now()
print(f"\n🏛️ 机构行为代理因子构建 — {NOW.strftime('%F %H:%M')}")
print("=" * 60)

OUTPUT_DIR = "data/new_factors"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 加载股票列表
print("[1/4] 准备数据...")
stock_list = pd.read_parquet("data/raw/stock_list.parquet")
codes = stock_list["ts_code"].tolist()
print(f"  股票数: {len(codes)}")

# 2. 批量下载股东户数
print("\n[2/4] 下载股东户数...")
all_holder = []
batch_size = 200
t0 = time.time()

for i in range(0, len(codes), batch_size):
    batch = codes[i:i+batch_size]
    for code in batch:
        try:
            df = pro.query('stk_holdernumber', ts_code=code, start_date='20170101', end_date=NOW.strftime('%Y%m%d'))
            if df is not None and len(df) > 0:
                all_holder.append(df)
        except:
            pass
        time.sleep(0.15)  # 限速
    if (i // batch_size) % 5 == 0:
        print(f"  进度: {i+batch_size}/{len(codes)}, 用时{time.time()-t0:.0f}s", flush=True)

if all_holder:
    holder_df = pd.concat(all_holder, ignore_index=True)
    holder_df.to_parquet(f"{OUTPUT_DIR}/holder_numbers.parquet")
    print(f"  股东户数: {len(holder_df):,}行, 用时{time.time()-t0:.0f}s")

# 3. 下载十大股东（只看最近4个季度）
print("\n[3/4] 下载十大股东...")
all_top10 = []
t0 = time.time()

for i in range(0, len(codes), 50):  # 一次多只
    batch = codes[i:i+50]
    codes_str = ','.join(batch)
    try:
        df = pro.query('top10_holders', ts_code=codes_str, start_date='20240101', end_date=NOW.strftime('%Y%m%d'))
        if df is not None and len(df) > 0:
            all_top10.append(df)
    except:
        pass
    time.sleep(0.3)
    if (i // 200) % 5 == 0 and (i // 50) % 4 == 0:
        print(f"  进度: {i+50}/{len(codes)}, 用时{time.time()-t0:.0f}s", flush=True)

if all_top10:
    top10_df = pd.concat(all_top10, ignore_index=True)
    top10_df.to_parquet(f"{OUTPUT_DIR}/top10_holders.parquet")
    print(f"  十大股东: {len(top10_df):,}行, 用时{time.time()-t0:.0f}s")

# 4. 构建因子
print("\n[4/4] 构建因子...")

# 股东户数因子
if all_holder:
    holder_df = pd.read_parquet(f"{OUTPUT_DIR}/holder_numbers.parquet")
    holder_df = holder_df.sort_values(["ts_code", "end_date"])
    holder_df["end_date"] = pd.to_datetime(holder_df["end_date"])
    
    # 户数变化率 (环比)
    holder_df["holder_change_pct"] = holder_df.groupby("ts_code")["holder_num"].pct_change()
    
    # 户数季度变化率
    holder_df["holder_change_qoq"] = holder_df.groupby("ts_code")["holder_num"].pct_change(3)
    
    # 保留最新记录（映射到交易日）
    print(f"  股东户数: {len(holder_df)}条记录, {holder_df['ts_code'].nunique()}只股票")
    print(f"  最新户数变化: 均值={holder_df['holder_change_pct'].mean():.4f}, std={holder_df['holder_change_pct'].std():.4f}")
    
    # 保存原始因子
    holder_factor = holder_df[["ts_code", "end_date", "holder_num", "holder_change_pct", "holder_change_qoq"]].copy()
    holder_factor.to_parquet(f"{OUTPUT_DIR}/holder_factors.parquet")

# 十大股东机构因子
if all_top10:
    top10_df = pd.read_parquet(f"{OUTPUT_DIR}/top10_holders.parquet")
    top10_df = top10_df.sort_values(["ts_code", "end_date"])
    top10_df["end_date"] = pd.to_datetime(top10_df["end_date"])
    
    # 标记是否是机构
    inst_types = ["金融机构", "保险公司", "基金", "证券", "投资"]
    top10_df["is_institution"] = top10_df["holder_type"].apply(
        lambda x: any(it in str(x) for it in inst_types)
    )
    
    # 每只股票每期的机构持股变化
    inst_change = top10_df.groupby(["ts_code", "end_date"]).agg(
        inst_count=("is_institution", "sum"),
        total_hold_change=("hold_change", "sum"),
        inst_hold_change=("hold_change", lambda x: x[top10_df.loc[x.index, "is_institution"]].sum()),
        inst_hold_ratio=("hold_ratio", lambda x: x[top10_df.loc[x.index, "is_institution"]].sum() / x.sum() if x.sum() > 0 else 0),
    ).reset_index()
    
    print(f"  十大股东机构因子: {len(inst_change)}条, {inst_change['ts_code'].nunique()}只股票")
    inst_change.to_parquet(f"{OUTPUT_DIR}/inst_top10_factors.parquet")

print(f"\n{'='*60}")
print(f"✅ 机构行为代理因子数据下载完成")
print(f"  → 需要继续: 因子IC测试 + 合并到面板")
print(f"{'='*60}")
