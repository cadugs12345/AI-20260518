"""
因子中性化 (快速版) - 提前合并全景数据, 一次搞定
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DATA_FACTORS, DATA_RAW

print("=" * 50, flush=True)
print("因子中性化 (快速版)", flush=True)
print("=" * 50, flush=True)

t0 = time.time()

# 1. 加载面板
panel = pd.read_parquet(os.path.join(DATA_FACTORS, "factor_panel_with_fwd.parquet"))
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
panel = panel[panel["trade_date"] >= "2018-01-01"].copy()
print(f"面板: {len(panel):,} 条", flush=True)

factor_cols = [c for c in panel.columns 
               if c not in ("ts_code","trade_date","fwd_20d_ret")
               and panel[c].dtype in ("float64","int64")]

# 2. 行业
indf = pd.read_parquet(os.path.join(DATA_RAW, "stock_list.parquet"))
industry_map = dict(zip(indf["ts_code"], indf["industry"].fillna("未知")))

# 3. 市值 - 从 daily_basic 预读, 只需要 ts_code, trade_date, total_mv
print("加载全景市值数据...", flush=True)
db_dir = os.path.join(DATA_RAW, "daily_basic")
db_files = sorted(os.listdir(db_dir))
db_list = []
for f in db_files:
    d = pd.read_parquet(os.path.join(db_dir, f))
    d["trade_date"] = pd.to_datetime(f.replace(".parquet", ""))
    db_list.append(d[["ts_code","trade_date","total_mv"]])
df_db = pd.concat(db_list, ignore_index=True)
del db_list
df_db["ln_mv"] = np.log(df_db["total_mv"].replace(0, np.nan))
print(f"  全景: {len(df_db):,} 条", flush=True)

# 合并市值到面板
panel = panel.merge(df_db[["ts_code","trade_date","ln_mv"]], on=["ts_code","trade_date"], how="left")
del df_db
print(f"  合并后: {len(panel):,} 条", flush=True)

# 4. 逐日中性化
dates = sorted(panel["trade_date"].unique())
all_industries = sorted(set(industry_map.values()))
n_ind = len(all_industries)
ind_to_idx = {ind: i for i, ind in enumerate(all_industries)}
n_factors = len(factor_cols)

print(f"交易日: {len(dates)} 天, 行业: {n_ind} 个, 因子: {n_factors} 个", flush=True)

neutral_list = []
dates_sample = dates[:5]  # 先跑5天测试
print(f"测试前5天...", flush=True)
for i, date in enumerate(dates_sample):
    df_day = panel[panel["trade_date"] == date].copy()
    n = len(df_day)
    codes = df_day["ts_code"].tolist()
    
    # 市值 (已经标准化)
    cap = df_day["ln_mv"].values
    cap = (cap - np.nanmean(cap)) / max(np.nanstd(cap), 1e-10)
    
    # 行业哑变量
    ind_mat = np.zeros((n, n_ind), dtype=np.float64)
    for j, c in enumerate(codes):
        idx = ind_to_idx.get(industry_map.get(c, "未知"))
        if idx is not None:
            ind_mat[j, idx] = 1.0
    
    # 回归矩阵
    X = np.column_stack([np.ones(n), cap, ind_mat])
    x_ok = ~np.isnan(X).any(axis=1)
    
    n_processed = 0
    for factor in factor_cols:
        y = df_day[factor].values.astype(float)
        y_ok = ~np.isnan(y)
        mask = y_ok & x_ok
        
        if mask.sum() < 100:
            continue
        
        Xc = X[mask]
        yc = y[mask]
        try:
            coeffs, _, _, _ = np.linalg.lstsq(Xc, yc, rcond=None)
            res = y - X @ coeffs
            res[~mask] = np.nan
            df_day[factor] = res
            n_processed += 1
        except:
            continue
    
    neutral_list.append(df_day)
    print(f"  第{i+1}天: {len(df_day)}只, {n_processed}/{n_factors}个因子已处理, 用时{time.time()-t0:.0f}s", flush=True)

print(f"5天测试完成, 总计用时{(time.time()-t0):.0f}s", flush=True)

df_neu = pd.concat(neutral_list, ignore_index=True)
print(f"[完成] {len(df_neu):,} 条", flush=True)

# 只保留原列(去掉ln_mv)
keep_cols = [c for c in panel.columns if c != "ln_mv"]
df_neu = df_neu[keep_cols]

out_path = os.path.join(DATA_FACTORS, "factor_panel_neutral.parquet")
df_neu.to_parquet(out_path, index=False)
print(f"[保存] {out_path}", flush=True)

# 相关性
corr = df_neu[factor_cols].corr().abs()
avg_corr = corr.values[np.triu_indices_from(corr.values, k=1)].mean()
print(f"[相关性] 平均绝对相关性: {avg_corr:.4f}")
print(f"[用时] {(time.time()-t0)/60:.1f}min")
print("Done!", flush=True)
