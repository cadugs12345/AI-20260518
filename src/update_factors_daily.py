"""
增量因子更新：只补缺失的交易日（比全量重建快100倍）
用法：python src/update_factors_daily.py
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd

# 强制无缓冲
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
os.environ['PYTHONUNBUFFERED'] = '1'

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import config.settings as S
pd.set_option("future.no_silent_downcasting", True)

# 直接写路径常量（避免导入问题）
DATA_FACTORS = S.DATA_FACTORS
DATA_RAW = S.DATA_RAW

t0 = time.time()
print(">>> update_factors_daily 启动", flush=True)

# ===== 加载现有面板 =====
panel_path = os.path.join(DATA_FACTORS, "factor_panel_v5_final.parquet")
print(f"加载现有面板...", end=" ", flush=True)
panel = pd.read_parquet(panel_path)
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
print(f"{len(panel):,}行, {panel['trade_date'].min().date()} ~ {panel['trade_date'].max().date()}", flush=True)

# ===== 确定缺失日期 =====
print("扫描缺失日期...", end=" ", flush=True)
px = pd.read_parquet(os.path.join(DATA_FACTORS, "full_prices.parquet"),
    columns=["trade_date"])
px["trade_date"] = pd.to_datetime(px["trade_date"])
all_dates = sorted(px["trade_date"].unique())
existing_dates = set(panel["trade_date"].unique())
missing = sorted(set(all_dates) - existing_dates)
print(f"{len(missing)} 天缺失: {missing[0].date() if missing else '无'} ~ {missing[-1].date() if missing else '无'}", flush=True)

if not missing:
    print("✅ 无缺失日期，面板已是最新")
    sys.exit(0)

print(f"缺失 {len(missing)} 天: {missing[0].date()} ~ {missing[-1].date()}")
missing_strs = set(d.strftime("%Y%m%d") for d in missing)

# ===== 1. 计算时序因子 ====
print("\n计算时序因子...")
stock_list = pd.read_parquet(os.path.join(DATA_RAW, "stock_list.parquet"))
all_codes = stock_list["ts_code"].tolist()
daily_dir = os.path.join(DATA_RAW, "daily")

def calc_stock_factors(daily):
    """计算时序技术因子（同build_factors.py）"""
    if daily.empty or len(daily) < 60:
        return pd.DataFrame()
    daily = daily.set_index("trade_date")
    close, high, low, vol, pct, amount = daily["close"], daily["high"], daily["low"], daily["vol"], daily["pct_chg"], daily["amount"]
    factors = pd.DataFrame(index=daily.index)
    # 反转与动量
    factors["短期反转"] = -close.pct_change(5)
    factors["20日动量"] = close.pct_change(20)
    factors["60日动量"] = close.pct_change(60)
    factors["120日动量"] = close.pct_change(120)
    factors["波动率"] = pct.rolling(20, min_periods=5).std() * np.sqrt(252) / 100.0
    tr = pd.concat([abs(high - low), abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    factors["ATR比率"] = tr.rolling(14).mean() / close * 100
    factors["Amihud非流动性"] = (pct.abs()/100 / (amount/1e8).replace(0, np.nan)).rolling(20, min_periods=5).mean()
    for p in [6, 12, 24]:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        ag, al = gain.rolling(p).mean(), loss.rolling(p).mean()
        factors[f"RSI_{p}"] = 100 - (100 / (1 + ag / al.replace(0, np.nan)))
    for p in [5, 10, 20]:
        factors[f"EMA{p}偏离"] = (close - close.ewm(span=p).mean()) / close * 100
    ma20 = close.rolling(20).mean()
    factors["BOLL位置"] = ((close - ma20) / (2 * close.rolling(20).std() + 1e-10)).clip(-3, 3)
    ema12, ema26 = close.ewm(span=12).mean(), close.ewm(span=26).mean()
    dif, dea = ema12 - ema26, (ema12 - ema26).ewm(span=9).mean()
    factors["MACD"] = ((dif - dea) * 2 / close * 100).clip(-10, 10)
    factors["量能趋势"] = (vol.ewm(span=20).mean() / close).pct_change(20).clip(-50, 50) * 100
    factors["威廉指标"] = (high.rolling(14).max() - close) / (high.rolling(14).max() - low.rolling(14).min() + 1e-10) * (-100)
    tp = (high + low + close) / 3
    factors["CCI"] = ((tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True) + 1e-10)).clip(-200, 200)
    obv = (vol * np.sign(pct)).cumsum()
    factors["OBV"] = ((obv - obv.expanding().mean()) / (obv.expanding().std() + 1e-10)).clip(-5, 5)
    rsv = (close - low.rolling(9).min()) / (high.rolling(9).max() - low.rolling(9).min() + 1e-10) * 100
    factors["KDJ_K"] = rsv.ewm(span=3).mean()
    return factors

print(f"开始遍历{len(all_codes)}只股票...", flush=True)
ts_rows = []
for i, code in enumerate(all_codes):
    path = os.path.join(daily_dir, f"{code}.parquet")
    if not os.path.exists(path):
        continue
    daily = pd.read_parquet(path)
    if daily.empty or len(daily) < 60:
        continue
    daily = daily.sort_values("trade_date")
    factors = calc_stock_factors(daily)
    if factors.empty:
        continue
    # 只取缺失的日期
    avail_dates = set(factors.index.strftime("%Y%m%d")) & missing_strs
    if not avail_dates:
        continue
    fy = factors[factors.index.isin(avail_dates)].copy()
    if fy.empty:
        continue
    fy["ts_code"] = code
    fy = fy.reset_index().rename(columns={"index": "trade_date"})
    ts_rows.append(fy)
    if (i + 1) % 200 == 0:
        print(f"  [{i+1}/{len(all_codes)}] 已找到{len(ts_rows)}只有新日期", flush=True)

if ts_rows:
    ts_new = pd.concat(ts_rows, ignore_index=True)
    print(f"  新时序因子: {len(ts_new)} 条", flush=True)
else:
    print("  没有新时序因子数据", flush=True)
    ts_new = pd.DataFrame()

# ===== 2. 合并截面数据（daily_basic）=====
print("\n合并截面因子（市值、BP、换手率等）...")
db_dir = os.path.join(DATA_RAW, "daily_basic")
db_files = sorted(os.listdir(db_dir))
db_parts = []
for fname in db_files:
    if not fname.endswith(".parquet"):
        continue
    db = pd.read_parquet(os.path.join(db_dir, fname))
    if not db.empty:
        db["trade_date"] = pd.to_datetime(db["trade_date"])
        db = db[db["trade_date"].isin(missing)]
        if not db.empty:
            db_parts.append(db)

if db_parts:
    df_db = pd.concat(db_parts, ignore_index=True)
    df_db["市值"] = np.log(df_db["total_mv"].replace(0, np.nan))
    df_db["流通市值"] = np.log(df_db["circ_mv"].replace(0, np.nan))
    df_db["BP"] = (1.0 / df_db["pb"].replace(0, np.nan).replace(np.inf, np.nan))
    df_db["EP"] = (1.0 / df_db["pe_ttm"].replace(0, np.nan).replace(np.inf, np.nan))
    df_db["SP"] = (1.0 / df_db["ps_ttm"].replace(0, np.nan).replace(np.inf, np.nan))
    df_db["股息率"] = df_db["dv_ttm"] / 100.0
    df_db["换手率"] = df_db["turnover_rate"]
    df_db["量比"] = df_db["volume_ratio"]
    sc = [c for c in ["ts_code","trade_date","市值","流通市值","BP","EP","SP","股息率","换手率","量比"] if c in df_db.columns]
    section_new = df_db[sc]
    print(f"  新截面数据: {len(section_new)} 条")
else:
    print("  没有新截面数据")
    section_new = pd.DataFrame()

# ===== 3. 合并财务因子 =====
print("\n合并财务因子...")
fin_dir = os.path.join(DATA_RAW, "financial")
fin_files = [f for f in os.listdir(fin_dir) if f.endswith(".parquet")]
fin_parts_new = []
for fname in fin_files:
    fd = pd.read_parquet(os.path.join(fin_dir, fname))
    if fd.empty:
        continue
    fd["ts_code"] = fname.replace(".parquet", "")
    if "ann_date" in fd.columns:
        fd["ann_date"] = pd.to_datetime(fd["ann_date"], errors="coerce")
    if "end_date" in fd.columns:
        fd["end_date"] = pd.to_datetime(fd["end_date"], errors="coerce")
    cols = ["ts_code","ann_date","end_date","roe","grossprofit_margin","netprofit_margin","yoy_gr_yoy","yoy_sales_gr_yoy","debt_to_assets"]
    has = [c for c in cols if c in fd.columns]
    fd = fd[has].copy()
    # 只取新日期有公告的
    if "ann_date" in fd.columns:
        fd = fd[fd["ann_date"].isin(missing)]
    if not fd.empty:
        fin_parts_new.append(fd)

if fin_parts_new:
    df_fin_new = pd.concat(fin_parts_new, ignore_index=True)
    df_fin_valid = df_fin_new[df_fin_new["ann_date"].notna()].copy()
    df_fin_valid["trade_date"] = df_fin_valid["ann_date"]
    print(f"  新财务数据: {len(df_fin_valid)} 条")
else:
    df_fin_valid = pd.DataFrame()

# ===== 4. 构建新行 =====
print("\n构建增量面板...")
# 先用时序因子为基础，左连截面
if not ts_new.empty:
    new_panel = ts_new.copy()
    if not section_new.empty:
        new_panel = new_panel.merge(section_new, on=["ts_code","trade_date"], how="left")
    # 加财务
    if not df_fin_valid.empty:
        fin_map = {"roe":"ROE","grossprofit_margin":"毛利率","netprofit_margin":"净利率","yoy_gr_yoy":"利润增速","yoy_sales_gr_yoy":"营收增速","debt_to_assets":"杠杆"}
        for src, tgt in fin_map.items():
            if src in df_fin_valid.columns:
                sub = df_fin_valid[["ts_code","trade_date",src]].dropna().rename(columns={src:tgt})
                sub = sub.drop_duplicates(["ts_code","trade_date"], keep="last")
                if not sub.empty:
                    new_panel = new_panel.merge(sub, on=["ts_code","trade_date"], how="left")

    # 加close列（全量prices里的close）
    px_missing = prices[prices["trade_date"].isin(missing)][["ts_code","trade_date","close"]]
    if not px_missing.empty:
        new_panel = new_panel.merge(px_missing, on=["ts_code","trade_date"], how="left")

    # 确保列顺序与现有面板一致
    existing_cols = panel.columns.tolist()
    new_cols = new_panel.columns.tolist()
    # 加新列（新增因子时会出现）
    for c in new_cols:
        if c not in existing_cols and c not in ["ts_code","trade_date"]:
            panel[c] = np.nan  # 给现有面板补空列
    existing_cols = panel.columns.tolist()
    # 确保新面板列一致
    for c in existing_cols:
        if c not in new_cols and c not in ["fwd_20d_ret","pet_quantile","rank"]:
            new_panel[c] = np.nan
    # 对齐列顺序
    common_cols = [c for c in existing_cols if c in new_panel.columns]
    new_panel = new_panel[common_cols]

    # ===== 5. 追加并保存 =====
    print(f"\n追加新数据: {len(new_panel)} 行")
    combined = pd.concat([panel, new_panel], ignore_index=True)
    combined = combined.sort_values(["trade_date","ts_code"]).reset_index(drop=True)

    # 保存为最终版
    combined.to_parquet(panel_path, index=False)
    print(f"✅ 更新完成")
    print(f"  原面板: {len(panel):,} 行")
    print(f"  新增: {len(new_panel):,} 行")
    print(f"  新总行数: {len(combined):,}")
    print(f"  新日期范围: {combined['trade_date'].min().date()} ~ {combined['trade_date'].max().date()}")
else:
    print("没有新数据可追加")

print(f"\n总用时: {time.time()-t0:.1f}s")
