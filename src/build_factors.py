"""
因子面板构建 (分批内存友好版)
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DATA_RAW, DATA_FACTORS

def calc_stock_factors(daily):
    if daily.empty or len(daily) < 60:
        return pd.DataFrame()
    daily = daily.set_index("trade_date") if "trade_date" in daily.columns else daily
    close, high, low, vol, pct, amount = daily["close"], daily["high"], daily["low"], daily["vol"], daily["pct_chg"], daily["amount"]
    factors = pd.DataFrame(index=daily.index)
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

def run():
    t0 = time.time()
    print("=" * 50, flush=True)
    print("因子面板构建 - 分批版", flush=True)
    print("=" * 50, flush=True)
    stock_list = pd.read_parquet(os.path.join(DATA_RAW, "stock_list.parquet"))
    all_codes = stock_list["ts_code"].tolist()
    print(f"股票池: {len(all_codes)} 只", flush=True)
    daily_dir = os.path.join(DATA_RAW, "daily")
    sample = pd.read_parquet(os.path.join(daily_dir, f"{all_codes[0]}.parquet"))
    all_dates = sorted(sample["trade_date"].dt.strftime("%Y%m%d").tolist())
    print(f"交易日: {len(all_dates)} 天", flush=True)
    os.makedirs(DATA_FACTORS, exist_ok=True)
    years = sorted(set(d[:4] for d in all_dates))
    for year in years:
        ty = time.time()
        print(f"\n[年份 {year}] 开始...", flush=True)
        year_start = pd.Timestamp(f"{year}-01-01")
        year_end = pd.Timestamp(f"{int(year)+1}-01-01") - pd.Timedelta(days=1)
        batch_size = 500
        total_batches = (len(all_codes) + batch_size - 1) // batch_size
        for batch_idx in range(total_batches):
            batch_codes = all_codes[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            ts_list = []
            for code in batch_codes:
                path = os.path.join(daily_dir, f"{code}.parquet")
                if not os.path.exists(path):
                    continue
                daily = pd.read_parquet(path)
                if daily.empty or len(daily) < 60:
                    continue
                daily["trade_date"] = pd.to_datetime(daily["trade_date"])
                daily = daily.sort_values("trade_date").set_index("trade_date")
                factors = calc_stock_factors(daily)
                if factors.empty:
                    continue
                year_mask = (factors.index >= year_start) & (factors.index <= year_end)
                fy = factors[year_mask].copy()
                if fy.empty:
                    continue
                fy["ts_code"] = code
                fy = fy.reset_index().rename(columns={"index": "trade_date"})
                ts_list.append(fy)
            if ts_list:
                df_batch = pd.concat(ts_list, ignore_index=True)
                bpath = os.path.join(DATA_FACTORS, f"ts_{year}_b{batch_idx}.parquet")
                df_batch.to_parquet(bpath, index=False)
            print(f"  [{year}] 批次 {batch_idx+1}/{total_batches}", flush=True)
        # 合并该年
        bfiles = sorted([os.path.join(DATA_FACTORS, f) for f in os.listdir(DATA_FACTORS) if f.startswith(f"ts_{year}_b")])
        if bfiles:
            df_year = pd.concat([pd.read_parquet(f) for f in bfiles], ignore_index=True)
            sp = os.path.join(DATA_FACTORS, f"timeseries_{year}.parquet")
            df_year.to_parquet(sp, index=False)
            print(f"[年份 {year}] 保存: {len(df_year)} 条, 用时 {time.time()-ty:.0f}s", flush=True)
            for f in bfiles:
                os.remove(f)
    # 合并截面和财务
    print(f"\n[合并] 加载所有年份时序因子...", flush=True)
    all_ts = []
    for year in years:
        p = os.path.join(DATA_FACTORS, f"timeseries_{year}.parquet")
        if os.path.exists(p):
            all_ts.append(pd.read_parquet(p))
    if all_ts:
        df_all = pd.concat(all_ts, ignore_index=True)
        df_all["trade_date"] = pd.to_datetime(df_all["trade_date"])
        print(f"[合并] 时序因子: {len(df_all)} 条", flush=True)
        db_dir = os.path.join(DATA_RAW, "daily_basic")
        db_files = sorted(os.listdir(db_dir))
        db_parts = []
        for j, fname in enumerate(db_files):
            if not fname.endswith(".parquet"):
                continue
            db = pd.read_parquet(os.path.join(db_dir, fname))
            if not db.empty:
                db["trade_date"] = pd.to_datetime(db["trade_date"])
                db_parts.append(db)
        df_db = pd.concat(db_parts, ignore_index=True) if db_parts else pd.DataFrame()
        print(f"[合并] 全景数据: {len(df_db)} 条", flush=True)
        if not df_db.empty:
            df_db["市值"] = np.log(df_db["total_mv"].replace(0, np.nan))
            df_db["流通市值"] = np.log(df_db["circ_mv"].replace(0, np.nan))
            df_db["BP"] = (1.0 / df_db["pb"].replace(0, np.nan).replace(np.inf, np.nan))
            df_db["EP"] = (1.0 / df_db["pe_ttm"].replace(0, np.nan).replace(np.inf, np.nan))
            df_db["SP"] = (1.0 / df_db["ps_ttm"].replace(0, np.nan).replace(np.inf, np.nan))
            df_db["股息率"] = df_db["dv_ttm"] / 100.0
            df_db["换手率"] = df_db["turnover_rate"]
            df_db["量比"] = df_db["volume_ratio"]
            sc = [c for c in ["ts_code","trade_date","市值","流通市值","BP","EP","SP","股息率","换手率","量比"] if c in df_db.columns]
            df_section = df_db[sc]
            df_panel = df_all.merge(df_section, on=["ts_code","trade_date"], how="left")
        else:
            df_panel = df_all
        print("[合并] 添加财务因子...", flush=True)
        fin_dir = os.path.join(DATA_RAW, "financial")
        fin_files = [f for f in os.listdir(fin_dir) if f.endswith(".parquet")]
        fin_parts = []
        for j, fname in enumerate(fin_files):
            fd = pd.read_parquet(os.path.join(fin_dir, fname))
            if not fd.empty:
                fd["ts_code"] = fname.replace(".parquet", "")
                if "ann_date" in fd.columns:
                    fd["ann_date"] = pd.to_datetime(fd["ann_date"], errors="coerce")
                if "end_date" in fd.columns:
                    fd["end_date"] = pd.to_datetime(fd["end_date"], errors="coerce")
                cols = ["ts_code","ann_date","end_date","roe","grossprofit_margin","netprofit_margin","yoy_gr_yoy","yoy_sales_gr_yoy","debt_to_assets"]
                fin_parts.append(fd[[c for c in cols if c in fd.columns]])
        if fin_parts:
            df_fin_all = pd.concat(fin_parts, ignore_index=True)
            df_fin_valid = df_fin_all[df_fin_all["ann_date"].notna()].copy()
            df_fin_valid["trade_date"] = df_fin_valid["ann_date"]
            fin_map = {"roe":"ROE","grossprofit_margin":"毛利率","netprofit_margin":"净利率","yoy_gr_yoy":"利润增速","yoy_sales_gr_yoy":"营收增速","debt_to_assets":"杠杆"}
            for src, tgt in fin_map.items():
                if src in df_fin_valid.columns:
                    sub = df_fin_valid[["ts_code","trade_date",src]].dropna().rename(columns={src:tgt})
                    sub = sub.drop_duplicates(["ts_code","trade_date"], keep="last")
                    df_panel = df_panel.merge(sub, on=["ts_code","trade_date"], how="left")
        result_path = os.path.join(DATA_FACTORS, "factor_panel.parquet")
        df_panel.to_parquet(result_path, index=False)
        print(f"\n{'='*50}", flush=True)
        print(f"完成! 因子面板: {len(df_panel)} 条, {len(df_panel.columns)} 列", flush=True)
        print(f"保存: {result_path}", flush=True)
        print(f"总用时: {(time.time()-t0)/60:.1f} 分钟", flush=True)
        print(f"{'='*50}", flush=True)
    else:
        print("[错误] 无数据生成", flush=True)

if __name__ == "__main__":
    run()
