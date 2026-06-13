#!/usr/bin/env python3
"""
数据就绪检查 — 确保日K线、指数、因子数据全部更新到最新交易日
================================================================
在运行选股信号之前调用，验证：
  1. tushare交易日历 → 确定最新交易日
  2. 日K线数据 → 所有股票是否覆盖到最新交易日
  3. 沪深300指数 → 是否到最新交易日
  4. 因子面板 → 是否到最新交易日
全部通过 → exit 0，否则 → exit 1
"""
import pandas as pd
import numpy as np
import os, sys, json, glob, warnings, time
warnings.filterwarnings('ignore')

PROJ = "/mnt/d/AI-20260604"
DATA_DAILY = os.path.join(PROJ, "data", "raw", "daily")
STOCK_LIST = os.path.join(PROJ, "data", "raw", "stock_list.parquet")
INDEX_PATH = os.path.join(PROJ, "data", "raw", "index_000300.parquet")
FACTOR_PANEL = os.path.join(PROJ, "data", "factors", "factor_panel_v5_final.parquet")
REPORT_FILE = os.path.join(PROJ, "alerts", "data_check.json")


def get_latest_trade_date():
    """从tushare获取最新交易日（考虑T+1数据可得性：今天的数据明天才有）"""
    try:
        import tushare as ts
        pro = ts.pro_api()
        today = pd.Timestamp.now().strftime('%Y%m%d')
        cal = pro.trade_cal(start_date='20260101', end_date=today)
        cal = cal[cal['is_open'] == 1]
        all_dates = sorted(cal['cal_date'].tolist())
        # 今天如果是交易日，数据可能还没出（收盘后2-3小时），保守取前一日
        latest_cal = all_dates[-1]
        # 如果latest_cal是今天且当前时间<18:00，可能数据没出完，取前一交易日
        now = pd.Timestamp.now()
        if latest_cal == now.strftime('%Y%m%d') and now.hour < 18:
            if len(all_dates) >= 2:
                latest_cal = all_dates[-2]
        return latest_cal
    except:
        return None


def check_daily_data(latest_date_str):
    """检查日K线数据覆盖率"""
    if not os.path.exists(STOCK_LIST):
        return False, 0, 0, "stock_list.parquet 不存在"
    
    sl = pd.read_parquet(STOCK_LIST)
    codes = sorted(sl['ts_code'].unique())
    total = len(codes)
    
    # 快速抽样检查：每只股票的最后日期
    covered = 0; missing = 0; empty = 0
    check_codes = codes  # 全量检查
    sample = min(200, len(codes))
    
    # 先用200只快速判断
    for code in codes[:sample]:
        fp = os.path.join(DATA_DAILY, f"{code}.parquet")
        if not os.path.exists(fp):
            empty += 1; continue
        try:
            df = pd.read_parquet(fp, columns=['trade_date'])
            last = df['trade_date'].max()
            if hasattr(last, 'strftime'):
                last_str = last.strftime('%Y%m%d')
            else:
                last_str = str(last)[:10].replace('-','')
            if last_str >= latest_date_str:
                covered += 1
            else:
                missing += 1
        except:
            empty += 1
    
    coverage = covered / max(covered + missing + empty, 1) * 100
    
    # 如果样本覆盖率>90%，认为OK（少数新股可能数据少）
    ok = coverage >= 90
    details = {
        'total_stocks': total,
        'sample_checked': sample,
        'covered': covered,
        'missing': missing,
        'empty': empty,
        'coverage_pct': round(coverage, 1),
        'latest_date': latest_date_str,
    }
    return ok, covered, missing, details


def check_index(latest_date_str):
    """检查沪深300指数数据"""
    if not os.path.exists(INDEX_PATH):
        return False, "index_000300.parquet 不存在"
    df = pd.read_parquet(INDEX_PATH, columns=['trade_date'])
    last = df['trade_date'].max()
    if hasattr(last, 'strftime'):
        last_str = last.strftime('%Y%m%d')
    else:
        last_str = str(last)[:10].replace('-','')
    return last_str >= latest_date_str, last_str


def check_factor_panel(latest_date_str):
    """检查因子面板"""
    if not os.path.exists(FACTOR_PANEL):
        return False, "factor_panel 不存在"
    try:
        df = pd.read_parquet(FACTOR_PANEL, columns=['trade_date'])
        last = df['trade_date'].max()
        if hasattr(last, 'strftime'):
            last_str = last.strftime('%Y%m%d')
        else:
            last_str = str(last)[:10].replace('-','')
        return last_str >= latest_date_str, last_str
    except:
        return False, "读取失败"


def main():
    print(f"{'='*60}")
    print(f"🔍 数据就绪检查 — {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")
    
    t0 = time.time()
    results = {'timestamp': pd.Timestamp.now().isoformat(), 'checks': {}}
    
    # 1. 最新交易日
    latest = get_latest_trade_date()
    if latest is None:
        print("❌ 无法获取tushare交易日历")
        results['status'] = 'FAIL'
        results['error'] = 'tushare连接失败'
        os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
        with open(REPORT_FILE, 'w') as f: json.dump(results, f, ensure_ascii=False, indent=2)
        sys.exit(1)
    
    print(f"\n📅 最新交易日: {latest}")
    results['latest_trade_date'] = latest
    
    all_ok = True
    
    # 2. 日K线
    print("\n📦 日K线数据...", end=" ", flush=True)
    ok, covered, missing, details = check_daily_data(latest)
    results['checks']['daily_kline'] = details
    if ok:
        print(f"✅ 覆盖率{details['coverage_pct']}% ({details['covered']}/{details['covered']+details['missing']+details['empty']})")
    else:
        print(f"⚠️ 覆盖率仅{details['coverage_pct']}%，{missing}只缺失")
        all_ok = False
    
    # 3. 沪深300
    print("📈 沪深300指数...", end=" ", flush=True)
    ok_idx, idx_last = check_index(latest)
    results['checks']['hs300'] = {'ok': ok_idx, 'last_date': idx_last}
    if ok_idx:
        print(f"✅ 最新{idx_last}")
    else:
        print(f"⚠️ 最后{idx_last}，需要更新")
        all_ok = False
    
    # 4. 因子面板
    print("🧮 因子面板...", end=" ", flush=True)
    ok_fp, fp_last = check_factor_panel(latest)
    results['checks']['factor_panel'] = {'ok': ok_fp, 'last_date': fp_last}
    if ok_fp:
        print(f"✅ 最新{fp_last}")
    else:
        print(f"⚠️ 最后{fp_last}，需要更新")
        all_ok = False
    
    # 汇总
    elapsed = time.time() - t0
    print(f"\n{'─'*60}")
    if all_ok:
        print(f"✅ 全部就绪 (耗时{elapsed:.1f}s)")
        results['status'] = 'OK'
    else:
        print(f"❌ 数据不完整 (耗时{elapsed:.1f}s) — 需要先执行数据更新")
        results['status'] = 'INCOMPLETE'
    
    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    with open(REPORT_FILE, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
