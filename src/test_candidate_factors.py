"""
新因子生成器 - 从已有数据中挖掘衍生因子
"""
import os, sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factor_engine import FactorEngine

def generate_candidate_factors(panel):
    """从面板已有数据生成候选衍生因子"""
    print("生成候选因子...")
    candidates = []
    
    # 1. 因子交互项 (已有因子间的加减乘除)
    base_factors = [c for c in panel.columns 
                    if c not in ("ts_code","trade_date","fwd_20d_ret","fwd_5d_ret",
                                  "pe","pe_ttm","pb","ps","ps_ttm","dv_ratio","dv_ttm",
                                  "total_mv","circ_mv","市值","流通市值")
                    and panel[c].dtype in ("float64","int64")]
    print(f"  基础因子: {base_factors}")
    
    # 取核心因子做组合
    core = ["20日动量","60日动量","RSI_6","RSI_24","BOLL位置","MACD","量能趋势","换手率"]
    core = [c for c in core if c in panel.columns]
    print(f"  核心因子: {core}")
    
    # 动量之间的差异（短期-长期）
    if "20日动量" in panel.columns and "60日动量" in panel.columns:
        candidates.append({
            "name": "动量加速",
            "data": panel[["trade_date","ts_code"]]  # 占位
        })
    
    # 动量反转复合
    if "20日动量" in panel.columns and "RSI_6" in panel.columns:
        pass
    
    return candidates


def quick_test_candidates():
    """快速测试几个候选衍生因子"""
    import warnings
    warnings.filterwarnings("ignore")
    
    eng = FactorEngine(
        data_path="data/factors/factor_panel_v3.parquet",
        price_path="data/factors/full_prices.parquet"
    )
    panel = eng.panel
    
    results = []
    
    def _build_with_ret(panel, value_series):
        """构建包含收益标签的测试数据"""
        df = panel[["trade_date","ts_code","fwd_20d_ret"]].copy()
        df["value"] = value_series
        return df.dropna(subset=["value","fwd_20d_ret"])
    
    # 候选1：动量加速因子 = 20日动量 - 60日动量
    print("\n=== 候选1: 动量加速(20-60) ===")
    r1 = eng.test_factor("动量加速", _build_with_ret(panel, panel["20日动量"] - panel["60日动量"]), verbose=True)
    results.append(r1)
    
    # 候选2：超跌信号(RSI低+动量负)
    print("\n=== 候选2: 超跌信号(RSI低+动量负) ===")
    r2 = eng.test_factor("超跌信号", _build_with_ret(panel, -panel["RSI_6"] * panel["20日动量"]), verbose=True)
    results.append(r2)
    
    # 候选3：量价背离
    print("\n=== 候选3: 量价背离 ===")
    if "量能趋势" in panel.columns and "MACD" in panel.columns:
        val = panel["量能趋势"] - panel["MACD"].abs() * np.sign(panel["MACD"])
        r3 = eng.test_factor("量价背离", _build_with_ret(panel, val), verbose=True)
        results.append(r3)
    
    # 候选4：高波动反转
    print("\n=== 候选4: 高波反转 ===")
    if "波动率" in panel.columns:
        val = panel["波动率"] * (-panel["20日动量"])
        r4 = eng.test_factor("高波反转", _build_with_ret(panel, val), verbose=True)
        results.append(r4)
    
    # 候选5：EMA多排强度
    print("\n=== 候选5: 多排强度 ===")
    cols_ok = all(c in panel.columns for c in ["EMA5偏离","EMA10偏离","EMA20偏离"])
    if cols_ok:
        ema5 = panel["EMA5偏离"].fillna(0)
        ema10 = panel["EMA10偏离"].fillna(0)
        ema20 = panel["EMA20偏离"].fillna(0)
        val = (ema5 + ema10 + ema20)/3 - np.abs(ema5-ema10) - np.abs(ema10-ema20)
        r5 = eng.test_factor("多排强度", _build_with_ret(panel, val), verbose=True)
        results.append(r5)
    
    # 候选6：ROE变化
    print("\n=== 候选6: ROE变化 ===")
    if "ROE" in panel.columns:
        panel_sorted = panel.sort_values(["ts_code","trade_date"])
        val = panel_sorted.groupby("ts_code")["ROE"].diff(1)
        r6 = eng.test_factor("ROE变化", _build_with_ret(panel, val.values), verbose=True)
        results.append(r6)
    
    # 汇总
    print(f"\n{'='*60}")
    print("候选因子测试汇总")
    print(f"{'='*60}")
    print(f"{'因子名':16s} | {'IC均值':>8s} | {'IC_IR':>6s} | {'IC夏普':>7s} | {'评分':>4s} | {'建议':16s}")
    print("-"*70)
    for r in results:
        if "error" in r:
            print(f"{r.get('factor_name','?'):16s} | {r['error']}")
        else:
            print(f"{r['factor_name']:16s} | {r['IC_mean']*100:+7.3f}% | "
                  f"{r['IC_IR']:+5.2f} | {r['IC_sharpe_annualized']:+5.2f} | "
                  f"{r['score']:3.0f} | {r['recommendation']:16s}")


if __name__ == "__main__":
    quick_test_candidates()
