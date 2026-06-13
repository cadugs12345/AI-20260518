"""
将新因子合并到主因子面板并测试IC
1. 北向因子（日频市场级，广播到个股）
2. 测试新因子的IC/ICIR/衰减

Usage:
    python src/integrate_new_factors.py [--test-only]

依赖:
    src/alert_system.py (FactorAlertSystem)
    src/factor_engine.py (FactorEngine)
"""
import os, sys, json, time
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config.settings import DATA_FACTORS, DATA_RAW
from src.factor_engine import FactorEngine

NEW_FACTOR_DIR = os.path.join(DATA_FACTORS, "new_factors")
PANEL_PATH = os.path.join(DATA_FACTORS, "factor_panel_with_fwd_v2.parquet")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

def log(msg):
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}")

def broadcast_northbound_factor(nb_df, panel):
    """
    北向资金是市场级数据（每天一个值），广播到个股
    """
    nb = nb_df.copy()
    nb["trade_date"] = pd.to_datetime(nb["trade_date"])
    
    # 选取因子列
    factor_cols = [c for c in nb.columns if c not in ['trade_date']]
    
    # 合并到panel
    merged = panel.merge(nb, on="trade_date", how="left")
    
    # 前向填充（北向数据可能缺周五等情况）
    for col in factor_cols:
        if col in merged.columns:
            merged[col] = merged.groupby("ts_code")[col].transform(
                lambda x: x.ffill().bfill())
    
    return merged, factor_cols


def test_new_factors(panel, factor_names, label="fwd_20d_ret"):
    """批量测试新因子"""
    log(f"测试 {len(factor_names)} 个新因子...")
    
    results = []
    for fname in factor_names:
        if fname not in panel.columns:
            log(f"  ⚠️ {fname} 不在面板中")
            continue
        
        day_data = panel[["trade_date", label, fname]].dropna()
        if len(day_data) < 10000:
            log(f"  ⚠️ {fname} 有效数据不足: {len(day_data):,}")
            continue
        
        # 按日计算Rank IC
        all_dates = sorted(day_data["trade_date"].unique())
        ic_vals = []
        for date in all_dates[::20]:  # 月频采样
            dd = day_data[day_data["trade_date"] == date]
            if len(dd) < 100:
                continue
            v = dd[fname].values.astype(np.float64)
            r = dd[label].values.astype(np.float64)
            mask = ~(np.isnan(v) | np.isnan(r)) & (np.abs(r) < 0.5)
            if mask.sum() < 100:
                continue
            from scipy import stats
            ic, _ = stats.spearmanr(v[mask], r[mask])
            ic_vals.append(ic)
        
        ic_vals = np.array(ic_vals)
        if len(ic_vals) < 10:
            log(f"  ⚠️ {fname} 采样不足: {len(ic_vals)}")
            continue
        
        ic_mean = np.mean(ic_vals)
        ic_std = np.std(ic_vals)
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0
        n_pos = np.sum(ic_vals > 0)
        win_rate = n_pos / len(ic_vals)
        
        log(f"  {fname:25s} IC={ic_mean*100:+6.2f}%  IR={ic_ir:+5.2f}  胜率={win_rate*100:.0f}%  n={len(ic_vals)}")
        
        results.append({
            "factor": fname,
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "ic_ir": ic_ir,
            "win_rate": win_rate,
            "n_samples": len(ic_vals),
        })
    
    return results


if __name__ == "__main__":
    t0 = time.time()
    
    print("="*60)
    print("新因子整合与IC测试")
    print("="*60)
    
    print("\n[1] 加载主因子面板...")
    panel = pd.read_parquet(PANEL_PATH)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    log(f"面板: {len(panel):,}行, {panel['trade_date'].min().date()}~{panel['trade_date'].max().date()}")
    
    all_new_factors = {}
    
    # 加载北向因子
    nb_path = os.path.join(NEW_FACTOR_DIR, "northbound_factors.parquet")
    if os.path.exists(nb_path):
        print("\n[2] 加载北向因子...")
        nb = pd.read_parquet(nb_path)
        log(f"北向数据: {len(nb)}行")
        panel_merged, nb_cols = broadcast_northbound_factor(nb, panel)
        log(f"合并后: {len(panel_merged):,}行")
        all_new_factors["northbound"] = [c for c in nb_cols if c not in panel.columns]
        print(f"  新增字段: {all_new_factors['northbound']}")
    else:
        print("\n[2] ⚠️ 北向因子文件不存在，跳过")
        panel_merged = panel
    
    # 测试北向因子
    if all_new_factors.get("northbound"):
        print("\n[3] 测试北向因子IC...")
        # 排除纯日期字段和衍生na字段
        nb_factors_test = [c for c in all_new_factors["northbound"] 
                          if c not in ['trade_date', 'ts_code'] 
                          and not c.startswith('north_net_ma')]
        results = test_new_factors(panel_merged, nb_factors_test)
        if results:
            result_df = pd.DataFrame(results).sort_values("ic_ir", ascending=False)
            print("\n北向因子IC排名:")
            print(result_df.to_string(index=False))
            
            result_df.to_csv(os.path.join(OUTPUT_DIR, "new_factor_ic_test.csv"), index=False)
    
    print(f"\n总用时: {time.time()-t0:.1f}s")
    print(f"\n✅ 北向因子已集成到面板（如有需要可保存合并面板）")
