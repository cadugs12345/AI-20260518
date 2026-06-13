"""
根据预警结果调整因子组合权重
- 严重衰减 → 降权50%
- 需关注 → 降权20%
- 增强 → 升权20%
- 正常 → 不变

Usage:
    python src/adjust_weights.py
"""
import sys, os, json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config.settings import DATA_FACTORS

ALERTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "alerts")

# 基于历史重要性的基准权重（取自factor_lifecycle分析）
BASE_WEIGHTS = {
    "60日动量": 0.127, "20日动量": 0.120, "市值": 0.117,
    "EMA20偏离": 0.059, "120日动量": 0.058, "换手率": 0.055,
    "EMA5偏离": 0.045, "波动率": 0.045,
}
DEFAULT_WEIGHT = 0.04

# 预警等级 → 权重调整系数
ADJUST_MAP = {
    "严重衰减": 0.5,
    "需关注": 0.8,
    "轻微衰减": 0.9,
    "正常": 1.0,
    "增强": 1.2,
}

if __name__ == "__main__":
    print("="*60)
    print("因子权重调整 — 基于IC衰减预警")
    print("="*60)
    
    # 加载最新预警
    latest_path = os.path.join(ALERTS_DIR, "latest.json")
    if not os.path.exists(latest_path):
        print("⚠️ 未找到预警数据，请先运行 alert_system.py")
        sys.exit(1)
    
    with open(latest_path) as f:
        latest = json.load(f)
    
    report_path = latest.get("path")
    if not report_path or not os.path.exists(report_path):
        print("⚠️ 预警报告文件不存在")
        sys.exit(1)
    
    with open(report_path) as f:
        report = json.load(f)
    
    alerts = {a["factor"]: a["level"] for a in report.get("alerts", [])}
    factors_status = {}
    for r in report.get("all_factors", []):
        factors_status[r["factor"]] = r["level"]
    
    print(f"\n加载预警: {len(alerts)} 个严重/关注, 共 {len(factors_status)} 个因子")
    print()
    print(f"{'因子':20s} | {'基准权重':>8s} | {'预警状态':>12s} | {'调整系数':>8s} | {'新权重':>8s}")
    print("-" * 65)
    
    new_weights = {}
    for f, base in BASE_WEIGHTS.items():
        status = factors_status.get(f, "正常")
        adj = ADJUST_MAP.get(status, 1.0)
        new_w = base * adj
        new_weights[f] = new_w
        print(f"{f:20s} | {base*100:6.1f}% | {status:>12s} | {adj:7.2f}x | {new_w*100:6.1f}%")
    
    # 其他因子按默认权重
    for f, status in factors_status.items():
        if f not in BASE_WEIGHTS:
            adj = ADJUST_MAP.get(status, 1.0)
            new_w = DEFAULT_WEIGHT * adj
            new_weights[f] = new_w
            print(f"{f:20s} | {DEFAULT_WEIGHT*100:6.1f}% | {status:>12s} | {adj:7.2f}x | {new_w*100:6.1f}%")
    
    # 归一化
    total = sum(new_weights.values())
    new_weights = {k: v/total for k, v in new_weights.items()}
    
    print("\n[归一化后权重]")
    print(f"{'因子':20s} | {'新权重':>8s}")
    print("-" * 35)
    for f in sorted(new_weights, key=new_weights.get, reverse=True):
        print(f"{f:20s} | {new_weights[f]*100:6.1f}%")
    
    # 保存
    result = {
        "base_on": latest.get("report_time", "unknown"),
        "weights": new_weights,
        "note": "严重衰减÷2, 需关注×0.8, 轻微衰减×0.9, 正常×1.0, 增强×1.2"
    }
    out_path = os.path.join(ALERTS_DIR, "adjusted_weights.json")
    with open(out_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 调整权重已保存: {out_path}")
    
    # 输出可用于 factor_combination.py 的配置
    print("\n📋 可复制到factor_combination.py的权重配置:")
    for f in sorted(new_weights, key=new_weights.get, reverse=True):
        print(f"    \"{f}\": {new_weights[f]:.6f},")
