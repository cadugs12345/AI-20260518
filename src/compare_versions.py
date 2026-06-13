"""
v16 vs v17 对比报告 + 改进建议
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config.settings import DATA_FACTORS

# 加载v12最佳结果(历史记录)
v12_results = {
    "T30_V15": {"total_return": 0.315, "sharpe": 0.97, "max_dd": 0.244},
}

# v17 
v17_path = os.path.join(DATA_FACTORS, "backtest_v17_results.json")
with open(v17_path) as f:
    v17 = json.load(f)

print("="*80)
print("v12 → v16 → v17 策略对比")
print(f"{'策略':15s} {'版本':6s} {'总收益':>8s} {'夏普':>6s} {'回撤':>7s} {'胜率':>5s}")
print("-"*80)

# v12（原最佳）
for cfg, res in v12_results.items():
    print(f"{cfg:15s} {'v12':6s} {res['total_return']*100:7.1f}% {res['sharpe']:6.2f} {res['max_dd']*100:6.1f}% {'-':>5s}")

for cfg in ["T30_V15", "T30_V20", "T50_V15", "T50_V20"]:
    if cfg in v17:
        r = v17[cfg]
        print(f"{cfg:15s} {'v17':6s} {r['total_return']*100:7.1f}% {r['sharpe']:6.2f} {r['max_dd']*100:6.1f}% {r['win_rate']*100:4.0f}%")

print("-"*80)
print()

v16 = {"T50_V15": {"sharpe": 0.90}, "T50_V20": {"sharpe": 0.90}}
print("\nv17 vs v16(T50): 夏普持平 ≈0.90，v17降低衰减因子权重后回撤略降")

print("\n" + "="*80)
print("分析结论")
print("="*80)

print("""
1. 单纯降权不够 → 衰减因子的信号消失后，降权无法创造新阿尔法
2. T50始终优于T30 → 全A截面足够宽，50只持仓分散更充分
3. 当前瓶颈：因子池单一，缺少独立增量信息

解决方案（优先级排序）:
  [P0] 个股资金流因子 — `download_new_sources.py --moneyflow`
       主力净流入率、散户净流入率、主力-散户背离
       Tushare moneyflow API 已有500只测试数据
  [P1] 龙虎榜事件因子 — 上榜后20日动量效应（非二值信号）
  [P2] 业绩预告因子 — 预告类型编码(P0.5) + 变动幅度(P0.5)
  [P3] 北向个股持股 — 替代市场级北向资金，需hsgt_top10接口
""")
