"""
因子权重自动优化器
基于预警衰减数据，自动给出最优因子权重配置

输入: alerts/latest.json (最新预警报告)
逻辑:
  严重衰减(decay<-0.3): 权重×0.5
  需关注(decay>-0.3 & recent_ir<-0.5): 权重×0.8
  正常(recent_ir>-0.5): 保留
  增强(recent_ir上升>0.1): 权重×1.2

输出: alerts/optimized_weights.json (可直接供回测使用)
"""
import json, os, sys
from datetime import datetime

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT)
ALERTS_DIR = "alerts"

# 1. 读取最新预警
with open(f"{ALERTS_DIR}/latest.json") as f:
    meta = json.load(f)

report_path = meta.get("path", "")
with open(report_path) as f:
    data = json.load(f)

alerts = data.get("alerts", [])
print(f"[权重优化] 加载{len(alerts)}个因子预警")

# 2. 基础权重（原始因子重要性）
base_weights = {
    "60日动量": 0.127, "20日动量": 0.120, "市值": 0.117,
    "EMA20偏离": 0.059, "120日动量": 0.058, "换手率": 0.050,
    "波动率": 0.049, "EMA5偏离": 0.049, "RSI_24": 0.046,
    "MACD": 0.044, "OBV": 0.041, "BOLL位置": 0.041,
    "RSI_12": 0.039, "RSI_6": 0.039, "量能趋势": 0.038,
    "EMA10偏离": 0.038, 
}

# 3. 根据预警调整
changes = {}
for a in alerts:
    name = a["factor"]
    if name not in base_weights:
        continue
    
    level = a.get("level", "")
    decay = a.get("decay", 0)
    recent_ir = a.get("recent_ic_ir", 0)
    
    old_w = base_weights[name]
    
    if level == "严重衰减":
        multiplier = 0.5
        reason = "严重衰减"
    elif level == "显著衰减":
        multiplier = 0.6
        reason = "显著衰减"
    elif level == "需关注" and recent_ir < -0.5:
        multiplier = 0.8
        reason = "需关注+低IR"
    elif decay < -0.15:
        multiplier = 0.9
        reason = "轻微衰减"
    elif recent_ir > -0.3 and recent_ir < 0:
        multiplier = 1.0
        reason = "正常"
    elif recent_ir >= 0:
        multiplier = 1.2
        reason = "近期转正"
    else:
        multiplier = 1.0
        reason = "保持"
    
    # 额外：如果换手率/波动率有增强趋势（decay>0.1且recent_ir上升），增强
    if name in ("换手率", "波动率") and decay > 0.1:
        multiplier = min(multiplier * 1.15, 1.3)
        reason = "近期增强"
    
    new_w = old_w * multiplier
    changes[name] = {"old": old_w, "new": new_w, "mult": multiplier, "reason": reason}
    base_weights[name] = new_w

# 对未预警的因子也保持
for name in base_weights:
    if name not in changes:
        changes[name] = {"old": base_weights[name], "new": base_weights[name], "mult": 1.0, "reason": "未预警"}

# 4. 归一化
total = sum(base_weights.values())
normalized = {k: v/total for k, v in base_weights.items()}

# 5. 输出
result = {
    "timestamp": data["timestamp"],
    "n_alerts": len(alerts),
    "adjustments": {},
    "weights": normalized,
}

# 只输出有变化的
for name, c in sorted(changes.items(), key=lambda x: x[1]["new"], reverse=True):
    if abs(c["mult"] - 1.0) > 0.01:
        result["adjustments"][name] = {
            "old_weight": round(c["old"]*100, 1),
            "new_weight": round(c["new"]*100, 1),
            "change": f"{c['mult']:.2f}x",
            "reason": c["reason"]
        }

# 格式化成百分比并排序
weights_pct = {k: f"{v*100:.1f}%" for k, v in sorted(normalized.items(), key=lambda x: x[1], reverse=True)}
result["weights_pct"] = weights_pct

with open(f"{ALERTS_DIR}/optimized_weights.json", "w") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

# 打印
print(f"权重调整:")
print(f"{'因子':15s} | {'原权':>6s} | {'新权':>6s} | {'调整'}")
print("-"*45)
for name, c in sorted(changes.items(), key=lambda x: x[1]["new"], reverse=True):
    if abs(c["mult"] - 1.0) > 0.01:
        print(f"{name:15s} | {c['old']*100:5.1f}% | {c['new']*100:5.1f}% | {c['mult']:.2f}x {c['reason']}")

print(f"\n✅ 已保存: {ALERTS_DIR}/optimized_weights.json")
print(f"  今日调整: {len(result['adjustments'])}个因子")
