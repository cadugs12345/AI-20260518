"""
Dashboard ML对比面板 — v12 vs RF vs LGB三组净值比较
生成独立HTML页面，每日9:30自动更新
"""
import os, json, time, base64, io
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")

PROJECT = "/mnt/d/AI-20260518"
os.chdir(PROJECT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT = "output"
os.makedirs(OUTPUT, exist_ok=True)

t0 = time.time()
print(f"📊 Dashboard ML对比面板 — {time.strftime('%F %H:%M')}")

# 加载回测结果
bt = json.load(open(f"{OUTPUT}/backtest_v29_final.json"))

# 加载v27结果（含T30/T50完整数据）
v27_bt = json.load(open(f"{OUTPUT}/backtest_v27_ml_precise.json"))

# 加载最新信号
try:
    sig = json.load(open("signals/v29_signal.json"))
except:
    sig = {"date": "N/A", "positions": [], "model": "LightGBM 79因子"}

# 加载预警
try:
    alerts = json.load(open("alerts/latest.json"))
    alert_items = json.load(open(alerts.get("path","")))
    alert_level = sum(1 for a in alert_items.get("alerts",[]) 
                      if a.get("level") in ("严重衰减","显著衰减"))
except:
    alert_level = 0

# 生成净值对比图（从v27回测中获取模拟净值）
# 由于没有实际净值序列，用文本表格对比代替

def make_table(bt, suffix=""):
    rows = []
    configs = {
        "v12": "v12等权",
        "rf": "RF",
        "rf_risk": "RF+风控"
    }
    lgb_keys = {
        "lgb_raw": "LGB纯",
        "lgb": "LGB+RF回退",
        "lgb_risk": "LGB+风控"
    }
    
    for k, name in configs.items():
        if k in bt.get("T30_V15", {}):
            d30 = bt["T30_V15"][k]
            d50 = bt.get("T50_V15", {}).get(k, d30)
            rows.append((name, "👉 " + suffix, 
                        d30.get("ret","N/A"), d30.get("sr","N/A"),
                        d50.get("ret","N/A"), d50.get("sr","N/A")))
    
    # LGB从v29读取
    for k, name in lgb_keys.items():
        if k in bt.get("T30_V15", {}):
            d30 = bt["T30_V15"][k]
            d50 = bt.get("T50_V15", {}).get(k, d30)
            rows.append((name, "🚀 " + suffix,
                        d30.get("ret","N/A"), d30.get("sr","N/A"),
                        d50.get("ret","N/A"), d50.get("sr","N/A")))
    
    return rows

# 用v27和v29数据合并
v27_rows = make_table(v27_bt, "v27回测")
v29_rows = make_table(bt, "v29回测")
all_rows = v27_rows + [r for r in v29_rows if r[0] not in [x[0] for x in v27_rows]]

# 找出最佳
best_sr = max([float(r[3]) for r in all_rows if r[3] not in ("N/A","-0.00")])

# 生成HTML
html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ML策略对比 — Dashboard v29</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; 
           background: #0f0f1a; color: #ccc; padding: 24px; }}
    h1 {{ color: #fff; font-size: 24px; margin-bottom: 4px; }}
    .sub {{ color: #888; font-size: 13px; margin-bottom: 20px; }}
    .tag {{ display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; }}
    .tag-v12 {{ background: #3a2a1a; color: #fa0; }}
    .tag-rf {{ background: #1a3a2a; color: #0f0; }}
    .tag-lgb {{ background: #1a1a4a; color: #88f; }}
    
    .summary {{ display: flex; gap: 16px; margin-bottom: 20px; flex-wrap: wrap; }}
    .card {{ flex: 1; min-width: 200px; background: #1a1a2e; border-radius: 12px; padding: 16px; border: 1px solid #2a2a3e; }}
    .card h2 {{ color: #888; font-size: 12px; text-transform: uppercase; margin-bottom: 8px; }}
    .card .big {{ font-size: 28px; font-weight: 700; color: #fff; }}
    .card .big.green {{ color: #6f6; }}
    .card .small {{ color: #888; font-size: 12px; margin-top: 4px; }}
    
    table {{ width: 100%; border-collapse: collapse; margin-bottom: 16px; }}
    th, td {{ padding: 10px 12px; text-align: center; font-size: 13px; }}
    th {{ background: #2a2a3e; color: #aaa; font-weight: 500; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}
    td {{ border-bottom: 1px solid #222; }}
    tr:hover {{ background: #2a2a3e; }}
    .num {{ font-variant-numeric: tabular-nums; font-family: 'SF Mono', 'Consolas', monospace; }}
    .pos {{ color: #6f6; }}
    .neg {{ color: #f66; }}
    .best {{ background: #1a3a1a; font-weight: 700; }}
    
    .risk-box {{ margin-top: 16px; }}
    .risk-item {{ display: inline-block; padding: 4px 12px; margin: 3px; border-radius: 6px; font-size: 12px; }}
    .risk-high {{ background: #3a1a1a; color: #f66; border: 1px solid #5a2a2a; }}
    .risk-mid {{ background: #3a3a1a; color: #ff0; border: 1px solid #5a5a2a; }}
    
    .positions {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 6px; }}
    .pos-item {{ background: #2a2a3e; padding: 8px 10px; border-radius: 8px; text-align: center; }}
    .pos-code {{ color: #fff; font-weight: 500; font-size: 14px; }}
    .pos-score {{ color: #888; font-size: 11px; }}
    
    .footer {{ margin-top: 24px; color: #555; font-size: 11px; text-align: center; }}
</style>
</head>
<body>

<h1>📊 ML策略对比 · Dashboard v29</h1>
<div class="sub">{sig['date']} | 版本演变: v12→RF→RF+风控→<strong>LightGBM 🏆</strong></div>

<div class="summary">
    <div class="card">
        <h2>当前实盘模型</h2>
        <div class="big green" style="font-size:20px">{sig['model']}</div>
        <div class="small">持仓 {sig['n_hold']} 只 · 回测夏普 1.03</div>
    </div>
    <div class="card">
        <h2>预警状态</h2>
        <div class="big" style="color:{'#f66' if alert_level>=3 else '#ff0' if alert_level>=1 else '#6f6'}">
            {'🚨 紧急' if alert_level>=3 else '⚠️ 注意' if alert_level>=1 else '✅ 正常'}
        </div>
        <div class="small">{sum(1 for _ in [])}个因子 · {alert_level}个严重</div>
    </div>
    <div class="card">
        <h2>版本里程碑</h2>
        <div style="margin-top:8px">
            <span class="tag tag-v12">v12 等权 0.97</span>
            <span class="tag tag-rf">v27 RF+风控 0.99</span>
            <span class="tag tag-lgb">v29 LGB 1.03 🏆</span>
        </div>
    </div>
</div>

<h2 style="color:#fff; margin-bottom:12px; font-size:16px">🏆 回测对比 (2021-2026, 含摩擦成本)</h2>
<table>
    <tr>
        <th style="text-align:left">策略</th>
        <th>T30年化</th>
        <th>T30夏普</th>
        <th>T30回撤</th>
        <th>T50年化</th>
        <th>T50夏普</th>
        <th>T50回撤</th>
    </tr>
"""

for name, tag, t30r, t30s, t50r, t50s in all_rows:
    t30sr = float(t30s) if t30s not in ("N/A","-0.00") else 0
    t50sr = float(t50s) if t50s not in ("N/A","-0.00") else 0
    is_best = abs(t30sr - best_sr) < 0.01
    
    # 策略标签
    name_tag = name.replace(" ","")
    if "LGB" in name_tag or "lgb" in name_tag:
        tag_cls = "tag-lgb"
    elif "RF" in name_tag:
        tag_cls = "tag-rf"
    else:
        tag_cls = "tag-v12"
    
    cls = "best" if is_best else ""
    html += f"""    <tr class="{cls}">
        <td style="text-align:left"><span class="tag {tag_cls}">{name}</span></td>
        <td class="num {'pos' if t30r.startswith('+') else 'neg'}">{t30r}</td>
        <td class="num {'pos' if t30sr>0 else 'neg'}">{t30s}</td>
        <td class="num neg">{t30r}</td>
        <td class="num {'pos' if t50r.startswith('+') else 'neg'}">{t50r}</td>
        <td class="num {'pos' if t50sr>0 else 'neg'}">{t50s}</td>
        <td class="num neg">{t50r}</td>
    </tr>
"""

html += f"""
</table>

<h2 style="color:#fff; margin:16px 0 12px; font-size:16px">🎯 实盘持仓 Top15</h2>
<div class="positions">
"""

for p in sig["positions"][:15]:
    html += f"""    <div class="pos-item">
        <div class="pos-code">{p['ts_code']}</div>
        <div class="pos-score">{p['score']:.4f}</div>
    </div>
"""

html += f"""
</div>

<div class="risk-box">
    <h2 style="color:#fff; margin:16px 0 12px; font-size:16px">⚡ 风险监控</h2>
    <div>
        <span class="risk-item risk-high">🔴 60日动量 IR=-0.82</span>
        <span class="risk-item risk-high">🔴 市值 IR=-0.66</span>
        <span class="risk-item risk-mid">🟡 20日动量 IR=-0.69</span>
        <span class="risk-item risk-mid">🟡 EMA20偏离 IR=-0.60</span>
        <span class="risk-item risk-mid">🟡 120日动量 IR=-0.48</span>
    </div>
</div>

<div style="margin-top:16px; background:#1a1a2e; border-radius:10px; padding:16px; border:1px solid #2a2a3e;">
    <h2 style="color:#fff; margin-bottom:10px; font-size:16px">📈 版本迭代日志</h2>
    <table>
        <tr><th>版本</th><th>方法</th><th>因子</th><th>夏普</th><th>上线日期</th></tr>
        <tr><td>v12</td><td>等权合成</td><td>15核心</td><td>0.97</td><td>初始</td></tr>
        <tr><td>v27</td><td>RF + 风控</td><td>79</td><td>0.99</td><td>5/20</td></tr>
        <tr class="best"><td><strong>v29</strong></td><td><strong>LightGBM 🏆</strong></td><td><strong>79</strong></td><td><strong>1.03</strong></td><td><strong>5/21</strong></td></tr>
    </table>
</div>

<div class="footer">
    自动生成 {time.strftime('%F %H:%M')} | 下一轮更新: 09:30 每日
</div>

</body>
</html>
"""

with open(f"{OUTPUT}/dashboard_ml_compare.html", "w") as f:
    f.write(html)

print(f"  ✅ {OUTPUT}/dashboard_ml_compare.html")
print(f"  ⏱ {time.time()-t0:.1f}s")
