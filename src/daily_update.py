"""
每日监控更新 — 直接import调用（避免subprocess子进程超时SIGKILL）
"""
import sys, os, json, time, shutil
from datetime import datetime

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M")
log_lines = []

def log(msg):
    log_lines.append(str(msg))
    print(msg)

log(f"📊 每日监控更新 — {TIMESTAMP}")
print()

# 1. 预警扫描 (独立进程运行，避免OOM和import错误)
log("[预警扫描] ============")
try:
    import subprocess
    r = subprocess.run(["bash", "-c", "cd /mnt/d/AI-20260518 && source .venv/bin/activate && python3 src/alert_system.py 2>/dev/null"],
                      capture_output=True, text=True, timeout=120)
    if r.returncode == 0:
        # 检查是否有最新报告
        with open("alerts/latest.json") as f:
            meta = json.load(f)
        n = meta.get("n_alerts", 0)
        log(f"  预警: {n}个")
        alert_ok = True
    else:
        log(f"  ⚠️ alert返回{r.returncode}: {r.stderr[:200]}")
        alert_ok = False
except Exception as e:
    log(f"  ⚠️ 预警: {e}")
    alert_ok = False

# 2. Dashboard生成 (subprocess调用，单独运行不超时)
log("[Dashboard] ============")
try:
    import subprocess
    r = subprocess.run(["bash", "-c", "cd /mnt/d/AI-20260518 && source .venv/bin/activate && python3 src/dashboard.py --no-refresh 2>/dev/null"],
                      capture_output=True, text=True, timeout=120)
    if r.returncode == 0:
        shutil.copy2("output/dashboard.html", "output/dashboard_latest.html")
        log(f"  → output/dashboard_latest.html")
        dash_ok = True
    else:
        log(f"  ⚠️ dashboard返回{r.returncode}")
        dash_ok = False
except Exception as e:
    log(f"  ⚠️ Dashboard: {e}")
    dash_ok = False

# 3. 权重优化 (独立子进程)
log("[权重优化] ============")
try:
    import subprocess
    r = subprocess.run(["bash", "-c", "cd /mnt/d/AI-20260518 && source .venv/bin/activate && python3 src/optimize_weights.py 2>/dev/null"],
                      capture_output=True, text=True, timeout=30)
    if r.returncode == 0:
        out = r.stdout.strip()[-200:] if r.stdout else ""
        log(f"  {out}")
        opt_ok = True
    else:
        log(f"  ⚠️ opt返回{r.returncode}")
        opt_ok = False
except Exception as e:
    log(f"  ⚠️ 权重优化: {e}")
    opt_ok = False

# 4. 生成日报
log("\n[日报生成] ============")
daily = []
daily.append(f"📊 A股量化监控日报 — {datetime.now().strftime('%F')}")
daily.append(f"{'='*48}")

# 预警
try:
    with open("alerts/latest.json") as f:
        meta = json.load(f)
    with open(meta.get("path","")) as f:
        rd = json.load(f)
    
    alerts_list = rd.get("alerts", [])
    if alerts_list:
        daily.append(f"\n🔴 因子预警 ({len(alerts_list)}个)")
        for a in alerts_list:
            icon = "🔴" if a.get("level") in ("严重衰减","显著衰减") else "🟡"
            daily.append(f"  {icon} {a['factor']}: {a['level']} (IR {a.get('recent_ic_ir',0):+.2f})")
except:
    pass

# 权重
try:
    with open("alerts/optimized_weights.json") as f:
        ow = json.load(f)
    weights_pct = ow.get("weights_pct", {})
    if weights_pct:
        daily.append(f"\n⚙️ 建议权重 (Top8)")
        for i, (name, w) in enumerate(list(weights_pct.items())[:8]):
            daily.append(f"  {i+1}. {name:12s}: {w}")
except:
    pass

# 策略
try:
    with open("data/factors/backtest_v17_results.json") as f:
        v17 = json.load(f)
    daily.append(f"\n📈 v17策略")
    for cfg in ["T50_V15", "T30_V15"]:
        r = v17.get(cfg, {})
        if r:
            daily.append(f"  {cfg}: 夏普{r.get('sharpe',0):.2f} | 收益{r.get('total_return',0)*100:.1f}% | 回撤{r.get('max_dd',0)*100:.1f}%")
except:
    pass

daily.append(f"\n{'='*48}")
daily.append(f"📅 {TIMESTAMP} | 明天9:30自动更新")

report_text = "\n".join(daily)
with open("output/daily_brief.txt", "w") as f:
    f.write(report_text)

print(report_text)

with open("output/daily_update.log", "a") as f:
    f.write(f"\n{'='*60}\n{TIMESTAMP}\n{'='*60}\n")
    f.write("\n".join(log_lines) + "\n")

print(f"\n✅ 更新完成")
print(f"  Dashboard: output/dashboard_latest.html")
print(f"  预警: alerts/latest.json")
print(f"  日报: output/daily_brief.txt")
