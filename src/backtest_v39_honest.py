"""
📈 v39 月频调仓 · 可信回测 (v3)
============================================
简化但严格的净值计算：
- 每期买入Top30，持有20个交易日
- 20天到期卖出（或调仓日卖出，取先到的）
- 扣冲击成本+佣金+印花税
- 固定模型（v38），不重训练
============================================
"""
import sys, os, time, warnings, joblib, json
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.chdir("/mnt/d/AI-20260518")
t0 = time.time()

N_HOLD = 30
COST_PER_TRADE = 0.005  # 佣金+滑点(单边)
STAMP = 0.001           # 印花税(单边)
OUTPUT_DIR = "backtest_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("📈 v39 月频调仓 · 可信回测(v3)")
print("="*70)

# ─── 1. 数据 ───
print("\n[1/4] 加载数据...")
m38 = joblib.load("models/live_lgb_v38_final.joblib")
model = m38["model"]
fc = m38["factor_cols"]

panel = pd.read_parquet("data/factors/factor_panel_v5_final.parquet",
    columns=["ts_code", "trade_date"] + fc)
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
panel = panel.sort_values(["trade_date", "ts_code"])

prices = pd.read_parquet("data/factors/full_prices.parquet")
prices["trade_date"] = pd.to_datetime(prices["trade_date"])
prices = prices.drop_duplicates(subset=["ts_code", "trade_date"])
prices = prices.sort_values(["ts_code", "trade_date"])

print(f"  面板: {len(panel):,}行 | 价格: {len(prices):,}行")

# 价格索引
print("\n[2/4] 构建价格索引...")
px_idx = {}
for code, grp in prices.groupby("ts_code"):
    grp = grp.sort_values("trade_date")
    px_idx[code] = {"d": grp["trade_date"].values, "c": grp["close"].values}
print(f"  {len(px_idx)}只")

def get_px(code, dt):
    i = px_idx.get(code)
    if i is None: return np.nan
    pos = np.searchsorted(i["d"], dt, side="right") - 1
    if pos < 0 or pos >= len(i["d"]) or i["d"][pos] != dt: return np.nan
    return i["c"][pos]

# ─── 3. 调仓日 ───
print("\n[3/4] 固定模型预测+回测...")
all_dates = sorted(panel["trade_date"].unique())
dfd = pd.DataFrame({"trade_date": all_dates})
dfd["ym"] = dfd["trade_date"].astype(str).str[:7]
edates = sorted(dfd.groupby("ym")["trade_date"].first().reset_index()["trade_date"].unique())
edates = [d for d in edates if d >= pd.Timestamp("2020-01-01")]
# 截止到有20天未来数据
edates = [d for d in edates if d <= prices["trade_date"].max() - pd.Timedelta(days=30)]
print(f"  调仓日: {len(edates)}个 ({edates[0].date()} ~ {edates[-1].date()})")

# 每期：买入Top30 → 20个交易日后卖出
nav = 1.0
navs = [1.0]
dates_rec = []
dd_rec = []
records = []

for i, ed in enumerate(edates):
    day = panel[panel["trade_date"] == ed].copy()
    if len(day) < 60: continue
    
    # 选股
    X = day[fc].fillna(0).values.astype(np.float32)
    day["score"] = model.predict(X)
    top = day.nlargest(N_HOLD, "score")
    
    # 找20个交易日后的日期
    # 用ed后第20个有数据的日期
    d_idx = all_dates.index(ed) if ed in all_dates else -1
    if d_idx < 0 or d_idx + 20 >= len(all_dates):
        continue
    sell_date = all_dates[d_idx + 20]
    
    # 单期收益
    code_rets = []
    for _, r in top.iterrows():
        buy = get_px(r["ts_code"], ed)
        if np.isnan(buy) or buy <= 0: continue
        sell = get_px(r["ts_code"], sell_date)
        if np.isnan(sell) or sell <= 0: continue
        gross_ret = sell / buy - 1
        # 成本 = 买入佣金滑点 + 卖出佣金滑点印花税
        cost = COST_PER_TRADE + COST_PER_TRADE + STAMP
        net_ret = (1 + gross_ret) / (1 + cost) - 1  # 近似
        code_rets.append(net_ret)
    
    if len(code_rets) >= 5:
        period_ret = np.mean(code_rets)
        nav *= (1 + period_ret)
        
        peak = max(navs)
        dd = nav / peak - 1
        
        navs.append(nav)
        dates_rec.append(ed)
        dd_rec.append(dd)
        
        records.append({
            "entry_date": ed,
            "sell_date": sell_date,
            "period_ret": period_ret,
            "n_valid": len(code_rets),
        })
        
        if (i+1) % 20 == 0:
            print(f"  [{i+1}/{len(edates)}] {ed.date()} ret={period_ret*100:+.2f}% nav={nav:.4f}")

# ─── 4. 统计 ───
print("\n[4/4] 统计结果...")

df = pd.DataFrame(records)
rets = df["period_ret"].values
n_months = len(rets)
total_years = (df["entry_date"].iloc[-1] - df["entry_date"].iloc[0]).days / 365.25

total_ret = navs[-1] - 1
annual_ret = navs[-1] ** (1 / total_years) - 1
annual_vol = rets.std() * np.sqrt(12)
sharpe = annual_ret / annual_vol if annual_vol > 0 else 0
max_dd = min(dd_rec)
win_rate = (rets > 0).mean()

# 滚动夏普
rs = []
for j in range(12, len(rets)):
    if rets[j-12:j].std() > 0:
        rs.append(rets[j-12:j].mean() / rets[j-12:j].std() * np.sqrt(12))

# 分年
df["year"] = df["entry_date"].astype(str).str[:4]
yearly = []
for yr in sorted(df["year"].unique()):
    yr_rets = df[df["year"] == yr]["period_ret"].values
    yr_nav = np.prod(1 + yr_rets)
    yearly.append({"year": int(yr), "N": len(yr_rets), 
                   "ret": yr_nav - 1, "win": (yr_rets > 0).mean()})
yearly = pd.DataFrame(yearly)

print("\n" + "="*70)
print("📊 v39 可信回测 v3 · 结果")
print("="*70)
print(f"  固定模型(v38) | Top{N_HOLD} | 总成本含印花税 ~1.2%/边")
print(f"  期: {df['entry_date'].iloc[0].date()} ~ {df['entry_date'].iloc[-1].date()} ({total_years:.1f}年)")
print(f"  期数: {n_months}")
print()
print("--- 核心指标 ---")
print(f"  累计净收益:     {total_ret*100:+.2f}%")
print(f"  年化收益:       {annual_ret*100:+.2f}%")
print(f"  年化波动(月频): {annual_vol*100:.2f}%")
print(f"  年化夏普:       {sharpe:.2f}")
print(f"  最大回撤:       {max_dd*100:.2f}%")
print(f"  月胜率:         {win_rate*100:.1f}%")
if rs:
    print(f"  12月滚动夏普:   均值 {np.mean(rs):.2f} | 当前 {rs[-1]:.2f}")

print()
print("--- 分年 ---")
for _, r in yearly.iterrows():
    print(f"  {r['year']}: {r['N']:2d}期  年收益{r['ret']*100:>+8.2f}%  月胜率{r['win']*100:>5.1f}%")

print()
print("--- 最近12个月 ---")
for _, r in df.tail(12).iterrows():
    print(f"  {r['entry_date'].date()} ~ {r['sell_date'].date()}  "
          f"ret={r['period_ret']*100:+.2f}%  n={r['n_valid']}")

# 保存
nav_df = pd.DataFrame({"entry_date": dates_rec, "nav": navs[1:], "dd": dd_rec})
nav_df.to_parquet(f"{OUTPUT_DIR}/v39_honest_nav.parquet")
summary = {
    "version": "v39 可信回测v3",
    "holdings": N_HOLD,
    "total_return_pct": round(total_ret * 100, 2),
    "annual_return_pct": round(annual_ret * 100, 2),
    "annual_vol_pct": round(annual_vol * 100, 2),
    "sharpe": round(sharpe, 2),
    "max_dd_pct": round(max_dd * 100, 2),
    "monthly_win_rate_pct": round(win_rate * 100, 1),
}
json.dump(summary, open(f"{OUTPUT_DIR}/v39_honest_summary.json", "w"), indent=2, ensure_ascii=False)

print(f"\n✅ {OUTPUT_DIR}/v39_honest_nav.parquet")
print(f"⏱ {(time.time()-t0)/60:.1f}分")
