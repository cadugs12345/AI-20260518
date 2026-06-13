#!/usr/bin/env python3
"""
信号对比监控 — 对比今日持仓 vs 昨日持仓，生成调仓建议（含买入/卖出手数）
"""
import os, sys, json, time
import pandas as pd
import warnings; warnings.filterwarnings("ignore")
os.chdir("/mnt/d/AI-20260518"); sys.path.insert(0, '.')
tt = time.time()

SIGNALS = "signals"
PER_STOCK = 100000  # 每只买10万

print(f"💹 持仓对比监控 — {time.strftime('%F %H:%M')}")
print("="*50)

# 加载最新收盘价
print("  加载最新价格...", end=" ", flush=True)
prices = pd.read_parquet("data/factors/full_prices.parquet")
prices["trade_date"] = pd.to_datetime(prices["trade_date"])
latest_px_date = prices["trade_date"].max()
px_latest = prices[prices["trade_date"] == latest_px_date][["ts_code", "close"]].copy()
px_dict = dict(zip(px_latest["ts_code"], px_latest["close"]))
print(f"ok ({latest_px_date.date()})", flush=True)

def calc_lots(code):
    """按每只10万算手数"""
    px = px_dict.get(code)
    if px is None or px <= 0:
        return 0, 0
    shares = int(PER_STOCK / px / 100) * 100
    actual_cost = shares * px
    lots = shares // 100
    return lots, actual_cost

# 找今日和昨日的持仓文件
files = sorted([f for f in os.listdir(SIGNALS) 
                if f.startswith("v38_positions_") and f.endswith(".csv")])
print(f"历史信号文件: {len(files)}个")

if len(files) < 2:
    print("⚠️ 不足2个信号文件，无法对比")
    sys.exit(0)

today_file = files[-1]
yesterday_file = files[-2]
print(f"  今日: {today_file}")
print(f"  昨日: {yesterday_file}")

# 读入需要的列（忽略可能的重复列）
today = pd.read_csv(f"{SIGNALS}/{today_file}")
yesterday = pd.read_csv(f"{SIGNALS}/{yesterday_file}")
# 去掉重复列尾缀
for col in list(today.columns):
    if col.endswith('.1'):
        today = today.drop(columns=[col])
for col in list(yesterday.columns):
    if col.endswith('.1'):
        yesterday = yesterday.drop(columns=[col])
# 给今日股票加价格和手数
today["close"] = today["ts_code"].map(px_dict).round(2)
lots_info = today["ts_code"].apply(lambda c: pd.Series(calc_lots(c), index=["lots", "actual_cost"]))
today = pd.concat([today, lots_info], axis=1)
# 清理所有尾缀重复列
for c in list(today.columns):
    if c.endswith(".1"):
        today = today.drop(columns=[c])
# 去重列名
today = today.loc[:,~today.columns.duplicated()].copy()

today_codes = set(today["ts_code"])
yesterday_codes = set(yesterday["ts_code"])

# 新买入
new_buy = today_codes - yesterday_codes
new_buy_df = today[today["ts_code"].isin(new_buy)].copy()
new_buy_df = new_buy_df.sort_values("rank")

# 需卖出
need_sell = yesterday_codes - today_codes
need_sell_df = yesterday[yesterday["ts_code"].isin(need_sell)].copy()
need_sell_df = need_sell_df.sort_values("rank")

# 保留的
kept = today_codes & yesterday_codes
kept_today = today[today["ts_code"].isin(kept)].copy()
kept_yesterday = yesterday[yesterday["ts_code"].isin(kept)].copy()
kept_merged = kept_today.merge(kept_yesterday[["ts_code","rank","weight"]], 
                                on="ts_code", suffixes=("_new","_old"))
kept_merged["rank_change"] = kept_merged["rank_old"] - kept_merged["rank_new"]
kept_merged = kept_merged.sort_values("rank_new")

# 卖出总金额估算
sell_total = 0
for _, r in need_sell_df.iterrows():
    px = px_dict.get(r["ts_code"], 0)
    buy_amount = PER_STOCK  # 当初买入约10万
    sell_total += buy_amount

print(f"\n📋 调仓建议")
print(f"  新买入: {len(new_buy)}只")
print(f"  需卖出: {len(need_sell)}只")
print(f"  继续持有: {len(kept)}只")
print()

if len(need_sell) > 0:
    print(f"🚨 需卖出 ({len(need_sell)}只):")
    for _, r in need_sell_df.iterrows():
        nm = r.get("name", "")
        ind = r.get("industry", "")
        print(f"  🔴 {r['ts_code']} {nm:8s} {ind:8s} "
              f"(昨日第{r['rank']}名, {r['weight']}%)")
    est_cash = len(need_sell) * PER_STOCK / 10000
    print(f"  → 预计回收资金: {est_cash:.0f}万")
    print()

if len(new_buy) > 0:
    print(f"🟢 新买入 ({len(new_buy)}只): 每只买10万（取整手）")
    total_cost = 0
    for _, r in new_buy_df.iterrows():
        nm = str(r.get("name", ""))
        ind = str(r.get("industry", ""))
        px = float(r["close"] if "close" in r.index and pd.notna(r["close"]) else 0)
        lots = int(r["lots"] if "lots" in r.index and pd.notna(r["lots"]) else 0)
        cost = float(r["actual_cost"] if "actual_cost" in r.index and pd.notna(r["actual_cost"]) else 0)
        total_cost += cost
        print(f"  🟢 {r['ts_code']} {nm:8s} {ind:8s} "
              f"{px:>8.2f}元 × {lots:>3d}手 = {cost/10000:>4.1f}万  (第{r['rank']}名)")
    print(f"  → 预计投入: {total_cost/10000:.1f}万")
    print()

if len(kept_merged) > 0:
    print(f"⏸️ 继续持有 (排名变化):")
    for _, r in kept_merged.head(10).iterrows():
        ch = r["rank_change"]
        arrow = "↑" if ch > 0 else ("↓" if ch < 0 else "—")
        nm = str(r.get("name_new", r.get("name", "")))
        ind = str(r.get("industry_new", r.get("industry", "")))
        lots = int(r["lots"] if "lots" in r.index and pd.notna(r["lots"]) else 0)
        px = float(r["close"] if "close" in r.index and pd.notna(r["close"]) else 0)
        print(f"  {arrow} {r['ts_code']} {nm:8s} {ind:8s} "
              f"{px:.2f}元 {lots}手 "
              f"排名: {int(r['rank_old'])}→{int(r['rank_new'])} ({ch:+d})")
    print()

# 汇总
buy_total = sum(today["actual_cost"].fillna(0).astype(float))
print(f"{'='*50}")
print(f"📌 周一开盘操作清单")
if len(need_sell) > 0:
    print(f"  卖出: {len(need_sell)}只（回收约{len(need_sell)*10:.0f}万）")
print(f"  买入: {len(new_buy)}只（投入约{buy_total/10000:.1f}万）")
print(f"  持有: {len(kept)}只不变")
print(f"  总持仓: {len(today)}只 | 总投入约{buy_total/10000:.1f}万")
print(f"{'='*50}")

# 保存调仓建议（CSV + 手数）
records = []
for _, r in need_sell_df.iterrows():
    records.append({
        "action": "卖出", "ts_code": r["ts_code"],
        "name": r.get("name",""), "industry": r.get("industry",""),
        "reason": f"退出Top10(昨日第{r['rank']}名)"
    })
for _, r in new_buy_df.iterrows():
    px = r.get("close", 0)
    lots = r.get("lots", 0)
    cost = r.get("actual_cost", 0)
    records.append({
        "action": "买入", "ts_code": r["ts_code"],
        "name": r.get("name",""), "industry": r.get("industry",""),
        "price": f"{px:.2f}" if px else "",
        "lots": int(lots),
        "cost": f"{cost:.0f}" if cost else "",
        "reason": f"新进Top10(今日第{r['rank']}名)"
    })

adj_df = pd.DataFrame(records)
adj_path = f"{SIGNALS}/trade_advice_{today_file.replace('v38_positions_','').replace('.csv','')}.csv"
adj_df.to_csv(adj_path, index=False, encoding="utf-8-sig")
print(f"\n  调仓建议保存: {adj_path}")

print(f"\n⏱ {time.time()-tt:.1f}s")
