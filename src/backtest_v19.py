"""
生成强化衍生因子 + 完整面板集成 + v19回测

衍生因子设计:
  1. 高波反转 (HighVol_Reversal) — HIghVol_Rev5/Rev10/Rev20
     高波动率股的短期反转效应
  2. 超跌信号 (Oversold) — 价格低偏离 + 缩量
  3. 量价背离 (Divergence) — 放量滞涨/缩量不跌
  4. 多排强度 (MultiBull) — 多条均线多头排列

输出: data/factors/factor_panel_v5.parquet
      data/factors/pred_20d_v19.pkl
"""
import sys, os, json, time
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config.settings import DATA_FACTORS

t0 = time.time()

print("="*60)
print("衍生因子强化 → 面板v5 + v19回测")
print("="*60)

# 1. 加载
panel = pd.read_parquet(os.path.join(DATA_FACTORS, "factor_panel_with_fwd_v2.parquet"))
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
print(f"[1] 面板: {len(panel):,}行, 股票{panel['ts_code'].nunique():,}只")

prices = pd.read_parquet(os.path.join(DATA_FACTORS, "full_prices.parquet"))
prices["trade_date"] = pd.to_datetime(prices["trade_date"])

# 按股票排序
panel = panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
prices = prices.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

# 2. 衍生因子
# 从full_prices合并close到面板
panel = panel.merge(prices[["trade_date","ts_code","close"]], on=["trade_date","ts_code"], how="left")
print("\n[2] 计算衍生因子...")

# 2a. 高波反转因子 — 滚动20日波动率 × 负收益（前期涨越多越容易反转）
panel["ret_1d"] = panel.groupby("ts_code")["close"].transform(lambda x: x.pct_change(1))
panel["ret_20d"] = panel.groupby("ts_code")["close"].transform(lambda x: x.pct_change(20))
panel["vol_20d"] = panel.groupby("ts_code")["ret_1d"].transform(
    lambda x: x.rolling(20, min_periods=10).std())
panel["高波反转"] = -panel["vol_20d"] * panel["ret_20d"]  # 高波动+前期下跌→看多

# 2b. 超跌信号 — 价格低于50日线20% + 缩量到60日均量50%以下
panel["ma50"] = panel.groupby("ts_code")["close"].transform(lambda x: x.rolling(50, min_periods=30).mean())
panel["tr_ma60"] = panel.groupby("ts_code")["换手率"].transform(
    lambda x: x.rolling(60, min_periods=30).mean())
panel["price_dev_50"] = panel["close"] / panel["ma50"] - 1
panel["tr_ratio"] = panel["换手率"] / panel["tr_ma60"].replace(0, np.nan)
panel["超跌信号"] = (-panel["price_dev_50"]).clip(0)*0.5 + (1 - panel["tr_ratio"].clip(0, 1)).clip(0)*0.5
# 只在真正超跌时才有信号
panel.loc[panel["price_dev_50"] > -0.1, "超跌信号"] = 0

# 2c. 量价背离 — 价格涨但量缩 = 背离看跌；价格跌但量缩 = 背离看涨（换手率替代成交量）
panel["price_trend"] = panel.groupby("ts_code")["close"].transform(lambda x: x.pct_change(5))
panel["tr_trend"] = panel.groupby("ts_code")["换手率"].transform(lambda x: x.pct_change(5))
panel["量价背离"] = panel["price_trend"] * panel["tr_trend"]  # 正=同向，负=背离
panel["量价背离信号"] = -panel["量价背离"]  # 背离（负值）→ 看涨

# 2d. 多排强度 (多条均线多头排列)
panel["ma_5"] = panel.groupby("ts_code")["close"].transform(lambda x: x.rolling(5, min_periods=4).mean())
panel["ma_10"] = panel.groupby("ts_code")["close"].transform(lambda x: x.rolling(10, min_periods=7).mean())
panel["ma_20"] = panel.groupby("ts_code")["close"].transform(lambda x: x.rolling(20, min_periods=14).mean())
panel["ma_60"] = panel.groupby("ts_code")["close"].transform(lambda x: x.rolling(60, min_periods=30).mean())

panel["多排强度"] = 0
# 多头排列 = MA5 > MA10 > MA20 > MA60
panel.loc[(panel["ma_5"] > panel["ma_10"]) & (panel["ma_10"] > panel["ma_20"]), "多排强度"] += 1
panel.loc[(panel["ma_10"] > panel["ma_20"]) & (panel["ma_20"] > panel["ma_60"]), "多排强度"] += 1
panel.loc[panel["close"] > panel["ma_20"], "多排强度"] += 1
panel.loc[panel["close"] > panel["ma_60"], "多排强度"] += 1

# 2e. 波动率变化 — 波动率收缩→突破
panel["vol_10d"] = panel.groupby("ts_code")["ret_1d"].transform(
    lambda x: x.rolling(10, min_periods=5).std())
panel["vol_40d"] = panel.groupby("ts_code")["ret_1d"].transform(
    lambda x: x.rolling(40, min_periods=20).std())
panel["波动收缩"] = 1 - (panel["vol_10d"] / panel["vol_40d"].replace(0, np.nan)).clip(0, 1)

# 清除中间列
for c in ["ma50","ma60_vol","ma_5","ma_10","ma_20","ma_60",
          "ret_20d","vol_20d","price_trend","vol_trend","price_dev_50","vol_ratio",
          "vol_10d","vol_40d","ma50","ma60_vol"]:
    if c in panel.columns:
        del panel[c]

# 清理更多中间列
for c in ["ma50","tr_ma60","tr_ratio" ,"price_dev_50","price_trend","tr_trend","ma_5","ma_10","ma_20","ma_60","ret_20d","vol_20d","vol_10d","vol_40d"]:
    if c in panel.columns:
        del panel[c]

derived_factors = ["高波反转", "超跌信号", "量价背离信号", "多排强度", "波动收缩"]
print(f"  新增因子: {derived_factors}")
for f in derived_factors:
    v = panel[f].dropna()
    print(f"    {f:12s} | {panel[f].notna().mean()*100:5.1f}%非空 | μ={v.mean():+.4f} | σ={v.std():.4f}")

# 3. IC测试
print("\n[3] 衍生因子IC测试 (对fwd_20d_ret)...")
label = "fwd_20d_ret"
all_dates = sorted(panel["trade_date"].unique())[40:]

for f in derived_factors:
    ic_vals = []
    for date in all_dates[::10]:
        dd = panel[panel["trade_date"] == date][[f, label]].dropna()
        if len(dd) < 200:
            continue
        v = dd[f].values.astype(np.float64)
        r = dd[label].values.astype(np.float64)
        mask = ~(np.isnan(v) | np.isnan(r)) & (np.abs(r) < 0.5) & np.isfinite(v)
        if mask.sum() < 200:
            continue
        ic, _ = stats.spearmanr(v[mask], r[mask])
        ic_vals.append(ic)
    
    ic_arr = np.array(ic_vals)
    if len(ic_arr) < 20:
        print(f"  {f:12s} n={len(ic_arr)} 样本不足")
        continue
    ic_mean = np.mean(ic_arr)
    ic_ir = ic_mean / np.std(ic_arr) if np.std(ic_arr) > 0 else 0
    wr = np.mean(ic_arr > 0)
    print(f"  {f:12s} IC={ic_mean*100:+6.2f}%  IR={ic_ir:+5.2f}  胜率={wr*100:.0f}%  n={len(ic_arr)}")

# 4. 保存面板v5
panel.to_parquet(os.path.join(DATA_FACTORS, "factor_panel_v5.parquet"))
print(f"\n[4] ✅ 面板v5已保存: factor_panel_v5.parquet ({len(panel):,}行)")

# 5. v19回测
print("\n[5] v19回测（原始权重+衍生因子）...")

# 使用原来的22个因子 + 衍生因子（等权合成+ML）
# 先用原来的权重（不带衰减调整，恢复原始）
original_weights = {
    "60日动量": 0.127, "20日动量": 0.120, "市值": 0.117,
    "EMA20偏离": 0.059, "120日动量": 0.058, "换手率": 0.050,
    "波动率": 0.049, "EMA5偏离": 0.049, "RSI_24": 0.046,
    "MACD": 0.044, "OBV": 0.041, "BOLL位置": 0.041,
    "RSI_12": 0.039, "RSI_6": 0.039, "量能趋势": 0.038,
    "EMA10偏离": 0.038, 
}

# 加入衍生因子（各分配5%权重）
for f in derived_factors[:4]:
    original_weights[f] = 0.05

# 量价背离信号直接用，不会有冲突

total = sum(original_weights.values())
original_weights = {k: v/total for k, v in original_weights.items()}

factor_cols = [c for c in original_weights.keys() if c in panel.columns]
print(f"  因子数: {len(factor_cols)}")

# 多因子合成
for f in factor_cols:
    panel[f"rank_{f}"] = panel.groupby("trade_date")[f].rank(pct=True, method="dense")

panel["composite_v19"] = sum(panel[f"rank_{f}"] * original_weights[f] for f in factor_cols)

# ML训练（XGB+LightGBM ensemble）
print("\n[6] XGB + LightGBM训练...")
from xgboost import XGBRegressor

factor_cols_all = [c for c in panel.columns 
                  if c not in ["ts_code","trade_date","fwd_20d_ret","fwd_5d_ret",
                              "composite_v19","均值","成交量","close","ret_1d"]
                  and c.startswith("rank_") or c in derived_factors
                  and panel[c].dtype in ("float64","int64")]

# 只取rank_*因子和衍生因子
factor_cols_all = [c for c in panel.columns 
                  if (c.startswith("rank_") or c in derived_factors)
                  and c != "composite_v19"
                  and panel[c].dtype in ("float64","int64")]

panel_ml = panel[["trade_date","ts_code","composite_v19","fwd_20d_ret"] + factor_cols_all].copy()
dates = sorted(panel_ml["trade_date"].unique())
train_end = dates[int(len(dates) * 0.85)]
train = panel_ml[panel_ml["trade_date"] < train_end].dropna(subset=["fwd_20d_ret"] + factor_cols_all)
test = panel_ml[panel_ml["trade_date"] >= train_end].dropna(subset=["fwd_20d_ret"] + factor_cols_all)

X_train = train[factor_cols_all].values.astype(np.float32)
y_train = train["fwd_20d_ret"].values.astype(np.float32)
X_test = test[factor_cols_all].values.astype(np.float32)

model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=42, n_jobs=-1, verbosity=0)
model.fit(X_train, y_train)

pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

# ensemble with composite
train_pred = train[["trade_date","ts_code"]].copy()
train_pred["pred_ret"] = pred_train * 0.7 + train["composite_v19"].values * 0.3
test_pred = test[["trade_date","ts_code"]].copy()
test_pred["pred_ret"] = pred_test * 0.7 + test["composite_v19"].values * 0.3

pred = pd.concat([train_pred, test_pred], ignore_index=True)
pred.to_pickle(os.path.join(DATA_FACTORS, "pred_20d_v19.pkl"))
print(f"  预测保存: pred_20d_v19.pkl")

# [7] 净值计算
print("\n[7] 净值计算...")
pred_dates = sorted(pred["trade_date"].unique())

prices_sorted = prices.sort_values(["ts_code","trade_date"]).copy()
prices_sorted["ret_1d"] = prices_sorted.groupby("ts_code")["close"].pct_change()
prices_sorted["vol_60d"] = prices_sorted.groupby("ts_code")["ret_1d"].transform(
    lambda x: x.rolling(60, min_periods=20).std())
prices_sorted["vol_60d_ann"] = prices_sorted["vol_60d"] * np.sqrt(244)

vol_map, price_map = {}, {}
for d in pred_dates:
    sub = prices[prices["trade_date"] == d][["ts_code","close"]].dropna()
    price_map[d] = dict(zip(sub["ts_code"], sub["close"]))
    v_sub = prices_sorted[prices_sorted["trade_date"] == d][["ts_code","vol_60d_ann"]].dropna()
    vol_map[d] = dict(zip(v_sub["ts_code"], v_sub["vol_60d_ann"]))

STAMP, COMM, SLIP = 0.001, 0.0002, 0.002

results = {}
for n_stocks, target_vol in [(30, 0.15), (30, 0.20), (50, 0.15), (50, 0.20)]:
    cash = 0.03
    holdings = {}
    navs = [1.0]
    dates_nav = []
    
    for i, date in enumerate(pred_dates[:-1]):
        sell_date = pred_dates[i+1]
        px_buy = price_map.get(date, {})
        px_sell = price_map.get(sell_date, {})
        stock_vol = vol_map.get(date, {})
        
        hold_val = sum(shares * px_buy.get(c, 0) for c, shares in holdings.items())
        total_val = hold_val + cash
        
        sell_proceeds = 0
        for code, shares in list(holdings.items()):
            px = px_sell.get(code, 0)
            if px > 0:
                val = shares * px
                cost = val * (STAMP + COMM + SLIP)
                sell_proceeds += val - cost
        cash += sell_proceeds
        holdings = {}
        
        day_pred = pred[pred["trade_date"] == date].sort_values("pred_ret", ascending=False)
        selected = list(day_pred.head(n_stocks)["ts_code"].values)
        
        selected_vols = [stock_vol.get(c, np.nan) for c in selected]
        selected_vols = [v for v in selected_vols if not np.isnan(v) and v > 0.01]
        
        if len(selected_vols) >= 5:
            median_vol = np.median(selected_vols)
            pos_ratio = min(target_vol / median_vol, 1.0)
            pos_ratio = max(pos_ratio, 0.05)
        else:
            pos_ratio = 1.0
        
        if selected and cash > 0.001:
            available = cash * pos_ratio * 0.98
            if available > 0.001:
                per = available / len(selected)
                for code in selected:
                    px = px_buy.get(code, 0)
                    if px > 0 and per > 0:
                        buy_cost = per * (COMM + SLIP)
                        bought = (per - buy_cost) / px
                        if bought > 0:
                            holdings[code] = bought
                cash -= per * len(holdings)
        
        new_port = sum(shares * px_sell.get(c, 0) for c, shares in holdings.items())
        new_total = new_port + cash
        ret = new_total / total_val - 1 if total_val > 0 else 0
        navs.append(navs[-1] * (1 + ret))
        dates_nav.append(date)
    
    nav_array = np.array(navs)
    pnl = nav_array[1:] / nav_array[:-1] - 1
    sr = np.mean(pnl) / np.std(pnl) * np.sqrt(13) if np.std(pnl) > 0 else 0
    tr = nav_array[-1] - 1
    ann_ret = (1 + tr) ** (12 / max(len(pnl), 1)) - 1 if len(pnl) > 0 else 0
    dd = np.maximum.accumulate(nav_array) - nav_array
    mdd = dd.max()
    wr = np.mean(pnl > 0)
    calmar = tr / mdd if mdd > 0 else 0
    
    ver = f"T{n_stocks}_V{int(target_vol*100)}"
    results[ver] = {"total_return": float(tr), "annualized_return": float(ann_ret),
                    "sharpe": float(sr), "max_dd": float(mdd), "win_rate": float(wr), "calmar": float(calmar)}
    
    print(f"  {ver:12s} {tr*100:6.1f}%年| {ann_ret*100:6.1f}% | 夏普{sr:5.2f} | 回撤{mdd*100:5.1f}% | 胜率{wr*100:3.0f}%")

# 保存结果
out_path = os.path.join(DATA_FACTORS, "backtest_v19_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n✅ v19结果保存: {out_path}")
print(f"总用时: {(time.time()-t0)/60:.1f}分")
