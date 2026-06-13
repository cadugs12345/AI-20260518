"""
v28 XGBoost滚动训练 — 轻量版
每半年滚一次，XGBoost 100棵树+早停，回测用v27引擎
"""
import os, sys, time, json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

PROJECT = "/mnt/d/AI-20260518"
os.chdir(PROJECT)
sys.path.insert(0, '.')
import joblib

t0 = time.time()
print("="*60)
print("v28 XGBoost滚动训练 (轻量版)")
print(f"{time.strftime('%F %H:%M')}")
print("="*60)

DATA = "data/factors"

# ===== 1. 加载 =====
print("\n[1] 加载数据...")
panel = pd.read_parquet(f"{DATA}/factor_panel_v6.parquet")
prices = pd.read_parquet(f"{DATA}/full_prices.parquet")
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
prices["trade_date"] = pd.to_datetime(prices["trade_date"])

factor_cols = [c for c in panel.columns 
               if c not in ["ts_code","trade_date","fwd_20d_ret","close","volume","ret_1d",
                   "__fragment_index","__batch_index","__last_in_fragment","__filename","board_break"]]
core15 = ["短期反转","20日动量","60日动量","120日动量","波动率",
          "换手率","量比","量能趋势","BP","EP","MACD","BOLL位置",
          "EMA5偏离","EMA10偏离","EMA20偏离"]
print(f"  面板: {len(panel):,}行, {len(factor_cols)}因子")

# ===== 2. 波动率 =====
print("\n[2] 预计算波动率...")
ps = prices.sort_values(["ts_code","trade_date"]).copy()
ps["ret_1d"] = ps.groupby("ts_code")["close"].pct_change()
ps["vol_60d"] = ps.groupby("ts_code")["ret_1d"].transform(lambda x: x.rolling(60, min_periods=20).std())
ps["vol_60d_ann"] = ps["vol_60d"] * np.sqrt(244)

# ===== 3. 周期节点 =====
all_dates = sorted(panel["trade_date"].unique())
period_dates = [all_dates[i] for i in range(0, len(all_dates), 20) 
                if all_dates[i] >= pd.Timestamp("2021-01-01")]
print(f"\n[3] 20日周期节点: {len(period_dates)}")

price_map, vol_map = {}, {}
for d in period_dates:
    sub = prices[prices["trade_date"] == d][["ts_code","close"]].dropna()
    price_map[d] = dict(zip(sub["ts_code"], sub["close"]))
    v_sub = ps[ps["trade_date"] == d][["ts_code","vol_60d_ann"]].dropna()
    vol_map[d] = dict(zip(v_sub["ts_code"], v_sub["vol_60d_ann"]))

# ===== 4. 生成预测信号（三组）=====
print("\n[4] 生成预测信号...")

# RF模型
rf_md = joblib.load("models/ml_ensemble_v1.joblib")
rf_model, rf_factors = rf_md["model"], rf_md["factor_cols"]

import xgboost as xgb

# 提前算好v12和RF分
print("  计算v12/RF基准...", end=" ", flush=True)
v12_z = panel[core15].rank(pct=True)
panel["s_v12"] = v12_z.mean(axis=1)

# RF分批预测
panel["s_rf"] = 0.0
for date in period_dates:
    idx = panel["trade_date"] == date
    day = panel[idx]
    X = np.column_stack([day[c].values for c in rf_factors])
    X = np.nan_to_num(X.astype(np.float32), nan=0)
    panel.loc[idx, "s_rf"] = rf_model.predict_proba(X)[:, 1]
print(f"done", flush=True)

# v12分数和RF分数做等权作为RF+风控的基础
risk_cols = ["repair_force_10d","board_repair_score","高波反转","量价背离"]
# 确认这些列在panel中
for c in risk_cols:
    if c not in panel.columns:
        print(f"  警告: {c}不在panel中，跳过风控")
        risk_cols.remove(c)

# 按半年重训XGBoost
print("  训练XGBoost（每半年滚动）...")
xgb_preds = []
train_years = {}  # 缓存已训练的模型

# 找到每半年对应的最后一个period日期
half_year_dates = []
for d in period_dates:
    key = f"{d.year}{'H1' if d.month <= 6 else 'H2'}"
    if not half_year_dates or half_year_dates[-1][0] != key:
        half_year_dates.append((key, d))
    else:
        half_year_dates[-1] = (key, d)

print(f"  训练轮次: {len(half_year_dates)}个半年期", flush=True)

for hi, (hk, train_cutoff) in enumerate(half_year_dates):
    if hi < 2:  # 前两个半年不训练（数据不足），用RF替代
        continue
    
    # 训练集：训练截止日期前2年
    train_end = train_cutoff - pd.Timedelta(days=5)
    train_start = train_end - pd.Timedelta(days=2*365)
    
    train = panel[(panel["trade_date"] >= train_start) & (panel["trade_date"] <= train_end)]
    train = train[train["fwd_20d_ret"].notna() & (train["fwd_20d_ret"].abs() < 0.5)]
    
    if len(train) < 20000:
        continue
    
    # 缩小训练数据加速（随机采样20万条）
    if len(train) > 200000:
        train = train.sample(200000, random_state=42)
    
    X_tr = np.nan_to_num(train[rf_factors].values.astype(np.float32), nan=0)
    y_tr = np.clip(train["fwd_20d_ret"].values.astype(np.float32), -0.3, 0.3)
    
    # 轻量XGBoost
    model = xgb.XGBRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=2.0,
        min_child_weight=10,
        random_state=42, verbosity=0, n_jobs=8,
        early_stopping_rounds=10
    )
    # 从训练集中切出验证集
    n_val = min(30000, int(len(train) * 0.2))
    model.fit(X_tr[:-n_val] if n_val > 0 else X_tr,
              y_tr[:-n_val] if n_val > 0 else y_tr,
              eval_set=[(X_tr[-n_val:] if n_val > 0 else X_tr, y_tr[-n_val:] if n_val > 0 else y_tr)],
              verbose=False)
    
    # 预测当前半年后的所有period
    for d in period_dates:
        if d <= train_cutoff:
            continue
        idx = panel["trade_date"] == d
        day = panel[idx]
        X_te = np.nan_to_num(day[rf_factors].values.astype(np.float32), nan=0)
        preds = model.predict(X_te)
        for j, code in enumerate(day["ts_code"].values):
            xgb_preds.append({"trade_date": d, "ts_code": code, "pred_ret": float(preds[j])})
    
    print(f"  {hk} trained: {len(train):,}条 -> 预测至{str(period_dates[-1])[:10]}", flush=True)

print(f"  XGBoost预测: {len(xgb_preds):,}条", flush=True)

# ===== 5. 构建预测表 =====
print("\n[5] 构建回测信号...")

# 对每个日期，各策略取Top50（作为后续回测的选股池）
pred_records = {"v12": [], "rf": [], "rf_risk": [], "xgb_raw": [], "xgb": []}
xgb_df = pd.DataFrame(xgb_preds) if xgb_preds else pd.DataFrame(columns=["trade_date","ts_code","pred_ret"])

for i, d in enumerate(period_dates):
    day = panel[panel["trade_date"] == d].copy()
    if len(day) == 0:
        continue
    
    # 各策略打分
    scores = {}
    scores["v12"] = day["s_v12"].values
    
    # RF
    scores["rf"] = day["s_rf"].values
    
    # RF+风控
    rf_risk = day["s_rf"].values.copy()
    for j, (_, row) in enumerate(day.iterrows()):
        triggered = False
        for rc in risk_cols:
            v = row.get(rc, np.nan)
            if rc == "量价背离" and not np.isnan(v) and v > 0.03: triggered = True
            if rc in ["repair_force_10d","board_repair_score"] and not np.isnan(v) and v < -0.05: triggered = True
            if rc == "高波反转" and not np.isnan(v) and v < -0.03: triggered = True
        if triggered:
            rf_risk[j] = -999
    scores["rf_risk"] = rf_risk
    
    # XGBoost原始分（未回退RF）
    xgb_day = xgb_df[xgb_df["trade_date"] == d]
    xgb_scores = np.full(len(day), -999.0)
    if len(xgb_day) > 0:
        xgb_map = dict(zip(xgb_day["ts_code"], xgb_day["pred_ret"]))
        for j, code in enumerate(day["ts_code"].values):
            if code in xgb_map:
                xgb_scores[j] = xgb_map[code]
    scores["xgb_raw"] = xgb_scores
    
    # XGBoost（无预测时回退RF）
    xgb_fallback = xgb_scores.copy()
    if len(xgb_day) > 0:
        xgb_map = dict(zip(xgb_day["ts_code"], xgb_day["pred_ret"]))
        for j, code in enumerate(day["ts_code"].values):
            xgb_fallback[j] = xgb_map.get(code, day["s_rf"].values[j])
    scores["xgb"] = xgb_fallback
    
    # 记录Top50
    for strategy in ["v12", "rf", "rf_risk", "xgb_raw", "xgb"]:
        sc = scores[strategy]
        order = np.argsort(-sc)
        top50_codes = day["ts_code"].values[order][:50]
        top50_scores = sc[order][:50]
        for j in range(50):
            pred_records[strategy].append({
                "trade_date": d, "ts_code": top50_codes[j],
                "pred_ret": float(top50_scores[j])
            })

# ===== 6. 回测 =====
print(f"\n[6] 回测引擎...")
STAMP, COMM, SLIP = 0.001, 0.0002, 0.002

preds = {}
for k in ["v12", "rf", "rf_risk", "xgb_raw", "xgb"]:
    preds[k] = pd.DataFrame(pred_records[k])
    print(f"  {k}: {len(preds[k]):,}条, {preds[k]['trade_date'].nunique()}期")

def backtest(pred_df, n_stocks=30, target_vol=0.15, label=""):
    pred_dates = sorted(pred_df["trade_date"].unique())
    cash = 0.03
    holdings = {}
    navs = [1.0]
    
    for i in range(len(pred_dates) - 1):
        date = pred_dates[i]
        sell_date = pred_dates[i+1]
        px_buy = price_map.get(date, {})
        px_sell = price_map.get(sell_date, {})
        stock_vol = vol_map.get(date, {})
        
        hold_val = sum(shares * px_buy.get(c, 0) for c, shares in holdings.items())
        total_val = hold_val + cash
        
        sell_proceeds = 0
        for code, shares in holdings.items():
            px = px_sell.get(code, 0)
            if px > 0:
                val = shares * px
                cost = val * (STAMP + COMM + SLIP)
                sell_proceeds += val - cost
        cash = cash + sell_proceeds
        holdings = {}
        
        day_pred = pred_df[pred_df["trade_date"] == date].sort_values("pred_ret", ascending=False)
        selected = list(day_pred.head(n_stocks)["ts_code"].values)
        
        selected_vols = [stock_vol.get(c, np.nan) for c in selected]
        selected_vols = [v for v in selected_vols if not np.isnan(v) and v > 0.01]
        
        if len(selected_vols) >= 5:
            median_vol = np.median(selected_vols)
            pos_ratio = min(target_vol / median_vol, 1.0)
            pos_ratio = max(pos_ratio, 0.05)
        else:
            median_vol = np.nan
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
                        if bought > 0: holdings[code] = bought
                cash -= per * len(holdings)
        
        new_port = sum(shares * px_sell.get(c, 0) for c, shares in holdings.items())
        new_total = new_port + cash
        ret = new_total / total_val - 1 if total_val > 0 else 0
        navs.append(navs[-1] * (1 + ret))
    
    pnl = np.array(navs[1:]) / np.array(navs[:-1]) - 1
    nav_arr = np.array(navs)
    tr = nav_arr[-1] - 1
    n_years = len(pnl) / 13
    ar = nav_arr[-1] ** (1/n_years) - 1 if n_years > 0 and nav_arr[-1] > 0 else 0
    vol = np.std(pnl) * np.sqrt(13)
    sr = np.mean(pnl) / np.std(pnl) * np.sqrt(13) if np.std(pnl) > 0 else 0
    dd = np.maximum.accumulate(nav_arr) - nav_arr
    mdd = dd.max()
    wr = np.mean(pnl > 0)
    calmar = ar / mdd if mdd > 0 else 0
    
    print(f"  {label:20s}: 年化{ar*100:+.1f}% | 夏普{sr:.2f} | 回撤{mdd*100:.1f}% | 胜率{wr*100:.0f}% | 卡玛{calmar:.2f} | {len(pnl)}期")
    
    return {"ret": f"{ar*100:+.1f}%", "sharpe": f"{sr:.2f}", "mdd": f"{mdd*100:.1f}%",
            "wr": f"{wr*100:.0f}%", "calmar": f"{calmar:.2f}", "vol": f"{vol*100:.1f}%",
            "n": len(pnl), "_ar": ar, "_sr": sr, "_mdd": mdd}

def run_all(ns=30, tv=0.15, label=""):
    print(f"\n回测 (T{ns} 目波{tv*100:.0f}%):")
    r = {}
    for k, name in [("v12","v12等权"), ("rf","RF"), ("rf_risk","RF+风控"),
                    ("xgb_raw","XGBoost纯"), ("xgb","XGBoost+RF回退")]:
        if len(preds[k]) > 0:
            r[k] = backtest(preds[k], ns, tv, name)
    return r

r30 = run_all(30, 0.15, "T30_V15")
r50 = run_all(50, 0.15, "T50_V15")

result = {"T30_V15": r30, "T50_V15": r50}
json.dump(result, open("output/backtest_v28_xgb.json", "w"), indent=2, default=str)

print(f"\n{'='*60}")
print(f"{'策略':20s} {'T30年化':>8s} {'T30夏普':>8s} {'T30回撤':>8s} {'T50年化':>8s} {'T50夏普':>8s}")
print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for k, name in [("v12","v12等权"), ("rf","RF"), ("rf_risk","RF+风控"),
                ("xgb_raw","XGBoost纯"), ("xgb","XGBoost+RF回退")]:
    if k in r30:
        d30 = r30[k]; d50 = r50[k]
        print(f"  {name:20s} {d30['ret']:>8s} {d30['sharpe']:>8s} {d30['mdd']:>8s} "
              f"{d50['ret']:>8s} {d50['sharpe']:>8s} {d50['mdd']:>8s}")

print(f"\n⏱ {(time.time()-t0)/60:.1f}分")
print("✅ output/backtest_v28_xgb.json")
