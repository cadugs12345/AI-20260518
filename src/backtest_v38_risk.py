#!/usr/bin/env python3
"""
backtest_v38_risk.py — v36c rank标签回测 + 风控规则

风控规则（基于 v27 经验 + 数据验证）：
1. repair_force_10d < -5% → 超跌修复失败（"断板"风险）
2. 高波反转 < -3% → 高波动/常见反转
3. 量价背离 > 0.5 → 量价显著背离（收紧到>50%, 原v27太松0.03命中18%）

评分 + 指衰权重 = 行业中性 + 风控过滤
"""
import os, sys, time, json, gc
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
os.chdir("/mnt/d/AI-20260518"); sys.path.insert(0, '.')
import joblib, lightgbm as lgb
from scipy.stats import spearmanr
tt = time.time()

STAMP, COMM, SLIP = 0.001, 0.0002, 0.002
print("="*60, flush=True)
print("v38 风控回测 — v36c rank标签 + 指衰 + 行业中性 + 风控", flush=True)
print(time.strftime('%F %H:%M'), flush=True)
print("="*60, flush=True)

# ====== 数据 ======
print("\n加载数据...", flush=True)
panel = pd.read_parquet("data/factors/factor_panel_v6.parquet")
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
panel["ts_code"] = panel["ts_code"].astype(str)
print(f"  面板: {len(panel):,}行 × {len(panel.columns)}列", flush=True)

prices = pd.read_parquet("data/factors/full_prices.parquet")
prices["trade_date"] = pd.to_datetime(prices["trade_date"])
prices["ret_1d"] = prices.groupby("ts_code")["close"].pct_change()
print(f"  价格: {len(prices):,}行", flush=True)

# ====== 构建rank标签 ======
print("\n构建rank标签...", flush=True)
mask = panel["fwd_20d_ret"].notna() & (panel["fwd_20d_ret"].abs() < 0.5)
panel["label_rank"] = np.nan
panel.loc[mask, "label_rank"] = (
    panel[mask].groupby("trade_date")["fwd_20d_ret"]
    .rank(pct=True, ascending=True)
)
print(f"  标签: {mask.sum():,}条有效", flush=True)

# ====== 回测参数 ======
ps = prices.sort_values(["ts_code","trade_date"]).copy()
ps["vol_60d_ann"] = ps.groupby("ts_code")["ret_1d"].transform(
    lambda x: x.rolling(60, min_periods=20).std()) * np.sqrt(244)

pdts = sorted(panel["trade_date"].unique())
period_dates = [pdts[i] for i in range(0, len(pdts), 20)
                if pdts[i] >= pd.Timestamp("2021-01-01")]

half_dates = []
for d in period_dates:
    k = f"{d.year}{'H1' if d.month <= 6 else 'H2'}"
    if not half_dates or half_dates[-1][0] != k:
        half_dates.append((k, d))
    else:
        half_dates[-1] = (k, d)

rf_md = joblib.load("models/ml_ensemble_v1.joblib")
factor_cols = rf_md["factor_cols"]
print(f"  因子: {len(factor_cols)}个 | 换仓期: {len(period_dates)} | 训窗口: {len(half_dates)}", flush=True)


# ====== 滚动训练（rank标签）=====#
def train_lgb_rank(name=""):
    """训练LightGBM + rank标签，返回预测表"""
    preds = {"trade_date": [], "ts_code": [], "pred_ret": [],
             "repair_force_10d": [], "高波反转": [], "量价背离": []}
    
    for hi, (hk, train_cutoff) in enumerate(half_dates):
        if hi < 3:
            continue  # 需要前几期积攒训练数据
        
        train_end = train_cutoff - pd.Timedelta(days=5)
        tr = panel[
            (panel["trade_date"] >= train_end - pd.Timedelta(days=730)) &
            (panel["trade_date"] <= train_end) &
            panel["label_rank"].notna()
        ].copy()
        
        if len(tr) < 20000:
            continue
        if len(tr) > 100000:
            tr = tr.sample(100000, random_state=42)
        
        X_tr = tr[factor_cols].fillna(0).values.astype(np.float32)
        y_tr = tr["label_rank"].values.astype(np.float32)
        nv = max(1, int(len(tr) * 0.15))
        
        m = lgb.LGBMRegressor(
            n_estimators=500, max_depth=3, learning_rate=0.02,
            subsample=0.7, colsample_bytree=0.7,
            reg_alpha=0.2, reg_lambda=1.0,
            min_child_weight=20, min_data_in_leaf=100,
            random_state=42, verbose=-1, n_jobs=8,
        )
        m.fit(
            X_tr[:-nv], y_tr[:-nv],
            eval_set=[(X_tr[-nv:], y_tr[-nv:])],
            callbacks=[lgb.early_stopping(30, verbose=False)],
            eval_metric="mse",
        )
        
        # 预测后续所有换仓日
        for d in period_dates:
            if d <= train_cutoff:
                continue
            day = panel[panel["trade_date"] == d]
            if len(day) == 0:
                continue
            pp = m.predict(day[factor_cols].fillna(0).values.astype(np.float32))
            
            for j, code in enumerate(day["ts_code"].values):
                preds["trade_date"].append(d)
                preds["ts_code"].append(code)
                preds["pred_ret"].append(float(pp[j]))
                preds["repair_force_10d"].append(
                    float(day.iloc[j].get("repair_force_10d", np.nan)))
                preds["高波反转"].append(
                    float(day.iloc[j].get("高波反转", np.nan)))
                preds["量价背离"].append(
                    float(day.iloc[j].get("量价背离", np.nan)))
        
        if (hi + 1) % 3 == 0:
            print(f"  {name}: {hk} ({hi+1}/{len(half_dates)-3})", flush=True)
    
    pdf = pd.DataFrame(preds)
    # 去重：同一date+code 取首次预测
    pdf = pdf.drop_duplicates(subset=["trade_date", "ts_code"], keep="first")
    print(f"  {name}: {len(pdf):,}预测, {pdf['trade_date'].nunique()}期", flush=True)
    return pdf


# ====== 回测引擎（支持风控）=====#
def backtest(pred_df, n_stocks=30, target_vol=0.15, label="",
             min_date=pd.Timestamp("2023-01-01"),
             risk_enabled=True, risk_config=None):
    """
    回测引擎，支持风控过滤 + 指数衰减权重 + 行业中性
    risk_config: {
        'repair_force_10d': -0.05,  # None=不启用
        '高波反转': -0.03,
        '量价背离': 0.5,
    }
    """
    if risk_config is None:
        risk_config = {
            "repair_force_10d": -0.05,
            "高波反转": -0.03,
            "量价背离": 0.5,
        }
    
    # 行业数据
    import tushare as ts
    from config.settings import TS_TOKEN
    ts.set_token(TS_TOKEN)
    pro = ts.pro_api()
    stk = pro.query("stock_basic", exchange="", list_status="L",
                    fields="ts_code,industry")
    si = dict(zip(stk["ts_code"], stk["industry"]))
    
    pred_dates = sorted(pred_df["trade_date"].unique())
    pred_dates = [d for d in pred_dates if d >= min_date]
    if len(pred_dates) < 2:
        return None
    
    cash = 0.03
    holdings = {}
    navs = [1.0]
    risk_log = {"total": 0, "removed": 0}
    
    for i in range(len(pred_dates) - 1):
        date = pred_dates[i]
        sell_date = pred_dates[i + 1]
        
        # 价格数据
        px_buy = {}
        px_sell = {}
        sv = {}
        for _, r in ps[ps["trade_date"] == date].iterrows():
            px_buy[r["ts_code"]] = r["close"]
            sv[r["ts_code"]] = (r["vol_60d_ann"]
                                if pd.notna(r.get("vol_60d_ann")) else 0.3)
        for _, r in ps[ps["trade_date"] == sell_date].iterrows():
            px_sell[r["ts_code"]] = r["close"]
        
        # 卖出
        hv = sum(shares * px_buy.get(c, 0) for c, shares in holdings.items())
        tv_ = hv + cash
        sp = 0
        for c, shares in holdings.items():
            px = px_sell.get(c, 0)
            if px > 0:
                sp += shares * px - shares * px * (STAMP + COMM + SLIP)
        cash += sp
        holdings = {}
        
        # 选股
        day = pred_df[pred_df["trade_date"] == date].sort_values(
            "pred_ret", ascending=False).reset_index(drop=True)
        
        codes = list(day["ts_code"])
        scores = day["pred_ret"].values
        
        # 风控过滤
        risk_log["total"] += 1
        if risk_enabled:
            r_mask = np.zeros(len(day), dtype=bool)
            for j, (_, row) in enumerate(day.iterrows()):
                for factor_name, threshold in risk_config.items():
                    if threshold is None:
                        continue
                    val = row.get(factor_name, np.nan)
                    if pd.isna(val):
                        continue
                    if threshold < 0 and val < threshold:
                        r_mask[j] = True
                    elif threshold > 0 and val > threshold:
                        r_mask[j] = True
            
            safe_idx = np.where(~r_mask)[0]
            risk_log["removed"] += int(r_mask.sum())
        else:
            safe_idx = np.arange(len(day))
        
        # 行业中性选择（从安全列表中选）
        selected_idx = []
        ind_count = {}
        for j in safe_idx:
            if j >= len(codes):
                continue
            ind = si.get(codes[j], "其他")
            if ind_count.get(ind, 0) < 3:
                selected_idx.append(j)
                ind_count[ind] = ind_count.get(ind, 0) + 1
            if len(selected_idx) >= n_stocks:
                break
        
        # 如果风控后不够，放宽行业限制
        if len(selected_idx) < n_stocks:
            for j in safe_idx:
                if j not in selected_idx:
                    selected_idx.append(j)
                    if len(selected_idx) >= n_stocks:
                        break
        
        sel_codes = [codes[j] for j in selected_idx]
        sel_scores = [scores[j] for j in selected_idx]
        
        # 指数衰减权重
        r = np.arange(1, len(sel_codes) + 1)
        weights = np.exp(-0.1 * r)
        weights = weights / weights.sum()
        
        # 波动率缩放
        sl = [sv.get(c, np.nan) for c in sel_codes]
        sl = [v for v in sl if not np.isnan(v) and v > 0.01]
        pr = max(min(target_vol / np.median(sl), 1.0)
                 if len(sl) >= 5 else 1.0, 0.05)
        
        # 买入
        if sel_codes and cash > 0.001:
            alloc = cash * pr * 0.98
            if alloc > 0.001:
                for code, wt in zip(sel_codes, weights):
                    px = px_buy.get(code, 0)
                    if px > 0 and wt > 0:
                        b = (alloc * wt) * (1 - COMM - SLIP) / px
                        if b > 0:
                            holdings[code] = b
                cash -= alloc
        
        # 净值
        np_ = sum(shares * px_sell.get(c, 0)
                  for c, shares in holdings.items())
        nt = np_ + cash
        ret = nt / tv_ - 1 if tv_ > 0 else 0
        navs.append(navs[-1] * (1 + ret))
    
    # 结果
    pnl = np.array(navs[1:]) / np.array(navs[:-1]) - 1
    na = np.array(navs)
    ny = len(pnl) / 13
    ar = na[-1] ** (1 / ny) - 1 if ny > 0 and na[-1] > 0 else 0
    sr = (np.mean(pnl) / np.std(pnl) * np.sqrt(13)) if np.std(pnl) > 0 else 0
    dd = np.maximum.accumulate(na) - na
    mdd = dd.max()
    wr = np.mean(pnl > 0)
    cal = ar / mdd if mdd > 0 else 0
    
    risk_note = ""
    if risk_enabled:
        risk_ratio = risk_log["removed"] / max(risk_log["total"], 1)
        risk_note = f" [风控剔除{risk_log['removed']}次]"
    
    print(f"  {label:32s}: 年化{ar*100:+7.1f}% | 夏普{sr:5.2f} | "
          f"回撤{mdd*100:5.1f}% | 胜率{wr*100:3.0f}% | 卡玛{cal:.2f} | "
          f"{len(pnl)}期{risk_note}", flush=True)
    
    return {
        "label": label,
        "navs": navs,
        "pnl": pnl.tolist(),
        "ar": ar,
        "sr": sr,
        "mdd": mdd,
        "wr": wr,
        "cal": cal,
        "n_periods": len(pnl),
        "risk_removed": risk_log["removed"],
        "risk_ratio": risk_log["removed"] / max(risk_log["total"], 1),
    }


# ====== 训练 ======
print("\n训练LightGBM (rank标签)...", flush=True)
pred_df = train_lgb_rank("v38 rank")
joblib.dump(pred_df, "output/pred_v38_rank.parquet")
print(f"  预测保存: output/pred_v38_rank.parquet", flush=True)

# ====== 回测 ======
print(f"\n{'='*60}", flush=True)
print("回测: T30 目波15% 2023-2026", flush=True)
print(f"{'='*60}", flush=True)

results = []

# 1. 基准：v36c 无风控
r = backtest(pred_df, 30, 0.15, "v36c 基准(无风控)", risk_enabled=False)
if r:
    results.append(r)

# 2. v27 原版风控（修复 < -5% | 高波 < -3% | 量价 > 0.03）
r = backtest(pred_df, 30, 0.15, "v38 v27原版风控",
             risk_config={
                 "repair_force_10d": -0.05,
                 "高波反转": -0.03,
                 "量价背离": 0.03,  # 原v27阈值
             })
if r:
    results.append(r)

# 3. 收紧版风控（量价背离收紧到 0.5）
r = backtest(pred_df, 30, 0.15, "v38 收紧风控(量价>0.5)",
             risk_config={
                 "repair_force_10d": -0.05,
                 "高波反转": -0.03,
                 "量价背离": 0.5,
             })
if r:
    results.append(r)

# 4. 仅修复力+高波反转（去掉量价背离）
r = backtest(pred_df, 30, 0.15, "v38 仅修复+高波(无量价)",
             risk_config={
                 "repair_force_10d": -0.05,
                 "高波反转": -0.03,
                 "量价背离": None,
             })
if r:
    results.append(r)

# 5. 严格版：修复<-5% | 高波<-2% | 量价>0.5
r = backtest(pred_df, 30, 0.15, "v38 严格版(高波<-2%)",
             risk_config={
                 "repair_force_10d": -0.05,
                 "高波反转": -0.02,  # 收紧到-2%
                 "量价背离": 0.5,
             })
if r:
    results.append(r)

# 输出汇总
print(f"\n{'='*60}", flush=True)
print("结果汇总：", flush=True)
print(f"{'配置':32s} {'年化':>7s} {'夏普':>6s} {'回撤':>6s} {'胜率':>4s} {'卡玛':>5s} {'期数':>5s}", flush=True)
print("-"*65, flush=True)
for r in results:
    print(f"{r['label']:32s} {r['ar']*100:>6.1f}% {r['sr']:>5.2f} "
          f"{r['mdd']*100:>5.1f}% {r['wr']*100:>3.0f}% {r['cal']:>4.2f} "
          f"{r['n_periods']:>5d}", flush=True)

# 保存结果
save = {k: v for k, v in enumerate(results)}
json.dump(save, open("output/backtest_v38_risk.json", "w"), indent=2, default=str)
print(f"\n✅ 完成 | ⏱ {(time.time()-tt)/60:.1f}分", flush=True)
print(f"  结果: output/backtest_v38_risk.json", flush=True)
