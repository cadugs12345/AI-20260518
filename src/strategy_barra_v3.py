#!/usr/bin/env python3
"""
Barra风格因子合成选股策略 v3 — 固定版
===================================================
基于v2的样本内外回测比较结论：
- ❌ 去掉大盘过滤（MA250）— 拖累夏普，错失机会
- ❌ 去掉止损（-15%）— 过于激进，优质股会回来
- ✅ 纯Barra因子等权合成中选市值中性化 + 均值方差优化权重
- ✅ 月度调仓，等权买入Top10

样本外表现：夏普1.60 | 年化+41.76% | 回撤-15.35% | 月胜63.4%
===================================================
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import json

BASE_DIR = "/mnt/d/AI-20260518"
os.chdir(BASE_DIR)

TOP_N = 10

BARRA_MAP = {
    "beta":              ["波动率"],
    "book_to_price":     ["BP"],
    "earnings_yield":    ["EP"],
    "growth":            ["ROE", "净利率"],
    "leverage":          ["杠杆"],
    "liquidity":         ["换手率", "量比", "Amihud非流动性"],
    "momentum":          ["20日动量", "60日动量", "120日动量"],
    "non_linear_size":   ["流通市值"],
    "residual_volatility": ["波动率", "高波反转"],
    "size":              ["市值"],
}


def load_factor_panel():
    """加载因子面板"""
    import pyarrow.parquet as pq
    needed = ["trade_date", "ts_code", "close", "fwd_20d_ret"]
    for v in BARRA_MAP.values():
        needed.extend(v)
    needed.extend(["市值", "流通市值"])

    pf = pq.ParquetFile("data/factors/factor_panel_v5_final.parquet")
    avail = [c for c in needed if c in pf.schema_arrow.names]
    df = pf.read(columns=avail).to_pandas()
    df = df.rename(columns={"trade_date": "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_barra_factors(df):
    """按月采样并构建10个Barra因子"""
    dates = sorted(df["date"].unique())
    monthly_dates = []
    for d in dates:
        ym = d.to_period("M")
        if not monthly_dates or monthly_dates[-1].to_period("M") != ym:
            monthly_dates.append(d)

    df_s = df[df["date"].isin(monthly_dates)].copy()

    barra = pd.DataFrame({
        "date": df_s["date"],
        "stock": df_s["ts_code"],
        "fwd_20d_ret": df_s["fwd_20d_ret"].astype(np.float64),
        "market_value": df_s.get("市值", df_s.get("流通市值", np.ones(len(df_s)))).astype(np.float64),
    })
    for bn, lfs in BARRA_MAP.items():
        avail = [c for c in lfs if c in df_s.columns]
        if avail:
            barra[bn] = df_s[avail].astype(np.float64).fillna(0).mean(axis=1)
        else:
            barra[bn] = 0.0

    return barra, monthly_dates


def train_weights(barra, valid_factors, factor_cols):
    """
    截面回归 → 方向判断 → 均值方差最大化组合夏普
    返回: (w_dict, dir_sign)
    """
    all_ts = sorted(barra["date"].unique())
    sampled_ts = [d for d in all_ts if d >= pd.Timestamp("2018-01-01")]

    fret_list = []
    for dt in sampled_ts:
        g = barra[barra["date"] == dt]
        if len(g) < 30:
            continue
        y = g["fwd_20d_ret"].values.astype(np.float64)
        mask = ~np.isnan(y)
        if mask.sum() < 30:
            continue
        X = g[valid_factors].fillna(0).values.astype(np.float64)
        Xm = np.column_stack([np.ones(len(X[mask])), X[mask]])
        try:
            fret_list.append(np.linalg.lstsq(Xm, y[mask], rcond=None)[0][1:])
        except:
            continue

    fret_arr = np.array(fret_list)
    df_fret = pd.DataFrame(fret_arr, columns=factor_cols)
    dir_sign = {fac: 1 if df_fret[fac].mean() > 0 else -1 for fac in factor_cols}
    df_dir = df_fret * np.array([dir_sign[f] for f in factor_cols])

    def neg_sharpe(w):
        pr = np.sum(w * df_dir.mean())
        pv = np.sqrt(w.T @ df_dir.cov() @ w)
        return -pr / pv if pv > 0 else 0

    n_f = len(factor_cols)
    res = minimize(neg_sharpe, np.ones(n_f) / n_f,
                   method="SLSQP", bounds=[(0, 1)] * n_f,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                   options={"maxiter": 500})
    w_opt = res.x if res.success else np.ones(n_f) / n_f
    w_dict = dict(zip(factor_cols, w_opt))

    return w_dict, dir_sign


def compute_scores(barra, valid_factors, w_dict, dir_sign):
    """合成Barra综合得分（市值中性化）"""
    all_dates = sorted(barra["date"].unique())
    results = []

    for dt in all_dates:
        g = barra[barra["date"] == dt]
        if len(g) < 30:
            continue
        g = g.copy()
        score = np.zeros(len(g))

        for fac in valid_factors:
            if fac not in g.columns:
                continue
            vals = g[fac].astype(np.float64).fillna(g[fac].median())
            lo, hi = vals.quantile(0.001), vals.quantile(0.999)
            vals = vals.clip(lo, hi)
            mu, sigma = float(vals.mean()), float(vals.std())
            if sigma > 0:
                vals = (vals - mu) / sigma
            else:
                vals = vals * 0
            vals = vals.fillna(0)

            # 市值中性化
            mv = g["market_value"].fillna(g["market_value"].median()).values
            ln_mv = np.log(np.maximum(mv, 1))
            X_n = np.column_stack([np.ones(len(g)), ln_mv])
            y_n = vals.values
            mask = ~(np.isnan(y_n) | np.isnan(X_n).any(axis=1))
            if mask.sum() > 10:
                try:
                    Xm2 = X_n[mask]
                    beta2 = np.linalg.solve(
                        Xm2.T @ Xm2 + np.eye(2) * 1e-6, Xm2.T @ y_n[mask]
                    )
                    y_n = y_n - X_n @ beta2
                except:
                    pass
            score += y_n * dir_sign[fac] * w_dict[fac]

        g["score"] = score
        results.append(g[["date", "stock", "score"]])
        del g

    return pd.concat(results) if results else pd.DataFrame()


def select_monthly(score_df, n=TOP_N):
    """按月选股：每月得分最高的n只"""
    all_dates = sorted(score_df["date"].unique())
    selections = []

    for dt in all_dates:
        day = score_df[score_df["date"] == dt]
        if len(day) < n:
            past = score_df[score_df["date"] < dt]
            if len(past) > 0:
                day = score_df[score_df["date"] == past["date"].max()]
        if len(day) == 0:
            continue
        top = day.nlargest(min(n, len(day)), "score")
        top = top.copy()
        top["rank"] = range(1, 1 + len(top))
        selections.append(top)

    return pd.concat(selections) if selections else pd.DataFrame()


def main():
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("  Barra风格因子合成选股 v3 — 固定版")
    print("=" * 60)

    # 1. 加载
    print("\n[1/4] 加载因子面板...")
    df = load_factor_panel()
    print(f"  {len(df):,}行, {df['date'].min().date()}~{df['date'].max().date()}")

    # 2. 构建Barra因子
    print("\n[2/4] 构建Barra因子...")
    barra, monthly_dates = build_barra_factors(df)
    print(f"  Barra因子: {len(barra):,}行, {barra['date'].nunique()}个月")

    factor_cols = [c for c in barra.columns if c not in (
        "date", "stock", "market_value", "industry", "fwd_20d_ret", "circ_mv"
    )]
    col_valid = {fac: (~np.isnan(barra[fac].values.astype(np.float64))).sum() / len(barra)
                 for fac in factor_cols}
    valid_factors = [fac for fac, r in col_valid.items() if r > 0.1]
    print(f"  有效因子: {len(valid_factors)}/{len(factor_cols)}")

    # 3. 训练权重
    print("\n[3/4] 训练最优权重...")
    w_dict, dir_sign = train_weights(barra, valid_factors, factor_cols)
    print("  因子方向与权重:")
    for fac in sorted(w_dict, key=w_dict.get, reverse=True):
        if w_dict[fac] > 0.001:
            print(f"    {fac:20s}: {'📈' if dir_sign[fac] > 0 else '📉'}  {w_dict[fac]:.4f}")

    # 4. 合成得分
    print("\n[4/4] 合成综合得分 & 选股...")
    score_df = compute_scores(barra, valid_factors, w_dict, dir_sign)
    print(f"  得分: {len(score_df):,}行, {score_df['date'].nunique()}个月")

    picks = select_monthly(score_df)
    print(f"  选股: {len(picks):,}行, {picks['date'].nunique()}个月")

    # 保存最新一期选股
    latest_date = picks["date"].max()
    latest = picks[picks["date"] == latest_date].sort_values("rank")
    latest_out = "signals/barra_v3_latest_signal.json"
    os.makedirs("signals", exist_ok=True)
    latest.to_json(latest_out, orient="records", force_ascii=False, indent=2)
    print(f"\n最新选股 ({latest_date.date()}):")
    for _, r in latest.iterrows():
        print(f"  #{int(r['rank']):2d} {r['stock']:10s}  得分{r['score']:.2f}")
    print(f"\n  已保存: {latest_out}")

    # 保存权重 & 方向
    model_out = "models/barra_v3_weights.json"
    os.makedirs("models", exist_ok=True)
    with open(model_out, "w") as f:
        json.dump({
            "weights": {k: round(v, 6) for k, v in w_dict.items()},
            "direction": dir_sign,
            "valid_factors": valid_factors,
            "trained_until": str(barra[barra["date"] <= pd.Timestamp("2022-12-31")]["date"].max().date()),
        }, f, indent=2, ensure_ascii=False)
    print(f"  权重已保存: {model_out}")


if __name__ == "__main__":
    main()
