"""
明日验收脚本 (2026-05-20)
1. 机构行为因子IC测试 -> 合成测试
2. 断板修复因子事件分解验证
3. v23回测 (核心15+turnover_persistence+hilo_signal)
4. ML重训 (面板81因子XGBoost)
"""
import sys, os, time, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

PROJECT = "/mnt/d/AI-20260518"
os.chdir(PROJECT)
sys.path.insert(0, '.')
from config.settings import TS_TOKEN

import tushare as ts
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

t0 = time.time()
OUT = "output/20260520"
os.makedirs(OUT, exist_ok=True)

print("=" * 60)
print("明日验收脚本 — 2026-05-20")
print("=" * 60)

# ============================================================
# TASK 1: 机构行为因子全量验证
# ============================================================
print("\n[Task 1] 机构行为因子 ... ", end="", flush=True)
try:
    panel = pd.read_parquet("data/factors/factor_panel_v6.parquet",
                            columns=["ts_code", "trade_date", "fwd_20d_ret"])

    inst_results = {}

    # 股东户数
    if os.path.exists("data/new_factors/holder_all.parquet"):
        h = pd.read_parquet("data/new_factors/holder_all.parquet")
        h["end_date"] = pd.to_datetime(h["end_date"])
        h = h.sort_values(["ts_code", "end_date"])
        h["holder_change"] = h.groupby("ts_code")["holder_num"].pct_change()
        h["holder_qoq"] = h.groupby("ts_code")["holder_num"].pct_change(3)
        h["trade_date"] = (h["end_date"] + pd.Timedelta(days=45)).apply(lambda x: x.replace(day=1))
        h["trade_date"] = pd.to_datetime(h["trade_date"])

        for fact in ["holder_change", "holder_qoq"]:
            merged = h.merge(panel, on=["ts_code", "trade_date"], how="inner").dropna(subset=[fact, "fwd_20d_ret"])
            ics = []
            for ym, g in merged.groupby(merged["trade_date"].dt.to_period("M")):
                gv = g[[fact, "fwd_20d_ret"]].dropna()
                if len(gv) < 30: continue
                r, _ = spearmanr(gv[fact], gv["fwd_20d_ret"])
                if not np.isnan(r): ics.append(r)
            if ics:
                ic_m = float(np.mean(ics))
                ic_s = float(np.std(ics))
                inst_results[f"holder_{fact}"] = {"ic": ic_m, "ir": ic_m / ic_s if ic_s > 0 else 0, "n": len(ics)}
        print(f"股东户数OK", end=" ", flush=True)

    # 十大股东
    if os.path.exists("data/new_factors/top10_all.parquet"):
        t10 = pd.read_parquet("data/new_factors/top10_all.parquet")
        t10["end_date"] = pd.to_datetime(t10["end_date"])
        inst_types = ["金融机构", "保险公司", "基金", "证券", "投资"]
        t10["is_inst"] = t10["holder_type"].apply(lambda x: any(it in str(x) for it in inst_types))
        
        # 用更高效的聚合
        t10 = t10.sort_values(["ts_code", "end_date"])
        t10["_inst_ratio"] = t10.groupby(["ts_code", "end_date"])["hold_ratio"].transform(
            lambda x: x[t10.loc[x.index, "is_inst"].values].sum() if x.index.isin(t10.index[t10["is_inst"]]).any() else 0
        )
        # 简化聚合
        isum = t10.groupby(["ts_code", "end_date"]).agg(
            inst_hold_ratio=("hold_ratio", lambda x: x[t10.loc[x.index, "is_inst"]].sum()),
            inst_change=("hold_change", lambda x: x[t10.loc[x.index, "is_inst"]].sum()),
        ).reset_index()
        isum["inst_ratio_change"] = isum.groupby("ts_code")["inst_hold_ratio"].diff()
        isum["trade_date"] = (isum["end_date"] + pd.Timedelta(days=45)).apply(lambda x: x.replace(day=1))
        isum["trade_date"] = pd.to_datetime(isum["trade_date"])

        for fact in ["inst_ratio_change", "inst_change"]:
            merged = isum.merge(panel, on=["ts_code", "trade_date"], how="inner").dropna(subset=[fact, "fwd_20d_ret"])
            ics = []
            for ym, g in merged.groupby(merged["trade_date"].dt.to_period("M")):
                gv = g[[fact, "fwd_20d_ret"]].dropna()
                if len(gv) < 30: continue
                r, _ = spearmanr(gv[fact], gv["fwd_20d_ret"])
                if not np.isnan(r): ics.append(r)
            if ics:
                ic_m = float(np.mean(ics))
                ic_s = float(np.std(ics))
                inst_results[f"top10_{fact}"] = {"ic": ic_m, "ir": ic_m / ic_s if ic_s > 0 else 0, "n": len(ics)}
        print(f"十大股东OK", flush=True)

    # 输出
    print(f"\n  机构行为因子 IC结果:")
    for k, v in sorted(inst_results.items(), key=lambda x: -abs(x[1]["ir"])):
        flag = "🟢" if abs(v["ir"]) > 0.5 else ("🟡" if abs(v["ir"]) > 0.3 else "🔴")
        print(f"    {k:35s} IC={v['ic']*100:+7.2f}% IR={v['ir']:7.2f} ({v['n']}个月) {flag}")

    # 综合因子构建 & 合成测试
    if inst_results and any(abs(v["ir"]) > 0.3 for v in inst_results.values()):
        best_key = max(inst_results, key=lambda k: abs(inst_results[k]["ir"]))
        print(f"\n  最优因子 [{best_key}] IR={inst_results[best_key]['ir']:.2f} → 准备合成测试")

    json.dump(inst_results, open(f"{OUT}/inst_factors.json", "w"), indent=2, default=str)
    
except Exception as e:
    print(f"❌ {e}", flush=True)

# ============================================================
# TASK 2: 断板修复因子事件分解
# ============================================================
print(f"\n[Task 2] 断板修复事件分解 ... ", end="", flush=True)
try:
    panel2 = pd.read_parquet("data/factors/factor_panel_v6.parquet",
                             columns=["ts_code", "trade_date", "close", "fwd_20d_ret",
                                      "board_break", "repair_force_5d", "repair_force_10d",
                                      "board_repair_score"])
    # 炸板后收益分析
    breakdown = panel2[panel2["board_break"] == True].copy()
    breakdown = breakdown.sort_values(["ts_code", "trade_date"])
    
    print(f"炸板事件: {len(breakdown):,}条", end=" ", flush=True)
    
    # 炸板后不同区间的平均收益
    result = {
        "炸板样本数": len(breakdown),
        "炸板后20日平均收益": float(breakdown["fwd_20d_ret"].mean()),
        "炸板后20日中位数": float(breakdown["fwd_20d_ret"].median()),
        "炸板后收益>0比例": float((breakdown["fwd_20d_ret"] > 0).mean()),
        "修复因子均值": float(breakdown["repair_force_10d"].mean()),
    }

    # 分层: repair_force_10d高低分组
    for q, label in [(0.2, "低修复"), (0.8, "高修复")]:
        cutoff = breakdown["repair_force_10d"].quantile(q)
        subset = breakdown[breakdown["repair_force_10d"] >= cutoff] if q > 0.5 else breakdown[breakdown["repair_force_10d"] <= cutoff]
        result[f"{label}组收益均值"] = float(subset["fwd_20d_ret"].mean())
        result[f"{label}组数量"] = len(subset)

    json.dump(result, open(f"{OUT}/board_break_analysis.json", "w"), indent=2, default=str)
    for k, v in result.items():
        if "收益" in k:
            print(f"\n    {k}: {v*100:.2f}%", end="", flush=True)
    print()
except Exception as e:
    print(f"❌ {e}", flush=True)

# ============================================================
# TASK 3: v23回测
# ============================================================
print(f"\n[Task 3] v23回测(核心15+2增量因子) ... ", end="", flush=True)
try:
    col_list = ["ts_code", "trade_date", "fwd_20d_ret",
                "短期反转", "20日动量", "60日动量", "120日动量", "波动率",
                "换手率", "量比", "量能趋势", "BP", "EP", "MACD", "BOLL位置",
                "EMA5偏离", "EMA10偏离", "EMA20偏离",
                "turnover_persistence", "hilo_signal"]
    panel3 = pd.read_parquet("data/factors/factor_panel_v6.parquet", columns=col_list)
    panel3["_ym"] = panel3["trade_date"].dt.to_period("M")

    def syn_ir(df, factors):
        valid = [f for f in factors if f in df.columns]
        if len(valid) < 2: return 0, 0, []
        zdf = df[valid].rank(pct=True)
        df["__zs"] = zdf.mean(axis=1)
        ics = []
        for ym, g in df.groupby("_ym"):
            gv = g[["__zs", "fwd_20d_ret"]].dropna()
            if len(gv) < 50: continue
            r, _ = spearmanr(gv["__zs"], gv["fwd_20d_ret"])
            if not np.isnan(r): ics.append(r)
        if not ics: return 0, 0, []
        ic_m = float(np.mean(ics))
        return ic_m, ic_m / float(np.std(ics)) if float(np.std(ics)) > 0 else 0, ics

    core = ["短期反转", "20日动量", "60日动量", "120日动量", "波动率",
            "换手率", "量比", "量能趋势", "BP", "EP", "MACD", "BOLL位置",
            "EMA5偏离", "EMA10偏离", "EMA20偏离"]
    v23 = core + ["turnover_persistence", "hilo_signal"]

    ic_base, ir_base, _ = syn_ir(panel3, core)
    ic_v23, ir_v23, ics_v23 = syn_ir(panel3, v23)

    v23_result = {
        "v12_ir": ir_base, "v12_ic": ic_base,
        "v23_ir": ir_v23, "v23_ic": ic_v23,
        "增量": ir_v23 - ir_base,
        "v23_ic月数": len(ics_v23),
        "v23_ic正比例": float(np.mean([1 for x in ics_v23 if x > 0])) if ics_v23 else 0,
    }
    json.dump(v23_result, open(f"{OUT}/v23_result.json", "w"), indent=2)
    print(f"v12 IR={ir_base:.2f} → v23 IR={ir_v23:.2f} (Δ={ir_v23-ir_base:+.3f})", flush=True)

except Exception as e:
    print(f"❌ {e}", flush=True)

# ============================================================
# TASK 4: ML重训
# ============================================================
print(f"\n[Task 4] ML重训(XGBoost, 81因子) ... ", end="", flush=True)
try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import xgboost as xgb

    panel4 = pd.read_parquet("data/factors/factor_panel_v6.parquet")
    panel4 = panel4[panel4["trade_date"] >= "2023-01-01"]
    
    factor_cols = [c for c in panel4.columns if c not in ["ts_code", "trade_date", "fwd_20d_ret",
        "close", "ret_1d", "volume", "__fragment_index", "__batch_index",
        "__last_in_fragment", "__filename", "board_break"]]
    
    # 按时间分割
    split_date = pd.Timestamp("2025-01-01")
    train = panel4[panel4["trade_date"] < split_date].dropna(subset=factor_cols + ["fwd_20d_ret"]).sample(min(500000, len(panel4)))
    test = panel4[panel4["trade_date"] >= split_date].dropna(subset=factor_cols + ["fwd_20d_ret"]).sample(min(100000, len(panel4)))
    
    X_train = train[factor_cols].fillna(0)
    y_train = np.where(train["fwd_20d_ret"].values > 0, 1, 0)
    X_test = test[factor_cols].fillna(0)
    y_test = np.where(test["fwd_20d_ret"].values > 0, 1, 0)

    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.5,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 月频IC
    test_df = test.copy()
    test_df["ml_score"] = model.predict_proba(X_test)[:, 1]
    test_df["_ym"] = test_df["trade_date"].dt.to_period("M")
    
    ics_ml = []
    for ym, g in test_df.dropna(subset=["ml_score", "fwd_20d_ret"]).groupby("_ym"):
        gv = g[["ml_score", "fwd_20d_ret"]].dropna()
        if len(gv) < 50: continue
        r, _ = spearmanr(gv["ml_score"], gv["fwd_20d_ret"])
        if not np.isnan(r): ics_ml.append(r)

    # 特征重要性
    importance = pd.DataFrame({
        "factor": factor_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).head(20)

    ml_result = {
        "ml_ic_mean": float(np.mean(ics_ml)) if ics_ml else 0,
        "ml_ic_ir": float(np.mean(ics_ml)) / float(np.std(ics_ml)) if ics_ml and float(np.std(ics_ml)) > 0 else 0,
        "ml_ic_n": len(ics_ml),
        "训练精度": float(model.score(X_train, y_train)),
        "测试精度": float(model.score(X_test, y_test)),
        "top_factor": importance.head(10).to_dict(orient="records")
    }
    json.dump(ml_result, open(f"{OUT}/ml_result.json", "w"), indent=2)
    
    print(f"ML IR={ml_result['ml_ic_ir']:.2f} (vs v12 IR={ir_base:.2f})", flush=True)
    print(f"  Top因子:", flush=True)
    for _, r in importance.head(5).iterrows():
        print(f"    {r['factor']:30s} {r['importance']:.4f}", flush=True)

except ImportError:
    print("sklearn/xgboost未安装", flush=True)
except Exception as e:
    print(f"❌ {e}", flush=True)

# ============================================================
print(f"\n{'='*60}")
print(f"✅ 全部完成! 结果保存在 {OUT}/")
print(f"⏱ {time.time()-t0:.0f}s")
print(f"{'='*60}")
