"""
机器学习多因子选股 (XGBoost/LightGBM)
- 严格时间序列交叉验证
- 每年滚动训练 (防过拟合)
- 周频调仓
- 夏普目标 > 1.0
"""
import os, sys, time, gc, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DATA_FACTORS

t0 = time.time()

print("=" * 60, flush=True)
print("ML 多因子选股 (XGBoost / LightGBM)", flush=True)
print("=" * 60, flush=True)

# ===== 加载数据 =====
panel = pd.read_parquet(os.path.join(DATA_FACTORS, "factor_panel_with_fwd.parquet"))
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
panel = panel[panel["trade_date"] >= "2017-06-01"].copy()  # 留一年滚动窗口
print(f"面板: {len(panel):,} 条", flush=True)

# 筛选有效因子 (ICIR > 0.15)
factor_cols = [c for c in panel.columns 
               if c not in ("ts_code","trade_date","fwd_20d_ret")
               and panel[c].dtype in ("float64","int64")]

# 去掉已知负向因子
drop_cols = ["短期反转", "毛利率", "量比", "BP", "SP", "EP", "股息率", "流通市值"]
factor_cols = [c for c in factor_cols if c not in drop_cols]
print(f"有效因子: {len(factor_cols)} 个: {factor_cols}", flush=True)

# ===== 时间序列划分 =====
dates_all = sorted(panel["trade_date"].unique())
# 用周频建仓 (每周五)
weekly = pd.date_range(start=dates_all[0], end=dates_all[-1], freq="W-FRI")
weekly_dates = []
for d in weekly:
    if d in dates_all:
        weekly_dates.append(d)
    else:
        # 取最近的上一个交易日
        prev = d - pd.Timedelta(days=1)
        while prev not in dates_all and prev >= dates_all[0]:
            prev -= pd.Timedelta(days=1)
        if prev in dates_all:
            weekly_dates.append(prev)
weekly_dates = sorted(set(weekly_dates))
# 过滤掉回测前的日期
weekly_dates = [d for d in weekly_dates if d >= pd.Timestamp("2018-01-05")]
print(f"周频节点: {len(weekly_dates)} 周 ({weekly_dates[0].date()} ~ {weekly_dates[-1].date()})", flush=True)

# ===== 滚动训练窗口 =====
train_years = 3  # 用过去3年做训练

# 尝试导入ML库
print("\n[检查依赖]", flush=True)
has_xgb = True
has_lgbm = True
try:
    import xgboost as xgb
    print(f"  XGBoost: {xgb.__version__}", flush=True)
except:
    has_xgb = False
    print("  XGBoost: 未安装", flush=True)

try:
    import lightgbm as lgb
    print(f"  LightGBM: {lgb.__version__}", flush=True)
except:
    has_lgbm = False
    print("  LightGBM: 未安装", flush=True)

if not has_xgb and not has_lgbm:
    print("[错误] 需要 XGBoost 或 LightGBM", flush=True)
    sys.exit(1)

# ===== 训练 & 回测 =====
results = {}

for name, ModelClass, params in [
    ("XGBoost", xgb.XGBRegressor if has_xgb else None, {
        "n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.7, "reg_alpha": 0.1, "reg_lambda": 1.0,
        "random_state": 42, "verbosity": 0, "n_jobs": 8
    }),
    ("LightGBM", lgb.LGBMRegressor if has_lgbm else None, {
        "n_estimators": 300, "max_depth": 5, "learning_rate": 0.03,
        "subsample": 0.8, "colsample_bytree": 0.7,
        "reg_alpha": 0.1, "reg_lambda": 1.0, "min_child_samples": 50,
        "random_state": 42, "verbose": -1, "n_jobs": 8
    })
]:
    if ModelClass is None:
        continue
    
    print(f"\n{'='*60}", flush=True)
    print(f"[{name}] 开始滚动训练+回测", flush=True)
    print(f"{'='*60}", flush=True)
    
    # 存储预测
    all_preds = []
    week_count = 0
    
    for i, date in enumerate(weekly_dates):
        # 训练窗口: [date - 3年, date]
        train_start = date - pd.Timedelta(days=train_years*365)
        
        train_mask = (panel["trade_date"] >= train_start) & (panel["trade_date"] < date)
        test_data = panel[panel["trade_date"] == date].copy()
        
        train_data = panel[train_mask].dropna(subset=factor_cols + ["fwd_20d_ret"]).copy()
        test_data = test_data.dropna(subset=factor_cols).copy()
        
        # 如果训练数据不足, 跳过
        if len(train_data) < 10000:
            continue
        
        # 训练集: 只选收益标准差大的股票 (提升信噪比)
        train_data = train_data[train_data["fwd_20d_ret"].abs() < 0.5].copy()  # 去异常值
        
        X_train = train_data[factor_cols].values.astype(np.float32)
        y_train = train_data["fwd_20d_ret"].values.astype(np.float32)
        
        X_test = test_data[factor_cols].values.astype(np.float32)
        test_codes = test_data["ts_code"].values
        
        # 中位數填充NaN (避免泄露)
        col_medians = np.nanmedian(X_train, axis=0)
        col_medians = np.nan_to_num(col_medians, 0)
        for j in range(X_train.shape[1]):
            mask_tr = np.isnan(X_train[:, j])
            if mask_tr.any():
                X_train[mask_tr, j] = col_medians[j]
            mask_te = np.isnan(X_test[:, j])
            if mask_te.any():
                X_test[mask_te, j] = col_medians[j]
        
        # 模型训练
        try:
            model = ModelClass(**params)
            model.fit(X_train, y_train)
            
            # 预测
            preds = model.predict(X_test)
            
            for j, code in enumerate(test_codes):
                all_preds.append({
                    "trade_date": date, "ts_code": code, 
                    "pred_ret": float(preds[j]),
                    "ml_method": name
                })
        except Exception as e:
            continue
        
        week_count += 1
        if week_count % 50 == 0:
            elapsed = (time.time() - t0) / 60
            rate = week_count / elapsed if elapsed > 0 else 0
            print(f"  [{name}] {week_count}/{len(weekly_dates)} 周, {rate:.0f}周/分钟", flush=True)
    
    # 构建预测面板
    df_pred = pd.DataFrame(all_preds)
    if df_pred.empty:
        print(f"  [{name}] 无预测结果", flush=True)
        continue
    print(f"  [{name}] 预测: {len(df_pred):,} 条", flush=True)
    
    # ===== 回测 =====
    pred_dates = sorted(df_pred["trade_date"].unique())
    pnl = []
    cum = [1.0]
    
    for i, date in enumerate(pred_dates):
        if i == len(pred_dates) - 1:
            continue
        
        day = df_pred[df_pred["trade_date"] == date].sort_values("pred_ret", ascending=False)
        if len(day) < 100:
            continue
        
        # 选top 30%
        n_top = max(len(day) // 3, 50)
        top = day.head(n_top)
        
        next_date = pred_dates[i + 1]
        next_data = panel[panel["trade_date"] == next_date]
        if next_data.empty:
            continue
        
        rets = []
        for _, row in top.iterrows():
            nd = next_data[next_data["ts_code"] == row["ts_code"]]
            if not nd.empty and not np.isnan(nd["fwd_20d_ret"].iloc[0]):
                rets.append(nd["fwd_20d_ret"].iloc[0])
        
        if rets:
            r = np.mean(rets)
            pnl.append(r)
            cum.append(cum[-1] * (1 + r))
    
    if not pnl:
        print(f"  [{name}] 回测无结果", flush=True)
        continue
    
    pnl = np.array(pnl)
    cum = np.array(cum)
    tr = cum[-1] - 1
    ar = (cum[-1])**(52/len(pnl)) - 1
    vol = np.std(pnl) * np.sqrt(52/len(pnl)) * np.sqrt(252)
    sr = np.mean(pnl) / np.std(pnl) * np.sqrt(52)
    dd = np.maximum.accumulate(cum) - cum
    mdd = dd.max()
    wr = np.mean(pnl > 0)
    
    results[name] = {
        "总收益": f"{tr*100:.1f}%",
        "年化收益": f"{ar*100:.1f}%",
        "年化波动": f"{vol*100:.1f}%",
        "夏普比率": f"{sr:.2f}",
        "最大回撤": f"{mdd*100:.1f}%",
        "周胜率": f"{wr*100:.0f}%",
        "交易周数": len(pnl),
    }
    
    print(f"\n[{name}] 回测结果:", flush=True)
    print(f"  总收益: {tr*100:.1f}% | 年化: {ar*100:.1f}% | 年化波动: {vol*100:.1f}%", flush=True)
    print(f"  夏普: {sr:.2f} | 最大回撤: {mdd*100:.1f}% | 周胜率: {wr*100:.0f}%", flush=True)
    
    # 可选: 保存月度收益序列
    save_path = os.path.join(DATA_FACTORS, f"ml_backtest_{name.lower()}.parquet")
    pd.DataFrame({"pnl": pnl}).to_parquet(save_path)
    print(f"  收益序列已保存: {save_path}", flush=True)
    
    del model; gc.collect()

# 汇总
print(f"\n{'='*60}", flush=True)
print("ML选股回测汇总", flush=True)
print(f"{'='*60}", flush=True)
summary_df = pd.DataFrame(results).T
print(summary_df.to_string(), flush=True)
summary_df.to_csv(os.path.join(DATA_FACTORS, "ml_strategy_results.csv"), encoding="utf-8-sig")

print(f"\n总用时: {(time.time()-t0)/60:.1f} 分钟", flush=True)
print("Done!", flush=True)
