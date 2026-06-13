"""
v10 - 20日调仓 + 目标波动 + 行业中性化 + 止损
"""
import os, sys, time, gc, pickle
import numpy as np
import pandas as pd
import tushare as ts

warnings = __import__('warnings')
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_FACTORS = "data/factors"

t0 = time.time()
print("="*60)
print("v10 - 20日调仓 + 行业中性化 + 止损 + 目标波动")
print("="*60)

# ===== 加载数据 =====
panel = pd.read_parquet(os.path.join(DATA_FACTORS, "factor_panel_with_fwd_v2.parquet"))
prices = pd.read_parquet(os.path.join(DATA_FACTORS, "full_prices.parquet"))
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
prices["trade_date"] = pd.to_datetime(prices["trade_date"])

# ===== 行业分类 =====
print("加载行业分类...")
pro = ts.pro_api('3e8953587c4c717c26e5cb99d028a66e044d184f2d464cab0950000e')
ind_df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,industry')
ind_map = dict(zip(ind_df['ts_code'], ind_df['industry']))
panel['行业'] = panel['ts_code'].map(ind_map)
panel['行业'] = panel['行业'].fillna('其他')

# 合并小行业（每期平均<10只的合并到"其他"）
ind_counts = panel.groupby(['trade_date', '行业']).size().reset_index(name='cnt')
avg_cnt = ind_counts.groupby('行业')['cnt'].mean()
small_inds = set(avg_cnt[avg_cnt < 10].index)
panel['行业_大类'] = panel['行业'].apply(lambda x: '其他' if x in small_inds else x)
n_inds = panel['行业_大类'].nunique()
print(f"  行业: {panel['行业'].nunique()} -> 合并后 {n_inds} 个大类")

# ===== 因子列 =====
factor_cols = [c for c in panel.columns 
               if c not in ("ts_code","trade_date","fwd_20d_ret","行业","行业_大类")
               and panel[c].dtype in ("float64","int64")]
factor_cols = [c for c in factor_cols if c not in ("短期反转","毛利率","量比","BP","SP","EP","股息率","流通市值")]

# ===== 20日节点 =====
all_dates = sorted(panel["trade_date"].unique())
period_dates = [all_dates[i] for i in range(0, len(all_dates), 20) 
                if all_dates[i] >= pd.Timestamp("2021-01-01")]

price_map = {}
for d in period_dates:
    sub = prices[prices["trade_date"] == d][["ts_code","close"]].dropna()
    price_map[d] = dict(zip(sub["ts_code"], sub["close"]))

# ===== ML预测缓存 =====
pred_path = os.path.join(DATA_FACTORS, "pred_20d_v10.pkl")
if os.path.exists(pred_path):
    with open(pred_path, 'rb') as f:
        pred = pickle.load(f)
    print(f"ML缓存: {len(pred):,}条, {pred['trade_date'].nunique()}期")
else:
    print("[ML] 新训练...")
    import xgboost as xgb, lightgbm as lgb
    all_preds = []
    for i, date in enumerate(period_dates):
        train_start = date - pd.Timedelta(days=3*365)
        val_start = date - pd.Timedelta(days=180)
        
        train = panel[(panel["trade_date"] >= train_start) & (panel["trade_date"] < val_start)]
        val = panel[(panel["trade_date"] >= val_start) & (panel["trade_date"] < date)]
        train = train.dropna(subset=factor_cols + ["fwd_20d_ret"])
        val = val.dropna(subset=factor_cols + ["fwd_20d_ret"])
        train = train[train["fwd_20d_ret"].abs() < 0.5]
        val = val[val["fwd_20d_ret"].abs() < 0.5]
        if len(train) < 10000 or len(val) < 2000: continue
        
        X_tr = np.nan_to_num(train[factor_cols].values.astype(np.float32), nan=0)
        y_tr = train["fwd_20d_ret"].values.astype(np.float32)
        X_va = np.nan_to_num(val[factor_cols].values.astype(np.float32), nan=0)
        y_va = val["fwd_20d_ret"].values.astype(np.float32)
        
        xgb_m = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbosity=0, n_jobs=8, early_stopping_rounds=30)
        xgb_m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        lgb_m = lgb.LGBMRegressor(n_estimators=500, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbose=-1, n_jobs=8)
        lgb_m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(30)], eval_metric="mse")
        
        day = panel[panel["trade_date"] == date]
        X_te = np.nan_to_num(day[factor_cols].values.astype(np.float32), nan=0)
        p = (xgb_m.predict(X_te) + lgb_m.predict(X_te)) / 2
        
        for j, code in enumerate(day["ts_code"].values):
            all_preds.append({"trade_date": date, "ts_code": code, "pred_ret": float(p[j])})
        if (i+1) % 30 == 0: print(f"  [{i+1}/{len(period_dates)}] train={len(train):,}")
    
    pred = pd.DataFrame(all_preds)
    with open(pred_path, 'wb') as f:
        pickle.dump(pred, f)
    print(f"ML完成: {len(pred):,}条, {pred['trade_date'].nunique()}期")

pred["trade_date"] = pd.to_datetime(pred["trade_date"])
pred_dates = sorted(pred["trade_date"].unique())

# ===== 行业中性化处理 =====
# 对每个日期，将预测值按行业做组内标准化（Z-score），再行业间均衡
print("做行业中性化...")
ind_neutralized = []
for d in pred_dates:
    day_pred = pred[pred["trade_date"] == d].copy()
    day_ind = panel[panel["trade_date"] == d][["ts_code","行业_大类"]].drop_duplicates()
    day_pred = day_pred.merge(day_ind, on="ts_code", how="left")
    day_pred["行业_大类"] = day_pred["行业_大类"].fillna("其他")
    
    # 行业内计算rank z-score
    def rank_zscore(s):
        r = s.rank(pct=True)
        return (r - r.mean()) / r.std()
    
    day_pred["pred_neutral"] = day_pred.groupby("行业_大类")["pred_ret"].transform(rank_zscore)
    ind_neutralized.append(day_pred)

pred_all = pd.concat(ind_neutralized, ignore_index=True)
print(f"中性化完成: {len(pred_all):,}条")

# ===== 无成本验证 =====
print(f"\n无成本验证:")
for use_neutral in [False, True]:
    label = "ML原始" if not use_neutral else "ML+行业中性"
    pcol = "pred_ret" if not use_neutral else "pred_neutral"
    for n in [30, 50]:
        rets = []
        for d in pred_dates:
            day = pred_all[pred_all["trade_date"] == d].sort_values(pcol, ascending=False)
            top = set(day.head(n)["ts_code"].values)
            actual = panel[panel["trade_date"] == d]
            rr = actual[actual["ts_code"].isin(top)]["fwd_20d_ret"].mean()
            if not np.isnan(rr): rets.append(rr)
        if rets:
            pnl = np.array(rets)
            sr = np.mean(pnl)/np.std(pnl)*np.sqrt(13)
            print(f"  {label:20s} Top{n:3d}: 均值{np.mean(pnl)*100:+.2f}% 夏普{sr:.2f} {len(rets)}期")

# ===== 含成本回测 =====
print(f"\n{'='*60}")
print("含成本 + 波动率控制 + 止损 + 行业中性")
print(f"{'='*60}")

STAMP, COMM, SLIP = 0.001, 0.0002, 0.002
STOP_LOSS = 0.08  # 单期跌超8%后清仓观望1期

def backtest_v10(n_stocks, target_vol=0.20, use_neutral=True, stop_loss=STOP_LOSS, label=""):
    pcol = "pred_ret" if not use_neutral else "pred_neutral"
    cash = 0.03
    holdings = {}
    navs = [1.0]
    hist_rets = []
    in_cooldown = 0  # 止损冷却期计数
    
    for i, date in enumerate(pred_dates[:-1]):
        sell_date = pred_dates[i+1]
        px_buy = price_map.get(date, {})
        px_sell = price_map.get(sell_date, {})
        
        hold_val = sum(shares * px_buy.get(c, 0) for c, shares in holdings.items())
        total_val = hold_val + cash
        
        # ---- 卖出 ----
        sell_proceeds = 0
        for code, shares in holdings.items():
            px = px_sell.get(code, 0)
            if px > 0:
                val = shares * px
                cost = val * (STAMP + COMM + SLIP)
                sell_proceeds += val - cost
        cash = cash + sell_proceeds
        holdings = {}
        
        # ---- 止损冷却期 ----
        if in_cooldown > 0:
            in_cooldown -= 1
            # 冷却期不买入，只持有现金
            new_total = cash
            ret = new_total / total_val - 1 if total_val > 0 else 0
            navs.append(navs[-1] * (1 + ret))
            hist_rets.append(ret)
            if i < 3 or (i+1) % 10 == 0:
                print(f"  p{i:3d} {str(date.date())}->{str(sell_date.date())} | 冷却{in_cooldown} | "
                      f"总{total_val:.3f}->{new_total:.3f} ret={ret*100:+6.2f}% nav={navs[-1]:.3f}")
            continue
        
        # ---- 波动率仓位 ----
        position_ratio = 1.0
        if len(hist_rets) >= 3:
            # EWMA波动率估计（lambda=0.7）
            ewma_vol = 0
            w_sum = 0
            for j, r in enumerate(reversed(hist_rets[-10:])):
                w = 0.7 ** j
                ewma_vol += w * r**2
                w_sum += w
            ewma_vol = np.sqrt(ewma_vol / w_sum) * np.sqrt(13)
            
            if ewma_vol > 0.01:
                position_ratio = min(target_vol / ewma_vol, 1.0)
                position_ratio = max(position_ratio, 0.1)
        
        # ---- 买入 ----
        day_pred = pred_all[pred_all["trade_date"] == date].sort_values(pcol, ascending=False)
        selected = list(day_pred.head(n_stocks)["ts_code"].values)
        
        if selected and cash > 0.001:
            available = cash * position_ratio * 0.98
            if available > 0.001:
                per = available / len(selected)
                for code in selected:
                    px = px_buy.get(code, 0)
                    if px > 0 and per > 0:
                        buy_cost = per * (COMM + SLIP)
                        bought = (per - buy_cost) / px
                        if bought > 0: holdings[code] = bought
                cash -= per * len(holdings)
        
        # ---- 收益 + 止损检查 ----
        new_port = sum(shares * px_sell.get(c, 0) for c, shares in holdings.items())
        new_total = new_port + cash
        ret = new_total / total_val - 1 if total_val > 0 else 0
        
        # 止损触发？
        if ret < -stop_loss:
            in_cooldown = 1  # 下一期不交易
            print(f"  ⚠️ 止损触发! ret={ret*100:.1f}% 冷却1期")
        
        navs.append(navs[-1] * (1 + ret))
        hist_rets.append(ret)
        
        if i < 3 or (i+1) % 10 == 0 or len(holdings) == 0:
            print(f"  p{i:3d} {str(date.date())}->{str(sell_date.date())} | "
                  f"持{len(holdings):3d} | 仓{position_ratio:.2f} | "
                  f"总{total_val:.3f}->{new_total:.3f} "
                  f"ret={ret*100:+6.2f}% nav={navs[-1]:.3f}")
    
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
    
    print(f"\n  {label} Top{n_stocks} 目波{target_vol*100:.0f}%:")
    print(f"    总收益: {tr*100:.1f}% | 年化: {ar*100:.1f}%")
    print(f"    实际波动: {vol*100:.1f}% | 夏普: {sr:.2f} | 回撤: {mdd*100:.1f}%")
    print(f"    卡玛: {calmar:.2f} | 胜率: {wr*100:.0f}% | {len(pnl)}期")
    print(f"    止损触发: {sum(1 for r in pnl if r < -STOP_LOSS)}次")
    return nav_arr, pnl

# 对比测试：行业中性 vs 非中性，加不加载止损
for n in [30, 50]:
    for use_n in [False, True]:
        label = "ML原始" if not use_n else "ML+行业中性"
        nav, pnl = backtest_v10(n, target_vol=0.20, use_neutral=use_n, stop_loss=STOP_LOSS, label=label)
        print()

print(f"\n总用时: {(time.time()-t0)/60:.1f}分")
