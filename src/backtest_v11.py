"""
v11 - 20日调仓 + GARCH波动率预测 + 止损 + 行业中性（可选）
"""
import os, sys, time, gc, pickle
import numpy as np
import pandas as pd
import tushare as ts
from arch import arch_model

warnings = __import__('warnings')
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_FACTORS = "data/factors"

t0 = time.time()
print("="*60)
print("v11 - 20日调仓 + GARCH波动率")
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
panel['行业'] = panel['ts_code'].map(ind_map).fillna('其他')
ind_counts = panel.groupby(['trade_date', '行业']).size().reset_index(name='cnt')
avg_cnt = ind_counts.groupby('行业')['cnt'].mean()
small_inds = set(avg_cnt[avg_cnt < 10].index)
panel['行业_大类'] = panel['行业'].apply(lambda x: '其他' if x in small_inds else x)

# ===== 因子列 =====
factor_cols = [c for c in panel.columns 
               if c not in ("ts_code","trade_date","fwd_20d_ret","行业","行业_大类")
               and panel[c].dtype in ("float64","int64")]
factor_cols = [c for c in factor_cols if c not in ("短期反转","毛利率","量比","BP","SP","EP","股息率","流通市值")]

# ===== 日频等权收益（用于GARCH）=====
print("构建日频等权收益...")
prices_sorted = prices.sort_values(['trade_date','ts_code']).copy()
prices_sorted['ret_1d'] = prices_sorted.groupby('ts_code')['close'].pct_change()
daily_ret = prices_sorted.groupby('trade_date')['ret_1d'].mean().dropna()
print(f"  日收益序列: {len(daily_ret)} 日, {daily_ret.index[0].date()} ~ {daily_ret.index[-1].date()}")
print(f"  年化波动: {daily_ret.std()*np.sqrt(244):.1%}")

# ===== 预计算GARCH预测波动 =====
print("预计算GARCH预测...")
garch_cache = os.path.join(DATA_FACTORS, "garch_forecasts.pkl")

# 获取period_dates
all_dates = sorted(panel["trade_date"].unique())
period_dates = [all_dates[i] for i in range(0, len(all_dates), 20) 
                if all_dates[i] >= pd.Timestamp("2021-01-01")]

if os.path.exists(garch_cache):
    with open(garch_cache, 'rb') as f:
        garch_pred = pickle.load(f)
    print(f"  GARCH缓存: {len(garch_pred)} 个预测")
else:
    garch_pred = {}
    # 对每个period_date，用截至该日的日收益训练GARCH，预测未来20日波动
    for i, date in enumerate(period_dates):
        hist_ret = daily_ret.loc[:date]
        if len(hist_ret) < 500: continue
        
        model = arch_model(hist_ret.values * 100, vol='Garch', p=1, q=1, dist='normal',
                          mean='zero')  # 均值归零更稳定
        try:
            res = model.fit(disp='off', options={'maxiter': 500})
            fc = res.forecast(horizon=20)
            # 预测方差：horizon 1..20的预测方差之和 = 未来20日总方差
            pred_vars = fc.variance.iloc[-1].values
            # 年化波动 = sqrt(总方差 / 20 * 244) 但需要转回百分比
            # fc方差是在 scale=100 时计算的（原始收益*100）
            total_var_20d = np.sum(pred_vars) / 10000  # 转回原始尺度
            annual_vol = np.sqrt(total_var_20d / 20 * 244)
            garch_pred[date] = annual_vol
            
            # 历史波动（滚动窗口，作为对比）
            hist_20d_vol = hist_ret.tail(20).std() * np.sqrt(244) if len(hist_ret) >= 20 else np.nan
        except Exception as e:
            garch_pred[date] = np.nan
        
        if (i+1) % 20 == 0:
            print(f"  [{i+1}/{len(period_dates)}] {str(date.date())} GARCH={garch_pred.get(date, 0):.1%}")
    
    with open(garch_cache, 'wb') as f:
        pickle.dump(garch_pred, f)
    print(f"  GARCH完成: {len(garch_pred)} 个预测")

# ===== ML预测缓存 =====
pred_path = os.path.join(DATA_FACTORS, "pred_20d_v10.pkl")
if os.path.exists(pred_path):
    with open(pred_path, 'rb') as f:
        pred_all = pickle.load(f)
    print(f"ML缓存: {len(pred_all):,}条")
else:
    print("[ML] 需要先跑v10生成缓存!")
    sys.exit(1)

# 检查是否有行业中性列
if "pred_neutral" not in pred_all.columns:
    print("  添加行业中性列...")
    panel_dates = panel[["ts_code","trade_date","行业_大类"]].drop_duplicates()
    pred_all = pred_all.merge(panel_dates, on=["ts_code","trade_date"], how="left")
    pred_all["行业_大类"] = pred_all["行业_大类"].fillna("其他")
    def rank_zscore(s):
        r = s.rank(pct=True)
        return (r - r.mean()) / r.std()
    pred_all["pred_neutral"] = pred_all.groupby("行业_大类")["pred_ret"].transform(rank_zscore)
    print(f"  完成: {len(pred_all):,}条")

pred_dates = sorted(pred_all["trade_date"].unique())

# ===== 无成本验证 =====
print(f"\n无成本验证:")
for pcol, label in [("pred_ret", "ML原始"), ("pred_neutral", "ML+行业中性")]:
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
print("含成本 + GARCH波动率 + 止损")
print(f"{'='*60}")

STAMP, COMM, SLIP = 0.001, 0.0002, 0.002
STOP_LOSS = 0.08

def backtest_v11(n_stocks, target_vol=0.20, use_neutral=True, stop_loss=STOP_LOSS, label=""):
    pcol = "pred_neutral" if use_neutral else "pred_ret"
    cash = 0.03
    holdings = {}
    navs = [1.0]
    in_cooldown = 0
    hist_rets = []
    
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
        
        # ---- 止损冷却 ----
        if in_cooldown > 0:
            in_cooldown -= 1
            new_total = cash
            ret = new_total / total_val - 1 if total_val > 0 else 0
            navs.append(navs[-1] * (1 + ret))
            hist_rets.append(ret)
            continue
        
        # ---- GARCH波动率仓位 ----
        est_vol = garch_pred.get(date, np.nan)
        if np.isnan(est_vol) or est_vol < 0.01:
            position_ratio = 1.0
        else:
            # 如果预测波动非常高，降仓
            position_ratio = min(target_vol / est_vol, 1.0)
            position_ratio = max(position_ratio, 0.1)
        
        # ---- 买入 ----
        day_pred = pred_all[pred_all["trade_date"] == date].sort_values(pcol, ascending=False)
        selected = list(day_pred.head(n_stocks)["ts_code"].values) if n_stocks == 30 else \
                   list(day_pred.head(n_stocks)["ts_code"].values)
        
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
        
        # ---- 收益 + 止损 ----
        new_port = sum(shares * px_sell.get(c, 0) for c, shares in holdings.items())
        new_total = new_port + cash
        ret = new_total / total_val - 1 if total_val > 0 else 0
        
        if ret < -stop_loss:
            in_cooldown = 1
        
        navs.append(navs[-1] * (1 + ret))
        hist_rets.append(ret)
        
        if i < 3 or (i+1) % 10 == 0:
            vol_str = f"{est_vol*100:.0f}%" if not np.isnan(est_vol) else "N/A"
            print(f"  p{i:3d} {str(date.date())}->{str(sell_date.date())} | "
                  f"持{len(holdings):3d} | GARCH波{vol_str} | "
                  f"仓{position_ratio:.2f} | "
                  f"总{total_val:.3f}->{new_total:.3f} "
                  f"ret={ret*100:+6.2f}% nav={navs[-1]:.3f}" +
                  (" ⚠️止损" if ret < -stop_loss else ""))
    
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

# 构建price_map
price_map = {}
for d in pred_dates:
    sub = prices[prices["trade_date"] == d][["ts_code","close"]].dropna()
    price_map[d] = dict(zip(sub["ts_code"], sub["close"]))

# 跑
for n in [30, 50]:
    for tv in [0.20, 0.25]:
        backtest_v11(n, target_vol=tv, use_neutral=False, label="ML原始")
        print()

print(f"\n总用时: {(time.time()-t0)/60:.1f}分")
