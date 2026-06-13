"""
v30 组合优化：不等权配置 + 行业中性
基于v29 LGB信号，测试三种分配策略
1. 等权 (当前v29)
2. 线性权重：rank得分从高到低线性递减
3. 凸权重：sqrt得分归一化，前几名权重更大
4. 行业中性：按中信一级行业分配，每行业最多3只
"""
import os, sys, time, json
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
os.chdir("/mnt/d/AI-20260518")
sys.path.insert(0, ".")
import joblib, lightgbm as lgb
tt = time.time()

print("="*60)
print("v30 组合优化 — 不等权 + 行业中性")
print(f"{time.strftime('%F %H:%M')}")
print("="*60)

panel = pd.read_parquet("data/factors/factor_panel_v6.parquet")
prices = pd.read_parquet("data/factors/full_prices.parquet")
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
prices["trade_date"] = pd.to_datetime(prices["trade_date"])

factor_cols = [c for c in panel.columns 
               if c not in ["ts_code","trade_date","fwd_20d_ret","close","volume","ret_1d",
                   "__fragment_index","__batch_index","__last_in_fragment","__filename","board_break"]]
core15 = ["短期反转","20日动量","60日动量","120日动量","波动率",
          "换手率","量比","量能趋势","BP","EP","MACD","BOLL位置",
          "EMA5偏离","EMA10偏离","EMA20偏离"]

# 波动率
ps = prices.sort_values(["ts_code","trade_date"]).copy()
ps["ret_1d"] = ps.groupby("ts_code")["close"].pct_change()
ps["vol_60d"] = ps.groupby("ts_code")["ret_1d"].transform(lambda x: x.rolling(60, min_periods=20).std())
ps["vol_60d_ann"] = ps["vol_60d"] * np.sqrt(244)

all_dates = sorted(panel["trade_date"].unique())
period_dates = [all_dates[i] for i in range(0, len(all_dates), 20)
                if all_dates[i] >= pd.Timestamp("2021-01-01")]

price_map, vol_map = {}, {}
for d in period_dates:
    sub = prices[prices["trade_date"]==d][["ts_code","close"]].dropna()
    price_map[d] = dict(zip(sub["ts_code"], sub["close"]))
    v_sub = ps[ps["trade_date"]==d][["ts_code","vol_60d_ann"]].dropna()
    vol_map[d] = dict(zip(v_sub["ts_code"], v_sub["vol_60d_ann"]))

# 行业分类 — 简单映射
ind_map = {}
# 用面板最近的ts_code获取前3位代码前缀做行业近似
all_codes = panel["ts_code"].unique()
for c in all_codes:
    prefix = c.split(".")[0][:3]
    # 行业按板块分类
    sec = c.split(".")[1]
    ind = int(c.split(".")[0][:3])
    # 粗略行业分类
    if sec == "SH" and ind >= 600000 and ind < 610000: ind_map[c] = "金融"
    elif sec == "SH" and ind >= 688000: ind_map[c] = "科创"
    elif sec == "SZ" and (c.startswith("00")): ind_map[c] = "主板"
    elif sec == "SZ" and (c.startswith("30")): ind_map[c] = "创业板"
    else: ind_map[c] = "其他"

# 更细致的行业
from config.settings import TS_TOKEN
import tushare as ts
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# 取股票基本信息
stk_basic = pro.query("stock_basic", exchange="", list_status="L",
                       fields="ts_code,name,industry")
stk_ind = dict(zip(stk_basic["ts_code"], stk_basic["industry"]))
print(f"行业数据: {len(stk_ind)}只", flush=True)

# LGB滚动训练（同v29）
rf_md = joblib.load("models/ml_ensemble_v1.joblib")
rf_factors = rf_md["factor_cols"]

half_dates = []
for d in period_dates:
    k = f"{d.year}{'H1' if d.month<=6 else 'H2'}"
    if not half_dates or half_dates[-1][0]!=k:
        half_dates.append((k,d))
    else:
        half_dates[-1]=(k,d)

lgb_all = pd.DataFrame({"trade_date": pd.Series(dtype="datetime64[ns]"),
    "ts_code": pd.Series(dtype="str"), "pred_ret": pd.Series(dtype="float64")})

for hi,(hk,train_cutoff) in enumerate(half_dates):
    if hi<3: continue
    train_end = train_cutoff - pd.Timedelta(days=5)
    train = panel[(panel["trade_date"]>=train_end-pd.Timedelta(days=730)) & (panel["trade_date"]<=train_end)]
    train = train[train["fwd_20d_ret"].notna() & (train["fwd_20d_ret"].abs()<0.5)]
    if len(train)<20000: continue
    if len(train)>150000: train=train.sample(150000,random_state=42)
    
    X_tr = train[rf_factors].fillna(0).values.astype(np.float32)
    y_tr = np.clip(train["fwd_20d_ret"].values.astype(np.float32),-0.3,0.3)
    nv = max(1,int(len(train)*0.15))
    
    lgb_m = lgb.LGBMRegressor(**dict(n_estimators=500,max_depth=3,learning_rate=0.02,
        subsample=0.7,colsample_bytree=0.7,reg_alpha=0.2,reg_lambda=1.0,
        min_child_weight=20,min_data_in_leaf=100,random_state=42,verbose=-1,n_jobs=8))
    lgb_m.fit(X_tr[:-nv],y_tr[:-nv],eval_set=[(X_tr[-nv:],y_tr[-nv:])],
              callbacks=[lgb.early_stopping(30,verbose=False)],eval_metric="mse")
    
    for d in period_dates:
        if d<=train_cutoff: continue
        idx = panel["trade_date"]==d
        day = panel[idx]
        X_te = day[rf_factors].fillna(0).values.astype(np.float32)
        preds = lgb_m.predict(X_te)
        lgb_all = pd.concat([lgb_all,pd.DataFrame({"trade_date":d,"ts_code":day["ts_code"].values,
            "pred_ret":preds.astype(np.float64)})], ignore_index=True)

lgb_map = {}
for d in lgb_all["trade_date"].unique():
    lgb_map[d] = dict(zip(lgb_all[lgb_all["trade_date"]==d]["ts_code"], 
                          lgb_all[lgb_all["trade_date"]==d]["pred_ret"]))

# 三种权重方案 + 行业中性
STAMP, COMM, SLIP = 0.001, 0.0002, 0.002

def calc_weights(scores, method="linear", n_hold=30):
    """计算分配权重"""
    scores = np.array(scores)
    order = np.argsort(-scores)
    top = order[:n_hold]
    
    if method == "equal":
        w = np.ones(n_hold) / n_hold
    elif method == "linear":
        # 线性递减: 1, 2/3, 1/3, ...
        w = np.linspace(1, 0.2, n_hold)
        w = w / w.sum()
    elif method == "sqrt":
        # 凸权重: sqrt(rank)归一化
        r = np.arange(1, n_hold+1)
        w = 1 / np.sqrt(r)
        w = w / w.sum()
    elif method == "convex":
        # 指数衰减: e^{-0.1 * rank}
        r = np.arange(1, n_hold+1)
        w = np.exp(-0.1 * r)
        w = w / w.sum()
    else:
        w = np.ones(n_hold) / n_hold
    
    return top, w

def apply_industry_neutral(codes, scores, ind_dict, max_per_ind=3, n_hold=30):
    """行业中性：同行业最多max_per_ind只"""
    order = np.argsort(-scores)
    selected = []
    ind_count = {}
    
    for idx in order:
        code = codes[idx]
        ind = ind_dict.get(code, "其他")
        cnt = ind_count.get(ind, 0)
        if cnt < max_per_ind:
            selected.append(idx)
            ind_count[ind] = cnt + 1
        if len(selected) >= n_hold:
            break
    
    if len(selected) < n_hold:
        # 补足
        for idx in order:
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= n_hold:
                break
    
    return np.array(selected[:n_hold])

def bt_weighted(pdf, method="equal", n=30, tv=0.15, label="", 
                industry_neutral=False, ind_dict=None, start_date=None):
    pdts = sorted(pdf["trade_date"].unique())
    if start_date:
        pdts = [d for d in pdts if d >= start_date]
    if len(pdts) < 2: return None
    cash, hold, navs = 0.03, {}, [1.0]
    
    for i in range(len(pdts)-1):
        d, sd = pdts[i], pdts[i+1]
        pb, psm = price_map.get(d, {}), price_map.get(sd, {})
        sv = vol_map.get(d, {})
        
        hv = sum(shares * pb.get(c,0) for c,shares in hold.items())
        tv_ = hv + cash
        
        # 卖出
        sp = 0
        for c,shares in hold.items():
            px = psm.get(c,0)
            if px>0:
                v = shares*px
                sp += v - v*(STAMP+COMM+SLIP)
        cash += sp
        hold = {}
        
        # 选择
        dp_ = pdf[pdf["trade_date"]==d].copy()
        dp_ = dp_.sort_values("pred_ret", ascending=False)
        
        codes = list(dp_["ts_code"])
        scores = dp_["pred_ret"].values
        
        if industry_neutral and ind_dict:
            top_idx = apply_industry_neutral(codes, scores, ind_dict, 3, n)
            sel_codes = [codes[j] for j in top_idx]
            _, weights = calc_weights(scores[top_idx], method, len(top_idx))
        else:
            top_idx, weights = calc_weights(scores, method, n)
            sel_codes = [codes[j] for j in top_idx]
        
        # 波动率调整总仓位
        sel_vols = [sv.get(c,np.nan) for c in sel_codes]
        sel_vols = [v for v in sel_vols if not np.isnan(v) and v>0.01]
        pr = max(min(tv/np.median(sel_vols), 1.0) if len(sel_vols)>=5 else 1.0, 0.05)
        
        if sel_codes and cash>0.001:
            alloc = cash * pr * 0.98
            if alloc>0.001:
                for code, w in zip(sel_codes, weights):
                    px = pb.get(code, 0)
                    if px>0 and w>0:
                        amt = alloc * w
                        cost = amt * (COMM+SLIP)
                        bought = (amt - cost) / px
                        if bought>0:
                            hold[code] = hold.get(code, 0) + bought
                cash -= alloc
        
        np_ = sum(shares * psm.get(c,0) for c,shares in hold.items())
        nt = np_ + cash
        r = nt/tv_ - 1 if tv_>0 else 0
        navs.append(navs[-1]*(1+r))
    
    pnl = np.array(navs[1:])/np.array(navs[:-1])-1
    na = np.array(navs)
    ny = len(pnl)/13
    ar = na[-1]**(1/ny)-1 if ny>0 and na[-1]>0 else 0
    vv = np.std(pnl)*np.sqrt(13)
    sr = np.mean(pnl)/np.std(pnl)*np.sqrt(13) if np.std(pnl)>0 else 0
    dd = np.maximum.accumulate(na)-na
    mdd = dd.max()
    wr = np.mean(pnl>0)
    cal = ar/mdd if mdd>0 else 0
    print(f"  {label:25s}: 年化{ar*100:+7.1f}% | 夏普{sr:6.2f} | 回撤{mdd*100:6.1f}% | 胜率{wr*100:3.0f}% | 卡玛{cal:.2f}")
    return {"ret":f"{ar*100:+.1f}%","sr":f"{sr:.2f}","mdd":f"{mdd*100:.1f}%","num":len(pnl)}

# 构建LGB信号
print("\n构建信号...", flush=True)
preds = {"trade_date":[], "ts_code":[], "pred_ret":[]}
for d in period_dates:
    day = panel[panel["trade_date"]==d]
    if len(day)==0: continue
    lgb_d = lgb_map.get(d, {})
    for _, row in day.iterrows():
        preds["trade_date"].append(d)
        preds["ts_code"].append(row["ts_code"])
        preds["pred_ret"].append(lgb_d.get(row["ts_code"], row.get("s_rf", 0)))
pred_df = pd.DataFrame(preds)

print(f"\n{'='*60}")
print("T30 目波15% 2021-2026")
print(f"{'='*60}")

results = []
configs = [
    ("equal_raw", "等权", False, None),
    ("linear_raw", "线性递减", False, None),
    ("sqrt_raw", "sqrt凸权", False, None),
    ("convex_raw", "指数衰减", False, None),
    ("equal_ind", "等权+行业中性", True, stk_ind),
    ("linear_ind", "线性+行业中性", True, stk_ind),
    ("sqrt_ind", "sqrt+行业中性", True, stk_ind),
    ("convex_ind", "指数+行业中性", True, stk_ind),
]

methods_map = {"equal":"equal","linear":"linear","sqrt":"sqrt","convex":"convex"}

for cfg in configs:
    key, name, ind, ind_d = cfg
    m_name = key.split("_")[0]
    method = methods_map.get(m_name, "equal")
    r = bt_weighted(pred_df, method=method, n=30, tv=0.15, label=name,
                    industry_neutral=ind, ind_dict=ind_d)
    if r: results.append((name, r))

# 存结果
res_j = {n: r for n, r in results}
json.dump(res_j, open("output/backtest_v30_weight.json","w"), indent=2, default=str)

print(f"\n⏱ {(time.time()-tt)/60:.1f}分")
print("✅ output/backtest_v30_weight.json")
