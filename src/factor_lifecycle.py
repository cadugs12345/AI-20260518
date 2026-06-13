"""
因子生命周期管理系统
- IC衰减追踪
- 因子淘汰/激活管理
- 因子贡献度分析
"""
import os, sys, time
import numpy as np
import pandas as pd

warnings = __import__('warnings')
warnings.filterwarnings("ignore")

class FactorLifecycle:
    """因子生命周期管理"""
    
    def __init__(self, panel_path=None, panel=None):
        if panel is not None:
            self.panel = panel
        else:
            base = panel_path or "data/factors/factor_panel_v3.parquet"
            self.panel = pd.read_parquet(base)
        self.panel["trade_date"] = pd.to_datetime(self.panel["trade_date"])
        self.all_dates = sorted(self.panel["trade_date"].unique())
        
        skip = {"ts_code","trade_date","fwd_20d_ret","fwd_5d_ret"}
        self.factors = [c for c in self.panel.columns 
                        if c not in skip and self.panel[c].dtype in ("float64","int64")]
        
        # 因子状态表
        self.registry = pd.DataFrame({
            "factor": self.factors,
            "active": True,
            "added_date": pd.Timestamp("2024-01-01"),
            "last_decay_check": pd.Timestamp("2020-01-01"),
            "ic_series": None,
        })
        
    def calc_ic_series(self, factor_name, label="fwd_20d_ret", window=20):
        """计算因子的IC时间序列"""
        fcol = factor_name if factor_name in self.panel.columns else None
        if fcol is None:
            return pd.DataFrame()
        
        ic_dates, ic_vals = [], []
        for date in self.all_dates[::window]:  # 每20日取一次减少计算量
            day = self.panel[self.panel["trade_date"] == date]
            day = day[[fcol, label]].dropna()
            if len(day) < 50:
                continue
            vals = day[fcol].values
            rets = day[label].values
            mask = np.abs(rets) < 0.5
            if mask.sum() < 50:
                continue
            from scipy import stats
            ic, _ = stats.spearmanr(vals[mask], rets[mask])
            ic_dates.append(date)
            ic_vals.append(ic)
        
        return pd.DataFrame({"trade_date": ic_dates, "IC": ic_vals})
    
    def compute_decay(self, factor_name, windows=[(0,60),(60,120),(120,240)]):
        """计算因子IC在不同窗口的衰减"""
        ic_df = self.calc_ic_series(factor_name)
        if len(ic_df) < 10:
            return None
        
        results = {}
        for w_name, start, end in [(f"{w[0]}-{w[1]}日", w[0], w[1]) for w in windows]:
            seg = ic_df.iloc[-end:-start] if start > 0 else ic_df.iloc[-end:]
            if len(seg) > 5:
                ir = seg["IC"].mean() / seg["IC"].std() if seg["IC"].std() > 0 else 0
                results[w_name] = {
                    "n": len(seg),
                    "ic_mean": seg["IC"].mean(),
                    "ic_std": seg["IC"].std(),
                    "ic_ir": ir,
                }
        return results
    
    def scan_all_factors_decay(self):
        """扫描所有活跃因子的IC衰减"""
        print("因子IC衰减扫描...")
        rows = []
        for f in self.factors:
            decay = self.compute_decay(f)
            if decay:
                # 最后100期 vs 更早100期
                ic_df = self.calc_ic_series(f)
                if len(ic_df) > 30:
                    recent = ic_df["IC"].tail(20)
                    early = ic_df["IC"].head(20)
                    recent_ir = recent.mean() / recent.std() if recent.std() > 0 else 0
                    early_ir = early.mean() / early.std() if early.std() > 0 else 0
                    decay_rate = recent_ir - early_ir if abs(early_ir) > 0.1 else 0
                else:
                    recent_ir, early_ir, decay_rate = 0, 0, 0
                
                row = {
                    "factor": f,
                    "ic_mean": decay.get("120-240日", {}).get("ic_mean", 0),
                    "ic_ir_recent": recent_ir,
                    "ic_ir_early": early_ir,
                    "decay_rate": decay_rate,
                    "abs_ic": abs(decay.get("120-240日", {}).get("ic_mean", 0)),
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        # 按|IC|排序
        df = df.sort_values("abs_ic", ascending=False)
        
        print(f"\n{'因子':20s} | {'IC均值':>8s} | {'早期IC_IR':>9s} | {'近期IC_IR':>9s} | {'衰减':>6s}")
        print("-"*65)
        for _, r in df.iterrows():
            decay_str = f"{r['decay_rate']:+.2f}" if abs(r.get('decay_rate',0)) > 0 else "0.00"
            print(f"{r['factor']:20s} | {r['ic_mean']*100:+7.3f}% | {r['ic_ir_early']:+7.3f} | "
                  f"{r['ic_ir_recent']:+7.3f} | {decay_str:>6s}")
        
        return df


class FactorContribution:
    """因子贡献度分析（在ML模型中的重要性）"""
    
    def __init__(self, panel, factor_cols, label_col="fwd_20d_ret"):
        self.panel = panel
        self.factor_cols = factor_cols
        self.label_col = label_col
        
    def train_and_analyze(self, n_estimators=300):
        """训练一个完整模型并输出因子重要性"""
        import xgboost as xgb
        data = self.panel[self.factor_cols + [self.label_col]].dropna()
        data = data[data[self.label_col].abs() < 0.5]
        if len(data) > 100000:
            data = data.sample(100000, random_state=42)
        
        X = np.nan_to_num(data[self.factor_cols].values.astype(np.float32), nan=0)
        y = data[self.label_col].values.astype(np.float32)
        
        model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=4,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0, n_jobs=8)
        model.fit(X, y, eval_set=[(X, y)], verbose=False)
        
        importance = pd.DataFrame({
            "factor": self.factor_cols,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        
        importance["cumsum"] = importance["importance"].cumsum()
        importance["rank"] = range(1, len(importance)+1)
        
        return importance, model


# ==================== 快速测试 ====================

if __name__ == "__main__":
    t0 = time.time()
    print("="*60)
    print("因子生命周期 & 贡献度分析")
    print("="*60)
    
    # 加载
    panel = pd.read_parquet("data/factors/factor_panel_with_fwd_v2.parquet")
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    
    # 因子列
    skip = {"ts_code","trade_date","fwd_20d_ret","fwd_5d_ret","均值","20日收益率",
            "短期反转","毛利率","量比","BP","SP","EP","股息率","流通市值"}
    factors = [c for c in panel.columns if c not in skip and panel[c].dtype in ("float64","int64")]
    print(f"因子数: {len(factors)}")
    print(f"因子: {factors}")
    
    # 1. 贡献度分析
    print(f"\n{'='*60}")
    print("因子贡献度分析 (XGBoost)")
    print(f"{'='*60}")
    contrib = FactorContribution(panel, factors)
    importance, model = contrib.train_and_analyze()
    
    print(f"\n{'因子':24s} | {'重要性':>8s} | {'累积':>8s}")
    print("-"*45)
    for _, r in importance.iterrows():
        bar = "█" * int(r["importance"] * 50)
        print(f"{r['factor']:24s} | {r['importance']*100:6.2f}% | {r['cumsum']*100:6.2f}% | {bar}")
    
    # 累计排名
    top5_pct = importance.iloc[:5]["importance"].sum()
    top10_pct = importance.iloc[:10]["importance"].sum()
    print(f"\nTop5因子累积: {top5_pct*100:.1f}%")
    print(f"Top10因子累积: {top10_pct*100:.1f}%")
    
    # 2. IC衰减扫描 (只扫描最重要的10个，速度快)
    print(f"\n{'='*60}")
    print("因子IC衰减扫描 (Top8因子)")
    print(f"{'='*60}")
    lifecycle = FactorLifecycle(panel=panel)
    top_factors = importance.head(8)["factor"].tolist()
    
    rows = []
    for f in top_factors:
        ic_df = lifecycle.calc_ic_series(f)
        if len(ic_df) > 30:
            recent = ic_df["IC"].tail(30)
            early = ic_df["IC"].head(30)
            recent_ir = recent.mean() / recent.std() if recent.std() > 0 else 0
            early_ir = early.mean() / early.std() if early.std() > 0 else 0
            decay = recent_ir - early_ir
        else:
            recent_ir = early_ir = decay = 0
        rows.append({
            "factor": f,
            "ic_mean": ic_df["IC"].mean(),
            "early_IR": early_ir,
            "recent_IR": recent_ir,
            "decay": decay,
        })
    
    decay_df = pd.DataFrame(rows).sort_values("ic_mean", ascending=False)
    print(f"\n{'因子':20s} | {'IC均值':>8s} | {'早期IR':>7s} | {'近期IR':>7s} | {'衰减':>6s} | {'状态':>6s}")
    print("-"*70)
    for _, r in decay_df.iterrows():
        if r["decay"] < -0.3:
            status = "⚠️衰减"
        elif r["decay"] > 0.3:
            status = "✅增强"
        else:
            status = "  稳定"
        print(f"{r['factor']:20s} | {r['ic_mean']*100:+7.3f}% | {r['early_IR']:+6.2f} | "
              f"{r['recent_IR']:+6.2f} | {r['decay']:+5.2f} | {status}")
    
    print(f"\n总用时: {time.time()-t0:.1f}秒")
