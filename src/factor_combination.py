"""
多因子合成与迭代优化
方案1: 滚动EWMA-IC加权 (基准组合)
方案2: 均值-方差最优加权 (约束优化)
方案3: 等权基准对比

输出: 每日每只股票的复合因子值 (Composite Score)
"""
import os, sys, warnings, time
import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DATA_FACTORS, DATA_RAW, BACKTEST_START


class FactorCombiner:
    """
    多因子合成器
    """

    def __init__(self):
        self.factor_panel = None
        self.factor_cols = []
        self.ic_history = {}      # {factor_name: [{date, ic, n_stocks}, ...]}
        self.weights_history = {} # {date: {factor_name: weight}}

    def load_data(self, panel_path: str = None):
        """加载因子面板"""
        if panel_path is None:
            panel_path = os.path.join(DATA_FACTORS, "factor_panel.parquet")
        self.factor_panel = pd.read_parquet(panel_path)
        self.factor_panel["trade_date"] = pd.to_datetime(self.factor_panel["trade_date"])

        # 识别因子列
        exclude = {"ts_code", "trade_date", "ann_date", "end_date"}
        self.factor_cols = [
            c for c in self.factor_panel.columns
            if c not in exclude
            and self.factor_panel[c].dtype in ["float64", "int64"]
        ]
        print(f"[组合] 加载因子面板: {len(self.factor_panel)} 条, {len(self.factor_cols)} 个因子")
        return self

    def _get_future_return(self, code: str, date_ts) -> float:
        """获取未来20日收益"""
        date_str = date_ts.strftime("%Y%m%d")
        daily_path = os.path.join(DATA_RAW, "daily", f"{code}.parquet")
        if not os.path.exists(daily_path):
            return np.nan
        daily = pd.read_parquet(daily_path).sort_values("trade_date")
        mask = daily["trade_date"] == date_str
        if not mask.any():
            return np.nan
        idx = mask.idxmax()
        future_idx = idx + 20
        if future_idx >= len(daily):
            return np.nan
        return (daily.loc[future_idx, "close"] - daily.loc[idx, "close"]) / daily.loc[idx, "close"]

    def calc_rolling_ic(self, lookback: int = 60):
        """
        计算每个因子每期的IC值
        lookback: 滚动IC窗口 (用于EWMA加权)
        """
        print(f"\n[IC滚动] 计算滚动IC, 窗口={lookback}天...")

        dates = sorted(self.factor_panel["trade_date"].unique())
        dates = [d for d in dates if str(d.date()) >= BACKTEST_START]

        # 按批次减少IO
        daily_cache = {}

        for factor in self.factor_cols:
            self.ic_history[factor] = []

        batch_size = 100
        total = len(dates)

        for batch_start in range(0, total, batch_size):
            batch_dates = dates[batch_start:batch_start + batch_size]

            for date in batch_dates:
                mask = self.factor_panel["trade_date"] == date
                df_day = self.factor_panel[mask].dropna(subset=self.factor_cols, thresh=1).copy()
                if len(df_day) < 50:
                    continue

                codes = df_day["ts_code"].tolist()

                for factor in self.factor_cols:
                    sub = df_day.dropna(subset=[factor])
                    if len(sub) < 50:
                        continue

                    # 获取未来收益
                    rets = []
                    valid_codes = []
                    for _, row in sub.iterrows():
                        ret = self._get_future_return(row["ts_code"], date)
                        if pd.notna(ret):
                            rets.append(ret)
                            valid_codes.append(row["ts_code"])

                    if len(valid_codes) < 50:
                        continue

                    f_vals = sub[sub["ts_code"].isin(valid_codes)][factor].values[:len(valid_codes)]
                    factor_map = dict(zip(sub["ts_code"], sub[factor]))
                    aligned_f = np.array([factor_map.get(c, np.nan) for c in valid_codes])

                    nan_mask = ~(np.isnan(aligned_f) | np.isnan(rets))
                    if nan_mask.sum() < 50:
                        continue

                    from scipy import stats as sp_stats
                    ic, _ = sp_stats.spearmanr(aligned_f[nan_mask], np.array(rets)[nan_mask])

                    self.ic_history[factor].append({
                        "date": date,
                        "ic": ic,
                        "n_stocks": nan_mask.sum(),
                    })

            if (batch_start + batch_size) % 200 == 0 or batch_start == 0:
                print(f"  [IC] {min(batch_start+batch_size, total)}/{total} 交易日")

        print(f"[IC] 完成, 因子IC样本数: {[len(v) for k,v in self.ic_history.items()][:5]}...")
        return self

    # ==============================
    # 方案1: 滚动EWMA-IC加权
    # ==============================

    def combine_ewma_ic(self, halflife: int = 40, min_periods: int = 20) -> pd.DataFrame:
        """
        用滚动EWMA-IC加权合成因子

        halflife: IC指数衰减半衰期(天), 新近IC权重越大
        min_periods: 最少需要的IC样本数

        返回: composite_scores (trade_date x ts_code 的复合因子值)
        """
        print(f"\n{'='*60}")
        print(f"方案1: EWMA-IC加权复合因子 (halflife={halflife}天)")
        print(f"{'='*60}")

        if not self.ic_history:
            self.calc_rolling_ic()

        dates = sorted(self.factor_panel["trade_date"].unique())
        dates = [d for d in dates if str(d.date()) >= BACKTEST_START]

        composite_list = []

        # 将IC历史转换为DataFrame
        ic_dfs = {}
        for factor in self.factor_cols:
            if self.ic_history.get(factor):
                ic_df = pd.DataFrame(self.ic_history[factor])
                ic_df["date"] = pd.to_datetime(ic_df["date"])
                ic_df = ic_df.sort_values("date").set_index("date")
                ic_dfs[factor] = ic_df

        # EWMA权重: 越近权重越大
        def ewma_weights(n, hl):
            lam = np.exp(-np.log(2) / hl)
            w = np.array([lam ** (n - 1 - i) for i in range(n)])
            return w / w.sum()

        for date in dates:
            factor_weights = {}

            for factor in self.factor_cols:
                if factor not in ic_dfs:
                    continue
                ic_df = ic_dfs[factor]
                ic_before = ic_df[ic_df.index <= date]
                if len(ic_before) < min_periods:
                    continue

                ic_series = ic_before["ic"].values[-min(len(ic_before), 120):]
                n = len(ic_series)

                # EWMA滚动IC均值
                w = ewma_weights(n, halflife)
                ewma_ic = np.sum(ic_series * w)

                factor_weights[factor] = {
                    "ic_ewma": ewma_ic,
                    "n_obs": n,
                }

            if len(factor_weights) < 3:
                continue

            # 仅保留正IC的因子 (有效因子), 归一化权重
            pos_factors = {k: v for k, v in factor_weights.items() if v["ic_ewma"] > 0}
            if not pos_factors:
                continue

            total_ic = sum(v["ic_ewma"] for v in pos_factors.values())
            norm_weights = {k: v["ic_ewma"] / total_ic for k, v in pos_factors.items()}

            # 保存权重历史
            self.weights_history[date] = norm_weights

            # 计算该日每只股票的复合因子值
            mask = self.factor_panel["trade_date"] == date
            df_day = self.factor_panel[mask].copy()

            if df_day.empty:
                continue

            # 因子标准化 (z-score)
            scores = np.zeros(len(df_day))
            valid_factors_used = []

            for factor, w in norm_weights.items():
                if factor not in df_day.columns:
                    continue
                vals = df_day[factor].values.astype(float)

                # 去极值 + 标准化
                mask_valid = ~np.isnan(vals)
                if mask_valid.sum() < 20:
                    continue

                vals_clipped = np.clip(
                    vals,
                    np.nanpercentile(vals, 1),
                    np.nanpercentile(vals, 99),
                )
                mean_v = np.nanmean(vals_clipped)
                std_v = np.nanstd(vals_clipped)
                if std_v > 0:
                    z_scores = (vals_clipped - mean_v) / std_v
                else:
                    z_scores = np.zeros_like(vals_clipped)
                z_scores[~mask_valid] = 0

                scores += w * z_scores
                valid_factors_used.append(factor)

            if len(valid_factors_used) < 3:
                continue

            df_result = pd.DataFrame({
                "trade_date": date,
                "ts_code": df_day["ts_code"].values,
                "composite_score": scores,
                "n_factors": len(valid_factors_used),
            })
            composite_list.append(df_result)

        df_composite = pd.concat(composite_list, ignore_index=True)
        print(f"[EWMA-IC] 复合因子生成: {len(df_composite)} 条, "
              f"日期范围: {df_composite['trade_date'].min().date()} ~ {df_composite['trade_date'].max().date()}")

        # 保存
        save_path = os.path.join(DATA_FACTORS, "composite_ewma.parquet")
        df_composite.to_parquet(save_path, index=False)
        print(f"[EWMA-IC] 保存: {save_path}")

        return df_composite

    # ==============================
    # 方案2: 均值-方差最优加权
    # ==============================

    def combine_mean_variance_optimal(self, lookback: int = 60,
                                       risk_aversion: float = 1.0) -> pd.DataFrame:
        """
        均值-方差 (Mean-Variance) 最优加权

        max  w' * mu - 0.5 * risk_aversion * w' * Sigma * w
        s.t. sum(w) = 1, w_i >= 0 (只做多)

        mu:    过去lookback天的滚动IC均值向量
        Sigma: 过去lookback天的IC协方差矩阵
        """
        print(f"\n{'='*60}")
        print(f"方案2: 均值-方差最优加权 (lookback={lookback}, λ={risk_aversion})")
        print(f"{'='*60}")

        if not self.ic_history:
            self.calc_rolling_ic()

        # 构建IC面板: [date x factor]
        ic_series_list = []
        for factor in self.factor_cols:
            if self.ic_history.get(factor):
                ic_df = pd.DataFrame(self.ic_history[factor])
                ic_df["date"] = pd.to_datetime(ic_df["date"])
                ic_df = ic_df.set_index("date")[["ic"]].rename(columns={"ic": factor})
                ic_series_list.append(ic_df)

        if not ic_series_list:
            print("[MVO] 无IC数据")
            return pd.DataFrame()

        ic_panel = pd.concat(ic_series_list, axis=1).sort_index()
        print(f"[MVO] IC面板: {ic_panel.shape[0]} 期, {ic_panel.shape[1]} 因子")

        # 优化函数
        def neg_sharpe_ratio(weights, mu, sigma):
            port_return = weights @ mu
            port_risk = np.sqrt(weights @ sigma @ weights)
            return -(port_return / port_risk) if port_risk > 0 else 0

        # 获取交易日序列
        dates = sorted(self.factor_panel["trade_date"].unique())
        dates = [d for d in dates if str(d.date()) >= BACKTEST_START]

        composite_list = []
        rebalance_dates = []
        current_weights = None

        for i, date in enumerate(dates):
            # 每月调仓 (大约21个交易日)
            is_rebalance = (i == 0) or (i % 21 == 0)

            if is_rebalance:
                # 用过去lookback天的IC数据估计mu和Sigma
                ic_window = ic_panel[ic_panel.index <= date].tail(lookback)
                if len(ic_window) < 20:
                    continue

                mu = ic_window.mean().values
                sigma = ic_window.cov().values

                # 处理NaN
                valid = ~(np.isnan(mu) | np.isnan(sigma).any(axis=1))
                if valid.sum() < 3:
                    continue

                mu_valid = mu[valid]
                sigma_valid = sigma[np.ix_(valid, valid)]
                valid_factors = ic_window.columns[valid].tolist()

                # 约束优化: 只做多权重
                n = len(mu_valid)
                bounds = [(0, 1) for _ in range(n)]
                constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
                x0 = np.ones(n) / n

                try:
                    result = minimize(
                        neg_sharpe_ratio, x0,
                        args=(mu_valid, sigma_valid),
                        method="SLSQP",
                        bounds=bounds,
                        constraints=constraints,
                        options={"maxiter": 1000, "ftol": 1e-12},
                    )
                    if result.success:
                        current_weights = dict(zip(valid_factors, result.x))
                        # 剔除小权重 (< 1%)
                        current_weights = {k: v for k, v in current_weights.items() if v > 0.01}
                        rebalance_dates.append(date)
                    else:
                        continue
                except Exception as e:
                    print(f"  [MVO] 优化异常 {date.date()}: {e}")
                    continue

            if current_weights is None or len(current_weights) < 3:
                continue

            # 计算该日复合因子
            mask = self.factor_panel["trade_date"] == date
            df_day = self.factor_panel[mask].copy()
            if df_day.empty:
                continue

            scores = np.zeros(len(df_day))
            factors_used = []

            for factor, w in current_weights.items():
                if factor not in df_day.columns:
                    continue
                vals = df_day[factor].values.astype(float)
                mask_valid = ~np.isnan(vals)

                if mask_valid.sum() < 20:
                    continue

                # 去极值标准化
                vals_clipped = np.clip(
                    vals,
                    np.nanpercentile(vals, 1),
                    np.nanpercentile(vals, 99),
                )
                mean_v = np.nanmean(vals_clipped)
                std_v = np.nanstd(vals_clipped)
                z = (vals_clipped - mean_v) / std_v if std_v > 0 else np.zeros_like(vals_clipped)
                z[~mask_valid] = 0
                scores += w * z
                factors_used.append(factor)

            if len(factors_used) < 3:
                continue

            df_result = pd.DataFrame({
                "trade_date": date,
                "ts_code": df_day["ts_code"].values,
                "composite_score_mvo": scores,
                "n_factors_mvo": len(factors_used),
            })
            composite_list.append(df_result)

        if not composite_list:
            print("[MVO] 无结果")
            return pd.DataFrame()

        df_composite = pd.concat(composite_list, ignore_index=True)
        print(f"[MVO] 复合因子: {len(df_composite)} 条, "
              f"调仓{len(rebalance_dates)}次")
        print(f"[MVO] 调仓日: {[d.date() for d in rebalance_dates[:5]]}...")

        save_path = os.path.join(DATA_FACTORS, "composite_mvo.parquet")
        df_composite.to_parquet(save_path, index=False)

        # 保存权重历史
        weights_df = pd.DataFrame(rebalance_dates, columns=["rebalance_date"])
        weights_df["factor_weights"] = [current_weights] * len(weights_df)
        weights_path = os.path.join(DATA_FACTORS, "mvo_weights_history.parquet")
        weights_df.to_parquet(weights_path, index=False)

        return df_composite

    # ==============================
    # 方案3: 等权基准
    # ==============================

    def combine_equal_weight(self) -> pd.DataFrame:
        """
        等权复合因子 (基准对照)
        """
        print(f"\n{'='*60}")
        print(f"方案3: 等权复合因子 (基准)")
        print(f"{'='*60}")

        dates = sorted(self.factor_panel["trade_date"].unique())
        dates = [d for d in dates if str(d.date()) >= BACKTEST_START]

        composite_list = []

        for date in dates:
            mask = self.factor_panel["trade_date"] == date
            df_day = self.factor_panel[mask].copy()
            if df_day.empty:
                continue

            scores = np.zeros(len(df_day))
            n_valid = 0

            for factor in self.factor_cols:
                if factor not in df_day.columns:
                    continue
                vals = df_day[factor].values.astype(float)
                mask_valid = ~np.isnan(vals)
                if mask_valid.sum() < 20:
                    continue

                vals_clipped = np.clip(
                    vals,
                    np.nanpercentile(vals, 1),
                    np.nanpercentile(vals, 99),
                )
                mean_v = np.nanmean(vals_clipped)
                std_v = np.nanstd(vals_clipped)
                z = (vals_clipped - mean_v) / std_v if std_v > 0 else np.zeros_like(vals_clipped)
                z[~mask_valid] = 0
                scores += z
                n_valid += 1

            if n_valid < 3:
                continue

            df_result = pd.DataFrame({
                "trade_date": date,
                "ts_code": df_day["ts_code"].values,
                "composite_score_eq": scores / n_valid,
                "n_factors_eq": n_valid,
            })
            composite_list.append(df_result)

        df_composite = pd.concat(composite_list, ignore_index=True)
        print(f"[等权] 复合因子: {len(df_composite)} 条")

        save_path = os.path.join(DATA_FACTORS, "composite_equal.parquet")
        df_composite.to_parquet(save_path, index=False)

        return df_composite

    # ==============================
    # 复合因子对比回测
    # ==============================

    def backtest_composites(self):
        """
        对比三种复合因子的分层回测表现
        """
        print(f"\n{'='*60}")
        print(f"复合因子回测对比")
        print(f"{'='*60}")

        results = []

        for name, file in [("EWMA-IC", "composite_ewma"),
                            ("MVO最优", "composite_mvo"),
                            ("等权基准", "composite_equal")]:
            path = os.path.join(DATA_FACTORS, f"{file}.parquet")
            if not os.path.exists(path):
                print(f"[{name}] 文件不存在, 跳过")
                continue

            df = pd.read_parquet(path)
            score_col = [c for c in df.columns if c.startswith("composite")][0]
            print(f"\n[{name}] 回测...")

            # 排序选股: 每组得分最高的20%
            dates = sorted(df["trade_date"].unique())
            daily_rets = []

            for date in dates:
                mask = df["trade_date"] == date
                day_df = df[mask].dropna(subset=[score_col]).copy()
                if len(day_df) < 100:
                    continue

                # 取前20%
                top_n = max(int(len(day_df) * 0.2), 20)
                top_codes = day_df.nlargest(top_n, score_col)["ts_code"].tolist()

                # 等权平均未来20日收益
                rets = []
                for code in top_codes:
                    r = self._get_future_return(code, date)
                    if pd.notna(r):
                        rets.append(r)
                if rets:
                    daily_rets.append(np.mean(rets))

            if not daily_rets:
                continue

            total_ret = np.prod([1 + r for r in daily_rets]) - 1
            ann_ret = (1 + total_ret) ** (252 / len(daily_rets)) - 1
            vol = np.std(daily_rets) * np.sqrt(252 / 20)
            sharpe = ann_ret / vol if vol > 0 else 0
            max_dd = np.min(
                np.minimum.accumulate(np.cumprod([1 + r for r in daily_rets]))
                / np.maximum.accumulate(np.cumprod([1 + r for r in daily_rets]))
                - 1
            )
            win_rate = np.mean([1 if r > 0 else 0 for r in daily_rets])

            results.append({
                "方案": name,
                "年化收益": ann_ret,
                "年化波动": vol,
                "夏普比率": sharpe,
                "最大回撤": max_dd,
                "胜率": win_rate,
                "交易次数": len(daily_rets),
            })

            print(f"  {name}: 年化={ann_ret:.2%}, 夏普={sharpe:.2f}, "
                  f"回撤={max_dd:.2%}, 胜率={win_rate:.1%}")

        df_result = pd.DataFrame(results)
        save_path = os.path.join(DATA_FACTORS, "composite_backtest.csv")
        df_result.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"\n[对比] 保存: {save_path}")
        return df_result

    def run_all(self):
        """运行全部合成方案"""
        self.calc_rolling_ic()
        self.combine_ewma_ic()
        self.combine_equal_weight()
        self.combine_mean_variance_optimal()
        self.backtest_composites()
        return self


if __name__ == "__main__":
    combiner = FactorCombiner()
    combiner.load_data()
    combiner.run_all()
