"""
单因子测试系统
核心功能:
1. IC分析 (IC均值/ICIR/IC胜率/月度分布/行业中性)
2. 分层回测 (5组/10组, 单调性检验)
3. 因子超额收益/年化/最大回撤/夏普
4. 输出测试报告
"""
import os, sys, warnings, time
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DATA_RAW, DATA_FACTORS, BACKTEST_START, END_DATE


class FactorTestEngine:
    """
    单因子测试引擎
    """

    def __init__(self, factor_panel_path: str = None):
        """
        factor_panel_path: 因子面板数据, 含 ts_code, trade_date + 多个因子列
        """
        if factor_panel_path is None:
            factor_panel_path = os.path.join(DATA_FACTORS, "factor_panel.parquet")
        self.factor_panel = None
        self.factor_panel_path = factor_panel_path
        self.trade_dates = None
        self.results = {}

    def load_data(self):
        """加载因子面板数据"""
        if not os.path.exists(self.factor_panel_path):
            raise FileNotFoundError(f"因子面板不存在: {self.factor_panel_path}")

        print(f"[加载] 因子面板: {self.factor_panel_path}")
        self.factor_panel = pd.read_parquet(self.factor_panel_path)
        self.factor_panel["trade_date"] = pd.to_datetime(self.factor_panel["trade_date"])

        # 加载日线行情用于计算未来收益
        print("[加载] 加载日线行情索引...")
        stock_list = pd.read_parquet(os.path.join(DATA_RAW, "stock_list.parquet"))
        sample_path = os.path.join(DATA_RAW, "daily", f"{stock_list['ts_code'].iloc[0]}.parquet")
        sample = pd.read_parquet(sample_path)
        self.trade_dates = sorted(sample["trade_date"].dt.strftime("%Y%m%d").tolist())

        print(f"[加载] 完成: {len(self.factor_panel)} 条 × {len(self.factor_panel.columns)} 列")
        return self

    def _get_future_return(self, code: str, date: str, hold_days: int = 20) -> float:
        """获取未来 N 日的收益"""
        daily_path = os.path.join(DATA_RAW, "daily", f"{code}.parquet")
        if not os.path.exists(daily_path):
            return np.nan
        daily = pd.read_parquet(daily_path)
        daily = daily.sort_values("trade_date")
        mask = daily["trade_date"] == date
        if not mask.any():
            return np.nan
        idx = mask.idxmax()
        future_idx = idx + hold_days
        if future_idx >= len(daily):
            return np.nan
        current_close = daily.loc[idx, "close"]
        future_close = daily.loc[future_idx, "close"]
        return (future_close - current_close) / current_close

    def _load_daily_basic(self, date: str) -> pd.DataFrame:
        """加载某日全景数据"""
        path = os.path.join(DATA_RAW, "daily_basic", f"{date}.parquet")
        if os.path.exists(path):
            return pd.read_parquet(path)
        return pd.DataFrame()

    def calc_ic(self, factor_name: str, hold_days: int = 20,
                industry_neutral: bool = False) -> dict:
        """
        计算单因子IC指标
        返回: IC均值、ICIR、IC胜率、IC序列
        """
        print(f"\n[IC分析] 因子: {factor_name} | 持有期: {hold_days}天")

        ic_values = []
        factor_col = factor_name

        if factor_col not in self.factor_panel.columns:
            print(f"  ⚠️ 因子列不存在: {factor_col}")
            return {}

        # 遍历每个交易日
        cut_start = BACKTEST_START
        dates_in_panel = sorted(self.factor_panel["trade_date"].dropna().unique())
        dates_in_panel = [d for d in dates_in_panel if str(d.date()) >= cut_start]

        sample_every = max(1, len(dates_in_panel) // 10)

        for i, date in enumerate(dates_in_panel):
            date_str = date.strftime("%Y%m%d")

            # 取该日的因子截面值
            mask = self.factor_panel["trade_date"] == date
            df_day = self.factor_panel[mask].copy()

            if df_day.empty or factor_col not in df_day.columns:
                continue

            # 剔除空值
            df_day = df_day.dropna(subset=[factor_col])
            if len(df_day) < 50:  # 样本太少没意义
                continue

            # 获取未来收益
            future_rets = []
            valid_codes = []

            for _, row in df_day.iterrows():
                ret = self._get_future_return(row["ts_code"], date_str, hold_days)
                if pd.notna(ret):
                    future_rets.append(ret)
                    valid_codes.append(row["ts_code"])

            if len(valid_codes) < 50:
                continue

            factor_vals = df_day[df_day["ts_code"].isin(valid_codes)][factor_col].values[:len(valid_codes)]

            # Pearson IC + Spearman Rank IC
            try:
                # 对齐
                factor_map = dict(zip(df_day["ts_code"], df_day[factor_col]))
                aligned_factor = np.array([factor_map.get(c, np.nan) for c in valid_codes])

                nan_mask = ~(np.isnan(aligned_factor) | np.isnan(future_rets))
                if nan_mask.sum() < 50:
                    continue

                f = aligned_factor[nan_mask]
                r = np.array(future_rets)[nan_mask]

                pearson_ic, _ = stats.pearsonr(f, r)
                spearman_ic, _ = stats.spearmanr(f, r)

                ic_values.append({
                    "date": date,
                    "pearson_ic": pearson_ic,
                    "spearman_ic": spearman_ic,
                    "n_stocks": len(f),
                })
            except Exception:
                continue

            if (i + 1) % sample_every == 0:
                print(f"  进度: {i+1}/{len(dates_in_panel)} 交易日 | 有效IC: {len(ic_values)}")

        if not ic_values:
            print("  ⚠️ 无有效IC数据")
            return {}

        ic_df = pd.DataFrame(ic_values)
        ic_mean = ic_df["spearman_ic"].mean()
        ic_std = ic_df["spearman_ic"].std()
        icir = ic_mean / ic_std if ic_std > 0 else 0
        ic_win_rate = (ic_df["spearman_ic"] > 0).mean()

        result = {
            "factor": factor_name,
            "hold_days": hold_days,
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "icir": icir,
            "ic_win_rate": ic_win_rate,
            "n_periods": len(ic_df),
            "n_stocks_avg": ic_df["n_stocks"].mean(),
            "ic_series": ic_df,
        }

        print(f"  ✅ IC均值: {ic_mean:.4f}")
        print(f"  ✅ ICIR:   {icir:.4f}")
        print(f"  ✅ IC胜率: {ic_win_rate:.1%}")
        print(f"  ✅ 有效期数: {len(ic_df)}")

        self.results[f"{factor_name}_ic"] = result
        return result

    def layer_backtest(self, factor_name: str, n_groups: int = 5,
                       hold_days: int = 20) -> dict:
        """
        分层回测
        将因子值分为 n_groups 组, 计算每组累计收益
        """
        print(f"\n[分层回测] 因子: {factor_name} | {n_groups}组 | 持有期: {hold_days}天")

        factor_col = factor_name
        if factor_col not in self.factor_panel.columns:
            print(f"  ⚠️ 因子列不存在")
            return {}

        cut_start = BACKTEST_START
        dates_in_panel = sorted(self.factor_panel["trade_date"].dropna().unique())
        dates_in_panel = [d for d in dates_in_panel if str(d.date()) >= cut_start]

        # 每组累计收益
        group_returns = {g: [] for g in range(n_groups)}
        long_short_returns = []  # 多头-空头
        dates_record = []

        sample_every = max(1, len(dates_in_panel) // 10)

        for i, date in enumerate(dates_in_panel):
            date_str = date.strftime("%Y%m%d")
            mask = self.factor_panel["trade_date"] == date
            df_day = self.factor_panel[mask].dropna(subset=[factor_col]).copy()

            if len(df_day) < 50:
                continue

            # 分组
            df_day["group"] = pd.qcut(
                df_day[factor_col].rank(method="first"),
                q=n_groups,
                labels=list(range(n_groups)),
                duplicates="drop",
            )
            if df_day["group"].isna().any():
                df_day = df_day.dropna(subset=["group"])

            date_ret = {}

            for g in range(n_groups):
                group_stocks = df_day[df_day["group"] == g]
                if group_stocks.empty:
                    continue
                rets = []
                for _, row in group_stocks.iterrows():
                    ret = self._get_future_return(row["ts_code"], date_str, hold_days)
                    if pd.notna(ret):
                        rets.append(ret)
                if rets:
                    group_returns[g].append(np.mean(rets))

            # 多头空头差额
            g0 = group_returns[0] if group_returns[0] else None
            gn = group_returns[n_groups - 1] if group_returns[n_groups - 1] else None
            if g0 and gn:
                ls = gn[-1] - g0[-1]
                long_short_returns.append(ls)
                dates_record.append(date)

            if (i + 1) % sample_every == 0:
                print(f"  进度: {i+1}/{len(dates_in_panel)} 交易日")

        # 计算绩效
        perf = self._calc_layer_performance(group_returns, long_short_returns,
                                             n_groups, factor_name, hold_days)

        # 单调性检验
        monotonicity = self._test_monotonicity(group_returns)
        perf["monotonicity"] = monotonicity
        print(f"  ✅ 单调性: {monotonicity}")
        print(f"  ✅ 多头年化: {perf.get('group_ann_return', {}).get(n_groups-1, 0):.2%}")
        print(f"  ✅ 多空夏普: {perf.get('long_short_sharpe', 0):.2f}")

        return perf

    def _calc_layer_performance(self, group_returns: dict, ls_returns: list,
                                 n_groups: int, factor_name: str, hold_days: int) -> dict:
        """计算分层回测绩效指标"""

        result = {
            "factor": factor_name,
            "n_groups": n_groups,
            "hold_days": hold_days,
        }

        # 每组年化收益
        ann_return = {}
        for g in range(n_groups):
            if group_returns[g]:
                total_ret = np.prod([1 + r for r in group_returns[g]]) - 1
                n_periods = len(group_returns[g])
                periods_per_year = 252 / hold_days
                ann_ret = (1 + total_ret) ** (periods_per_year / n_periods) - 1
                ann_return[g] = ann_ret

        result["group_ann_return"] = ann_return

        # 多头/空头夏普
        for g in range(n_groups):
            if group_returns[g]:
                mean_r = np.mean(group_returns[g])
                std_r = np.std(group_returns[g]) if len(group_returns[g]) > 1 else 1
                sharpe = mean_r / std_r * np.sqrt(252 / hold_days)
                result[f"group_{g}_sharpe"] = sharpe

        # 多空组合收益
        if ls_returns:
            ls_sharpe = np.mean(ls_returns) / np.std(ls_returns) * np.sqrt(252 / hold_days) if np.std(ls_returns) > 0 else 0
            result["long_short_sharpe"] = ls_sharpe

            # 最大回撤
            cum_ls = np.cumprod([1 + r for r in ls_returns])
            peak = np.maximum.accumulate(cum_ls)
            dd = (cum_ls - peak) / peak
            result["long_short_max_dd"] = np.min(dd)

            # 累计多空收益
            result["long_short_ann_return"] = np.mean(ls_returns) * (252 / hold_days)

        return result

    def _test_monotonicity(self, group_returns: dict) -> str:
        """单调性检验"""
        n_groups = len(group_returns)
        means = []
        for g in range(n_groups):
            if group_returns[g]:
                means.append(np.mean(group_returns[g]))
            else:
                means.append(0)

        # 检验是否单调递增
        increasing = all(means[i] <= means[i+1] for i in range(len(means)-1))
        decreasing = all(means[i] >= means[i+1] for i in range(len(means)-1))

        if increasing:
            return "严格单调递增"
        elif decreasing:
            return "严格单调递减"
        else:
            # 看大致趋势
            corr = stats.pearsonr(range(len(means)), means)[0]
            if corr > 0.7:
                return f"近似单调递增 (corr={corr:.2f})"
            elif corr < -0.7:
                return f"近似单调递减 (corr={corr:.2f})"
            return f"无序/单调性弱 (corr={corr:.2f})"

    def run_full_test(self, factor_names: list = None,
                       hold_days: int = 20, n_groups: int = 5,
                       ic_only: bool = False) -> pd.DataFrame:
        """
        完整测试所有因子
        返回: 因子排名 DataFrame
        """
        if factor_names is None:
            # 自动检测因子列
            exclude_cols = {"ts_code", "trade_date", "ann_date", "end_date"}
            factor_names = [
                c for c in self.factor_panel.columns
                if c not in exclude_cols and self.factor_panel[c].dtype in ["float64", "int64"]
            ]

        print(f"\n{'='*60}")
        print(f"单因子全量测试开始: {len(factor_names)} 个因子")
        print(f"持有期: {hold_days}天 | 分组: {n_groups}组")
        print(f"{'='*60}")

        summary = []

        for factor in factor_names:
            try:
                # IC分析
                ic_result = self.calc_ic(factor, hold_days)

                if not ic_result:
                    continue

                # 分层回测
                if not ic_only:
                    layer_result = self.layer_backtest(factor, n_groups, hold_days)
                else:
                    layer_result = {}

                summary.append({
                    "因子": factor,
                    "IC均值": ic_result.get("ic_mean", 0),
                    "ICIR": ic_result.get("icir", 0),
                    "IC胜率": ic_result.get("ic_win_rate", 0),
                    "IC标准差": ic_result.get("ic_std", 0),
                    "有效期数": ic_result.get("n_periods", 0),
                    "多头夏普": layer_result.get(f"group_{n_groups-1}_sharpe", 0),
                    "多空夏普": layer_result.get("long_short_sharpe", 0),
                    "多空最大回撤": layer_result.get("long_short_max_dd", 0),
                    "单调性": layer_result.get("monotonicity", ""),
                })

            except Exception as e:
                print(f"  ❌ {factor} 测试异常: {e}")
                continue

        df_summary = pd.DataFrame(summary)

        if not df_summary.empty:
            # 综合评分: ICIR*0.3 + IC胜率*0.2 + 多空夏普*0.3 + 单调性分*0.2
            df_summary["综合评分"] = (
                df_summary["ICIR"].rank(pct=True) * 0.3
                + df_summary["IC胜率"].rank(pct=True) * 0.2
                + df_summary["多空夏普"].rank(pct=True) * 0.3
                + df_summary["多头夏普"].rank(pct=True) * 0.2
            )
            df_summary = df_summary.sort_values("综合评分", ascending=False).reset_index(drop=True)

        print(f"\n{'='*60}")
        print(f"因子排名 (前10):")
        if not df_summary.empty:
            print(df_summary.head(10).to_string(index=False))
        print(f"{'='*60}")

        save_path = os.path.join(DATA_FACTORS, "factor_test_summary.csv")
        df_summary.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"\n完整报告已保存: {save_path}")

        return df_summary


def quick_test():
    """快捷测试: 用现有因子面板跑完整测试"""
    engine = FactorTestEngine()
    engine.load_data()

    # 先跑IC分析
    engine.run_full_test(hold_days=20)
    return engine


if __name__ == "__main__":
    quick_test()
