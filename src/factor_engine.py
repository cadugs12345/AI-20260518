"""
多因子迭代系统 - 因子自动测试流水线
功能：输入一个新因子候选，自动跑IC分析、分层回测、决定是否纳入因子库

用法示例:
```python
from factor_engine import FactorEngine
engine = FactorEngine()

# 测试一个新因子
result = engine.test_factor(data_df, factor_name="新因子名", factor_col="col_name")
# result 包含 IC序列、分层收益、建议
```
"""
import os, sys, time, gc
import numpy as np
import pandas as pd
from scipy import stats

class FactorEngine:
    """因子引擎 - 自动化因子测试、评估和管理"""
    
    def __init__(self, data_path="data/factors/factor_panel_v3.parquet",
                 price_path="data/factors/full_prices.parquet",
                 factor_list_path=None):
        self.data_path = data_path
        self.price_path = price_path
        self._load_data()
        self.factor_registry = {}  # 因子注册表 {name: metadata}
        self._load_registry(factor_list_path)
        
    def _load_data(self):
        """加载基础数据"""
        print("[FactorEngine] 加载数据...")
        self.panel = pd.read_parquet(self.data_path)
        self.prices = pd.read_parquet(self.price_path)
        self.panel["trade_date"] = pd.to_datetime(self.panel["trade_date"])
        self.prices["trade_date"] = pd.to_datetime(self.prices["trade_date"])
        
        # 确定已有因子列表
        skip = {"ts_code","trade_date","fwd_20d_ret","fwd_5d_ret"}
        self.existing_factors = [c for c in self.panel.columns 
                                 if c not in skip and self.panel[c].dtype in ("float64","int64")]
        print(f"  加载完成：{len(self.panel):,}行, {len(self.existing_factors)}个已有因子")
    
    def _load_registry(self, path=None):
        """加载因子注册表"""
        if path and os.path.exists(path):
            self.factor_registry = pd.read_parquet(path).to_dict('records')
    
    # ==================== 单因子测试 ====================
    
    def rank_ic(self, factor_values, forward_returns):
        """计算Rank IC（Spearman相关系数）"""
        mask = ~(np.isnan(factor_values) | np.isnan(forward_returns))
        if mask.sum() < 30:
            return np.nan, np.nan
        ic, pval = stats.spearmanr(factor_values[mask], forward_returns[mask])
        return ic, pval
    
    def test_factor(self, factor_name, factor_data=None, label_col="fwd_20d_ret",
                    verbose=True):
        """
        完整测试一个因子
        
        Parameters:
        -----------
        factor_name: str - 因子名称
        factor_data: pd.DataFrame - 必须包含[trade_date, ts_code, value, fwd_20d_ret]
        label_col: str - 收益标签列名
        
        Returns:
        --------
        dict - 测试结果
        """
        t0 = time.time()
        
        if factor_data is not None:
            # 外部传入的新因子
            data = factor_data.copy()
        elif factor_name in self.panel.columns:
            # 面板中已有的因子
            data = self.panel[["trade_date","ts_code",factor_name,label_col]].copy()
            data = data.rename(columns={factor_name: "value"})
        else:
            return {"error": f"因子 '{factor_name}' 未找到"}
        
        data = data.dropna(subset=["value", label_col])
        data = data[data[label_col].abs() < 0.5]  # 去极端值
        
        if len(data) < 1000:
            return {"error": f"有效样本不足: {len(data)}"}
        
        # 按日期分组计算IC
        ic_records = []
        for date, group in data.groupby("trade_date"):
            ic, pval = self.rank_ic(
                group["value"].values.astype(np.float64),
                group[label_col].values.astype(np.float64)
            )
            if not np.isnan(ic):
                ic_records.append({"trade_date": date, "IC": ic, "p_value": pval, "n": len(group)})
        
        ic_df = pd.DataFrame(ic_records)
        if len(ic_df) == 0:
            return {"error": "IC计算失败"}
        
        # IC统计
        ic_mean = ic_df["IC"].mean()
        ic_std = ic_df["IC"].std()
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0
        ic_positive_pct = (ic_df["IC"] > 0).mean()
        ic_sharpe = ic_mean / ic_std * np.sqrt(13)  # 年化IC_IR
        
        # IC序列的t检验（是否显著非零）
        t_stat, p_value_ttest = stats.ttest_1samp(ic_df["IC"], 0)
        
        # 分层测试
        decile_results = self._decile_test(data, label_col)
        
        # 方向性
        if ic_mean > 0:
            direction = "positive"  # 值越大越好 -> TOP买入
        else:
            direction = "negative"  # 值越小越好 -> BOTTOM买入
        
        result = {
            "factor_name": factor_name,
            "n_samples": len(data),
            "n_dates": len(ic_df),
            "IC_mean": ic_mean,
            "IC_std": ic_std,
            "IC_IR": ic_ir,
            "IC_sharpe_annualized": ic_sharpe,
            "IC_positive_pct": ic_positive_pct,
            "t_statistic": t_stat,
            "p_value_ttest": p_value_ttest,
            "direction": direction,
            "deciles": decile_results,
            "ic_series": ic_df,
            "time_seconds": time.time() - t0,
        }
        
        # 综合评分 (用于自动决策)
        result["score"] = self._compute_score(result)
        result["recommendation"] = self._recommend(result)
        
        if verbose:
            self._print_result(result)
        
        return result
    
    def _decile_test(self, data, label_col):
        """分层回测：按因子值分成10组，计算每组平均收益"""
        deciles = []
        for date, group in data.groupby("trade_date"):
            if len(group) < 100: continue
            group = group.sort_values("value")
            group["decile"] = pd.qcut(group["value"], 10, labels=False, duplicates='drop')
            for d in range(10):
                d_group = group[group["decile"] == d]
                if len(d_group) > 0:
                    deciles.append({"trade_date": date, "decile": d, 
                                    "ret": d_group[label_col].mean()})
        
        decile_df = pd.DataFrame(deciles)
        results = {}
        for d in range(10):
            dd = decile_df[decile_df["decile"] == d]
            if len(dd) > 0:
                results[f"decile_{d}"] = {
                    "mean_ret": dd["ret"].mean(),
                    "std": dd["ret"].std(),
                    "sharpe": dd["ret"].mean() / dd["ret"].std() * np.sqrt(13) if dd["ret"].std() > 0 else 0,
                }
        
        # 多空夏普 (Top - Bottom)
        if "decile_9" in results and "decile_0" in results:
            d9 = decile_df[decile_df["decile"] == 9]
            d0 = decile_df[decile_df["decile"] == 0]
            # 对齐日期
            merged = pd.merge(d9[["trade_date","ret"]], d0[["trade_date","ret"]], 
                              on="trade_date", suffixes=("_top", "_bottom"))
            ls_ret = merged["ret_top"] - merged["ret_bottom"]
            results["long_short"] = {
                "mean_ret": ls_ret.mean(),
                "std": ls_ret.std(),
                "sharpe": ls_ret.mean() / ls_ret.std() * np.sqrt(13) if ls_ret.std() > 0 else 0,
            }
        
        return results
    
    def _compute_score(self, result):
        """综合评分（0~100）"""
        score = 0
        
        # IC_IR > 0.3 加分
        ir = abs(result["IC_IR"])
        score += min(ir * 30, 30)
        
        # 统计显著性
        if result["p_value_ttest"] < 0.05:
            score += 20
        elif result["p_value_ttest"] < 0.10:
            score += 10
        
        # 正向比例
        pos_pct = max(result["IC_positive_pct"], 1 - result["IC_positive_pct"])
        score += min((pos_pct - 0.5) * 60, 20)
        
        # 分层单调性
        deciles = result.get("deciles", {})
        if "decile_9" in deciles and "decile_0" in deciles:
            ls = deciles.get("long_short", {})
            ls_sr = ls.get("sharpe", 0)
            score += min(abs(ls_sr) * 15, 20)
        
        # 样本量
        score += min(result["n_dates"] * 0.3, 10)
        
        return min(score, 100)
    
    def _recommend(self, result):
        """生成建议"""
        score = result["score"]
        if score >= 70:
            return "strong_accept"  # 强烈建议纳入
        elif score >= 50:
            return "conditional_accept"  # 有条件纳入（需要进一步分析）
        elif score >= 30:
            return "borderline"  # 边缘，可观察
        else:
            return "reject"  # 拒绝
    
    def _print_result(self, result):
        """打印测试结果"""
        print(f"\n{'='*50}")
        print(f"因子测试: {result['factor_name']}")
        print(f"{'='*50}")
        print(f"  样本: {result['n_samples']:,} ｜ {result['n_dates']}个交易日")
        print(f"  IC均值: {result['IC_mean']*100:+.3f}% ｜ IC_IR: {result['IC_IR']:.3f}")
        print(f"  年化IC夏普: {result['IC_sharpe_annualized']:.2f}")
        print(f"  正向占比: {result['IC_positive_pct']*100:.0f}%")
        print(f"  t检验p值: {result['p_value_ttest']:.4f}")
        print(f"  方向: {result['direction']}")
        print(f"  ----- 分层收益 (每20日) -----")
        deciles = result["deciles"]
        for d in range(10):
            if f"decile_{d}" in deciles:
                dd = deciles[f"decile_{d}"]
                print(f"    组{d}: 均值{dd['mean_ret']*100:+.2f}% 夏普{dd['sharpe']:.2f}")
        if "long_short" in deciles:
            ls = deciles["long_short"]
            print(f"    多空: 均值{ls['mean_ret']*100:+.2f}% 夏普{ls['sharpe']:.2f}")
        print(f"  ----- 评估 -----")
        print(f"  综合评分: {result['score']:.0f}/100")
        print(f"  建议: {result['recommendation']}")
        print(f"  用时: {result['time_seconds']:.1f}秒")
    
    # ==================== 批量测试 ====================
    
    def batch_test_factors(self, factor_list, label_col="fwd_20d_ret"):
        """
        批量测试多个因子
        
        factor_list: list of dict [{"name": "...", "data": DataFrame或None}]
        """
        results = []
        for f in factor_list:
            name = f.get("name", "unknown")
            data = f.get("data", None)
            result = self.test_factor(name, factor_data=data, label_col=label_col, verbose=False)
            results.append(result)
        
        summary = pd.DataFrame([{
            "因子": r["factor_name"],
            "IC均值": r["IC_mean"],
            "IC_IR": r["IC_IR"],
            "IC夏普": r["IC_sharpe_annualized"],
            "正向占比": r["IC_positive_pct"],
            "p值": r["p_value_ttest"],
            "评分": r["score"],
            "建议": r["recommendation"],
        } for r in results])
        
        return summary
    
    # ==================== 因子管理 ====================
    
    def register_factor(self, name, series_dir, ic_record=None):
        """注册一个因子到系统"""
        self.factor_registry[name] = {
            "name": name,
            "registered_date": pd.Timestamp.now(),
            "ic_record": ic_record,
            "active": True,
        }
    
    def ic_decay_monitor(self, lookback_windows=[60, 120, 240]):
        """监控因子IC衰减 - 比较不同窗口的IC"""
        pass
    
    # ==================== 新因子生成 ====================
    
    def add_custom_factor(self, factor_data, factor_name, label_col="fwd_20d_ret"):
        """
        添加一个新因子到面板并测试
        
        factor_data: DataFrame with [trade_date, ts_code, value]
        """
        # 先测试
        result = self.test_factor(factor_name, factor_data, label_col)
        
        # 如果建议接受，则合并到面板
        if result["recommendation"] in ["strong_accept", "conditional_accept"]:
            merge_data = factor_data.rename(columns={"value": factor_name})
            self.panel = self.panel.merge(
                merge_data[["trade_date","ts_code",factor_name]], 
                on=["trade_date","ts_code"], how="left"
            )
            self.existing_factors.append(factor_name)
            print(f"[FactorEngine] 因子 '{factor_name}' 已加入面板")
        else:
            print(f"[FactorEngine] 因子 '{factor_name}' 未被采纳（评分{result['score']:.0f}）")
        
        return result


# ==================== 工具函数 ====================

def build_derived_factors(base_factor, transformations=["rank", "zscore", "sign", "square"]):
    """生成衍生因子"""
    pass


if __name__ == "__main__":
    print("="*60)
    print("FactorEngine - 多因子迭代系统")
    print("="*60)
    
    engine = FactorEngine()
    
    # 示例：测试已有因子
    test_factor_name = "20日动量"
    result = engine.test_factor(test_factor_name)
    
    print(f"\n{'='*60}")
    print("因子引擎初始化完成，可继续测试其他因子")
    print(f"现有因子: {len(engine.existing_factors)}个")
