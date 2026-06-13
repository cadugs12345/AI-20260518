"""
因子中性化 & 正交化处理
在多因子合成之前必须做，消除因子间的冗余暴露。

流程:
1. 逐个因子对市值/行业做回归，取残差 (市值中性 + 行业中性)
2. 对中性化后的因子做正交化 (Schmidt正交 / PCA去相关)
3. 输出处理后的因子面板
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DATA_FACTORS, DATA_RAW


class FactorNeutralizer:
    """
    因子中性化 & 正交化
    """

    def __init__(self):
        self.factor_panel = None
        self.factor_cols = []
        self.industry_dummies = None  # 行业虚拟变量
        self.ortho_matrix = None      # 正交矩阵

    def load_data(self, panel_path: str = None):
        """加载因子面板"""
        if panel_path is None:
            panel_path = os.path.join(DATA_FACTORS, "factor_panel.parquet")
        self.factor_panel = pd.read_parquet(panel_path)
        self.factor_panel["trade_date"] = pd.to_datetime(self.factor_panel["trade_date"])

        exclude = {"ts_code", "trade_date", "ann_date", "end_date"}
        self.factor_cols = [c for c in self.factor_panel.columns if c not in exclude]
        print(f"[中性化] 加载面板: {len(self.factor_panel)} 条, {len(self.factor_cols)} 个原始因子")
        return self

    def load_industry(self):
        """加载行业分类"""
        stock_path = os.path.join(DATA_RAW, "stock_list.parquet")
        stocks = pd.read_parquet(stock_path)[["ts_code", "industry"]]
        self.industry_map = dict(zip(stocks["ts_code"], stocks["industry"]))
        print(f"[中性化] 行业分类加载: {len(self.industry_map)} 只股票, "
              f"{len(set(self.industry_map.values()))} 个行业")

    # ==============================
    # 市值中性 + 行业中性 (回归残差法)
    # ==============================

    def neutralize_factors(self) -> pd.DataFrame:
        """
        对每个因子做市值/行业中性的回归残差

        对每个交易日 t, 对每个因子 f:
        f_i = α + β1 * log_mktcap_i + Σ β_k * industry_k + ε_i
        中性化后的因子 = ε_i (残差)

        如果日全景数据未就绪, 用流通市值或总市值的对数值代替
        """
        self.load_industry()

        # 先检查是否已有 daily_basic (需要市值数据)
        daily_basic_dir = os.path.join(DATA_RAW, "daily_basic")
        has_market_cap = False
        if os.path.exists(daily_basic_dir):
            files = os.listdir(daily_basic_dir)
            if files:
                has_market_cap = True
                first_basic = pd.read_parquet(os.path.join(daily_basic_dir, files[0]))
                cap_map = dict(zip(first_basic["ts_code"],
                                   np.log(first_basic["total_mv"].replace(0, np.nan))))

        print(f"\n[中性化] 市值数据就绪: {has_market_cap}")

        dates = sorted(self.factor_panel["trade_date"].unique())
        neutralized_list = []

        for date in dates:
            mask = self.factor_panel["trade_date"] == date
            df_day = self.factor_panel[mask].copy()
            if df_day.empty:
                continue

            # 获取该日市值
            date_str = date.strftime("%Y%m%d")
            if has_market_cap:
                basic_path = os.path.join(daily_basic_dir, f"{date_str}.parquet")
                if os.path.exists(basic_path):
                    basic = pd.read_parquet(basic_path)
                    cap_map = dict(zip(basic["ts_code"],
                                       np.log(basic["total_mv"].replace(0, np.nan))))

            # 为每只股票构建回归特征: [log_mktcap, industry_dummies]
            codes = df_day["ts_code"].tolist()
            industries = [self.industry_map.get(c, "未知") for c in codes]

            # 独热编码行业
            all_industries = sorted(set(industries))
            ind_df = pd.get_dummies(pd.Series(industries), prefix="ind")
            # 对齐行业列
            for col in ind_df.columns:
                ind_df[col] = ind_df[col].astype(float)

            # 市值
            log_cap = pd.Series([cap_map.get(c, np.nan) for c in codes], index=df_day.index)
            log_cap_scaled = (log_cap - log_cap.mean()) / log_cap.std()  # 标准化

            # 回归特征
            X_features = pd.concat([log_cap_scaled, ind_df], axis=1).values

            for factor in self.factor_cols:
                if factor not in df_day.columns:
                    continue
                y = df_day[factor].values.astype(float)

                mask_y = ~(np.isnan(y) | np.isnan(X_features).any(axis=1))
                if mask_y.sum() < 30:
                    df_day[factor] = np.nan
                    continue

                y_clean = y[mask_y]
                X_clean = X_features[mask_y]

                try:
                    model = LinearRegression()
                    model.fit(X_clean, y_clean)
                    residuals = y - model.predict(X_features)
                    residuals[~mask_y] = np.nan
                    df_day[factor] = residuals
                except Exception:
                    pass

            neutralized_list.append(df_day)

            if len(neutralized_list) % 200 == 0:
                print(f"  [中性化] {len(neutralized_list)}/{len(dates)} 交易日")

        df_neutral = pd.concat(neutralized_list, ignore_index=True)
        print(f"[中性化] 完成: {len(df_neutral)} 条")

        save_path = os.path.join(DATA_FACTORS, "factor_panel_neutral.parquet")
        df_neutral.to_parquet(save_path, index=False)
        print(f"[中性化] 保存: {save_path}")

        self.factor_panel = df_neutral
        return df_neutral

    # ==============================
    # 施密特正交化 (Gram-Schmidt)
    # ==============================

    def orthogonalize_schmidt(self, factor_order: list = None) -> pd.DataFrame:
        """
        用 Gram-Schmidt 过程对因子做正交化

        factor_order: 因子顺序 (按优先级依次正交)
        若为 None, 按IC排序
        """
        print(f"\n[正交化] Gram-Schmidt 正交化...")

        if factor_order is None:
            # 按因子方差大小排序 (方差大的优先保留)
            factor_order = sorted(self.factor_cols,
                                  key=lambda f: self.factor_panel[f].std(),
                                  reverse=True)

        # 对每个交易日单独正交
        dates = sorted(self.factor_panel["trade_date"].unique())
        orthogonal_list = []

        for date in dates:
            mask = self.factor_panel["trade_date"] == date
            df_day = self.factor_panel[mask].copy()
            if df_day.empty:
                continue

            # 取因子值矩阵
            F = df_day[factor_order].values.astype(float)
            n_codes, n_factors = F.shape

            # Gram-Schmidt
            Q = np.zeros_like(F)
            for j in range(n_factors):
                v = F[:, j].copy()
                for k in range(j):
                    # 投影到已正交的向量上
                    v_k = Q[:, k]
                    # 只对共同非NAN的部分计算投影
                    mask_common = ~(np.isnan(v) | np.isnan(v_k))
                    if mask_common.sum() < 10:
                        continue
                    v_clean = v[mask_common]
                    vk_clean = v_k[mask_common]
                    if np.std(vk_clean) > 0:
                        proj = np.dot(vk_clean, v_clean) / np.dot(vk_clean, vk_clean)
                        v[mask_common] -= proj * vk_clean
                Q[:, j] = v

            # 替换因子值
            for j, factor in enumerate(factor_order):
                df_day[factor] = Q[:, j]
                # 标准化
                series = df_day[factor]
                mask_s = ~np.isnan(series)
                if mask_s.sum() > 10:
                    s_mean = series[mask_s].mean()
                    s_std = series[mask_s].std()
                    if s_std > 0:
                        df_day.loc[mask_s, factor] = (series[mask_s] - s_mean) / s_std

            orthogonal_list.append(df_day)

            if len(orthogonal_list) % 200 == 0:
                print(f"  [正交化] {len(orthogonal_list)}/{len(dates)} 交易日")

        df_ortho = pd.concat(orthogonal_list, ignore_index=True)
        print(f"[正交化] 完成: {len(df_ortho)} 条")

        save_path = os.path.join(DATA_FACTORS, "factor_panel_ortho.parquet")
        df_ortho.to_parquet(save_path, index=False)
        print(f"[正交化] 保存: {save_path}")

        return df_ortho

    # ==============================
    # PCA去相关 (另一种正交化方法)
    # ==============================

    def orthogonalize_pca(self, n_components: int = None) -> pd.DataFrame:
        """
        用PCA主成分做因子去相关

        保留所有主成分但只去相关, 不降维
        输出: 各主成分暴露 (荷载)
        """
        from sklearn.decomposition import PCA

        print(f"\n[正交化-PCA] PCA去相关...")

        dates = sorted(self.factor_panel["trade_date"].unique())
        pca_list = []

        for date in dates:
            mask = self.factor_panel["trade_date"] == date
            df_day = self.factor_panel[mask].copy()
            if df_day.empty:
                continue

            # 取因子值
            F = df_day[self.factor_cols].values.astype(float)

            # 逐行删除NaN (每只股票缺失不同)
            # 用均值填充 (因为前面已经中性化了, 均值接近0)
            col_means = np.nanmean(F, axis=0)
            F_filled = np.where(np.isnan(F), col_means, F)

            if n_components is None:
                n_components = len(self.factor_cols)
            n_components = min(n_components, len(self.factor_cols), F_filled.shape[0])

            try:
                pca = PCA(n_components=n_components)
                components = pca.fit_transform(F_filled)

                # 每个主成分是一个正交的新"因子"
                for j in range(n_components):
                    pc_name = f"PC{j+1}"
                    df_day[pc_name] = components[:, j]
                    # 标准化
                    mask_s = ~np.isnan(df_day[pc_name])
                    if mask_s.sum() > 10:
                        s = df_day[pc_name][mask_s]
                        df_day.loc[mask_s, pc_name] = (s - s.mean()) / s.std()

                # 删除原始因子列, 保留主成分
                for col in self.factor_cols:
                    df_day.drop(columns=[col], inplace=True, errors="ignore")

            except Exception:
                pass

            pca_list.append(df_day)

            if len(pca_list) % 200 == 0:
                print(f"  [PCA] {len(pca_list)}/{len(dates)} 交易日")

        df_pca = pd.concat(pca_list, ignore_index=True)
        print(f"[PCA] 完成: {len(df_pca)} 条, "
              f"主成分数: {n_components}")

        save_path = os.path.join(DATA_FACTORS, "factor_panel_pca.parquet")
        df_pca.to_parquet(save_path, index=False)
        print(f"[PCA] 保存: {save_path}")

        return df_pca

    def run_all(self):
        """全流程: 中性化 → 正交化"""
        print("=" * 60)
        print("因子预处理全流程")
        print("=" * 60)

        # 1. 市值/行业中性化
        self.neutralize_factors()

        # 2. Gram-Schmidt正交化
        self.orthogonalize_schmidt()

        # 3. PCA去相关 (可选)
        self.orthogonalize_pca(n_components=len(self.factor_cols))

        print("[预处理] 全部完成!")
        print(f"  中性化: data/factors/factor_panel_neutral.parquet")
        print(f"  GS正交: data/factors/factor_panel_ortho.parquet")
        print(f"  PCA:    data/factors/factor_panel_pca.parquet")


if __name__ == "__main__":
    neutralizer = FactorNeutralizer()
    neutralizer.load_data()
    neutralizer.run_all()
