"""Single-factor validation focused on predictive, stability, and redundancy tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from quant_shared.contracts import FactorResearchReport, PanelData


# 中文说明：定义 `ResearchConfig`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class ResearchConfig:
    quantiles: int = 5
    annualization: float = 252.0
    min_cross_section: int = 20


# 中文说明：定义 `FactorResearchEngine`，封装本模块对应的数据、配置与行为。
class FactorResearchEngine:
    """A data-source-independent replacement for the legacy analyzer core."""

    # 中文说明：`__init__`：初始化对象及其运行依赖。
    def __init__(self, config: ResearchConfig | None = None) -> None:
        self.config = config or ResearchConfig()

    # 中文说明：`analyze`：分析输入并生成诊断结果。
    def analyze(self, data: PanelData) -> FactorResearchReport:
        data.validate()
        ic = self.information_coefficients(data.factors, data.forward_returns)
        quantile_returns = {
            factor: self.quantile_returns(data.factors[factor], data.forward_returns)
            for factor in data.factors.columns
        }
        summary = self._summary(ic, quantile_returns, data.factors)
        correlations = self.average_cross_sectional_correlation(data.factors)
        return FactorResearchReport(
            summary=summary,
            ic=ic,
            quantile_returns=quantile_returns,  # {factor: groupret_df}
            factor_correlations=correlations,
            diagnostics={
                "n_dates": len(data.dates),
                "n_assets": len(data.assets),
                "n_factors": data.factors.shape[1],
                "label_alignment": "explicit_on_signal_date",
            },
        )

    # 中文说明：`information_coefficients`：执行该名称对应的业务计算，并返回调用方所需结果。
    def information_coefficients(
        self, factors: pd.DataFrame, forward_returns: pd.Series
    ) -> pd.DataFrame:
        rows: list[pd.Series] = []
        for date, cross_section in factors.groupby(level=0, sort=True):
            y = forward_returns.xs(date, level=0)
            x = cross_section.droplevel(0)
            values: dict[str, float] = {}
            for factor in factors.columns:
                valid = x[factor].notna() & y.notna()
                if valid.sum() < self.config.min_cross_section:
                    values[factor] = np.nan
                else:
                    values[factor] = float(spearmanr(x.loc[valid, factor], y.loc[valid]).statistic)
            rows.append(pd.Series(values, name=pd.Timestamp(date)))
        return pd.DataFrame(rows)

    # 中文说明：`quantile_returns`：执行该名称对应的业务计算，并返回调用方所需结果。
    def quantile_returns(self, factor: pd.Series, returns: pd.Series) -> pd.DataFrame:
        records: list[pd.Series] = []
        for date, signal in factor.groupby(level=0, sort=True):
            x = signal.droplevel(0)
            y = returns.xs(date, level=0)
            valid = x.notna() & y.notna()
            if valid.sum() < self.config.quantiles * 4:
                continue
            ranks = x.loc[valid].rank(method="first")
            buckets = pd.qcut(ranks, self.config.quantiles, labels=False) + 1
            bucket_return = y.loc[valid].groupby(buckets).mean()
            bucket_return.name = pd.Timestamp(date)
            records.append(bucket_return)
        result = pd.DataFrame(records).sort_index()
        result.columns = [f"Q{int(column)}" for column in result.columns]
        if not result.empty:
            result["long_short"] = result.iloc[:, -1] - result.iloc[:, 0]
        return result

    # 中文说明：`average_cross_sectional_correlation`：执行该名称对应的业务计算，并返回调用方所需结果。
    @staticmethod
    def average_cross_sectional_correlation(factors: pd.DataFrame) -> pd.DataFrame:
        matrices = [
            frame.droplevel(0).corr(method="spearman").to_numpy(float)
            for _, frame in factors.groupby(level=0, sort=False)
        ]
        average = np.nanmean(matrices, axis=0)
        return pd.DataFrame(average, index=factors.columns, columns=factors.columns)

    # 中文说明：`_summary`：内部辅助步骤，不作为稳定公共接口。
    def _summary(
        self,
        ic: pd.DataFrame,
        quantile_returns: dict[str, pd.DataFrame],
        factors: pd.DataFrame,
    ) -> pd.DataFrame:
        rows: dict[str, dict[str, float]] = {}
        for factor in factors.columns:
            series = ic[factor].dropna()
            spread = quantile_returns[factor].get("long_short", pd.Series(dtype=float)).dropna()
            ic_std = series.std(ddof=1)
            ann_return = spread.mean() * self.config.annualization if len(spread) else np.nan
            ann_vol = spread.std(ddof=1) * np.sqrt(self.config.annualization)
            rows[factor] = {
                "ic_mean": series.mean(),
                "ic_std": ic_std,
                "icir": series.mean() / ic_std if ic_std > 0 else np.nan,
                "ic_positive_ratio": (series > 0).mean(),
                "long_short_ann_return": ann_return,
                "long_short_ann_vol": ann_vol,
                "long_short_sharpe": ann_return / ann_vol if ann_vol > 0 else np.nan,
                "coverage": factors[factor].notna().mean(),
            }
        return pd.DataFrame.from_dict(rows, orient="index")
