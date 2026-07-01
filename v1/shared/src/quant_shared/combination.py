"""Factor-value layer: redundancy control and stable composite construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import CompositeConfig
from .math_utils import normalize_weights


# 中文说明：`information_coefficients`：执行该名称对应的业务计算，并返回调用方所需结果。
def information_coefficients(
    factors: pd.DataFrame,
    forward_returns: pd.Series,
    min_cross_section: int = 20,
) -> pd.DataFrame:
    """Compute laggable model IC history without invoking research diagnostics."""

    rows: list[pd.Series] = []
    for date, cross_section in factors.groupby(level=0, sort=True):
        y = forward_returns.xs(date, level=0)
        x = cross_section.droplevel(0)
        values: dict[str, float] = {}
        for factor in factors.columns:
            valid = x[factor].notna() & y.notna()
            values[factor] = (
                float(spearmanr(x.loc[valid, factor], y.loc[valid]).statistic)
                if valid.sum() >= min_cross_section
                else np.nan
            )
        rows.append(pd.Series(values, name=pd.Timestamp(date)))
    return pd.DataFrame(rows)


# 中文说明：定义 `FactorCombiner`，封装本模块对应的数据、配置与行为。
class FactorCombiner:
    # 中文说明：`__init__`：初始化对象及其运行依赖。
    def __init__(self, config: CompositeConfig | None = None) -> None:
        self.config = config or CompositeConfig()

    # 中文说明：`combine`：组合多个输入并生成统一结果。
    def combine(
        self,
        factors: pd.DataFrame,
        ic: pd.DataFrame | None = None,
    ) -> tuple[pd.Series, pd.DataFrame]:
        if self.config.method == "equal":
            weights = pd.DataFrame(
                1.0 / factors.shape[1],
                index=pd.Index(factors.index.get_level_values(0).unique()),
                columns=factors.columns,
            )
        elif self.config.method == "icir":
            if ic is None:
                raise ValueError("IC history is required for ICIR weighting")
            weights = self._rolling_icir_weights(ic.reindex(columns=factors.columns))
        else:
            raise ValueError(f"unsupported combination method: {self.config.method}")
        score = self._apply_weights(factors, weights)
        return score.rename("composite_score"), weights

    # 中文说明：`_rolling_icir_weights`：内部辅助步骤，不作为稳定公共接口。
    def _rolling_icir_weights(self, ic: pd.DataFrame) -> pd.DataFrame:
        rolling_mean = ic.rolling(
            self.config.lookback, min_periods=self.config.min_periods
        ).mean()
        rolling_std = ic.rolling(
            self.config.lookback, min_periods=self.config.min_periods
        ).std(ddof=1)
        raw = rolling_mean.div(rolling_std.replace(0.0, np.nan))
        raw = (1.0 - self.config.ic_shrinkage) * raw
        if not self.config.allow_negative:
            raw = raw.clip(lower=0.0)
        rows = [
            normalize_weights(
                row.to_numpy(float),
                allow_negative=self.config.allow_negative,
                max_weight=self.config.max_factor_weight,
            )
            for _, row in raw.iterrows()
        ]
        weights = pd.DataFrame(rows, index=raw.index, columns=raw.columns)
        weights = weights.ewm(alpha=1.0 - self.config.weight_smoothing, adjust=False).mean()
        # Today's signal may only use IC observations available before today.
        return weights.shift(1)

    # 中文说明：`_apply_weights`：内部辅助步骤，不作为稳定公共接口。
    @staticmethod
    def _apply_weights(factors: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
        dates = factors.index.get_level_values(0)
        aligned_weights = weights.reindex(dates).set_axis(factors.index)
        numerator = (factors * aligned_weights).sum(axis=1, min_count=1)
        denominator = aligned_weights.where(factors.notna()).abs().sum(axis=1)
        return numerator.div(denominator.replace(0.0, np.nan))

    # 中文说明：`select_representatives`：执行该名称对应的业务计算，并返回调用方所需结果。
    def select_representatives(
        self,
        correlation: pd.DataFrame,
        quality: pd.Series,
    ) -> list[str]:
        """Greedy quality-first pruning for highly redundant factors."""
        ordered = quality.sort_values(ascending=False).index
        selected: list[str] = []
        for factor in ordered:
            if all(
                abs(float(correlation.loc[factor, incumbent]))
                < self.config.correlation_threshold
                for incumbent in selected
            ):
                selected.append(str(factor))
        return selected
