"""Observable market-regime classification and bounded dynamic weighting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from quant_shared.config import RegimeConfig


# 中文说明：定义 `RegimeResult`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class RegimeResult:
    probabilities: pd.DataFrame
    states: pd.Series


# 中文说明：定义 `ObservableRegimeModel`，封装本模块对应的数据、配置与行为。
class ObservableRegimeModel:
    """Classify regimes using lagged volatility, trend, or supplied indicators."""

    # 中文说明：`__init__`：初始化对象及其运行依赖。
    def __init__(self, config: RegimeConfig | None = None) -> None:
        self.config = config or RegimeConfig()
        if self.config.lookback <= 1:
            raise ValueError("lookback 必须大于 1")
        if not 0.0 < self.config.threshold_quantile < 1.0:
            raise ValueError("threshold_quantile 必须位于 (0, 1) 区间")
        if not 0.0 <= self.config.transition_smoothing < 1.0:
            raise ValueError("transition_smoothing 必须位于 [0, 1) 区间")
    
    @staticmethod
    def _normalize_rows(scores: pd.DataFrame) -> pd.DataFrame:
        scores = scores.clip(lower=0.0)
        row_sum = scores.sum(axis=1)
        probabilities = scores.div(
            row_sum.replace(0.0, float("nan")),
            axis=0,
        )
        return probabilities.fillna(0.0)
    
    @staticmethod
    def _expanding_percentile(values: pd.Series, min_periods: int,) -> pd.Series:
        values = (
            values.astype(float)
            .replace([float("inf"), float("-inf")], float("nan"))
        )
        return values.expanding(min_periods=min_periods).rank(method="average", pct=True)
    
    def _high_vol_probability(self, volatility_percentile: pd.Series) -> pd.Series:
        '''threshold_quantile 对应 0.5 的高波动概率'''
        quantile = self.config.threshold_quantile
        rank = volatility_percentile.clip(0.0, 1.0)
        below_threshold = 0.5 * rank / quantile
        above_threshold = 0.5 + 0.5*(rank-quantile)/(1.0-quantile)
        probability = below_threshold.where(rank <= quantile, above_threshold)
        return probability.clip(0.0, 1.0)

    @staticmethod
    def _positive_trend_probability(trend: pd.Series, trend_percentile: pd.Series) -> pd.Series:
        """
        负趋势对应 [0, 0.5), 零趋势对应 0.5, 正趋势对应 (0.5, 1]
        """
        rank = trend_percentile.clip(0.0, 1.0)
        probability = pd.Series(0.5, index=trend.index, dtype=float)
        negative = trend < 0.0
        positive = trend > 0.0
        probability.loc[negative] = 0.5 * rank.loc[negative]
        probability.loc[positive] = 0.5 + 0.5 * rank.loc[positive]
        # 历史样本不足、分位数尚未形成时保留 NaN。
        probability = probability.where(rank.notna())
        return probability.clip(0.0, 1.0)

    @staticmethod
    def _indicator_high_score(
        indicator_percentile: pd.Series,
    ) -> pd.Series:
        """
        不直接使用原始分位数，而使用： max(0, 2 * percentile - 1)
        对应关系：
            percentile = 0.30 -> score = 0.00
            percentile = 0.50 -> score = 0.00
            percentile = 0.60 -> score = 0.20
            percentile = 0.75 -> score = 0.50
            percentile = 0.90 -> score = 0.80
            percentile = 1.00 -> score = 1.00
        这样做有两个目的：
        1. 只有指标处于历史较高位置时，才产生“高状态”证据；
        2. 该分数长期平均尺度约为 0.25，与四个基础状态的平均尺度接近。
        """
        return (2.0 * indicator_percentile - 1.0).clip(lower=0.0,upper=1.0)

    # 中文说明：`fit_predict`：拟合模型参数并预测市场状态。
    def fit_predict(self, market_returns: pd.Series, indicators: pd.DataFrame | None = None) -> RegimeResult:
        returns = market_returns.sort_index().astype(float).replace([float("inf"), float("-inf")], float("nan"))
        min_rolling_periods = max(10, self.config.lookback // 3)
        min_expanding_periods = max(20, self.config.lookback)
        # 1. 计算滞后波动率和滞后趋势
        volatility = returns.rolling(self.config.lookback, min_periods=min_rolling_periods).std().shift(1)
        trend = returns.rolling(self.config.lookback, min_periods=min_rolling_periods).mean().shift(1)
        # 2. 将波动率、趋势转为连续的历史分位数
        volatility_percentile = self._expanding_percentile(volatility, min_periods=min_expanding_periods)
        trend_percentile = self._expanding_percentile(trend, min_periods=min_expanding_periods)
        high_vol_probability = self._high_vol_probability(volatility_percentile)
        positive_trend_probability = (self._positive_trend_probability(trend,trend_percentile))
        low_vol_probability = (1.0 - high_vol_probability)
        non_positive_trend_probability = (1.0 - positive_trend_probability)
        # 3. 构造四种连续基础状态, 在有效数据条件下，四个基础状态分数之和等于 1。
        scores = pd.DataFrame(
            {
                "risk_on": low_vol_probability * positive_trend_probability,
                "risk_off": high_vol_probability * non_positive_trend_probability,
                "high_vol_trend": high_vol_probability * positive_trend_probability,
                "low_vol_range": low_vol_probability * non_positive_trend_probability,
            },
            index=returns.index,
        )
        # 4. 外部指标仍然保留为独立状态
        if indicators is not None:
            aligned = (
                indicators.sort_index().reindex(scores.index)
                .apply(pd.to_numeric, errors="coerce")
                .replace(
                    [float("inf"), float("-inf")],
                    float("nan"),
                )
                .shift(1)
            )
            for column in aligned.columns:
                indicator_percentile = self._expanding_percentile(aligned[column], min_periods=20)
                scores[f"indicator_{column}_high"] = self._indicator_high_score(indicator_percentile)
        # 5. 先沿时间维度平滑每个状态的原始证据
        # transition_smoothing 越大，历史权重越高。
        alpha = 1.0 - self.config.transition_smoothing
        smoothed_scores = scores.fillna(0.0).ewm(alpha=alpha, adjust=False).mean()
        # 6. 再在当前时点对所有状态进行横截面归一化
        probabilities = self._normalize_rows(smoothed_scores)
        # 7. 选择当前概率最大的状态
        states = probabilities.idxmax(axis=1).rename("regime")
        # 如果某个时点所有状态概率都是 0，说明仍处于预热期。
        no_available_state = probabilities.sum(axis=1) <= 0.0
        states.loc[no_available_state] = "unknown"
        return RegimeResult(probabilities, states)


class RegimeWeightController:
    """Tilt long-run weights by regime, with bounded and smoothed deviations."""
    def __init__(self, config: RegimeConfig | None = None) -> None:
        self.config = config or RegimeConfig()
    def apply(
            self,
            base_weights: pd.Series, 
            probabilities: pd.DataFrame,
            tilts: Mapping[str, pd.Series],  # 每种状态下各因子权重如何偏离，相对基础权重的调整比例
    ) -> pd.DataFrame:
        base = base_weights / base_weights.sum()
        rows: list[pd.Series] = []
        for date, probability in probabilities.iterrows():
            adjustment = pd.Series(0.0, index=base.index)
            for regime, regime_probability in probability.items():
                if regime in tilts:
                    adjustment = adjustment.add(
                        tilts[regime].reindex(base.index).fillna(0.0)
                        * float(regime_probability),
                        fill_value=0.0,
                    )
            adjustment = adjustment.clip(-self.config.max_tilt, self.config.max_tilt)
            weight = (base * (1.0 + adjustment)).clip(lower=0.0)
            weight = weight / weight.sum() if weight.sum() else base
            weight.name = date
            rows.append(weight)
        return pd.DataFrame(rows).ewm(alpha=1.0 - self.config.transition_smoothing, adjust=False).mean()


class MixtureOfExperts:
    """Combine expert alpha forecasts with observable regime probabilities."""
    @staticmethod
    def combine(
        expert_forecasts: Mapping[str, pd.Series],
        probabilities: pd.DataFrame,
    ) -> pd.Series:
        if not expert_forecasts:
            raise ValueError("at least one expert forecast is required")
        template = next(iter(expert_forecasts.values()))
        dates = template.index.get_level_values(0)
        combined = pd.Series(0.0, index=template.index)
        available = pd.Series(0.0, index=template.index)
        for name, forecast in expert_forecasts.items():
            if name not in probabilities.columns:
                continue
            probability = probabilities[name].reindex(dates).set_axis(template.index)
            aligned = forecast.reindex(template.index)
            combined = combined.add(aligned.fillna(0.0) * probability.fillna(0.0))
            available = available.add(aligned.notna().astype(float) * probability.fillna(0.0))
        return combined.div(available.replace(0.0, np.nan)).rename("expected_return")
