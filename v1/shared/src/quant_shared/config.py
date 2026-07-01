"""Typed configuration for the factor-to-portfolio workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


# 中文说明：定义 `StrategyType`，封装本模块对应的数据、配置与行为。
class StrategyType(str, Enum):
    LONG_ONLY = "long_only"
    INDEX_ENHANCED = "index_enhanced"
    MARKET_NEUTRAL = "market_neutral"


# 中文说明：定义 `PreprocessConfig`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class PreprocessConfig:
    winsor_method: str = "mad"
    winsor_limit: float = 5.0
    standardize: str = "zscore"
    neutralize: bool = True
    ridge: float = 1e-8
    min_observations: int = 20


# 中文说明：定义 `CompositeConfig`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class CompositeConfig:
    method: str = "icir"
    lookback: int = 60
    min_periods: int = 20
    ic_shrinkage: float = 0.50
    weight_smoothing: float = 0.80
    max_factor_weight: float = 0.35
    allow_negative: bool = False
    correlation_threshold: float = 0.90


# 中文说明：定义 `TransformConfig`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class TransformConfig:
    method: str = "none"
    n_components: int = 3
    lookback: int = 120
    min_periods: int = 40
    ridge: float = 1e-8
    orthogonalization: str = "symmetric"


# 中文说明：定义 `AlphaConfig`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class AlphaConfig:
    method: str = "ridge"
    lookback: int = 120
    min_periods: int = 40
    ridge: float = 10.0
    decay_halflife: float = 60.0
    clip_sigma: float = 4.0
    annualization: float = 252.0
    l1_ratio: float = 0.50
    n_components: int = 3
    random_state: int = 7
    max_iter: int = 500


# 中文说明：定义 `SleeveConfig`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class SleeveConfig:
    method: str = "risk_parity"
    lookback: int = 120
    min_periods: int = 40
    return_shrinkage: float = 0.70
    covariance_shrinkage: float = 0.30
    max_weight: float = 0.50
    turnover_penalty: float = 0.05
    risk_aversion: float = 5.0
    weight_smoothing: float = 0.80


# 中文说明：定义 `RegimeConfig`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class RegimeConfig:
    method: str = "volatility"
    lookback: int = 60
    threshold_quantile: float = 0.60
    transition_smoothing: float = 0.85
    max_tilt: float = 0.25


# 中文说明：定义 `RiskConfig`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class RiskConfig:
    factor_halflife: float = 60.0
    specific_halflife: float = 60.0
    newey_west_lags: int = 2
    covariance_shrinkage: float = 0.20
    specific_shrinkage: float = 0.30
    variance_floor: float = 1e-8
    annualization: float = 252.0


# 中文说明：定义 `OptimizerConfig`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class OptimizerConfig:
    strategy: StrategyType = StrategyType.LONG_ONLY
    risk_aversion: float = 8.0
    linear_cost_penalty: float = 1.0
    impact_cost_penalty: float = 1.0
    turnover_penalty: float = 0.0
    min_weight: float = 0.0
    max_weight: float = 0.05
    max_active_weight: float = 0.02
    benchmark_constituents_only: bool = True
    benchmark_weight_tolerance: float = 1e-8
    max_turnover: float | None = 0.40
    max_adv_participation: float | None = 0.10
    gross_exposure: float = 2.0
    net_exposure: float = 0.0
    tracking_error_limit: float | None = None
    exposure_lower: Mapping[str, float] = field(default_factory=dict)
    exposure_upper: Mapping[str, float] = field(default_factory=dict)
    solver: str = "CLARABEL"
    solver_options: Mapping[str, object] = field(default_factory=dict)
