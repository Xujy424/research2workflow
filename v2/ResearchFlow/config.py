"""Configuration objects for the v2 research-to-portfolio flow."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PortfolioRoute(str, Enum):
    SLEEVE = "sleeve"
    UNIFIED_ALPHA = "unified_alpha"


class StrategyType(str, Enum):
    LONG_ONLY = "long_only"
    INDEX_ENHANCED = "index_enhanced"
    MARKET_NEUTRAL = "market_neutral"


@dataclass(frozen=True)
class FamilyConfig:
    corr_threshold: float = 0.70
    ic_corr_threshold: float = 0.70
    min_ic_obs: int = 60
    representative_metric: str = "icir"
    clustering_method: str = "greedy"
    distance_threshold: float = 0.30
    transform_method: str = "raw"
    orthogonalization: str = "symmetric"
    transform_ridge: float = 1e-6
    n_components: int = 3
    min_component_abs_ic: float = 0.0
    min_component_abs_icir: float = 0.0
    composite_method: str = "icir"
    lookback: int = 252
    max_member_weight: float = 0.50


@dataclass(frozen=True)
class AlphaConfig:
    method: str = "icir"
    lookback: int = 252
    min_periods: int = 60
    max_family_weight: float = 0.50
    ridge_lambda: float = 10.0
    l1_ratio: float = 0.50
    n_components: int = 3
    random_state: int = 7
    max_iter: int = 500
    hidden_layer_sizes: tuple[int, ...] = (32, 16)
    clip_sigma: float = 4.0


@dataclass(frozen=True)
class SleeveConfig:
    quantile: float = 0.20
    long_only: bool = True
    allocation_method: str = "icir"
    lookback: int = 252
    min_periods: int = 60
    max_sleeve_weight: float = 0.60
    smoothing: float = 0.80


@dataclass(frozen=True)
class RiskConfig:
    factor_halflife: float = 60.0
    specific_halflife: float = 60.0
    newey_west_lags: int = 5
    covariance_shrinkage: float = 0.20
    specific_shrinkage: float = 0.20
    variance_floor: float = 1e-8
    annualization: float = 252.0


@dataclass(frozen=True)
class OptimizerConfig:
    max_stock_weight: float = 0.02
    max_turnover: float | None = 0.40
    max_adv_participation: float | None = 0.10
    industry_upper: float | None = None
    industry_lower: float | None = None
    benchmark_blend: float = 0.0
    turnover_penalty: float = 0.0

    strategy: StrategyType = StrategyType.LONG_ONLY
    min_weight: float = 0.0
    max_weight: float = 0.05
    max_active_weight: float = 0.02
    net_exposure: float = 0.0
    gross_exposure: float = 1.0
    risk_aversion: float = 8.0
    linear_cost_penalty: float = 1.0
    impact_cost_penalty: float = 1.0
    tracking_error_limit: float | None = None
    benchmark_constituents_only: bool = False
    benchmark_weight_tolerance: float = 1e-8
    exposure_lower: dict[int, float] = field(default_factory=dict)
    exposure_upper: dict[int, float] = field(default_factory=dict)
    solver: str = "CLARABEL"
    solver_options: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ResearchFlowV2Config:
    data_root: str = "D:/data"
    route: PortfolioRoute = PortfolioRoute.UNIFIED_ALPHA
    registry_path: str = "D:/data/factorpool/registry.json"
    label_category: str = "label"
    label_field: str = "Y.1D"
    tradable_category: str = "mask"
    tradable_field: str = "tradable"
    industry_category: str = "mask"
    industry_field: str = "industry"
    market_cap_category: str = "d_field"
    market_cap_field: str = "mv"
    adv_category: str = "d_field"
    adv_field: str = "amount"
    current_weight_category: str = "position"
    current_weight_field: str = "current_weight"
    output_category: str = "position"
    output_weight_field: str = "target_weight"
    output_alpha_category: str = "factorpool"
    output_alpha_field: str = "composite_alpha"
    family: FamilyConfig = field(default_factory=FamilyConfig)
    alpha: AlphaConfig = field(default_factory=AlphaConfig)
    sleeve: SleeveConfig = field(default_factory=SleeveConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
