"""Configuration objects for the v2 research-to-portfolio flow."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PortfolioRoute(str, Enum):
    SLEEVE = "sleeve"
    UNIFIED_ALPHA = "unified_alpha"


@dataclass(frozen=True)
class PreprocessConfig:
    winsor_method: str = "mad"
    standardize: bool = True
    neutralize: bool = True


@dataclass(frozen=True)
class FamilyConfig:
    corr_threshold: float = 0.85
    ic_corr_threshold: float = 0.80
    min_ic_obs: int = 60
    representative_metric: str = "icir"
    clustering_method: str = "greedy"
    composite_method: str = "icir"
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
class OptimizerConfig:
    max_stock_weight: float = 0.02
    max_turnover: float | None = 0.40
    max_adv_participation: float | None = 0.10
    industry_upper: float | None = None
    industry_lower: float | None = None
    benchmark_blend: float = 0.0
    turnover_penalty: float = 0.0


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
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    family: FamilyConfig = field(default_factory=FamilyConfig)
    alpha: AlphaConfig = field(default_factory=AlphaConfig)
    sleeve: SleeveConfig = field(default_factory=SleeveConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
