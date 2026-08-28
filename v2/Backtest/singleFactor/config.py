"""Typed configuration: no data paths or mutable global state."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class Method(str, Enum):
    QUANTILE_LONG_SHORT = "quantile_long_short"
    QUANTILE_LONG_ONLY = "quantile_long_only"
    QUANTILE_SHORT_ONLY = "quantile_short_only"
    BENCHMARK_HEDGED = "benchmark_hedged"
    EVENT = "event"


class Weighting(str, Enum):
    EQUAL = "equal"
    SIGNAL = "signal"


class ActiveSide(str, Enum):
    LONG = "long"
    SHORT = "short"
    LONG_SHORT = "long_short"


class SignalInput(str, Enum):
    SCORE = "score"
    PREBUILT_WEIGHT = "prebuilt_weight"


class EventTrigger(str, Enum):
    NONZERO = "nonzero"
    CHANGE = "change"
    CROSS = "cross"


class EventPortfolioMode(str, Enum):
    ACTIVE = "active"
    TRIGGERED_EQUAL_WEIGHT = "triggered_equal_weight"


class RebalanceFrequency(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    EVERY_N_DAYS = "every_n_days"


@dataclass(frozen=True)
class ExecutionConfig:
    signal_lag: int = 1
    rebalance_days: int = 1
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.DAILY
    cost_bps: float = 0.0
    short_cost_bps_annual: float = 0.0
    annual_days: int = 242

    def __post_init__(self):
        if self.signal_lag < 1 or self.rebalance_days < 1:
            raise ValueError("signal_lag and rebalance_days must be >= 1")


@dataclass(frozen=True)
class PortfolioConfig:
    method: Method = Method.QUANTILE_LONG_SHORT
    quantiles: int = 10
    top_groups: int = 1
    bottom_groups: int = 1
    weighting: Weighting = Weighting.EQUAL
    gross_exposure: float = 1.0
    industry_align: bool = False
    active_side: ActiveSide = ActiveSide.LONG
    signal_input: SignalInput = SignalInput.SCORE
    active_gross: float = 1.0

    def __post_init__(self):
        if self.quantiles < 2 or self.top_groups < 1:
            raise ValueError("quantiles >= 2 and top_groups >= 1 are required")
        if self.top_groups + self.bottom_groups > self.quantiles:
            raise ValueError("long and short groups overlap")
        if self.active_gross <= 0:
            raise ValueError("active_gross must be positive")


@dataclass(frozen=True)
class EventConfig:
    trigger: EventTrigger = EventTrigger.NONZERO
    threshold: float = 0.0
    holding_days: int = 5
    cooldown_days: int = 0
    cross_sectionalize: bool = True
    pair_within_industry: bool = False
    portfolio_mode: EventPortfolioMode = EventPortfolioMode.ACTIVE

    def __post_init__(self):
        if self.holding_days < 1 or self.cooldown_days < 0:
            raise ValueError("invalid event holding/cooldown period")


@dataclass(frozen=True)
class BacktestConfig:
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    event: EventConfig = field(default_factory=EventConfig)


@dataclass(frozen=True)
class PairDefinition:
    left: str
    right: str
    hedge_ratio: float = 1.0
    rationale: str = ""

    def __post_init__(self):
        if self.left == self.right or self.hedge_ratio <= 0:
            raise ValueError("a pair needs two assets and a positive hedge ratio")
        if not self.rationale.strip():
            raise ValueError("pair rationale is required; pairs are not inferred")


@dataclass(frozen=True)
class CapacityConfig:
    capital: tuple[float, ...] = (1e7, 5e7, 1e8, 5e8)
    max_participation: float = 0.10
    commission_bps: float = 10.0
    impact_coefficient: float = 0.001
    lot_size: int = 100

    def __post_init__(self):
        if not 0 < self.max_participation <= 1:
            raise ValueError("max_participation must be in (0, 1]")
        if self.lot_size < 1 or any(x <= 0 for x in self.capital):
            raise ValueError("capital and lot_size must be positive")
