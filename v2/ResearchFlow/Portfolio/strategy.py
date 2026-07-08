"""Portfolio strategy constraints used by the convex optimizer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import OptimizerConfig, StrategyType


@dataclass(frozen=True)
class StrategyInputs:
    alpha: np.ndarray
    covariance: np.ndarray
    current: np.ndarray
    benchmark: np.ndarray
    exposures: np.ndarray
    tradable: np.ndarray | None
    adv_weight: np.ndarray | None
    benchmark_member_mask: np.ndarray | None


class PortfolioStrategy:
    """Base class for strategy-specific portfolio constraints."""

    def __init__(self, config: OptimizerConfig) -> None:
        self.config = config

    def prepare(self, inputs: StrategyInputs) -> StrategyInputs:
        return inputs


    def active_weights(self, weights: np.ndarray, benchmark: np.ndarray) -> np.ndarray:
        return weights

    def constraints(self, cp, weights, benchmark) -> list:
        raise NotImplementedError

    def benchmark_output(self, benchmark: np.ndarray) -> np.ndarray | None:
        return None


class LongOnlyStrategy(PortfolioStrategy):
    """Long-only absolute-return portfolio."""

    def constraints(self, cp, weights, benchmark) -> list:
        cfg = self.config
        return [cp.sum(weights)==1.0, weights>=cfg.min_weight, weights<=cfg.max_weight]


class IndexEnhancedStrategy(PortfolioStrategy):
    """Long-only active portfolio around a benchmark."""

    def prepare(self, inputs: StrategyInputs) -> StrategyInputs:
        benchmark = inputs.benchmark
        self._validate_benchmark(benchmark)
        if not self.config.benchmark_constituents_only:
            return inputs
        if inputs.benchmark_member_mask is None:
            raise ValueError("benchmark_constituents_only=True requires benchmark_member_mask")
        allowed = np.asarray(inputs.benchmark_member_mask, dtype=bool)
        if allowed.shape != benchmark.shape:
            raise ValueError("benchmark_member_mask must align with benchmark_weight")
        return StrategyInputs(
            alpha=inputs.alpha[allowed],
            covariance=inputs.covariance[np.ix_(allowed, allowed)],
            current=inputs.current[allowed],
            benchmark=benchmark[allowed],
            exposures=inputs.exposures[allowed],
            tradable=None if inputs.tradable is None else np.asarray(inputs.tradable, dtype=bool)[allowed],
            adv_weight=None if inputs.adv_weight is None else np.asarray(inputs.adv_weight, dtype=float)[allowed],
            benchmark_member_mask=allowed[allowed],
        )


    def active_weights(self, weights: np.ndarray, benchmark: np.ndarray) -> np.ndarray:
        return weights - benchmark

    def constraints(self, cp, weights, benchmark) -> list:
        cfg = self.config
        return [
            cp.sum(weights) == 1.0,
            weights >= cfg.min_weight,
            weights <= cfg.max_weight,
            weights - benchmark <= cfg.max_active_weight,
            weights - benchmark >= -cfg.max_active_weight,
        ]

    def benchmark_output(self, benchmark: np.ndarray) -> np.ndarray | None:
        return benchmark

    def _validate_benchmark(self, benchmark: np.ndarray) -> None:
        tol = 1e-8
        if benchmark.size == 0 or not np.isfinite(benchmark).all() or np.any(benchmark < -tol):
            raise ValueError("benchmark_weight must be finite and non-negative")
        if not np.isclose(float(benchmark.sum()), 1.0, atol=tol):
            raise ValueError("benchmark_weight must sum to 1")


class MarketNeutralStrategy(PortfolioStrategy):
    """Long-short market-neutral portfolio."""

    def constraints(self, cp, weights, benchmark) -> list:
        cfg = self.config
        return [
            cp.sum(weights) == cfg.net_exposure,
            cp.norm1(weights) <= cfg.gross_exposure,
            weights <= cfg.max_weight,
            weights >= -cfg.max_weight,
        ]


def make_strategy(config: OptimizerConfig) -> PortfolioStrategy:
    strategy = StrategyType(config.strategy)
    if strategy == StrategyType.LONG_ONLY:
        return LongOnlyStrategy(config)
    if strategy == StrategyType.INDEX_ENHANCED:
        return IndexEnhancedStrategy(config)
    if strategy == StrategyType.MARKET_NEUTRAL:
        return MarketNeutralStrategy(config)
    raise ValueError(f"unsupported strategy: {config.strategy}")




