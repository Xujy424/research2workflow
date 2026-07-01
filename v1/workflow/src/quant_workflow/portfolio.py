"""Strategy-aware convex portfolio optimiser and post-solve diagnostics."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import cvxpy as cp
import numpy as np
import pandas as pd

from quant_shared.config import OptimizerConfig, StrategyType
from quant_shared.contracts import OptimizationResult, RiskModelOutput
from .costs import TransactionCostModel


# 中文说明：定义 `PortfolioOptimizer`，封装本模块对应的数据、配置与行为。
class PortfolioOptimizer:
    # 中文说明：`__init__`：初始化对象及其运行依赖。
    def __init__(
        self,
        config: OptimizerConfig | None = None,
        cost_model: TransactionCostModel | None = None,
    ) -> None:
        self.config = config or OptimizerConfig()
        self.cost_model = cost_model or TransactionCostModel()

    # 中文说明：`optimize`：求解组合权重和约束。
    def optimize(
        self,
        alpha: pd.Series,
        risk: RiskModelOutput,
        current_weights: pd.Series | None = None,
        benchmark_weights: pd.Series | None = None,
        adv_fraction: pd.Series | None = None,
        tradable: pd.Series | None = None,
    ) -> OptimizationResult:
        assets = risk.assets.intersection(alpha.index)
        if self.config.strategy == StrategyType.INDEX_ENHANCED:
            if benchmark_weights is None:
                raise ValueError("benchmark_weights are required for index enhancement")
            self._validate_benchmark(benchmark_weights)
            if self.config.benchmark_constituents_only:
                constituents = benchmark_weights[
                    benchmark_weights > self.config.benchmark_weight_tolerance
                ].index
                assets = assets.intersection(constituents)
        if len(assets) == 0:
            raise ValueError("alpha and risk model have no common assets")
        alpha_vector = alpha.reindex(assets).fillna(0.0).to_numpy(float)
        current = self._aligned(current_weights, assets)
        has_current = current_weights is not None
        benchmark = self._aligned(benchmark_weights, assets)
        if self.config.strategy == StrategyType.INDEX_ENHANCED:
            covered_weight = float(benchmark.sum())
            if not np.isclose(covered_weight, 1.0, atol=self.config.benchmark_weight_tolerance):
                raise ValueError(
                    "risk/alpha universe does not cover the full benchmark; "
                    f"covered benchmark weight={covered_weight:.12f}"
                )
        sigma = risk.stock_covariance.loc[assets, assets].to_numpy(float)
        exposures = risk.exposures.loc[assets].fillna(0.0)

        weights = cp.Variable(len(assets), name="weights")
        trades = weights - current
        risk_vector = weights - benchmark if self.config.strategy == StrategyType.INDEX_ENHANCED else weights
        # For index enhancement alpha@benchmark is constant, so maximizing
        # alpha@weights is exactly equivalent to maximizing alpha@active_weight.
        expected_return = alpha_vector @ weights
        risk_penalty = cp.quad_form(risk_vector, cp.psd_wrap(sigma))
        linear_cost, impact_cost = self.cost_model.component_expressions(
            trades, assets, adv_fraction
        )
        objective = cp.Maximize(
            expected_return
            - 0.5 * self.config.risk_aversion * risk_penalty
            - self.config.linear_cost_penalty * linear_cost
            - self.config.impact_cost_penalty * impact_cost
            - self.config.turnover_penalty * cp.sum_squares(trades)
        )
        constraints = self._constraints(
            weights,
            trades,
            risk_vector,
            sigma,
            exposures,
            current,
            benchmark,
            tradable,
            adv_fraction,
            assets,
            has_current,
        )
        problem = cp.Problem(objective, constraints)
        status, solver = self._solve(problem)
        if weights.value is None or status not in {"optimal", "optimal_inaccurate"}:
            raise RuntimeError(f"portfolio optimisation failed: {status}")
        solution = np.asarray(weights.value, dtype=float)
        solution[np.abs(solution) < 1e-10] = 0.0
        weight_series = pd.Series(solution, index=assets, name="target_weight")
        trade_series = weight_series - pd.Series(current, index=assets)
        active = solution - benchmark if self.config.strategy == StrategyType.INDEX_ENHANCED else solution
        benchmark_series = pd.Series(
            benchmark, index=assets, name="benchmark_weight"
        )
        active_series = pd.Series(active, index=assets, name="active_weight")
        cost_estimate = self.cost_model.estimate(trade_series, adv_fraction)
        predicted_variance = max(float(active @ sigma @ active), 0.0)
        exposure_result = exposures.T @ pd.Series(active, index=assets)
        return OptimizationResult(
            weights=weight_series,
            trades=trade_series.rename("trade_weight"),
            status=status,
            expected_return=float(alpha_vector @ active)
            if self.config.strategy == StrategyType.INDEX_ENHANCED
            else float(alpha_vector @ solution),
            predicted_volatility=float(np.sqrt(predicted_variance)),
            turnover=float(np.abs(trade_series).sum()),
            expected_cost=cost_estimate.total,
            exposures=exposure_result.rename("active_exposure"),
            constraint_usage=self._constraint_usage(weight_series, benchmark, sigma),
            diagnostics={
                "solver": solver,
                "objective_value": float(problem.value),
                "absolute_expected_return": float(alpha_vector @ solution),
                "active_expected_return": float(alpha_vector @ active),
                "linear_cost": cost_estimate.linear,
                "impact_cost": cost_estimate.impact,
            },
            benchmark_weights=benchmark_series,
            active_weights=active_series,
        )

    # 中文说明：`with_config`：执行该名称对应的业务计算，并返回调用方所需结果。
    def with_config(self, **changes: object) -> "PortfolioOptimizer":
        return PortfolioOptimizer(replace(self.config, **changes), self.cost_model)

    # 中文说明：`_aligned`：内部辅助步骤，不作为稳定公共接口。
    @staticmethod
    def _aligned(values: pd.Series | None, assets: pd.Index) -> np.ndarray:
        if values is None:
            return np.zeros(len(assets))
        return values.reindex(assets).fillna(0.0).to_numpy(float)

    # 中文说明：`_validate_benchmark`：内部辅助步骤，不作为稳定公共接口。
    def _validate_benchmark(self, benchmark: pd.Series) -> None:
        """Validate the long-only index portfolio used as the neutral anchor."""
        if benchmark.empty:
            raise ValueError("benchmark_weights cannot be empty")
        if benchmark.index.has_duplicates:
            raise ValueError("benchmark_weights contains duplicate assets")
        values = benchmark.astype(float)
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError("benchmark_weights contains non-finite values")
        if (values < -self.config.benchmark_weight_tolerance).any():
            raise ValueError("benchmark_weights must be non-negative")
        if not np.isclose(
            float(values.sum()),
            1.0,
            atol=self.config.benchmark_weight_tolerance,
        ):
            raise ValueError("benchmark_weights must sum to 1")

    # 中文说明：`_constraints`：内部辅助步骤，不作为稳定公共接口。
    def _constraints(
        self,
        weights: cp.Variable,
        trades: cp.Expression,
        risk_vector: cp.Expression,
        sigma: np.ndarray,
        exposures: pd.DataFrame,
        current: np.ndarray,
        benchmark: np.ndarray,
        tradable: pd.Series | None,
        adv_fraction: pd.Series | None,
        assets: pd.Index,
        has_current: bool,
    ) -> list[cp.Constraint]:
        cfg = self.config
        constraints: list[cp.Constraint] = []
        if cfg.strategy == StrategyType.LONG_ONLY:
            constraints.extend(
                [
                    cp.sum(weights) == 1.0,
                    weights >= cfg.min_weight,
                    weights <= cfg.max_weight,
                ]
            )
        elif cfg.strategy == StrategyType.INDEX_ENHANCED:
            constraints.extend(
                [
                    cp.sum(weights) == 1.0,
                    weights >= cfg.min_weight,
                    weights <= cfg.max_weight,
                    weights - benchmark <= cfg.max_active_weight,
                    weights - benchmark >= -cfg.max_active_weight,
                ]
            )
        elif cfg.strategy == StrategyType.MARKET_NEUTRAL:
            constraints.extend(
                [
                    cp.sum(weights) == cfg.net_exposure,
                    cp.norm1(weights) <= cfg.gross_exposure,
                    weights <= cfg.max_weight,
                    weights >= -cfg.max_weight,
                ]
            )
        else:
            raise ValueError(f"unsupported strategy: {cfg.strategy}")
        if cfg.max_turnover is not None and has_current:
            constraints.append(cp.norm1(trades) <= cfg.max_turnover)
        if cfg.max_adv_participation is not None and adv_fraction is not None:
            trade_capacity = (
                adv_fraction.reindex(assets)
                .fillna(0.0)
                .clip(lower=0.0)
                .to_numpy(float)
                * cfg.max_adv_participation
            )
            constraints.extend([trades <= trade_capacity, trades >= -trade_capacity])
        if cfg.tracking_error_limit is not None:
            constraints.append(
                cp.quad_form(risk_vector, cp.psd_wrap(sigma))
                <= cfg.tracking_error_limit**2
            )
        exposure_matrix = exposures.to_numpy(float).T
        for factor, lower in cfg.exposure_lower.items():
            if factor not in exposures.columns:
                raise KeyError(f"unknown exposure constraint: {factor}")
            position = exposures.columns.get_loc(factor)
            constraints.append(exposure_matrix[position] @ risk_vector >= lower)
        for factor, upper in cfg.exposure_upper.items():
            if factor not in exposures.columns:
                raise KeyError(f"unknown exposure constraint: {factor}")
            position = exposures.columns.get_loc(factor)
            constraints.append(exposure_matrix[position] @ risk_vector <= upper)
        if tradable is not None:
            locked = ~tradable.reindex(assets).fillna(False).to_numpy(bool)
            if locked.any():
                constraints.append(weights[locked] == current[locked])
        return constraints

    # 中文说明：`_solve`：内部辅助步骤，不作为稳定公共接口。
    def _solve(self, problem: cp.Problem) -> tuple[str, str]:
        installed = set(cp.installed_solvers())
        candidates: Iterable[str] = (
            self.config.solver,
            "CLARABEL",
            "ECOS",
            "SCS",
        )
        errors: list[str] = []
        for solver in dict.fromkeys(candidates):
            if solver not in installed:
                continue
            try:
                problem.solve(solver=solver, **dict(self.config.solver_options))
                if problem.status in {"optimal", "optimal_inaccurate"}:
                    return str(problem.status), solver
            except cp.error.SolverError as exc:
                errors.append(f"{solver}: {exc}")
        raise RuntimeError("no solver produced a solution; " + "; ".join(errors))

    # 中文说明：`_constraint_usage`：内部辅助步骤，不作为稳定公共接口。
    def _constraint_usage(
        self,
        weights: pd.Series,
        benchmark: np.ndarray,
        sigma: np.ndarray,
    ) -> dict[str, float]:
        active = weights.to_numpy() - (
            benchmark if self.config.strategy == StrategyType.INDEX_ENHANCED else 0.0
        )
        return {
            "gross_exposure": float(np.abs(weights).sum()),
            "net_exposure": float(weights.sum()),
            "largest_position": float(weights.abs().max()),
            "tracking_error_or_volatility": float(np.sqrt(max(active @ sigma @ active, 0.0))),
        }
