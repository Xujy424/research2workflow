"""Final stock-weight projection and execution-aware constraints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import OptimizerConfig, StrategyType
from ..matrix_math import cap_and_renormalize


@dataclass(frozen=True)
class PortfolioProjectionResult:
    weights: np.ndarray
    turnover: np.ndarray
    diagnostics: dict[str, float]


class StockWeightProjector:
    """Fast long-only projection used after either branch produces a raw score."""

    def __init__(self, config: OptimizerConfig | None = None) -> None:
        self.config = config or OptimizerConfig()

    def project(
        self,
        score: np.ndarray,
        *,
        tradable: np.ndarray,
        current_weight: np.ndarray | None = None,
        benchmark_weight: np.ndarray | None = None,
        adv: np.ndarray | None = None,
        industry: np.ndarray | None = None,
    ) -> PortfolioProjectionResult:
        raw = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        raw = np.where(tradable, np.maximum(raw, 0.0), 0.0)
        if benchmark_weight is not None and self.config.benchmark_blend > 0:
            raw = (1.0 - self.config.benchmark_blend) * raw + self.config.benchmark_blend * np.maximum(benchmark_weight, 0.0)
        weights = np.zeros_like(raw)
        turnover = np.zeros(raw.shape[0], dtype=float)
        prev = np.zeros(raw.shape[1], dtype=float) if current_weight is None else np.nan_to_num(current_weight[0], nan=0.0)
        for t in range(raw.shape[0]):
            row = raw[t]
            if row.sum() <= 1e-12:
                eligible = tradable[t].astype(float)
                row = eligible
            w = row / row.sum() if row.sum() > 1e-12 else row
            if industry is not None and self.config.industry_upper is not None:
                w = self._limit_industry(w, industry[t])
            w = cap_and_renormalize(w, max_weight=self.config.max_stock_weight)
            if adv is not None and self.config.max_adv_participation is not None:
                w = self._limit_adv_trade(w, prev, adv[t])
            if self.config.max_turnover is not None:
                w = self._limit_turnover(w, prev, self.config.max_turnover)
            weights[t] = w
            turnover[t] = float(np.abs(w - prev).sum())
            prev = w
        return PortfolioProjectionResult(
            weights=weights,
            turnover=turnover,
            diagnostics={
                "avg_turnover": float(np.nanmean(turnover)),
                "max_turnover": float(np.nanmax(turnover)),
                "avg_holding_count": float(np.nanmean((weights > 1e-12).sum(axis=1))),
            },
        )

    def _limit_industry(self, weights: np.ndarray, industry: np.ndarray) -> np.ndarray:
        upper = self.config.industry_upper
        if upper is None:
            return weights
        out = weights.copy()
        for code in np.unique(industry[np.isfinite(industry)]):
            m = industry == code
            total = out[m].sum()
            if total > upper and total > 0:
                excess = total - upper
                out[m] *= upper / total
                free = ~m & (out > 0)
                if free.any():
                    out[free] += excess * out[free] / out[free].sum()
        return out / out.sum() if out.sum() > 1e-12 else out

    @staticmethod
    def _limit_turnover(target: np.ndarray, current: np.ndarray, max_turnover: float) -> np.ndarray:
        trade = target - current
        gross = np.abs(trade).sum()
        if gross <= max_turnover or gross <= 1e-12:
            return target
        return current + trade * (max_turnover / gross)

    def _limit_adv_trade(self, target: np.ndarray, current: np.ndarray, adv: np.ndarray) -> np.ndarray:
        capacity = np.nan_to_num(adv, nan=0.0, posinf=0.0, neginf=0.0)
        if capacity.max(initial=0.0) > 1.0:
            capacity = capacity / max(capacity.sum(), 1e-12)
        cap = capacity * float(self.config.max_adv_participation or 0.0)
        trade = np.clip(target - current, -cap, cap)
        out = np.maximum(current + trade, 0.0)
        return out / out.sum() if out.sum() > 1e-12 else out




@dataclass(frozen=True)
class OptimizationResult:
    weights: np.ndarray
    trades: np.ndarray
    status: str
    expected_return: float
    predicted_volatility: float
    turnover: float
    expected_cost: float
    exposures: np.ndarray
    constraint_usage: dict[str, float]
    diagnostics: dict[str, float | str]
    benchmark_weights: np.ndarray | None = None
    active_weights: np.ndarray | None = None


class CvxPortfolioOptimizer:
    """Strategy-aware convex optimizer migrated from v1 with matrix inputs."""

    def __init__(self, config: OptimizerConfig | None = None) -> None:
        self.config = config or OptimizerConfig()

    def optimize(
        self,
        alpha: np.ndarray,
        covariance: np.ndarray,
        *,
        current_weight: np.ndarray | None = None,
        benchmark_weight: np.ndarray | None = None,
        adv_weight: np.ndarray | None = None,
        tradable: np.ndarray | None = None,
        exposures: np.ndarray | None = None,
    ) -> OptimizationResult:
        cp = self._cvxpy()
        cfg = self.config
        a = np.nan_to_num(np.asarray(alpha, dtype=float).reshape(-1), nan=0.0)
        sigma = np.asarray(covariance, dtype=float)
        if sigma.shape != (a.size, a.size):
            raise ValueError(f"covariance shape mismatch: {sigma.shape} vs {(a.size, a.size)}")
        current = np.zeros_like(a) if current_weight is None else np.nan_to_num(current_weight, nan=0.0).reshape(-1)
        benchmark = np.zeros_like(a) if benchmark_weight is None else np.nan_to_num(benchmark_weight, nan=0.0).reshape(-1)
        x = np.zeros((a.size, 0), dtype=float) if exposures is None else np.nan_to_num(exposures, nan=0.0)
        if x.shape[0] != a.size:
            raise ValueError("exposures must be shaped N x K")

        if cfg.strategy == StrategyType.INDEX_ENHANCED:
            if benchmark_weight is None:
                raise ValueError("benchmark_weight is required for index enhancement")
            self._validate_benchmark(benchmark)
            if cfg.benchmark_constituents_only:
                allowed = benchmark > cfg.benchmark_weight_tolerance
                a, sigma, current, benchmark, x = self._subset(a, sigma, current, benchmark, x, allowed)
                if tradable is not None:
                    tradable = np.asarray(tradable, dtype=bool)[allowed]
                if adv_weight is not None:
                    adv_weight = np.asarray(adv_weight, dtype=float)[allowed]

        n = a.size
        w = cp.Variable(n, name="weights")
        trades = w - current
        risk_vector = w - benchmark if cfg.strategy == StrategyType.INDEX_ENHANCED else w
        linear_cost, impact_cost = self._cost_expressions(cp, trades, n, adv_weight)
        objective = cp.Maximize(
            a @ w
            - 0.5 * cfg.risk_aversion * cp.quad_form(risk_vector, cp.psd_wrap(sigma))
            - cfg.linear_cost_penalty * linear_cost
            - cfg.impact_cost_penalty * impact_cost
            - cfg.turnover_penalty * cp.sum_squares(trades)
        )
        constraints = self._constraints(cp, w, trades, risk_vector, sigma, benchmark, tradable, adv_weight, x)
        problem = cp.Problem(objective, constraints)
        status, solver = self._solve(cp, problem)
        if w.value is None or status not in {"optimal", "optimal_inaccurate"}:
            raise RuntimeError(f"portfolio optimisation failed: {status}")

        weights = np.asarray(w.value, dtype=float)
        weights[np.abs(weights) < 1e-10] = 0.0
        trades_value = weights - current
        active = weights - benchmark if cfg.strategy == StrategyType.INDEX_ENHANCED else weights
        cost = self._estimate_cost(trades_value, adv_weight)
        variance = max(float(active @ sigma @ active), 0.0)
        return OptimizationResult(
            weights=weights,
            trades=trades_value,
            status=status,
            expected_return=float(a @ active if cfg.strategy == StrategyType.INDEX_ENHANCED else a @ weights),
            predicted_volatility=float(np.sqrt(variance)),
            turnover=float(np.abs(trades_value).sum()),
            expected_cost=cost,
            exposures=x.T @ active if x.size else np.empty(0, dtype=float),
            constraint_usage=self._constraint_usage(weights, benchmark, sigma),
            diagnostics={
                "solver": solver,
                "objective_value": float(problem.value),
                "linear_cost": float(self._estimate_linear_cost(trades_value)),
                "impact_cost": float(cost - self._estimate_linear_cost(trades_value)),
            },
            benchmark_weights=benchmark if cfg.strategy == StrategyType.INDEX_ENHANCED else None,
            active_weights=active if cfg.strategy == StrategyType.INDEX_ENHANCED else None,
        )

    def _constraints(self, cp, weights, trades, risk_vector, sigma, benchmark, tradable, adv_weight, exposures) -> list:
        cfg = self.config
        constraints = []
        if cfg.strategy == StrategyType.LONG_ONLY:
            constraints += [cp.sum(weights) == 1.0, weights >= cfg.min_weight, weights <= cfg.max_weight]
        elif cfg.strategy == StrategyType.INDEX_ENHANCED:
            constraints += [
                cp.sum(weights) == 1.0,
                weights >= cfg.min_weight,
                weights <= cfg.max_weight,
                weights - benchmark <= cfg.max_active_weight,
                weights - benchmark >= -cfg.max_active_weight,
            ]
        elif cfg.strategy == StrategyType.MARKET_NEUTRAL:
            constraints += [
                cp.sum(weights) == cfg.net_exposure,
                cp.norm1(weights) <= cfg.gross_exposure,
                weights <= cfg.max_weight,
                weights >= -cfg.max_weight,
            ]
        else:
            raise ValueError(f"unsupported strategy: {cfg.strategy}")
        if cfg.max_turnover is not None:
            constraints.append(cp.norm1(trades) <= cfg.max_turnover)
        if cfg.max_adv_participation is not None and adv_weight is not None:
            cap = np.clip(np.nan_to_num(adv_weight, nan=0.0), 0.0, None) * cfg.max_adv_participation
            constraints += [trades <= cap, trades >= -cap]
        if cfg.tracking_error_limit is not None:
            constraints.append(cp.quad_form(risk_vector, cp.psd_wrap(sigma)) <= cfg.tracking_error_limit**2)
        for idx, lower in cfg.exposure_lower.items():
            constraints.append(exposures[:, int(idx)] @ risk_vector >= lower)
        for idx, upper in cfg.exposure_upper.items():
            constraints.append(exposures[:, int(idx)] @ risk_vector <= upper)
        if tradable is not None:
            locked = ~np.asarray(tradable, dtype=bool)
            if locked.any():
                constraints.append(trades[locked] == 0.0)
        return constraints

    def _cost_expressions(self, cp, trades, n_assets: int, adv_weight: np.ndarray | None):
        linear = np.full(n_assets, 0.001, dtype=float)
        if adv_weight is None:
            impact = np.zeros(n_assets, dtype=float)
        else:
            impact = 0.001 / np.sqrt(np.clip(np.nan_to_num(adv_weight, nan=1e-5), 1e-5, None))
        return linear @ cp.abs(trades), impact @ cp.power(cp.abs(trades), 1.5)

    @staticmethod
    def _estimate_linear_cost(trades: np.ndarray) -> float:
        return float(0.001 * np.abs(trades).sum())

    def _estimate_cost(self, trades: np.ndarray, adv_weight: np.ndarray | None) -> float:
        linear = self._estimate_linear_cost(trades)
        if adv_weight is None:
            return linear
        impact = 0.001 / np.sqrt(np.clip(np.nan_to_num(adv_weight, nan=1e-5), 1e-5, None))
        return float(linear + impact @ np.power(np.abs(trades), 1.5))

    @staticmethod
    def _subset(a, sigma, current, benchmark, exposures, mask):
        return a[mask], sigma[np.ix_(mask, mask)], current[mask], benchmark[mask], exposures[mask]

    def _validate_benchmark(self, benchmark: np.ndarray) -> None:
        if benchmark.size == 0 or not np.isfinite(benchmark).all() or np.any(benchmark < -self.config.benchmark_weight_tolerance):
            raise ValueError("benchmark_weight must be finite and non-negative")
        if not np.isclose(float(benchmark.sum()), 1.0, atol=self.config.benchmark_weight_tolerance):
            raise ValueError("benchmark_weight must sum to 1")

    def _solve(self, cp, problem) -> tuple[str, str]:
        installed = set(cp.installed_solvers())
        for solver in dict.fromkeys((self.config.solver, "CLARABEL", "ECOS", "SCS")):
            if solver not in installed:
                continue
            try:
                problem.solve(solver=solver, **dict(self.config.solver_options))
                if problem.status in {"optimal", "optimal_inaccurate"}:
                    return str(problem.status), solver
            except cp.error.SolverError:
                continue
        return str(problem.status), ""

    def _constraint_usage(self, weights: np.ndarray, benchmark: np.ndarray, sigma: np.ndarray) -> dict[str, float]:
        active = weights - benchmark if self.config.strategy == StrategyType.INDEX_ENHANCED else weights
        return {
            "gross_exposure": float(np.abs(weights).sum()),
            "net_exposure": float(weights.sum()),
            "largest_position": float(np.max(np.abs(weights))) if weights.size else 0.0,
            "tracking_error_or_volatility": float(np.sqrt(max(active @ sigma @ active, 0.0))),
        }

    @staticmethod
    def _cvxpy():
        try:
            import cvxpy as cp  # type: ignore
        except ImportError as exc:
            raise ImportError("CvxPortfolioOptimizer requires cvxpy. Install cvxpy to use this optimizer.") from exc
        return cp
