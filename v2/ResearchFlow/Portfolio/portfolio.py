"""Final stock-weight projection and execution-aware constraints."""

from __future__ import annotations

from dataclasses import dataclass
import cvxpy as cp
import numpy as np

from ..config import OptimizerConfig
from ..matrix_math import cap_and_renormalize
from .cost import TransactionCostModel
from .strategy import StrategyInputs, make_strategy


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
        raw = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)  # T,N
        raw = np.where(tradable, np.maximum(raw, 0.0), 0.0)
        if benchmark_weight is not None and self.config.benchmark_blend > 0:
            raw = (1.0 - self.config.benchmark_blend) * raw + self.config.benchmark_blend * np.maximum(benchmark_weight, 0.0)
        weights = np.zeros_like(raw)
        turnover = np.zeros(raw.shape[0], dtype=float)  # T,
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

    def __init__(self, config: OptimizerConfig | None = None, *, cost_model: TransactionCostModel | None = None) -> None:
        self.config = config or OptimizerConfig()
        self.cost_model = cost_model or TransactionCostModel()

    def optimize(
        self,
        alpha: np.ndarray,
        covariance: np.ndarray,
        *,
        current_weight: np.ndarray | None = None,
        benchmark_weight: np.ndarray | None = None,
        benchmark_member_mask: np.ndarray | None = None,
        adv_weight: np.ndarray | None = None,
        tradable: np.ndarray | None = None,
        exposures: np.ndarray | None = None,
    ) -> OptimizationResult:
        cfg = self.config

        alpha_arr = np.asarray(alpha, dtype=float)
        if alpha_arr.ndim != 1:
            raise ValueError("CvxPortfolioOptimizer.optimize expects one cross-sectional alpha vector; loop by date in workflow")
        a = np.nan_to_num(alpha_arr, nan=0.0)

        sigma = np.asarray(covariance, dtype=float)
        if sigma.shape != (a.size, a.size):
            raise ValueError(f"covariance shape mismatch: {sigma.shape} vs {(a.size, a.size)}")
        
        current = np.zeros_like(a) if current_weight is None else np.nan_to_num(current_weight, nan=0.0).reshape(-1)
        benchmark = np.zeros_like(a) if benchmark_weight is None else np.nan_to_num(benchmark_weight, nan=0.0).reshape(-1)
        x = np.zeros((a.size, 0), dtype=float) if exposures is None else np.nan_to_num(exposures, nan=0.0)
        if x.shape[0] != a.size:
            raise ValueError("exposures must be shaped N x K")

        strategy = make_strategy(cfg)
        prepared = strategy.prepare(StrategyInputs(a, sigma, current, benchmark, x, tradable, adv_weight, benchmark_member_mask))
        a, sigma, current, benchmark, x = (
            prepared.alpha,
            prepared.covariance,
            prepared.current,
            prepared.benchmark,
            prepared.exposures,
        )
        tradable = prepared.tradable
        adv_weight = prepared.adv_weight

        n = a.size
        w = cp.Variable(n, name="weights")
        trades = w - current
        active_weights = strategy.active_weights(w, benchmark)
        linear_cost, impact_cost = self.cost_model.cost_expressions(cp, trades, n, adv_weight=adv_weight)
        objective = cp.Maximize(
            a @ w
            - 0.5 * cfg.risk_aversion * cp.quad_form(active_weights, cp.psd_wrap(sigma))
            - cfg.linear_cost_penalty * linear_cost
            - cfg.impact_cost_penalty * impact_cost
            - cfg.turnover_penalty * cp.sum_squares(trades)
        )
        constraints = strategy.constraints(cp, w, benchmark)
        constraints += self._common_constraints(cp, trades, active_weights, sigma, tradable, adv_weight, x)
        problem = cp.Problem(objective, constraints)
        status, solver = self._solve(cp, problem)
        if w.value is None or status not in {"optimal", "optimal_inaccurate"}:
            raise RuntimeError(f"portfolio optimisation failed: {status}")

        weights = np.asarray(w.value, dtype=float)
        weights[np.abs(weights) < 1e-10] = 0.0
        trades_value = weights - current
        active = strategy.active_weights(weights, benchmark)
        cost_estimate = self.cost_model.estimate(trades_value, adv_weight=adv_weight)
        linear_cost_value = float(cost_estimate.linear)
        impact_cost_value = float(cost_estimate.impact)
        cost = float(cost_estimate.total)
        variance = max(float(active @ sigma @ active), 0.0)
        return OptimizationResult(
            weights=weights,
            trades=trades_value,
            status=status,
            expected_return=float(a @ active),
            predicted_volatility=float(np.sqrt(variance)),
            turnover=float(np.abs(trades_value).sum()),
            expected_cost=cost,
            exposures=x.T @ active if x.size else np.empty(0, dtype=float),
            constraint_usage=self._constraint_usage(weights, active, sigma),
            diagnostics={
                "solver": solver,
                "objective_value": float(problem.value),
                "linear_cost": linear_cost_value,
                "impact_cost": impact_cost_value,
            },
            benchmark_weights=strategy.benchmark_output(benchmark),
            active_weights=active,
        )

    def _common_constraints(self, cp, trades, active_weights, sigma, tradable, adv_weight, exposures) -> list:
        cfg = self.config
        constraints = []
        if cfg.max_turnover is not None:
            constraints.append(cp.norm1(trades) <= cfg.max_turnover)
        if cfg.max_adv_participation is not None and adv_weight is not None:
            cap = np.clip(np.nan_to_num(adv_weight, nan=0.0), 0.0, None) * cfg.max_adv_participation
            constraints += [trades <= cap, trades >= -cap]
        if cfg.tracking_error_limit is not None:
            constraints.append(cp.quad_form(active_weights, cp.psd_wrap(sigma)) <= cfg.tracking_error_limit**2)
        for idx, lower in cfg.exposure_lower.items():
            constraints.append(exposures[:, int(idx)] @ active_weights >= lower)
        for idx, upper in cfg.exposure_upper.items():
            constraints.append(exposures[:, int(idx)] @ active_weights <= upper)
        if tradable is not None:
            locked = ~np.asarray(tradable, dtype=bool)
            if locked.any():
                constraints.append(trades[locked] == 0.0)
        return constraints

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

    def _constraint_usage(self, weights: np.ndarray, active: np.ndarray, sigma: np.ndarray) -> dict[str, float]:
        return {
            "gross_exposure": float(np.abs(weights).sum()),
            "net_exposure": float(weights.sum()),
            "largest_position": float(np.max(np.abs(weights))) if weights.size else 0.0,
            "tracking_error_or_volatility": float(np.sqrt(max(active @ sigma @ active, 0.0))),
        }












