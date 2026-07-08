"""Capital allocation methods shared by alpha and sleeve paths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from scipy.optimize import minimize
except ImportError:  # optional; analytic fallbacks keep the pipeline usable
    minimize = None

from ..matrix_math import cap_and_renormalize, nearest_psd


EPS = 1e-12


@dataclass(frozen=True)
class AllocationParams:
    method: str = "icir"
    lookback: int = 252
    min_periods: int = 60
    max_weight: float = 0.60
    smoothing: float = 0.80
    return_shrinkage: float = 0.70
    covariance_shrinkage: float = 0.30
    turnover_penalty: float = 0.05
    risk_aversion: float = 5.0


class CapitalAllocator:
    """Rolling allocator for family alpha weights or sleeve capital weights."""

    ANALYTIC_METHODS = {"equal", "icir", "correlation_adjusted"}
    OPTIMIZATION_METHODS = {"minimum_variance", "mean_variance", "risk_parity"}

    def __init__(self, params: AllocationParams) -> None:
        self.params = params

    def allocate(self, returns: np.ndarray | None, n_dates: int, n_assets: int) -> np.ndarray:
        method = self.params.method.lower()
        if returns is None or method == "equal":
            return np.full((n_dates, n_assets), 1.0 / n_assets, dtype=float)
        if method not in self.ANALYTIC_METHODS | self.OPTIMIZATION_METHODS:
            raise ValueError(f"unsupported allocation method: {self.params.method}")

        values = np.asarray(returns, dtype=float)
        out = np.full((n_dates, n_assets), 1.0 / n_assets, dtype=float)
        prev = out[0]
        for t in range(n_dates):
            hist = values[max(0, t - self.params.lookback):t]
            hist = hist[np.isfinite(hist).all(axis=1)]
            if len(hist) < self.params.min_periods:
                out[t] = prev
                continue
            target = self._allocate_window(hist, prev, method)
            out[t] = self.params.smoothing * prev + (1.0 - self.params.smoothing) * target
            out[t] = self._normalize(out[t])
            prev = out[t]
        return out

    def _allocate_window(self, returns: np.ndarray, previous: np.ndarray, method: str) -> np.ndarray:
        mean = np.nanmean(returns, axis=0) * (1.0 - self.params.return_shrinkage)
        if method == "icir":
            std = np.nanstd(returns, axis=0)
            score = np.divide(mean, std, out=np.zeros_like(mean), where=std > EPS)
            return self._normalize(score)

        cov = self._covariance(returns)
        if method == "correlation_adjusted":
            score = np.linalg.pinv(cov) @ np.maximum(mean, 0.0)
            return self._normalize(score)
        if method == "minimum_variance":
            return self._convex_allocate(np.zeros_like(mean), cov, previous, mean_variance=False)
        if method == "mean_variance":
            return self._convex_allocate(mean, cov, previous, mean_variance=True)
        if method == "risk_parity":
            return self._risk_parity(cov, previous)
        return np.full_like(previous, 1.0 / len(previous))

    def _covariance(self, returns: np.ndarray) -> np.ndarray:
        cov = np.cov(returns, rowvar=False)
        if cov.ndim == 0:
            cov = np.asarray([[float(cov)]])
        diag = np.diag(np.diag(cov))
        cov = (1.0 - self.params.covariance_shrinkage) * cov + self.params.covariance_shrinkage * diag
        return nearest_psd(cov, 1e-10)

    def _convex_allocate(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        previous: np.ndarray,
        *,
        mean_variance: bool,
    ) -> np.ndarray:
        if minimize is None:
            raw = np.linalg.pinv(covariance) @ (np.maximum(mean, 0.0) if mean_variance else np.ones_like(mean))
            return self._normalize(raw)

        def objective(weight: np.ndarray) -> float:
            risk = 0.5 * self.params.risk_aversion * float(weight @ covariance @ weight)
            ret = float(mean @ weight) if mean_variance else 0.0
            turnover = self.params.turnover_penalty * float(np.sum((weight - previous) ** 2))
            return risk - ret + turnover

        result = minimize(
            objective,
            previous,
            method="SLSQP",
            bounds=[(0.0, self.params.max_weight)] * len(previous),
            constraints={"type": "eq", "fun": lambda weight: weight.sum() - 1.0},
            options={"maxiter": 500, "ftol": 1e-12},
        )
        return self._normalize(result.x) if result.success else previous

    def _risk_parity(self, covariance: np.ndarray, previous: np.ndarray) -> np.ndarray:
        if minimize is None:
            vol = np.sqrt(np.maximum(np.diag(covariance), EPS))
            return self._normalize(np.divide(1.0, vol, out=np.zeros_like(vol), where=vol > EPS))

        def objective(weight: np.ndarray) -> float:
            marginal = covariance @ weight
            contribution = weight * marginal
            target = contribution.sum() / len(weight)
            return float(np.sum((contribution - target) ** 2))

        result = minimize(
            objective,
            previous,
            method="SLSQP",
            bounds=[(1e-8, self.params.max_weight)] * len(previous),
            constraints={"type": "eq", "fun": lambda weight: weight.sum() - 1.0},
            options={"maxiter": 500, "ftol": 1e-12},
        )
        return self._normalize(result.x) if result.success else previous

    def _normalize(self, raw: np.ndarray) -> np.ndarray:
        values = np.maximum(np.asarray(raw, dtype=float), 0.0)
        if np.nansum(values) <= EPS:
            values = np.ones_like(values)
        return cap_and_renormalize(values, max_weight=self.params.max_weight)



