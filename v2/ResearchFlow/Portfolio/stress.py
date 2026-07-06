"""Portfolio stress testing for stock weights and risk inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StressResult:
    scenario_pnl: dict[str, float]
    factor_pnl: dict[str, np.ndarray]
    stressed_volatility: dict[str, float]


class StressTester:
    """Deterministic stress test using exposures, factor shocks and covariance."""

    def run(
        self,
        weights: np.ndarray,
        exposures: np.ndarray,
        covariance: np.ndarray,
        factor_shocks: dict[str, np.ndarray],
        *,
        specific_var: np.ndarray | None = None,
        specific_vol_multipliers: dict[str, float] | None = None,
    ) -> StressResult:
        w = np.asarray(weights, dtype=float).reshape(-1)
        x = np.asarray(exposures, dtype=float)
        if x.shape[0] != len(w):
            raise ValueError("exposures must have shape N x K aligned with weights")
        portfolio_exposure = x.T @ w
        scenario_pnl: dict[str, float] = {}
        factor_pnl: dict[str, np.ndarray] = {}
        stressed_vol: dict[str, float] = {}
        for scenario, shock in factor_shocks.items():
            shock_arr = np.asarray(shock, dtype=float).reshape(-1)
            contribution = portfolio_exposure * shock_arr
            scenario_pnl[scenario] = float(np.nansum(contribution))
            factor_pnl[scenario] = contribution
            cov = np.asarray(covariance, dtype=float).copy()
            if specific_var is not None:
                multiplier = 1.0 if specific_vol_multipliers is None else specific_vol_multipliers.get(scenario, 1.0)
                cov = cov + np.diag(np.asarray(specific_var, dtype=float) * (multiplier**2 - 1.0))
            stressed_vol[scenario] = float(np.sqrt(max(w @ cov @ w, 0.0)))
        return StressResult(scenario_pnl, factor_pnl, stressed_vol)
