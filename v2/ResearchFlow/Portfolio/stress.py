"""Deterministic portfolio stress testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..matrix_math import cov_to_vol_corr, nearest_psd


CovarianceMode = Literal["factor", "asset"]


@dataclass(frozen=True)
class StressResult:
    scenario_pnl: dict[str, float]
    factor_pnl: dict[str, np.ndarray]
    stressed_vol: dict[str, float]
    stressed_var: dict[str, float]
    portfolio_exposure: np.ndarray
    covariance_mode: CovarianceMode
    stressed_asset_covariance: dict[str, np.ndarray]


class StressTester:
    """Scenario shocks and covariance stress for one portfolio cross-section."""

    def __init__(self, *, repair_psd: bool = False, psd_floor: float = 1e-10) -> None:
        self.repair_psd = repair_psd
        self.psd_floor = psd_floor

    def run(
        self,
        weights: np.ndarray,
        exposures: np.ndarray,
        factor_shocks: dict[str, np.ndarray],
        *,
        factor_covariance: np.ndarray | None = None,
        asset_covariance: np.ndarray | None = None,
        specific_var: np.ndarray | None = None,
        covariance_vol_multipliers: dict[str, float | np.ndarray] | None = None,
        covariance_corr_blends: dict[str, float] | None = None,
        covariance_crisis_corrs: dict[str, np.ndarray] | None = None,
        specific_vol_multipliers: dict[str, float] | None = None,
    ) -> StressResult:
        w = np.asarray(weights, dtype=float).reshape(-1)
        x = np.asarray(exposures, dtype=float)
        mode: CovarianceMode = "factor" if factor_covariance is not None else "asset"
        covariance = np.asarray(factor_covariance if mode == "factor" else asset_covariance, dtype=float)
        spec = None if specific_var is None else np.asarray(specific_var, dtype=float).reshape(-1)

        portfolio_exposure = x.T @ w
        scenario_pnl: dict[str, float] = {}
        factor_pnl: dict[str, np.ndarray] = {}
        stressed_vol: dict[str, float] = {}
        stressed_var: dict[str, float] = {}
        stressed_asset_covariance: dict[str, np.ndarray] = {}

        for scenario, shock in factor_shocks.items():
            contribution = portfolio_exposure * np.asarray(shock, dtype=float).reshape(-1)
            scenario_pnl[scenario] = float(np.nansum(contribution))
            factor_pnl[scenario] = contribution

            asset_cov = self._scenario_asset_covariance(
                mode=mode,
                exposures=x,
                covariance=covariance,
                specific_var=spec,
                covariance_vol_multiplier=_scenario_value(covariance_vol_multipliers, scenario, 1.0),
                covariance_corr_blend=float(_scenario_value(covariance_corr_blends, scenario, 0.0)),
                covariance_crisis_corr=_scenario_value(covariance_crisis_corrs, scenario, None),
                specific_vol_multiplier=float(_scenario_value(specific_vol_multipliers, scenario, 1.0)),
            )
            stressed_asset_covariance[scenario] = asset_cov
            stressed_var[scenario] = max(float(w @ asset_cov @ w), 0.0)
            stressed_vol[scenario] = float(np.sqrt(stressed_var[scenario]))

        return StressResult(
            scenario_pnl=scenario_pnl,
            factor_pnl=factor_pnl,
            stressed_vol=stressed_vol,
            stressed_var=stressed_var,
            portfolio_exposure=portfolio_exposure,
            covariance_mode=mode,
            stressed_asset_covariance=stressed_asset_covariance,
        )

    def _scenario_asset_covariance(
        self,
        *,
        mode: CovarianceMode,
        exposures: np.ndarray,
        covariance: np.ndarray,
        specific_var: np.ndarray | None,
        covariance_vol_multiplier: float | np.ndarray,
        covariance_corr_blend: float,
        covariance_crisis_corr: np.ndarray | None,
        specific_vol_multiplier: float,
    ) -> np.ndarray:
        if mode == "factor":
            factor_cov = self._stress_covariance(
                covariance,
                vol_multiplier=covariance_vol_multiplier,
                corr_blend=covariance_corr_blend,
                crisis_corr=covariance_crisis_corr,
            )
            asset_cov = exposures @ factor_cov @ exposures.T
            if specific_var is not None:
                asset_cov += np.diag(specific_var * specific_vol_multiplier**2)
        else:
            asset_cov = covariance.copy()
            if covariance_crisis_corr is not None and covariance_corr_blend > 0.0:
                asset_cov = self._stress_covariance(
                    asset_cov,
                    vol_multiplier=covariance_vol_multiplier,
                    corr_blend=covariance_corr_blend,
                    crisis_corr=covariance_crisis_corr,
                )
            if specific_var is not None and specific_vol_multiplier != 1.0:
                diag = np.diag_indices_from(asset_cov)
                asset_cov[diag] += specific_var * (specific_vol_multiplier**2 - 1.0)
        return self._finalize_covariance(asset_cov)

    def _stress_covariance(
        self,
        covariance: np.ndarray,
        *,
        vol_multiplier: float | np.ndarray,
        corr_blend: float,
        crisis_corr: np.ndarray | None,
    ) -> np.ndarray:
        vol, corr = cov_to_vol_corr(np.asarray(covariance, dtype=float))
        multiplier = np.asarray(vol_multiplier, dtype=float)
        stressed_vol = vol * (multiplier if multiplier.ndim else float(multiplier))

        if crisis_corr is not None and corr_blend > 0.0:
            crisis = np.asarray(crisis_corr, dtype=float)
            corr = (1.0 - corr_blend) * corr + corr_blend * crisis
            corr = 0.5 * (corr + corr.T)
            np.fill_diagonal(corr, 1.0)

        return corr * np.outer(stressed_vol, stressed_vol)

    def _finalize_covariance(self, covariance: np.ndarray) -> np.ndarray:
        cov = 0.5 * (covariance + covariance.T)
        return nearest_psd(cov, self.psd_floor) if self.repair_psd else cov


def _scenario_value(data: dict | None, scenario: str, default):
    return default if data is None else data.get(scenario, default)
