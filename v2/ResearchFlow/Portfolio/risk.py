"""Lightweight matrix risk helpers for portfolio construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RiskEstimate:
    covariance: np.ndarray
    specific_var: np.ndarray
    diagnostics: dict[str, float]


def nearest_psd(matrix: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2
    values, vectors = np.linalg.eigh(symmetric)
    values = np.maximum(values, floor)
    repaired = (vectors * values) @ vectors.T
    return (repaired + repaired.T) / 2


class MatrixRiskModel:
    """Estimate stock covariance from returns with diagonal shrinkage."""

    def __init__(self, *, shrinkage: float = 0.30, variance_floor: float = 1e-8) -> None:
        self.shrinkage = shrinkage
        self.variance_floor = variance_floor

    def fit(self, returns: np.ndarray, *, lookback: int | None = None) -> RiskEstimate:
        hist = np.asarray(returns[-lookback:] if lookback else returns, dtype=float)
        hist = hist[np.isfinite(hist).all(axis=1)]
        if len(hist) < 2:
            raise ValueError("insufficient clean return history")
        cov = np.cov(hist, rowvar=False)
        diag = np.diag(np.diag(cov))
        cov = (1.0 - self.shrinkage) * cov + self.shrinkage * diag
        cov = nearest_psd(cov, self.variance_floor)
        return RiskEstimate(
            covariance=cov,
            specific_var=np.maximum(np.diag(cov), self.variance_floor),
            diagnostics={
                "condition_number": float(np.linalg.cond(cov)),
                "min_eigenvalue": float(np.linalg.eigvalsh(cov).min()),
            },
        )




@dataclass(frozen=True)
class FactorRiskEstimate:
    factor_returns: np.ndarray
    residual_returns: np.ndarray
    factor_covariance: np.ndarray
    specific_var: np.ndarray
    stock_covariance: np.ndarray
    diagnostics: dict[str, float]


class MatrixFactorRiskModel:
    """Barra-style factor risk model on matrix inputs."""

    def __init__(
        self,
        *,
        factor_halflife: float = 60.0,
        specific_halflife: float = 60.0,
        newey_west_lags: int = 5,
        covariance_shrinkage: float = 0.20,
        specific_shrinkage: float = 0.20,
        variance_floor: float = 1e-8,
        annualization: float = 252.0,
    ) -> None:
        self.factor_halflife = factor_halflife
        self.specific_halflife = specific_halflife
        self.newey_west_lags = newey_west_lags
        self.covariance_shrinkage = covariance_shrinkage
        self.specific_shrinkage = specific_shrinkage
        self.variance_floor = variance_floor
        self.annualization = annualization

    def fit(
        self,
        asset_returns: np.ndarray,
        exposure_history: np.ndarray,
        current_exposures: np.ndarray,
        *,
        market_cap_history: np.ndarray | None = None,
        mask: np.ndarray | None = None,
    ) -> FactorRiskEstimate:
        factor_returns, residual_returns, skipped = self.estimate_factor_returns(
            asset_returns,
            exposure_history,
            market_cap_history=market_cap_history,
            mask=mask,
        )
        factor_cov = self._factor_covariance(factor_returns)
        specific_var = self._specific_variance(residual_returns)
        current_x = np.nan_to_num(np.asarray(current_exposures, dtype=float), nan=0.0)
        stock_cov = current_x @ factor_cov @ current_x.T + np.diag(specific_var)
        stock_cov = nearest_psd(stock_cov, self.variance_floor)
        return FactorRiskEstimate(
            factor_returns=factor_returns,
            residual_returns=residual_returns,
            factor_covariance=factor_cov,
            specific_var=specific_var,
            stock_covariance=stock_cov,
            diagnostics={
                "n_factor_return_dates": float(np.isfinite(factor_returns).all(axis=1).sum()),
                "n_skipped_dates": float(skipped),
                "condition_number": float(np.linalg.cond(stock_cov)),
                "min_eigenvalue": float(np.linalg.eigvalsh(stock_cov).min()),
            },
        )

    def estimate_factor_returns(
        self,
        asset_returns: np.ndarray,
        exposure_history: np.ndarray,
        *,
        market_cap_history: np.ndarray | None = None,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        returns = np.asarray(asset_returns, dtype=float)
        exposures = np.asarray(exposure_history, dtype=float)
        if exposures.ndim != 3 or exposures.shape[:2] != returns.shape:
            raise ValueError("exposure_history must be shaped T x N x K and aligned with returns")
        valid = np.isfinite(returns) & np.isfinite(exposures).all(axis=2)
        if mask is not None:
            valid &= np.asarray(mask, dtype=bool)
        cap = None if market_cap_history is None else np.asarray(market_cap_history, dtype=float)
        t_count, n_assets, n_factors = exposures.shape
        factor_returns = np.full((t_count, n_factors), np.nan, dtype=float)
        residual_returns = np.full((t_count, n_assets), np.nan, dtype=float)
        skipped = 0
        for t in range(t_count):
            ok = valid[t]
            if ok.sum() <= n_factors + 2:
                skipped += 1
                continue
            x = exposures[t, ok]
            y = returns[t, ok]
            if cap is None:
                weights = np.ones_like(y)
            else:
                weights = np.sqrt(np.clip(cap[t, ok], 1.0, None))
                weights /= max(float(np.nanmedian(weights)), 1e-12)
            xw = x * weights[:, None]
            yw = y * weights
            ridge = self.variance_floor * np.eye(n_factors)
            beta = np.linalg.solve(xw.T @ xw + ridge, xw.T @ yw)
            factor_returns[t] = beta
            residual_returns[t, ok] = y - x @ beta
        return factor_returns, residual_returns, skipped

    def _factor_covariance(self, factor_returns: np.ndarray) -> np.ndarray:
        clean = np.asarray(factor_returns, dtype=float)
        clean = clean[np.isfinite(clean).all(axis=1)]
        if len(clean) < 2:
            raise ValueError("insufficient history to estimate factor covariance")
        weights = exponential_weights(len(clean), self.factor_halflife)
        centered = clean - np.average(clean, axis=0, weights=weights)
        cov = (centered * weights[:, None]).T @ centered
        for lag in range(1, min(self.newey_west_lags, len(clean) - 1) + 1):
            kernel = 1.0 - lag / (self.newey_west_lags + 1.0)
            cross = (centered[lag:] * weights[lag:, None]).T @ centered[:-lag]
            cov += kernel * (cross + cross.T)
        diag = np.diag(np.diag(cov))
        cov = (1.0 - self.covariance_shrinkage) * cov + self.covariance_shrinkage * diag
        return nearest_psd(cov * self.annualization, self.variance_floor)

    def _specific_variance(self, residuals: np.ndarray) -> np.ndarray:
        values = np.asarray(residuals, dtype=float)
        weights = exponential_weights(values.shape[0], self.specific_halflife)
        valid = np.isfinite(values)
        denom = (valid * weights[:, None]).sum(axis=0)
        means = np.divide(
            np.nansum(np.where(valid, values, 0.0) * weights[:, None], axis=0),
            denom,
            out=np.zeros(values.shape[1]),
            where=denom > 0,
        )
        var = np.divide(
            np.nansum(np.where(valid, (values - means) ** 2, 0.0) * weights[:, None], axis=0),
            denom,
            out=np.full(values.shape[1], np.nan),
            where=denom > 0,
        )
        median = np.nanmedian(var)
        shrunk = (1.0 - self.specific_shrinkage) * var + self.specific_shrinkage * median
        return np.maximum(np.nan_to_num(shrunk * self.annualization, nan=median), self.variance_floor)


def exponential_weights(length: int, halflife: float) -> np.ndarray:
    if length <= 0:
        return np.empty(0, dtype=float)
    ages = np.arange(length - 1, -1, -1, dtype=float)
    weights = np.power(0.5, ages / max(float(halflife), 1e-12))
    return weights / weights.sum()


def risk_attribution(weights: np.ndarray, exposures: np.ndarray, factor_covariance: np.ndarray, specific_var: np.ndarray) -> dict[str, np.ndarray | float]:
    w = np.asarray(weights, dtype=float).reshape(-1)
    x = np.asarray(exposures, dtype=float)
    factor_cov = np.asarray(factor_covariance, dtype=float)
    spec = np.asarray(specific_var, dtype=float).reshape(-1)
    factor_exposure = x.T @ w
    factor_contribution = factor_exposure * (factor_cov @ factor_exposure)
    specific_contribution = float(np.sum((w * w) * spec))
    total = float(factor_contribution.sum() + specific_contribution)
    return {
        "factor_exposure": factor_exposure,
        "factor_variance_contribution": factor_contribution,
        "specific_variance_contribution": specific_contribution,
        "total_variance": total,
        "factor_share": factor_contribution / total if total > 1e-12 else np.full_like(factor_contribution, np.nan),
        "specific_share": specific_contribution / total if total > 1e-12 else np.nan,
    }
