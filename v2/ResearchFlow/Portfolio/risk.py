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


