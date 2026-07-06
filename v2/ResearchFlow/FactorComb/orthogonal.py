"""Orthogonalization methods for factor matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..matrix_math import cross_sectional_zscore, neutralize_by_exposures


@dataclass(frozen=True)
class OrthogonalResult:
    residual: np.ndarray
    diagnostics: dict[str, float]


class FactorOrthogonalizer:
    """Cross-sectional residualization utilities.

    Inputs are matrix-native: target is ``T x N`` and exposures are
    ``T x N x K``. The class does not load data and does not depend on v1.
    """

    def residualize(
        self,
        target: np.ndarray,
        exposures: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        weights: np.ndarray | None = None,
        standardize: bool = True,
    ) -> OrthogonalResult:
        residual = neutralize_by_exposures(target, exposures, mask=mask, weights=weights)
        if standardize:
            residual = cross_sectional_zscore(residual, mask=mask)
        return OrthogonalResult(
            residual=residual,
            diagnostics={"n_exposures": float(exposures.shape[2])},
        )

    def residualize_against_pool(
        self,
        target: np.ndarray,
        pool: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        standardize: bool = True,
    ) -> OrthogonalResult:
        if pool.ndim != 3:
            raise ValueError("pool must have shape T x N x K")
        return self.residualize(target, pool, mask=mask, standardize=standardize)

    def sequential_orthogonalize(
        self,
        factors: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        order: np.ndarray | None = None,
        standardize: bool = True,
    ) -> OrthogonalResult:
        if factors.ndim != 3:
            raise ValueError("factors must have shape T x N x K")
        k = factors.shape[2]
        order_arr = np.arange(k) if order is None else np.asarray(order, dtype=int)
        out = np.full_like(factors, np.nan, dtype=float)
        accepted: list[np.ndarray] = []
        for position in order_arr:
            target = factors[:, :, position]
            if accepted:
                exposures = np.stack(accepted, axis=2)
                residual = self.residualize(target, exposures, mask=mask, standardize=standardize).residual
            else:
                residual = cross_sectional_zscore(target, mask=mask) if standardize else target.astype(float)
            out[:, :, position] = residual
            accepted.append(residual)
        return OrthogonalResult(out, {"n_factors": float(k)})
