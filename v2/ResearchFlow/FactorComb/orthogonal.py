"""Orthogonalization methods for factor matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..matrix_math import cross_sectional_zscore, neutralize_by_exposures


@dataclass(frozen=True)
class OrthogonalResult:
    residual: np.ndarray
    diagnostics: dict[str, float | str]


class FactorOrthogonalizer:
    """Cross-sectional residualization and orthogonalization utilities."""

    def residualize(
        self,
        target: np.ndarray,
        exposures: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        weights: np.ndarray | None = None,
        standardize: bool = True,
        ridge: float = 1e-8,
    ) -> OrthogonalResult:
        residual = neutralize_by_exposures(target, exposures, mask=mask, weights=weights, ridge=ridge)
        if standardize:
            residual = cross_sectional_zscore(residual, mask=mask)
        return OrthogonalResult(residual=residual, diagnostics={"n_exposures": float(exposures.shape[2])})

    def residualize_against_pool(
        self,
        target: np.ndarray,
        pool: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        standardize: bool = True,
        ridge: float = 1e-8,
    ) -> OrthogonalResult:
        if pool.ndim != 3:
            raise ValueError("pool must have shape T x N x K")
        return self.residualize(target, pool, mask=mask, standardize=standardize, ridge=ridge)

    def orthogonalize(
        self,
        factors: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        method: str = "symmetric",
        ridge: float = 1e-6,
        standardize: bool = True,
    ) -> OrthogonalResult:
        """Orthogonalize factors cross-sectionally by date.

        ``method='symmetric'`` uses symmetric whitening from the covariance
        eigen-decomposition. ``method='sequential'`` uses QR orthogonalization.
        Both methods match the v1 transform semantics but operate on T x N x K
        numpy matrices.
        """
        if factors.ndim != 3:
            raise ValueError("factors must have shape T x N x K")
        if method not in {"symmetric", "sequential"}:
            raise ValueError("orthogonalization must be sequential or symmetric")
        
        valid_mask = np.ones(factors.shape[:2], dtype=bool) if mask is None else mask.astype(bool)
        out = np.full_like(factors, np.nan, dtype=float)
        for t in range(factors.shape[0]):
            valid = valid_mask[t] & np.isfinite(factors[t]).all(axis=1)
            min_required = factors.shape[2] + 2 if method == "sequential" else 2
            if valid.sum() < min_required:
                continue
            centered = factors[t, valid] - factors[t, valid].mean(axis=0, keepdims=True)
            out[t, valid] = self._orthogonal_cross_section(centered, method=method, ridge=ridge)
        if standardize:
            out = np.stack([cross_sectional_zscore(out[:, :, k], mask=valid_mask) for k in range(out.shape[2])], axis=2)
        return OrthogonalResult(out, {"method": method, "n_factors": float(factors.shape[2])})

    def ordered_residualize(
        self,
        factors: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        order: np.ndarray | None = None,
        standardize: bool = True,
    ) -> OrthogonalResult:
        """Residualize each factor against previously accepted factors.

        This is not QR orthogonalization; it is an ordered incremental-residual
        transform useful when representative quality defines the factor order.
        """
        if factors.ndim != 3:
            raise ValueError("factors must have shape T x N x K")
        order_arr = np.arange(factors.shape[2]) if order is None else np.asarray(order, dtype=int)
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
        return OrthogonalResult(out, {"method": "ordered_residual", "n_factors": float(factors.shape[2])})

    @staticmethod
    def _orthogonal_cross_section(values: np.ndarray, *, method: str, ridge: float) -> np.ndarray:
        if method == "sequential":
            if values.shape[0] < values.shape[1]:
                raise ValueError("sequential QR orthogonalization requires observations >= factors; use symmetric or ordered_residual for wide matrices")
            q, _ = np.linalg.qr(values, mode="reduced")
            return q * np.sqrt(len(q))

        u, singular, vt = np.linalg.svd(values, full_matrices=False)
        scale = np.divide(
            singular,
            np.sqrt(singular * singular / len(values) + ridge),
            out=np.zeros_like(singular),
            where=singular > 1e-12,
        )
        return (u * scale) @ vt
