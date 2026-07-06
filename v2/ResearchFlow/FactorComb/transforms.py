"""Orthogonalization and dimensionality reduction transforms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..matrix_math import neutralize_by_exposures


@dataclass(frozen=True)
class TransformResult:
    values: np.ndarray
    diagnostics: dict[str, float]


class FactorTransformer:
    """Reusable transforms for ``T x N`` and ``T x N x K`` matrices."""

    def orthogonalize(self, y: np.ndarray, x: np.ndarray, *, mask: np.ndarray | None = None) -> TransformResult:
        residual = neutralize_by_exposures(y, x, mask=mask)
        return TransformResult(residual, {"n_exposures": float(x.shape[2])})

    def pca(self, factors: np.ndarray, *, n_components: int, mask: np.ndarray | None = None) -> TransformResult:
        if factors.ndim != 3:
            raise ValueError("factors must have shape T x N x K")
        flat = factors.reshape(-1, factors.shape[2])
        if mask is not None:
            flat = flat[mask.reshape(-1)]
        valid = np.isfinite(flat).all(axis=1)
        clean = flat[valid]
        if clean.shape[0] <= n_components:
            raise ValueError("insufficient observations for PCA")
        mean = clean.mean(axis=0)
        centered = clean - mean
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[:n_components]
        projected = np.full((factors.shape[0] * factors.shape[1], n_components), np.nan)
        all_flat = factors.reshape(-1, factors.shape[2])
        all_valid = np.isfinite(all_flat).all(axis=1)
        projected[all_valid] = (all_flat[all_valid] - mean) @ components.T
        explained = singular[:n_components] ** 2 / np.maximum(np.sum(singular**2), 1e-12)
        return TransformResult(
            projected.reshape(factors.shape[0], factors.shape[1], n_components),
            {"explained_variance": float(explained.sum())},
        )

    def pls_one_component(self, factors: np.ndarray, labels: np.ndarray, *, mask: np.ndarray | None = None) -> TransformResult:
        if factors.ndim != 3:
            raise ValueError("factors must have shape T x N x K")
        x = factors.reshape(-1, factors.shape[2])
        y = labels.reshape(-1)
        valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
        if mask is not None:
            valid &= mask.reshape(-1)
        if valid.sum() <= factors.shape[2]:
            raise ValueError("insufficient observations for PLS")
        xv = x[valid] - np.nanmean(x[valid], axis=0)
        yv = y[valid] - np.nanmean(y[valid])
        coef = xv.T @ yv
        norm = np.linalg.norm(coef)
        coef = coef / norm if norm > 1e-12 else np.ones(factors.shape[2]) / factors.shape[2]
        score = np.full(x.shape[0], np.nan)
        row_valid = np.isfinite(x).all(axis=1)
        score[row_valid] = x[row_valid] @ coef
        return TransformResult(score.reshape(labels.shape), {"coef_norm": float(norm)})


