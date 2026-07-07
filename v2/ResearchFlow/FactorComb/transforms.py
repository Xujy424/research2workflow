"""Matrix-native factor transforms: residualisation, orthogonalisation, PCA, and PLS."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..matrix_math import neutralize_by_exposures
from .orthogonal import FactorOrthogonalizer


@dataclass(frozen=True)
class TransformResult:
    values: np.ndarray
    diagnostics: dict[str, object]


class FactorTransformer:
    """Reusable transforms for ``T x N`` and ``T x N x K`` matrices."""

    def __init__(self) -> None:
        self._orthogonalizer = FactorOrthogonalizer()

    def orthogonalize(
        self,
        factors: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        method: str = "symmetric",
        ridge: float = 1e-6,
    ) -> TransformResult:
        result = self._orthogonalizer.orthogonalize(factors, mask=mask, method=method, ridge=ridge)
        return TransformResult(result.residual, dict(result.diagnostics))

    def residualize(self, y: np.ndarray, x: np.ndarray, *, mask: np.ndarray | None = None) -> TransformResult:
        residual = neutralize_by_exposures(y, x, mask=mask)
        return TransformResult(residual, {"method": "residualize", "n_exposures": float(x.shape[2])})

    def pca(self, factors: np.ndarray, *, n_components: int, mask: np.ndarray | None = None) -> TransformResult:
        return self._static_projection(factors, None, n_components=n_components, mask=mask, supervised=False)

    def pls_one_component(self, factors: np.ndarray, labels: np.ndarray, *, mask: np.ndarray | None = None) -> TransformResult:
        return self.pls(factors, labels, n_components=1, mask=mask)

    def pls(self, factors: np.ndarray, labels: np.ndarray, *, n_components: int, mask: np.ndarray | None = None) -> TransformResult:
        return self._static_projection(factors, labels, n_components=n_components, mask=mask, supervised=True)

    def walk_forward_pca(
        self,
        factors: np.ndarray,
        *,
        n_components: int,
        lookback: int,
        min_periods: int,
        mask: np.ndarray | None = None,
    ) -> TransformResult:
        return self._walk_forward_projection(
            factors,
            None,
            n_components=n_components,
            lookback=lookback,
            min_periods=min_periods,
            mask=mask,
            supervised=False,
        )

    def walk_forward_pls(
        self,
        factors: np.ndarray,
        labels: np.ndarray,
        *,
        n_components: int,
        lookback: int,
        min_periods: int,
        mask: np.ndarray | None = None,
    ) -> TransformResult:
        return self._walk_forward_projection(
            factors,
            labels,
            n_components=n_components,
            lookback=lookback,
            min_periods=min_periods,
            mask=mask,
            supervised=True,
        )

    def _static_projection(
        self,
        factors: np.ndarray,
        labels: np.ndarray | None,
        *,
        n_components: int,
        mask: np.ndarray | None,
        supervised: bool,
    ) -> TransformResult:
        if factors.ndim != 3:
            raise ValueError("factors must have shape T x N x K")
        n_components = min(n_components, factors.shape[2])
        x = factors.reshape(-1, factors.shape[2])
        valid = np.isfinite(x).all(axis=1)
        if mask is not None:
            valid &= mask.reshape(-1)
        y = None if labels is None else labels.reshape(-1)
        if supervised:
            if y is None:
                raise ValueError("PLS requires labels")
            valid &= np.isfinite(y)
        if valid.sum() <= n_components:
            raise ValueError("insufficient observations for transform")
        x_train = x[valid]
        x_mean, x_std = self._fit_scaler(x_train)
        x_scaled = (x_train - x_mean) / x_std
        if supervised:
            rotation, diagnostics = self._fit_pls_rotation(x_scaled, y[valid], n_components)
        else:
            rotation, diagnostics = self._fit_pca_rotation(x_scaled, n_components)
        projected = np.full((x.shape[0], n_components), np.nan, dtype=float)
        row_valid = np.isfinite(x).all(axis=1)
        projected[row_valid] = ((x[row_valid] - x_mean) / x_std) @ rotation
        return TransformResult(projected.reshape(factors.shape[0], factors.shape[1], n_components), diagnostics)

    def _walk_forward_projection(
        self,
        factors: np.ndarray,
        labels: np.ndarray | None,
        *,
        n_components: int,
        lookback: int,
        min_periods: int,
        mask: np.ndarray | None,
        supervised: bool,
    ) -> TransformResult:
        if factors.ndim != 3:
            raise ValueError("factors must have shape T x N x K")
        if supervised and labels is None:
            raise ValueError("PLS requires labels")
        n_components = min(n_components, factors.shape[2])
        out = np.full((factors.shape[0], factors.shape[1], n_components), np.nan, dtype=float)
        explained: dict[int, list[float]] = {}
        valid_mask = np.ones(factors.shape[:2], dtype=bool) if mask is None else mask.astype(bool)
        for t in range(factors.shape[0]):
            start = max(0, t - lookback)
            if t - start < min_periods:
                continue
            x_hist = factors[start:t].reshape(-1, factors.shape[2])
            train_valid = np.isfinite(x_hist).all(axis=1) & valid_mask[start:t].reshape(-1)
            y_hist = None if labels is None else labels[start:t].reshape(-1)
            if supervised:
                train_valid &= np.isfinite(y_hist)
            if train_valid.sum() <= n_components:
                continue
            x_train = x_hist[train_valid]
            x_mean, x_std = self._fit_scaler(x_train)
            x_scaled = (x_train - x_mean) / x_std
            if supervised:
                rotation, diagnostics = self._fit_pls_rotation(x_scaled, y_hist[train_valid], n_components)
            else:
                rotation, diagnostics = self._fit_pca_rotation(x_scaled, n_components)
                explained[t] = diagnostics["explained_variance"]
            x_now = factors[t]
            now_valid = valid_mask[t] & np.isfinite(x_now).all(axis=1)
            if now_valid.any():
                out[t, now_valid] = ((x_now[now_valid] - x_mean) / x_std) @ rotation
        diagnostics = {"method": "pls" if supervised else "pca", "lookback": lookback, "min_periods": min_periods}
        if not supervised:
            diagnostics["explained_variance"] = explained
        return TransformResult(out, diagnostics)

    @staticmethod
    def _fit_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0)
        std[std <= 1e-12] = 1.0
        return mean, std

    @staticmethod
    def _fit_pca_rotation(x_scaled: np.ndarray, n_components: int) -> tuple[np.ndarray, dict[str, object]]:
        _, singular, vt = np.linalg.svd(x_scaled, full_matrices=False)
        rotation = vt[:n_components].T
        explained = (singular[:n_components] ** 2 / np.maximum(np.sum(singular**2), 1e-12)).tolist()
        return rotation, {"method": "pca", "explained_variance": explained}

    @staticmethod
    def _fit_pls_rotation(x_scaled: np.ndarray, y: np.ndarray, n_components: int) -> tuple[np.ndarray, dict[str, object]]:
        try:
            from sklearn.cross_decomposition import PLSRegression
        except ImportError as exc:
            raise ImportError("PLS transform requires scikit-learn") from exc
        model = PLSRegression(n_components=n_components, scale=False)
        model.fit(x_scaled, y)
        return model.x_rotations_, {"method": "pls", "n_components": n_components}
