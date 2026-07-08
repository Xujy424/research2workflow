"""Unified family transform entry for orthogonal, PCA, and PLS methods."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from sklearn.cross_decomposition import PLSRegression
except ImportError:  # optional dependency, only required when transform_method="pls"
    PLSRegression = None

from ..config import FamilyConfig
from ..matrix_math import cross_sectional_zscore, neutralize_by_exposures


EPS = 1e-12


@dataclass(frozen=True)
class FamilyTransformResult:
    values: np.ndarray
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class ProjectionFit:
    rotation: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    diagnostics: dict[str, object]


class OrthogonalTransform:
    """Orthogonal and ordered-residual transforms for family factors."""

    def run(
        self,
        factors: np.ndarray,
        *,
        mask: np.ndarray,
        method: str,
        ridge: float,
        quality: np.ndarray,
    ) -> FamilyTransformResult:
        if method == "ordered_residual":
            order = np.argsort(-np.nan_to_num(np.abs(quality), nan=-np.inf))
            values = self._ordered_residualize(factors, mask=mask, order=order)
            return FamilyTransformResult(values, {"method": "ordered_residual", "n_factors": float(factors.shape[2])})
        values = self._orthogonalize(factors, mask=mask, method=method, ridge=ridge)
        return FamilyTransformResult(values, {"method": method, "n_factors": float(factors.shape[2])})

    def _orthogonalize(self, factors: np.ndarray, *, mask: np.ndarray, method: str, ridge: float) -> np.ndarray:
        if factors.ndim != 3:
            raise ValueError("factors must have shape T x N x K")
        if method not in {"symmetric", "sequential"}:
            raise ValueError("orthogonalization must be symmetric, sequential, or ordered_residual")
        valid_mask = np.asarray(mask, dtype=bool)
        out = np.full_like(factors, np.nan, dtype=float)
        for t in range(factors.shape[0]):
            valid = valid_mask[t] & np.isfinite(factors[t]).all(axis=1)
            min_required = factors.shape[2] + 2 if method == "sequential" else 2
            if valid.sum() < min_required:
                continue
            centered = factors[t, valid] - factors[t, valid].mean(axis=0, keepdims=True)
            out[t, valid] = self._orthogonal_cross_section(centered, method=method, ridge=ridge)
        return np.stack([cross_sectional_zscore(out[:, :, k], mask=valid_mask) for k in range(out.shape[2])], axis=2)

    def _ordered_residualize(self, factors: np.ndarray, *, mask: np.ndarray, order: np.ndarray) -> np.ndarray:
        if factors.ndim != 3:
            raise ValueError("factors must have shape T x N x K")
        out = np.full_like(factors, np.nan, dtype=float)
        accepted: list[np.ndarray] = []
        for position in np.asarray(order, dtype=int):
            target = factors[:, :, position]
            if accepted:
                residual = neutralize_by_exposures(target, np.stack(accepted, axis=2), mask=mask)
                residual = cross_sectional_zscore(residual, mask=mask)
            else:
                residual = cross_sectional_zscore(target, mask=mask)
            out[:, :, position] = residual
            accepted.append(residual)
        return out

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
            where=singular > EPS,
        )
        return (u * scale) @ vt


class ProjectionTransform:
    """Shared PCA/PLS static and walk-forward projection logic."""

    def run(
        self,
        factors: np.ndarray,
        labels: np.ndarray,
        *,
        method: str,
        fit: str,
        n_components: int,
        lookback: int,
        min_periods: int,
        mask: np.ndarray,
    ) -> FamilyTransformResult:
        if method not in {"pca", "pls"}:
            raise ValueError("projection method must be pca or pls")
        if fit == "static":
            return self.static(factors, labels, method=method, n_components=n_components, mask=mask)
        if fit == "walk_forward":
            return self.walk_forward(
                factors,
                labels,
                method=method,
                n_components=n_components,
                lookback=lookback,
                min_periods=min_periods,
                mask=mask,
            )
        raise ValueError("transform_fit must be static or walk_forward")

    def static(
        self,
        factors: np.ndarray,
        labels: np.ndarray,
        *,
        method: str,
        n_components: int,
        mask: np.ndarray,
    ) -> FamilyTransformResult:
        self._validate_inputs(factors, labels if method == "pls" else None)
        n_components = min(n_components, factors.shape[2])
        x_all = factors.reshape(-1, factors.shape[2])
        y_all = labels.reshape(-1) if method == "pls" else None
        train_valid = self._valid(x_all, y_all, mask.reshape(-1))
        if train_valid.sum() <= n_components:
            raise ValueError("insufficient observations for projection")
        fit = self._fit(x_all[train_valid], None if y_all is None else y_all[train_valid], method, n_components)
        flat_values = np.full((x_all.shape[0], n_components), np.nan, dtype=float)
        flat_values[train_valid] = self._project(x_all[train_valid], fit)
        values = flat_values.reshape(factors.shape[0], factors.shape[1], n_components)
        diagnostics = dict(fit.diagnostics)
        diagnostics["fit"] = "static"
        return FamilyTransformResult(values, diagnostics)

    def walk_forward(
        self,
        factors: np.ndarray,
        labels: np.ndarray,
        *,
        method: str,
        n_components: int,
        lookback: int,
        min_periods: int,
        mask: np.ndarray,
    ) -> FamilyTransformResult:
        self._validate_inputs(factors, labels if method == "pls" else None)
        n_components = min(n_components, factors.shape[2])
        out = np.full((factors.shape[0], factors.shape[1], n_components), np.nan, dtype=float)
        explained: dict[int, list[float]] = {}
        valid_mask = np.asarray(mask, dtype=bool)

        for t in range(factors.shape[0]):
            start = max(0, t - lookback)
            if t - start < min_periods:
                continue
            x_hist = factors[start:t].reshape(-1, factors.shape[2])
            y_hist = labels[start:t].reshape(-1) if method == "pls" else None
            train_valid = self._valid(x_hist, y_hist, valid_mask[start:t].reshape(-1))
            if train_valid.sum() <= n_components:
                continue
            fit = self._fit(x_hist[train_valid], None if y_hist is None else y_hist[train_valid], method, n_components)
            if method == "pca":
                explained[t] = fit.diagnostics["explained_variance"]
            now_valid = self._valid(factors[t], None, valid_mask[t])
            if now_valid.any():
                out[t, now_valid] = self._project(factors[t, now_valid], fit)

        diagnostics = {"method": method, "fit": "walk_forward", "lookback": lookback, "min_periods": min_periods}
        if method == "pca":
            diagnostics["explained_variance"] = explained
        return FamilyTransformResult(out, diagnostics)

    def _fit(self, x: np.ndarray, y: np.ndarray | None, method: str, n_components: int) -> ProjectionFit:
        scaled, mean, std = self._standardize(x)
        if method == "pca":
            rotation, diagnostics = self._fit_pca(scaled, n_components)
        else:
            rotation, diagnostics = self._fit_pls(scaled, y, n_components)
        return ProjectionFit(rotation=rotation, mean=mean, std=std, diagnostics=diagnostics)

    @staticmethod
    def _project(x: np.ndarray, fit: ProjectionFit) -> np.ndarray:
        centered = np.asarray(x, dtype=float) - fit.mean
        scaled = np.divide(centered, fit.std, out=np.zeros_like(centered, dtype=float), where=fit.std > EPS)
        return scaled @ fit.rotation
    
    @staticmethod
    def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = np.nanmean(x, axis=0)
        centered = x - mean
        std = np.nanstd(centered, axis=0)
        std = np.where(std > EPS, std, 1.0)
        scaled = np.divide(centered, std, out=np.zeros_like(centered, dtype=float), where=std > EPS)
        return scaled, mean, std

    @staticmethod
    def _valid(x: np.ndarray, y: np.ndarray | None, mask: np.ndarray | None) -> np.ndarray:
        valid = np.isfinite(x).all(axis=1)
        if mask is not None:
            valid &= np.asarray(mask, dtype=bool)
        if y is not None:
            valid &= np.isfinite(y)
        return valid

    @staticmethod
    def _validate_inputs(factors: np.ndarray, labels: np.ndarray | None) -> None:
        if factors.ndim != 3:
            raise ValueError("factors must have shape T x N x K")
        if labels is not None and labels.shape != factors.shape[:2]:
            raise ValueError("labels must have shape T x N and align with factors")

    @staticmethod
    def _fit_pca(x_scaled: np.ndarray, n_components: int) -> tuple[np.ndarray, dict[str, object]]:
        _, singular, vt = np.linalg.svd(x_scaled, full_matrices=False)
        rotation = vt[:n_components].T
        explained = (singular[:n_components] ** 2 / np.maximum(np.sum(singular**2), EPS)).tolist()
        return rotation, {"method": "pca", "explained_variance": explained}

    @staticmethod
    def _fit_pls(x_scaled: np.ndarray, y: np.ndarray | None, n_components: int) -> tuple[np.ndarray, dict[str, object]]:
        if y is None:
            raise ValueError("PLS requires labels")
        if PLSRegression is None:
            raise ImportError("PLS transform requires scikit-learn")
        model = PLSRegression(n_components=n_components, scale=False)
        model.fit(x_scaled, y)
        return model.x_rotations_, {"method": "pls", "n_components": n_components}


class FamilyTransform:
    """Light dispatcher used by FactorFamilyBuilder."""

    def __init__(self, config: FamilyConfig | None = None) -> None:
        self.config = config or FamilyConfig()
        self.orthogonal = OrthogonalTransform()
        self.projection = ProjectionTransform()

    def run(
        self,
        factors: np.ndarray,
        labels: np.ndarray,
        *,
        mask: np.ndarray,
        quality: np.ndarray,
    ) -> FamilyTransformResult:
        method = self.config.transform_method
        if method == "raw" or factors.shape[2] == 1:
            return FamilyTransformResult(factors, {"method": "raw"})
        if method == "orthogonal":
            return self.orthogonal.run(
                factors,
                mask=mask,
                method=self.config.orthogonalization,
                ridge=self.config.transform_ridge,
                quality=quality,
            )
        if method in {"pca", "pls"}:
            return self.projection.run(
                factors,
                labels,
                method=method,
                fit=self.config.transform_fit,
                n_components=min(self.config.n_components, factors.shape[2]),
                lookback=self.config.lookback,
                min_periods=self.config.min_ic_obs,
                mask=mask,
            )
        raise ValueError(f"unsupported family transform_method: {method}")

