"""Walk-forward alpha models used by the unified-alpha branch."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ..config import AlphaConfig
from ..matrix_math import cross_sectional_zscore


@dataclass(frozen=True)
class AlphaModelResult:
    alpha: np.ndarray
    coefficients: np.ndarray


class WalkForwardSklearnAlpha:
    """Walk-forward sklearn alpha models for ``T x N x K`` features."""

    SUPPORTED = {
        "elastic_net",
        "lasso",
        "bayesian_ridge",
        "pls",
        "random_forest",
        "gbdt",
        "hist_gbdt",
        "rank_gbdt",
        "mlp",
    }

    def __init__(self, config: AlphaConfig) -> None:
        if config.method not in self.SUPPORTED:
            raise ValueError(f"unsupported sklearn alpha method: {config.method}")
        self.config = config

    def fit_predict(self, features: np.ndarray, labels: np.ndarray, *, mask: np.ndarray) -> AlphaModelResult:
        t_count, _, n_features = features.shape
        alpha = np.full(features.shape[:2], np.nan, dtype=float)
        importance = np.full((t_count, n_features), np.nan, dtype=float)
        for t in range(t_count):
            start = max(0, t - self.config.lookback)
            x_hist = features[start:t].reshape(-1, n_features)
            y_hist = labels[start:t].reshape(-1)
            m_hist = mask[start:t].reshape(-1)
            valid = m_hist & np.isfinite(y_hist) & np.isfinite(x_hist).all(axis=1)
            if valid.sum() <= max(self.config.min_periods, n_features + 5):
                continue
            x = x_hist[valid]
            y = y_hist[valid]
            if self.config.method == "rank_gbdt":
                y = simple_rank_target(y)
            mean = x.mean(axis=0)
            std = x.std(axis=0)
            std[std == 0] = 1.0
            model = self._make_model()
            model.fit((x - mean) / std, y)
            valid_now = mask[t] & np.isfinite(features[t]).all(axis=1)
            if valid_now.any():
                alpha[t, valid_now] = model.predict((features[t, valid_now] - mean) / std).reshape(-1)
            importance[t] = model_importance(model, n_features)
        return AlphaModelResult(cross_sectional_zscore(alpha, mask=mask), importance)

    def _make_model(self) -> object:
        method = self.config.method
        if method in {"elastic_net", "lasso"}:
            from sklearn.linear_model import ElasticNet
            return ElasticNet(
                alpha=self.config.ridge_lambda,
                l1_ratio=1.0 if method == "lasso" else self.config.l1_ratio,
                max_iter=self.config.max_iter,
                random_state=self.config.random_state,
            )
        if method == "bayesian_ridge":
            from sklearn.linear_model import BayesianRidge
            return BayesianRidge(max_iter=self.config.max_iter)
        if method == "pls":
            from sklearn.cross_decomposition import PLSRegression
            return PLSRegression(n_components=self.config.n_components, scale=False, max_iter=self.config.max_iter)
        if method == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                min_samples_leaf=20,
                max_features="sqrt",
                n_jobs=-1,
                random_state=self.config.random_state,
            )
        if method in {"gbdt", "hist_gbdt", "rank_gbdt"}:
            from sklearn.ensemble import HistGradientBoostingRegressor
            return HistGradientBoostingRegressor(
                max_iter=min(self.config.max_iter, 200),
                max_leaf_nodes=15,
                l2_regularization=self.config.ridge_lambda,
                random_state=self.config.random_state,
            )
        if method == "mlp":
            from sklearn.neural_network import MLPRegressor
            return MLPRegressor(
                hidden_layer_sizes=self.config.hidden_layer_sizes,
                activation="relu",
                alpha=self.config.ridge_lambda,
                early_stopping=True,
                max_iter=self.config.max_iter,
                random_state=self.config.random_state,
            )
        raise ValueError(f"unsupported sklearn alpha method: {method}")


class DynamicLinearAlpha:
    """
    递推最小二乘：Recursive least squares alpha with drifting coefficients.
    适合因子收益结构缓慢变化、需要日频自适应的场景
    """

    def __init__(self, config: AlphaConfig, *, forgetting_factor: float = 0.99, initial_uncertainty: float = 100.0) -> None:
        self.config = config
        self.forgetting_factor = forgetting_factor
        self.initial_uncertainty = initial_uncertainty

    def fit_predict(self, features: np.ndarray, labels: np.ndarray, *, mask: np.ndarray) -> AlphaModelResult:
        t_count, _, n_features = features.shape
        beta = np.zeros(n_features, dtype=float)
        covariance = np.eye(n_features) * self.initial_uncertainty
        alpha = np.full(features.shape[:2], np.nan, dtype=float)
        coef = np.full((t_count, n_features), np.nan, dtype=float)
        for t in range(t_count):
            if t >= self.config.min_periods:
                valid_now = mask[t] & np.isfinite(features[t]).all(axis=1)
                alpha[t, valid_now] = features[t, valid_now] @ beta
                coef[t] = beta
            valid = mask[t] & np.isfinite(labels[t]) & np.isfinite(features[t]).all(axis=1)
            for x, y in zip(features[t, valid], labels[t, valid]):
                projected = covariance @ x
                denominator = self.forgetting_factor + x @ projected
                gain = projected / max(float(denominator), 1e-12)
                beta = beta + gain * (y - x @ beta)
                covariance = (covariance - np.outer(gain, x) @ covariance) / self.forgetting_factor
        return AlphaModelResult(cross_sectional_zscore(alpha, mask=mask), coef)


def simple_rank_target(values: np.ndarray) -> np.ndarray:
    target = np.asarray(values, dtype=float).reshape(-1)
    finite = np.isfinite(target)
    out = np.zeros_like(target, dtype=float)
    if finite.sum() <= 1:
        return out
    order = np.argsort(target[finite], kind="mergesort")
    ranks = np.empty(finite.sum(), dtype=float)
    ranks[order] = (np.arange(finite.sum(), dtype=float) + 0.5) / finite.sum() - 0.5
    out[finite] = ranks
    return out


def model_importance(model: object, n_features: int) -> np.ndarray:
    if hasattr(model, "coef_"):
        values = np.asarray(getattr(model, "coef_"), dtype=float).reshape(-1)
        return values[:n_features]
    if hasattr(model, "feature_importances_"):
        return np.asarray(getattr(model, "feature_importances_"), dtype=float)
    return np.full(n_features, np.nan)


def rank_by_blocks(values: np.ndarray, mask_block: np.ndarray) -> np.ndarray:
    lengths = mask_block.sum(axis=1).astype(int)
    out = np.empty_like(values, dtype=float)
    start = 0
    for length in lengths:
        end = start + length
        if length > 0:
            order = np.argsort(values[start:end])
            ranks = np.empty(length, dtype=float)
            ranks[order] = (np.arange(length) + 1) / length - 0.5
            out[start:end] = ranks
        start = end
    if start < len(values):
        out[start:] = values[start:]
    return out


