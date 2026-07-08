"""Branch B: combine family composites into a unified stock alpha."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import AlphaConfig
from ..matrix_math import cap_and_renormalize, cross_sectional_zscore, factor_ic_history
from .allocation import AllocationParams, CapitalAllocator
from .alpha_models import DynamicLinearAlpha, WalkForwardSklearnAlpha


@dataclass(frozen=True)
class UnifiedAlphaResult:
    alpha: np.ndarray
    family_weights: np.ndarray
    family_ic: np.ndarray
    diagnostics: dict[str, float | str]


class UnifiedAlphaPath:
    """Class-level alpha combiner with configurable model choices.

    Supported methods: equal, icir, correlation_adjusted, ridge,
    fama_macbeth, score_slope, dynamic_linear, elastic_net, lasso,
    bayesian_ridge, pls, random_forest, gbdt, hist_gbdt, rank_gbdt, mlp.
    """

    def __init__(
        self,
        config: AlphaConfig | None = None,
        *,
        method: str | None = None,
        lookback: int | None = None,
        min_periods: int | None = None,
        max_family_weight: float | None = None,
        ridge_lambda: float | None = None,
    ) -> None:
        base = config or AlphaConfig()
        self.config = AlphaConfig(
            method=method or base.method,
            lookback=lookback or base.lookback,
            min_periods=min_periods or base.min_periods,
            max_family_weight=max_family_weight if max_family_weight is not None else base.max_family_weight,
            ridge_lambda=ridge_lambda if ridge_lambda is not None else base.ridge_lambda,
            l1_ratio=base.l1_ratio,
            n_components=base.n_components,
            random_state=base.random_state,
            max_iter=base.max_iter,
            hidden_layer_sizes=base.hidden_layer_sizes,
            clip_sigma=base.clip_sigma,
        )
        self.method = self.config.method
        self.lookback = self.config.lookback
        self.min_periods = self.config.min_periods
        self.max_family_weight = self.config.max_family_weight
        self.ridge_lambda = self.config.ridge_lambda
        self.allocator = CapitalAllocator(
            AllocationParams(
                method=self.method,
                lookback=self.lookback,
                min_periods=self.min_periods,
                max_weight=self.max_family_weight,
            )
        )

    def run(self, family_scores: np.ndarray, labels: np.ndarray, *, tradable: np.ndarray | None = None) -> UnifiedAlphaResult:
        mask = np.ones(family_scores.shape[:2], dtype=bool) if tradable is None else tradable.astype(bool)
        family_ic = factor_ic_history(family_scores, labels, mask=mask)
        if self.method in {"equal", "icir", "correlation_adjusted"}:
            weights = self._weights(family_ic)
            alpha = self._combine(family_scores, weights, mask)
        elif self.method == "ridge":
            alpha, weights = self._rolling_ridge_alpha(family_scores, labels, mask)
        elif self.method == "fama_macbeth":
            alpha, weights = self._fama_macbeth_alpha(family_scores, labels, mask)
        elif self.method == "score_slope":
            weights = self._weights(family_ic)
            score = self._combine(family_scores, weights, mask)
            alpha = self._score_slope_alpha(score, labels, mask)
        elif self.method == "dynamic_linear":
            model_result = DynamicLinearAlpha(self.config).fit_predict(family_scores, labels, mask=mask)
            alpha = model_result.alpha
            weights = self._weights_from_coefficients(model_result.coefficients, family_scores.shape[2])
        elif self.method in WalkForwardSklearnAlpha.SUPPORTED:
            model_result = WalkForwardSklearnAlpha(self.config).fit_predict(family_scores, labels, mask=mask)
            alpha = model_result.alpha
            weights = self._weights_from_coefficients(model_result.coefficients, family_scores.shape[2])
        else:
            raise ValueError(f"unsupported unified alpha method: {self.method}")
        return UnifiedAlphaResult(
            alpha=alpha,
            family_weights=weights,
            family_ic=family_ic,
            diagnostics={"method": self.method, "lookback": float(self.lookback)},
        )

    def _weights(self, family_ic: np.ndarray) -> np.ndarray:
        return self.allocator.allocate(family_ic, family_ic.shape[0], family_ic.shape[1])

    @staticmethod
    def _combine(family_scores: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> np.ndarray:
        valid = np.isfinite(family_scores)
        numerator = np.where(valid, family_scores * weights[:, None, :], 0.0).sum(axis=2)
        denom = np.where(valid, weights[:, None, :], 0.0).sum(axis=2)
        alpha = np.divide(numerator, denom, out=np.full(family_scores.shape[:2], np.nan), where=denom > 0)
        return cross_sectional_zscore(alpha, mask=mask)

    def _rolling_ridge_alpha(self, family_scores: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        '''
        多变量回归 + L2 正则
        适合 family 较多、相关性较强
        '''
        t_count, _, n_family = family_scores.shape
        alpha = np.full(family_scores.shape[:2], np.nan, dtype=float)
        weights = np.full((t_count, n_family), 1.0 / n_family, dtype=float)
        for t in range(t_count):
            start = max(0, t - self.lookback)
            x_hist = family_scores[start:t].reshape(-1, n_family)
            y_hist = labels[start:t].reshape(-1)
            m_hist = mask[start:t].reshape(-1)
            valid = m_hist & np.isfinite(y_hist) & np.isfinite(x_hist).all(axis=1)
            if valid.sum() < max(self.min_periods, n_family + 5):
                continue
            x = x_hist[valid]
            y = y_hist[valid]
            x_design = np.column_stack([np.ones(len(x)), x])
            penalty = self.ridge_lambda * np.eye(x_design.shape[1])
            penalty[0, 0] = 0.0
            beta = np.linalg.solve(x_design.T @ x_design + penalty, x_design.T @ y)
            weights[t] = self._normalize_beta(beta[1:])
            valid_now = mask[t] & np.isfinite(family_scores[t]).all(axis=1)
            alpha[t, valid_now] = np.column_stack([np.ones(valid_now.sum()), family_scores[t, valid_now]]) @ beta
        return cross_sectional_zscore(alpha, mask=mask), weights

    def _fama_macbeth_alpha(self, family_scores: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        '''
        每日横截面回归 beta，再滚动平均 beta
        适合需要解释每类因子收益贡献
        '''
        t_count, _, n_family = family_scores.shape
        daily_beta = np.full((t_count, n_family), np.nan, dtype=float)
        for t in range(t_count):
            valid = mask[t] & np.isfinite(labels[t]) & np.isfinite(family_scores[t]).all(axis=1)
            if valid.sum() <= n_family + 2:
                continue
            x = np.column_stack([np.ones(valid.sum()), family_scores[t, valid]])
            beta = np.linalg.lstsq(x, labels[t, valid], rcond=None)[0]
            daily_beta[t] = beta[1:]
        weights = np.full((t_count, n_family), 1.0 / n_family, dtype=float)
        alpha = np.full(family_scores.shape[:2], np.nan, dtype=float)
        for t in range(t_count):
            hist = daily_beta[max(0, t - self.lookback):t]
            if len(hist) < self.min_periods:
                continue
            beta = np.nanmean(hist, axis=0)
            weights[t] = self._normalize_beta(beta)
            valid_now = mask[t] & np.isfinite(family_scores[t]).all(axis=1)
            alpha[t, valid_now] = family_scores[t, valid_now] @ beta
        return cross_sectional_zscore(alpha, mask=mask), weights

    def _score_slope_alpha(self, score: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
        '''
        先合成 score，再用单变量斜率校准
        适合保留已有组合逻辑，只调整信号强度
        '''
        out = np.full_like(score, np.nan, dtype=float)
        for t in range(score.shape[0]):
            hist_score = score[max(0, t - self.lookback):t].reshape(-1)
            hist_label = labels[max(0, t - self.lookback):t].reshape(-1)
            valid = np.isfinite(hist_score) & np.isfinite(hist_label)
            if valid.sum() < self.min_periods:
                out[t] = score[t]
                continue
            denom = float(hist_score[valid] @ hist_score[valid])
            slope = float(hist_score[valid] @ hist_label[valid] / denom) if denom > 1e-12 else 1.0
            out[t] = score[t] * slope
        return cross_sectional_zscore(out, mask=mask)

    def _weights_from_coefficients(self, coefficients: np.ndarray, n_family: int) -> np.ndarray:
        out = np.full((coefficients.shape[0], n_family), 1.0 / n_family, dtype=float)
        for t in range(coefficients.shape[0]):
            coef = coefficients[t]
            if np.isfinite(coef).any():
                out[t] = self._normalize_beta(coef)
        return out

    def _normalize_beta(self, beta: np.ndarray) -> np.ndarray:
        raw = np.maximum(np.nan_to_num(beta, nan=0.0), 0.0)
        if raw.sum() <= 1e-12:
            return np.full(len(beta), 1.0 / len(beta), dtype=float)
        return cap_and_renormalize(raw / raw.sum(), max_weight=self.max_family_weight)
