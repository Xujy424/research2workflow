"""Observable regime probabilities, mixture of experts, and weight tilts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegimeProbabilityResult:
    probabilities: np.ndarray
    states: np.ndarray
    state_names: tuple[str, ...]


@dataclass(frozen=True)
class RegimeResult:
    probabilities: np.ndarray
    tilted_weights: np.ndarray


class ObservableRegimeModel:
    """Classify observable market regimes from market returns and indicators."""

    def __init__(self, *, lookback: int = 60, threshold_quantile: float = 0.60, smoothing: float = 0.80) -> None:
        self.lookback = lookback
        self.threshold_quantile = threshold_quantile
        self.smoothing = smoothing
        self.state_names = ("risk_on", "risk_off", "high_vol_trend", "low_vol_range")

    def fit_predict(self, market_returns: np.ndarray, indicators: np.ndarray | None = None) -> RegimeProbabilityResult:
        returns = np.asarray(market_returns, dtype=float).reshape(-1)
        vol = rolling_nanstd(returns, self.lookback)
        trend = rolling_nanmean(returns, self.lookback)
        vol_pct = expanding_percentile(vol)
        trend_pct = expanding_percentile(trend)
        high_vol = smooth_probability(vol_pct, self.threshold_quantile)
        positive_trend = trend_probability(trend, trend_pct)
        scores = np.column_stack([
            (1.0 - high_vol) * positive_trend,
            high_vol * (1.0 - positive_trend),
            high_vol * positive_trend,
            (1.0 - high_vol) * (1.0 - positive_trend),
        ])
        if indicators is not None:
            extra = np.asarray(indicators, dtype=float)
            if extra.ndim == 1:
                extra = extra[:, None]
            extra_scores = []
            for j in range(extra.shape[1]):
                extra_scores.append(np.maximum(2.0 * expanding_percentile(extra[:, j]) - 1.0, 0.0))
            scores = np.column_stack([scores, *extra_scores])
            names = self.state_names + tuple(f"indicator_{i}_high" for i in range(extra.shape[1]))
        else:
            names = self.state_names
        alpha = 1.0 - self.smoothing
        smoothed = ewm_nan_to_zero(scores, alpha)
        probabilities = normalize_rows(smoothed)
        states = np.asarray([names[i] if row.sum() > 0 else "unknown" for i, row in zip(np.nanargmax(probabilities, axis=1), probabilities)])
        return RegimeProbabilityResult(probabilities=probabilities, states=states, state_names=names)


class RegimeWeightController:
    """Apply bounded regime tilts to already valid weight rows."""

    def __init__(self, *, max_tilt: float = 0.20, smoothing: float = 0.80) -> None:
        self.max_tilt = max_tilt
        self.smoothing = smoothing

    def tilt(self, base_weights: np.ndarray, probabilities: np.ndarray, tilt_matrix: np.ndarray) -> RegimeResult:
        if probabilities.shape[0] != base_weights.shape[0]:
            raise ValueError("probabilities must align with weights by date")
        raw_multiplier = 1.0 + np.clip(probabilities @ tilt_matrix, -self.max_tilt, self.max_tilt)
        out = np.zeros_like(base_weights)
        prev = base_weights[0]
        for t in range(base_weights.shape[0]):
            target = np.maximum(base_weights[t] * raw_multiplier[t], 0.0)
            target = target / target.sum() if target.sum() > 1e-12 else base_weights[t]
            out[t] = self.smoothing * prev + (1.0 - self.smoothing) * target
            out[t] = out[t] / out[t].sum() if out[t].sum() > 1e-12 else out[t]
            prev = out[t]
        return RegimeResult(probabilities=probabilities, tilted_weights=out)


class MixtureOfExperts:
    """Combine state-specific alpha forecasts by observable regime probability."""

    @staticmethod
    def combine(expert_forecasts: dict[str, np.ndarray], probabilities: np.ndarray, state_names: tuple[str, ...]) -> np.ndarray:
        if not expert_forecasts:
            raise ValueError("at least one expert forecast is required")
        template = next(iter(expert_forecasts.values()))
        out = np.zeros_like(template, dtype=float)
        available = np.zeros_like(template, dtype=float)
        for j, name in enumerate(state_names):
            if name not in expert_forecasts or j >= probabilities.shape[1]:
                continue
            forecast = np.asarray(expert_forecasts[name], dtype=float)
            prob = probabilities[:, j][:, None]
            valid = np.isfinite(forecast)
            out += np.where(valid, forecast, 0.0) * prob
            available += valid.astype(float) * prob
        return np.divide(out, available, out=np.full_like(out, np.nan), where=available > 1e-12)


def rolling_nanmean(x: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(x), np.nan, dtype=float)
    for i in range(len(x)):
        start = max(0, i - window)
        values = x[start:i]
        if np.isfinite(values).sum() >= max(5, window // 3):
            out[i] = np.nanmean(values)
    return out


def rolling_nanstd(x: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(x), np.nan, dtype=float)
    for i in range(len(x)):
        start = max(0, i - window)
        values = x[start:i]
        if np.isfinite(values).sum() >= max(5, window // 3):
            out[i] = np.nanstd(values)
    return out


def expanding_percentile(x: np.ndarray) -> np.ndarray:
    out = np.full(len(x), np.nan, dtype=float)
    for i in range(len(x)):
        hist = x[: i + 1]
        valid = np.isfinite(hist)
        if valid.sum() >= 20 and np.isfinite(x[i]):
            out[i] = (hist[valid] <= x[i]).mean()
    return out


def smooth_probability(rank: np.ndarray, threshold: float) -> np.ndarray:
    rank = np.clip(rank, 0.0, 1.0)
    below = 0.5 * rank / threshold
    above = 0.5 + 0.5 * (rank - threshold) / max(1.0 - threshold, 1e-12)
    return np.where(rank <= threshold, below, above)


def trend_probability(trend: np.ndarray, rank: np.ndarray) -> np.ndarray:
    prob = np.full(len(trend), 0.5, dtype=float)
    neg = trend < 0
    pos = trend > 0
    prob[neg] = 0.5 * rank[neg]
    prob[pos] = 0.5 + 0.5 * rank[pos]
    return np.clip(prob, 0.0, 1.0)


def ewm_nan_to_zero(x: np.ndarray, alpha: float) -> np.ndarray:
    values = np.nan_to_num(x, nan=0.0)
    out = np.zeros_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def normalize_rows(x: np.ndarray) -> np.ndarray:
    row_sum = np.nansum(np.maximum(x, 0.0), axis=1, keepdims=True)
    return np.divide(np.maximum(x, 0.0), row_sum, out=np.zeros_like(x), where=row_sum > 1e-12)
