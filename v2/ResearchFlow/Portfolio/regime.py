"""Observable regime probabilities, mixture of experts, and bounded weight tilts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-12


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
    """Classify observable market regimes from market returns and optional indicators."""

    def __init__(self, *, lookback: int = 60, threshold_quantile: float = 0.60, smoothing: float = 0.80) -> None:
        self.lookback = int(lookback)
        self.threshold_quantile = float(threshold_quantile)
        self.smoothing = float(smoothing)
        self.state_names = ("risk_on", "risk_off", "high_vol_trend", "low_vol_range")

    def fit_predict(self, market_returns: np.ndarray, indicators: np.ndarray | None = None) -> RegimeProbabilityResult:
        returns = np.asarray(market_returns, dtype=float).reshape(-1)
        vol_pct = _expanding_percentile(_rolling_stat(returns, self.lookback, kind="std"))
        trend = _rolling_stat(returns, self.lookback, kind="mean")
        trend_pct = _expanding_percentile(trend)

        high_vol = _smooth_probability(vol_pct, self.threshold_quantile)
        positive_trend = _trend_probability(trend, trend_pct)
        scores = np.column_stack(
            [
                (1.0 - high_vol) * positive_trend,
                high_vol * (1.0 - positive_trend),
                high_vol * positive_trend,
                (1.0 - high_vol) * (1.0 - positive_trend),
            ]
        )
        names = self.state_names

        if indicators is not None:
            extra = np.asarray(indicators, dtype=float)
            extra = extra[:, None] if extra.ndim == 1 else extra
            indicator_scores = np.maximum(2.0 * np.column_stack([_expanding_percentile(extra[:, j]) for j in range(extra.shape[1])]) - 1.0, 0.0)
            scores = np.column_stack([scores, indicator_scores])
            names += tuple(f"indicator_{i}_high" for i in range(extra.shape[1]))

        probabilities = _normalize_rows(_ewm_nan_to_zero(scores, 1.0 - self.smoothing))
        has_state = probabilities.sum(axis=1) > EPS
        states = np.full(probabilities.shape[0], "unknown", dtype=object)
        states[has_state] = np.asarray(names, dtype=object)[np.argmax(probabilities[has_state], axis=1)]
        return RegimeProbabilityResult(probabilities=probabilities, states=states, state_names=names)


class RegimeWeightController:
    """Apply bounded regime tilts to already valid weight rows."""

    def __init__(self, *, max_tilt: float = 0.20, smoothing: float = 0.80) -> None:
        self.max_tilt = float(max_tilt)
        self.smoothing = float(smoothing)

    def tilt(self, base_weights: np.ndarray, probabilities: np.ndarray, tilt_matrix: np.ndarray) -> RegimeResult:
        base = np.asarray(base_weights, dtype=float)
        probs = np.asarray(probabilities, dtype=float)
        tilt = np.asarray(tilt_matrix, dtype=float)
        if probs.shape[0] != base.shape[0]:
            raise ValueError("probabilities must align with weights by date")
        if tilt.shape != (probs.shape[1], base.shape[1]):
            raise ValueError(f"tilt_matrix must be {probs.shape[1]} x {base.shape[1]}, got {tilt.shape}")

        multiplier = 1.0 + np.clip(probs @ tilt, -self.max_tilt, self.max_tilt)
        targets = _normalize_rows(base * multiplier)
        out = np.empty_like(targets)
        out[0] = targets[0]
        for t in range(1, targets.shape[0]):
            out[t] = self.smoothing * out[t - 1] + (1.0 - self.smoothing) * targets[t]
        return RegimeResult(probabilities=probs, tilted_weights=_normalize_rows(out))


class MixtureOfExperts:
    """Combine state-specific alpha forecasts by observable regime probability."""

    @staticmethod
    def combine(expert_forecasts: dict[str, np.ndarray], probabilities: np.ndarray, state_names: tuple[str, ...]) -> np.ndarray:
        if not expert_forecasts:
            raise ValueError("at least one expert forecast is required")
        probs = np.asarray(probabilities, dtype=float)
        template = np.asarray(next(iter(expert_forecasts.values())), dtype=float)
        weighted = np.zeros_like(template, dtype=float)
        coverage = np.zeros_like(template, dtype=float)

        for j, name in enumerate(state_names[: probs.shape[1]]):
            if name not in expert_forecasts:
                continue
            forecast = np.asarray(expert_forecasts[name], dtype=float)
            if forecast.shape != template.shape:
                raise ValueError(f"forecast shape mismatch for {name}: {forecast.shape} vs {template.shape}")
            prob = probs[:, j][:, None]
            valid = np.isfinite(forecast)
            weighted += np.where(valid, forecast, 0.0) * prob
            coverage += valid * prob
        return np.divide(weighted, coverage, out=np.full_like(weighted, np.nan), where=coverage > EPS)


def _rolling_stat(x: np.ndarray, window: int, *, kind: str) -> np.ndarray:
    values = np.asarray(x, dtype=float).reshape(-1)
    valid = np.isfinite(values)
    clean = np.where(valid, values, 0.0)
    count = _window_sum(valid.astype(float), window)
    total = _window_sum(clean, window)
    mean = np.divide(total, count, out=np.full_like(values, np.nan), where=count > 0)
    min_obs = max(5, window // 3)
    if kind == "mean":
        return np.where(count >= min_obs, mean, np.nan)
    sq_total = _window_sum(clean * clean, window)
    var = np.divide(sq_total, count, out=np.full_like(values, np.nan), where=count > 0) - mean * mean
    return np.where(count >= min_obs, np.sqrt(np.clip(var, 0.0, None)), np.nan)


def _window_sum(x: np.ndarray, window: int) -> np.ndarray:
    padded = np.r_[0.0, np.asarray(x, dtype=float)]
    cumsum = np.cumsum(padded)
    start = np.maximum(np.arange(x.size) - int(window), 0)
    return cumsum[1:] - cumsum[start]


def _expanding_percentile(x: np.ndarray, min_obs: int = 20) -> np.ndarray:
    values = np.asarray(x, dtype=float).reshape(-1)
    out = np.full(values.size, np.nan, dtype=float)
    for i, value in enumerate(values):
        hist = values[: i + 1]
        valid = np.isfinite(hist)
        if valid.sum() >= min_obs and np.isfinite(value):
            out[i] = np.mean(hist[valid] <= value)
    return out


def _smooth_probability(rank: np.ndarray, threshold: float) -> np.ndarray:
    r = np.clip(np.nan_to_num(rank, nan=0.5), 0.0, 1.0)
    below = 0.5 * r / max(threshold, EPS)
    above = 0.5 + 0.5 * (r - threshold) / max(1.0 - threshold, EPS)
    return np.where(r <= threshold, below, above)


def _trend_probability(trend: np.ndarray, rank: np.ndarray) -> np.ndarray:
    r = np.clip(np.nan_to_num(rank, nan=0.5), 0.0, 1.0)
    return np.where(trend < 0, 0.5 * r, np.where(trend > 0, 0.5 + 0.5 * r, 0.5))


def _ewm_nan_to_zero(x: np.ndarray, alpha: float) -> np.ndarray:
    values = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0)
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, values.shape[0]):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    values = np.maximum(np.nan_to_num(np.asarray(x, dtype=float), nan=0.0), 0.0)
    row_sum = values.sum(axis=1, keepdims=True)
    return np.divide(values, row_sum, out=np.zeros_like(values), where=row_sum > EPS)
