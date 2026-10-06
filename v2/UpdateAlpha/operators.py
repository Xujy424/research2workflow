"""Shared numerical operators for alpha-factor calculations."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def safe_ratio_return(numerator, denominator):
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=(
            np.isfinite(numerator)
            & np.isfinite(denominator)
            & (numerator > 0)
            & (denominator > 0)
        ),
    ) - 1.0


def cross_sectional_residual(y, regressors, min_observations, *, mask=None):
    """OLS residuals of one cross-section on one or more regressors."""
    target = np.asarray(y, dtype=np.float64)
    features = np.asarray(regressors, dtype=np.float64)
    if target.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if features.ndim == 1:
        features = features[:, None]
    if features.ndim != 2 or features.shape[0] != target.shape[0]:
        raise ValueError("regressors must be shaped observations x features")

    valid = np.isfinite(target) & np.all(np.isfinite(features), axis=1)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != target.shape:
            raise ValueError("mask must match the target cross-section")
        valid &= mask
    result = np.full_like(target, np.nan)
    if valid.sum() < min_observations:
        return result
    design = np.column_stack((np.ones(valid.sum()), features[valid]))
    coefficient, *_ = np.linalg.lstsq(design, target[valid], rcond=None)
    result[valid] = target[valid] - design @ coefficient
    return result


def cross_sectional_rank(values, *, mask=None):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    valid = np.isfinite(values)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("mask must match the ranked cross-section")
        valid &= mask
    result = np.full_like(values, np.nan)
    if valid.any():
        result[valid] = rankdata(values[valid], method="average") / valid.sum()
    return result


def time_series_zscore(values, min_observations):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("values must be shaped time x observations")
    valid = np.isfinite(values)
    count = valid.sum(axis=0)
    mean = np.divide(
        np.where(valid, values, 0.0).sum(axis=0),
        count,
        out=np.full(values.shape[1], np.nan),
        where=count > 0,
    )
    centered = np.where(valid, values - mean, 0.0)
    std = np.sqrt(np.divide(
        np.sum(centered * centered, axis=0),
        count,
        out=np.full(values.shape[1], np.nan),
        where=count >= min_observations,
    ))
    return np.divide(
        values - mean,
        std,
        out=np.full_like(values, np.nan),
        where=valid & (std > 0),
    )


def selected_mean_diff(values, selector, selected_count, min_observations):
    """Mean(values) on top selector rows minus its bottom-row mean."""
    values = np.asarray(values, dtype=np.float64)
    selector = np.asarray(selector, dtype=np.float64)
    if values.ndim != 2 or selector.shape != values.shape:
        raise ValueError("values and selector must share a time x observations shape")
    if not 1 <= selected_count * 2 <= values.shape[0]:
        raise ValueError("selected_count must not exceed half the time window")

    valid = np.isfinite(values) & np.isfinite(selector)
    count = valid.sum(axis=0)
    high_rows = np.argpartition(
        np.where(valid, selector, -np.inf), -selected_count, axis=0
    )[-selected_count:]
    low_rows = np.argpartition(
        np.where(valid, selector, np.inf), selected_count - 1, axis=0
    )[:selected_count]
    high = np.take_along_axis(values, high_rows, axis=0)
    low = np.take_along_axis(values, low_rows, axis=0)
    high_valid = np.isfinite(high)
    low_valid = np.isfinite(low)
    high_mean = np.divide(
        np.where(high_valid, high, 0.0).sum(axis=0),
        high_valid.sum(axis=0),
        out=np.full(values.shape[1], np.nan),
        where=high_valid.sum(axis=0) > 0,
    )
    low_mean = np.divide(
        np.where(low_valid, low, 0.0).sum(axis=0),
        low_valid.sum(axis=0),
        out=np.full(values.shape[1], np.nan),
        where=low_valid.sum(axis=0) > 0,
    )
    required = max(min_observations, 2 * selected_count)
    return np.where(count >= required, high_mean - low_mean, np.nan)


def time_series_range(values, window, min_observations):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not 1 <= window <= values.shape[0]:
        raise ValueError("invalid time-series range window")
    sample = values[-window:]
    valid = np.isfinite(sample)
    maximum = np.max(np.where(valid, sample, -np.inf), axis=0)
    minimum = np.min(np.where(valid, sample, np.inf), axis=0)
    return np.where(
        valid.sum(axis=0) >= min_observations,
        maximum - minimum,
        np.nan,
    )


def time_series_corr(x, y, min_observations):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.shape != x.shape:
        raise ValueError("x and y must share a time x observations shape")
    valid = np.isfinite(x) & np.isfinite(y)
    count = valid.sum(axis=0)
    x_mean = np.divide(
        np.where(valid, x, 0.0).sum(axis=0), count,
        out=np.full(x.shape[1], np.nan), where=count > 0,
    )
    y_mean = np.divide(
        np.where(valid, y, 0.0).sum(axis=0), count,
        out=np.full(y.shape[1], np.nan), where=count > 0,
    )
    xc = np.where(valid, x - x_mean, 0.0)
    yc = np.where(valid, y - y_mean, 0.0)
    denominator = np.sqrt(np.sum(xc * xc, axis=0) * np.sum(yc * yc, axis=0))
    return np.divide(
        np.sum(xc * yc, axis=0), denominator,
        out=np.full(x.shape[1], np.nan),
        where=(count >= min_observations) & (denominator > 0),
    )


__all__ = [
    "cross_sectional_rank",
    "cross_sectional_residual",
    "safe_ratio_return",
    "selected_mean_diff",
    "time_series_corr",
    "time_series_range",
    "time_series_zscore",
]
