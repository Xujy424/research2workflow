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
    scalar = x.ndim == 1 and y.ndim == 1
    if scalar:
        x, y = x[:, None], y[:, None]
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
    result = np.divide(
        np.sum(xc * yc, axis=0), denominator,
        out=np.full(x.shape[1], np.nan),
        where=(count >= min_observations) & (denominator > 0),
    )
    return float(result[0]) if scalar else result


def least_trimmed_squares_slope(x, y, trim, *, rtol=1e-13, max_intervals=8192):
    """Globally minimize the sum of the N-trim smallest (y-beta*x)**2.

    This is a scalar regression without an intercept. Bounds come from the
    quadratic curvature of *every* possible retained subset, so active-set
    refinements supply upper bounds only, never an unverified local answer.
    The returned solution is certified to ``rtol`` relative objective error
    (with a floating-point floor). A work limit raises rather than silently
    returning an uncertified approximation.
    """
    from heapq import heappop, heappush

    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 1 or y.shape != x.shape or not len(x):
        raise ValueError("x and y must be matching nonempty vectors")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("least-trimmed-squares inputs must be finite")
    if not isinstance(trim, (int, np.integer)) or not 0 <= trim < len(x):
        raise ValueError("trim must be an integer in [0, N)")
    if not 0 < rtol < 1 or max_intervals < 1:
        raise ValueError("invalid solver accuracy or work limit")
    x_scale, y_scale = np.max(np.abs(x)), np.max(np.abs(y))
    if x_scale == 0 or y_scale == 0:
        return 0.0
    a, b = x / x_scale, y / y_scale
    if trim == 0:
        return float(np.dot(a, b) / np.dot(a, a) * y_scale / x_scale)
    keep = len(a) - trim
    ratios = b[a != 0] / a[a != 0]
    if not np.isfinite(ratios).all():
        raise ValueError("predictor scale produces an unbounded floating-point search interval")
    lo, hi = float(ratios.min()), float(ratios.max())
    if lo == hi:
        return lo * y_scale / x_scale
    squares = a * a
    curvature = np.partition(squares, trim)[trim:].sum()
    best, slope = np.inf, lo

    def loss(beta):
        residual = b - beta * a
        return np.partition(residual * residual, keep - 1)[:keep].sum()

    def improve(beta):
        nonlocal best, slope
        for _ in range(16):
            residual = b - beta * a
            selected = np.argpartition(residual * residual, keep - 1)[:keep]
            value = np.dot(residual[selected], residual[selected])
            if value < best:
                best, slope = value, beta
            denominator = squares[selected].sum()
            if denominator == 0:
                break
            fitted = np.dot(a[selected], b[selected]) / denominator
            if fitted == beta:
                break
            beta = fitted

    def lower_bound(left, right, f_left, f_right):
        # For every retained subset S, its quadratic equals the chord of
        # its endpoint losses minus A_S*(beta-left)*(right-beta).
        # Endpoint losses >= F(endpoint), and A_S <= curvature. Therefore
        # this parabola bounds the lower envelope F throughout the interval.
        left_residual, right_residual = b - left * a, b - right * a
        left_square, right_square = left_residual ** 2, right_residual ** 2
        row_lower = np.where(left_residual * right_residual <= 0, 0,
                             np.minimum(left_square, right_square))
        row_upper = np.maximum(left_square, right_square)
        threshold = np.partition(row_upper, keep - 1)[keep - 1]
        possible = squares[row_lower <= threshold]
        # Rows exceeding the k-th smallest uniform upper bound can never
        # enter a retained subset anywhere in this interval. Dropping them
        # tightens curvature, including exactly flat optima with zero x.
        local_curvature = np.sort(possible)[-keep:].sum()
        bend = local_curvature * (right - left) ** 2
        weight = np.clip(0.5 - (f_right - f_left) / (2 * bend), 0, 1) if bend > 0 else 0
        quadratic_bound = (1 - weight) * f_left + weight * f_right - bend * weight * (1 - weight)
        independent_bound = np.partition(row_lower, keep - 1)[:keep].sum()
        return max(quadratic_bound, independent_bound)

    for initial in (np.dot(a, b) / squares.sum(), *np.quantile(ratios, (0.1, 0.5, 0.9))):
        improve(initial)
    f_lo, f_hi = loss(lo), loss(hi)
    queue = [(lower_bound(lo, hi, f_lo, f_hi), lo, hi, f_lo, f_hi)]
    intervals = 0
    floor = np.finfo(float).eps * len(a)
    while queue:
        bound, left, right, f_left, f_right = heappop(queue)
        tolerance = rtol * max(best, floor)
        if bound >= best - tolerance:
            continue
        midpoint = left + (right - left) / 2
        if midpoint == left or midpoint == right:
            improve(left)
            improve(right)
            continue
        intervals += 1
        if intervals > max_intervals:
            raise RuntimeError(
                f"least-trimmed-squares exceeded {max_intervals} intervals; "
                "no uncertified solution was returned"
            )
        f_mid = loss(midpoint)
        improve(midpoint)
        for low, high, f_low, f_high in (
            (left, midpoint, f_left, f_mid), (midpoint, right, f_mid, f_right),
        ):
            bound = lower_bound(low, high, f_low, f_high)
            if bound < best - tolerance:
                heappush(queue, (bound, low, high, f_low, f_high))
    return float(slope * y_scale / x_scale)


def linear_quantile_residual(y, design, quantile):
    """Residuals from linear check-loss minimization on finite observations.

    ``design`` contains any desired intercept. No intercept, lag, selection
    rule or domain-specific feature is implicitly added to the regression.
    """
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix, eye, hstack

    y, design = np.asarray(y, float), np.asarray(design, float)
    if y.ndim != 1 or design.ndim != 2 or len(y) != len(design) or not len(y):
        raise ValueError("expected nonempty y[N] and design[N, P]")
    if not 0 < quantile < 1:
        raise ValueError("quantile must be strictly between zero and one")
    if not np.isfinite(y).all() or not np.isfinite(design).all():
        raise ValueError("quantile regression inputs must be finite")
    n, p = design.shape
    scale = np.linalg.norm(design, axis=0)
    scale[scale == 0] = 1
    x = design / scale
    constraints = hstack((csr_matrix(x), eye(n), -eye(n)), format="csr")
    fit = linprog(
        np.r_[np.zeros(p), np.full(n, quantile), np.full(n, 1 - quantile)],
        A_eq=constraints, b_eq=y,
        bounds=[(None, None)] * p + [(0, None)] * (2 * n), method="highs",
        options={"time_limit": 30.0},
    )
    if not fit.success:
        raise RuntimeError(f"quantile regression failed: {fit.message}")
    return y - x @ fit.x[:p]


__all__ = [
    "cross_sectional_rank",
    "cross_sectional_residual",
    "safe_ratio_return",
    "selected_mean_diff",
    "time_series_corr",
    "linear_quantile_residual",
    "least_trimmed_squares_slope",
    "time_series_range",
    "time_series_zscore",
]
