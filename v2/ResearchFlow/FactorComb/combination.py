"""Common weighting methods used by family, sleeve, and alpha modules."""

from __future__ import annotations

import numpy as np

from ..matrix_math import calc_icir, cap_and_renormalize


def rolling_icir_weights(
    ic: np.ndarray,
    *,
    lookback: int = 252,
    min_periods: int = 60,
    max_weight: float = 0.50,
    allow_negative: bool = False,
) -> np.ndarray:
    n_dates, n_items = ic.shape
    out = np.full((n_dates, n_items), 1.0 / n_items, dtype=float)
    for t in range(n_dates):
        hist = ic[max(0, t - lookback):t]
        valid_count = np.isfinite(hist).sum(axis=0)
        if len(hist) < min_periods or valid_count.max(initial=0) < min_periods:
            continue
        score = calc_icir(hist)
        if not allow_negative:
            score = np.maximum(score, 0.0)
        scale = np.abs(score).sum() if allow_negative else score.sum()
        if scale > 1e-12:
            out[t] = cap_and_renormalize(score / scale, max_weight=max_weight)
    return out


def equal_weights(n_dates: int, n_items: int) -> np.ndarray:
    return np.full((n_dates, n_items), 1.0 / n_items, dtype=float)


