"""Vectorized numerical primitives for factor matrices."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
import bottleneck as bn


EPS = 1e-12


def winsorize(
    x: np.ndarray,
    *,
    method: str = "mad",
    p: float = 0.01,
    n_sigma: float = 3.0,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    valid = np.isfinite(values)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    work = np.where(valid, values, np.nan)
    if method == "quantile":
        lower = np.nanquantile(work, p, axis=1, keepdims=True)
        upper = np.nanquantile(work, 1.0 - p, axis=1, keepdims=True)
    elif method == "sigma":
        mean = np.nanmean(work, axis=1, keepdims=True)
        std = np.nanstd(work, axis=1, keepdims=True)
        lower = mean - n_sigma * std
        upper = mean + n_sigma * std
    elif method == "mad":
        median = np.nanmedian(work, axis=1, keepdims=True)
        mad = np.nanmedian(np.abs(work - median), axis=1, keepdims=True)
        radius = n_sigma * 1.4826 * mad
        lower = median - radius
        upper = median + radius
    else:
        raise ValueError(f"unsupported winsorization method: {method}")
    return np.where(valid, np.clip(work, lower, upper), np.nan)


def cross_sectional_zscore(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    valid = np.isfinite(values)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    work = np.where(valid, values, np.nan)
    mean = np.nanmean(work, axis=1, keepdims=True)
    std = np.nanstd(work, axis=1, keepdims=True)
    out = np.divide(work - mean, std, out=np.full_like(work, np.nan), where=std > EPS)
    return np.where(valid, out, np.nan)


def cross_sectional_rank(x: np.ndarray, *, pct: bool = True) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    out = np.full_like(values, np.nan, dtype=float)
    for t in range(values.shape[0]):
        valid = np.isfinite(values[t])
        if valid.sum() == 0:
            continue
        ranks = rankdata(values[t, valid], method="average")
        out[t, valid] = ranks / valid.sum() if pct else ranks
    return out


def neutralize_by_exposures(
    y: np.ndarray,
    exposures: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    add_intercept: bool = True,
    min_obs: int | None = None,
) -> np.ndarray:
    target = np.asarray(y, dtype=float)
    x = np.asarray(exposures, dtype=float)
    if x.ndim != 3 or x.shape[:2] != target.shape:
        raise ValueError("exposures must be shaped T x N x K and aligned with y")
    
    valid = np.isfinite(target) & np.isfinite(x).all(axis=2)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if weights is not None:
        w_all = np.asarray(weights, dtype=float)
        valid &= np.isfinite(w_all) & (w_all > 0)
    threshold = min_obs or (x.shape[2] + int(add_intercept) + 5)
    out = np.full_like(target, np.nan, dtype=float)

    for t in range(target.shape[0]):
        ok = valid[t]
        if ok.sum() < threshold:
            continue
        design = x[t, ok]
        if add_intercept:
            design = np.column_stack([np.ones(ok.sum()), design])
        yy = target[t, ok]
        if weights is None:
            beta = np.linalg.lstsq(design, yy, rcond=None)[0]
        else:
            sqrt_w = np.sqrt(w_all[t, ok])
            beta = np.linalg.lstsq(design * sqrt_w[:, None], yy * sqrt_w, rcond=None)[0]
        out[t, ok] = yy - design @ beta
    return out


def industry_size_neutralize(
    factor: np.ndarray,
    industry: np.ndarray,
    market_cap: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    standardize: bool = True,
) -> np.ndarray:
    industry_arr = np.asarray(industry, dtype=float)
    finite_codes = np.unique(industry_arr[np.isfinite(industry_arr)]).astype(int)
    dummies = [(industry_arr == code).astype(float) for code in finite_codes]
    log_size = np.log(np.clip(np.asarray(market_cap, dtype=float), EPS, None))
    exposures = np.stack([log_size, *dummies], axis=2)
    residual = neutralize_by_exposures(factor, exposures, mask=mask)
    return cross_sectional_zscore(residual, mask=mask) if standardize else residual


def nan_corr_by_row(x: np.ndarray, y: np.ndarray, *, rank: bool = False, min_obs: int = 20) -> np.ndarray:
    '''按日期逐行计算相关系数'''
    xx = cross_sectional_rank(x, pct=False) if rank else np.asarray(x, dtype=float)
    yy = cross_sectional_rank(y, pct=False) if rank else np.asarray(y, dtype=float)
    out = np.full(xx.shape[0], np.nan, dtype=float)
    for t in range(xx.shape[0]):
        valid = np.isfinite(xx[t]) & np.isfinite(yy[t])
        if valid.sum() >= min_obs:
            out[t] = np.corrcoef(xx[t, valid], yy[t, valid])[0, 1]
    return out


def cap_and_renormalize(weights: np.ndarray, *, max_weight: float, iterations: int = 50) -> np.ndarray:
    '''单票权重上限约束后的重新归一化'''
    w = np.nan_to_num(np.asarray(weights, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    active = w > EPS
    if not active.any():
        return w
    
    cap = max(max_weight, 1.0 / active.sum())
    w = w / w.sum()

    for _ in range(iterations):
        over = active & (w > cap)
        if not over.any():
            break

        excess = float((w[over] - cap).sum())
        w[over] = cap

        free = active & (w < cap - EPS)
        free_sum = float(w[free].sum())
        if free_sum <= EPS:
            break

        w[free] *= (free_sum + excess) / free_sum

    return w


def normalize_long_only(score: np.ndarray, *, mask: np.ndarray | None = None, max_weight: float = 0.02) -> np.ndarray:
    '''
        把 alpha/score 矩阵转成 long-only 权重矩阵
        score = composite_alpha
        weights = normalize_long_only(score, mask=tradable, max_weight=0.02)
    '''
    raw = np.nan_to_num(np.asarray(score, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if mask is not None:
        raw = np.where(mask, raw, 0.0)
    raw = np.maximum(raw, 0.0)
    out = np.zeros_like(raw)
    for t in range(raw.shape[0]):
        row = raw[t]
        if row.sum() <= EPS:
            eligible = np.asarray(mask[t], dtype=bool) if mask is not None else np.ones_like(row, dtype=bool)
            row = eligible.astype(float)
        w = row / row.sum() if row.sum() > EPS else row
        out[t] = cap_and_renormalize(w, max_weight=max_weight) if max_weight > 0 else w
    return out


# def calc_indmv_neutral_longshort(ind_signal, temp_mv):
#     ix = ~(np.isnan(ind_signal) | np.isinf(ind_signal) | np.isnan(temp_mv) | np.isinf(temp_mv))
#     ind_signal[~ix] = np.nan
#     temp_mv[~ix] = np.nan

#     mv_mean = bn.nanmean(temp_mv, axis=1)
#     signal_mean = bn.nanmean(ind_signal, axis=1)
#     m = (mv_mean * signal_mean - bn.nanmean(temp_mv * ind_signal, axis=1)) / (mv_mean**2 - bn.nanmean(temp_mv**2, axis=1) + 1e-6)
#     b = signal_mean - m * mv_mean
#     residual = (ind_signal.T - (temp_mv.T * m) - b).T
#     ind_signal = (residual.T - bn.nanmean(residual, axis=1)) / (bn.nanstd(residual, axis=1) + 1e-6)
#     return ind_signal.T

# def indmv_neutral_longshort(alpha_vec, ind_arr, mv_arr):
#     new_signal = np.full_like(alpha_vec, np.nan)   # [T,N]
#     ln_mv = np.log(mv_arr)
#     for i in range(31):
#         ind_ix = ind_arr == i
#         ind_select = ind_ix.any(axis=0)
#         ind_ix_select = ind_ix[:, ind_select]
#         ind_signal = alpha_vec[:, ind_select].copy()
#         ind_signal[~ind_ix_select] = np.nan
#         temp_mv = ln_mv[:, ind_select].copy()
#         new_signal[ind_ix] = calc_indmv_neutral_longshort(ind_signal, temp_mv)[ind_ix_select]
#     return new_signal

