"""Vectorized numerical primitives for factor matrices.

This module is the shared numerical layer for FactorTest, FactorRegistry,
FactorComb, and Portfolio.  Functions here operate on numpy arrays and avoid
application-specific state, file paths, or registry concepts.
"""

from __future__ import annotations

import numpy as np
import bottleneck as bn
import pandas as pd


EPS = 1e-12


# ---------------------------------------------------------------------------
# Cross-sectional preprocessing
# ---------------------------------------------------------------------------
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


def robust_extreme_ratio_by_row(x: np.ndarray, *, mad_multiple: float = 5.0, mask: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    valid = np.isfinite(values)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    work = np.where(valid, values, np.nan)
    median = np.nanmedian(work, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(work - median), axis=1, keepdims=True)
    fallback = np.abs(work - median) > EPS
    normal = np.abs(work - median) > mad_multiple * mad
    extreme = np.where((np.isfinite(mad) & (mad > EPS)), normal, fallback)
    counts = np.sum(valid, axis=1)
    hits = np.sum(extreme & valid, axis=1)
    return np.divide(hits, counts, out=np.full(values.shape[0], np.nan, dtype=float), where=counts > 0)


# ---------------------------------------------------------------------------
# Ranking, IC, and correlation
# ---------------------------------------------------------------------------
def corr(a, b, axis):
    b[np.isnan(a)] = np.nan
    a[np.isnan(b)] = np.nan
    arr = (
            (bn.nanmean(a * b, axis=axis) - bn.nanmean(a, axis=axis) * bn.nanmean(b, axis=axis))
            / (bn.nanstd(a, axis=axis) + 1e-6)
            / (bn.nanstd(b, axis=axis) + 1e-6)
    )
    bn.replace(arr, np.nan, 0)
    arr[np.isinf(arr)] = 0
    return arr

def IC(y_, y):
    ics = corr(y_.copy(), y.copy(), axis=-1)
    return ics

def rankIC(y_, y):
    rank_ics = corr(bn.nanrankdata(y_.copy(), axis=-1), bn.nanrankdata(y.copy(), axis=-1), axis=-1)
    return rank_ics


def nan_corr_by_row(x: np.ndarray, y: np.ndarray, *, rank: bool = False, min_obs: int = 20) -> np.ndarray:
    left = np.asarray(x, dtype=float)
    right = np.asarray(y, dtype=float)
    out = rankIC(left, right) if rank else IC(left, right)
    valid_count = np.sum(np.isfinite(left) & np.isfinite(right), axis=-1)
    return np.where(valid_count >= min_obs, out, np.nan)


def rank_1d(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if valid.sum() == 0:
        return out
    out[valid] = bn.nanrankdata(values[valid], axis=-1)
    return out


def corr_1d(x: np.ndarray, y: np.ndarray, *, min_obs: int = 2) -> float:
    left = np.asarray(x, dtype=float)
    right = np.asarray(y, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < min_obs:
        return np.nan
    value = corr(left[valid], right[valid], axis=0)
    return float(value) if np.isfinite(value) else np.nan


# ---------------------------------------------------------------------------
# Neutralization
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Group, return, and drawdown diagnostics
# ---------------------------------------------------------------------------
# 中文说明：`calc_group_ret`：计算研究或生产指标。
def calc_group_ret(alpha, label, num_group=10):
    rank = bn.nanrankdata(alpha, axis=-1)
    num_signal = np.nanmax(rank, axis=-1)
    stock_each_group = num_signal // num_group
    group_ret = np.full((num_group, num_signal.shape[0]), np.nan)
    for i in range(num_group):
        if i==num_group-1:
            group_ix = (rank.T > stock_each_group * i) & (rank.T <= num_signal)
        else:
            group_ix = (rank.T > stock_each_group * i) & (rank.T <= stock_each_group * (i + 1)) # n_stock, n_date
        temp_ret = label.copy()
        temp_ret[~group_ix.T] = np.nan
        group_ret[i] = np.nanmean(temp_ret, axis=-1)
    group_ret = group_ret - np.nanmean(group_ret, axis=0)
    col_list = list(range(1, num_group + 1))[::-1]
    group_ret = pd.DataFrame(
        group_ret.T,
        columns=col_list,
        index=alpha.index,
    )
    return group_ret

# 中文说明：`calc_annret`：计算研究或生产指标。
def calc_annret(ret_df):
    nav = np.nancumprod(1+ret_df.values)
    years = (ret_df.index[-1] - ret_df.index[0]).days / 242
    total_ret = nav[-1]/nav[0]-1
    annret = (1+total_ret)**(1/years) - 1
    return annret

# 中文说明：`calc_annvol`：计算研究或生产指标。
def calc_annvol(ret_df):
    annvol = np.nanstd(ret_df.values) * np.sqrt(242)
    return annvol

# 中文说明：`calc_sharpe`：计算研究或生产指标。
def calc_sharpe(ret_df):
    annret = calc_annret(ret_df)
    annvol = calc_annvol(ret_df)
    sharpe = annret / annvol if annvol>0 else 0
    return sharpe

# 中文说明：`calc_maxdrawdown`：计算研究或生产指标。
def calc_maxdrawdown(ret_df):
    nav = np.nancumprod(1+ret_df.values)
    return ((nav - np.maximum.accumulate(nav)) / np.maximum.accumulate(nav)).min()

# 中文说明：`calc_calmar`：计算研究或生产指标。
def calc_calmar(ret_df):
    annret = calc_annret(ret_df)
    max_dd = calc_maxdrawdown(ret_df)
    calmar = annret / abs(max_dd) if max_dd<0 else np.nan
    return calmar

# 中文说明：`calc_weekly_bps`：计算研究或生产指标。
def calc_weekly_bps(ret_df):
    weekly_rets = ret_df.resample('W').apply(lambda x: (1+x).prod()-1).dropna()
    weekly_avg_bps = weekly_rets.mean() * 10000
    return weekly_avg_bps

# 中文说明：`calc_holdings`：计算研究或生产指标。
def calc_holdings(alpha: pd.DataFrame, num_group: int = 10) -> pd.DataFrame:
    rank = bn.nanrankdata(alpha.values, axis=-1)   # (n_dates, n_stocks)
    n_valid = np.nansum(~np.isnan(alpha.values), axis=-1)
    stock_each_group = n_valid // num_group
    topgroup_ix = (rank.T > stock_each_group * (num_group-1) ) & (rank.T <= n_valid) # (n_stocks, n_dates)
    bottomgroup_ix = (rank.T > 0) & (rank.T <= stock_each_group)
    long_holds = pd.DataFrame(topgroup_ix.T, index=alpha.index, columns=alpha.columns)
    short_holds= pd.DataFrame(bottomgroup_ix.T, index=alpha.index, columns=alpha.columns)
    holds = pd.DataFrame(0, index=alpha.index, columns=alpha.columns, dtype=int)
    holds[long_holds.values] = 1
    holds[short_holds.values] = -1
    return holds

# 中文说明：`calc_turnover`：计算研究或生产指标。
def calc_turnover(holds,  freq: str = 'D') -> pd.Series:
    # 1. 生成调仓日期列表
    if freq == 'D':
        rebalance_dates = holds.index.sort_values()  # 所有交易日
    elif freq == 'W':
        rebalance_dates = pd.Series(holds.index).groupby(holds.index.to_period('W-MON')).first().values
    elif freq == 'M':
        rebalance_dates = pd.Series(holds.index).groupby(holds.index.to_period('M')).first().values
    else:
        raise ValueError("freq 必须是 'D', 'W', 'M'")
    
    #holdings = calc_top_holdings(alpha)
    curr = holds.loc[rebalance_dates].astype(int)
    prev = curr.shift(1).fillna(0).astype(int)
    
    # 双边平均换手率 = (买入+卖出) / (前总持仓+现总持仓)
    change = (curr - prev).abs().sum(axis=1)          # 操作次数加权和
    total = prev.abs().sum(axis=1) + curr.abs().sum(axis=1)
    turnover = change / total
    turnover = turnover.fillna(0.0)
    turnover.iloc[0] = np.nan
    return turnover.rename('turnover').to_frame(name='turnover')

# ---------------------------------------------------------------------------
# Portfolio weight helpers
# ---------------------------------------------------------------------------
def cap_and_renormalize(weights: np.ndarray, *, max_weight: float, iterations: int = 50) -> np.ndarray:
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
