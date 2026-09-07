"""Event-time paths and inference, separate from portfolio backtests."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import norm
from .builders import event_directions, event_triggers
from .config import EventConfig
from .data import FactorData


@dataclass(frozen=True)
class EventStudyResult:
    observations: pd.DataFrame
    statistics: pd.DataFrame
    average_path: pd.Series


def _hac_mean_test(values, max_lag):
    """Newey-West t test for a mean with Bartlett kernel."""
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return np.nan, np.nan
    demeaned = x - x.mean()
    lag = min(max(int(max_lag), 0), n - 1)
    long_run_variance = np.dot(demeaned, demeaned) / n
    for k in range(1, lag + 1):
        covariance = np.dot(demeaned[k:], demeaned[:-k]) / n
        long_run_variance += 2 * (1 - k / (lag + 1)) * covariance
    standard_error = np.sqrt(max(long_run_variance, 0) / n)
    t_value = x.mean() / standard_error if standard_error > 0 else np.nan
    p_value = 2 * norm.sf(abs(t_value)) if np.isfinite(t_value) else np.nan
    return t_value, p_value


def _forward_return_matrix(returns, horizon):
    """Vectorized forward compounded returns for every date and asset."""
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizons must contain positive integers")
    out = np.full(returns.shape, np.nan, dtype=float)
    if len(returns) <= horizon:
        return out

    windows = np.lib.stride_tricks.sliding_window_view(
        returns[2:], window_shape=horizon, axis=0
    )
    valid = np.isfinite(windows).all(axis=-1)
    compounded = np.prod(
        1.0 + np.where(np.isfinite(windows), windows, 0.0),
        axis=-1,
    ) - 1.0
    out[:len(compounded)] = np.where(valid, compounded, np.nan)
    return out


def _row_peer_mean(forward, eligible):
    """Nan-aware equal-weight peer return for every date."""
    valid = eligible & np.isfinite(forward)
    count = valid.sum(axis=1)
    total = np.where(valid, forward, 0.0).sum(axis=1)
    return np.divide(
        total, count,
        out=np.full(forward.shape[0], np.nan, dtype=float),
        where=count > 0,
    )


def _industry_peer_return(forward, tradable, industry, event_t, event_j):
    """Industry peer means for event observations without per-event loops."""
    peer_return = np.full(len(event_t), np.nan, dtype=float)
    if not len(event_t):
        return peer_return   # event_t是有重复日期的事件序列，非时间序列

    # 找出每个事件日期的起始位置
    dates, starts = np.unique(event_t, return_index=True)
    stops = np.r_[starts[1:], len(event_t)]
    for t, start, stop in zip(dates, starts, stops):
        positions = np.arange(start, stop)
        event_inds = industry[t, event_j[positions]]
        for ind in np.unique(event_inds[np.isfinite(event_inds)]):
            same_code = event_inds == ind  # 某个行业的filter
            peers = (
                tradable[t]
                & np.isfinite(forward[t])
                & (industry[t] == ind)
            )
            if peers.any():
                peer_return[positions[same_code]] = forward[t, peers].mean()
    return peer_return


def run_event_study(raw: FactorData, event=EventConfig(),
                    horizons=(1, 3, 5, 10, 20), adjustment="none"):
    """Measure event returns after t, optionally market/industry adjusted."""
    horizons = tuple(horizons)
    data = raw.aligned()
    if adjustment not in {"none", "market", "industry"}:
        raise ValueError("adjustment must be none, market or industry")
    if adjustment == "industry" and data.industry is None:
        raise ValueError("industry adjustment requires industry")
    
    signal, returns = data.signal.to_numpy(), data.returns.to_numpy()
    tradable = data.tradable.to_numpy()
    if data.benchmark_weight is not None:
        benchmark = data.benchmark_weight.to_numpy(float)
        tradable = (
            tradable
            & np.isfinite(benchmark)
            & (benchmark > 0)
        )
    trigger = event_triggers(signal, tradable, event)
    direction = event_directions(signal, event)
    trigger &= direction != 0

    event_t, event_j = np.where(trigger)
    event_direction = direction[event_t, event_j]
    event_dates = data.signal.index.to_numpy()[event_t]
    event_assets = data.signal.columns.to_numpy()[event_j]
    industry = (
        None if data.industry is None else data.industry.to_numpy()
    )

    adjusted_returns = []
    for horizon in horizons:
        forward = _forward_return_matrix(returns, horizon)
        event_return = forward[event_t, event_j]
        if adjustment == "market":
            peer_return = _row_peer_mean(forward, tradable)[event_t]
            event_return = event_return - peer_return
        elif adjustment == "industry":
            peer_return = _industry_peer_return(
                forward, tradable, industry, event_t, event_j
            )
            event_return = event_return - peer_return

        adjusted_returns.append(event_return)

    if len(event_t) and horizons:
        return_matrix = (
            event_direction[:, None] * np.column_stack(adjusted_returns)
        )
        observations = pd.DataFrame({
            "event_date": np.repeat(event_dates, len(horizons)),
            "asset": np.repeat(event_assets, len(horizons)),
            "horizon": np.tile(horizons, len(event_t)), # eg:(1,3,5)*2
            "direction": np.repeat(event_direction, len(horizons)),
            "return": return_matrix.ravel(),
        })
    else:
        observations = pd.DataFrame(columns=[
            "event_date", "asset", "horizon", "direction", "return"
        ])
    
    stats = []
    for horizon in horizons:
        values = observations.loc[
            observations.horizon == horizon, "return"].dropna().to_numpy()
        # Multiple stocks can fire on one day. Aggregate by event date first,
        # then HAC-correct the overlapping horizon-return time series.
        dated = observations.loc[
            observations.horizon == horizon, ["event_date", "return"]
        ].dropna().groupby("event_date").mean()["return"]
        hac_lag = max(horizon - 1, 0)
        t_value, p_value = _hac_mean_test(dated.to_numpy(), hac_lag)
        stats.append((horizon, len(values), len(dated),
                      np.mean(values) if len(values) else np.nan,
                      np.mean(values > 0) if len(values) else np.nan,
                      t_value, p_value, hac_lag))
    statistics = pd.DataFrame(stats, columns=[
        "horizon", "sample_count", "event_date_count", "mean_return",
        "win_rate", "hac_t_stat", "hac_p_value", "hac_lags"
    ]).set_index("horizon")
    average_path = statistics.mean_return.rename("average_event_path")
    return EventStudyResult(observations, statistics, average_path)
