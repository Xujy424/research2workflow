"""Event-time paths and inference, separate from portfolio backtests."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import norm
from .builders import event_triggers
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


def _forward_path(returns, start, horizon):
    block = returns[start + 1:start + horizon + 1]
    if len(block) < horizon or not np.isfinite(block).all():
        return np.nan
    return np.prod(1 + block) - 1


def run_event_study(raw: FactorData, event=EventConfig(),
                    horizons=(1, 3, 5, 10, 20), adjustment="none"):
    """Measure event returns after t, optionally market/industry adjusted."""
    data = raw.aligned()
    if adjustment not in {"none", "market", "industry"}:
        raise ValueError("adjustment must be none, market or industry")
    if adjustment == "industry" and data.industry is None:
        raise ValueError("industry adjustment requires industry")
    signal, returns = data.signal.to_numpy(), data.returns.to_numpy()
    tradable = data.tradable.to_numpy()
    trigger = event_triggers(signal, tradable, event)
    rows = []
    for t, j in zip(*np.where(trigger)):
        direction = np.sign(signal[t, j]) or 1.0
        peers = tradable[t].copy()
        if adjustment == "industry":
            peers &= data.industry.to_numpy()[t] == data.industry.to_numpy()[t, j]
        for horizon in horizons:
            value = _forward_path(returns[:, j], t, horizon)
            if adjustment != "none":
                peer_values = [_forward_path(returns[:, k], t, horizon)
                               for k in np.flatnonzero(peers)]
                value -= np.nanmean(peer_values)
            rows.append((data.signal.index[t], data.signal.columns[j], horizon,
                         direction, direction * value))
    observations = pd.DataFrame(rows, columns=[
        "event_date", "asset", "horizon", "direction", "return"])
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
