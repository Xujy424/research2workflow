"""Growth minus value return-spread percentiles extracted from style.py.

Only local pct, circ_mv and growth1_mask are required; no Wind calls.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from v2.GetData import DataPool
from v2.UpdateData.config import ROOT
from v2.Monitor.utils import compound_return, weighted_mean, rolling_percentile


def growth_value_spreads(mask, returns, weights, periods=(1, 5, 10, 20)):
    """Start-of-window groups/weights and buy-and-hold compounded stock returns.

    Inputs share [date, stock] shape; returns are decimals. An incomplete
    stock-return window excludes that stock, matching style.py exactly.
    """
    if mask.shape != returns.shape or weights.shape != returns.shape:
        raise ValueError("mask, returns and weights must have the same shape")
    if any(not isinstance(p, (int, np.integer)) or p < 1 for p in periods):
        raise ValueError("periods must be positive integers")
    result = np.full((len(returns), len(periods)), np.nan)
    for column, period in enumerate(periods):
        for end in range(period, len(returns)):
            start = end - period
            r = compound_return(returns[start + 1:end + 1])
            m, w = mask[start], weights[start]
            high, low = m == 1, m == -1
            result[end, column] = (weighted_mean(r[high], w[high])
                                   - weighted_mean(r[low], w[low]))
    return result


def growth_value_history(data, start="2018-01-01", end=None, *,
                         periods=(1, 5, 10, 20), history=252, chunk_size=256):
    """Compute once per endpoint in bounded chunks, then rank the spread series."""
    periods = tuple(periods)
    if not periods or any(not isinstance(p, (int, np.integer)) or p < 1 for p in periods):
        raise ValueError("periods must contain positive integers")
    if history < 1 or chunk_size < 1:
        raise ValueError("history and chunk_size must be positive")
    dates = pd.DatetimeIndex(data.axis.trade_dates)
    end = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None).normalize() if end is None else pd.Timestamp(end)
    selected = np.flatnonzero((dates >= pd.Timestamp(start)) & (dates <= end))
    if not len(selected):
        raise ValueError("no local trading dates in requested range")
    first, last = int(selected[0]), int(selected[-1])
    warm_start = max(0, first - (history - 1))
    spreads = np.full((last - warm_start + 1, len(periods)), np.nan)
    for left in range(warm_start, last + 1, chunk_size):
        right = min(last, left + chunk_size - 1)
        read_start = max(0, left - max(periods))
        returns = np.asarray(data.read("d_essentials/pct", right, read_start), dtype=float) / 100
        mask = data.read("barra/growth1_mask", right, read_start)
        weights = data.read("d_essentials/circ_mv", right, read_start)
        values = growth_value_spreads(mask, returns, weights, periods)
        spreads[left - warm_start:right - warm_start + 1] = values[left - read_start:]
    result = pd.DataFrame(index=dates[warm_start:last + 1])
    result.index.name = "date"
    for column, period in enumerate(periods):
        result[f"spread_{period}d"] = spreads[:, column]
        result[f"percentile_{period}d"] = rolling_percentile(spreads[:, column], history)
    return result.loc[dates[first]:dates[last]]


def plot_growth_value(result, periods=(1, 5, 10, 20), history=252):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, axes = plt.subplots(len(periods), 1, figsize=(14, 10), sharex=True, squeeze=False)
    for ax, period in zip(axes[:, 0], periods):
        ax.plot(result.index, result[f"percentile_{period}d"], linewidth=0.8, color="#246a9c")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7)
        ax.axhline(0.2, color="#be6a4b", linestyle=":", linewidth=0.6)
        ax.axhline(0.8, color="#be6a4b", linestyle=":", linewidth=0.6)
        ax.set(title=f"{period}-day growth minus value return: {history}-day historical percentile", ylim=(0, 1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=0.15)
    fig.suptitle(f"Growth / value style | {result.index[0]:%Y-%m-%d} to {result.index[-1]:%Y-%m-%d}")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    started = perf_counter()
    periods = (1, 5, 10, 20)
    history = 252
    output = Path(__file__).resolve().parent / "output" / "growth_value"
    output.mkdir(parents=True, exist_ok=True)
    with DataPool(ROOT, asset="stock") as data:
        result = growth_value_history(data, start="2018-01-01", periods=periods, history=history)
    if not result.filter(like="percentile_").notna().any().any():
        raise ValueError("all percentiles are missing; check growth1_mask, pct and circ_mv")
    suffix = f"{result.index[0]:%Y%m%d}_{result.index[-1]:%Y%m%d}"
    result.to_csv(output / f"growth_value_{suffix}.csv")
    figure = plot_growth_value(result, periods, history)
    figure.savefig(output / f"growth_value_{suffix}.png", dpi=160)
    print(result.tail().to_string())
    print(f"Saved to {output}; elapsed={perf_counter() - started:.2f}s", flush=True)
    plt.show()
