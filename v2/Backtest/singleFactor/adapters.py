"""Adapters from axis-backed numpy/memmap storage to the public data contract."""

from __future__ import annotations
import numpy as np
import pandas as pd
from .data import FactorData


def _frame(values, dates, ticks, name, dtype=None):
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional date x stock matrix")
    if array.shape != (len(dates), len(ticks)):
        raise ValueError(
            f"{name} shape {array.shape} does not match axes "
            f"({len(dates)}, {len(ticks)})")
    return pd.DataFrame(
        array, index=pd.DatetimeIndex(dates, name="date"),
        columns=pd.Index([str(x) for x in ticks], name="tick"), copy=False)


def factor_data_from_arrays(signal, returns, dates, ticks, *, tradable=None,
                            industry=None, benchmark_weight=None,
                            market_weight=None):
    """Wrap aligned numpy or memmap matrices without copying when possible."""
    signal_df = _frame(signal, dates, ticks, "signal", float)
    return_df = _frame(returns, dates, ticks, "returns", float)
    optional = {
        "tradable": None if tradable is None else
            _frame(tradable, dates, ticks, "tradable", bool),
        "industry": None if industry is None else
            _frame(industry, dates, ticks, "industry"),
        "benchmark_weight": None if benchmark_weight is None else
            _frame(benchmark_weight, dates, ticks, "benchmark_weight", float),
        "market_weight": None if market_weight is None else
            _frame(market_weight, dates, ticks, "market_weight", float),
    }
    return FactorData(signal_df, return_df, **optional)
