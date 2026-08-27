"""Stable performance and cross-sectional diagnostics."""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def performance(returns: pd.Series, annual_days=242) -> pd.Series:
    x = returns.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if not len(x):
        return pd.Series(dtype=float)
    nav = np.cumprod(1 + x)
    ann = nav[-1] ** (annual_days / len(x)) - 1
    vol = np.std(x, ddof=1) * np.sqrt(annual_days) if len(x) > 1 else np.nan
    dd = nav / np.maximum.accumulate(nav) - 1
    return pd.Series({"total_return": nav[-1] - 1, "annual_return": ann,
                      "annual_vol": vol, "sharpe": ann / vol if vol > 0 else np.nan,
                      "max_drawdown": dd.min(), "win_rate": np.mean(x > 0)})


def cross_sectional_ic(signal: pd.DataFrame, future_return: pd.DataFrame,
                       rank=True, min_obs=20) -> pd.Series:
    out = np.full(len(signal), np.nan)
    for t, (x, y) in enumerate(zip(signal.to_numpy(), future_return.to_numpy())):
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < min_obs:
            continue
        xx, yy = x[ok], y[ok]
        if rank:
            xx, yy = rankdata(xx), rankdata(yy)
        out[t] = np.corrcoef(xx, yy)[0, 1]
    return pd.Series(out, index=signal.index, name="rank_ic" if rank else "ic")
