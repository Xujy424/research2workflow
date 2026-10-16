"""Shared monitor calculations; data access is delegated to context."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .context import row

INDEX_NAMES = {'沪深300': 'hs300', '中证500': 'zz500', '中证1000': 'zz1000', '中证2000': 'zz2000', '000300': 'hs300', '000905': 'zz500', '000852': 'zz1000', '932000': 'zz2000', '000985': 'zzfull'}


def position(ctx, asof):
    i = ctx['dates'].searchsorted(pd.Timestamp(asof), side='right') - 1
    if i < 0:
        raise ValueError('asof precedes available trading dates')
    if pd.Timestamp(asof) > ctx['dates'][-1]:
        raise ValueError('asof exceeds data calendar; refresh axes or use a remote-only context')
    return int(i)


def weighted_average(values, weights):
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return np.average(values[valid], weights=weights[valid]) if valid.any() else np.nan


def weights(ctx, i, universe=None, industry=None):
    if universe in (None, 'self', 'all'):
        w = row(ctx, 'd_essentials/circ_mv', i).copy()
    else:
        name = INDEX_NAMES.get(universe, universe)
        w = row(ctx, f'index/weight/{name}_weight', i).copy()
    if industry is not None:
        if isinstance(industry, str):
            from ..UpdateData.stock.update_industry import industry_to_id
            industry = industry_to_id[industry]
        w[row(ctx, 'industry/industry', i) != industry] = np.nan
    return np.where(w > 0, w, np.nan)


def returns(ctx, i, period):
    if not isinstance(period, int) or period < 1 or i < period:
        raise ValueError('period must be a positive integer with sufficient history')
    r = np.array([row(ctx, 'd_essentials/pct', j) / 100 for j in range(i - period + 1, i + 1)])
    return np.prod(1 + r, axis=0) - 1


def portfolio_return(ctx, i, period=1, universe=None, industry=None):
    return weighted_average(returns(ctx, i, period), weights(ctx, i - period, universe, industry))


def smooth(ctx, asof, period, fn):
    i = position(ctx, asof)
    if not isinstance(period, int) or period < 1 or i < period - 1:
        raise ValueError('invalid period or insufficient history')
    values = np.array([fn(j) for j in range(i - period + 1, i + 1)])
    return float(values.mean())


def percentile(ctx, asof, period, history, fn):
    i = position(ctx, asof)
    if not isinstance(history, int) or history < 2 or i < history + period - 2:
        raise ValueError('insufficient percentile history')
    values = np.array([smooth(ctx, ctx['dates'][j], period, fn) for j in range(i - history + 1, i + 1)])
    return historical_percentile(values)


def portfolio_nav(ctx, i, universe, industry):
    key = (universe, industry)
    nav = ctx.setdefault('_nav_cache', {}).setdefault(key, [1.0])
    while len(nav) <= i:
        nav.append(nav[-1] * (1 + portfolio_return(ctx, len(nav), 1, universe, industry)))
    return nav[i]


def beta(ctx, i):
    if i < 252:
        raise ValueError('beta requires 252 trading days')
    r = np.array([row(ctx, 'd_essentials/pct', j) / 100 for j in range(i - 251, i + 1)])
    market = np.array([weighted_average(r[k], weights(ctx, i - 252 + k)) for k in range(252)])
    valid = np.isfinite(r) & np.isfinite(market[:, None])
    x = np.where(valid, market[:, None], np.nan)
    y = np.where(valid, r, np.nan)
    xc, yc = (x - np.nanmean(x, axis=0), y - np.nanmean(y, axis=0))
    den = np.nansum(xc ** 2, axis=0)
    return np.divide(np.nansum(xc * yc, axis=0), den, out=np.full(len(ctx['ticks']), np.nan), where=(den > 0) & (valid.sum(0) >= 200))


def historical_percentile(values):
    """Rank the latest value using midpoint ranks for ties; missing values propagate."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        return np.nan
    return float(((values < values[-1]).sum() + 0.5 * (values == values[-1]).sum()) / len(values))
