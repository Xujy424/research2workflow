"""市场与风格监控。收益/分位均用小数表示，详见 style.md。"""
from __future__ import annotations
from functools import partial
import numpy as np
import pandas as pd
from ..UpdateData.config import ROOT
from .context import create_context, close_context, row, bond_yield
from .utils import (
    position, weighted_average, weights, returns, portfolio_return,
    smooth, percentile, portfolio_nav, historical_percentile,
)

def _ctx_level_percentile(ctx, asof, period=1, universe='hs300', industry=None, history=252):
    """Reconstructed constituent-portfolio NAV percentile (not official index close)."""
    return percentile(ctx, asof, period, history, lambda i: portfolio_nav(ctx, i, universe, industry))


def _ctx_relative_return(ctx, asof, period=1, universe='hs300', industry=None, benchmark='zzfull'):
    i = position(ctx, asof)
    return portfolio_return(ctx, i, period, universe, industry) - portfolio_return(ctx, i, period, benchmark)


def _ctx_new_high_low_share(ctx, asof, period=1, universe='hs300', industry=None, lookback=252):

    def daily(i):
        if lookback < 2 or i < lookback:
            raise ValueError('insufficient high/low lookback')
        r = np.array([row(ctx, 'd_essentials/pct', j) / 100 for j in range(i - lookback + 1, i + 1)])
        nav = np.vstack([np.ones(len(ctx['ticks'])), np.cumprod(1 + r, axis=0)])
        valid = np.isfinite(nav).all(0) & np.isfinite(weights(ctx, i, universe, industry))
        high = nav[-1] > np.max(nav[:-1], axis=0)
        low = nav[-1] < np.min(nav[:-1], axis=0)
        return float((high[valid].sum() - low[valid].sum()) / valid.sum()) if valid.any() else np.nan
    return smooth(ctx, asof, period, daily)


def _ctx__valuation(ctx, i, field, universe, industry):
    v = row(ctx, field, i)
    reciprocal = np.divide(1, v, out=np.full_like(v, np.nan), where=v > 0)
    avg = weighted_average(reciprocal, weights(ctx, i, universe, industry))
    return 1 / avg if np.isfinite(avg) and avg > 0 else np.nan


def _ctx_pe_percentile(ctx, asof, period=1, universe='hs300', industry=None, history=252):
    return percentile(ctx, asof, period, history, lambda i: _ctx__valuation(ctx, i, 'valuation/pe_ttm', universe, industry))


def _ctx_pb_percentile(ctx, asof, period=1, universe='hs300', industry=None, history=252):
    return percentile(ctx, asof, period, history, lambda i: _ctx__valuation(ctx, i, 'valuation/pb_lf', universe, industry))


def _ctx__amount(ctx, i, universe, industry):
    a = row(ctx, 'd_essentials/amount', i)
    valid = np.isfinite(weights(ctx, i, universe, industry)) & np.isfinite(a)
    return float(a[valid].sum()) if valid.any() else np.nan


def _ctx_amount_percentile(ctx, asof, period=1, universe='hs300', industry=None, history=252):
    return percentile(ctx, asof, period, history, lambda i: _ctx__amount(ctx, i, universe, industry))


def _ctx_amount_share_percentile(ctx, asof, period=1, universe='hs300', industry=None, history=252):

    def share(i):
        total = _ctx__amount(ctx, i, None, None)
        return _ctx__amount(ctx, i, universe, industry) / total if total > 0 else np.nan
    return percentile(ctx, asof, period, history, share)


def _ctx_earnings_bond_spread(ctx, asof, period=1, universe='hs300', industry=None):
    return smooth(ctx, asof, period, lambda i: 1 / _ctx__valuation(ctx, i, 'valuation/pe_ttm', universe, industry) - bond_yield(ctx, i))


def _ctx_industry_relative_return(ctx, asof, period=1, universe=None, industry=None, benchmark='zzfull'):
    if industry is None:
        raise ValueError('industry is required')
    return _ctx_relative_return(ctx, asof, period, universe, industry, benchmark)


def _ctx__style_spread(ctx, i, period, field, universe, industry, quantile, high_first, leg=None):
    if not 0 < quantile < 0.5:
        raise ValueError('quantile must be between 0 and 0.5')
    start = i - period
    w = weights(ctx, start, universe, industry)
    if field == 'growth':

        def profit(j):
            pe = row(ctx, 'valuation/pe_ttm', j)
            return np.divide(row(ctx, 'd_essentials/total_mv', j), pe, out=np.full_like(pe, np.nan), where=pe > 0)
        old = profit(start - 252)
        score = np.divide(profit(start), old, out=np.full_like(old, np.nan), where=old > 0) - 1
    else:
        score = row(ctx, field, start)
    valid = np.isfinite(w) & np.isfinite(score)
    if field.startswith('valuation/'):
        valid &= score > 0
    if valid.sum() < 2:
        return np.nan
    ordered = np.flatnonzero(valid)[np.argsort(score[valid], kind='stable')]
    n = max(1, int(len(ordered) * quantile))
    low, high = (ordered[:n], ordered[-n:])
    r = returns(ctx, i, period)
    if leg is not None:
        selected = high if leg == 'high' else low
        return weighted_average(r[selected], w[selected])
    spread = weighted_average(r[high], w[high]) - weighted_average(r[low], w[low])
    return spread if high_first else -spread


def _ctx__style_percentile(ctx, asof, period, universe, industry, history, quantile, field, high_first=True):
    i = position(ctx, asof)
    if i < history + period - 1 or history < 2:
        raise ValueError('insufficient style history')
    v = np.array([_ctx__style_spread(ctx, j, period, field, universe, industry, quantile, high_first) for j in range(i - history + 1, i + 1)])
    return historical_percentile(v)


def _ctx_size_spread_percentile(ctx, asof, period=1, universe=None, industry=None, history=252, quantile=0.3):
    return _ctx__style_percentile(ctx, asof, period, universe, industry, history, quantile, 'd_essentials/total_mv')


def _ctx_growth_value_spread_percentile(ctx, asof, period=1, universe=None, industry=None, history=252, quantile=0.3):
    i = position(ctx, asof)
    if history < 2 or i < history + period + 251:
        raise ValueError('growth/value requires an extra 252 trading days')

    def spread(j):
        return _ctx__style_spread(ctx, j, period, 'growth', universe, industry, quantile, True, leg='high') - _ctx__style_spread(ctx, j, period, 'valuation/pb_lf', universe, industry, quantile, True, leg='low')
    v = np.array([spread(j) for j in range(i - history + 1, i + 1)])
    return historical_percentile(v)


def _ctx_pe_spread_percentile(ctx, asof, period=1, universe=None, industry=None, history=252, quantile=0.3):
    return _ctx__style_percentile(ctx, asof, period, universe, industry, history, quantile, 'valuation/pe_ttm')


def _ctx_pb_spread_percentile(ctx, asof, period=1, universe=None, industry=None, history=252, quantile=0.3):
    return _ctx__style_percentile(ctx, asof, period, universe, industry, history, quantile, 'valuation/pb_lf')


def _ctx_beta_spread_percentile(ctx, asof, period=1, universe=None, industry=None, history=252, quantile=0.3):
    return _ctx__style_percentile(ctx, asof, period, universe, industry, history, quantile, 'barra/beta')


def _ctx_rotation_speed_percentile(ctx, asof, period=1, universe=None, industry=None, history=252, rank_period=20):
    if industry is not None:
        raise ValueError('rotation speed compares all industries; leave industry=None')

    def speed(i):
        codes = np.unique(row(ctx, 'industry/industry', i - rank_period - 1))
        pairs = [(portfolio_return(ctx, i, rank_period, universe, c), portfolio_return(ctx, i - 1, rank_period, universe, c)) for c in codes if np.isfinite(c)]
        frame = pd.DataFrame(pairs).dropna()
        if len(frame) < 2:
            return np.nan
        ranks = frame.rank(pct=True)
        return float((ranks[0] - ranks[1]).abs().mean())
    return percentile(ctx, asof, period, history, speed)


def _ctx_snapshot(ctx, asof, periods=(1, 5, 10, 20), universes=('hs300', 'zz500', 'zz1000', 'zz2000'), industries=None, benchmark='zzfull', history=252):
    """Return the three screenshot blocks as DataFrames; data errors propagate."""
    metrics = {'点位252历史分位': partial(_ctx_level_percentile, ctx), '相对收益': partial(_ctx_relative_return, ctx), '新高新低差占比': partial(_ctx_new_high_low_share, ctx), 'PE_TTM历史分位': partial(_ctx_pe_percentile, ctx), 'PB_LF历史分位': partial(_ctx_pb_percentile, ctx), '成交额历史分位': partial(_ctx_amount_percentile, ctx), '成交额占比历史分位': partial(_ctx_amount_share_percentile, ctx), '盈利-10年国债收益率': partial(_ctx_earnings_bond_spread, ctx)}
    columns = pd.MultiIndex.from_product([universes, periods], names=['universe', 'period'])
    market = pd.DataFrame(index=metrics, columns=columns, dtype=float)
    for name, fn in metrics.items():
        for u, p in columns:
            kw = {'history': history} if '分位' in name else {}
            if name == '相对收益':
                kw['benchmark'] = benchmark
            market.loc[name, (u, p)] = fn(asof, p, u, **kw)
    if industries is None:
        from ..UpdateData.stock.update_industry import industry_to_id
        industries = list(industry_to_id)
    sector = pd.DataFrame({p: {ind: _ctx_industry_relative_return(ctx, asof, p, industry=ind, benchmark=benchmark) for ind in industries} for p in periods})
    styles = {'大小盘收益差分位': partial(_ctx_size_spread_percentile, ctx), '成长价值收益差分位': partial(_ctx_growth_value_spread_percentile, ctx), 'PE收益差分位': partial(_ctx_pe_spread_percentile, ctx), 'PB收益差分位': partial(_ctx_pb_spread_percentile, ctx), '高低Beta收益差分位': partial(_ctx_beta_spread_percentile, ctx), '行业轮动速度分位': partial(_ctx_rotation_speed_percentile, ctx)}
    style = pd.DataFrame({p: {name: fn(asof, p, history=history) for name, fn in styles.items()} for p in periods})
    return {'market': market, 'industry': sector, 'style': style}


def level_percentile(asof, period=1, universe='hs300', industry=None, history=252, *, data=None, conn=None, root=ROOT, context=None):
    ctx = context if context is not None else create_context(root=root, data=data, conn=conn)
    try:
        return _ctx_level_percentile(ctx, asof, period, universe, industry, history)
    finally:
        if context is None:
            close_context(ctx)


def relative_return(asof, period=1, universe='hs300', industry=None, benchmark='zzfull', *, data=None, conn=None, root=ROOT, context=None):
    ctx = context if context is not None else create_context(root=root, data=data, conn=conn)
    try:
        return _ctx_relative_return(ctx, asof, period, universe, industry, benchmark)
    finally:
        if context is None:
            close_context(ctx)


def new_high_low_share(asof, period=1, universe='hs300', industry=None, lookback=252, *, data=None, conn=None, root=ROOT, context=None):
    ctx = context if context is not None else create_context(root=root, data=data, conn=conn)
    try:
        return _ctx_new_high_low_share(ctx, asof, period, universe, industry, lookback)
    finally:
        if context is None:
            close_context(ctx)


def pe_percentile(asof, period=1, universe='hs300', industry=None, history=252, *, data=None, conn=None, root=ROOT, context=None):
    ctx = context if context is not None else create_context(root=root, data=data, conn=conn)
    try:
        return _ctx_pe_percentile(ctx, asof, period, universe, industry, history)
    finally:
        if context is None:
            close_context(ctx)


def pb_percentile(asof, period=1, universe='hs300', industry=None, history=252, *, data=None, conn=None, root=ROOT, context=None):
    ctx = context if context is not None else create_context(root=root, data=data, conn=conn)
    try:
        return _ctx_pb_percentile(ctx, asof, period, universe, industry, history)
    finally:
        if context is None:
            close_context(ctx)


def amount_percentile(asof, period=1, universe='hs300', industry=None, history=252, *, data=None, conn=None, root=ROOT, context=None):
    ctx = context if context is not None else create_context(root=root, data=data, conn=conn)
    try:
        return _ctx_amount_percentile(ctx, asof, period, universe, industry, history)
    finally:
        if context is None:
            close_context(ctx)


def amount_share_percentile(asof, period=1, universe='hs300', industry=None, history=252, *, data=None, conn=None, root=ROOT, context=None):
    ctx = context if context is not None else create_context(root=root, data=data, conn=conn)
    try:
        return _ctx_amount_share_percentile(ctx, asof, period, universe, industry, history)
    finally:
        if context is None:
            close_context(ctx)


def earnings_bond_spread(asof, period=1, universe='hs300', industry=None, *, data=None, conn=None, root=ROOT, context=None):
    ctx = context if context is not None else create_context(root=root, data=data, conn=conn)
    try:
        return _ctx_earnings_bond_spread(ctx, asof, period, universe, industry)
    finally:
        if context is None:
            close_context(ctx)


def industry_relative_return(asof, period=1, universe=None, industry=None, benchmark='zzfull', *, data=None, conn=None, root=ROOT, context=None):
    ctx = context if context is not None else create_context(root=root, data=data, conn=conn)
    try:
        return _ctx_industry_relative_return(ctx, asof, period, universe, industry, benchmark)
    finally:
        if context is None:
            close_context(ctx)


def size_spread_percentile(asof, period=1, universe=None, industry=None, history=252, quantile=0.3, *, data=None, conn=None, root=ROOT, context=None):
    ctx = context if context is not None else create_context(root=root, data=data, conn=conn)
    try:
        return _ctx_size_spread_percentile(ctx, asof, period, universe, industry, history, quantile)
    finally:
        if context is None:
            close_context(ctx)


def growth_value_spread_percentile(asof, period=1, universe=None, industry=None, history=252, quantile=0.3, *, data=None, conn=None, root=ROOT, context=None):
    ctx = context if context is not None else create_context(root=root, data=data, conn=conn)
    try:
        return _ctx_growth_value_spread_percentile(ctx, asof, period, universe, industry, history, quantile)
    finally:
        if context is None:
            close_context(ctx)


def pe_spread_percentile(asof, period=1, universe=None, industry=None, history=252, quantile=0.3, *, data=None, conn=None, root=ROOT, context=None):
    ctx = context if context is not None else create_context(root=root, data=data, conn=conn)
    try:
        return _ctx_pe_spread_percentile(ctx, asof, period, universe, industry, history, quantile)
    finally:
        if context is None:
            close_context(ctx)


def pb_spread_percentile(asof, period=1, universe=None, industry=None, history=252, quantile=0.3, *, data=None, conn=None, root=ROOT, context=None):
    ctx = context if context is not None else create_context(root=root, data=data, conn=conn)
    try:
        return _ctx_pb_spread_percentile(ctx, asof, period, universe, industry, history, quantile)
    finally:
        if context is None:
            close_context(ctx)


def beta_spread_percentile(asof, period=1, universe=None, industry=None, history=252, quantile=0.3, *, data=None, conn=None, root=ROOT, context=None):
    ctx = context if context is not None else create_context(root=root, data=data, conn=conn)
    try:
        return _ctx_beta_spread_percentile(ctx, asof, period, universe, industry, history, quantile)
    finally:
        if context is None:
            close_context(ctx)


def rotation_speed_percentile(asof, period=1, universe=None, industry=None, history=252, rank_period=20, *, data=None, conn=None, root=ROOT, context=None):
    ctx = context if context is not None else create_context(root=root, data=data, conn=conn)
    try:
        return _ctx_rotation_speed_percentile(ctx, asof, period, universe, industry, history, rank_period)
    finally:
        if context is None:
            close_context(ctx)


def snapshot(asof, periods=(1, 5, 10, 20), universes=('hs300', 'zz500', 'zz1000', 'zz2000'), industries=None, benchmark='zzfull', history=252, *, data=None, conn=None, root=ROOT, context=None):
    ctx = context if context is not None else create_context(root=root, data=data, conn=conn)
    try:
        return _ctx_snapshot(ctx, asof, periods, universes, industries, benchmark, history)
    finally:
        if context is None:
            close_context(ctx)
