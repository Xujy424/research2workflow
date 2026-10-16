"""市场与风格监控。收益/分位均用小数表示，详见 style.md。"""
from __future__ import annotations
from functools import partial
import numpy as np
import pandas as pd
from ..GetData import DataPool
from ..UpdateData.config import ROOT
INDEX_NAMES = {'沪深300': 'hs300', '中证500': 'zz500', '中证1000': 'zz1000', '中证2000': 'zz2000', '000300': 'hs300', '000905': 'zz500', '000852': 'zz1000', '932000': 'zz2000', '000985': 'zzfull'}

def create_context(root=ROOT, *, data=None, conn=None, start='2005-01-01', end=None):
    ctx = {}
    ctx['conn'] = conn
    ctx['_owns_conn'] = False
    ctx['_owns_data'] = data is None
    if data is None:
        try:
            data = DataPool(root, asset='stock')
        except FileNotFoundError:
            data = None
    ctx['data'] = data
    ctx['_nav_cache'] = {}
    if data is not None:
        ctx['dates'] = pd.DatetimeIndex(data.axis.trade_dates)
        ctx['ticks'] = np.asarray(data.axis.ticks)
    else:
        end = pd.Timestamp(end or pd.Timestamp.now(tz='Asia/Shanghai').date())
        frame = pd.read_sql('SELECT DISTINCT TradingDay FROM QT_StockPerformance WHERE TradingDay BETWEEN %s AND %s ORDER BY TradingDay', _ctx_connection(ctx), params=(pd.Timestamp(start).date(), end.date()))
        ctx['dates'] = pd.DatetimeIndex(frame.iloc[:, 0])
        frame = pd.read_sql('SELECT DISTINCT SecuCode FROM SecuMain WHERE SecuCategory=1 AND SecuMarket IN (83,90) ORDER BY SecuCode', _ctx_connection(ctx))
        ctx['ticks'] = frame.iloc[:, 0].astype(str).to_numpy()
    return ctx

def _ctx_connection(ctx):
    if ctx['conn'] is None:
        from ..UpdateData.config import get_jy_conn
        ctx['conn'] = get_jy_conn()
        ctx['_owns_conn'] = True
    return ctx['conn']

def _ctx_position(ctx, asof):
    i = ctx['dates'].searchsorted(pd.Timestamp(asof), side='right') - 1
    if i < 0:
        raise ValueError('asof precedes available trading dates')
    if pd.Timestamp(asof) > ctx['dates'][-1]:
        raise ValueError('asof exceeds data calendar; refresh axes or use a remote-only context')
    return int(i)

def _ctx_row(ctx, field, i):
    if i < 0 or i >= len(ctx['dates']):
        raise ValueError('insufficient trading history')
    if ctx['data'] is not None:
        try:
            values = np.asarray(ctx['data'].read(field, ctx['dates'][i]), float)
            if np.isfinite(values).any():
                return values
        except (FileNotFoundError, KeyError):
            pass
    values = np.asarray(_ctx__remote(ctx, field, i), float)
    if not np.isfinite(values).any():
        raise ValueError(f"no observations for {field} on {ctx['dates'][i].date()}")
    return values

def _ctx__daily(ctx, i):
    from ..UpdateData.stock.update_essentials import update_dfield
    return update_dfield(str(ctx['dates'][i].date()), ctx['ticks'], _ctx_connection(ctx))

def _ctx__remote(ctx, field, i):
    date = str(ctx['dates'][i].date())
    if field.startswith('d_essentials/'):
        return _ctx__daily(ctx, i)[field.split('/')[-1]]
    if field.startswith('index/weight/'):
        from ..UpdateData.stock.update_index import _read_latest_index_snapshot
        from ..UpdateData.stock.update_index import INDEX_CODE_TO_NAME
        name = field.split('/')[-1].removesuffix('_weight')
        code = next((k for k, v in INDEX_CODE_TO_NAME.items() if v == name))
        frame = _read_latest_index_snapshot(date, _ctx_connection(ctx)).to_pandas()
        return frame.loc[frame.index_code == code].set_index('tick').weight.reindex(ctx['ticks']).to_numpy()
    if field == 'industry/industry':
        from ..UpdateData.stock.update_industry import _daily_industry_codes
        return _daily_industry_codes(date, ctx['ticks'], _ctx_connection(ctx))[0]
    if field in {'valuation/pe_ttm', 'valuation/pb_lf'}:
        names = ('PE_TTM', 'PETTM', 'PE') if field.endswith('pe_ttm') else ('PB', 'PB_LF')
        parts = []
        for table in ('LC_DIndicesForValuation', 'LC_STIBDIndices'):
            cols = pd.read_sql('SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME=%s', _ctx_connection(ctx), params=(table,))
            lookup = {str(c).lower(): str(c) for c in cols.iloc[:, 0]}
            candidates = names[:2] if field.endswith('pe_ttm') else names
            col = next((lookup[n.lower()] for n in candidates if n.lower() in lookup), None)
            if col is None:
                continue
            out = pd.read_sql(f'SELECT s.SecuCode AS tick, a.[{col}] AS value FROM [{table}] a JOIN SecuMain s ON a.InnerCode=s.InnerCode WHERE a.TradingDay=%s AND s.SecuCategory=1 AND s.SecuMarket IN (83,90)', _ctx_connection(ctx), params=(date,))
            if not out.empty:
                parts.append(out)
        if parts:
            return pd.concat(parts).drop_duplicates('tick').set_index('tick').value.reindex(ctx['ticks']).to_numpy()
        raise ValueError(f'Juyuan valuation schema unavailable: {field}')
    if field == 'barra/beta':
        if i < 252:
            raise ValueError('beta requires 252 trading days')
        r = np.array([_ctx_row(ctx, 'd_essentials/pct', j) / 100 for j in range(i - 251, i + 1)])
        market = np.array([_ctx__average(ctx, r[k], _ctx_weights(ctx, i - 252 + k)) for k in range(252)])
        valid = np.isfinite(r) & np.isfinite(market[:, None])
        x = np.where(valid, market[:, None], np.nan)
        y = np.where(valid, r, np.nan)
        xc, yc = (x - np.nanmean(x, axis=0), y - np.nanmean(y, axis=0))
        den = np.nansum(xc ** 2, axis=0)
        return np.divide(np.nansum(xc * yc, axis=0), den, out=np.full(len(ctx['ticks']), np.nan), where=(den > 0) & (valid.sum(0) >= 200))
    raise ValueError(f'no Juyuan fallback registered for {field}')

def _ctx__average(ctx, values, weights):
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return np.average(values[valid], weights=weights[valid]) if valid.any() else np.nan

def _ctx_weights(ctx, i, universe=None, industry=None):
    if universe in (None, 'self', 'all'):
        w = _ctx_row(ctx, 'd_essentials/circ_mv', i).copy()
    else:
        name = INDEX_NAMES.get(universe, universe)
        w = _ctx_row(ctx, f'index/weight/{name}_weight', i).copy()
    if industry is not None:
        if isinstance(industry, str):
            from ..UpdateData.stock.update_industry import industry_to_id
            industry = industry_to_id[industry]
        w[_ctx_row(ctx, 'industry/industry', i) != industry] = np.nan
    return np.where(w > 0, w, np.nan)

def _ctx_returns(ctx, i, period):
    if not isinstance(period, int) or period < 1 or i < period:
        raise ValueError('period must be a positive integer with sufficient history')
    r = np.array([_ctx_row(ctx, 'd_essentials/pct', j) / 100 for j in range(i - period + 1, i + 1)])
    return np.prod(1 + r, axis=0) - 1

def _ctx_portfolio_return(ctx, i, period=1, universe=None, industry=None):
    return _ctx__average(ctx, _ctx_returns(ctx, i, period), _ctx_weights(ctx, i - period, universe, industry))

def _ctx__smooth(ctx, asof, period, fn):
    i = _ctx_position(ctx, asof)
    if not isinstance(period, int) or period < 1 or i < period - 1:
        raise ValueError('invalid period or insufficient history')
    values = np.array([fn(j) for j in range(i - period + 1, i + 1)])
    return float(values.mean())

def _ctx__percentile(ctx, asof, period, history, fn):
    i = _ctx_position(ctx, asof)
    if not isinstance(history, int) or history < 2 or i < history + period - 2:
        raise ValueError('insufficient percentile history')
    values = np.array([_ctx__smooth(ctx, ctx['dates'][j], period, fn) for j in range(i - history + 1, i + 1)])
    if not np.isfinite(values).all():
        return np.nan
    return float(((values < values[-1]).sum() + 0.5 * (values == values[-1]).sum()) / history)

def _ctx__nav(ctx, i, universe, industry):
    key = (universe, industry)
    nav = ctx['_nav_cache'].setdefault(key, [1.0])
    while len(nav) <= i:
        nav.append(nav[-1] * (1 + _ctx_portfolio_return(ctx, len(nav), 1, universe, industry)))
    return nav[i]

def _ctx_level_percentile(ctx, asof, period=1, universe='hs300', industry=None, history=252):
    """Reconstructed constituent-portfolio NAV percentile (not official index close)."""
    return _ctx__percentile(ctx, asof, period, history, lambda i: _ctx__nav(ctx, i, universe, industry))

def _ctx_relative_return(ctx, asof, period=1, universe='hs300', industry=None, benchmark='zzfull'):
    i = _ctx_position(ctx, asof)
    return _ctx_portfolio_return(ctx, i, period, universe, industry) - _ctx_portfolio_return(ctx, i, period, benchmark)

def _ctx_new_high_low_share(ctx, asof, period=1, universe='hs300', industry=None, lookback=252):

    def daily(i):
        if lookback < 2 or i < lookback:
            raise ValueError('insufficient high/low lookback')
        r = np.array([_ctx_row(ctx, 'd_essentials/pct', j) / 100 for j in range(i - lookback + 1, i + 1)])
        nav = np.vstack([np.ones(len(ctx['ticks'])), np.cumprod(1 + r, axis=0)])
        valid = np.isfinite(nav).all(0) & np.isfinite(_ctx_weights(ctx, i, universe, industry))
        high = nav[-1] > np.max(nav[:-1], axis=0)
        low = nav[-1] < np.min(nav[:-1], axis=0)
        return float((high[valid].sum() - low[valid].sum()) / valid.sum()) if valid.any() else np.nan
    return _ctx__smooth(ctx, asof, period, daily)

def _ctx__valuation(ctx, i, field, universe, industry):
    v = _ctx_row(ctx, field, i)
    reciprocal = np.divide(1, v, out=np.full_like(v, np.nan), where=v > 0)
    avg = _ctx__average(ctx, reciprocal, _ctx_weights(ctx, i, universe, industry))
    return 1 / avg if np.isfinite(avg) and avg > 0 else np.nan

def _ctx_pe_percentile(ctx, asof, period=1, universe='hs300', industry=None, history=252):
    return _ctx__percentile(ctx, asof, period, history, lambda i: _ctx__valuation(ctx, i, 'valuation/pe_ttm', universe, industry))

def _ctx_pb_percentile(ctx, asof, period=1, universe='hs300', industry=None, history=252):
    return _ctx__percentile(ctx, asof, period, history, lambda i: _ctx__valuation(ctx, i, 'valuation/pb_lf', universe, industry))

def _ctx__amount(ctx, i, universe, industry):
    a = _ctx_row(ctx, 'd_essentials/amount', i)
    valid = np.isfinite(_ctx_weights(ctx, i, universe, industry)) & np.isfinite(a)
    return float(a[valid].sum()) if valid.any() else np.nan

def _ctx_amount_percentile(ctx, asof, period=1, universe='hs300', industry=None, history=252):
    return _ctx__percentile(ctx, asof, period, history, lambda i: _ctx__amount(ctx, i, universe, industry))

def _ctx_amount_share_percentile(ctx, asof, period=1, universe='hs300', industry=None, history=252):

    def share(i):
        total = _ctx__amount(ctx, i, None, None)
        return _ctx__amount(ctx, i, universe, industry) / total if total > 0 else np.nan
    return _ctx__percentile(ctx, asof, period, history, share)

def _ctx__bond_yield(ctx, i):
    out = pd.read_sql('SELECT TOP 1 Yield FROM Bond_CBYieldCurve WHERE CurveCode=10 AND YieldTypeCode=1 AND YearsToMaturity=10 AND EndDate<=%s ORDER BY EndDate DESC', _ctx_connection(ctx), params=(str(ctx['dates'][i].date()),))
    return float(out.iloc[0, 0]) / 100 if not out.empty else np.nan

def _ctx_earnings_bond_spread(ctx, asof, period=1, universe='hs300', industry=None):
    return _ctx__smooth(ctx, asof, period, lambda i: 1 / _ctx__valuation(ctx, i, 'valuation/pe_ttm', universe, industry) - _ctx__bond_yield(ctx, i))

def _ctx_industry_relative_return(ctx, asof, period=1, universe=None, industry=None, benchmark='zzfull'):
    if industry is None:
        raise ValueError('industry is required')
    return _ctx_relative_return(ctx, asof, period, universe, industry, benchmark)

def _ctx__style_spread(ctx, i, period, field, universe, industry, quantile, high_first, leg=None):
    if not 0 < quantile < 0.5:
        raise ValueError('quantile must be between 0 and 0.5')
    start = i - period
    w = _ctx_weights(ctx, start, universe, industry)
    if field == 'growth':

        def profit(j):
            pe = _ctx_row(ctx, 'valuation/pe_ttm', j)
            return np.divide(_ctx_row(ctx, 'd_essentials/total_mv', j), pe, out=np.full_like(pe, np.nan), where=pe > 0)
        old = profit(start - 252)
        score = np.divide(profit(start), old, out=np.full_like(old, np.nan), where=old > 0) - 1
    else:
        score = _ctx_row(ctx, field, start)
    valid = np.isfinite(w) & np.isfinite(score)
    if field.startswith('valuation/'):
        valid &= score > 0
    if valid.sum() < 2:
        return np.nan
    ordered = np.flatnonzero(valid)[np.argsort(score[valid], kind='stable')]
    n = max(1, int(len(ordered) * quantile))
    low, high = (ordered[:n], ordered[-n:])
    r = _ctx_returns(ctx, i, period)
    if leg is not None:
        selected = high if leg == 'high' else low
        return _ctx__average(ctx, r[selected], w[selected])
    spread = _ctx__average(ctx, r[high], w[high]) - _ctx__average(ctx, r[low], w[low])
    return spread if high_first else -spread

def _ctx__style_percentile(ctx, asof, period, universe, industry, history, quantile, field, high_first=True):
    i = _ctx_position(ctx, asof)
    if i < history + period - 1 or history < 2:
        raise ValueError('insufficient style history')
    v = np.array([_ctx__style_spread(ctx, j, period, field, universe, industry, quantile, high_first) for j in range(i - history + 1, i + 1)])
    return float(((v < v[-1]).sum() + 0.5 * (v == v[-1]).sum()) / history) if np.isfinite(v).all() else np.nan

def _ctx_size_spread_percentile(ctx, asof, period=1, universe=None, industry=None, history=252, quantile=0.3):
    return _ctx__style_percentile(ctx, asof, period, universe, industry, history, quantile, 'd_essentials/total_mv')

def _ctx_growth_value_spread_percentile(ctx, asof, period=1, universe=None, industry=None, history=252, quantile=0.3):
    i = _ctx_position(ctx, asof)
    if history < 2 or i < history + period + 251:
        raise ValueError('growth/value requires an extra 252 trading days')

    def spread(j):
        return _ctx__style_spread(ctx, j, period, 'growth', universe, industry, quantile, True, leg='high') - _ctx__style_spread(ctx, j, period, 'valuation/pb_lf', universe, industry, quantile, True, leg='low')
    v = np.array([spread(j) for j in range(i - history + 1, i + 1)])
    return float(((v < v[-1]).sum() + 0.5 * (v == v[-1]).sum()) / history) if np.isfinite(v).all() else np.nan

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
        codes = np.unique(_ctx_row(ctx, 'industry/industry', i - rank_period - 1))
        pairs = [(_ctx_portfolio_return(ctx, i, rank_period, universe, c), _ctx_portfolio_return(ctx, i - 1, rank_period, universe, c)) for c in codes if np.isfinite(c)]
        frame = pd.DataFrame(pairs).dropna()
        if len(frame) < 2:
            return np.nan
        ranks = frame.rank(pct=True)
        return float((ranks[0] - ranks[1]).abs().mean())
    return _ctx__percentile(ctx, asof, period, history, speed)

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


def close_context(context):
    """Close only resources opened by create_context; caller-owned objects stay open."""
    if context['_owns_data'] and context['data'] is not None:
        context['data'].close()
    if context['_owns_conn'] and context['conn'] is not None:
        context['conn'].close()
        context['conn'] = None
        context['_owns_conn'] = False
    context.pop('_cache', None)
    context['_nav_cache'].clear()


def _cached(function):
    def read(context, *args):
        cache = context.setdefault('_cache', {})
        key = (function.__name__, args)
        if key not in cache:
            cache[key] = function(context, *args)
        return cache[key]
    return read


_ctx_row = _cached(_ctx_row)
_ctx__daily = _cached(_ctx__daily)
_ctx__bond_yield = _cached(_ctx__bond_yield)
