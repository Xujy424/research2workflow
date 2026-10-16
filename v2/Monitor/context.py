"""Monitor data access: local DataPool reads and missing-data fallbacks."""
from __future__ import annotations

import numpy as np
import pandas as pd
from functools import wraps

from ..GetData import DataPool
from ..UpdateData.config import ROOT

def create_context(root=ROOT, *, data=None, conn=None, start='2005-01-01', end=None):
    """Create a reusable data context; injected resources remain caller-owned."""
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
    if data is not None:
        ctx['dates'] = pd.DatetimeIndex(data.axis.trade_dates)
        ctx['ticks'] = np.asarray(data.axis.ticks)
    else:
        end = pd.Timestamp(end or pd.Timestamp.now(tz='Asia/Shanghai').date())
        frame = pd.read_sql('SELECT DISTINCT TradingDay FROM QT_StockPerformance WHERE TradingDay BETWEEN %s AND %s ORDER BY TradingDay', connection(ctx), params=(pd.Timestamp(start).date(), end.date()))
        ctx['dates'] = pd.DatetimeIndex(frame.iloc[:, 0])
        frame = pd.read_sql('SELECT DISTINCT SecuCode FROM SecuMain WHERE SecuCategory=1 AND SecuMarket IN (83,90) ORDER BY SecuCode', connection(ctx))
        ctx['ticks'] = frame.iloc[:, 0].astype(str).to_numpy()
    return ctx


def connection(ctx):
    if ctx['conn'] is None:
        from ..UpdateData.config import get_jy_conn
        ctx['conn'] = get_jy_conn()
        ctx['_owns_conn'] = True
    return ctx['conn']


def row(ctx, field, i):
    """Read one aligned local row, falling back when absent or entirely missing."""
    if i < 0 or i >= len(ctx['dates']):
        raise ValueError('insufficient trading history')
    if ctx['data'] is not None:
        try:
            values = np.asarray(ctx['data'].read(field, ctx['dates'][i]), float)
            if np.isfinite(values).any():
                return values
        except (FileNotFoundError, KeyError):
            pass
    values = np.asarray(remote(ctx, field, i), float)
    if not np.isfinite(values).any():
        raise ValueError(f"no observations for {field} on {ctx['dates'][i].date()}")
    return values


def daily(ctx, i):
    from ..UpdateData.stock.update_essentials import update_dfield
    return update_dfield(str(ctx['dates'][i].date()), ctx['ticks'], connection(ctx))


def remote(ctx, field, i):
    date = str(ctx['dates'][i].date())
    if field.startswith('d_essentials/'):
        return daily(ctx, i)[field.split('/')[-1]]
    if field.startswith('index/weight/'):
        from ..UpdateData.stock.update_index import _read_latest_index_snapshot
        from ..UpdateData.stock.update_index import INDEX_CODE_TO_NAME
        name = field.split('/')[-1].removesuffix('_weight')
        code = next((k for k, v in INDEX_CODE_TO_NAME.items() if v == name))
        frame = _read_latest_index_snapshot(date, connection(ctx)).to_pandas()
        return frame.loc[frame.index_code == code].set_index('tick').weight.reindex(ctx['ticks']).to_numpy()
    if field == 'industry/industry':
        from ..UpdateData.stock.update_industry import _daily_industry_codes
        return _daily_industry_codes(date, ctx['ticks'], connection(ctx))[0]
    if field in {'valuation/pe_ttm', 'valuation/pb_lf'}:
        names = ('PE_TTM', 'PETTM', 'PE') if field.endswith('pe_ttm') else ('PB', 'PB_LF')
        parts = []
        for table in ('LC_DIndicesForValuation', 'LC_STIBDIndices'):
            cols = pd.read_sql('SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME=%s', connection(ctx), params=(table,))
            lookup = {str(c).lower(): str(c) for c in cols.iloc[:, 0]}
            candidates = names[:2] if field.endswith('pe_ttm') else names
            col = next((lookup[n.lower()] for n in candidates if n.lower() in lookup), None)
            if col is None:
                continue
            out = pd.read_sql(f'SELECT s.SecuCode AS tick, a.[{col}] AS value FROM [{table}] a JOIN SecuMain s ON a.InnerCode=s.InnerCode WHERE a.TradingDay=%s AND s.SecuCategory=1 AND s.SecuMarket IN (83,90)', connection(ctx), params=(date,))
            if not out.empty:
                parts.append(out)
        if parts:
            return pd.concat(parts).drop_duplicates('tick').set_index('tick').value.reindex(ctx['ticks']).to_numpy()
        raise ValueError(f'Juyuan valuation schema unavailable: {field}')
    if field == 'barra/beta':
        from .utils import beta
        return beta(ctx, i)
    raise ValueError(f'no Juyuan fallback registered for {field}')


def bond_yield(ctx, i):
    out = pd.read_sql(
        'SELECT TOP 1 Yield FROM Bond_CBYieldCurve ' \
        'WHERE CurveCode=10 ' \
        'AND YieldTypeCode=1 ' \
        'AND YearsToMaturity=10 ' \
        'AND EndDate<=%s ORDER BY EndDate DESC', 
        connection(ctx), 
        params=(str(ctx['dates'][i].date()),)
    )
    return float(out.iloc[0, 0]) / 100 if not out.empty else np.nan


def close_context(context):
    """Close only resources opened by create_context; caller-owned objects stay open."""
    if context['_owns_data'] and context['data'] is not None:
        context['data'].close()
    if context['_owns_conn'] and context['conn'] is not None:
        context['conn'].close()
        context['conn'] = None
        context['_owns_conn'] = False
    context.pop('_cache', None)
    context.pop('_nav_cache', None)


def _cached(function):
    @wraps(function)
    def read(context, *args):
        cache = context.setdefault('_cache', {})
        key = (function.__name__, args)
        if key not in cache:
            cache[key] = function(context, *args)
        return cache[key]
    return read

row = _cached(row)
daily = _cached(daily)
bond_yield = _cached(bond_yield)
