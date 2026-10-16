"""Daily Barra-style exposures with validated input types and PIT annual growth.

Conventions: 252 trading days/year; 3-MAD clipping; cap-weighted centering,
equal-weighted scaling; sqrt(cap)-weighted regressions. Available descriptors
are reweighted; all-missing stays NaN. Two-year consensus growth remains a
proxy for long-term analyst growth. Cash earnings add reported depreciation
and amortization, using known fiscal-year values when interim data is absent.
These conventions do not reproduce a proprietary estimation universe exactly.

Momentum uses adjacent ratio-adjusted closes (not IPO issue-price returns).
EPS growth divides the annual slope by mean absolute EPS. Residual volatility
is standardized after Beta removal, then residualized against Beta and Size.
The latter sequence preserves the validated clipping/orthogonalization order.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if not __package__:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from v2.UpdateData.utils import date_index, ensure_memmap

BARRA_NAMES = ('beta', 'btop', 'size', 'nonlinear_size', 'momentom',
               'residual_vol', 'liquidity', 'leverage', 'growth1', 'earnings_yield')
YEAR_DAYS = 252
MIN_RETURN_OBS = 126
MIN_MOMENTUM_OBS = 252
MIN_ANNUAL_OBS = 3
MOMENTUM_DAYS = 504
MOMENTUM_LAG = 21
MONTH_DAYS = 21


# Input readers: validate each writer's dtype before touching financial values.


def _find(root, *names):
    for name in names:
        path = Path(root) / name
        if path.exists():
            return path
    raise FileNotFoundError(f"missing input: {', '.join(names)}")


def _mat(root, dates, ticks, *names, start=0, end=None):
    """Read a daily matrix using its writer's explicit dtype contract."""
    path = _find(root, *names)
    relative = path.relative_to(Path(root))
    dtype = np.dtype(np.float64 if relative.parts[0] == 'zyyx' else np.float32)
    shape = (len(dates), len(ticks))
    expected = int(np.prod(shape)) * dtype.itemsize
    if path.stat().st_size != expected:
        raise ValueError(f'{path}: expected {expected} bytes for {dtype}, {shape}')
    matrix = np.memmap(path, dtype=dtype, mode='r', shape=shape)
    return np.array(matrix[start:end], dtype=np.float64)


def _row(root, dates, ticks, dt, *names):
    return _mat(root, dates, ticks, *names, start=dt, end=dt + 1)[0]


def _divide(a, b):
    with np.errstate(all='ignore'):
        value = np.asarray(a, dtype=float) / b
    return np.where(np.isfinite(value), value, np.nan)


def _w(n, hl):
    value = np.exp2(-np.arange(n - 1, -1, -1) / hl)
    return value / value.sum()


def _mean(x, w, minimum=1):
    valid = np.isfinite(x)
    weights = valid * np.asarray(w)[:, None]
    result = _divide(np.sum(np.where(valid, x, 0) * weights, axis=0), weights.sum(0))
    result[valid.sum(0) < minimum] = np.nan
    return result


def _std(x, w, minimum=1):
    center = _mean(x, w, minimum)
    return np.sqrt(_mean((x - center) ** 2, w, minimum))


def _standardize(x, cap, *, clip=True):
    value = np.array(x, dtype=float, copy=True)
    valid = np.isfinite(value) & np.isfinite(cap) & (cap > 0)
    value[~valid] = np.nan
    if valid.sum() < 3:
        return np.full_like(value, np.nan)
    if clip:
        median = np.median(value[valid])
        mad = np.median(abs(value[valid] - median)) * 1.4826
        if mad > 0:
            value[valid] = np.clip(value[valid], median - 3 * mad, median + 3 * mad)
    center = np.average(value[valid], weights=cap[valid])
    scale = np.std(value[valid])
    if scale == 0:
        value[valid] = 0
        return value
    return _divide(value - center, scale)


def _residual(y, xs, cap):
    design = np.column_stack([np.ones(len(y)), *xs])
    valid = np.isfinite(y) & np.isfinite(design).all(1) & np.isfinite(cap) & (cap > 0)
    result = np.full(len(y), np.nan)
    if valid.sum() > design.shape[1]:
        # sqrt(cap) objective weight => premultiply design by cap**0.25.
        weights = cap[valid] ** .25
        weights /= weights.max()
        coef = np.linalg.lstsq(design[valid] * weights[:, None], y[valid] * weights, rcond=None)[0]
        result[valid] = y[valid] - design[valid] @ coef
    return result


def _combine(descriptors, weights, cap):
    values = np.array([_standardize(x, cap) for x in descriptors])
    effective = np.asarray(weights)[:, None] * np.isfinite(values)
    combined = _divide(np.nansum(values * effective, axis=0), effective.sum(0))
    return _standardize(combined, cap)


def _orthogonalize(value, exposures, cap):
    """Remove exposures with the shared weighted regression, then normalize."""
    return _standardize(_residual(value, exposures, cap), cap)


def _market_return(ret, mv):
    valid = np.isfinite(ret) & np.isfinite(mv) & (mv > 0)
    weights = np.where(valid, mv, 0)
    return _divide(np.sum(np.where(valid, ret, 0) * weights, axis=1), weights.sum(1))


def _beta(ret, market):
    x = np.broadcast_to(market[:, None], ret.shape).copy()
    valid = np.isfinite(ret) & np.isfinite(x)
    x[~valid] = np.nan
    y = np.where(valid, ret, np.nan)
    weights = _w(len(ret), 63)
    mx = _mean(x, weights, MIN_RETURN_OBS)
    my = _mean(y, weights, MIN_RETURN_OBS)
    return _divide(_mean((x - mx) * (y - my), weights, MIN_RETURN_OBS),
                   _mean((x - mx) ** 2, weights, MIN_RETURN_OBS))


def _risk_free(conn, dates):
    """Use latest known ten-year yield on each date, without backfill."""
    sql = """
        SELECT EndDate, Yield FROM Bond_CBYieldCurve
        WHERE CurveCode=10 AND YieldTypeCode=1 AND YearsToMaturity=10
          AND EndDate<=%(end)s
          AND (EndDate>=%(start)s OR EndDate=(
              SELECT MAX(EndDate) FROM Bond_CBYieldCurve
              WHERE CurveCode=10 AND YieldTypeCode=1 AND YearsToMaturity=10
                AND EndDate<%(start)s))
        ORDER BY EndDate
    """
    frame = pd.read_sql(sql, conn, params={'start': str(dates[0])[:10], 'end': str(dates[-1])[:10]})
    if frame.empty:
        raise ValueError('No risk-free yield history available')
    frame['EndDate'] = pd.to_datetime(frame.EndDate).dt.normalize()
    series = frame.drop_duplicates('EndDate', keep='last').set_index('EndDate')['Yield']
    annual = pd.to_numeric(series, errors='coerce').reindex(pd.DatetimeIndex(dates), method='ffill').to_numpy() / 100
    if np.any(~np.isfinite(annual) | (annual <= -1)):
        raise ValueError('Missing or invalid risk-free yield on requested dates')
    return np.expm1(np.log1p(annual) / YEAR_DAYS)


def _common_equity(root, dates, ticks, dt):
    net = _row(root, dates, ticks, dt, 'fundamental/balance/SEWithoutMI.bin')
    assets = _row(root, dates, ticks, dt, 'fundamental/balance/TotalAssets.bin')
    preferred = _row(root, dates, ticks, dt, 'fundamental/balance/EPreferStock.bin')
    # Optional preferred equity is not applicable on a populated balance
    # sheet when blank. Required financial totals are never zero-filled.
    preferred = np.where(np.isnan(preferred) & np.isfinite(net) & np.isfinite(assets), 0, preferred)
    return net - preferred, preferred, assets


def _calc_btop(root, dates, ticks, dt, cap):
    book, _, _ = _common_equity(root, dates, ticks, dt)
    return _standardize(_divide(book, cap), cap)


def _calc_size(cap):
    return np.log(np.where(np.isfinite(cap) & (cap > 0), cap, np.nan))


def _calc_nonlinear_size(size, cap):
    z = _standardize(size, cap, clip=False)
    return _orthogonalize(z ** 3, [z], cap)


def _calc_momentom(root, dates, ticks, dt, cap, rf):
    """504-day excess log return, lagged 21 days, half-life 126 days.

    Require two finite positive adjacent closes for each return. This leaves
    IPO first-day returns and unavailable prices missing, without zero fills.
    """
    end = dt + 1 - MOMENTUM_LAG
    start = end - MOMENTUM_DAYS
    if start < 1:
        return np.full(len(ticks), np.nan)
    prices = _mat(root, dates, ticks, 'd_essentials/close_ratio_adj.bin', start=start-1, end=end)
    prices = np.where(np.isfinite(prices) & (prices > 0), prices, np.nan)
    with np.errstate(all='ignore'):
        excess_logs = np.log(_divide(prices[1:], prices[:-1])) - np.log1p(rf[:MOMENTUM_DAYS, None])
    return _standardize(_mean(excess_logs, _w(MOMENTUM_DAYS, 126), MIN_MOMENTUM_OBS), cap)


def _calc_residual_vol(ret, mv, rf, beta, cap):
    if len(ret) != YEAR_DAYS:
        return np.full(len(cap), np.nan)
    excess = ret - rf[:, None]
    dastd = _std(excess, _w(YEAR_DAYS, 42), MIN_RETURN_OBS)
    market = _market_return(ret, mv) - rf
    weights = _w(YEAR_DAYS, 63)
    joint = np.isfinite(excess) & np.isfinite(market[:, None])
    x = np.where(joint, market[:, None], np.nan)
    y = np.where(joint, excess, np.nan)
    errors = y - _mean(y, weights, MIN_RETURN_OBS) - beta * (x - _mean(x, weights, MIN_RETURN_OBS))
    hsigma = _std(errors, weights, MIN_RETURN_OBS)
    with np.errstate(all='ignore'):
        logs = np.log1p(ret) - np.log1p(rf[:, None])
    # Complete 21-day months for CMRA; missing returns stay missing.
    months = logs.reshape(12, MONTH_DAYS, -1).sum(1)[::-1]
    cumulative = np.cumsum(months, axis=0)
    cmra = cumulative.max(0) - cumulative.min(0)
    vol = _combine([dastd, cmra, hsigma], [.74, .16, .10], cap)
    beta_exposure = _standardize(beta, cap)
    beta_residual = _orthogonalize(vol, [beta_exposure], cap)
    size_exposure = _standardize(_calc_size(cap), cap, clip=False)
    return _orthogonalize(beta_residual, [beta_exposure, size_exposure], cap)


def _calc_liquidity(root, dates, ticks, dt, cap, size):
    if dt + 1 < YEAR_DAYS:
        return np.full(len(ticks), np.nan)
    # TurnoverRate uses circulating shares. Barra's V/S descriptor uses
    # total shares outstanding; both local fields are in shares, not percent.
    volume = _mat(root, dates, ticks, 'd_essentials/volume.bin', start=dt-251, end=dt+1)
    shares = _mat(root, dates, ticks, 'd_essentials/total_share.bin', start=dt-251, end=dt+1)
    turnover = _divide(volume, np.where(shares > 0, shares, np.nan))
    turnover[turnover < 0] = np.nan
    months = turnover.reshape(12, MONTH_DAYS, -1).sum(1)
    with np.errstate(all='ignore'):
        stom = np.log(months[-1])
        stoq = np.log(months[-3:].mean(0))
        stoa = np.log(months.mean(0))
    value = _combine([stom, stoq, stoa], [.35, .35, .30], cap)
    return _orthogonalize(value, [_standardize(size, cap, clip=False)], cap)


def _calc_leverage(root, dates, ticks, dt, cap):
    book, preferred, assets = _common_equity(root, dates, ticks, dt)
    # Existing debt proxies are broader than interest-bearing debt.
    debt = _row(root, dates, ticks, dt, 'fundamental/balance/TotalNonCurrentLiability.bin')
    liabilities = _row(root, dates, ticks, dt, 'fundamental/balance/TotalLiability.bin')
    mlev = _divide(cap + preferred + debt, cap)
    dtoa = _divide(liabilities, np.where(assets > 0, assets, np.nan))
    blev = _divide(book + preferred + debt, book)
    return _combine([mlev, dtoa, blev], [.38, .35, .27], cap)


def _growth(y, *, absolute_mean=False):
    """Annual OLS; mean absolute observations handle mixed-sign EPS."""
    x = np.broadcast_to(np.arange(len(y))[:, None], y.shape)
    valid = np.isfinite(y)
    count = valid.sum(0)
    mx = _divide(np.sum(np.where(valid, x, 0), axis=0), count)
    my = _divide(np.nansum(y, axis=0), count)
    slope = _divide(np.nansum((x - mx) * (y - my), axis=0),
                    np.sum(np.where(valid, (x - mx) ** 2, 0), axis=0))
    denominator = _divide(np.sum(np.where(valid, abs(y), 0), axis=0), count) if absolute_mean else my
    result = _divide(slope, np.where(abs(my) > 1e-12, denominator, np.nan))
    result[count < MIN_ANNUAL_OBS] = np.nan
    return result


def _annual_financials(date, conn):
    """Fetch annual reports known at the target date, across both source tables."""
    target = pd.Timestamp(date).normalize()
    frames = []
    for table in ('LC_IncomeStatementAll', 'LC_STIBIncomeState'):
        sql = f"""
            SELECT s.SecuCode AS tick, f.EndDate AS end_date,
                   f.InfoPublDate AS publish_date, f.ID AS record_id,
                   f.BasicEPS, f.OperatingRevenue
            FROM {table} f LEFT JOIN SecuMain s ON f.CompanyCode=s.CompanyCode
            WHERE f.EndDate>=%(start)s AND f.EndDate<=%(date)s
              AND MONTH(f.EndDate)=12 AND DAY(f.EndDate)=31
              AND f.InfoPublDate<=%(date)s
              AND f.IfMerged=1 AND f.IfAdjusted=2 AND f.IfComplete=1
              AND s.SecuCategory=1 AND s.SecuMarket IN (83,90)
        """
        frames.append(pd.read_sql(sql, conn, params={
            'start': f'{target.year-8}-01-01', 'date': target.date()}))
    return pd.concat(frames, ignore_index=True)


def _annual_growth_inputs(date, dates, ticks, conn, root):
    """Align the last five fiscal years, retaining missing calendar-year gaps."""
    target = pd.Timestamp(date).normalize()
    frame = _annual_financials(target, conn)
    empty = np.full((5, len(ticks)), np.nan)
    if frame.empty:
        return empty.copy(), empty.copy()
    frame['tick'] = frame.tick.astype(str).str.zfill(6)
    frame['end_date'] = pd.to_datetime(frame.end_date)
    frame['publish_date'] = pd.to_datetime(frame.publish_date)
    frame = frame[(frame.publish_date <= target) & (frame.end_date <= target)].copy()
    frame['year'] = frame.end_date.dt.year
    frame = frame.sort_values(['tick', 'year', 'publish_date', 'record_id']).drop_duplicates(['tick', 'year'], keep='last')
    frame['offset'] = frame.year - frame.groupby('tick').year.transform('max')
    frame = frame[frame.offset >= -4].copy()
    frame['shares'] = np.nan
    for year in frame.year.unique():
        positions = np.flatnonzero(~np.isnat(dates) & (dates <= np.datetime64(f'{year}-12-31')))
        if len(positions):
            shares = _row(root, dates, ticks, int(positions[-1]), 'd_essentials/total_share.bin')
            lookup = pd.Series(shares, index=ticks)
            lookup = lookup[lookup.index != '']
            selected = frame.year == year
            frame.loc[selected, 'shares'] = frame.loc[selected, 'tick'].map(lookup)
    # Year-end shares approximate annual reported sales per share.
    frame['sales_per_share'] = _divide(frame.OperatingRevenue.to_numpy(), np.where(frame.shares > 0, frame.shares, np.nan))
    values = []
    for field in ('BasicEPS', 'sales_per_share'):
        frame[field] = pd.to_numeric(frame[field], errors='coerce')
        values.append(frame.pivot(index='offset', columns='tick', values=field)
                      .reindex(index=range(-4, 1), columns=ticks).to_numpy(dtype=float))
    return tuple(values)


def _calc_growth1(root, dates, ticks, dt, cap, conn):
    eps, sales = _annual_growth_inputs(dates[dt], dates, ticks, conn, root)
    egro, sgro = _growth(eps, absolute_mean=True), _growth(sales)
    # Explicit vendor proxies; convert percentage growth to decimals.
    long = _row(root, dates, ticks, dt, 'zyyx/con_forecast_roll/con_npcgrate_2y_roll.bin') / 100
    short = _row(root, dates, ticks, dt, 'zyyx/con_forecast_roll/con_np_yoy_roll.bin') / 100
    return _combine([sgro, egro, long, short], [.47, .24, .18, .11], cap)


def _cashflow_depreciation_reports(date, conn):
    """Read canonical cash-flow fields before main/STAR schema intersection.

    Query source reports rather than stale binaries: IntAAmort was previously
    lost when the writer intersected its name with IntangibleAssetAmortization.
    """
    from v2.UpdateData.stock.update_fundamental import _columns

    target = pd.Timestamp(date).normalize()
    frames = []
    for table, amortization in [('LC_CashFlowStatementAll', 'IntangibleAssetAmortization'),
                                ('LC_STIBCashFlowState', 'IntAAmort')]:
        names = {name.lower() for name in _columns(conn, table)}
        bulletin = 'AND f.BulletinType IN (20,30)' if 'bulletintype' in names else ''
        sql = f"""
            SELECT s.SecuCode AS tick, f.EndDate AS end_date,
                   f.InfoPublDate AS publish_date, f.ID AS record_id,
                   f.FixedAssetDepreciation,
                   f.{amortization} AS IntangibleAssetAmortization,
                   f.DeferredExpenseAmort
            FROM {table} f JOIN SecuMain s ON f.CompanyCode=s.CompanyCode
            WHERE f.InfoPublDate>=%(start)s AND f.InfoPublDate<=%(date)s
              AND f.EndDate<=%(date)s AND f.IfMerged=1
              AND f.IfAdjusted=2 AND f.IfComplete=1
              AND s.SecuCategory=1 AND s.SecuMarket IN (83,90) {bulletin}
        """
        frames.append(pd.read_sql(sql, conn, params={
            'start': (target - pd.DateOffset(years=3)).date(), 'date': target.date()}))
    return pd.concat(frames, ignore_index=True)


def _cash_earnings(date, ticks, profit, conn):
    """Parent-company TTM profit plus known depreciation/amortization.

    Missing interim TTM additions fall back to the latest reported fiscal
    values. Depreciation must be known; unreported amortization is optional.
    This follows the independently validated policy without future reports.
    """
    from v2.UpdateData.stock.update_fundamental import _ttm_values

    frame = _cashflow_depreciation_reports(date, conn).copy()
    if frame.empty:
        return np.full(len(ticks), np.nan)
    target = pd.Timestamp(date).normalize()
    frame['tick'] = frame.tick.astype(str).str.zfill(6)
    for field in ['end_date', 'publish_date']:
        frame[field] = pd.to_datetime(frame[field])
    frame = frame[(frame.publish_date <= target) & (frame.end_date <= target)
                  & (frame.publish_date >= target - pd.DateOffset(years=3))]
    frame = frame.sort_values(['tick', 'end_date', 'publish_date', 'record_id'])
    annual = frame[(frame.end_date.dt.month == 12) & (frame.end_date.dt.day == 31)]
    annual = annual.drop_duplicates('tick', keep='last').set_index('tick')
    positions = np.flatnonzero(ticks != '')
    valid_ticks = ticks[positions].tolist()
    additions = []
    for field in ['FixedAssetDepreciation', 'IntangibleAssetAmortization', 'DeferredExpenseAmort']:
        frame[field] = pd.to_numeric(frame[field], errors='coerce')
        interim = _ttm_values(frame, field, valid_ticks, ticks, positions).astype(float)
        fiscal = pd.to_numeric(annual[field], errors='coerce').reindex(ticks).to_numpy(dtype=float)
        additions.append(np.where(np.isfinite(interim), interim, fiscal))
    depreciation, amortization, deferred = additions
    return (profit + depreciation + np.where(np.isfinite(amortization), amortization, 0)
            + np.where(np.isfinite(deferred), deferred, 0))


def _calc_earnings_yield(root, dates, ticks, dt, cap, conn):
    forecast = _row(root, dates, ticks, dt, 'zyyx/con_forecast_roll/con_np_roll.bin') * 10000
    profit = _row(root, dates, ticks, dt, 'fundamental/income_ttm/NPParentCompanyOwners.bin')
    cash = _cash_earnings(dates[dt], ticks, profit, conn)
    return _combine([_divide(forecast, cap), _divide(profit, cap), _divide(cash, cap)], [.68, .11, .21], cap)


def calculate_barra(date, dates, ticks, conn, root):
    """Calculate production exposures without writing output files."""
    dates, ticks = np.asarray(dates), np.asarray(ticks)
    dt = date_index(date, dates)
    root = Path(root)
    cap = _row(root, dates, ticks, dt, 'd_essentials/total_mv.bin')
    cap = np.where((ticks != '') & np.isfinite(cap) & (cap > 0), cap, np.nan)
    start = max(0, dt + 1 - MOMENTUM_LAG - MOMENTUM_DAYS)
    rf = _risk_free(conn, dates[start:dt + 1])
    ret = _mat(root, dates, ticks, 'd_essentials/pct.bin', start=max(0, dt-251), end=dt+1) / 100
    ret[ret <= -1] = np.nan
    mv = _mat(root, dates, ticks, 'd_essentials/total_mv.bin', start=max(0, dt-251), end=dt+1)
    raw_beta = np.full(len(ticks), np.nan)
    if len(ret) == YEAR_DAYS:
        raw_beta = _beta(ret - rf[-YEAR_DAYS:, None], _market_return(ret, mv) - rf[-YEAR_DAYS:])
    size = _calc_size(cap)
    values = {
        'beta': _standardize(raw_beta, cap),
        'btop': _calc_btop(root, dates, ticks, dt, cap),
        # Keep the established raw log-market-cap size output.
        'size': size,
        'nonlinear_size': _calc_nonlinear_size(size, cap),
        'momentom': _calc_momentom(root, dates, ticks, dt, cap, rf),
        'residual_vol': _calc_residual_vol(ret, mv, rf[-YEAR_DAYS:], raw_beta, cap),
        'liquidity': _calc_liquidity(root, dates, ticks, dt, cap, size),
        'leverage': _calc_leverage(root, dates, ticks, dt, cap),
        'growth1': _calc_growth1(root, dates, ticks, dt, cap, conn),
        'earnings_yield': _calc_earnings_yield(root, dates, ticks, dt, cap, conn),
    }
    for name, value in values.items():
        values[name] = np.where(np.isfinite(value) & np.isfinite(cap), value, np.nan)
    return values


def _save(root, dates, ticks, dt, values):
    for name, value in values.items():
        matrix = ensure_memmap(Path(root) / 'barra' / f'{name}.bin',
                               (len(dates), len(ticks)), np.float64, np.nan)
        matrix[dt] = value
        matrix.flush()


def update_barra(date, dates, ticks, conn, root, *, output_root=None):
    """Compute and save one date; output_root isolates validation outputs.

    Both roots are stock directories. Daily callers retain their original
    five-argument interface and save to root/barra as before.
    """
    values = calculate_barra(date, dates, ticks, conn, root)
    _save(root if output_root is None else output_root, dates, ticks,
          date_index(date, dates), values)
    return values


__all__ = ['BARRA_NAMES', 'calculate_barra', 'update_barra']


if __name__ == '__main__':
    import argparse
    from v2.UpdateData.config import ROOT, get_jy_conn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', required=True)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    dates = np.load(args.root / 'axis/dates.npy', allow_pickle=False)
    ticks = np.load(args.root / 'axis/stock_ticks.npy', allow_pickle=False)
    connection = get_jy_conn()
    try:
        update_barra(args.date, dates, ticks, connection, args.root / 'stock')
    finally:
        connection.close()
