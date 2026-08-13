"""Daily incremental Barra factor maintenance."""
from pathlib import Path
import numpy as np
import pandas as pd
BARRA_NAMES = ('beta', 'btop', 'size', 'nonlinear_size', 'momentom', 'residual_vol', 'liquidity', 'leverage', 'growth1', 'earnings_yield')

def _find(root, *names):
    for name in names:
        path = Path(root) / name
        if path.exists():
            return path
    raise FileNotFoundError(f"missing input: {', '.join(names)}")

def _mat(root, dates, ticks, *names, start=0, end=None):
    a = np.memmap(_find(root, *names), dtype=float, mode='r', shape=(len(dates), len(ticks)))
    return np.asarray(a[start:end])

def _row(root, dates, ticks, dt, *names):
    return _mat(root, dates, ticks, *names, start=dt, end=dt + 1)[0]

def _w(n, hl):
    x = np.exp(-np.log(2) / hl * np.arange(n - 1, -1, -1))
    return x / x.sum()

def _mean(x, w):
    ew = np.isfinite(x) * w[:, None]
    d = ew.sum(0)
    return np.divide(np.nansum(ew * x, 0), d, out=np.full(x.shape[1], np.nan), where=d != 0)

def _std(x, w):
    m = _mean(x, w)
    ew = np.isfinite(x) * w[:, None]
    d = ew.sum(0)
    return np.sqrt(np.divide(np.nansum(ew * (x - m) ** 2, 0), d, out=np.full(x.shape[1], np.nan), where=d != 0))

def _beta(r, rm):
    w = _w(len(rm), 62)
    ew = (np.isfinite(r) & np.isfinite(rm[:, None])) * w[:, None]
    d = ew.sum(0)
    mx = np.divide(np.nansum(ew * rm[:, None], 0), d, out=np.full(r.shape[1], np.nan), where=d != 0)
    my = np.divide(np.nansum(ew * r, 0), d, out=np.full(r.shape[1], np.nan), where=d != 0)
    cov = np.nansum(ew * (rm[:, None] - mx) * (r - my), 0)
    var = np.nansum(ew * (rm[:, None] - mx) ** 2, 0)
    return np.divide(cov, var, out=np.full_like(cov, np.nan), where=var != 0)

def _growth(y):
    x = np.arange(1, len(y) + 1.0)
    xc = x - x.mean()
    m = np.nanmean(y, 0)
    slope = np.nansum(xc[:, None] * (y - m), 0) / np.sum(xc ** 2)
    return np.divide(slope, m, out=np.full_like(m, np.nan), where=(abs(m) > 1e-12) & (np.isfinite(y).sum(0) >= 2))

def _save(root, dates, ticks, dt, values):
    folder = Path(root) / 'barra'
    folder.mkdir(parents=True, exist_ok=True)
    size = len(dates) * len(ticks) * 8
    for name, value in values.items():
        p = folder / f'{name}.bin'
        if not p.exists():
            with p.open('wb') as f:
                f.truncate(size)
            a = np.memmap(p, dtype=float, mode='r+', shape=(len(dates), len(ticks)))
            a[:] = np.nan
            a.flush()
        if p.stat().st_size != size:
            raise ValueError(f'{p} size does not match axes')
        a = np.memmap(p, dtype=float, mode='r+', shape=(len(dates), len(ticks)))
        a[dt] = value
        a.flush()

def update_barra(date, dates, ticks, conn, root):
    """Compute all ten factors for date and overwrite only its matrix row."""
    ds = np.asarray(dates, dtype='datetime64[D]')
    target = np.datetime64(date)
    dt = int(np.searchsorted(ds, target))
    if dt >= len(ds) or ds[dt] != target:
        raise ValueError(f'{date} is not in dates')
    n = len(ticks)
    sql = f"select top 1 Yield from Bond_CBYieldCurve where CurveCode=10 and YieldTypeCode=1 and YearsToMaturity=10 and EndDate<='{date}' order by EndDate desc"
    df = pd.read_sql(sql, conn)
    rf = np.nan if df.empty else (1 + float(df.iloc[0, 0])) ** (1 / 242) - 1
    start = max(0, dt - 241)
    ret = _mat(root, dates, ticks, 'd_essentials/pct.bin', 'd_field/pct.bin', start=start, end=dt + 1)
    mv = _mat(root, dates, ticks, 'd_essentials/total_mv.bin', 'd_field/mv.bin', 'fundamental/mcap.bin', start=start, end=dt + 1)
    total = np.nansum(mv, 1, keepdims=True)
    rm = np.nansum(np.divide(mv, total, out=np.zeros_like(mv), where=total != 0) * ret, 1) - rf
    excess = ret - rf
    beta = _beta(excess, rm) if len(ret) == 242 else np.full(n, np.nan)
    if len(ret) == 242:
        lr = np.log1p(excess)
        lr[~np.isfinite(lr)] = 0
        cs = np.cumsum(lr, 0)
        residual = 0.74 * _std(excess, _w(242, 41)) + 0.16 * (cs.max(0) - cs.min(0)) + 0.1 * _std(excess - beta[None, :] * rm[:, None], _w(242, 62))
    else:
        residual = np.full(n, np.nan)
    mend = dt - 20
    mstart = mend - 484
    momentum = _mean(
        np.log1p(
            _mat(root, dates, ticks, 'd_essentials/pct.bin', 'd_field/pct.bin', start=mstart, end=mend)
        )
        , _w(484, 124)
    ) if mstart >= 0 else np.full(n, np.nan)
    mcap = _row(root, dates, ticks, dt, 'fundamental/mcap.bin', 'd_essentials/total_mv.bin', 'd_field/mv.bin')
    with np.errstate(all='ignore'):
        size = np.log(mcap)
    ok = np.isfinite(size)
    z = np.full(n, np.nan)
    if ok.any() and np.nanstd(size) != 0:
        z[ok] = (size[ok] - np.nanmean(size)) / np.nanstd(size)
    nonlinear = np.full(n, np.nan)
    if ok.sum() >= 2:
        X = np.column_stack((np.ones(ok.sum()), z[ok]))
        nonlinear[ok] = z[ok] ** 3 - X @ np.linalg.lstsq(X, z[ok] ** 3, rcond=None)[0]
    book = _row(root, dates, ticks, dt, 'fundamental/bookvalue.bin')
    btop = np.divide(book, mcap, out=np.full(n, np.nan), where=mcap != 0)
    turnover = _mat(root, dates, ticks, 'd_essentials/turnover.bin', 'd_field/turnover.bin', start=max(0, dt - 31), end=dt + 1)
    if len(turnover) == 32:
        with np.errstate(all='ignore'):
            stom = np.array([np.log(np.sum(turnover[i:i + 21], 0)) for i in range(12)])
            liquidity = 0.35 * stom[-1] + 0.35 * np.log(stom[-3:].mean(0)) + 0.3 * np.log(stom.mean(0))
    else:
        liquidity = np.full(n, np.nan)
    pref = _row(root, dates, ticks, dt, 'fundamental/e_preferstock_bookvalue.bin')
    debt = _row(root, dates, ticks, dt, 'fundamental/long_liability.bin')
    liability = _row(root, dates, ticks, dt, 'fundamental/total_liability.bin')
    net = _row(root, dates, ticks, dt, 'fundamental/net_asset.bin')
    asset = _row(root, dates, ticks, dt, 'fundamental/total_asset.bin', 'fundamental/bookvalue.bin')
    with np.errstate(all='ignore'):
        leverage = 0.38 * (mcap + pref + debt) / mcap + 0.35 * liability / asset + 0.27 * (net + debt) / (net - pref)
    gs = max(0, dt - 1209)
    eps = _mat(root, dates, ticks, 'fundamental/eps.bin', start=gs, end=dt + 1)
    revenue = _mat(root, dates, ticks, 'fundamental/operating_revenue.bin', start=gs, end=dt + 1)
    egro = _growth(eps) if len(eps) == 1210 else np.full(n, np.nan)
    sgro = _growth(revenue) if len(revenue) == 1210 else np.full(n, np.nan)
    egrlf = _row(root, dates, ticks, dt, 'con_forecast/con_npcgrate_2y_roll.bin')
    con_eps = _row(root, dates, ticks, dt, 'con_forecast/con_eps_ttm.bin')
    eps_ttm = _row(root, dates, ticks, dt, 'fundamental/eps_ttm.bin')
    growth = 0.18 * egrlf + 0.11 * (con_eps / (eps_ttm + 1e-08) - 1) + 0.24 * egro + 0.47 * sgro
    cash = _mat(root, dates, ticks, 'fundamental/cashdiv.bin', start=max(0, dt - 241), end=dt + 1)
    close = _row(root, dates, ticks, dt, 'd_essentials/close.bin', 'd_field/close.bin')
    profit = _row(root, dates, ticks, dt, 'fundamental/netprofit_ttm.bin')
    con_np = _row(root, dates, ticks, dt, 'con_forecast/con_np_ttm.bin')
    ce = np.nansum(cash, 0) / close if len(cash) == 242 else np.full(n, np.nan)
    earnings = 0.68 * con_np / mcap + 0.21 * ce + 0.11 * profit / mcap
    values = dict(beta=beta, btop=btop, size=size, nonlinear_size=nonlinear, momentom=momentum, residual_vol=residual, liquidity=liquidity, leverage=leverage, growth1=growth, earnings_yield=earnings)
    _save(root, dates, ticks, dt, values)
    return values
__all__ = ['BARRA_NAMES', 'update_barra']
