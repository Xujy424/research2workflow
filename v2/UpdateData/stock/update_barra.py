"""Daily incremental Barra factor maintenance."""
from pathlib import Path
import numpy as np
import pandas as pd
import sys

if __package__:
    from ..config import get_jy_conn
else:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from v2.UpdateData.config import get_jy_conn



BARRA_NAMES = ('beta', 'btop', 'size', 'nonlinear_size', 'momentom', 'residual_vol', 'liquidity', 'leverage', 'growth1', 'earnings_yield')

def _find(root, *names):
    for name in names:
        path = Path(root) / name
        if path.exists():
            return path
    raise FileNotFoundError(f"missing input: {', '.join(names)}")

def _mat(root, dates, ticks, *names, start=0, end=None):
    a = np.memmap(_find(root, *names), dtype=np.float32, mode='r', shape=(len(dates), len(ticks)))
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
                f.truncate(size)   # 扩展到二维 float64 矩阵所需大小
            a = np.memmap(p, dtype=float, mode='r+', shape=(len(dates), len(ticks)))
            a[:] = np.nan
            a.flush()
        if p.stat().st_size != size:
            raise ValueError(f'{p} size does not match axes')
        a = np.memmap(p, dtype=float, mode='r+', shape=(len(dates), len(ticks)))
        a[dt] = value
        a.flush()


def _calc_beta(ret, mv, rf, n):
    if len(ret) != 242:
        return np.full(n, np.nan)
    total = np.nansum(mv, 1, keepdims=True)
    rm = np.nansum(np.divide(mv, total, out=np.zeros_like(mv), where=total != 0) * ret, 1) - rf   # 市值加权个股收益率得到市场收益率
    return _beta(ret - rf, rm)


def _calc_btop(root, dates, ticks, dt, mcap):
    book = _row(root, dates, ticks, dt, 'fundamental/balance/TotalAssets.bin')   # bookvalue
    return np.divide(book, mcap, out=np.full(len(ticks), np.nan), where=mcap != 0)


def _calc_size(mcap):
    with np.errstate(all='ignore'):
        return np.log(mcap)


def _calc_nonlinear_size(size):
    ok = np.isfinite(size)
    result = np.full(len(size), np.nan)
    if not ok.any() or np.nanstd(size) == 0:
        return result
    z = np.full(len(size), np.nan)
    z[ok] = (size[ok] - np.nanmean(size)) / np.nanstd(size)
    if ok.sum() >= 2:
        design = np.column_stack((np.ones(ok.sum()), z[ok]))
        result[ok] = z[ok] ** 3 - design @ np.linalg.lstsq(design, z[ok] ** 3, rcond=None)[0]
    return result


def _calc_momentom(root, dates, ticks, dt):
    ''' 过去 525 个交易日（剔除最近 21 天）的加权收益率，半衰期 126 天 '''
    end = dt - 20
    start = end - 484
    if start < 0:
        return np.full(len(ticks), np.nan)
    ret = _mat(root, dates, ticks, 'd_essentials/pct.bin', start=start, end=end)
    return _mean(np.log1p(ret), _w(484, 124))


def _calc_residual_vol(ret, mv, rf, beta, n):
    '''由 DASTD（超额收益波动率）、CMRA（12 个月收益率区间）、HSIGMA（Beta 残差波动）加权合成，并对 BETA 正交化'''
    if len(ret) != 242:
        return np.full(n, np.nan)

    total = np.nansum(mv, 1, keepdims=True)
    rm = np.nansum(np.divide(mv, total, out=np.zeros_like(mv), where=total != 0) * ret, 1) - rf
    excess = ret - rf

    # DASTD: 过去 242 个交易日个股超额收益率的指数加权标准差，半衰期 41 天。
    dastd = _std(excess, _w(242, 41))

    # CMRA: 过去 242 个交易日累计对数超额收益的最大值与最小值之差。
    lr = np.log1p(excess)
    lr[~np.isfinite(lr)] = 0
    cumulative = np.cumsum(lr, 0)
    cmra = cumulative.max(0) - cumulative.min(0)

    # HSIGMA: 市场模型残差的指数加权标准差，半衰期 62 天。
    residual = excess - beta[None, :] * rm[:, None]
    hsigma = _std(residual, _w(242, 62))

    return 0.74 * dastd + 0.16 * cmra + 0.10 * hsigma


def _calc_liquidity(root, dates, ticks, dt):
    turnover = _mat(root, dates, ticks, 'd_essentials/turnover.bin', start=max(0, dt - 31), end=dt + 1)
    if len(turnover) != 32:
        return np.full(len(ticks), np.nan)
    with np.errstate(all='ignore'):
        stom = np.array([np.log(np.sum(turnover[i:i + 21], 0)) for i in range(12)])
        return 0.35 * stom[-1] + 0.35 * np.log(stom[-3:].mean(0)) + 0.3 * np.log(stom.mean(0))


def _calc_leverage(root, dates, ticks, dt, mcap):
    pref = _row(root, dates, ticks, dt, 'fundamental/balance/EPreferStock.bin')
    debt = _row(root, dates, ticks, dt, 'fundamental/balance/TotalNonCurrentLiability.bin')
    liability = _row(root, dates, ticks, dt, 'fundamental/balance/TotalLiability.bin')
    net = _row(root, dates, ticks, dt, 'fundamental/balance/SEWithoutMI.bin')
    asset = _row(root, dates, ticks, dt, 'fundamental/balance/TotalAssets.bin')
    with np.errstate(all='ignore'):
        return 0.38 * (mcap + pref + debt) / mcap + 0.35 * liability / asset + 0.27 * (net + debt) / (net - pref)


def _calc_growth1(root, dates, ticks, dt):
    '''由过去五年销售增长率、每股盈利增长率、分析师预测的长期和短期盈利增长率加权合成'''
    # 定义：0.18 × EGRLF + 0.11 × EGRSF + 0.24 × EGRO + 0.47 × SGRO。
    # EGRLF：分析师预测长期（3-5年）EPS增长率。
    # EGRSF：分析师预测短期（1年）EPS增长率。
    # EGRO：过去5年EPS增长率，回归斜率 / 平均EPS。
    # SGRO：过去5年营业收入增长率，回归斜率 / 平均营收。
    # 需基本面数据 (年度频率):
    # egro: slope(eps_5y) / mean(eps_5y)
    # sgro: slope(revenue_5y) / mean(revenue_5y)
    # growth = 0.18 * egrlf + 0.11 * egrsf + 0.24 * egro + 0.47 * sgro
    n = len(ticks)
    start = max(0, dt - 1209)
    eps = _mat(root, dates, ticks, 'fundamental/income_ttm/BasicEPS.bin', start=start, end=dt + 1)
    revenue = _mat(root, dates, ticks, 'fundamental/income_ttm/OperatingRevenue.bin', start=start, end=dt + 1)
    egro = _growth(eps) if len(eps) == 1210 else np.full(n, np.nan)
    sgro = _growth(revenue) if len(revenue) == 1210 else np.full(n, np.nan)
    egrlf = _row(root, dates, ticks, dt, 'zyyx/con_forecast_roll/con_npcgrate_2y_roll.bin')
    con_eps = _row(root, dates, ticks, dt, 'zyyx/con_forecast_roll/con_eps_roll.bin')
    eps_ttm = _row(root, dates, ticks, dt, 'fundamental/income_ttm/BasicEPS.bin')
    return 0.18 * egrlf + 0.11 * (con_eps / (eps_ttm + 1e-08) - 1) + 0.24 * egro + 0.47 * sgro


def _calc_earnings_yield(root, dates, ticks, dt, mcap):
    n = len(ticks)
    cash = _mat(root, dates, ticks, 'fundamental/dividend/cash_dividend_ttm_adjusted.bin', start=max(0, dt - 241), end=dt + 1)
    close = _row(root, dates, ticks, dt, 'd_essentials/close_adj.bin')
    profit = _row(root, dates, ticks, dt, 'fundamental/income_ttm/NetProfit.bin')
    con_np = _row(root, dates, ticks, dt, 'zyyx/con_forecast_roll/con_np_roll.bin')
    ce = np.nansum(cash, 0) / close if len(cash) == 242 else np.full(n, np.nan)
    return 0.68 * con_np / mcap + 0.21 * ce + 0.11 * profit / mcap


def update_barra(date, dates, ticks, conn, root):
    """Compute all ten factors for date and overwrite only its matrix row."""
    target = np.datetime64(date)
    dt = int(np.searchsorted(dates, target))
    if dt >= len(dates) or dates[dt] != target:
        raise ValueError(f'{date} is not in dates')

    sql = f"""
    select 
        top 1 Yield 
    from Bond_CBYieldCurve 
    where CurveCode=10 
        and YieldTypeCode=1 
        and YearsToMaturity=10 
        and EndDate<='{date}' order by EndDate desc
    """
    df = pd.read_sql(sql, conn)
    rf = np.nan if df.empty else (1 + float(df.iloc[0, 0])) ** (1 / 242) - 1
    start = max(0, dt - 241)
    ret = _mat(root, dates, ticks, 'd_essentials/pct.bin', start=start, end=dt + 1)
    mv = _mat(root, dates, ticks, 'd_essentials/circ_mv.bin', start=start, end=dt + 1)
    mcap = _row(root, dates, ticks, dt, 'd_essentials/total_mv.bin')

    beta = _calc_beta(ret, mv, rf, len(ticks))
    size = _calc_size(mcap)
    values = {
        'beta': beta,
        'btop': _calc_btop(root, dates, ticks, dt, mcap),
        'size': size,
        'nonlinear_size': _calc_nonlinear_size(size),
        'momentom': _calc_momentom(root, dates, ticks, dt),
        'residual_vol': _calc_residual_vol(ret, mv, rf, beta, len(ticks)),
        'liquidity': _calc_liquidity(root, dates, ticks, dt),
        'leverage': _calc_leverage(root, dates, ticks, dt, mcap),
        'growth1': _calc_growth1(root, dates, ticks, dt),
        'earnings_yield': _calc_earnings_yield(root, dates, ticks, dt, mcap),
    }
    _save(root, dates, ticks, dt, values)
    return values


__all__ = ['BARRA_NAMES', 'update_barra']




if __name__ == '__main__':

    date = '2024-06-14'
    dates = np.load('D:/data/axis/dates.npy',allow_pickle=True)
    ticks = np.load('D:/data/axis/stock_ticks.npy', allow_pickle=False)
    conn = get_jy_conn()
    root = Path('D:/data')

    update_barra(
        date, dates, ticks, conn, root
    )