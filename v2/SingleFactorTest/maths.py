"""作者：xujiayi | 创建日期：2026-11-11
单因子检验的纯数学函数；输入收益率均为小数，矩阵为日期×股票。
"""
from __future__ import annotations

import numpy as np
import bottleneck as bn
from scipy import stats


def normalize(values):
    """非负行归一化，空行保留零仓位。
    Params: values: ndarray (T, N)，非负原始权重。
    Return: ndarray (T, N)，每行权重和为1或0。
    """
    x = np.asarray(values, dtype=np.float64)
    x = np.where(np.isfinite(x), np.maximum(x, 0), 0)
    total = np.sum(x, axis=1, keepdims=True)
    return np.divide(
        x, total,
        out=np.zeros_like(x),
        where=total > 0
    )


def standardize(values):
    """截面等权Z-score。
    Params: values: ndarray (T, N)。
    Return: ndarray (T, N)，常数截面为NaN。
    """
    x = np.asarray(values, dtype=np.float64)
    mean = bn.nanmean(x, axis=1)[:, None]
    std = bn.nanstd(x, axis=1)[:, None]
    return np.divide(
        x - mean, std,
        out=np.full_like(x, np.nan),
        where=std > 0
    )


def transform(values, method="asinh"):
    """截面稳健压缩，参数全部仅使用当日截面。
    Params:
        values: ndarray (T, N)。
        method: asinh/softsign/piecewise/rank_gaussian/yeo_johnson/none。
    Return:
        ndarray (T, N)，保留缺失值；常数截面为0。
    """
    x = np.asarray(values, dtype=np.float64)

    if method == "none":
        return x.copy()

    if method == "rank_gaussian":
        ranks = stats.rankdata(
            x, axis=1, method="average", nan_policy="omit"
        )
        count = np.sum(np.isfinite(x), axis=1, keepdims=True)
        p = (ranks - 0.5) / np.maximum(count, 1)
        return stats.norm.ppf(p)

    if method == "yeo_johnson":
        out = np.full_like(x, np.nan)
        for i, row in enumerate(x):
            valid = np.isfinite(row)
            v = row[valid]
            if v.size:
                out[i, valid] = (
                    0 if np.all(v == v[0])
                    else stats.yeojohnson(v)[0]
                )
        return out

    if method not in ("asinh", "softsign", "piecewise"):
        raise ValueError(f"未知变换方法: {method}")

    center = bn.nanmedian(x, axis=1)[:, None]
    dev = np.abs(x - center)
    scale = bn.nanmedian(dev, axis=1)[:, None] * 1.4826

    # MAD为0时退化到标准差，最终避免除零
    std = bn.nanstd(x, axis=1, ddof=1)[:, None]
    scale = np.where(scale > 0, scale, std)
    scale = np.where(scale > 0, scale, 1)

    z = (x - center) / scale

    if method == "asinh":
        return np.arcsinh(z)

    if method == "softsign":
        return z / (1 + np.abs(z))

    if method == "piecewise":
        return (
            np.clip(z, -3, 3)
            + np.sign(z) * np.maximum(np.abs(z) - 3, 0) * 0.1
        )


def calc_indmv_neutral_longshort(ind_signal, temp_mv):
    """计算行业内市值中性化残差"""
    ix = ~(np.isnan(ind_signal) | np.isinf(ind_signal) | np.isnan(temp_mv) | np.isinf(temp_mv))
    ind_signal[~ix] = np.nan
    temp_mv[~ix] = np.nan

    mv_mean = bn.nanmean(temp_mv, axis=1)
    signal_mean = bn.nanmean(ind_signal, axis=1)
    m = (mv_mean * signal_mean - bn.nanmean(temp_mv * ind_signal, axis=1)) / (mv_mean**2 - bn.nanmean(temp_mv**2, axis=1) + 1e-6)
    b = signal_mean - m * mv_mean
    residual = (ind_signal.T - (temp_mv.T * m) - b).T
    ind_signal = (residual.T - bn.nanmean(residual, axis=1)) / (bn.nanstd(residual, axis=1) + 1e-6)
    return ind_signal.T


def indmv_neutral_longshort(alpha_vec, ind_arr, mv_arr):
    """逐行业执行市值中性化"""
    new_signal = np.full_like(alpha_vec, np.nan)
    ln_mv = np.log(mv_arr)
    for i in range(31):
        ind_ix = ind_arr == i
        ind_select = ind_ix.any(axis=0)
        ind_ix_select = ind_ix[:, ind_select]
        ind_signal = alpha_vec[:, ind_select].copy()
        ind_signal[~ind_ix_select] = np.nan
        temp_mv = ln_mv[:, ind_select].copy()
        new_signal[ind_ix] = calc_indmv_neutral_longshort(ind_signal, temp_mv)[ind_ix_select]
    return new_signal


def neutralize(values, industry=None, market_cap=None):
    """执行行业与市值中性化。
    Params:
        values: ndarray (T, N)，原始因子值。
        industry: ndarray (T, N)，行业编码；None表示不进行行业中性化。
        market_cap: ndarray (T, N)，市值；None表示不进行市值中性化。
    Return:
        ndarray (T, N)，中性化后的因子值。
    """
    alpha = np.asarray(values, dtype=float).copy()
    industry_values = (
        np.zeros(alpha.shape, dtype=int)
        if industry is None else np.asarray(industry)
    )
    market_values = (
        np.ones(alpha.shape, dtype=float)
        if market_cap is None else np.asarray(market_cap, dtype=float)
    )
    return indmv_neutral_longshort(
        alpha, industry_values, market_values
    )


def cross_ic(factor, labels, rank=False, min_obs=3):
    """计算逐日截面Pearson IC或Spearman RankIC。
    Params:
        factor/labels: ndarray (T, N)，同轴矩阵。
        rank: bool，是否计算Spearman RankIC。
        min_obs: int，最小有效样本数。
    Return:
        ndarray (T,)，不足样本或零方差为NaN。
    """
    x = np.asarray(factor, dtype=float)
    y = np.asarray(labels, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = np.where(valid, x, np.nan)
    y = np.where(valid, y, np.nan)

    if rank:
        x = bn.nanrankdata(x, axis=1)
        y = bn.nanrankdata(y, axis=1)

    dx = x - bn.nanmean(x, axis=1)[:, None]
    dy = y - bn.nanmean(y, axis=1)[:, None]

    cov = bn.nansum(dx * dy, axis=1)
    sx = np.sqrt(bn.nansum(dx ** 2, axis=1))
    sy = np.sqrt(bn.nansum(dy ** 2, axis=1))

    denominator = sx * sy
    return np.divide(
        cov, denominator,
        out=np.full(x.shape[0], np.nan),
        where=(denominator > 0) & (valid.sum(axis=1) >= min_obs)
    )


def forward_returns(returns, horizon=1, start=2):
    """向量化计算信号日之后指定区间的复合收益。
    Params:
        returns: ndarray (T, N)，日收益率矩阵。
        horizon: int，未来收益计算窗口。
        start: int，相对信号日的起始偏移。
    Return:
        ndarray (T, N)，不足窗口的位置为NaN。
    """
    gross = 1.0 + np.asarray(returns, dtype=float)
    t_count, n_count = gross.shape
    result = np.full((t_count, n_count), np.nan)
    valid_count = t_count - start - horizon + 1
    if valid_count > 0:
        windows = np.lib.stride_tricks.sliding_window_view(gross, horizon, axis=0)
        result[:valid_count] = (np.prod(windows[start:start + valid_count], axis=-1) - 1)
    return result


def weighted_returns(weights, returns):
    """加权收益，不把持仓股票的缺失收益默认为零。
    Params:
        weights/returns: ndarray (T, N)，同轴权重与收益。
    Return:
        ndarray (T,)，每日组合收益。
    """
    w = np.asarray(weights, dtype=float)
    r = np.asarray(returns, dtype=float)
    missing = ((w != 0) & np.isnan(r)).any(axis=1)
    result = np.sum(np.where(np.isnan(r), 0, w * r), axis=1)
    result[missing] = np.nan
    return result


def nav(returns):
    """净值；缺失收益会使后续净值继续缺失。
    Params:
        returns: ndarray (T,) 或 (T, N)。
    Return:
        ndarray，同型净值。
    """
    return np.cumprod(1.0 + np.asarray(returns, dtype=float), axis=0)


def performance(returns, turnover=None, annual_days=252):
    """复合年化、样本波动、零无风险利率夏普和含初始净值1的回撤。
    Params:
        returns: ndarray
        turnover: 可选双边交易额/2/NAV
        annual_days: 年交易日
    Return:
        dict，turnover为日均单边换手，Calmar使用回撤绝对值。
    """
    r = np.asarray(returns, dtype=float)
    valid = np.isfinite(r)
    r = r[valid]

    if r.size == 0:
        return dict.fromkeys(
            ["annual_return", "annual_vol", "sharpe", "max_drawdown", "calmar", "turnover"],
            np.nan
        )

    n = nav(r)
    dd = n / np.maximum.accumulate(np.maximum(n, 1.0)) - 1
    mdd = dd.min()
    ann = n[-1] ** (annual_days / len(r)) - 1 if n[-1] > 0 else np.nan
    vol = np.std(r, ddof=1) * np.sqrt(annual_days)
    
    return {
        "annual_return": ann,
        "annual_vol": vol,
        "sharpe": r.mean() * annual_days / vol if vol > 0 else np.nan,
        "max_drawdown": mdd,
        "calmar": ann / abs(mdd) if mdd < 0 else np.nan,
        "turnover": np.nan if turnover is None else np.nanmean(np.asarray(turnover, dtype=float)[valid])
    }

def periods(index, years=5):
    """报告年份；以数据末日为报告日，包含当前年及前years个自然年，Overall为全部已载入样本。
    Params: 日期索引/回看完整年份数；
    Return: {标签: 布尔掩码}。
    """
    end = index.max().year
    return {**{str(y): index.year == y for y in range(end - years, end + 1)},
            "Overall": np.ones(len(index), dtype=bool)}


def hac_mean_test(values, lags=0):
    """Newey-West均值检验，处理重叠标签。
    Params:
        values: ndarray (T,)，待检验序列。
        lags: HAC滞后阶数。
    Return:
        tuple (t值, 双侧渐近p值)，样本不足为NaN。
    """
    x = np.asarray(values, dtype=float)
    x = np.where(np.isfinite(x), x, np.nan)

    count = np.isfinite(x).sum()
    if count < 3:
        return np.nan, np.nan

    mean = bn.nanmean(x)
    z = np.where(np.isfinite(x), x - mean, 0.0)

    variance = z @ z
    for k in range(1, min(lags + 1, len(z))):
        variance += 2 * (1 - k / (lags + 1)) * (z[k:] @ z[:-k])

    t = mean / np.sqrt(variance / count**2) if variance > 0 else np.nan
    return t, 2 * stats.norm.sf(abs(t))
