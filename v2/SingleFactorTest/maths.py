"""作者：xujiayi | 创建日期：2026-11-11
单因子检验的纯数学函数；输入收益率均为小数，矩阵为日期×股票。
"""
from __future__ import annotations

import bottleneck as bn
import numpy as np
import pandas as pd
from scipy import stats


def normalize(values):
    """非负行归一化，空行保留零仓位。

    Params: values: DataFrame，非负原始权重。
    Return: DataFrame，每行权重和为1或0。
    """
    values = values.replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)
    return values.div(values.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)


def standardize(values):
    """截面等权Z-score；Params: values: DataFrame；Return: DataFrame。"""
    return values.sub(values.mean(axis=1), axis=0).div(
        values.std(axis=1, ddof=0).replace(0, np.nan), axis=0)


def transform(values, method="asinh"):
    """截面稳健压缩，参数全部仅使用当日截面。

    Params: values: DataFrame；method: asinh/softsign/piecewise/rank_gaussian/yeo_johnson/none。
    Return: DataFrame，保留缺失值；常数截面为0。
    """
    if method == "none":
        return values.copy()
    center = values.median(axis=1)
    scale = values.sub(center, axis=0).abs().median(axis=1) * 1.4826
    scale = scale.where(scale > 0, values.std(axis=1)).replace(0, 1)
    z = values.sub(center, axis=0).div(scale, axis=0)
    if method == "asinh":
        return np.arcsinh(z)
    if method == "softsign":
        return z / (1 + z.abs())
    if method == "piecewise":
        return z.clip(-3, 3) + np.sign(z) * (z.abs() - 3).clip(lower=0) * .1
    if method == "rank_gaussian":
        p = values.rank(axis=1, method="average").sub(.5).div(values.count(axis=1), axis=0)
        return pd.DataFrame(stats.norm.ppf(p), values.index, values.columns)
    if method == "yeo_johnson":
        out = values.copy() * np.nan
        for date, row in values.iterrows():
            valid = row.dropna()
            if len(valid):
                out.loc[date, valid.index] = (0 if valid.nunique() < 2 else stats.yeojohnson(valid)[0])
        return out
    raise ValueError(f"未知变换方法: {method}")


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
    """执行行业与市值中性化"""
    alpha = values.to_numpy(dtype=float, copy=True)
    industry_values = np.zeros(alpha.shape, dtype=int) if industry is None else industry.to_numpy(copy=False)
    market_values = np.ones(alpha.shape, dtype=float) if market_cap is None else market_cap.to_numpy(dtype=float, copy=False)
    result = indmv_neutral_longshort(alpha, industry_values, market_values)
    return pd.DataFrame(result, index=values.index, columns=values.columns)


def cross_ic(factor, labels, rank=False, min_obs=3):
    """成对有效样本上的逐日相关系数。

    Params: factor/labels: 同轴DataFrame；rank: Spearman；min_obs: 最小截面样本。
    Return: Series，不足样本或零方差为NaN。
    """
    valid = factor.notna() & labels.notna()
    x, y = factor.where(valid), labels.where(valid)
    if rank:
        x, y = x.rank(axis=1), y.rank(axis=1)
    return x.corrwith(y, axis=1).where(valid.sum(axis=1) >= min_obs)


def forward_returns(returns, horizon=1, start=1):
    """向量化计算信号日之后指定区间的复合收益"""
    gross = 1.0 + returns.to_numpy(dtype=float, copy=False)
    t_count, n_count = gross.shape
    result = np.full((t_count, n_count), np.nan)
    valid_count = t_count - start - horizon + 1
    if valid_count > 0:
        windows = np.lib.stride_tricks.sliding_window_view(gross, horizon, axis=0)
        result[:valid_count] = np.prod(windows[start:start + valid_count], axis=-1) - 1.0
    return pd.DataFrame(result, index=returns.index, columns=returns.columns)


def weighted_returns(weights, returns):
    """加权收益；Params: 同轴权重/收益DataFrame；Return: Series。

    活跃持仓收益缺失则当日收益为NaN，不把缺失收益默认为零。
    """
    missing = ((weights != 0) & returns.isna()).any(axis=1)
    return (weights * returns).sum(axis=1).mask(missing)


def nav(returns):
    """净值；Params: 收益Series/DataFrame；Return: 同型净值，缺失日保留空值。"""
    return (1 + returns).cumprod()


def performance(returns, turnover=None, annual_days=252):
    """复合年化、样本波动、零无风险利率夏普和含初始净值1的回撤。

    Params: returns: Series；turnover: 可选双边交易额/2/NAV；annual_days: 年交易日。
    Return: Series，turnover为日均单边换手，Calmar使用回撤绝对值。
    """
    r = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return pd.Series(dict.fromkeys(["annual_return", "annual_vol", "sharpe", "max_drawdown", "calmar", "turnover"], np.nan))
    n = nav(r)
    dd = n / n.cummax().clip(lower=1) - 1
    ann = n.iloc[-1] ** (annual_days / len(r)) - 1 if n.iloc[-1] > 0 else np.nan
    vol = r.std(ddof=1) * np.sqrt(annual_days)
    mdd = dd.min()
    return pd.Series({"annual_return": ann, "annual_vol": vol,
                      "sharpe": r.mean() * annual_days / vol if vol > 0 else np.nan,
                      "max_drawdown": mdd, "calmar": ann / abs(mdd) if mdd < 0 else np.nan,
                      "turnover": np.nan if turnover is None else turnover.reindex(r.index).mean()})


def periods(index, years=5):
    """报告年份；Params: 日期索引/回看完整年份数；Return: {标签: 布尔掩码}。

    以数据末日为报告日，包含当前年及前years个自然年，Overall为全部已载入样本。
    """
    end = index.max().year
    return {**{str(y): index.year == y for y in range(end - years, end + 1)},
            "Overall": np.ones(len(index), dtype=bool)}


def hac_mean_test(values, lags=0):
    """Newey-West均值检验，处理重叠标签。

    Params: values: Series；lags: HAC滞后阶数。
    Return: (t值, 双侧渐近p值)，样本不足为NaN。
    """
    series = pd.Series(values).replace([np.inf, -np.inf], np.nan)
    count = series.count()
    if count < 3:
        return np.nan, np.nan
    # 未触发日的score为0，保留交易日间距，避免压缩稀疏事件造成错误HAC滞后。
    z = (series - series.mean()).fillna(0).to_numpy(float)
    variance = z @ z
    for k in range(1, min(lags + 1, len(z))):
        variance += 2 * (1 - k / (lags + 1)) * (z[k:] @ z[:-k])
    t = series.mean() / np.sqrt(variance / count ** 2) if variance > 0 else np.nan
    return t, 2 * stats.norm.sf(abs(t))
