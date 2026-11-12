"""作者：xujiayi | 创建日期：2026-11-11
单因子检验的纯数学函数；输入收益率均为小数，矩阵为日期×股票。
"""
from __future__ import annotations

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


def neutralize(values, industry=None, market_cap=None):
    """逐日OLS残差化，含截距、行业哑变量及log流通市值。

    Params: values/industry/market_cap: 同轴DataFrame，后两项可为None。
    Return: DataFrame，缺少暴露或样本不足的位置为NaN。
    """
    out = values.copy() * np.nan
    for i in range(len(values)):
        y = values.iloc[i].to_numpy(float)
        valid = np.isfinite(y)
        if industry is not None:
            ind = industry.iloc[i]
            valid &= ind.notna().to_numpy()
        if market_cap is not None:
            mv = market_cap.iloc[i].to_numpy(float)
            valid &= np.isfinite(mv) & (mv > 0)
        if not valid.any():
            continue
        # Frisch–Waugh–Lovell：先去行业均值，再回归去均值的log市值。
        # 与含行业哑变量的OLS残差等价，避免每日构造大矩阵并重复SVD。
        group = pd.factorize(ind[valid], sort=False)[0] if industry is not None else np.zeros(valid.sum(), dtype=int)
        count = np.bincount(group)
        residual = y[valid] - (np.bincount(group, weights=y[valid]) / count)[group]
        rank = len(count)
        if market_cap is not None:
            x = np.log(mv[valid])
            x -= (np.bincount(group, weights=x) / count)[group]
            variance = x @ x
            if variance > 1e-20:
                residual -= x * ((x @ residual) / variance)
                rank += 1
        if valid.sum() > rank:
            out.iloc[i, np.flatnonzero(valid)] = residual
    return out


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
    """计算t+start至t+start+horizon-1的复合收益，不填补尾端或中途缺失。

    Params: returns: 日收益DataFrame；horizon/start: 正整数交易日。
    Return: DataFrame，索引为信号日。
    """
    if horizon < 1 or start < 1:
        raise ValueError("horizon/start须为正整数")
    gross = 1 + returns
    result = gross.shift(-start)
    for lag in range(start + 1, start + horizon):
        result = result * gross.shift(-lag)
    return result - 1


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
