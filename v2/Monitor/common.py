"""v3 指标共用的日期、域成员和横截面聚合工具。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import bottleneck as bn
from math import gamma
from statistics import NormalDist

from .config import STYLE_MASKS, INDEX_MAP, INDUSTRY_MAP


def minute_log_returns(prices, times=None):
    """价格矩阵转对数收益，同时标记连续一分钟区间，不填补缺失价格。"""
    prices = np.asarray(prices, dtype=float)
    ret = np.diff(np.log(np.where(np.isfinite(prices) & (prices > 0), prices, np.nan)), axis=0)
    consecutive = (np.ones(len(ret), dtype=bool) if times is None else
                   np.diff(pd.to_timedelta(times).asi8) == 60_000_000_000)
    return ret, consecutive


def event_acd_intensity(events, valid, clock, *, alpha=0.1, beta=0.8):
    """按事件间隔递推固定参数ACD强度；无事件为0，单事件或无效列为NaN。"""
    events = np.asarray(events, dtype=bool)
    minutes, stocks = events.shape
    result = np.full(stocks, np.nan)
    count = events.sum(axis=0)
    result[valid & (count == 0)] = 0
    estimable = valid & (count >= 2)
    if not estimable.any():
        return result
    first = events.argmax(axis=0)
    last = minutes - 1 - events[::-1].argmax(axis=0)
    mean_wait = np.divide(clock[last] - clock[first], count - 1,
        out=np.zeros(stocks), where=estimable)
    omega = (1 - alpha - beta) * mean_wait
    psi = np.maximum(mean_wait, 1)
    previous = clock[first].copy()
    for t in np.flatnonzero(events[:, estimable].any(axis=1)):
        active = events[t] & estimable & (t > first)
        psi[active] = (omega[active] + alpha * (clock[t] - previous[active])
                       + beta * psi[active])
        previous[active] = clock[t]
    result[estimable] = 1 / psi[estimable]
    return result


def bns_jump_variation(returns, consecutive=None, *, alpha=0.01, min_returns=60):
    """批量BNS检验，返回逐股(RV、显著跳跃变差、Z)，无效股票为NaN。
    returns为[时点, 股票]的等间隔对数收益；consecutive标记可用时间区间。
    BV/TQ乘积不跨时间断点，个股在可用区间须完整；有限样本按乘积数修正。
    """
    if not 0 < alpha < 1 or min_returns < 3:
        raise ValueError('alpha须在(0, 1)内，min_returns至少为3')
    ret = np.asarray(returns, dtype=float)
    if ret.ndim != 2:
        raise ValueError('returns须为[时点, 股票]矩阵')
    mask = np.ones(len(ret), dtype=bool) if consecutive is None else np.asarray(consecutive, bool)
    if mask.shape != (len(ret),):
        raise ValueError('consecutive须与收益时点数一致')
    empty = np.full(ret.shape[1], np.nan)
    pairs = mask[1:] & mask[:-1]
    triples = pairs[1:] & mask[:-2]
    n, n2, n3 = mask.sum(), pairs.sum(), triples.sum()
    if n < min_returns or n2 == 0 or n3 == 0:
        return empty.copy(), empty.copy(), empty.copy()
    complete = np.isfinite(ret[mask]).all(axis=0)
    r = np.abs(np.where(np.isfinite(ret), ret, 0))
    rv = (r[mask] ** 2).sum(axis=0)
    bv = (np.pi / 2) * (n / n2) * (r[1:] * r[:-1])[pairs].sum(axis=0)
    power = r ** (4 / 3)
    mu43 = 2 ** (2 / 3) * gamma(7 / 6) / np.sqrt(np.pi)
    tq = n * (n / n3) / mu43 ** 3 * (
        power[2:] * power[1:-1] * power[:-2])[triples].sum(axis=0)
    valid = complete & (rv > 0) & (bv > 0)
    ratio = np.divide(bv, rv, out=empty.copy(), where=valid)
    scale = np.divide(tq, bv ** 2, out=empty.copy(), where=valid)
    constant = (np.pi / 2) ** 2 + np.pi - 5
    z = (1 - ratio) / np.sqrt(constant / n * np.maximum(1, scale))
    jump = np.where(z > NormalDist().inv_cdf(1 - alpha), np.maximum(rv - bv, 0), 0)
    return np.where(valid, rv, np.nan), np.where(valid, jump, np.nan), z


def significant_jump_events(returns, consecutive=None, *, alpha=0.01, min_returns=60):
    """BNS日度筛选后以ABD方法定位分钟事件，返回(事件矩阵, 逐股有效标记)。
    ABD使用sqrt(BV/n)标准化收益，以双侧Bonferroni阈值校正日内多次检验。
    假定日内连续方差近似均匀，未校正波动季节性；仅适用于事后日度监控。
    """
    ret = np.asarray(returns, dtype=float)
    rv, jump, _ = bns_jump_variation(ret, consecutive, alpha=alpha, min_returns=min_returns)
    mask = np.ones(len(ret), dtype=bool) if consecutive is None else np.asarray(consecutive, bool)
    valid = np.isfinite(rv)
    events = np.zeros(ret.shape, dtype=bool)
    n = mask.sum()
    if n:
        # 对BNS显著日，RV-J恰为BV；其余日由jump>0排除。
        threshold = NormalDist().inv_cdf(1 - alpha / (2 * n))
        events = (mask[:, None] & valid & (jump > 0)
                  & (np.abs(ret) > threshold * np.sqrt((rv - jump) / n)))
    return events, valid


def domain_returns(context, domain, name, end, returns):
    """按域内前一日权重计算日收益；Barra 多空域保留有符号收益差。"""
    dates = context.dates[:end + 1]
    weights = context.get_weight(domain, name, end)
    lagged = np.vstack((np.full_like(weights[:1], np.nan), weights[:-1]))
    valid = np.isfinite(returns) & np.isfinite(lagged)
    weighted = np.sum(np.where(valid, returns * lagged, 0.0), axis=1) / 100
    total_weight = np.sum(np.where(valid, lagged, 0.0), axis=1)
    result = np.divide(weighted, total_weight, out=np.full(len(dates), np.nan),
                        where=total_weight != 0)
    signed = np.any(valid & (lagged < 0), axis=1)
    result[signed] = weighted[signed]
    result[~valid.any(axis=1)] = np.nan
    return pd.Series(result, index=dates)


def rolling_compound_return(values, period):
    """计算滚动 period 日复合收益，窗口不足时返回 NaN。"""
    if period < 1:
        raise ValueError("period必须为正整数")
    values = pd.Series(values, copy=False)
    return (1 + values).rolling(period, min_periods=period).apply(np.prod, raw=True) - 1


def weighted_series(values, weights, dates, name=None, *, signed=False):
    """逐日加权聚合；默认计算单组加权均值，signed模式保留多空权重差。"""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (
        (weights != 0) if signed else (weights > 0)
    )
    numerator = np.sum(np.where(valid, values * weights, 0), axis=1)
    denominator = np.sum(np.where(valid, weights, 0), axis=1)
    result = np.divide(numerator, denominator, out=np.full(len(dates), np.nan),
                       where=denominator != 0)
    if signed:
        signed_rows = np.any(valid & (weights < 0), axis=1)
        result[signed_rows] = numerator[signed_rows]
    result[~valid.any(axis=1)] = np.nan
    return pd.Series(result, index=dates, name=name)


def rolling_zscore(series, window=252, min_periods=60, *, ddof=1, name=None,
                   exclude_current=False):
    """逐列滚动标准化，返回Series或DataFrame；exclude_current排除当期。"""
    if window < 1 or not 1 <= min_periods <= window:
        raise ValueError("window须为正数，且min_periods须介于1和window之间")
    values = (pd.DataFrame(series, copy=False) if np.ndim(series) == 2
              else pd.Series(series, copy=False))
    history = (values.shift(1) if exclude_current else values).rolling(
        window, min_periods=min_periods)
    mean = history.mean()
    std = history.std(ddof=ddof)
    result = values.sub(mean).div(std.where(std > 0))
    return result.rename(name or values.name) if isinstance(result, pd.Series) else result


def rolling_percentile(series, window=252, min_periods=60, *, name=None):
    """计算当前值在含当日的滚动窗口内的经验中秩分位，返回0到1之间的序列。"""
    if window < 1 or not 1 <= min_periods <= window:
        raise ValueError("window须为正数，且min_periods须介于1和window之间")
    values = pd.Series(series, copy=False)

    def percentile(window_values):
        current = window_values[-1]
        valid = window_values[np.isfinite(window_values)]
        if not np.isfinite(current) or len(valid) < min_periods:
            return np.nan
        # 平值按平均名次处理，避免相同观测因排序顺序产生不同分位。
        below = np.count_nonzero(valid < current)
        equal = np.count_nonzero(valid == current)
        return (below + 0.5 * equal) / len(valid)

    result = values.rolling(window, min_periods=1).apply(percentile, raw=True)
    return result.rename(name or values.name)


