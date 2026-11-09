"""风格、估值与相对表现指标；返回日期索引的 Series 或 DataFrame。"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "v2.Monitor"

from .context import MonitorContext
from v2.UpdateData.config import ROOT
from .config import INDUSTRY_MAP, INDEX_MAP, STYLE_MASKS
from .common import (
    domain_returns, rolling_compound_return, weighted_series,
)

RETURNS = "d_essentials/pct"
INDEX_CODES = {
    "hs300": "000300.SH", "zz500": "000905.SH", "zz1000": "000852.SH",
    "zz2000": "932000.CSI", "zzfull": "000985.SH",
}


def index_level(context, date, domain, name):
    """读取指定指数官方收盘点位。"""
    if domain != "index":
        raise ValueError("指数收盘点位要求 domain='index'")
    index_name = INDEX_MAP.get(name, name)
    code = INDEX_CODES.get(index_name)
    if code is None:
        raise ValueError(f"不支持的指数: {name}")
    end = context.dates[context.get_position(date)]
    return context.read("wsd", (code, "close"), context.dates[0], end, kind="series").rename("指数收盘点位")


def relative_return(context, date, domain, name, benchmark="zzfull", period=20):
    """域内滞后成分权重组合 period 日收益减全市场同期收益。"""
    end = context.get_position(date)
    returns = context.read("local", RETURNS, 0, end, kind="matrix")
    target = domain_returns(context, domain, name, end, returns)
    base = domain_returns(context, "index", benchmark, end, returns)
    return (
        rolling_compound_return(target, period) - rolling_compound_return(base, period)
    ).rename("相对收益")


def new_high_low_share(context, date, domain, name, lookback=120):
    """域内过去 lookback 日创新高股票占比减创新低股票占比。60/120/252"""
    end = context.get_position(date)
    dates = context.dates[:end + 1]
    returns = context.read("local", RETURNS, 0, end, kind="matrix") / 100
    wealth = np.exp(np.nancumsum(np.log1p(np.where(returns > -1, returns, np.nan)), axis=0))
    wealth[~np.isfinite(returns)] = np.nan
    frame = pd.DataFrame(wealth)
    prior_high = frame.shift(1).rolling(lookback, min_periods=lookback).max().to_numpy()
    prior_low = frame.shift(1).rolling(lookback, min_periods=lookback).min().to_numpy()
    observed = pd.DataFrame(np.isfinite(returns)).shift(1).rolling(
        lookback, min_periods=lookback
    ).sum().to_numpy()
    weights = context.get_weight(domain, name, end)
    valid = np.isfinite(returns) & np.isfinite(weights) & (weights != 0) & (observed >= lookback)
    highs, lows = valid & (wealth > prior_high), valid & (wealth < prior_low)
    count = valid.sum(axis=1)
    result = np.divide(highs.sum(axis=1) - lows.sum(axis=1), count,
                       out=np.full(len(dates), np.nan), where=count > 0)
    return pd.Series(result, index=dates, name="新高占比减新低占比")


def _harmonic_level(values, weights):
    """按给定权重计算每日调和平均估值。"""
    valid = np.isfinite(values) & (values > 0) & np.isfinite(weights) & (weights != 0)
    numerator = np.sum(np.where(valid, weights, 0), axis=1)
    denominator = np.sum(np.where(valid, weights / np.where(valid, values, 1), 0), axis=1)
    return np.divide(
        numerator, denominator, 
        out=np.full(len(values), np.nan),
        where=denominator != 0
    )

def _valuation_level(context, date, domain, name, field, label):
    end = context.get_position(date)
    dates = context.dates[:end + 1]
    values = context.read("local", field, 0, end, kind="matrix")
    weights = context.get_weight(domain, name, end)
    if np.any(weights < 0):
        high = _harmonic_level(values, np.where(weights > 0, weights, np.nan))
        low = _harmonic_level(values, np.where(weights < 0, -weights, np.nan))
        result = high - low
    else:
        result = _harmonic_level(values, weights)
    return pd.Series(result, index=dates, name=label)


def pe_level(context, date, domain, name):
    """域内市值加权 PE_TTM 调和平均水平。"""
    return _valuation_level(context, date, domain, name,
                            "fundamental/valuation/pe_ttm", "PE_TTM")


def pb_level(context, date, domain, name):
    """域内市值加权 PB_LF 调和平均水平。"""
    return _valuation_level(context, date, domain, name,
                            "fundamental/valuation/pb_lf", "PB_LF")


def earnings_bond_spread(context, date, domain, name, rate_code="S0059749"):
    """域内加权盈利收益率减10年期国债收益率，单位为小数。"""
    end = context.get_position(date)
    dates = context.dates[:end + 1]
    pe = context.read("local", "fundamental/valuation/pe_ttm", 0, end, kind="matrix")
    weights = context.get_weight(domain, name, end)
    bond = context.read("edb", rate_code, dates[0], dates[-1], kind="series").reindex(dates)
    if bond.abs().median() > 1:
        bond = bond / 100
    # 10年期国债收益率不应为零或负数；EDB异常零值按缺失处理后沿用最近有效值。
    bond = bond.where(bond > 0).ffill()
    earnings_yield = np.divide(1, pe, out=np.full_like(pe, np.nan), where=pe > 0)
    if np.any(weights < 0):
        high = weighted_series(earnings_yield, np.where(weights > 0, weights, np.nan), dates)
        low = weighted_series(earnings_yield, np.where(weights < 0, -weights, np.nan), dates)
        result = high - low
    else:
        result = weighted_series(earnings_yield, weights, dates) - bond
    return result.rename("盈利收益率减10Y国债")


def amount(context, date, domain, name):
    """域内股票日成交额总和。"""
    end = context.get_position(date)
    values = context.read("local", "d_essentials/amount", 0, end, kind="matrix")
    weights = context.get_weight(domain, name, end)
    included = np.isfinite(weights) & (weights != 0) & np.isfinite(values)
    result = np.sum(np.where(included, values, 0), axis=1)
    result[~included.any(axis=1)] = np.nan
    return pd.Series(result, index=context.dates[:end + 1], name="成交额")

def amount_share(context, date, domain, name):
    """域内日成交额占全市场成交额比例。"""
    end = context.get_position(date)
    values = context.read("local", "d_essentials/amount", 0, end, kind="matrix")
    total = np.nansum(np.where(np.isfinite(values), values, 0), axis=1)
    selected = amount(context, date, domain, name)
    return selected.div(pd.Series(total, index=selected.index).where(total > 0)).rename("成交额占比")



def _industry_daily_returns(context, end):
    """按前一日行业归属和流通市值，批量计算各行业当日收益。"""
    dates = context.dates[:end + 1]
    names = tuple(INDUSTRY_MAP)
    daily = np.full((len(dates), len(names)), np.nan)
    if len(dates) < 2:
        return pd.DataFrame(daily, index=dates, columns=names)

    returns = context.read("local", RETURNS, 0, end, kind="matrix") / 100
    industry = context.read("local", "industry/industry", 0, end, kind="matrix")
    cap = context.read("local", "d_essentials/circ_mv", 0, end, kind="matrix")
    codes = industry[:-1]
    valid = (np.isfinite(codes) & (codes >= 0) & (codes < len(names))
             & np.isfinite(cap[:-1]) & (cap[:-1] > 0) & np.isfinite(returns[1:]))
    rows, stocks = np.where(valid)
    bins = rows * len(names) + codes[rows, stocks].astype(np.intp)
    weights = cap[:-1][rows, stocks]
    size = (len(dates) - 1) * len(names)
    numerator = np.bincount(bins, weights=returns[1:][rows, stocks] * weights,
                            minlength=size).reshape(-1, len(names))
    denominator = np.bincount(bins, weights=weights, minlength=size).reshape(-1, len(names))
    daily[1:] = np.divide(numerator, denominator, out=np.full_like(denominator, np.nan),
                          where=denominator > 0)
    return pd.DataFrame(daily, index=dates, columns=names)


def industry_relative_return(context, date, domain, name, benchmark="zzfull", period=20):
    """返回各行业相对全市场的 period 日复合收益，列为行业名称。"""
    if period < 1:
        raise ValueError("period必须为正整数")
    end = context.get_position(date)
    dates = context.dates[:end + 1]
    industry_returns = _industry_daily_returns(context, end).to_numpy()
    windows = np.lib.stride_tricks.sliding_window_view(industry_returns, period, axis=0)
    compounded = np.prod(1 + windows, axis=-1) - 1
    values = np.full_like(industry_returns, np.nan)
    values[period - 1:] = compounded
    market_returns = context.read("local", RETURNS, 0, end, kind="matrix")
    benchmark = domain_returns(context, "index", benchmark, end, market_returns)
    benchmark = rolling_compound_return(benchmark, period).to_numpy()
    return pd.DataFrame(values - benchmark[:, None], index=dates,
                        columns=tuple(INDUSTRY_MAP))



def _factor_return_spread(context, date, factor, period, weights=None):
    end = context.get_position(date)
    returns = context.read("local", RETURNS, 0, end, kind="matrix") / 100
    if weights is None:
        mask = context.read("local", STYLE_MASKS[factor], 0, end, kind="matrix")
        cap = context.read("local", "d_essentials/circ_mv", 0, end, kind="matrix")
        weights = np.where(mask > 0, cap, np.where(mask < 0, -cap, 0.0))
    dates = context.dates[:end + 1]
    group_returns = []
    for group_weight in (np.where(weights > 0, weights, np.nan),
                         np.where(weights < 0, -weights, np.nan)):
        lagged = np.full_like(group_weight, np.nan)
        lagged[1:] = group_weight[:-1]
        group_returns.append(weighted_series(returns, lagged, dates))
    return (rolling_compound_return(group_returns[0], period) -
            rolling_compound_return(group_returns[1], period))

def size_spread(context, date, domain, name, period=5):
    """全市场 Barra 规模因子高组减低组的 period 日收益差。"""
    return _factor_return_spread(context, date, "size", period).rename("大小盘收益差")

def growth_value_spread(context, date, domain, name, period=5):
    """全市场成长价值因子正组减负组的 period 日收益差。"""
    return _factor_return_spread(context, date, "growth_value", period).rename("成长价值收益差")

def beta_spread(context, date, domain, name, period=5):
    """全市场 Barra Beta 高组减低组的 period 日收益差。"""
    return _factor_return_spread(context, date, "beta", period).rename("高低Beta收益差")


def _spearman_corr(a, b, axis=1):
    """沿指定轴计算两组序列的 Spearman 相关，按共同有效值排名。"""
    valid = np.isfinite(a) & np.isfinite(b)
    rank_a = np.nan_to_num(rankdata(np.where(valid, a, np.nan), axis=axis, method="average", nan_policy="omit"))
    rank_b = np.nan_to_num(rankdata(np.where(valid, b, np.nan), axis=axis, method="average", nan_policy="omit"))
    count = valid.sum(axis=axis, keepdims=True)
    centered_a = np.where(valid, rank_a - rank_a.sum(axis=axis, keepdims=True)/np.maximum(count, 1), 0.0)
    centered_b = np.where(valid, rank_b - rank_b.sum(axis=axis, keepdims=True)/np.maximum(count, 1), 0.0)
    covariance = np.sum(centered_a * centered_b, axis=axis)
    scale = np.sqrt(np.sum(centered_a**2, axis=axis)*np.sum(centered_b**2, axis=axis))
    return np.divide(
        covariance, scale, out=np.full_like(covariance, np.nan, dtype=float),
        where=(count.squeeze(axis=axis) > 1) & (scale > 0)
    ).clip(-1, 1)


def rotation_speed(context, date, domain, name, rank_period=20):
    """全市场行业过去 rank_period 日收益排序的日变化度，数值为1减Spearman相关。"""
    if rank_period < 1:
        raise ValueError("rank_period必须为正整数")
    end = context.get_position(date)
    dates = context.dates[:end + 1]
    if len(dates) <= rank_period:
        return pd.Series(np.nan, index=dates, name="行业轮动速度")
    daily = _industry_daily_returns(context, end).to_numpy()

    # 相邻两个滚动复合收益窗口对应前后两期行业排序。
    windows = np.lib.stride_tricks.sliding_window_view(
        daily, rank_period, axis=0
    )
    compounded = np.prod(1 + windows, axis=-1) - 1
    previous, current = compounded[:-1], compounded[1:]
    correlation = _spearman_corr(previous, current, axis=1)
    result = np.full(len(dates), np.nan)
    result[rank_period:] = 1 - correlation
    return pd.Series(result, index=dates, name="行业轮动速度")


def _valuation_return_spread(context, date, domain, name, field, period):
    """每日按估值构造市值加权十分位组合，计算低估值减高估值复合收益。"""
    end = context.get_position(date)
    dates = context.dates[:end + 1]
    values = context.read("local", field, 0, end, kind="matrix")
    cap = context.read("local", "d_essentials/circ_mv", 0, end, kind="matrix")
    valid = np.isfinite(values) & (values > 0) & np.isfinite(cap) & (cap > 0)
    ranks = rankdata(np.where(valid, values, np.nan), axis=1,
                     method="average", nan_policy="omit")
    counts = valid.sum(axis=1, keepdims=True)
    tail = np.maximum((counts * .1).astype(int), 1)
    enough = counts >= 10
    bottom = valid & enough & (ranks <= tail)
    top = valid & enough & (ranks > counts - tail)
    weights = np.where(top, cap, np.where(bottom, -cap, 0.0))
    # helper返回正组减负组；这里需要低估值组减高估值组。
    result = -_factor_return_spread(context, date, None, period, weights=weights)
    return result.reindex(dates)

def pe_return_spread(context, date, domain, name, period=5):
    """全市场 PE 低估值组减高估值组收益差。"""
    return _valuation_return_spread(context, date, domain, name,
                                    "fundamental/valuation/pe_ttm", period).rename("PE高低组收益差")

def pb_return_spread(context, date, domain, name, period=5):
    """全市场 PB 低估值组减高估值组收益差。"""
    return _valuation_return_spread(context, date, domain, name,
                                    "fundamental/valuation/pb_lf", period).rename("PB高低组收益差")


STYLE_INDICATORS = {
    function.__name__: function
    for function in (
        index_level, relative_return, new_high_low_share, pe_level, pb_level,
        earnings_bond_spread, amount, amount_share, industry_relative_return,
        size_spread, growth_value_spread, beta_spread, rotation_speed,
        pe_return_spread, pb_return_spread,
    )
}


def update_style(root=ROOT, domain="index", name="hs300", indicators=None):
    """计算指定域的风格指标，保存为共享日期轴对齐的单列二进制文件。"""
    from v2.UpdateData.utils import ensure_memmap

    context = MonitorContext(root=root)
    try:
        axis_dates = context.data.axis.full_dates
        valid_dates = axis_dates[~np.isnat(axis_dates)]
        if not valid_dates.size:
            raise ValueError("日期轴没有有效交易日")
        start, end = context.dates[0], context.dates[-1]
        output_root = Path(root) / "median_variables" / "monitor" / "style"
        for indicator in indicators or STYLE_INDICATORS:
            if indicator == "index_level" and domain != "index":
                print(f"[SKIP] {indicator}: 仅支持指数域", flush=True)
                continue
            values = STYLE_INDICATORS[indicator](context, end, domain, name).reindex(context.dates)
            if isinstance(values, pd.Series):
                values = values.to_frame()
            path = output_root / f"{indicator}.bin"
            shape = (len(axis_dates), values.shape[1])
            expected_size = int(np.prod(shape, dtype=np.int64)) * np.dtype(np.float32).itemsize
            if path.exists() and path.stat().st_size != expected_size:
                path.unlink()
            array = ensure_memmap(
                path, shape, dtype=np.float32
            )
            array[:] = np.nan
            array[:len(valid_dates)] = values.to_numpy(dtype=np.float32)
            array.flush()
            del array
            print(f"[OK] {indicator}: {start.date()} - {end.date()}", flush=True)
    finally:
        context.close()


if __name__ == "__main__":
    # 全市场专属指标使用 domain="market", name="all"；其他指标按需设置域。
    update_style(domain="index", name="hs300")
