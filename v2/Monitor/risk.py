"""风险指标日序列；每个函数接收一个域并返回该域的 pandas Series。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

if __package__:
    from .context import MonitorContext
else:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from v2.Monitor.v3.context import MonitorContext

from v2.UpdateData.config import ROOT
from v2.Monitor.v3.common import domain_returns, rolling_compound_return

RETURNS = "d_essentials/pct"
CLOSE = "d_essentials/close"
HIT_FLOOR = "basic/hit_floor"
PRICE_FLOOR = "basic/price_floor"
INDEX_NAMES = {
    "沪深300": "hs300", "中证500": "zz500", "中证1000": "zz1000",
    "中证2000": "zz2000", "全A": "zzfull",
}


def corr(values):
    """计算单个时间窗口内的平均两两相关性。"""
    values = np.array(values, dtype=float, copy=True)
    if values.ndim != 2:
        raise ValueError("corr输入必须是[时间,股票]二维矩阵")
    finite = np.isfinite(values)
    has_values = finite.any(axis=0)
    values, finite = values[:, has_values], finite[:, has_values]
    if values.shape[1] < 2:
        return np.nan

    median = np.nanmedian(np.where(finite, values, np.nan), axis=0)
    values = np.where(finite, values, median)
    values -= values.mean(axis=0)
    norm = np.linalg.norm(values, axis=0)
    valid = norm > 0
    if valid.sum() < 2:
        return np.nan

    values = values[:, valid] / norm[valid]
    count = values.shape[1]
    summed = values.sum(axis=1)
    return float(np.clip((summed @ summed - count) / (count * (count - 1)), -1, 1))


def pca(sample):
    """计算单个收益窗口第一主成分的方差解释比例。"""
    sample = np.asarray(sample, dtype=float)
    finite = np.isfinite(sample)
    keep = finite.sum(axis=0) >= int(np.ceil(len(sample) * .8))
    sample, finite = sample[:, keep], finite[:, keep]
    if sample.shape[1] < 2:
        return np.nan

    mean = np.nanmean(sample, axis=0)
    sample = np.where(finite, sample - mean, 0)
    norm = np.linalg.norm(sample, axis=0)
    valid = norm > 0
    if valid.sum() < 2:
        return np.nan

    sample = sample[:, valid] / norm[valid]
    gram = sample @ sample.T if sample.shape[0] <= sample.shape[1] else sample.T @ sample
    return np.linalg.eigvalsh(gram)[-1] / np.trace(gram)


def cross_section_dispersion(context, date, domain, name):
    """计算指定单一域内股票日收益的横截面标准差，单位为百分数。"""
    end = context.get_position(date)
    values = context.read("local", RETURNS, 0, end, kind="matrix")
    weight = context.get_weight(domain, name, end)
    selected = np.where(np.isfinite(weight) & (weight != 0), values, np.nan)
    result = np.nanstd(selected, axis=1, ddof=1)
    return pd.Series(result, index=context.dates[:end + 1], name="横截面收益离散度")


def average_correlation(context, date, domain, name, window=60):
    """按滚动窗口计算指定域股票的平均两两收益相关性。"""
    end = context.get_position(date)
    returns = context.read("local", RETURNS, 0, end, kind="matrix")
    weight = context.get_weight(domain, name, end)
    values = np.where(np.isfinite(weight) & (weight != 0), returns / 100, np.nan)
    dates = context.dates[:end + 1]
    result = np.full(len(dates), np.nan)
    if window < 1 or len(values) < window:
        return pd.Series(result, index=dates, name="平均收益相关性")

    windows = np.lib.stride_tricks.sliding_window_view(values, window, axis=0).transpose(0, 2, 1)
    for i, sample in enumerate(windows):
        result[i + window - 1] = corr(sample)
    return pd.Series(result, index=dates, name="平均收益相关性")


def downside_correlation(context, date, domain, name, window=120):
    """计算指定域市场收益最低20%窗口内的平均两两相关性。"""
    end = context.get_position(date)
    returns = context.read("local", RETURNS, 0, end, kind="matrix")
    weight = context.get_weight(domain, name, end)
    values = np.where(np.isfinite(weight) & (weight != 0), returns / 100, np.nan)
    market = domain_returns(context, domain, name, end, returns).to_numpy()
    dates = context.dates[:end + 1]
    result = np.full(len(dates), np.nan)
    if window < 1 or window > len(dates):
        return pd.Series(result, index=dates, name="下跌条件相关性")

    for i in range(window - 1, len(dates)):
        start = i - window + 1
        market_window = market[start:i + 1]
        valid = np.isfinite(market_window)
        if valid.sum() == 0:
            continue
        cutoff = np.nanquantile(market_window[valid], .20)
        sample = values[start:i + 1][valid & (market_window <= cutoff)]
        if len(sample) >= 3:
            result[i] = corr(sample)

    return pd.Series(result, index=dates, name="下跌条件相关性")


def pca_explained_ratio(context, date, domain, name, window=60):
    """计算指定域滚动收益矩阵第一主成分的方差解释比例。"""
    end = context.get_position(date)
    values = context.read("local", RETURNS, 0, end, kind="matrix")
    weight = context.get_weight(domain, name, end)
    values = np.where(np.isfinite(weight) & (weight != 0), values, np.nan)
    result = np.full(len(values), np.nan)
    if window < 1:
        return pd.Series(result, index=context.dates[:end + 1], name="首主成分解释率")
    for i in range(window - 1, len(values)):
        result[i] = pca(values[i - window + 1:i + 1])
    return pd.Series(result, index=context.dates[:end + 1], name="首主成分解释率")


def negative_return_ratio(context, date, domain, name):
    """指定域内负收益股票占收盘价有效股票的比例。"""
    end = context.get_position(date)
    returns = context.read("local", RETURNS, 0, end, kind="matrix")
    close = context.read("local", CLOSE, 0, end, kind="matrix")
    weight = context.get_weight(domain, name, end)
    valid = np.isfinite(weight) & (weight != 0) & np.isfinite(close)
    count = valid.sum(axis=1)
    ratio = np.divide(((returns < 0) & valid).sum(axis=1), count, out=np.full(len(count), np.nan), where=count > 0)
    return pd.Series(ratio, index=context.dates[:end + 1], name="负收益占比")

def extreme_loss_ratio(context, date, domain, name):
    """指定域内日跌幅超过5%的股票占收盘价有效股票的比例。"""
    end = context.get_position(date)
    returns = context.read("local", RETURNS, 0, end, kind="matrix")
    close = context.read("local", CLOSE, 0, end, kind="matrix")
    weight = context.get_weight(domain, name, end)
    valid = np.isfinite(weight) & (weight != 0) & np.isfinite(close)
    count = valid.sum(axis=1)
    ratio = np.divide(((returns < -5) & valid).sum(axis=1), count, out=np.full(len(count), np.nan), where=count > 0)
    return pd.Series(ratio, index=context.dates[:end + 1], name="极端下跌占比")

def limit_down_ratio(context, date, domain, name):
    """指定域内跌停股票占收盘价有效且适用涨跌幅限制股票的比例。"""
    end = context.get_position(date)
    floor = context.read("local", HIT_FLOOR, 0, end, kind="matrix")
    close = context.read("local", CLOSE, 0, end, kind="matrix")
    eligible = context.read("local", PRICE_FLOOR, 0, end, kind="matrix") > 0
    weight = context.get_weight(domain, name, end)
    valid = np.isfinite(weight) & (weight != 0) & np.isfinite(close) & eligible
    count = valid.sum(axis=1)
    ratio = np.divide(((floor > 0) & valid).sum(axis=1), count, out=np.full(len(count), np.nan), where=count > 0)
    return pd.Series(ratio, index=context.dates[:end + 1], name="跌停扩散比例")

def downside_resonance(context, date, domain, name):
    """指定域负收益、极端下跌和跌停占比的等权均值。"""
    components = [
        negative_return_ratio(context, date, domain, name),
        extreme_loss_ratio(context, date, domain, name),
        limit_down_ratio(context, date, domain, name),
    ]
    return (sum(components) / len(components)).rename("市场下行共振")


def worst_rolling_loss(context, date, domain, name, lookback=252, loss_window=5):
    """计算指定单一域过去lookback日内最差loss_window日复合收益。"""
    end = context.get_position(date)
    returns = context.read("local", RETURNS, 0, end, kind="matrix")
    result = domain_returns(context, domain, name, end, returns)
    compound = rolling_compound_return(result, loss_window)
    return compound.rolling(lookback, min_periods=lookback).min().rename("最差滚动收益")


def realized_volatility(context, date, domain, name, window=20):
    """计算指定单一域滞后权重组合收益的年化滚动标准差。"""
    end = context.get_position(date)
    returns = context.read("local", RETURNS, 0, end, kind="matrix")
    result = domain_returns(context, domain, name, end, returns)
    return result.rolling(window, min_periods=window).std(ddof=1).mul(np.sqrt(252)).rename(f"RV{window}")

def rv_ratio(context, date, domain, name, short_window=5, long_window=20):
    """计算指定单一域短期与长期实现波动率之比。"""
    end = context.get_position(date)
    returns = context.read("local", RETURNS, 0, end, kind="matrix")
    result = domain_returns(context, domain, name, end, returns)
    shortterm = result.rolling(short_window, min_periods=short_window).std(ddof=1)
    longterm = result.rolling(long_window, min_periods=long_window).std(ddof=1)
    return shortterm.div(longterm.where(longterm > 0)).rename(f"RV{short_window}_RV{long_window}")


def index_iv30(context, date, name):
    """读取本地按共享日期轴保存的指数IV30，转换为小数年化波动率。"""
    name = INDEX_NAMES.get(name, name)
    fields = {"hs300": "option/HS300_IV30", "zz1000": "option/CSI1000_IV30"}
    values = context.read("local", fields[name], 0, kind="series") / 100
    end = pd.Timestamp(date).normalize()
    return values.loc[:end].rename(f"{name}_IV30")

def imply_vol_premium(context, date, domain, name):
    """计算指定指数IV30方差减30日历史实现方差。"""
    if domain != "index":
        raise ValueError("imply_vol_premium只支持domain='index'")
    # 与update_risk保存的IV30口径一致：仅用过去有效值填补最多3个交易日。
    iv = index_iv30(context, date, name).ffill(limit=3)
    rv = realized_volatility(context, date, domain, name, window=30)
    return (iv.pow(2) - rv.pow(2)).rename("IV30方差减RV30方差")


def iv_regime_score(context, date, domain, name, iv_weight=0.7):
    """计算IV与价格动量的平滑综合分数，阈值判定由调用方完成。"""
    if domain != "index":
        raise ValueError("iv_regime_score只支持domain='index'")
    if not 0 <= iv_weight <= 1:
        raise ValueError("iv_weight须在[0, 1]内")

    iv = index_iv30(context, date, name).where(lambda x: x > 0).ffill(limit=5)
    log_iv = np.log(iv)
    fast_iv = log_iv.ewm(span=5, adjust=False, min_periods=5).mean()
    slow_iv = log_iv.ewm(span=20, adjust=False, min_periods=20).mean()
    iv_scale = log_iv.diff().rolling(60, min_periods=40).std() * np.sqrt(10)
    iv_score = -(fast_iv - slow_iv) / iv_scale.where(iv_scale > 1e-8)

    end = context.get_position(date)
    returns = context.read("local", RETURNS, 0, end, kind="matrix")
    daily = domain_returns(context, domain, name, end, returns)
    nav = (1 + daily).cumprod()
    price_scale = daily.rolling(20).std() * np.sqrt(5)
    price_score = nav.pct_change(5, fill_method=None) / price_scale.where(price_scale > 1e-8)

    score = (iv_weight * iv_score + (1 - iv_weight) * price_score)
    score = score.clip(-3, 3).ewm(span=3, adjust=False, min_periods=3).mean().where(score.notna())
    return score.rename("IV风险分数")


def liquidity_pressure(context, date, domain, name, window=252):
    """价差、深度收缩和Amihud的滚动标准分等权合成流动性压力。"""
    from .micro import amihud_illiquidity, five_level_depth_notional, effective_spread

    spread = effective_spread(context, date, domain, name)
    depth = five_level_depth_notional(context, date, domain, name)
    amihud = amihud_illiquidity(context, date, domain, name)
    depth_decline = -depth.pct_change(20)
    components = []
    for values in (spread, depth_decline, amihud):
        mean = values.rolling(window, min_periods=60).mean()
        std = values.rolling(window, min_periods=60).std(ddof=1)
        components.append(values.sub(mean).div(std.where(std > 0)))
    return pd.concat(components, axis=1).mean(axis=1, skipna=False).rename("流动性压力")



RISK_INDICATORS = {
    function.__name__: function
    for function in (
        cross_section_dispersion, average_correlation, downside_correlation,
        pca_explained_ratio, negative_return_ratio, extreme_loss_ratio,
        limit_down_ratio, downside_resonance, worst_rolling_loss,
        realized_volatility, rv_ratio, index_iv30, imply_vol_premium,
        iv_regime_score, liquidity_pressure,
    )
}


def update_risk(root=ROOT, domain="index", name="hs300", indicators=None):
    """计算风险指标并保存为按共享日期轴对齐的单列二进制文件。"""
    from v2.UpdateData.utils import ensure_memmap

    context = MonitorContext(root=root)
    try:
        axis_dates = context.data.axis.full_dates
        valid_dates = axis_dates[~np.isnat(axis_dates)]
        if not valid_dates.size:
            raise ValueError("日期轴没有有效交易日")
        start, end = context.dates[0], context.dates[-1]
        output_root = Path(root) / "median_variables" / "monitor" / "risk"

        for indicator in indicators or RISK_INDICATORS:
            calculate = RISK_INDICATORS[indicator]
            values = (calculate(context, end, name) if indicator == "index_iv30"
                      else calculate(context, end, domain, name))
            values = values.reindex(context.dates)
            if indicator == "index_iv30":
                # 仅用于本地risk模型输入；原始option IV30文件保留真实缺失。
                values = values.ffill(limit=3)
            array = ensure_memmap(
                output_root / f"{indicator}.bin", (len(axis_dates), 1), dtype=np.float32
            )
            array[:] = np.nan
            array[:len(valid_dates), 0] = values.to_numpy(dtype=np.float32)
            array.flush()
            del array
            print(f"[OK] {indicator}: {start.date()} - {end.date()}", flush=True)
    finally:
        context.close()


if __name__ == "__main__":
    # 在 IDE 中设置域和名称；需要时可把指标名元组传给 update_risk。
    update_risk(domain="index", name="hs300")



