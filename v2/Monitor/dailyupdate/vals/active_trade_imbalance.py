"""主动成交不平衡; Context entry for fetching and calculating this indicator."""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..utils import safe_divide
from ..config import MonitorConfig
from ..utils import metric_result
from ..utils import _weighted_rows, _history_percentile


def _compute(context, date):
    """(主动买入成交额-主动卖出成交额)/两侧主动成交额。"""
    # 按本指标定义计算，沿用原有窗口、单位和缺失值处理口径。
    n = len(context.ticks)
    positions = pd.Index([str(t).zfill(6)[:6] for t in context.ticks])
    import polars as pl
    exchange_frames = []
    for exchange in ("sh", "sz"):
        path = context.l2_path(date, exchange, "cj")
        if not path.exists():
            continue
        frame = pl.scan_parquet(path).select("SecurityID", "Side", "Price", "OrderQty", "TransactTime").filter(
            pl.col("TransactTime").cast(pl.String).str.slice(0, 8) < "14:57:00"
        ).with_columns(
            pl.col("SecurityID").cast(pl.String).str.zfill(6).str.slice(0, 6).alias("_tick"),
            pl.when(pl.col("Side") > 0).then(pl.col("Price") * pl.col("OrderQty")).otherwise(0).alias("_buy"),
            pl.when(pl.col("Side") < 0).then(pl.col("Price") * pl.col("OrderQty")).otherwise(0).alias("_sell"),
        ).group_by("_tick").agg(pl.sum("_buy"), pl.sum("_sell"))
        exchange_frames.append(frame)
    if not exchange_frames:
        return np.full(n, np.nan)
    frame = pl.concat(exchange_frames).group_by("_tick").agg(
        pl.sum("_buy"), pl.sum("_sell")
    ).collect(engine="streaming")
    indices = positions.get_indexer(frame["_tick"].to_numpy())
    matched = indices >= 0
    buy, sell = np.zeros(n), np.zeros(n)
    np.add.at(buy, indices[matched], frame["_buy"].to_numpy().astype(float)[matched])
    np.add.at(sell, indices[matched], frame["_sell"].to_numpy().astype(float)[matched])
    observed = np.zeros(n, dtype=bool)
    observed[np.unique(indices[matched])] = True
    result = safe_divide(buy - sell, buy + sell)
    result[~observed] = np.nan
    return result


def calculate(context, date, config=None, *, scope='index'):
    """从统一Context读取本指标输入，计算当日各域/周期的数值。"""
    config = config or MonitorConfig()
    warmup = max(config.history + max(config.periods) + 30, 252 + max(config.periods) + 30)
    start, end = context.window(date, warmup)
    daily = np.full((end-start+1, len(context.ticks)), np.nan)
    parameters = {}
    for row, idx in enumerate(range(start, end+1)):
        key = ('active_trade_imbalance', tuple(sorted(parameters.items())), idx)
        values = context.factor_cache.get(key)
        if values is None:
            values = _compute(context, context.dates[idx])
            values = np.full(len(context.ticks), np.nan) if values is None else values
            context.cache_factor(key, values)
        daily[row] = values
    # 按域取权重；barra域按高低组分别加权后相减。
    rows = []
    for domain in context.eligible_domains(config.domains, scope):
        high, low = context.domain_pair_weights(start, end, domain)
        series = _weighted_rows(daily, high)
        if low is not None:
            series = series - _weighted_rows(daily, low)
        for period in config.periods:
            rows.append(dict(domain=domain, period=period,
                             percentile=_history_percentile(series, config.history, period)))
    return metric_result(rows)
