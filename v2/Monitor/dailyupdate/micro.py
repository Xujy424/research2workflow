"""微观结构监控：Context 取数 -> 独立指标函数 -> Monitor 汇总展示。"""
from __future__ import annotations

from functools import reduce
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

try:
    from .context import MarketContext
    from .utils import historical_percentile, safe_divide
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from v2.Monitor.dailyupdate.context import MarketContext
    from v2.Monitor.dailyupdate.utils import historical_percentile, safe_divide

PERIODS = (1, 5, 10, 20)
INDEX_DOMAINS = ("沪深300", "中证500", "中证1000", "中证2000")
STYLE_DOMAINS = tuple(f"barra{i}" for i in range(1, 11))
DECAY_HORIZONS = (1, 5, 15, 30)
PATH_HORIZONS = (10, 30, 60, 120)


def _time_key(value):
    text = str(value).strip()
    if len(text) == 5 and text[2] == ":":
        return text + ":00.000000"
    if len(text) == 8 and text[2] == ":":
        return text + ".000000"
    try:
        return pd.Timestamp(f"2000-01-01 {text}").strftime("%H:%M:%S.%f")
    except (TypeError, ValueError):
        return text


def _matrix_from_indexed(indexed, column, shape):
    if column not in indexed:
        return np.full(shape, np.nan)
    return pd.to_numeric(indexed[column], errors="coerce").to_numpy(float).reshape(shape)


def _sum_preserve_nan(values, axis=0):
    x = np.asarray(values, float)
    total = np.nansum(x, axis=axis)
    total[np.all(~np.isfinite(x), axis=axis)] = np.nan
    return total


def _nanmean_rows(values):
    x = np.asarray(values, float)
    valid = np.isfinite(x).any(axis=0)
    result = np.full(x.shape[1:], np.nan)
    if valid.any():
        result[valid] = np.nanmean(x[:, valid], axis=0)
    return result


def _weighted_rows(values, weights):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    numerator = np.sum(np.where(valid, values * weights, 0.0), axis=1)
    denominator = np.sum(np.where(valid, weights, 0.0), axis=1)
    return safe_divide(numerator, denominator)


def _rolling_mean(series, period):
    x = np.asarray(series, float)
    result = np.full(len(x), np.nan)
    if period <= 0:
        raise ValueError("period must be positive")
    if len(x) < period:
        return result
    windows = np.lib.stride_tricks.sliding_window_view(x, period)
    valid = np.isfinite(windows).all(axis=1)
    means = windows.mean(axis=1)
    result[period - 1:] = np.where(valid, means, np.nan)
    return result


def _lag_correlation_columns(values, lag):
    """逐列计算固定滞后的相关系数，列方向向量化。"""
    x = np.asarray(values, float)
    if x.shape[0] <= lag:
        return np.full(x.shape[1], np.nan)
    lhs, rhs = x[lag:], x[:-lag]
    valid = np.isfinite(lhs) & np.isfinite(rhs)
    count = valid.sum(axis=0)
    left = np.where(valid, lhs, 0.0)
    right = np.where(valid, rhs, 0.0)
    left_mean = safe_divide(left.sum(axis=0), count)
    right_mean = safe_divide(right.sum(axis=0), count)
    left_centered = np.where(valid, lhs - left_mean, 0.0)
    right_centered = np.where(valid, rhs - right_mean, 0.0)
    covariance = safe_divide((left_centered * right_centered).sum(axis=0), count)
    left_std = np.sqrt(safe_divide((left_centered ** 2).sum(axis=0), count))
    right_std = np.sqrt(safe_divide((right_centered ** 2).sum(axis=0), count))
    corr = safe_divide(covariance, left_std * right_std)
    corr[(count < 5) | (left_std == 0) | (right_std == 0)] = np.nan
    return corr


def _history_percentile(series, history, period):
    values = _rolling_mean(series, period)
    return historical_percentile(values[-history:])


class MicroContext(MarketContext):
    """微观结构取数上下文，读取 sh_shot_1m / sz_shot_1m 快照和 L2 事件。"""

    def __init__(self, *args, snapshot_root=None,
                 snapshot_pattern="proc/{date}/{exchange}_shot_1m.pq", **kwargs):
        super().__init__(*args, **kwargs)
        self.snapshot_root = Path(snapshot_root or (Path(self.data.asset_root) / "l2"))
        self.snapshot_pattern = snapshot_pattern
        self._snapshot_cache = {}
        self._aligned_snapshot_cache = {}

    def snapshots(self, date, exchanges=("sh", "sz")):
        day = pd.Timestamp(date).strftime("%Y%m%d")
        if day not in self._snapshot_cache:
            frames = []
            import polars as pl
            for exchange in exchanges:
                path = self.snapshot_root / self.snapshot_pattern.format(date=day, exchange=exchange)
                if not path.exists():
                    continue
                try:
                    frame = pl.read_parquet(path).to_pandas()
                except Exception as exc:
                    warnings.warn(f"盘口快照读取失败 {path}: {exc}", RuntimeWarning)
                    continue
                if not frame.empty:
                    frame["SecurityID"] = frame["SecurityID"].astype(str).str.zfill(6)
                    frames.append(frame)
            self._snapshot_cache[day] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return self._snapshot_cache[day]

    def aligned_snapshot(self, date, topn=5):
        day = pd.Timestamp(date).strftime("%Y%m%d")
        key = (day, topn)
        if key not in self._aligned_snapshot_cache:
            self._aligned_snapshot_cache[key] = align_snapshot(self.snapshots(date), self.ticks, topn)
        return self._aligned_snapshot_cache[key]

    def l2_path(self, date, exchange, suffix):
        return self.snapshot_root / "proc" / pd.Timestamp(date).strftime("%Y%m%d") / f"{exchange}{suffix}.pq"

    def close(self):
        self._snapshot_cache.clear()
        self._aligned_snapshot_cache.clear()
        super().close()


def align_snapshot(frame, ticks, topn=5):
    """将宽表快照对齐到 [分钟, 股票]，盘口深度保留 topn 档、OFI 仅算一档。"""
    if frame is None or frame.empty:
        return None
    source = frame.copy()
    source["_tick"] = source["SecurityID"].astype(str).str.zfill(6).str[:6]
    source["_time"] = source["BarTime"].map(_time_key)
    source = source[
        ((source["_time"] >= "09:30:00.000000") & (source["_time"] <= "11:30:00.000000"))
        | ((source["_time"] >= "13:00:00.000000") & (source["_time"] <= "14:57:00.000000"))
    ]
    if source.empty:
        return None
    source = source.sort_values(["_tick", "_time"]).drop_duplicates(["_tick", "_time"], keep="last")
    times = np.array(sorted(source["_time"].unique()), dtype=object)
    shape = (len(times), len(ticks))
    stock_codes = [str(tick).zfill(6)[:6] for tick in ticks]
    indexed = source.set_index(["_time", "_tick"]).reindex(pd.MultiIndex.from_product([times, stock_codes]))
    matrices = {name: [] for name in ("bid_px", "bid_qty", "ask_px", "ask_qty")}
    columns = {"bid_px": "BidPrice", "bid_qty": "BidQty", "ask_px": "AskPrice", "ask_qty": "AskQty"}
    for level in range(1, topn + 1):
        for name, prefix in columns.items():
            matrices[name].append(_matrix_from_indexed(indexed, f"{prefix}{level}", shape))
    bid_px, bid_qty, ask_px, ask_qty = (np.asarray(matrices[k]) for k in columns)
    midpoint = (bid_px[0] + ask_px[0]) / 2
    spread = safe_divide(ask_px[0] - bid_px[0], midpoint) * 10000
    depth = _sum_preserve_nan(bid_qty, 0) + _sum_preserve_nan(ask_qty, 0)
    best_imbalance = safe_divide(bid_qty[0] - ask_qty[0], bid_qty[0] + ask_qty[0])
    ofi = np.full_like(midpoint, np.nan)
    if len(times) > 1:
        bp, bq, ap, aq = bid_px[0], bid_qty[0], ask_px[0], ask_qty[0]
        bid_flow = np.where(bp[1:] > bp[:-1], bq[1:], np.where(bp[1:] < bp[:-1], -bq[:-1], bq[1:] - bq[:-1]))
        ask_flow = np.where(ap[1:] < ap[:-1], aq[1:], np.where(ap[1:] > ap[:-1], -aq[:-1], aq[1:] - aq[:-1]))
        valid = np.isfinite(bp[1:]) & np.isfinite(bp[:-1]) & np.isfinite(ap[1:]) & np.isfinite(ap[:-1])
        ofi[1:] = np.where(valid, bid_flow - ask_flow, np.nan)
    return {"times": times, "mid": midpoint, "spread": spread, "depth": depth,
            "best_imbalance": best_imbalance, "ofi": ofi,
            "bid_px": bid_px, "bid_qty": bid_qty, "ask_px": ask_px, "ask_qty": ask_qty}


def effective_spread(quotes):
    """日内相对买卖价差中位数，单位 bp。"""
    if quotes is None:
        return None
    x = quotes["spread"]
    return np.nanmedian(x, axis=0)


def five_level_depth(quotes):
    """日内五档买卖盘数量深度中位数。"""
    if quotes is None:
        return None
    return np.nanmedian(quotes["depth"], axis=0)


def five_level_depth_notional(quotes):
    """日内五档买卖盘名义深度中位数。"""
    if quotes is None:
        return None
    bid = _sum_preserve_nan(quotes["bid_px"] * quotes["bid_qty"], 0)
    ask = _sum_preserve_nan(quotes["ask_px"] * quotes["ask_qty"], 0)
    return np.nanmedian(bid + ask, axis=0)


def order_flow_imbalance(quotes):
    """最优档 OFI 净流量/绝对流量总和。"""
    if quotes is None:
        return None
    flow = quotes["ofi"]
    return safe_divide(_sum_preserve_nan(flow, 0), _sum_preserve_nan(np.abs(flow), 0))


def active_trade_imbalance(context, date):
    """(主动买入成交额-主动卖出成交额)/两侧主动成交额。"""
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


def _order_flow_autocorrelation(quotes, lag):
    if quotes is None:
        return None
    return _lag_correlation_columns(quotes["ofi"], lag)


def order_flow_autocorrelation_lag1(quotes):
    """最优档 OFI 的一阶滞后自相关。"""
    return _order_flow_autocorrelation(quotes, 1)


def order_flow_autocorrelation_lag5(quotes):
    """最优档 OFI 的五阶滞后自相关。"""
    return _order_flow_autocorrelation(quotes, 5)


def best_level_imbalance(quotes):
    """日内最优买卖队列失衡均值。"""
    if quotes is None:
        return None
    return _nanmean_rows(quotes["best_imbalance"])


def impact_decay(quotes, horizon):
    """逐个最优档 OFI 冲击计算 horizon 分钟后剩余响应比例，再取日内中位数。"""
    if quotes is None:
        return None
    mid = quotes["mid"]
    ret = np.diff(np.log(mid), axis=0)
    flow = quotes["ofi"][1:]
    if len(ret) <= horizon:
        return np.full(mid.shape[1], np.nan)
    future_windows = np.lib.stride_tricks.sliding_window_view(ret, horizon, axis=0)
    future_returns = np.moveaxis(future_windows, -1, 1)[1:]
    immediate_return = ret[:-horizon]
    signed_flow = np.sign(flow[:-horizon])
    valid = (np.isfinite(future_returns).all(axis=1)
             & np.isfinite(immediate_return)
             & np.isfinite(signed_flow) & (signed_flow != 0))
    immediate = signed_flow * immediate_return
    remaining = signed_flow * future_returns.sum(axis=1)
    ratios = safe_divide(remaining, immediate + remaining)
    ratios[(np.abs(immediate) <= 1e-12) | ~valid] = np.nan
    valid_columns = np.isfinite(ratios).any(axis=0)
    result = np.full(mid.shape[1], np.nan)
    if valid_columns.any():
        result[valid_columns] = np.nanmedian(ratios[:, valid_columns], axis=0)
    return result


def impact_path_structure(quotes, horizon):
    """逐时点计算 horizon 窗口累计对数收益/首分钟对数收益，再取日内中位数。"""
    if quotes is None:
        return None
    mid = quotes["mid"]
    if len(mid) <= horizon:
        return np.full(mid.shape[1], np.nan)
    start, first, end = mid[:-horizon], mid[0:len(mid) - horizon], mid[horizon:]
    # For horizon == 1 the first-step return is the full window return; the ratio is undefined.
    if horizon == 1:
        return np.full(mid.shape[1], np.nan)
    ratios = safe_divide(np.log(safe_divide(end, start)), np.log(safe_divide(first, start)))
    result = np.full(mid.shape[1], np.nan)
    valid = np.isfinite(ratios).any(axis=0)
    if valid.any():
        result[valid] = np.nanmedian(ratios[:, valid], axis=0)
    return result


def impact_decay_1m(quotes):
    """1分钟冲击保留率。"""
    return impact_decay(quotes, 1)


def impact_decay_5m(quotes):
    """5分钟冲击保留率。"""
    return impact_decay(quotes, 5)


def impact_decay_15m(quotes):
    """15分钟冲击保留率。"""
    return impact_decay(quotes, 15)


def impact_decay_30m(quotes):
    """30分钟冲击保留率。"""
    return impact_decay(quotes, 30)


def impact_path_10m(quotes):
    """10分钟冲击路径结构。"""
    return impact_path_structure(quotes, 10)


def impact_path_30m(quotes):
    """30分钟冲击路径结构。"""
    return impact_path_structure(quotes, 30)


def impact_path_60m(quotes):
    """60分钟冲击路径结构。"""
    return impact_path_structure(quotes, 60)


def impact_path_120m(quotes):
    """120分钟冲击路径结构。"""
    return impact_path_structure(quotes, 120)


def jump_frequency_30m(quotes):
    """滚动30分钟波动率三倍标准差阈值以上跳跃次数/有效观测次数。"""
    if quotes is None:
        return None
    ret = np.diff(np.log(quotes["mid"]), axis=0)
    result = np.full(ret.shape[1], np.nan)
    if len(ret) <= 30:
        return result
    windows = np.lib.stride_tricks.sliding_window_view(ret, 30, axis=0)
    samples = np.moveaxis(windows, -1, 1)[:-1]
    valid_count = np.isfinite(samples).sum(axis=1)
    sigma = np.nanstd(samples, axis=1)
    eligible = np.isfinite(ret[30:]) & (valid_count[:len(ret) - 30] >= 25) & (sigma[:len(ret) - 30] > 0)
    events = eligible & (np.abs(ret[30:]) > 3 * sigma[:len(ret) - 30])
    denominator = eligible.sum(axis=0)
    numerator = events.sum(axis=0)
    valid = denominator > 0
    result[valid] = numerator[valid] / denominator[valid]
    return result


def acd_jump_intensity(quotes):
    """由30分钟跳跃事件间隔估计 ACD(1,1) 的预期跳跃强度 1/psi。"""
    if quotes is None:
        return None
    ret = np.diff(np.log(quotes["mid"]), axis=0)
    n_minutes, n_stocks = ret.shape
    result = np.full(n_stocks, np.nan)
    if n_minutes <= 30:
        return result

    # rolling[k] = ret[k:k+30]；第 i 根收益对应此前样本 rolling[i-30]。
    windows = np.lib.stride_tricks.sliding_window_view(ret, 30, axis=0)
    samples = np.moveaxis(windows, -1, 1)
    past_samples = samples[:-1]
    valid_count = np.isfinite(past_samples).sum(axis=1)
    sigma = np.nanstd(past_samples, axis=1)
    eligible_tail = np.isfinite(ret[30:]) & (valid_count[:n_minutes - 30] >= 25) & (sigma[:n_minutes - 30] > 0)
    jump_tail = eligible_tail & (np.abs(ret[30:]) > 3 * sigma[:n_minutes - 30])
    jump = np.zeros(ret.shape, dtype=bool)
    jump[30:] = jump_tail

    # 把逐分钟事件矩阵压成逐股事件位置；仅对实际发生事件的股票构造等待时间。
    alpha, beta = 0.10, 0.80
    for j in np.flatnonzero(jump.sum(axis=0) >= 2):
        events = np.flatnonzero(jump[:, j])
        waits = np.diff(events).astype(float)
        if waits.size == 0:
            continue
        omega = (1 - alpha - beta) * np.mean(waits)
        initial_psi = max(float(np.mean(waits)), 1.0)
        psi = reduce(
            lambda previous_psi, wait: omega + alpha * wait + beta * previous_psi,
            waits,
            initial_psi,
        )
        result[j] = 1 / psi
    return result


def standardized_acd_intensity(current, prior_history, min_observations=10):
    """当前 ACD 强度相对之前20个交易日的横截面逐股 z-score。"""
    x = np.asarray(prior_history, float)[-20:]
    current = np.asarray(current, float)
    valid = np.isfinite(x)
    count = valid.sum(axis=0)
    mean = safe_divide(np.where(valid, x, 0.0).sum(axis=0), count)
    centered = np.where(valid, x - mean, 0.0)
    std = np.sqrt(safe_divide((centered ** 2).sum(axis=0), count))
    zscore = safe_divide(current - mean, std)
    zscore[(count < min_observations) | (std <= 0) | ~np.isfinite(current)] = np.nan
    return zscore


def jump_volume_divergence(acd_z, volume_history):
    """ACD跳跃强度z分数减去近5日成交量z分数。"""
    volume = np.asarray(volume_history, float)[-5:]
    valid = np.isfinite(volume)
    count = valid.sum(axis=0)
    mean = safe_divide(np.where(valid, volume, 0.0).sum(axis=0), count)
    centered = np.where(valid, volume - mean, 0.0)
    std = np.sqrt(safe_divide((centered ** 2).sum(axis=0), count))
    volume_z = safe_divide(volume[-1] - mean, std)
    result = np.asarray(acd_z, float) - volume_z
    result[(count < 3) | (std <= 0) | ~np.isfinite(acd_z)] = np.nan
    return result


def amihud_illiquidity(daily_return, daily_amount):
    """Amihud 非流动性：绝对日收益/日成交额。"""
    return safe_divide(np.abs(daily_return), daily_amount)


def estimate_kyle_lambda(quotes):
    """以最优档OFI名义额回归分钟中间价收益，逐股估计 Kyle lambda。"""
    if quotes is None:
        return None
    mid = quotes["mid"]
    ret = np.diff(np.log(mid), axis=0)
    flow_notional = quotes["ofi"][1:] * mid[1:]
    valid = np.isfinite(ret) & np.isfinite(flow_notional)
    count = valid.sum(axis=0)
    x = np.where(valid, flow_notional, 0.0)
    y = np.where(valid, ret, 0.0)
    mean_x = safe_divide(x.sum(axis=0), count)
    mean_y = safe_divide(y.sum(axis=0), count)
    x_centered = np.where(valid, flow_notional - mean_x, 0.0)
    y_centered = np.where(valid, ret - mean_y, 0.0)
    variance = safe_divide((x_centered ** 2).sum(axis=0), count)
    covariance = safe_divide((x_centered * y_centered).sum(axis=0), count)
    result = safe_divide(covariance, variance)
    result[(count < 30) | (variance <= 0)] = np.nan
    return result


def order_impact_cost(quotes, order_notional, kyle_lambda=None):
    """给定委托名义金额的成本估计：半价差 + Kyle lambda × 委托名义金额。"""
    if quotes is None:
        return None
    if kyle_lambda is None:
        kyle_lambda = estimate_kyle_lambda(quotes)
    return effective_spread(quotes) / 20000 + np.asarray(kyle_lambda) * np.asarray(order_notional)


def executable_capacity(quotes, adtv, participation=0.10):
    """可成交金额代理值：min(10% ADTV, 五档盘口名义深度)。"""
    if quotes is None:
        return None
    return np.minimum(participation * np.asarray(adtv), five_level_depth_notional(quotes))


def liquidity_coverage(executable_amount, demand_amount):
    """可成交金额/需求金额。"""
    return safe_divide(executable_amount, demand_amount)


def cancellation_rate(cancel_amount, effective_order_amount):
    """撤单金额/有效委托金额；需由撤单、委托事件数据提供分子分母。"""
    return safe_divide(cancel_amount, effective_order_amount)


def daily_cancellation_rate(context, date):
    """每日撤单金额/有效申报金额；委托初始名义额按该日委托表计算。"""
    n = len(context.ticks)
    import polars as pl
    daily_frames = []
    order_frames = []
    cancel_frames = []
    for exchange in ("sh", "sz"):
        order_path = context.l2_path(date, exchange, "wt")
        cancel_path = context.l2_path(date, exchange, "cancel")
        if not order_path.exists():
            continue
        orders = pl.scan_parquet(order_path).select(
            "ChannelNo", "SecurityID", "Side", "ApplSeqNum", "Price", "OrderQty", "TransactTime"
        ).filter(
            (pl.col("Price") > 0) & (pl.col("OrderQty") > 0)
            & (pl.col("TransactTime").cast(pl.String).str.slice(0, 8) < "14:57:00")
        )
        order_totals = orders.with_columns(
            pl.col("SecurityID").cast(pl.String).str.zfill(6).str.slice(0, 6).alias("_tick"),
            (pl.col("Price") * pl.col("OrderQty")).alias("_submitted"),
        ).group_by("_tick").agg(pl.sum("_submitted").alias("submitted"))
        order_frames.append(order_totals)
        cancel_totals = None
        if cancel_path.exists():
            cancels = pl.scan_parquet(cancel_path).select(
                "ChannelNo", "SecurityID", "BidApplSeqNum", "OfferApplSeqNum", "OrderQty", "TransactTime"
            ).filter(pl.col("TransactTime").cast(pl.String).str.slice(0, 8) < "14:57:00")
            buy_cancel = cancels.select(
                "ChannelNo", "SecurityID", pl.col("BidApplSeqNum").alias("ApplSeqNum"),
                pl.lit(1).alias("Side"), "OrderQty",
            )
            sell_cancel = cancels.select(
                "ChannelNo", "SecurityID", pl.col("OfferApplSeqNum").alias("ApplSeqNum"),
                pl.lit(-1).alias("Side"), "OrderQty",
            )
            cancel_legs = pl.concat([buy_cancel, sell_cancel]).filter(
                pl.col("ApplSeqNum").is_not_null() & (pl.col("ApplSeqNum") != 0)
            )
            order_prices = orders.select("ChannelNo", "SecurityID", "Side", "ApplSeqNum", "Price")
            cancel_totals = cancel_legs.join(
                order_prices, on=["ChannelNo", "SecurityID", "Side", "ApplSeqNum"], how="inner"
            ).with_columns(
                pl.col("SecurityID").cast(pl.String).str.zfill(6).str.slice(0, 6).alias("_tick"),
                (pl.col("Price") * pl.col("OrderQty")).alias("_cancelled"),
            ).group_by("_tick").agg(pl.sum("_cancelled").alias("cancelled"))
        if cancel_totals is None:
            cancel_totals = pl.DataFrame(schema={"_tick": pl.String, "cancelled": pl.Float64}).lazy()
        cancel_frames.append(cancel_totals)
    if not order_frames:
        return np.full(n, np.nan)
    positions = pd.Index([str(t).zfill(6)[:6] for t in context.ticks])
    order_totals = pl.concat(order_frames).group_by("_tick").agg(pl.sum("submitted")).rename(
        {"submitted": "submitted"}
    ).collect(engine="streaming")
    cancel_totals = pl.concat(cancel_frames).group_by("_tick").agg(pl.sum("cancelled")).collect(engine="streaming")
    codes = order_totals["_tick"].to_numpy()
    indices = positions.get_indexer(codes)
    matched = indices >= 0
    submitted = np.zeros(n)
    submitted[indices[matched]] = order_totals["submitted"].to_numpy().astype(float)[matched]
    cancel_indices = positions.get_indexer(cancel_totals["_tick"].to_numpy())
    cancel_matched = cancel_indices >= 0
    cancelled = np.zeros(n)
    np.add.at(cancelled, cancel_indices[cancel_matched], cancel_totals["cancelled"].to_numpy().astype(float)[cancel_matched])
    observed = np.zeros(n, dtype=bool)
    observed[np.unique(indices[matched])] = True
    result = cancellation_rate(cancelled, submitted)
    result[~observed] = np.nan
    return result


def _daily_return(context, index):
    """读取 d_essentials/pct 并转换成收益率小数，pct 原值为百分数。"""
    return np.asarray(context.row("d_essentials/pct", index), float) / 100.0


class MicroMonitor:
    """调用独立的个股指标函数，再按当前域市值权重计算历史分位表。"""

    METRIC_FUNCTIONS = {
        "相对买卖价差": effective_spread,
        "五档盘口深度": five_level_depth_notional,
        "标准化OFI": order_flow_imbalance,
        "订单流自相关-lag1": order_flow_autocorrelation_lag1,
        "订单流自相关-lag5": order_flow_autocorrelation_lag5,
        "B1队列失衡": best_level_imbalance,
        "30分钟跳跃频率": jump_frequency_30m,
        "ACD预期跳跃强度": acd_jump_intensity,
        "KyleLambda": estimate_kyle_lambda,
        "冲击保留率_1m": impact_decay_1m,
        "冲击保留率_5m": impact_decay_5m,
        "冲击保留率_15m": impact_decay_15m,
        "冲击保留率_30m": impact_decay_30m,
        "冲击路径_10m": impact_path_10m,
        "冲击路径_30m": impact_path_30m,
        "冲击路径_60m": impact_path_60m,
        "冲击路径_120m": impact_path_120m,
    }

    DOMAIN_POLICY = {
        "相对买卖价差": "wide",
        "五档盘口深度": "wide",
        "标准化OFI": "wide",
        "订单流自相关-lag1": "wide",
        "订单流自相关-lag5": "wide",
        "B1队列失衡": "wide",
        "30分钟跳跃频率": "wide",
        "ACD预期跳跃强度": "wide",
        "主动成交不平衡": "index",
        "Amihud非流动性": "index",
        "跳跃-成交量偏离": "index",
        "订单冲击成本": "wide",
        "可成交量": "wide",
        "撤单率": "index",
    }

    def __init__(self, context):
        self.context = context

    def _daily_individual_factors(self, start, end, metrics=None, show_progress=True):
        c = self.context
        all_names = tuple(self.METRIC_FUNCTIONS) + (
            "主动成交不平衡", "订单流自相关-lag1", "订单流自相关-lag5",
            "Amihud非流动性", "跳跃-成交量偏离", "标准化跳跃强度",
            "订单冲击成本", "可成交量", "撤单率",
        )
        requested = None if metrics is None else set(metrics)
        names = all_names if requested is None else tuple(name for name in all_names if name in requested)
        selected = set(names)
        needs_acd = bool(selected.intersection(("ACD预期跳跃强度", "标准化跳跃强度", "跳跃-成交量偏离")))
        needs_amount = bool(selected.intersection(("Amihud非流动性", "跳跃-成交量偏离", "订单冲击成本", "可成交量")))
        needs_quotes = bool(selected.intersection(self.METRIC_FUNCTIONS)) or bool(
            selected.intersection(("订单冲击成本", "可成交量"))
        )
        factors = {name: np.full((end - start + 1, len(c.ticks)), np.nan) for name in names}
        acd_history = []
        amount_history = []
        date_indices = range(start, end + 1)
        total_days = end - start + 1
        for row, idx in enumerate(date_indices):
            date = c.dates[idx]
            quotes = c.aligned_snapshot(date) if needs_quotes else None
            amount = None
            if needs_amount:
                amount = np.asarray(c.row("m_essentials/amount", idx), float)
                if amount.ndim > 1:
                    amount = np.nansum(amount, axis=0)
            daily_values = {
                name: fn(quotes) for name, fn in self.METRIC_FUNCTIONS.items()
                if name in selected and name != "ACD预期跳跃强度"
            }
            if "ACD预期跳跃强度" in selected or needs_acd:
                acd = acd_jump_intensity(quotes)
                acd_history.append(acd)
                if "ACD预期跳跃强度" in selected:
                    daily_values["ACD预期跳跃强度"] = acd
            if "主动成交不平衡" in selected:
                daily_values["主动成交不平衡"] = active_trade_imbalance(c, date)
            if "撤单率" in selected:
                daily_values["撤单率"] = daily_cancellation_rate(c, date)
            if "Amihud非流动性" in selected:
                daily_values["Amihud非流动性"] = amihud_illiquidity(_daily_return(c, idx), amount)
            if "订单流自相关-lag1" in selected:
                daily_values["订单流自相关-lag1"] = order_flow_autocorrelation_lag1(quotes)
            if "订单流自相关-lag5" in selected:
                daily_values["订单流自相关-lag5"] = order_flow_autocorrelation_lag5(quotes)
            if "可成交量" in selected:
                daily_values["可成交量"] = executable_capacity(quotes, amount)
            if "订单冲击成本" in selected:
                kyle = daily_values.get("KyleLambda")
                if kyle is None:
                    kyle = estimate_kyle_lambda(quotes)
                daily_values["订单冲击成本"] = order_impact_cost(quotes, 0.10 * amount, kyle)
            for name, values in daily_values.items():
                if name in factors and values is not None:
                    factors[name][row] = values
            if needs_acd and len(acd_history) > 20:
                acd_matrix = np.asarray(acd_history, float)
                current_acd_z = standardized_acd_intensity(acd_matrix[-1], acd_matrix[:-1])
                if "标准化跳跃强度" in selected:
                    factors["标准化跳跃强度"][row] = current_acd_z
                if "跳跃-成交量偏离" in selected:
                    amount_history.append(amount)
                    factors["跳跃-成交量偏离"][row] = jump_volume_divergence(
                        current_acd_z, np.asarray(amount_history[-5:], float),
                    )
            elif "跳跃-成交量偏离" in selected:
                amount_history.append(amount)
            if show_progress:
                date_text = pd.Timestamp(date).strftime("%Y-%m-%d")
                print(f"[{row + 1}/{total_days}] {date_text} completed", flush=True)
        return factors

    def _domain_series(self, values, start, end, domain):
        c = self.context
        if domain.startswith("barra"):
            factor = list(c.style_masks)[int(domain[5:]) - 1]
            mask = c.matrix(c.style_masks[factor], start, end)
            weights = c.stock_weight_matrix(start, end)
            high = _weighted_rows(values, np.where(mask > 0, weights, np.nan))
            low = _weighted_rows(values, np.where(mask < 0, weights, np.nan))
            return high - low
        return _weighted_rows(values, c.domain_weights(start, end, domain))

    def snapshot(self, asof, domains=("全市场",), history=252, periods=PERIODS, metrics=None):
        c = self.context
        end = c.position(asof)
        if end < 0:
            return pd.DataFrame()
        warmup = max(history + max(periods) + 30, 252 + max(periods) + 30)
        start = max(0, end - warmup)
        daily = self._daily_individual_factors(start, end, metrics)
        selected_metrics = tuple(daily)
        all_domains = ("全市场",) + INDEX_DOMAINS + STYLE_DOMAINS + tuple(c.industry_map.keys())
        requested = None if tuple(domains) == ("全市场",) else set(domains)
        rows = []
        for metric in selected_metrics:
            if metric not in daily:
                continue
            scope = self.DOMAIN_POLICY.get(metric, "all")
            candidates = all_domains if scope == "all" else (
                ("全市场",) + INDEX_DOMAINS if scope == "index" else
                ("全市场",) + INDEX_DOMAINS + STYLE_DOMAINS
            )
            if requested is not None:
                candidates = tuple(domain for domain in candidates if domain in requested)
            for domain in candidates:
                series = self._domain_series(daily[metric], start, end, domain)
                for period in periods:
                    rows.append({"指标": metric, "域": domain, "周期": period,
                                 "历史分位": _history_percentile(series, history, period)})
        if not rows:
            return pd.DataFrame()
        result = pd.DataFrame(rows).pivot(index=["指标", "域"], columns="周期", values="历史分位")
        result.columns = [f"{int(period)}日历史分位" for period in result.columns]
        return result

    @staticmethod
    def table(result, decimals=4):
        return result.round(decimals)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    context = MicroContext()
    try:
        print(MicroMonitor(context).snapshot("2026-09-30").to_string())
    finally:
        context.close()
