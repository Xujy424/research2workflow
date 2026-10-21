"""微观结构监控：Context 取数 -> 独立指标函数 -> Monitor 汇总展示。"""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

try:
    from .context import MarketContext
    from .utils import historical_percentile, safe_divide, weighted_mean
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from v2.Monitor.context import MarketContext
    from v2.Monitor.utils import historical_percentile, safe_divide, weighted_mean

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
    return np.array([weighted_mean(value, weight) for value, weight in zip(values, weights)])


def _rolling_mean(series, period):
    x = np.asarray(series, float)
    result = np.full(len(x), np.nan)
    for i in range(period - 1, len(x)):
        sample = x[i - period + 1:i + 1]
        if np.isfinite(sample).all():
            result[i] = np.mean(sample)
    return result


def _lag_correlation(series, lag):
    x = np.asarray(series, float)
    if len(x) <= lag:
        return np.nan
    lhs, rhs = x[lag:], x[:-lag]
    valid = np.isfinite(lhs) & np.isfinite(rhs)
    if valid.sum() < 5 or np.std(lhs[valid]) == 0 or np.std(rhs[valid]) == 0:
        return np.nan
    return float(np.corrcoef(lhs[valid], rhs[valid])[0, 1])


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
    buy, sell, observed = np.zeros(n), np.zeros(n), np.zeros(n, dtype=bool)
    positions = {str(t).zfill(6)[:6]: i for i, t in enumerate(context.ticks)}
    import polars as pl
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
        ).group_by("_tick").agg(pl.sum("_buy"), pl.sum("_sell")).collect(engine="streaming")
        for code, b, s in zip(frame["_tick"].to_list(), frame["_buy"].to_list(), frame["_sell"].to_list()):
            j = positions.get(code)
            if j is not None:
                buy[j] += float(b or 0)
                sell[j] += float(s or 0)
                observed[j] = True
    result = safe_divide(buy - sell, buy + sell)
    result[~observed] = np.nan
    return result


def _order_flow_autocorrelation(quotes, lag):
    if quotes is None:
        return None
    flow = quotes["ofi"]
    result = np.full(flow.shape[1], np.nan)
    for j in range(flow.shape[1]):
        scale = np.nansum(np.abs(flow[:, j]))
        if scale > 0:
            result[j] = _lag_correlation(flow[:, j] / scale, lag)
    return result


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
    result = np.full(mid.shape[1], np.nan)
    for j in range(mid.shape[1]):
        ratios = []
        for i in range(len(ret) - horizon):
            future = ret[i + 1:i + horizon + 1]
            if not np.isfinite(flow[i, j]) or not np.isfinite(ret[i, j]) or not np.isfinite(future).all():
                continue
            immediate = np.sign(flow[i, j]) * ret[i, j]
            if abs(immediate) > 1e-12:
                remaining = np.sign(flow[i, j]) * np.sum(future)
                ratios.append(safe_divide(remaining, immediate + remaining))
        if ratios:
            result[j] = np.median(ratios)
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
    for j in range(ret.shape[1]):
        events = eligible = 0
        for i in range(30, len(ret)):
            history = ret[i - 30:i, j]
            if not np.isfinite(ret[i, j]) or np.isfinite(history).sum() < 25:
                continue
            sigma = np.nanstd(history)
            eligible += 1
            events += int(sigma > 0 and abs(ret[i, j]) > 3 * sigma)
        if eligible:
            result[j] = events / eligible
    return result


def acd_jump_intensity(quotes):
    """由30分钟跳跃事件间隔估计 ACD(1,1) 的预期跳跃强度 1/psi。"""
    if quotes is None:
        return None
    ret = np.diff(np.log(quotes["mid"]), axis=0)
    result = np.full(ret.shape[1], np.nan)
    for j in range(ret.shape[1]):
        events = []
        for i in range(30, len(ret)):
            sample = ret[i - 30:i, j]
            if np.isfinite(ret[i, j]) and np.isfinite(sample).sum() >= 25:
                sigma = np.nanstd(sample)
                if sigma > 0 and abs(ret[i, j]) > 3 * sigma:
                    events.append(i)
        waits = np.diff(events).astype(float)
        if len(waits):
            alpha, beta = 0.10, 0.80
            omega = (1 - alpha - beta) * np.mean(waits)
            psi = max(float(np.mean(waits)), 1.0)
            for wait in waits:
                psi = omega + alpha * wait + beta * psi
            result[j] = 1 / psi
    return result


def standardized_acd_intensity(current, prior_history, min_observations=10):
    """当前 ACD 强度相对之前20个交易日的横截面逐股 z-score。"""
    x = np.asarray(prior_history, float)[-20:]
    current = np.asarray(current, float)
    result = np.full(current.shape, np.nan)
    for j in range(len(current)):
        sample = x[:, j]
        sample = sample[np.isfinite(sample)]
        if np.isfinite(current[j]) and len(sample) >= min_observations and np.std(sample) > 0:
            result[j] = (current[j] - np.mean(sample)) / np.std(sample)
    return result


def jump_volume_divergence(acd_z, volume_history):
    """ACD跳跃强度z分数减去近5日成交量z分数。"""
    volume = np.asarray(volume_history, float)[-5:]
    result = np.full(np.asarray(acd_z).shape, np.nan)
    for j in range(result.size):
        sample = volume[:, j]
        sample = sample[np.isfinite(sample)]
        if np.isfinite(acd_z[j]) and len(sample) >= 3 and np.std(sample) > 0:
            result[j] = acd_z[j] - (sample[-1] - np.mean(sample)) / np.std(sample)
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
    result = np.full(mid.shape[1], np.nan)
    for j in range(mid.shape[1]):
        valid = np.isfinite(ret[:, j]) & np.isfinite(flow_notional[:, j])
        if valid.sum() >= 30 and np.var(flow_notional[valid, j]) > 0:
            x, y = flow_notional[valid, j], ret[valid, j]
            result[j] = np.cov(x, y, ddof=0)[0, 1] / np.var(x)
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
    cancelled = np.zeros(n)
    submitted = np.zeros(n)
    observed = np.zeros(n, dtype=bool)
    positions = {str(t).zfill(6)[:6]: i for i, t in enumerate(context.ticks)}
    import polars as pl
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
        ).group_by("_tick").agg(pl.sum("_submitted").alias("submitted")).collect(engine="streaming")
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
            ).group_by("_tick").agg(pl.sum("_cancelled").alias("cancelled")).collect(engine="streaming")
        submitted_map = dict(zip(order_totals["_tick"].to_list(), order_totals["submitted"].to_list()))
        cancelled_map = {} if cancel_totals is None else dict(
            zip(cancel_totals["_tick"].to_list(), cancel_totals["cancelled"].to_list())
        )
        for code in set(submitted_map) | set(cancelled_map):
            j = positions.get(code)
            if j is not None:
                submitted[j] += float(submitted_map.get(code) or 0)
                cancelled[j] += float(cancelled_map.get(code) or 0)
                observed[j] = True
    result = cancellation_rate(cancelled, submitted)
    result[~observed] = np.nan
    return result


def _daily_return(context, index):
    close = np.asarray(context.row("d_essentials/close", index), float)
    if index == 0:
        return np.full(close.shape, np.nan)
    previous = np.asarray(context.row("d_essentials/close", index - 1), float)
    return safe_divide(close, previous) - 1


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

    def _daily_individual_factors(self, start, end):
        c = self.context
        names = tuple(self.METRIC_FUNCTIONS) + (
            "主动成交不平衡", "订单流自相关-lag1", "订单流自相关-lag5",
            "Amihud非流动性", "跳跃-成交量偏离", "标准化跳跃强度",
            "订单冲击成本", "可成交量", "撤单率",
        )
        factors = {name: np.full((end - start + 1, len(c.ticks)), np.nan) for name in names}
        acd_history = []
        amount_history = []
        for row, idx in enumerate(range(start, end + 1)):
            date = c.dates[idx]
            quotes = c.aligned_snapshot(date)
            amount = np.asarray(c.row("m_essentials/amount", idx), float)
            if amount.ndim > 1:
                amount = np.nansum(amount, axis=0)
            acd_history.append(acd_jump_intensity(quotes))
            amount_history.append(amount)
            daily_values = {name: fn(quotes) for name, fn in self.METRIC_FUNCTIONS.items()}
            daily_values["主动成交不平衡"] = active_trade_imbalance(c, date)
            daily_values["撤单率"] = daily_cancellation_rate(c, date)
            daily_values["Amihud非流动性"] = amihud_illiquidity(_daily_return(c, idx), amount)
            daily_values["订单流自相关-lag1"] = order_flow_autocorrelation_lag1(quotes)
            daily_values["订单流自相关-lag5"] = order_flow_autocorrelation_lag5(quotes)
            daily_values["可成交量"] = executable_capacity(quotes, amount)
            daily_values["订单冲击成本"] = order_impact_cost(quotes, 0.10 * amount, daily_values["KyleLambda"])
            for name, values in daily_values.items():
                if values is not None:
                    factors[name][row] = values
            acd_matrix = np.asarray(acd_history, float)
            if len(acd_history) > 20:
                current_acd_z = standardized_acd_intensity(acd_matrix[-1], acd_matrix[:-1])
                factors["标准化跳跃强度"][row] = current_acd_z
                factors["跳跃-成交量偏离"][row] = jump_volume_divergence(
                    current_acd_z,
                    np.asarray(amount_history[-5:], float),
                )
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
        daily = self._daily_individual_factors(start, end)
        selected_metrics = tuple(daily) if metrics is None else tuple(metrics)
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
