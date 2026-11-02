"""微观结构指标；每个函数接收一个域，返回该域的日度时间序列。"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
if __package__:
    from .context import MonitorContext
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from v2.Monitor.v3.context import MonitorContext
from v2.UpdateData.config import ROOT
from v2.Monitor.v3.common import (
    weighted_series, bns_jump_variation, significant_jump_events,
    minute_log_returns, event_acd_intensity, rolling_zscore)


def effective_spread(context, date, domain, name, *, stock_level=False):
    """有效分钟相对报价价差的均值，单位bp；再按域权重聚合。"""
    end = context.get_position(date)
    dates = context.dates[:end + 1]
    daily_values = np.full((len(dates), len(context.ticks)), np.nan)
    for i, day in enumerate(dates):
        quotes = context.aligned_snapshot(day)
        if quotes is None:
            continue
        spread = np.asarray(quotes['spread'], dtype=float)
        valid = np.isfinite(spread)
        count = valid.sum(axis=0)
        daily_values[i] = np.divide(
            np.where(valid, spread, 0).sum(axis=0), count,
            out=np.full(len(context.ticks), np.nan), where=count > 0)
    if stock_level:
        return pd.DataFrame(daily_values, index=dates, columns=context.ticks)
    weights = context.get_weight(domain, name, end)
    return weighted_series(daily_values, weights, dates, name='相对买卖价差', signed=True)


def five_level_depth_notional(context, date, domain, name, *, stock_level=False,
                              normalize=True):
    """五档金额/此前20日日均成交额的日内均值，再按域权重聚合。
    默认返回无量纲比例；normalize=False保留元单位供可执行容量使用。
    历史窗口不含当日，需20个有效日成交额；缺失或非正分母返回NaN。
    """
    end = context.get_position(date)
    dates = context.dates[:end + 1]
    if normalize:
        amount = np.asarray(context.read(
            'local', 'd_essentials/amount', 0, end, kind='matrix'), dtype=float)
        average_amount = pd.DataFrame(np.where(
            np.isfinite(amount) & (amount >= 0), amount, np.nan)
        ).shift(1).rolling(20, min_periods=20).mean().to_numpy()
    daily_values = np.full((len(dates), len(context.ticks)), np.nan)
    for i, day in enumerate(dates):
        quotes = context.aligned_snapshot(day)
        if quotes is None:
            continue
        bid = quotes['bid_px'] * quotes['bid_qty']
        ask = quotes['ask_px'] * quotes['ask_qty']
        total = np.nansum(bid, axis=0) + np.nansum(ask, axis=0)
        all_missing = np.all(~np.isfinite(bid), axis=0) & np.all(~np.isfinite(ask), axis=0)
        total[all_missing] = np.nan
        valid = np.isfinite(total)
        count = valid.sum(axis=0)
        daily_values[i] = np.divide(
            np.where(valid, total, 0).sum(axis=0), count,
            out=np.full(len(context.ticks), np.nan), where=count > 0)
        if normalize:
            # 分母在日内固定，先取均值再相除与逐分钟归一化后取均值等价。
            denominator = average_amount[i]
            daily_values[i] = np.divide(
                daily_values[i], denominator,
                out=np.full(len(context.ticks), np.nan),
                where=np.isfinite(denominator) & (denominator > 0))
    if stock_level:
        return pd.DataFrame(daily_values, index=dates, columns=context.ticks)
    weights = context.get_weight(domain, name, end)
    label = '五档深度占日均成交额比例' if normalize else '五档盘口名义深度'
    return weighted_series(daily_values, weights, dates, name=label, signed=True)


def order_flow_imbalance(context, date, domain, name):
    """日内最优档OFI净流量除以绝对流量总和。"""
    end = context.get_position(date)
    dates = context.dates[:end + 1]
    daily_values = np.full((len(dates), len(context.ticks)), np.nan)
    for i, day in enumerate(dates):
        quotes = context.aligned_snapshot(day)
        if quotes is None:
            continue
        flow = quotes['ofi']
        numerator = np.nansum(flow, axis=0)
        denominator = np.nansum(np.abs(flow), axis=0)
        numerator[np.all(~np.isfinite(flow), axis=0)] = np.nan
        daily_values[i] = np.divide(numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator > 0)
    weights = context.get_weight(domain, name, end)
    return weighted_series(daily_values, weights, dates, name='标准化OFI', signed=True)


def order_flow_autocorrelation(context, date, domain, name, lag=1):
    """日内OFI序列的逐股 lag 阶自相关，再按域加权。"""
    end = context.get_position(date)
    dates = context.dates[:end + 1]
    daily_values = np.full((len(dates), len(context.ticks)), np.nan)
    if lag < 1:
        raise ValueError('lag must be positive')
    for i, day in enumerate(dates):
        quotes = context.aligned_snapshot(day)
        if quotes is None:
            continue
        values = quotes['ofi']
        values = np.asarray(values, dtype=float)
        result = np.full(values.shape[1], np.nan)
        if values.shape[0] <= lag:
            daily_values[i] = result
            continue
        left, right = (values[:-lag], values[lag:])
        valid = np.isfinite(left) & np.isfinite(right)
        count = valid.sum(axis=0)
        mean_left = np.divide(np.where(valid, left, 0).sum(axis=0),
            count,
            out=np.full(values.shape[1], np.nan),
            where=count > 0)
        mean_right = np.divide(np.where(valid, right, 0).sum(axis=0),
            count,
            out=np.full(values.shape[1], np.nan),
            where=count > 0)
        dx, dy = (left - mean_left, right - mean_right)
        covariance = np.where(valid, dx * dy, 0).sum(axis=0)
        denominator = np.sqrt(np.where(valid, dx*dx, 0).sum(axis=0) * np.where(valid, dy*dy, 0).sum(axis=0))
        result = np.divide(covariance, denominator, out=result, where=(count >= 3) & (denominator > 0))
        daily_values[i] = result
    weights = context.get_weight(domain, name, end)
    return weighted_series(daily_values, weights, dates, name='订单流自相关', signed=True)


def best_level_imbalance(context, date, domain, name):
    """日内最优档买卖队列失衡均值。"""
    end = context.get_position(date)
    dates = context.dates[:end + 1]
    daily_values = np.full((len(dates), len(context.ticks)), np.nan)
    for i, day in enumerate(dates):
        quotes = context.aligned_snapshot(day)
        if quotes is None:
            continue
        daily_values[i] = np.nanmean(quotes['best_imbalance'], axis=0)
    weights = context.get_weight(domain, name, end)
    return weighted_series(daily_values, weights, dates, name='B1队列失衡', signed=True)


def significant_jump_variance_ratio(context, date, domain, name, *, stock_level=False):
    """日度BNS显著跳跃变差占比，单侧1%检验；不显著为0，无效为NaN。
    仅使用连续一分钟收益，不跨午休或时间缺口；至少60个完整收益。
    数学计算复用common.bns_jump_variation。
    """
    end = context.get_position(date)
    dates = context.dates[:end + 1]
    daily_values = np.full((len(dates), len(context.ticks)), np.nan)
    for i, day in enumerate(dates):
        quotes = context.aligned_snapshot(day)
        if quotes is None:
            continue
        ret, consecutive = minute_log_returns(quotes['mid'], quotes.get('times'))
        rv, jump, _ = bns_jump_variation(ret, consecutive)
        daily_values[i] = np.divide(
            jump, rv, out=np.full(len(context.ticks), np.nan), where=rv > 0)
    if stock_level:
        return pd.DataFrame(daily_values, index=dates, columns=context.ticks)
    weights = context.get_weight(domain, name, end)
    return weighted_series(daily_values, weights, dates, name='显著跳跃变差占比', signed=True)


def acd_jump_intensity(context, date, domain, name, *, stock_level=False):
    """BNS显著日内以ABD定位跳跃，按交易分钟间隔估计ACD强度。
    有效日无事件为0；仅一次事件无法估计间隔，返回NaN。日内参数固定，非拟合ACD。
    """
    end = context.get_position(date)
    dates = context.dates[:end + 1]
    daily_values = np.full((len(dates), len(context.ticks)), np.nan)
    for i, day in enumerate(dates):
        quotes = context.aligned_snapshot(day)
        if quotes is None:
            continue
        ret, consecutive = minute_log_returns(quotes['mid'], quotes.get('times'))
        jump, valid = significant_jump_events(ret, consecutive)
        daily_values[i] = event_acd_intensity(jump, valid, np.cumsum(consecutive))
    if stock_level:
        return pd.DataFrame(daily_values, index=dates, columns=context.ticks)
    weights = context.get_weight(domain, name, end)
    return weighted_series(daily_values, weights, dates, name='ACD跳跃强度', signed=True)

def standardized_acd_intensity(context, date, domain, name, min_observations=10, *, stock_level=False):
    """BNS筛选、ABD定位的ACD强度，相对此前20日逐股标准化后聚合。"""
    end = context.get_position(date)
    dates = context.dates[:end + 1]
    raw = acd_jump_intensity(context, date, domain, name, stock_level=True).to_numpy()
    values = rolling_zscore(raw, window=20, min_periods=min_observations,
                            ddof=0, exclude_current=True).to_numpy()
    values[:20] = np.nan
    if stock_level:
        return pd.DataFrame(values, index=dates, columns=context.ticks)
    weights = context.get_weight(domain, name, end)
    return weighted_series(values, weights, dates, name='标准化ACD强度', signed=True)

def jump_volume_divergence(context, date, domain, name):
    """BNS筛选、ABD定位的ACD强度z分数减去此前5日成交额z分数。"""
    end = context.get_position(date)
    dates = context.dates[:end + 1]
    acd_z = standardized_acd_intensity(context, date, domain, name, stock_level=True).to_numpy()
    amount = context.read('local', 'd_essentials/amount', 0, end, kind='matrix')
    volume_z = rolling_zscore(amount, window=5, min_periods=3,
                              ddof=0, exclude_current=True).to_numpy()
    volume_z[:5] = np.nan
    values = acd_z - volume_z
    weights = context.get_weight(domain, name, end)
    return weighted_series(values, weights, dates, name='跳跃成交量偏离', signed=True)


def estimate_kyle_lambda(context, date, domain, name, *, stock_level=False):
    """以最优档OFI名义额回归分钟中间价收益估算逐股Kyle lambda。"""
    end = context.get_position(date)
    dates = context.dates[:end + 1]
    daily_values = np.full((len(dates), len(context.ticks)), np.nan)
    for i, day in enumerate(dates):
        quotes = context.aligned_snapshot(day)
        if quotes is None:
            continue
        mid = quotes['mid']
        ret = np.diff(np.log(mid), axis=0)
        flow = quotes['ofi'][1:] * mid[1:]
        valid = np.isfinite(ret) & np.isfinite(flow)
        count = valid.sum(axis=0)
        x = np.where(valid, flow, 0)
        y = np.where(valid, ret, 0)
        mx = np.divide(x.sum(axis=0), count, out=np.full(mid.shape[1], np.nan), where=count > 0)
        my = np.divide(y.sum(axis=0), count, out=np.full(mid.shape[1], np.nan), where=count > 0)
        xc, yc = (np.where(valid, flow - mx, 0), np.where(valid, ret - my, 0))
        variance = np.divide((xc * xc).sum(axis=0), count, out=np.full(mid.shape[1], np.nan), where=count > 0)
        covariance = np.divide((xc * yc).sum(axis=0),
            count,
            out=np.full(mid.shape[1], np.nan),
            where=count > 0)
        result = np.divide(covariance, variance, out=np.full(mid.shape[1], np.nan), where=variance > 0)
        result[(count < 30) | (variance <= 0)] = np.nan
        daily_values[i] = result
    if stock_level:
        return pd.DataFrame(daily_values, index=dates, columns=context.ticks)
    weights = context.get_weight(domain, name, end)
    return weighted_series(daily_values, weights, dates, name='KyleLambda', signed=True)


def amihud_illiquidity(context, date, domain, name):
    """逐股绝对日收益除以成交额，再按域聚合。"""
    end = context.get_position(date)
    returns = context.read('local', 'd_essentials/pct', 0, end, kind='matrix') / 100
    amount = context.read('local', 'd_essentials/amount', 0, end, kind='matrix')
    values = np.divide(np.abs(returns),
        amount,
        out=np.full_like(returns, np.nan),
        where=np.isfinite(amount) & (amount > 0))
    weights = context.get_weight(domain, name, end)
    return weighted_series(values, weights, context.dates[:end + 1], name='Amihud非流动性', signed=True)




def order_impact_cost(context, date, domain, name, order_fraction=0.1):
    """半价差成本加Kyle lambda乘以给定委托名义金额。"""
    end = context.get_position(date)
    dates = context.dates[:end + 1]
    spread = effective_spread(context, date, domain, name, stock_level=True).to_numpy()
    kyle = estimate_kyle_lambda(context, date, domain, name, stock_level=True).to_numpy()
    amount = context.read('local', 'd_essentials/amount', 0, end, kind='matrix')
    values = spread / 20000 + kyle * order_fraction * amount
    weights = context.get_weight(domain, name, end)
    return weighted_series(values, weights, dates, name='订单冲击成本', signed=True)


def executable_capacity(context, date, domain, name, participation=0.1):
    """可成交金额代理：min(participation乘日成交额，五档名义深度)。"""
    end = context.get_position(date)
    dates = context.dates[:end + 1]
    depth = five_level_depth_notional(
        context, date, domain, name, stock_level=True, normalize=False).to_numpy()
    amount = context.read('local', 'd_essentials/amount', 0, end, kind='matrix')
    values = np.minimum(depth, participation * amount)
    weights = context.get_weight(domain, name, end)
    return weighted_series(values, weights, dates, name='可成交金额', signed=True)


def impact_decay(context, date, domain, name, horizon=1):
    """逐股估算OFI价格冲击在 horizon 分钟后的保留比例，再按域聚合。"""
    end = context.get_position(date)
    dates = context.dates[:end + 1]
    daily_values = np.full((len(dates), len(context.ticks)), np.nan)
    if horizon < 1:
        raise ValueError('horizon must be positive')
    for i, day in enumerate(dates):
        quotes = context.aligned_snapshot(day)
        if quotes is None:
            continue
        mid = quotes['mid']
        ret = np.diff(np.log(mid), axis=0)
        flow = quotes['ofi'][1:]
        if len(ret) <= horizon:
            daily_values[i] = np.full(mid.shape[1], np.nan)
            continue
        future = np.moveaxis(np.lib.stride_tricks.sliding_window_view(ret, horizon, axis=0), -1, 1)[1:]
        initial = ret[:-horizon]
        signed = np.sign(flow[:-horizon])
        valid = np.isfinite(future).all(axis=1) & np.isfinite(initial) & np.isfinite(signed) & (signed != 0)
        first, remaining = (signed * initial, signed * future.sum(axis=1))
        ratio = np.divide(remaining,
            first + remaining,
            out=np.full_like(remaining, np.nan),
            where=np.abs(first + remaining) > 1e-12)
        ratio[~valid | (np.abs(first) <= 1e-12)] = np.nan
        daily_values[i] = np.nanmedian(ratio, axis=0)
    weights = context.get_weight(domain, name, end)
    return weighted_series(daily_values, weights, dates, name='冲击保留率', signed=True)


def impact_path(context, date, domain, name, horizon=10):
    """horizon 分钟累计对数收益与首分钟收益的逐股中位比值。"""
    end = context.get_position(date)
    dates = context.dates[:end + 1]
    daily_values = np.full((len(dates), len(context.ticks)), np.nan)
    for i, day in enumerate(dates):
        quotes = context.aligned_snapshot(day)
        if quotes is None:
            continue
        mid = quotes['mid']
        if horizon <= 1 or len(mid) <= horizon:
            daily_values[i] = np.full(mid.shape[1], np.nan)
            continue
        start, first, finish = (mid[:-horizon], mid[1:len(mid) - horizon + 1], mid[horizon:])
        first_return = np.log(first / start)
        total_return = np.log(finish / start)
        ratio = np.divide(total_return,
            first_return,
            out=np.full_like(total_return, np.nan),
            where=np.abs(first_return) > 1e-12)
        daily_values[i] = np.nanmedian(ratio, axis=0)
    weights = context.get_weight(domain, name, end)
    return weighted_series(daily_values, weights, dates, name='冲击路径', signed=True)


def active_trade_imbalance(context, date, domain, name):
    """当日主动买入成交额减主动卖出成交额，再除以双边成交额。"""
    import polars as pl
    end = context.get_position(date)
    dates = context.dates[:end + 1]
    daily_values = np.full((len(dates), len(context.ticks)), np.nan)
    for i, day in enumerate(dates):
        ticks = pd.Index([str(t).zfill(6)[:6] for t in context.ticks])
        buy, sell = (np.zeros(len(ticks)), np.zeros(len(ticks)))
        observed = np.zeros(len(ticks), dtype=bool)
        found = False
        for exchange in ('sh', 'sz'):
            path = context.l2_path(day, exchange, 'cj')
            if not path.exists():
                continue
            found = True
            frame = pl.read_parquet(path).filter(pl.col('TransactTime').cast(pl.String).str.slice(0,
                8) < '14:57:00')
            if frame.is_empty():
                continue
            codes = frame['SecurityID'].cast(pl.String).str.zfill(6).str.slice(0, 6).to_numpy()
            index = ticks.get_indexer(codes)
            valid = index >= 0
            amount = (frame['Price'] * frame['OrderQty']).to_numpy().astype(float)
            side = frame['Side'].to_numpy()
            np.add.at(buy, index[valid & (side > 0)], amount[valid & (side > 0)])
            np.add.at(sell, index[valid & (side < 0)], amount[valid & (side < 0)])
            observed[index[valid]] = True
        if not found:
            daily_values[i] = np.full(len(ticks), np.nan)
            continue
        result = np.divide(buy - sell, buy + sell, out=np.full(len(ticks), np.nan), where=buy + sell > 0)
        result[~observed] = np.nan
        daily_values[i] = result
    weights = context.get_weight(domain, name, end)
    return weighted_series(daily_values, weights, dates, name='主动成交不平衡', signed=True)


def daily_cancellation_rate(context, date, domain, name):
    """逐股撤单名义金额除以申报名义金额，再按域聚合。"""
    import polars as pl
    end = context.get_position(date)
    dates = context.dates[:end + 1]
    daily_values = np.full((len(dates), len(context.ticks)), np.nan)
    for i, day in enumerate(dates):
        tick_index = pd.Index([str(t).zfill(6)[:6] for t in context.ticks])
        submitted, cancelled = (np.zeros(len(tick_index)), np.zeros(len(tick_index)))
        observed = np.zeros(len(tick_index), dtype=bool)
        any_orders = False
        for exchange in ('sh', 'sz'):
            order_path = context.l2_path(day, exchange, 'wt')
            cancel_path = context.l2_path(day, exchange, 'cancel')
            if not order_path.exists():
                continue
            any_orders = True
            orders = pl.read_parquet(order_path).filter((pl.col('Price') > 0) & (pl.col('OrderQty') > 0) & (pl.col('TransactTime').cast(pl.String).str.slice(0,
                8) < '14:57:00'))
            order_codes = orders['SecurityID'].cast(pl.String).str.zfill(6).str.slice(0, 6).to_numpy()
            order_ids = tick_index.get_indexer(order_codes)
            valid = order_ids >= 0
            amounts = (orders['Price'] * orders['OrderQty']).to_numpy().astype(float)
            np.add.at(submitted, order_ids[valid], amounts[valid])
            observed[order_ids[valid]] = True
            if not cancel_path.exists():
                continue
            cancels = pl.read_parquet(cancel_path).filter(pl.col('TransactTime').cast(pl.String).str.slice(0,
                8) < '14:57:00')
            order_keys = orders.select('ChannelNo', 'SecurityID', 'Side', 'ApplSeqNum', 'Price')
            buy = cancels.select('ChannelNo',
                'SecurityID',
                pl.col('BidApplSeqNum').alias('ApplSeqNum'),
                pl.lit(1).alias('Side'),
                'OrderQty')
            sell = cancels.select('ChannelNo',
                'SecurityID',
                pl.col('OfferApplSeqNum').alias('ApplSeqNum'),
                pl.lit(-1).alias('Side'),
                'OrderQty')
            legs = pl.concat([buy,
                sell]).filter(pl.col('ApplSeqNum').is_not_null() & (pl.col('ApplSeqNum') != 0))
            matched = legs.join(order_keys, on=['ChannelNo', 'SecurityID', 'Side', 'ApplSeqNum'], how='inner')
            if matched.is_empty():
                continue
            codes = matched['SecurityID'].cast(pl.String).str.zfill(6).str.slice(0, 6).to_numpy()
            ids = tick_index.get_indexer(codes)
            good = ids >= 0
            amount = (matched['Price'] * matched['OrderQty']).to_numpy().astype(float)
            np.add.at(cancelled, ids[good], amount[good])
        if not any_orders:
            daily_values[i] = np.full(len(tick_index), np.nan)
            continue
        result = np.divide(cancelled, submitted, out=np.full(len(tick_index), np.nan), where=submitted > 0)
        result[~observed] = np.nan
        daily_values[i] = result
    weights = context.get_weight(domain, name, end)
    return weighted_series(daily_values, weights, dates, name='撤单率', signed=True)


MICRO_INDICATORS = {function.__name__: function for function in (effective_spread,
    five_level_depth_notional,
    order_flow_imbalance,
    order_flow_autocorrelation,
    best_level_imbalance,
    significant_jump_variance_ratio,
    acd_jump_intensity,
    estimate_kyle_lambda,
    amihud_illiquidity,
    standardized_acd_intensity,
    jump_volume_divergence,
    order_impact_cost,
    executable_capacity,
    impact_decay,
    impact_path,
    active_trade_imbalance,
    daily_cancellation_rate)}


def update_micro(root=ROOT, domain='index', name='hs300', indicators=None):
    """计算17个独立指标，保存完整日期轴对齐的float32单列时序bin。

    路径沿用macro/style/risk：median_variables/monitor/micro/<指标名>.bin。
    每次运行对应一个域；indicators=None计算全部，也可传单个指标名或名称序列。
    """
    from v2.UpdateData.utils import ensure_memmap
    selected = tuple(MICRO_INDICATORS if indicators is None else (indicators,
        ) if isinstance(indicators,
        str) else indicators)
    unknown = set(selected) - MICRO_INDICATORS.keys()
    if unknown:
        raise ValueError(f"未知micro指标: {', '.join(sorted(unknown))}")
    if not selected:
        return {}
    context = MonitorContext(root=root)
    try:
        axis_dates = context.data.axis.full_dates
        valid_dates = axis_dates[~np.isnat(axis_dates)]
        if not valid_dates.size:
            raise ValueError('日期轴没有有效交易日')
        start, end = (context.dates[0], context.dates[-1])
        output_root = Path(root) / 'median_variables' / 'monitor' / 'micro'
        outputs = {}
        for indicator in selected:
            values = MICRO_INDICATORS[indicator](context, end, domain, name)
            if not isinstance(values, pd.Series):
                raise TypeError(f'{indicator}必须返回日期索引的pd.Series')
            values = values.reindex(context.dates)
            path = output_root / f'{indicator}.bin'
            array = ensure_memmap(path, (len(axis_dates), 1), dtype=np.float32)
            try:
                array[:] = np.nan
                array[:len(valid_dates), 0] = values.to_numpy(dtype=np.float32)
                array.flush()
            finally:
                del array
            outputs[indicator] = path
            print(f'[OK] {indicator}: {start.date()} - {end.date()}', flush=True)
        return outputs
    finally:
        context.close()


__all__ = [*MICRO_INDICATORS, 'MICRO_INDICATORS', 'update_micro']


if __name__ == '__main__':
    DOMAIN = 'index'
    NAME = 'hs300'
    INDICATORS = None
    update_micro(root=ROOT, domain=DOMAIN, name=NAME, indicators=INDICATORS)
