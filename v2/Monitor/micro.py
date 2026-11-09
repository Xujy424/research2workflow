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

def depth_decline(context, date, domain, name, window=20, *, stock_level=False):
    """盘口深度下降幅度：1-当日平均五档金额/此前window日均值。
    基准不含当日；正值为下降，负值为增加，0.2表示下降20%。先逐股计算再聚合。
    """
    depth = five_level_depth_notional(context, date, domain, name, stock_level=True, normalize=False)
    baseline = depth.shift(1).rolling(window).mean()
    values = 1 - depth / baseline.where(baseline > 0)
    if stock_level:
        return values
    end = context.get_position(date)
    weights = context.get_weight(domain, name, end)
    return weighted_series(values.to_numpy(), weights, values.index, name='盘口深度下降幅度', signed=True)


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

def order_impact_cost(context, date, domain, name, order_fraction=0.1, *,
                      impact_coefficient=1.0, alpha=0.5, window=20):
    """成本率=半价差+κ*(Q/ADV)^alpha，默认平方根模型，κ=Y*日波动率。
    Q=order_fraction*ADV，因此Q/ADV直接为order_fraction；ADV为此前window日日均成交额。
    日波动率用此前window日日收益标准差，不含当日；Y默认1，须由执行数据校准。
    输出小数比例。冲击项沿用容量指标的峰值口径，非精确平均执行成本。
    """
    end = context.get_position(date)
    dates = context.dates[:end + 1]
    spread = effective_spread(context, date, domain, name, stock_level=True).to_numpy()
    amount = context.read('local', 'd_essentials/amount', 0, end, kind='matrix')
    adv = pd.DataFrame(amount).shift(1).rolling(window).mean().to_numpy()
    returns = context.read('local', 'd_essentials/pct', 0, end, kind='matrix') / 100
    sigma = pd.DataFrame(returns).shift(1).rolling(window).std(ddof=1).to_numpy()
    kappa = impact_coefficient * sigma
    values = np.where(adv > 0, spread / 20000 + kappa * order_fraction ** alpha, np.nan)
    weights = context.get_weight(domain, name, end)
    return weighted_series(values, weights, dates, name='订单冲击成本', signed=True)


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

def turnover_illiquidity(context, date, domain, name):
    """逐股绝对日收益率/流通股换手率，再按域聚合；两者均用小数比例。
    换手率为成交股数/流通股数，消除成交规模量纲，不代表剔除报价价差。
    """
    end = context.get_position(date)
    returns = context.read('local', 'd_essentials/pct', 0, end, kind='matrix') / 100
    turnover = context.read('local', 'd_essentials/circ_turnover', 0, end, kind='matrix')
    values = np.divide(np.abs(returns), turnover,
        out=np.full_like(returns, np.nan),
        where=np.isfinite(returns) & np.isfinite(turnover) & (turnover > 0))
    weights = context.get_weight(domain, name, end)
    return weighted_series(values, weights, context.dates[:end + 1], name='换手率非流动性', signed=True)


def executable_capacity(context, date, domain, name, *, max_impact=0.005,
                        impact_coefficient=1.0, window=20, stock_level=False):
    """可成交金额代理（元）：min(原始五档金额，ADV*(max_impact/Y/sigma)^2)。
    ADV与日收益标准差均用此前window日，不含当日；五档金额关闭归一化。
    max_impact为峰值相对冲击预算，默认50bp，Y默认1；不含报价价差成本。
    当前沿用日内平均五档深度，未额外施加压力折扣。
    """
    end = context.get_position(date)
    dates = context.dates[:end + 1]
    depth = five_level_depth_notional(context, date, domain, name, stock_level=True, normalize=False).to_numpy()
    amount = context.read('local', 'd_essentials/amount', 0, end, kind='matrix')
    adv = pd.DataFrame(amount).shift(1).rolling(window).mean().to_numpy()
    returns = context.read('local', 'd_essentials/pct', 0, end, kind='matrix') / 100
    sigma = pd.DataFrame(returns).shift(1).rolling(window).std(ddof=1).to_numpy()
    rho = np.divide(max_impact, impact_coefficient * sigma, out=np.full_like(sigma, np.nan), where=sigma > 0) ** 2
    values = np.minimum(depth, rho * adv)
    if stock_level:
        return pd.DataFrame(values, index=dates, columns=context.ticks)
    weights = context.get_weight(domain, name, end)
    return weighted_series(values, weights, dates, name='可成交金额', signed=True)


def impact_decay(context, date, domain, name, horizon=1, *, stock_level=False):
    """初始一分钟收益及随后horizon分钟收益之和/初始收益，日内取中位数。
    1为全部保留，0为完全反转，负数为过度反转，超过1为同向放大。
    仅选OFI有效且非零、初始收益非零的连续分钟窗口；为事后价格保留代理。
    """
    end = context.get_position(date)
    dates = context.dates[:end + 1]
    daily_values = np.full((len(dates), len(context.ticks)), np.nan)
    if horizon < 1:
        raise ValueError('horizon must be positive')
    for i, day in enumerate(dates):
        quotes = context.aligned_snapshot(day)
        if quotes is None:
            continue
        ret, consecutive = minute_log_returns(quotes['mid'], quotes.get('times'))
        flow = quotes['ofi'][1:]
        if len(ret) <= horizon:
            continue
        future = np.lib.stride_tricks.sliding_window_view(ret, horizon, axis=0)[1:]
        initial = ret[:-horizon]
        signed = np.sign(flow[:-horizon])
        continuous = np.lib.stride_tricks.sliding_window_view(
            consecutive, horizon + 1).all(axis=-1)
        valid = (continuous[:, None] & np.isfinite(future).all(axis=-1)
                 & np.isfinite(initial) & np.isfinite(signed) & (signed != 0))
        first, remaining = (signed * initial, signed * future.sum(axis=-1))
        ratio = np.divide(first + remaining,
            first,
            out=np.full_like(remaining, np.nan),
            where=valid & (np.abs(first) > 1e-12))
        daily_values[i] = np.nanmedian(ratio, axis=0)
    if stock_level:
        return pd.DataFrame(daily_values, index=dates, columns=context.ticks)
    weights = context.get_weight(domain, name, end)
    return weighted_series(daily_values, weights, dates, name='冲击保留率', signed=True)


def active_trade_imbalance(context, date, domain, name):
    """连续竞价时段主动买卖成交额之差/之和，排除集合竞价与午休。"""
    import polars as pl
    end = context.get_position(date)
    dates = context.dates[:end + 1]
    daily_values = np.full((len(dates), len(context.ticks)), np.nan)
    for i, day in enumerate(dates):
        ticks = pd.Index([str(t).zfill(6)[:6] for t in context.ticks])
        buy, sell = (np.zeros(len(ticks)), np.zeros(len(ticks)))
        observed = np.zeros(len(ticks), dtype=bool)
        frame = context.l2_table(day, 'cj', columns=[
            'TransactTime', 'SecurityID', 'Price', 'OrderQty', 'Side'])
        if frame is None or frame.is_empty():
            continue
        amount = pl.col('Price').cast(pl.Float64) * pl.col('OrderQty')
        frame = frame.group_by('SecurityID').agg(
            amount.filter(pl.col('Side') > 0).sum().alias('buy'),
            amount.filter(pl.col('Side') < 0).sum().alias('sell'))
        index = ticks.get_indexer(frame['SecurityID'].to_numpy())
        valid = index >= 0
        buy[index[valid]] = frame['buy'].to_numpy()[valid]
        sell[index[valid]] = frame['sell'].to_numpy()[valid]
        observed[index[valid]] = True
        result = np.divide(buy - sell, buy + sell, out=np.full(len(ticks), np.nan), where=buy + sell > 0)
        result[~observed] = np.nan
        daily_values[i] = result
    weights = context.get_weight(domain, name, end)
    return weighted_series(daily_values, weights, dates, name='主动成交不平衡', signed=True)


def daily_cancellation_rate(context, date, domain, name):
    """连续竞价逐股撤单金额/申报金额；撤单先按委托归并，单次关联并流式聚合。"""
    import polars as pl
    end = context.get_position(date)
    dates = context.dates[:end + 1]
    daily_values = np.full((len(dates), len(context.ticks)), np.nan)
    weights = context.get_weight(domain, name, end)
    tick_index = pd.Index([str(t).zfill(6)[:6] for t in context.ticks])
    for i, day in enumerate(dates):
        securities = tick_index[np.isfinite(weights[i]) & (weights[i] != 0)].tolist()
        summaries = []
        for exchange in ('sh', 'sz'):
            orders = context.l2_table(day, 'wt', lazy=True, exchanges=(exchange,), securities=securities, columns=[
                'TransactTime', 'SecurityID', 'ChannelNo', 'Side', 'ApplSeqNum', 'Price', 'OrderQty'])
            cancels = context.l2_table(day, 'cancel', lazy=True, exchanges=(exchange,), securities=securities, columns=[
                'TransactTime', 'SecurityID', 'ChannelNo', 'BidApplSeqNum', 'OfferApplSeqNum', 'OrderQty'])
            if orders is None or cancels is None:
                continue
            orders = orders.filter(
                (pl.col('Price') > 0) & (pl.col('OrderQty') > 0)
                & pl.col('SecurityID').is_in(tick_index.tolist()))
            keys = ['Exchange', 'ChannelNo', 'SecurityID', 'Side', 'ApplSeqNum']
            # 单次展开买卖委托号，先合并同一订单多次撤单，避免关联后重复计入申报金额。
            legs = cancels.unpivot(
                on=['BidApplSeqNum', 'OfferApplSeqNum'],
                index=['Exchange', 'ChannelNo', 'SecurityID', 'OrderQty'],
                variable_name='leg', value_name='ApplSeqNum'
            ).filter(pl.col('ApplSeqNum') != 0).with_columns(
                pl.when(pl.col('leg') == 'BidApplSeqNum').then(1).otherwise(-1).alias('Side')
            ).group_by(keys).agg(pl.col('OrderQty').sum().alias('cancel_qty'))
            summary = orders.join(legs, on=keys, how='left').group_by('SecurityID').agg(
                (pl.col('Price').cast(pl.Float64) * pl.col('OrderQty')).sum().alias('submitted'),
                (pl.col('Price').cast(pl.Float64) * pl.col('cancel_qty').fill_null(0)).sum().alias('cancelled')
            ).collect(engine='streaming')
            summaries.append(summary)
        if not summaries:
            continue
        summary = pl.concat(summaries).group_by('SecurityID').agg(
            pl.col('submitted').sum(), pl.col('cancelled').sum()
        ).select('SecurityID', (pl.col('cancelled') / pl.col('submitted')).alias('ratio'))
        ids = tick_index.get_indexer(summary['SecurityID'].to_numpy())
        daily_values[i, ids] = summary['ratio'].to_numpy()
    return weighted_series(daily_values, weights, dates, name='撤单率', signed=True)



def liquidity_coverage_ratio(context, date, domain, name, *, target_amount=None,
                             max_impact=0.005, impact_coefficient=1.0, window=20):
    """目标股票容量之和/目标交易金额之和，非市值加权平均。
    target_amount为日期×股票DataFrame（元，非负，0表示不交易），轴须明确覆盖。
    未提供需求、需求缺失或目标股票容量不可用时返回NaN；允许覆盖率超过1。
    域仅界定成员（含风格多空两侧），不将域权重再乘入元金额。
    """
    end = context.get_position(date)
    dates = context.dates[:end + 1]
    result = pd.Series(np.nan, index=dates, name='流动性覆盖率')
    if target_amount is None:
        return result
    demand = target_amount.reindex(index=dates, columns=context.ticks).to_numpy(dtype=float)
    capacity = executable_capacity(context, date, domain, name, stock_level=True,
        max_impact=max_impact, impact_coefficient=impact_coefficient, window=window).to_numpy()
    weights = context.get_weight(domain, name, end)
    # 域外需求置零，域内无效需求保留NaN，让普通sum自然传播缺失。
    demand = np.where(np.isfinite(demand) & (demand >= 0), demand, np.nan)
    demand = np.where(np.isfinite(weights) & (weights != 0), demand, 0)
    capacity = np.where(np.isfinite(capacity), capacity, np.nan)
    total = demand.sum(axis=1)
    available = np.where(demand > 0, capacity, 0).sum(axis=1)
    result[:] = np.divide(available, total, out=np.full(len(dates), np.nan),
                          where=total > 0)
    return result




MICRO_INDICATORS = {
    function.__name__: function
    for function in (
        effective_spread,
        five_level_depth_notional,
        depth_decline,
        order_flow_imbalance,
        order_flow_autocorrelation,
        best_level_imbalance,
        significant_jump_variance_ratio,
        acd_jump_intensity,
        standardized_acd_intensity,
        jump_volume_divergence,
        estimate_kyle_lambda,
        order_impact_cost,
        amihud_illiquidity,
        turnover_illiquidity,
        executable_capacity,
        impact_decay,
        active_trade_imbalance,
        daily_cancellation_rate,
    )
}


def update_micro(root=ROOT, domain='index', name='hs300', indicators=None, *, indicator_params=None):
    """计算注册指标，每项保存完整日期轴对齐的float32单列时序bin。

    路径沿用macro/style/risk：median_variables/monitor/micro/<指标名>.bin。
    每次运行对应一个域；indicators=None计算全部，也可传单个指标名或名称序列。
    indicator_params按指标名传关键字参数。
    """
    from v2.UpdateData.utils import ensure_memmap
    selected = (tuple(MICRO_INDICATORS) if indicators is None else
                (indicators,) if isinstance(indicators, str) else tuple(indicators))
    unknown = set(selected) - MICRO_INDICATORS.keys()
    if unknown:
        raise ValueError(f"未知micro指标: {', '.join(sorted(unknown))}")
    if not selected:
        return {}
    context = MonitorContext(root=root)
    indicator_params = indicator_params or {}
    try:
        axis_dates = context.data.axis.full_dates
        valid_dates = axis_dates[~np.isnat(axis_dates)]
        if not valid_dates.size:
            raise ValueError('日期轴没有有效交易日')
        start, end = (context.dates[0], context.dates[-1])
        output_root = Path(root) / 'median_variables' / 'monitor' / 'micro'
        outputs = {}
        for indicator in selected:
            values = MICRO_INDICATORS[indicator](context, end, domain, name,
                **indicator_params.get(indicator, {}))
            values = values.reindex(context.dates).to_numpy(dtype=np.float32)
            path = output_root / f'{indicator}.bin'
            array = ensure_memmap(path, (len(axis_dates), 1), dtype=np.float32)
            try:
                array[:] = np.nan
                array[:len(context.dates), 0] = values
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
