"""Pure signal-to-target-weight transformations."""

from __future__ import annotations
import numpy as np
from scipy.stats import rankdata
from .config import (ActiveSide, EventConfig, EventTrigger, PortfolioConfig,
                     SignalInput, Weighting)


def _allocate(score, selected, weighting):
    '''等权或按因子值作权重后归一化'''
    if not selected.any():
        return np.zeros_like(score)
    raw = np.ones_like(score) if weighting == Weighting.EQUAL else np.abs(score)
    raw = np.where(selected, raw, 0.0)
    return raw / raw.sum()


def quantile_weights(signal, tradable, config: PortfolioConfig, long_only=False):
    out = np.zeros_like(signal, dtype=float)
    for t in range(len(signal)):
        ok = tradable[t] & np.isfinite(signal[t])
        if ok.sum() < config.quantiles:
            continue
        ranks = np.zeros(signal.shape[1]); ranks[ok] = rankdata(signal[t, ok])
        size = ok.sum() / config.quantiles
        top = ok & (ranks > size * (config.quantiles - config.top_groups))
        lw = _allocate(signal[t], top, config.weighting)
        if long_only:
            out[t] = lw * config.gross_exposure
            continue
        short = ok & (ranks <= size * config.bottom_groups)
        sw = _allocate(signal[t], short, config.weighting)
        out[t] = .5 * config.gross_exposure * (lw - sw)
    return out


def short_only_weights(signal, tradable, config: PortfolioConfig):
    """Short only the low-score tail; high scores carry no position."""
    return -quantile_weights(-signal, tradable, config, long_only=True)


def industry_aligned_long(signal, tradable, industry, benchmark, config):
    benchmark = np.nan_to_num(benchmark, nan=0.0).clip(min=0)
    benchmark /= np.where(benchmark.sum(1, keepdims=True) > 0,
                          benchmark.sum(1, keepdims=True), 1)
    out = np.zeros_like(signal)
    for t in range(len(signal)):
        for code in np.unique(industry[t, np.isfinite(industry[t])]):
            member = industry[t] == code
            budget = benchmark[t, member].sum()
            eligible = member & (benchmark[t] > 0) & tradable[t] & np.isfinite(signal[t])
            if budget <= 0 or eligible.sum() == 0:
                continue
            ranks = rankdata(signal[t, eligible])
            selected = np.flatnonzero(eligible)[ranks > eligible.sum() * (1-config.top_groups/config.quantiles)]
            if not len(selected):
                selected = np.array([np.flatnonzero(eligible)[np.argmax(signal[t, eligible])]])
            local = _allocate(signal[t], np.isin(np.arange(signal.shape[1]), selected), config.weighting)
            out[t] += budget * local
    return out, benchmark


def _normalise_rows(weight):
    '''权重归一化'''
    total = np.sum(weight, axis=1, keepdims=True)
    return np.divide(weight, total, out=np.zeros_like(weight), where=total > 0)


def _benchmark_sleeve(signal, tradable, industry, benchmark, config, side):
    """计算行业配比权重."""
    sleeve = np.zeros_like(signal, dtype=float)
    for t in range(len(signal)):
        valid = tradable[t] & (benchmark[t] > 0) & np.isfinite(signal[t])
        groups = ([None] if not config.industry_align else   # 获取第t天，基准指数成分股中所有有效的行业编号，并去重
                  np.unique(industry[t, (benchmark[t] > 0) & np.isfinite(industry[t])]))
        for code in groups:
            # 基准成分、当前行业、可交易、因子有效
            member = valid if code is None else valid & (industry[t] == code)
            positions = np.flatnonzero(member)
            # 基准成分、当前行业
            benchmark_member = ((benchmark[t] > 0) if code is None else
                                (benchmark[t] > 0) & (industry[t] == code))
            if not len(positions):
                sleeve[t, benchmark_member] += benchmark[t, benchmark_member]
                continue
            # 当前行业的资金预算，如果不做行业对齐，所有股票作为一个整体处理
            budget = (1.0 if code is None else benchmark[t, industry[t] == code].sum())
            if config.signal_input == SignalInput.PREBUILT_WEIGHT:
                selected = positions[np.abs(signal[t, positions]) > 0]
            else:
                count = max(1, int(np.ceil(
                    len(positions) * (config.top_groups if side == ActiveSide.LONG else config.bottom_groups) / config.quantiles
                    )))
                order = np.argsort(signal[t, positions], kind="stable")
                selected = positions[order[-count:] if side == ActiveSide.LONG else order[:count]]
            if not len(selected) or budget <= 0:
                sleeve[t, benchmark_member] += benchmark[t, benchmark_member]
                continue
            raw = (np.ones(len(selected)) if config.weighting == Weighting.EQUAL
                   else np.abs(signal[t, selected]))
            if raw.sum() <= 0:
                raw = np.ones(len(selected))
            sleeve[t, selected] += budget * raw / raw.sum()
    return _normalise_rows(sleeve)


def benchmark_active_weights(signal, tradable, industry, benchmark, config):
    """计算主动权重及基准权重."""
    benchmark = _normalise_rows(
        np.where(np.isfinite(benchmark) & (benchmark > 0), benchmark, 0.0)
    )
    if config.industry_align and industry is None:
        raise ValueError("industry_align requires industry")
    long_sleeve = short_sleeve = np.zeros_like(signal, dtype=float)
    if config.active_side in {ActiveSide.LONG, ActiveSide.LONG_SHORT}:
        long_sleeve = _benchmark_sleeve(
            signal, tradable, industry, benchmark, config, ActiveSide.LONG
        )
    if config.active_side in {ActiveSide.SHORT, ActiveSide.LONG_SHORT}:
        short_sleeve = _benchmark_sleeve(
            signal, tradable, industry, benchmark, config, ActiveSide.SHORT
        )
    if config.active_side == ActiveSide.LONG:
        active = long_sleeve - benchmark
    elif config.active_side == ActiveSide.SHORT:
        active = benchmark - short_sleeve
    else:
        active = 0.5 * (long_sleeve - short_sleeve)
    # 每天主动权重归一化，让组合权重和基准权重每天可横向比较
    #如果不归一化，回测收益变化既来自因子选股能力，也来自每天不同的主动仓位规模，无法公平比较
    gross = np.abs(active).sum(axis=1, keepdims=True)
    active = np.divide(
        active * config.active_gross, gross,
        out=np.zeros_like(active), where=gross > 0)
    return {
        "active": active,
        "benchmark": benchmark,
        "portfolio": benchmark + active,
        "long_sleeve": long_sleeve,
        "short_sleeve": short_sleeve,
    }


def event_triggers(signal, tradable, event: EventConfig):
    '''标记事件触发'''
    if event.trigger == EventTrigger.NONZERO:
        trigger = np.isfinite(signal) & (np.abs(signal) > event.threshold)
    elif event.trigger == EventTrigger.CHANGE:  # 双向变化程度大于阈值
        trigger = np.vstack((np.zeros((1, signal.shape[1]), bool),
                             np.abs(np.diff(signal, axis=0)) > event.threshold))
    else:
        finite_pair = np.isfinite(signal[1:]) & np.isfinite(signal[:-1])
        above = signal > event.threshold
        crossed = finite_pair & (above[1:] != above[:-1])
        trigger = np.vstack((
            np.zeros((1, signal.shape[1]), dtype=bool),
            crossed,
        ))
    return trigger & tradable


def event_directions(signal, event: EventConfig):
    """Return the economic direction implied by each potential event.

    NONZERO is a level event. CHANGE and CROSS are transition events, so their
    direction comes from the change rather than the current signal level.
    Undefined or zero directions remain zero and must not be treated as
    positive events.
    """
    if event.trigger == EventTrigger.NONZERO:
        direction = np.sign(signal)
    else:
        change = np.vstack((
            np.full((1, signal.shape[1]), np.nan),
            np.diff(signal, axis=0),
        ))
        direction = np.sign(change)
    return np.where(np.isfinite(direction), direction, 0.0)


def event_weights(signal, tradable, industry, event: EventConfig):
    '''事件多空纯Alpha'''
    trigger = event_triggers(signal, tradable, event)
    direction = event_directions(signal, event)
    trigger &= direction != 0
    out, last = np.zeros_like(signal), np.full(signal.shape[1], -10**9)
    active = []
    for t in range(len(signal)):
        fresh = np.flatnonzero(trigger[t] & (t-last > event.cooldown_days))
        last[fresh] = t
        active.extend(
            (j, t + event.holding_days, direction[t, j]) for j in fresh
        )
        active = [x for x in active if x[1] > t]
        for j, _, value in active:
            out[t, j] += np.sign(value)
        if event.cross_sectionalize and event.neutralize_within_industry:
            if industry is None:
                raise ValueError("neutralize_within_industry requires industry")
            for code in np.unique(industry[t, np.isfinite(industry[t])]):
                peer = tradable[t] & (industry[t] == code)
                if peer.any():
                    out[t, peer] -= np.mean(out[t, peer])
        elif event.cross_sectionalize:
            if tradable[t].any():
                out[t, tradable[t]] -= np.mean(out[t, tradable[t]])
        gross = np.abs(out[t]).sum()
        if gross > 0:
            out[t] /= gross  # 多空仓位0.5，净敞口0，总杠杆1
    return out


def event_equal_weight_components(signal, tradable, event: EventConfig, benchmark=None):
    """真实持有事件股票的纯多头组合."""
    if benchmark is None:
        benchmark = _normalise_rows(tradable.astype(float))
    else:
        benchmark = _normalise_rows(
            np.where(np.isfinite(benchmark) & (benchmark > 0), benchmark, 0.0)
        )
    universe = tradable & (benchmark > 0)
    trigger = event_triggers(signal, universe, event)
    portfolio = np.zeros_like(signal, dtype=float)
    last = np.full(signal.shape[1], -10**9)
    active = []
    for t in range(len(signal)):
        # 0/1 条件数组中值为 True 的位置索引
        fresh = np.flatnonzero(trigger[t] & (t - last > event.cooldown_days))
        last[fresh] = t
        active.extend((j, t + event.holding_days) for j in fresh)
        active = [item for item in active if item[1] > t]
        selected = np.unique(np.asarray([j for j, _ in active], dtype=int))
        selected = selected[universe[t, selected]]
        if len(selected):
            portfolio[t, selected] = 1.0 / len(selected)
    active_weight = portfolio - benchmark
    return {
        "active": active_weight,
        "benchmark": benchmark,
        "portfolio": portfolio,
        # Backward-compatible alias for callers using the old key.
        "total_portfolio": portfolio,
        "event_portfolio": portfolio,
    }
