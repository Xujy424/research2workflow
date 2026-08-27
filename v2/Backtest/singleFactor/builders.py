"""Pure signal-to-target-weight transformations."""

from __future__ import annotations
import numpy as np
from scipy.stats import rankdata
from .config import (ActiveSide, EventConfig, EventTrigger, PortfolioConfig,
                     SignalInput, Weighting)


def _allocate(score, selected, weighting):
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
        long = ok & (ranks > size * (config.quantiles - config.top_groups))
        lw = _allocate(signal[t], long, config.weighting)
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
    total = np.sum(weight, axis=1, keepdims=True)
    return np.divide(weight, total, out=np.zeros_like(weight), where=total > 0)


def _benchmark_sleeve(signal, tradable, industry, benchmark, config, side):
    """Build a unit long or short-name sleeve inside benchmark constituents."""
    sleeve = np.zeros_like(signal, dtype=float)
    for t in range(len(signal)):
        valid = tradable[t] & (benchmark[t] > 0) & np.isfinite(signal[t])
        groups = ([None] if not config.industry_align else
                  np.unique(industry[t, (benchmark[t] > 0)
                                     & np.isfinite(industry[t])]))
        for code in groups:
            member = valid if code is None else valid & (industry[t] == code)
            positions = np.flatnonzero(member)
            benchmark_member = ((benchmark[t] > 0) if code is None else
                                (benchmark[t] > 0) & (industry[t] == code))
            if not len(positions):
                # No usable factor name: hold the benchmark in this industry,
                # which contributes exactly zero active industry exposure.
                sleeve[t, benchmark_member] += benchmark[t, benchmark_member]
                continue
            budget = (1.0 if code is None else
                      benchmark[t, industry[t] == code].sum())
            if config.signal_input == SignalInput.PREBUILT_WEIGHT:
                selected = positions[np.abs(signal[t, positions]) > 0]
            else:
                count = max(1, int(np.ceil(
                    len(positions) * (config.top_groups if side == ActiveSide.LONG
                                      else config.bottom_groups)
                    / config.quantiles)))
                order = np.argsort(signal[t, positions], kind="stable")
                selected = positions[order[-count:] if side == ActiveSide.LONG
                                     else order[:count]]
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
    """Return comparable active weights and their benchmark/total components."""
    benchmark = _normalise_rows(np.where(
        np.isfinite(benchmark) & (benchmark > 0), benchmark, 0.0))
    if config.industry_align and industry is None:
        raise ValueError("industry_align requires industry")
    long_sleeve = short_sleeve = np.zeros_like(signal, dtype=float)
    if config.active_side in {ActiveSide.LONG, ActiveSide.LONG_SHORT}:
        long_sleeve = _benchmark_sleeve(
            signal, tradable, industry, benchmark, config, ActiveSide.LONG)
    if config.active_side in {ActiveSide.SHORT, ActiveSide.LONG_SHORT}:
        short_sleeve = _benchmark_sleeve(
            signal, tradable, industry, benchmark, config, ActiveSide.SHORT)
    if config.active_side == ActiveSide.LONG:
        active = long_sleeve - benchmark
    elif config.active_side == ActiveSide.SHORT:
        active = benchmark - short_sleeve
    else:
        active = 0.5 * (long_sleeve - short_sleeve)
    gross = np.abs(active).sum(axis=1, keepdims=True)
    active = np.divide(
        active * config.active_gross, gross,
        out=np.zeros_like(active), where=gross > 0)
    return {
        "active": active,
        "benchmark": benchmark,
        "total_portfolio": benchmark + active,
        "long_sleeve": long_sleeve,
        "short_sleeve": short_sleeve,
    }


def event_triggers(signal, tradable, event: EventConfig):
    if event.trigger == EventTrigger.NONZERO:
        trigger = np.isfinite(signal) & (np.abs(signal) > event.threshold)
    elif event.trigger == EventTrigger.CHANGE:
        trigger = np.vstack((np.zeros((1, signal.shape[1]), bool),
                             np.abs(np.diff(signal, axis=0)) > event.threshold))
    else:
        above = signal > event.threshold
        trigger = np.vstack((np.zeros((1, signal.shape[1]), bool), above[1:] & ~above[:-1]))
    return trigger & tradable


def event_weights(signal, tradable, industry, event: EventConfig):
    trigger = event_triggers(signal, tradable, event)
    out, last = np.zeros_like(signal), np.full(signal.shape[1], -10**9)
    active = []
    for t in range(len(signal)):
        fresh = np.flatnonzero(trigger[t] & (t-last > event.cooldown_days))
        last[fresh] = t
        active.extend((j, t + event.holding_days, signal[t, j]) for j in fresh)
        active = [x for x in active if x[1] > t]
        for j, _, value in active:
            out[t, j] += np.sign(value)
        if event.cross_sectionalize and event.pair_within_industry:
            if industry is None:
                raise ValueError("pair_within_industry requires industry")
            for code in np.unique(industry[t, np.isfinite(industry[t])]):
                peer = tradable[t] & (industry[t] == code)
                if peer.any():
                    out[t, peer] -= np.mean(out[t, peer])
        elif event.cross_sectionalize:
            out[t] -= np.mean(out[t, tradable[t]]) if tradable[t].any() else 0
        gross = np.abs(out[t]).sum()
        if gross > 0:
            out[t] /= gross
    return out
