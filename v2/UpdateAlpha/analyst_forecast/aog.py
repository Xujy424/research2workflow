"""Earnings-announcement price reactions (Orient Securities, 2025-04-22).

Daily close-time signals: sample on an announcement reaction day, then carry
until the next event (optionally expire). Date-only announcements default to
the strictly following session; set announcement_same_day=True only after
verifying the provider's pre-open publication-date convention.
The original AOG market return is a synthetic zzfull weighted return, not an
official index open. No early-session money-flow proxy is silently substituted.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import rankdata

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from v2.UpdateAlpha.alphabase import AlphaBase, AlphaContext, AlphaMeta
from v2.GetData import DataPool
from v2.UpdateData.config import ROOT, get_jy_conn

DEFAULT_ROOT = Path("Z:/") if Path("Z:/axis/dates.npy").is_file() else ROOT


@dataclass(frozen=True)
class AOGConfig:
    lookback_days: int = 20
    quantile: float = 0.2
    min_observations: int = 20
    max_event_age: int | None = None
    announcement_same_day: bool = False
    benchmark_field: str = "index/weight/zzfull_weight"
    cache_days: int = 64

    def __post_init__(self):
        if not 1 <= self.min_observations <= self.lookback_days:
            raise ValueError("require 1 <= min_observations <= lookback_days")
        if not 0 <= self.quantile <= 1 or self.cache_days < 1:
            raise ValueError("invalid quantile or cache_days")
        if self.max_event_age is not None and self.max_event_age < 0:
            raise ValueError("max_event_age must be non-negative")


def _rank(values):
    result = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if valid.any():
        result[valid] = rankdata(values[valid], method="average") / valid.sum()
    return result


def _event_statistics(current, history, config):
    """History contains pre-event DAILY CROSS-SECTIONAL percentile ranks."""
    count = np.isfinite(history).sum(axis=0)
    maximum = np.full(current.shape, np.nan)
    quantile = maximum.copy()
    for j in np.flatnonzero(count >= config.min_observations):
        valid = history[:, j][np.isfinite(history[:, j])]
        maximum[j] = valid.max()
        quantile[j] = np.quantile(valid, config.quantile)
    return current - maximum, quantile


class AOGContext(AlphaContext):
    """Accept an optional announcement DataFrame for offline reproduction.

    announcements columns: tick, end_date, publish_date. The earliest disclosure
    per tick/report period defines the event; subsequent restatements do not.
    Supplied future events are filtered at every requested asof.
    """

    def __init__(self, root=DEFAULT_ROOT, conn=None, config=AOGConfig(),
                 announcements=None):
        super().__init__(DataPool(root, asset="stock"))
        self.config = config
        self.conn = conn
        self._owns_conn = False
        self._supplied = announcements is not None
        self._events = (pd.DataFrame(columns=["tick", "end_date", "publish_date"])
                        if announcements is None else pd.DataFrame(announcements).copy())
        self._loaded_until = None
        self._daily_cache = OrderedDict()
        self._event_cache = OrderedDict()
        self._result_cache = {}

    def _announcements(self, asof):
        asof = pd.Timestamp(asof).normalize()
        if not self._supplied and (
            self._loaded_until is None or asof > self._loaded_until
        ):
            if self.conn is None:
                self.conn = get_jy_conn()
                self._owns_conn = True
            lower = ("1900-01-01" if self._loaded_until is None
                     else self._loaded_until.strftime("%Y-%m-%d"))
            upper = asof.strftime("%Y-%m-%d")
            parts = []
            for table in ("LC_IncomeStatementAll", "LC_STIBIncomeState"):
                bulletin = "AND f.BulletinType IN (20, 30)" if table == "LC_IncomeStatementAll" else ""
                parts.append(f"""
                    SELECT s.SecuCode AS tick, f.EndDate AS end_date,
                           MIN(f.InfoPublDate) AS publish_date
                    FROM dbo.{table} f
                    INNER JOIN dbo.SecuMain s ON f.CompanyCode=s.CompanyCode
                    WHERE f.InfoPublDate > '{lower}' AND f.InfoPublDate < DATEADD(day, 1, '{upper}')
                      AND f.IfMerged=1 AND f.IfAdjusted=2 AND f.IfComplete=1
                      {bulletin}
                      AND s.SecuCategory=1 AND s.SecuMarket IN (83, 90)
                    GROUP BY s.SecuCode, f.EndDate
                """)
            frame = pl.read_database(" UNION ALL ".join(parts), self.conn,
                                     infer_schema_length=None).to_pandas()
            self._events = pd.concat([self._events, frame], ignore_index=True)
            self._loaded_until = asof
        frame = self._events.copy()
        frame["tick"] = frame["tick"].astype(str).str.strip().str.zfill(6)
        for col in ("publish_date", "end_date"):
            frame[col] = pd.to_datetime(frame[col]).dt.normalize()
        frame = frame[frame.publish_date <= asof].dropna()
        return (frame.sort_values("publish_date")
                .drop_duplicates(["tick", "end_date"], keep="first"))

    def daily_feature(self, day, feature):
        key = (day, feature)
        if key in self._daily_cache:
            self._daily_cache.move_to_end(key)
            return self._daily_cache[key]
        n = self.data.axis.tick_count
        excess = np.full(n, np.nan)
        ranks = excess.copy()
        if day > 0:
            price = np.asarray(self.data.read(f"d_essentials/{feature}_adj", day), float)
            previous = np.asarray(self.data.read("d_essentials/close_adj", day - 1), float)
            valid = (np.isfinite(price) & np.isfinite(previous)
                     & (price > 0) & (previous > 0))
            # Exclude non-trading observations from daily ranking.
            amount = self.data.read("d_essentials/amount", day)
            valid &= np.isfinite(amount) & (amount > 0)
            raw = np.divide(price, previous, out=np.full(n, np.nan), where=valid) - 1
            ranks = _rank(raw)
            # Prior-session weights avoid using closing weights in the gap.
            weights = np.asarray(self.data.read(self.config.benchmark_field, day - 1), float)
            eligible = valid & np.isfinite(weights) & (weights > 0)
            if eligible.any():
                market = np.average(raw[eligible], weights=weights[eligible])
                excess = raw - market
        self._daily_cache[key] = (excess, ranks)
        while len(self._daily_cache) > self.config.cache_days:
            self._daily_cache.popitem(last=False)
        return excess, ranks

    def values(self, asof, feature="open"):
        end = self.data.axis.date_position(pd.Timestamp(asof).date())
        cache_key = (end, feature)
        if cache_key in self._result_cache:
            return self._result_cache[cache_key]
        dates = self.data.axis.trade_dates
        frame = self._announcements(dates[end])
        event_days = np.searchsorted(
            dates, frame.publish_date.to_numpy(dtype="datetime64[D]"),
            side="left" if self.config.announcement_same_day else "right",
        )
        frame = frame.assign(event_day=event_days)
        frame = (frame[frame.event_day <= end].sort_values(["event_day", "end_date"])
                 .drop_duplicates("tick", keep="last"))
        outputs = {name: np.full(self.data.axis.tick_count, np.nan)
                   for name in ("raw", "rank", "demax", "quantile")}
        for event_day, group in frame.groupby("event_day"):
            event_day = int(event_day)
            if (self.config.max_event_age is not None
                    and end - event_day > self.config.max_event_age):
                continue
            event_key = (event_day, feature)
            if event_key not in self._event_cache:
                excess, current = self.daily_feature(event_day, feature)
                days = range(max(0, event_day - self.config.lookback_days), event_day)
                history = [self.daily_feature(day, feature)[1] for day in days]
                history = np.stack(history) if history else np.empty((0, len(current)))
                demax, quantile = _event_statistics(current, history, self.config)
                self._event_cache[event_key] = (excess, current, demax, quantile)
                while len(self._event_cache) > 512:
                    self._event_cache.popitem(last=False)
            self._event_cache.move_to_end(event_key)
            excess, current, demax, quantile = self._event_cache[event_key]
            for tick in group.tick:
                j = self.data.axis._tick_positions.get(tick)
                if j is None:
                    continue
                for name, value in zip(outputs, (excess, current, demax, quantile)):
                    outputs[name][j] = value[j]
        # One asof retained; repeated families share work without unbounded RAM.
        self._result_cache = {cache_key: outputs}
        return outputs


class _AOGFactor(AlphaBase):
    feature = "open"
    statistic = "raw"
    dependencies = ("LC_IncomeStatementAll", "LC_STIBIncomeState",
                    "d_essentials/open_adj", "d_essentials/close_adj",
                    "d_essentials/amount", "index/weight/zzfull_weight")

    def calculate(self, asof):
        return self.context.values(asof, self.feature)[self.statistic].astype(np.float32)


class AOGFactor(_AOGFactor):
    meta = AlphaMeta("aog", "announcement open gap minus synthetic market gap")


class AOGRankFactor(_AOGFactor):
    meta = AlphaMeta("aog_rank", "announcement open-gap cross-sectional percentile")
    statistic = "rank"


class AOGDemaxFactor(_AOGFactor):
    meta = AlphaMeta("aog_rank_demax_20d", "event rank minus pre-event 20D maximum")
    statistic = "demax"


class AOGQuantileFactor(_AOGFactor):
    meta = AlphaMeta("aog_rank_pre_quantile_20_20d", "pre-event 20D rank 20th percentile")
    statistic = "quantile"


class AOGLowDemaxFactor(AOGDemaxFactor):
    meta = AlphaMeta("aog_low_rank_demax_20d", "announcement low-return rank DEMAX")
    feature = "low"
    dependencies = _AOGFactor.dependencies + ("d_essentials/low_adj",)


class AOGLowQuantileFactor(AOGQuantileFactor):
    meta = AlphaMeta("aog_low_rank_pre_quantile_20_20d", "pre-event low-return rank quantile")
    feature = "low"
    dependencies = AOGLowDemaxFactor.dependencies


AOG_FACTORS = (AOGFactor, AOGRankFactor, AOGDemaxFactor, AOGQuantileFactor,
               AOGLowDemaxFactor, AOGLowQuantileFactor)
