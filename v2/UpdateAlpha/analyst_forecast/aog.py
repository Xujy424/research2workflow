"""Earnings-announcement price reactions (Orient Securities, 2025-04-22).

Daily close-time signals: sample on an announcement reaction day, then carry
until the next event (optionally expire). Date-only announcements default to
the strictly following session; set announcement_same_day=True only after
verifying the provider's pre-open publication-date convention.
The default market return is synthetic zzfull. With benchmark_field=None,
use current-day tradable stocks weighted by prior-session market value.
Signals are available at day close; tradable/amount can contain intraday data.
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
    announcement_lookback_days: int = 3 * 365
    decay_half_life_days: float = 120.0
    decay_max_events: int = 8
    event_cache_size: int = 2048
    max_event_age: int | None = None
    announcement_same_day: bool = False
    benchmark_field: str | None = "index/weight/zzfull_weight"
    market_value_field: str = "d_essentials/circ_mv"

    def __post_init__(self):
        if not 1 <= self.min_observations <= self.lookback_days:
            raise ValueError("require 1 <= min_observations <= lookback_days")
        if not 0 <= self.quantile <= 1:
            raise ValueError("quantile must be between zero and one")
        if self.announcement_lookback_days < 1:
            raise ValueError("announcement_lookback_days must be positive")
        if self.decay_half_life_days <= 0 or self.decay_max_events < 1:
            raise ValueError("decay parameters must be positive")
        if self.event_cache_size < 1:
            raise ValueError("event_cache_size must be positive")
        if self.max_event_age is not None and self.max_event_age < 0:
            raise ValueError("max_event_age must be non-negative")


def _rank(values):
    result = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if valid.any():
        result[valid] = rankdata(values[valid], method="average") / valid.sum()
    return result


class AOGContext(AlphaContext):
    """Read point-in-time latest announcements and local market rows."""

    def __init__(self, root=DEFAULT_ROOT, conn=None, config=AOGConfig(),
                 announcements=None):
        super().__init__(DataPool(root, asset="stock"))
        self.config, self.conn = config, conn
        self._owns_conn = False
        self._supplied = (
            None if announcements is None else pd.DataFrame(announcements).copy()
        )
        self._announcement_date = None
        self._announcement_frame = None

    def announcements(self, asof):
        history = self.announcement_history(asof)
        return history.drop_duplicates("tick", keep="last")

    def announcement_history(self, asof):
        asof = pd.Timestamp(asof).normalize()
        cutoff = (
            asof if self.config.announcement_same_day
            else asof - pd.Timedelta(days=1)
        )
        if cutoff == self._announcement_date:
            return self._announcement_frame
        start = cutoff - pd.Timedelta(
            days=self.config.announcement_lookback_days
        )
        if self._supplied is None:
            if self.conn is None:
                self.conn = get_jy_conn()
                self._owns_conn = True
            previous = pd.DataFrame()
            query_start = start
            if self._announcement_date is not None and cutoff > self._announcement_date:
                previous = self._announcement_frame
                query_start = self._announcement_date + pd.Timedelta(days=1)
            parts = []
            for table in ("LC_IncomeStatementAll", "LC_STIBIncomeState"):
                bulletin = "AND f.BulletinType IN (20, 30)" if table == "LC_IncomeStatementAll" else ""
                parts.append(f"""
                    SELECT DISTINCT
                        s.SecuCode AS tick, 
                        f.EndDate AS end_date,
                        f.InfoPublDate AS publish_date
                    FROM dbo.{table} f
                    INNER JOIN dbo.SecuMain s ON f.CompanyCode=s.CompanyCode
                    WHERE f.InfoPublDate >= '{query_start:%Y-%m-%d}'
                      AND f.InfoPublDate < DATEADD(day, 1, '{cutoff:%Y-%m-%d}')
                      AND f.IfMerged=1 AND f.IfAdjusted=2 AND f.IfComplete=1
                      {bulletin}
                      AND s.SecuCategory=1 AND s.SecuMarket IN (83, 90)
                """)
            union = " UNION ALL ".join(parts)
            sql = f"""
                WITH announcements AS ({union}), ranked AS (
                    SELECT tick, end_date, publish_date,
                           ROW_NUMBER() OVER (
                               PARTITION BY tick
                               ORDER BY end_date DESC, publish_date DESC
                           ) AS row_number
                    FROM announcements
                )
                SELECT tick, end_date, publish_date
                FROM ranked WHERE row_number <= {self.config.decay_max_events}
            """
            frame = pl.read_database(sql, self.conn, infer_schema_length=None).to_pandas()
            if frame.empty:
                frame = previous
            elif not previous.empty:
                frame = pd.concat([previous, frame], ignore_index=True)
        else:
            frame = self._supplied
        result = self._history(frame, start, cutoff, self.config.decay_max_events)
        self._announcement_date = cutoff
        self._announcement_frame = result
        return result

    @staticmethod
    def _history(frame, start, asof, max_events):
        if frame.empty:
            return pd.DataFrame(columns=["tick", "end_date", "publish_date"])
        frame = frame[["tick", "end_date", "publish_date"]].dropna().copy()
        frame["tick"] = frame["tick"].astype(str).str.strip().str.zfill(6)
        frame["end_date"] = pd.to_datetime(frame["end_date"]).dt.normalize()
        frame["publish_date"] = pd.to_datetime(frame["publish_date"]).dt.normalize()
        frame = frame[frame.publish_date.between(start, asof)]
        return (
            frame.drop_duplicates(["tick", "end_date", "publish_date"])
            .sort_values(["tick", "end_date", "publish_date"])
            .groupby("tick", group_keys=False)
            .tail(max_events)
        )

    def read_row(self, field, day):
        return np.asarray(self.data.read(field, int(day)), dtype=float)


class _AOGFactor(AlphaBase):
    feature = "open"
    dependencies = ("LC_IncomeStatementAll", "LC_STIBIncomeState",
                    "d_essentials/open_adj", "d_essentials/close_adj",
                    "d_essentials/amount")

    def __init__(self, context):
        super().__init__(context)
        self._event_cache = OrderedDict()

    def _raw(self, day):
        n = self.context.data.axis.tick_count
        if day <= 0:
            return np.full(n, np.nan)
        read = self.context.read_row
        price = read(f"d_essentials/{self.feature}_adj", day)
        previous = read("d_essentials/close_adj", day - 1)
        amount = read("d_essentials/amount", day)
        valid = (np.isfinite(price) & (price > 0) & np.isfinite(previous)
                 & (previous > 0) & np.isfinite(amount) & (amount > 0))
        if self.context.config.benchmark_field is None:
            tradable = read("basic/tradable", day)
            valid &= np.isfinite(tradable) & (tradable == 1)
        return np.divide(price, previous, out=np.full(n, np.nan), where=valid) - 1

    def _daily_value(self, day, daily=None):
        raise NotImplementedError

    def _event_value(self, day, daily=None):
        return self._daily_value(day, daily)

    def calculate(self, asof):
        axis, config = self.context.data.axis, self.context.config
        end = axis.date_position(pd.Timestamp(asof).date())
        frame = self.context.announcements(axis.trade_dates[end])
        out = np.full(axis.tick_count, np.nan, dtype=np.float32)
        if frame.empty:
            return out
        frame = frame.assign(
            event_day=np.searchsorted(
                axis.trade_dates,
                frame.publish_date.to_numpy(dtype="datetime64[D]"),
                side="left" if config.announcement_same_day else "right",
            ),
            position=frame.tick.map(axis._tick_positions),
        )
        selected = frame[
            (frame.event_day <= end) & frame.position.notna()
        ]
        if config.max_event_age is not None:
            selected = selected[end - selected.event_day <= config.max_event_age]
        daily = {}
        for day, group in selected.groupby("event_day", sort=False):
            day = int(day)
            positions = group.position.to_numpy(dtype=np.intp)
            if day not in self._event_cache:
                self._event_cache[day] = self._event_value(day, daily)
                while len(self._event_cache) > config.event_cache_size:
                    self._event_cache.popitem(last=False)
            self._event_cache.move_to_end(day)
            values = self._event_cache[day]
            out[positions] = values[positions]
        return out


class AOGFactor(_AOGFactor):
    meta = AlphaMeta("aog", "announcement open gap minus synthetic market gap")

    def _daily_value(self, day, daily=None):
        raw = self._raw(day)
        if day <= 0:
            return raw
        config = self.context.config
        field = config.market_value_field if config.benchmark_field is None else config.benchmark_field
        weights = self.context.read_row(field, day - 1)
        eligible = np.isfinite(raw) & np.isfinite(weights) & (weights > 0)
        if not eligible.any():
            return np.full(raw.shape, np.nan)
        return raw - np.average(raw[eligible], weights=weights[eligible])


class AOGRankFactor(_AOGFactor):
    meta = AlphaMeta("aog_rank", "announcement open-gap cross-sectional percentile")

    def _daily_value(self, day, daily=None):
        key = (self.feature, day)
        if daily is None:
            return _rank(self._raw(day))
        if key not in daily:
            daily[key] = _rank(self._raw(day))
        return daily[key]


class _AOGWindowFactor(AOGRankFactor):
    def _event_value(self, day, daily=None):
        config = self.context.config
        days = range(max(0, day - config.lookback_days), day)
        rows = [self._daily_value(t, daily) for t in days]
        n = self.context.data.axis.tick_count
        history = np.stack(rows) if rows else np.empty((0, n))
        eligible = np.isfinite(history).sum(axis=0) >= config.min_observations
        result = np.full(n, np.nan)
        if eligible.any():
            result[eligible] = self._reduce(history[:, eligible])
        return self._finish(day, result, daily)

    def _finish(self, day, result, daily=None):
        return result


class AOGDemaxFactor(_AOGWindowFactor):
    meta = AlphaMeta("aog_rank_demax_20d", "event rank minus pre-event 20D maximum")

    def _reduce(self, history):
        return np.nanmax(history, axis=0)

    def _finish(self, day, result, daily=None):
        return self._daily_value(day, daily) - result



class AOGQuantileFactor(_AOGWindowFactor):
    meta = AlphaMeta("aog_rank_pre_quantile_20_20d", "pre-event 20D rank 20th percentile")

    def _reduce(self, history):
        return np.nanquantile(history, self.context.config.quantile, axis=0)


class _AOGDecayFactor:
    """Exponentially weight several past announcement-event observations."""

    def calculate(self, asof):
        axis, config = self.context.data.axis, self.context.config
        end = axis.date_position(pd.Timestamp(asof).date())
        frame = self.context.announcement_history(axis.trade_dates[end])
        out = np.full(axis.tick_count, np.nan, dtype=np.float32)
        if frame.empty:
            return out
        frame = frame.assign(
            event_day=np.searchsorted(
                axis.trade_dates,
                frame.publish_date.to_numpy(dtype="datetime64[D]"),
                side="left" if config.announcement_same_day else "right",
            ),
            position=frame.tick.map(axis._tick_positions),
        )
        frame = frame[
            (frame.event_day <= end) & frame.position.notna()
        ].copy()
        if config.max_event_age is not None:
            frame = frame[end - frame.event_day <= config.max_event_age]
        if frame.empty:
            return out
        frame = frame.reset_index(drop=True)

        event_signal = np.full(len(frame), np.nan)
        daily = {}
        for day, rows in frame.groupby("event_day", sort=False):
            day = int(day)
            if day not in self._event_cache:
                self._event_cache[day] = self._event_value(day, daily)
                while len(self._event_cache) > config.event_cache_size:
                    self._event_cache.popitem(last=False)
            self._event_cache.move_to_end(day)
            positions = rows.position.to_numpy(dtype=np.intp)
            event_signal[rows.index.to_numpy()] = self._event_cache[day][positions]

        positions = frame.position.to_numpy(dtype=np.intp)
        weights = np.exp2(
            -(end - frame.event_day.to_numpy()) / config.decay_half_life_days
        )
        valid = np.isfinite(event_signal) & np.isfinite(weights)
        numerator = np.bincount(
            positions[valid],
            weights=event_signal[valid] * weights[valid],
            minlength=axis.tick_count,
        )
        denominator = np.bincount(
            positions[valid], weights=weights[valid], minlength=axis.tick_count
        )
        np.divide(
            numerator, denominator, out=out, where=denominator > 0
        )
        return out


class AOGDemaxDecayFactor(_AOGDecayFactor, AOGDemaxFactor):
    meta = AlphaMeta(
        "aog_rank_demax_20d_decay",
        "time-weighted DEMAX across recent announcement events",
    )



class AOGQuantileDecayFactor(_AOGDecayFactor, AOGQuantileFactor):
    meta = AlphaMeta(
        "aog_rank_pre_quantile_20_20d_decay",
        "time-weighted pre-event rank quantile across announcements",
    )


class AOGLowDemaxFactor(AOGDemaxFactor):
    meta = AlphaMeta("aog_low_rank_demax_20d", "announcement low-return rank DEMAX")
    feature = "low"
    dependencies = _AOGFactor.dependencies + ("d_essentials/low_adj",)



class AOGLowQuantileFactor(AOGQuantileFactor):
    meta = AlphaMeta("aog_low_rank_pre_quantile_20_20d", "pre-event low-return rank quantile")
    feature = "low"
    dependencies = AOGLowDemaxFactor.dependencies


AOG_FACTORS = (
    AOGFactor, AOGRankFactor,
    AOGDemaxFactor, AOGQuantileFactor,
    AOGDemaxDecayFactor, AOGQuantileDecayFactor,
    AOGLowDemaxFactor, AOGLowQuantileFactor,
)




