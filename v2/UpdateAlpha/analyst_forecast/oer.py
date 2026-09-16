"""Announcement-window excess return (OER) from Kaiyuan Securities.

For an announcement published on trading day T, the signal compounds the
stock return from the T-1 close through the T+1 close and subtracts the same
return for a configurable benchmark.  With date-only announcements, the
signal is available only after the reaction session closes.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from v2.UpdateAlpha.alphabase import AlphaBase, AlphaMeta
from v2.UpdateAlpha.analyst_forecast.aog import AOGConfig, AOGContext
from v2.UpdateData.config import ROOT


DEFAULT_ROOT = Path("Z:/") if Path("Z:/axis/dates.npy").is_file() else ROOT


@dataclass(frozen=True)
class OERConfig(AOGConfig):
    """OER uses periodic reports plus quantitative performance forecasts."""

    include_performance_forecasts: bool = True


class OERContext(AOGContext):
    def __init__(self, root=DEFAULT_ROOT, conn=None, config=OERConfig(),
                 announcements=None, universe="self"):
        super().__init__(
            root=root, conn=conn, config=config, announcements=announcements,
            universe=universe,
        )


class OERFactor(AlphaBase):
    """Close-to-close [T-1, T+1] return minus benchmark return."""

    meta = AlphaMeta(
        "oer", "announcement [T-1,T+1] cumulative excess return"
    )
    dependencies = (
        "LC_IncomeStatementAll", "LC_STIBIncomeState",
        "DZ_PerformanceForecast", "d_essentials/close_adj",
        "d_essentials/circ_mv", "basic/tradable",
    )

    def __init__(self, context):
        super().__init__(context)
        self._event_cache = OrderedDict()

    def _event_value(self, reaction_day):
        n = self.context.data.axis.tick_count
        if reaction_day < 2:
            return np.full(n, np.nan, dtype=np.float32)

        close = self.context.read_row("d_essentials/close_adj", reaction_day)
        start_close = self.context.read_row(
            "d_essentials/close_adj", reaction_day - 2
        )
        valid = (
            np.isfinite(close) & (close > 0)
            & np.isfinite(start_close) & (start_close > 0)
        )
        stock_return = np.divide(
            close, start_close, out=np.full(n, np.nan), where=valid
        ) - 1.0

        benchmark = self.context.calculate_benchmark(
            stock_return, reaction_day - 1
        )
        if not np.isfinite(benchmark):
            return np.full(n, np.nan, dtype=np.float32)
        excess = stock_return - benchmark
        return self.context.filter_factor_universe(
            excess, reaction_day
        ).astype(np.float32)

    def calculate(self, asof):
        axis, config = self.context.data.axis, self.context.config
        end = axis.date_position(pd.Timestamp(asof).date())
        frame = self.context.announcements(axis.trade_dates[end])
        out = np.full(axis.tick_count, np.nan, dtype=np.float32)
        if frame.empty:
            return out

        frame = frame.assign(
            reaction_day=np.searchsorted(
                axis.trade_dates,
                frame.publish_date.to_numpy(dtype="datetime64[D]"),
                side="left" if config.announcement_same_day else "right",
            ),
            position=frame.tick.map(axis._tick_positions),
        )
        selected = frame[
            (frame.reaction_day <= end) & frame.position.notna()
        ]
        if config.max_event_age is not None:
            selected = selected[
                end - selected.reaction_day <= config.max_event_age
            ]

        for day, group in selected.groupby("reaction_day", sort=False):
            day = int(day)
            if day not in self._event_cache:
                self._event_cache[day] = self._event_value(day)
                while len(self._event_cache) > config.event_cache_size:
                    self._event_cache.popitem(last=False)
            self._event_cache.move_to_end(day)
            positions = group.position.to_numpy(dtype=np.intp)
            out[positions] = self._event_cache[day][positions]
        return out


__all__ = ["OERConfig", "OERContext", "OERFactor"]
