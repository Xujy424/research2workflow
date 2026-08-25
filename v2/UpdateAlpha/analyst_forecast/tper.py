"""Point-in-time target-price level and revision factors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import polars as pl

if __package__:
    from ..alphabase import AlphaBase, AlphaContext, AlphaMeta
    from ...GetData import DataPool
    from ...UpdateData.config import ROOT, get_zyyx_conn
    from .utils import _date, aggregate
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from v2.UpdateAlpha.alphabase import AlphaBase, AlphaContext, AlphaMeta
    from v2.GetData import DataPool
    from v2.UpdateData.config import ROOT, get_zyyx_conn
    from v2.UpdateAlpha.analyst_forecast.utils import _date, aggregate


@dataclass(frozen=True)
class TPERConfig:
    lookback_days: int = 180
    revision_periods: int = 60
    close_field: str = "d_essentials/close"


class TPERContext(AlphaContext):
    """Read point-in-time target-price reports and local market fields."""

    def __init__(self, root=ROOT, conn=None, config=TPERConfig()):
        self.config = config
        self.conn = conn or get_zyyx_conn()
        self._owns_conn = conn is None
        self._cache = {}
        super().__init__(DataPool(root, asset="stock"))

    def reports(self, asof):
        """Read recent target-price reports and expand them to analysts."""
        asof = _date(asof)
        if self._cache.get("asof") == asof:
            return self._cache["reports"]

        start = asof - pd.Timedelta(days=self.config.lookback_days)
        sql = f"""
        SELECT
            f.id, f.report_id, f.stock_code, f.organ_id, ra.author_id,
            f.create_date, f.entrytime,
            f.target_price_ceiling, f.target_price_floor
        FROM rpt_forecast_stk f
        JOIN rpt_report_author ra ON ra.report_id = f.report_id
        WHERE f.create_date BETWEEN '{start}' AND '{asof}'
            AND f.entrytime <= '{asof} 23:59:59'
            AND DATEDIFF(day, f.create_date, f.entrytime) BETWEEN 0 AND 7
            AND (f.reliability >= 5 OR f.reliability IS NULL)
            AND f.organ_id IS NOT NULL
            AND ra.author_id IS NOT NULL
        """
        reports = (
            pl.read_database(sql, self.conn, infer_schema_length=None)
            .with_columns(
                pl.col("stock_code").cast(pl.String).str.zfill(6).alias("tick"),
                pl.col("organ_id").cast(pl.Int64, strict=False),
                pl.col("author_id").cast(pl.Int64, strict=False),
                pl.col("create_date").cast(pl.Date, strict=False),
                pl.col("entrytime").cast(pl.Datetime, strict=False),
                pl.col("target_price_ceiling").cast(pl.Float64, strict=False),
                pl.col("target_price_floor").cast(pl.Float64, strict=False),
            )
            .filter(pl.col("tick").is_not_null())
            .sort([
                "tick", "organ_id", "author_id", "create_date", "entrytime", "report_id", "id",
            ])
            .unique(
                ["report_id", "tick", "organ_id", "author_id"],
                keep="last", maintain_order=True,
            )
        )
        self._cache = {"asof": asof, "reports": reports}
        return reports

    def local_values(self, field, asof, ticks):
        """Read a local daily field and align it to report ticks."""
        values = np.asarray(self.data.read(field, asof), dtype=float)
        positions = self.data.axis._tick_positions
        return np.asarray([
            values[position]
            if (position := positions.get(str(tick))) is not None
            else np.nan
            for tick in ticks
        ])


class _TPERFactor(AlphaBase):
    column = ""

    def cross_section(self, asof):
        raise NotImplementedError

    def calculate(self, asof):
        frame = self.cross_section(_date(asof))
        return self.context.align(frame, self.column)

    def target_consensus(self, asof, alias="target_price"):
        """Aggregate latest analysts, then equal-weight their institutions."""
        reports = (
            self.context.reports(asof)
            .with_columns(
                pl.mean_horizontal(
                    "target_price_ceiling", "target_price_floor"
                ).alias("target_price")
            )
            .filter(
                pl.col("target_price").is_finite()
                & (pl.col("target_price") > 0)
            )
        )
        return aggregate(reports, "target_price", alias=alias)

    def trade_date_offset(self, asof, periods):
        """Return the trading date a fixed number of observations before asof."""
        dates = self.context.data.axis.trade_dates
        target = np.datetime64(_date(asof), "D")
        position = int(np.searchsorted(dates, target, side="right") - 1)
        lag_position = position - int(periods)
        if position < 0 or lag_position < 0:
            return None
        return pd.Timestamp(dates[lag_position]).date()


class TPERFactor(_TPERFactor):
    """Consensus target price relative to the current close."""

    meta = AlphaMeta("tper", "consensus target price / current price - 1")
    dependencies = (
        "rpt_forecast_stk", "rpt_report_author", "d_essentials/close",
    )
    column = "tper"

    def cross_section(self, asof):
        frame = self.target_consensus(asof)
        ticks = frame["tick"].to_list()
        close = self.context.local_values(
            self.context.config.close_field, asof, ticks
        )
        return (
            frame.with_columns(pl.Series("close", close))
            .with_columns(
                pl.when(
                    pl.col("close").is_finite() & (pl.col("close") > 0)
                )
                .then(pl.col("target_price") / pl.col("close") - 1)
                .otherwise(None)
                .alias(self.column)
            )
            .select("tick", self.column)
        )


class TPRevision60Factor(_TPERFactor):
    """Sixty-trading-day revision in consensus target price."""

    meta = AlphaMeta(
        "tp_revision60",
        "60-trading-day consensus target-price revision",
    )
    dependencies = ("rpt_forecast_stk", "rpt_report_author")
    column = "tp_revision60"

    def cross_section(self, asof):
        lag_asof = self.trade_date_offset(asof, self.context.config.revision_periods)
        if lag_asof is None:
            return pl.DataFrame(
                schema={"tick": pl.String, self.column: pl.Float64}
            )

        current = self.target_consensus(asof, "target_price_current")
        previous = self.target_consensus(lag_asof, "target_price_lag60")
        return (
            current.join(previous, on="tick", how="inner")
            .with_columns(
                pl.when(
                    pl.col("target_price_current").is_finite()
                    & pl.col("target_price_lag60").is_finite()
                    & (pl.col("target_price_lag60") > 0)
                )
                .then(
                    pl.col("target_price_current") / pl.col("target_price_lag60") - 1
                )
                .otherwise(None)
                .alias(self.column)
            )
            .select("tick", self.column)
        )


__all__ = [
    "TPERConfig", "TPERContext", "TPERFactor", "TPRevision60Factor",
]