"""Point-in-time analyst target-price factors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import polars as pl

if __package__:
    from ...alphabase import AlphaBase, AlphaContext, AlphaMeta
    from ....GetData import DataPool
    from ....UpdateData.config import ROOT, get_zyyx_conn
    from ..utils import _date, aggregate
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
    history_days: int = 365
    revision_lookback_days: int = 30
    revision_history_days: int = 90
    min_revision_history: int = 2
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
        """Read one as-of PIT report window without choosing factor grain."""
        asof = _date(asof)
        if self._cache.get("asof") == asof:
            return self._cache["reports"]

        start = asof - pd.Timedelta(days=self.config.history_days)
        if hasattr(self, "_report_history"):
            cutoff = pd.Timestamp(asof) + pd.Timedelta(days=1)
            reports = self._report_history.filter(
                pl.col("create_date").is_between(start, asof)
                & (pl.col("entrytime") < cutoff)
            )
            self._cache = {"asof": asof, "reports": reports}
            return reports

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
        )
        self._cache = {"asof": asof, "reports": reports}
        return reports

    def preload_reports(self, start, end):
        """Load a report history once for repeated point-in-time calculations."""
        start, end = _date(start), _date(end)
        sql = f"""
        SELECT
            f.id, f.report_id, f.stock_code, f.organ_id, ra.author_id,
            f.create_date, f.entrytime,
            f.target_price_ceiling, f.target_price_floor
        FROM rpt_forecast_stk f
        JOIN rpt_report_author ra ON ra.report_id = f.report_id
        WHERE f.create_date BETWEEN '{start}' AND '{end}'
            AND f.entrytime <= '{end} 23:59:59'
            AND DATEDIFF(day, f.create_date, f.entrytime) BETWEEN 0 AND 7
            AND (f.reliability >= 5 OR f.reliability IS NULL)
            AND f.organ_id IS NOT NULL
            AND ra.author_id IS NOT NULL
        """
        self._report_history = (
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
                "tick", "organ_id", "author_id", "create_date", "entrytime",
                "report_id", "id",
            ])
        )
        self._cache = {}
        return self._report_history

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

    def field_values(self, field, dates, ticks, *, strictly_before=False):
        """Read paired date/tick values at the latest eligible trade date."""
        if len(dates) != len(ticks):
            raise ValueError("dates and ticks must have the same length")

        axes = self.data.axis
        rows = np.searchsorted(
            axes.trade_dates,
            np.asarray(dates, dtype="datetime64[D]"),
            side="left" if strictly_before else "right",
        ) - 1
        normalized_ticks = [str(tick).strip().zfill(6) for tick in ticks]
        cols = np.asarray(
            [axes._tick_positions.get(tick, -1) for tick in normalized_ticks],
            dtype=np.int64,
        )
        values = np.full(len(rows), np.nan)
        valid = (
            (rows >= 0)
            & (rows < axes.date_count)
            & (cols >= 0)
        )
        matrix = self.data.load(field)
        values[valid] = matrix[rows[valid], cols[valid]]
        return values


class _TPERFactor(AlphaBase):
    column = ""
    report_grain = ("report_id", "tick", "organ_id", "author_id")

    def factor_reports(self, asof):
        """Apply this factor's report grain to the context's raw PIT rows."""
        frame = self.context.reports(asof).filter(
            pl.col("create_date")
            >= asof - pd.Timedelta(days=self.context.config.lookback_days)
        )
        grain = list(self.report_grain)
        missing = set(grain).difference(frame.columns)
        if missing:
            raise ValueError(
                f"TPER report grain contains missing columns: {sorted(missing)}"
            )
        order = [
            column for column in [
                "tick", "organ_id", "author_id", "create_date", "entrytime", "report_id", "id",
            ]
            if column in frame.columns
        ]
        return frame.sort(order).unique(
            grain, keep="last", maintain_order=True,
        )

    def cross_section(self, asof):
        raise NotImplementedError

    def calculate(self, asof):
        frame = self.cross_section(_date(asof))
        return self.context.align(frame, self.column)

    def target_consensus(self, asof, alias="target_price"):
        """Aggregate latest analysts, then equal-weight their institutions."""
        reports = (
            self.factor_reports(asof)
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


class WTRFactor(_TPERFactor):
    """Mean report target return, based on its prior-trading-day close."""

    meta = AlphaMeta("wtr", "weighted target-price expected return")
    dependencies = ("rpt_forecast_stk", "rpt_report_author", "d_essentials/close")
    column = "wtr"

    def cross_section(self, asof):
        reports = self.factor_reports(asof).with_columns(
            pl.mean_horizontal(
                "target_price_ceiling", "target_price_floor"
            ).alias("target_price")
        ).filter(pl.col("target_price").is_finite() & (pl.col("target_price") > 0))
        if reports.is_empty():
            return pl.DataFrame(schema={"tick": pl.String, self.column: pl.Float64})
        close = self.context.field_values(
            self.context.config.close_field,
            reports["create_date"].to_list(),
            reports["tick"].to_list(),
            strictly_before=True,
        )
        reports = reports.with_columns(pl.Series("base_close", close)).with_columns(
            pl.when(pl.col("base_close").is_finite() & (pl.col("base_close") > 0))
            .then(pl.col("target_price") / pl.col("base_close") - 1)
            .otherwise(None).alias("expected_return")
        )
        return aggregate(reports, "expected_return", alias=self.column)


class TargetPriceMeanRevisionFactor(_TPERFactor):
    """Latest target price versus that analyst's preceding 90-day mean."""

    meta = AlphaMeta(
        "target_price_mean_revision",
        "latest target price / prior 90-day analyst mean - 1",
    )
    dependencies = ("rpt_forecast_stk", "rpt_report_author")
    column = "target_price_mean_revision"

    def cross_section(self, asof):
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
            .sort([
                "tick", "organ_id", "author_id", "create_date", "entrytime", "report_id", "id",
            ])
            .unique(
                list(self.report_grain), keep="last", maintain_order=True,
            )
        )
        keys = ["tick", "organ_id", "author_id"]
        events = (
            reports.with_columns(
                pl.col("target_price")
                .rolling_mean_by(
                    "create_date",
                    window_size=f"{self.context.config.revision_history_days}d",
                    min_samples=self.context.config.min_revision_history,
                    closed="left",
                )
                .over(keys)
                .alias("prior_target_mean")
            )
            .with_columns(
                (pl.col("target_price") / pl.col("prior_target_mean") - 1).alias(self.column),
            )
            .filter(
                (pl.col("create_date")>=asof-pd.Timedelta(days=self.context.config.revision_lookback_days))
                & pl.col("prior_target_mean").is_finite()
                & (pl.col("prior_target_mean") > 0)
            )
            .sort([
                "tick", "organ_id", "author_id", "create_date", "entrytime", "report_id", "id",
            ])
            .unique(keys, keep="last", maintain_order=True)
        )
        return aggregate(events, self.column, alias=self.column)




__all__ = [
    "TPERConfig", "TPERContext",
    "TPERFactor", "WTRFactor", "TargetPriceMeanRevisionFactor", "TargetPriceMeanRevisionFactor"
]
