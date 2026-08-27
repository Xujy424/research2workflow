"""Point-in-time analyst target-price factors."""

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
    report_grain = ("report_id", "tick", "organ_id", "author_id")

    def factor_reports(self, asof):
        """Apply this factor's report grain to the context's raw PIT rows."""
        frame = self.context.reports(asof)
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
        dates, pieces = self.context.data.axis.trade_dates, []
        for date in reports["create_date"].unique().to_list():
            part = reports.filter(pl.col("create_date") == date)
            pos = int(np.searchsorted(dates, np.datetime64(date, "D"), side="left") - 1)
            if pos < 0:
                continue
            close = self.context.local_values(
                self.context.config.close_field,
                pd.Timestamp(dates[pos]).date(),
                part["tick"].to_list(),
            )
            pieces.append(part.with_columns(pl.Series("base_close", close)).with_columns(
                pl.when(pl.col("base_close").is_finite() & (pl.col("base_close") > 0))
                .then(pl.col("target_price") / pl.col("base_close") - 1)
                .otherwise(None).alias("expected_return")
            ))
        if not pieces:
            return pl.DataFrame(schema={"tick": pl.String, self.column: pl.Float64})
        return aggregate(pl.concat(pieces), "expected_return", alias=self.column)


class ConsensusExpectedReturnFactor(TPERFactor):
    """Consensus expected return under an explicit factor name."""

    meta = AlphaMeta("consensus_expected_return", "consensus target return")
    column = "consensus_expected_return"


class WeightedTargetPriceYoYFactor(_TPERFactor):
    """Current equal-weight target price minus the prior-year value."""

    meta = AlphaMeta("weighted_target_price_yoy", "weighted target-price YoY change")
    dependencies = ("rpt_forecast_stk", "rpt_report_author")
    column = "weighted_target_price_yoy"

    def cross_section(self, asof):
        lag_asof = (pd.Timestamp(asof) - pd.DateOffset(years=1)).date()
        current = self.target_consensus(asof, "target_price_current")
        previous = self.target_consensus(lag_asof, "target_price_last_year")
        return (
            current.join(previous, on="tick", how="inner")
            .with_columns(
                pl.when(
                    pl.col("target_price_current").is_finite()
                    & pl.col("target_price_last_year").is_finite()
                )
                .then(pl.col("target_price_current") - pl.col("target_price_last_year"))
                .otherwise(None)
                .alias(self.column)
            )
            .select("tick", self.column)
        )


__all__ = [
    "TPERConfig", "TPERContext", "TPERFactor", "WTRFactor",
    "ConsensusExpectedReturnFactor", "WeightedTargetPriceYoYFactor",
]
