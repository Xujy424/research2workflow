"""Analyst consensus valuation, growth, rating and attention factors."""

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
    from .utils import (
        _date, aggregate, institution_values, latest_analyst_values,
    )
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from v2.UpdateAlpha.alphabase import AlphaBase, AlphaContext, AlphaMeta
    from v2.GetData import DataPool
    from v2.UpdateData.config import ROOT, get_zyyx_conn
    from v2.UpdateAlpha.analyst_forecast.utils import (
        _date, aggregate, institution_values, latest_analyst_values,
    )



def _consensus_fy1_year(asof):
    """FY1 for consensus earnings: switch to the current year on May 1."""
    asof = _date(asof)
    return asof.year if (asof.month, asof.day) >= (5, 1) else asof.year - 1


def _detail_fy1_year(asof):
    """FY1 for broker-level forecast details: switch on January 1."""
    return _date(asof).year


@dataclass(frozen=True)
class ConsensusConfig:
    lookback_days: int = 180
    min_dispersion_analysts: int = 5
    market_value_field: str = "d_essentials/total_mv"
    close_field: str = "d_essentials/close"

@dataclass(frozen=True)
class ConsensusSnapshot:
    """Common point-in-time inputs shared by all consensus factors."""

    asof: object
    reports: pl.DataFrame
    annual: pl.DataFrame
    forecasts: pl.DataFrame

class ConsensusContext(AlphaContext):
    """Point-in-time rpt_forecast_stk data and local market fields."""

    def __init__(self, root=ROOT, conn=None, config=ConsensusConfig()):
        self.config = config
        self.conn = conn or get_zyyx_conn()
        self._owns_conn = conn is None
        self._cache = {}
        self._snapshot_cache = {}
        super().__init__(DataPool(root, asset="stock"))

    def reports(self, asof):
        """Read recent reports and expand every report to its analysts."""
        asof = _date(asof)
        if self._cache.get("asof") == asof:
            return self._cache["frame"]
        start = asof - pd.Timedelta(days=self.config.lookback_days)
        sql = f"""
        SELECT
            f.id, f.report_id, f.stock_code, f.organ_id, ra.author_id,
            f.create_date, f.entrytime,
            f.report_year, f.report_quarter,
            f.forecast_np,
            CASE f.gg_rating_code
                WHEN '1' THEN 0.00
                WHEN '2' THEN 0.25
                WHEN '3' THEN 0.50
                WHEN '5' THEN 0.75
                WHEN '7' THEN 1.00
                ELSE NULL
            END AS rating_score,
            f.target_price_ceiling, f.target_price_floor
        FROM rpt_forecast_stk f
        JOIN rpt_report_author ra ON ra.report_id = f.report_id
        WHERE f.create_date BETWEEN '{start}' AND '{asof}'
            AND f.entrytime <= '{asof} 23:59:59'
            AND DATEDIFF(day, f.create_date, f.entrytime) BETWEEN 0 AND 7
            AND f.report_year BETWEEN {asof.year - 2} AND {asof.year + 1}
            AND (f.reliability >= 5 OR f.reliability IS NULL)
            AND f.organ_id IS NOT NULL
            AND ra.author_id IS NOT NULL
        """
        frame = (
            pl.read_database(sql, self.conn, infer_schema_length=None)
            .with_columns(
                pl.col("stock_code").cast(pl.String).str.zfill(6).alias("tick"),
                pl.col("organ_id").cast(pl.Int64, strict=False),
                pl.col("author_id").cast(pl.Int64, strict=False),
                pl.col("create_date").cast(pl.Date, strict=False),
                pl.col("entrytime").cast(pl.Datetime, strict=False),
                pl.col("report_year").cast(pl.Int32, strict=False),
                pl.col("report_quarter").cast(pl.Int32, strict=False),
                pl.col("forecast_np").cast(pl.Float64, strict=False),
                pl.col("rating_score").cast(pl.Float64, strict=False),
                pl.col("target_price_ceiling").cast(pl.Float64, strict=False),
                pl.col("target_price_floor").cast(pl.Float64, strict=False),
            )
            .filter(pl.col("tick").is_not_null())
            .sort([
                "tick", "organ_id", "author_id", "report_year",
                "report_quarter", "create_date", "entrytime", "report_id", "id",
            ])
            .unique(
                [
                    "report_id", "tick", "organ_id", "author_id",
                    "report_year", "report_quarter",
                ],
                keep="last", maintain_order=True,
            )
        )
        self._cache = {"asof": asof, "frame": frame}
        return frame
    def local_values(self, field, asof, ticks):
        """Return a local daily field for report ticks without raising on delistings."""
        values = np.asarray(self.data.read(field, asof), dtype=float)
        positions = self.data.axis._tick_positions
        return np.asarray([
            values[position] if (position := positions.get(str(tick))) is not None
            else np.nan
            for tick in ticks
        ])

    def snapshot(self, asof):
        """Prepare only the point-in-time inputs shared by all factors."""
        asof = _date(asof)
        if self._snapshot_cache.get("asof") == asof:
            return self._snapshot_cache["snapshot"]

        reports = self.reports(asof)
        annual = (
            reports.filter(
                (pl.col("report_quarter") == 4) & pl.col("forecast_np").is_finite()
            )
            .sort([
                "tick", "organ_id", "author_id", "report_year", "create_date", "entrytime", "report_id", "id",
            ])
            .unique(
                ["tick", "organ_id", "author_id", "report_year"],
                keep="last", maintain_order=True,
            )
        )
        forecasts = aggregate(
            annual, "forecast_np", alias="forecast_np", extra_keys=("report_year",),
        )
        snapshot = ConsensusSnapshot(asof, reports, annual, forecasts)
        self._snapshot_cache = {"asof": asof, "snapshot": snapshot}
        return snapshot


class _ConsensusFactor(AlphaBase):
    column = ""

    def cross_section(self, snapshot):
        raise NotImplementedError

    def calculate(self, asof):
        frame = self.cross_section(self.context.snapshot(asof))
        return self.context.align(frame, self.column)

    @staticmethod
    def aggregate(frame, value, alias=None, extra_keys=()):
        return aggregate(
            frame, value, alias=alias or value, extra_keys=extra_keys
        )

    @staticmethod
    def institution_values(frame, value, alias=None, extra_keys=()):
        return institution_values(
            frame, value, alias=alias, extra_keys=extra_keys
        )

    @staticmethod
    def analyst_values(frame, value=None, extra_keys=()):
        return latest_analyst_values(frame, value, extra_keys)

    @staticmethod
    def forecast(snapshot, year, alias):
        return snapshot.forecasts.filter(
            pl.col("report_year") == year
        ).select("tick", pl.col("forecast_np").alias(alias))


class EPFY1Factor(_ConsensusFactor):
    meta = AlphaMeta("ep_fy1", "FY1 consensus net profit / market value")
    dependencies = (
        "rpt_forecast_stk", "rpt_report_author", "d_essentials/total_mv",
    )
    column = "ep_fy1"

    def cross_section(self, snapshot):
        fy1_year = _consensus_fy1_year(snapshot.asof)
        frame = self.forecast(snapshot, fy1_year, "np_fy1")
        ticks = frame["tick"].to_list()
        market_value = self.context.local_values(
            self.context.config.market_value_field, snapshot.asof, ticks
        )
        return frame.with_columns(
            pl.Series("market_value", market_value)
        ).with_columns(
            pl.when(
                pl.col("np_fy1").is_finite()
                & pl.col("market_value").is_finite()
                & (pl.col("market_value") > 0)
            ).then(
                pl.col("np_fy1") / pl.col("market_value")
            ).otherwise(None).alias(self.column)
        ).select("tick", self.column)


class PEGFactor(_ConsensusFactor):
    meta = AlphaMeta(
        "peg", "FY1 consensus PE / FY1-to-FY2 growth", direction=-1
    )
    dependencies = (
        "rpt_forecast_stk", "rpt_report_author", "d_essentials/total_mv",
    )
    column = "peg"

    def cross_section(self, snapshot):
        fy1_year = _consensus_fy1_year(snapshot.asof)
        fy1 = self.forecast(snapshot, fy1_year, "np_fy1")
        fy2 = self.forecast(snapshot, fy1_year + 1, "np_fy2")
        frame = fy1.join(fy2, on="tick", how="left")
        ticks = frame["tick"].to_list()
        market_value = self.context.local_values(
            self.context.config.market_value_field, snapshot.asof, ticks
        )
        return frame.with_columns(
            pl.Series("market_value", market_value)
        ).with_columns(
            pl.when(
                (pl.col("np_fy1") > 0)
                & pl.col("market_value").is_finite()
                & (pl.col("market_value") > 0)
            ).then(
                pl.col("np_fy1") / pl.col("market_value")
            ).otherwise(None).alias("ep_fy1"),
            pl.when(
                (pl.col("np_fy1") > 0) & (pl.col("np_fy2") > 0)
            ).then(
                pl.col("np_fy2") / pl.col("np_fy1") - 1
            ).otherwise(None).alias("growth"),
        ).with_columns(
            pl.when(
                (pl.col("ep_fy1") > 0) & (pl.col("growth") > 0)
            ).then(
                (1 / pl.col("ep_fy1")) / pl.col("growth")
            ).otherwise(None).alias(self.column)
        ).select("tick", self.column)


class SCOREFactor(_ConsensusFactor):
    meta = AlphaMeta("score", "consensus standardized analyst rating")
    dependencies = ("rpt_forecast_stk", "rpt_report_author")
    column = "score"

    def cross_section(self, snapshot):
        fy1_year = _detail_fy1_year(snapshot.asof)
        reports = snapshot.reports.filter(
            (pl.col("report_year") == fy1_year)
            & (pl.col("report_quarter") == 4)
            & pl.col("rating_score").is_finite()
        )
        return self.aggregate(reports, "rating_score", self.column)


class TPERFactor(_ConsensusFactor):
    meta = AlphaMeta("tper", "consensus target price / current price - 1")
    dependencies = (
        "rpt_forecast_stk", "rpt_report_author", "d_essentials/close",
    )
    column = "tper"

    def cross_section(self, snapshot):
        reports = snapshot.reports.with_columns(
            pl.mean_horizontal(
                "target_price_ceiling", "target_price_floor"
            ).alias("target_price")
        ).filter(pl.col("target_price") > 0)
        frame = self.aggregate(reports, "target_price")
        ticks = frame["tick"].to_list()
        close = self.context.local_values(
            self.context.config.close_field, snapshot.asof, ticks
        )
        return frame.with_columns(
            pl.Series("close", close)
        ).with_columns(
            pl.when(
                pl.col("close").is_finite() & (pl.col("close") > 0)
            ).then(
                pl.col("target_price") / pl.col("close") - 1
            ).otherwise(None).alias(self.column)
        ).select("tick", self.column)


class COVFactor(_ConsensusFactor):
    meta = AlphaMeta("cov", "square root of analysts covering in 6 months")
    dependencies = ("rpt_forecast_stk", "rpt_report_author")
    column = "cov"

    def cross_section(self, snapshot):
        authors = self.analyst_values(snapshot.reports)
        institutions = authors.group_by(["tick", "organ_id"]).agg(
            pl.col("author_id").n_unique().alias("analyst_score")
        )
        return institutions.group_by("tick").agg(
            pl.col("analyst_score").sum().sqrt().alias(self.column)
        )


class DISPFactor(_ConsensusFactor):
    meta = AlphaMeta(
        "disp", "FY1 analyst forecast dispersion", direction=-1
    )
    dependencies = ("rpt_forecast_stk", "rpt_report_author")
    column = "disp"

    def cross_section(self, snapshot):
        fy1_year = _detail_fy1_year(snapshot.asof)
        reports = snapshot.annual.filter(pl.col("report_year")==fy1_year)
        analysts = self.analyst_values(reports, "forecast_np")
        return (
            analysts.group_by("tick")
            .agg(
                pl.col("forecast_np").mean().alias("mean"),
                pl.col("forecast_np").std(ddof=1).alias("std"),
                pl.len().alias("count"),
            )
            .with_columns(
                pl.when(
                    (pl.col("count") >= self.context.config.min_dispersion_analysts)
                    & (pl.col("mean").abs() > 0)
                ).then(
                    pl.col("std") / pl.col("mean").abs()
                ).otherwise(None).alias(self.column)
            )
            .select("tick", self.column)
        )

def _factor_classes():
    return EPFY1Factor, PEGFactor, SCOREFactor, TPERFactor, COVFactor, DISPFactor


def calculate_consensus_family(
    asof, root=ROOT, conn=None, config=ConsensusConfig()
):
    """Calculate all factors while sharing one cached PIT snapshot."""
    with ConsensusContext(root, conn, config) as context:
        return {
            cls.meta.name: cls(context).run(asof)
            for cls in _factor_classes()
        }


def update_consensus_family(
    asof, root=ROOT, conn=None, config=ConsensusConfig(), folder="alpha"
):
    """Update all factor matrices while sharing one cached PIT snapshot."""
    with ConsensusContext(root, conn, config) as context:
        return {
            cls.meta.name: cls(context).update(asof, folder)
            for cls in _factor_classes()
        }



__all__ = [
    "ConsensusConfig", "ConsensusSnapshot", "ConsensusContext",
    "EPFY1Factor", "PEGFactor", "SCOREFactor", "TPERFactor",
    "COVFactor", "DISPFactor",
    "calculate_consensus_family", "update_consensus_family",
]
