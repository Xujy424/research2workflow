"""Analyst rating revision, broker-bias and industry-relative factors."""

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
    from .utils import _date, aggregate, latest_analyst_values
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from v2.UpdateAlpha.alphabase import AlphaBase, AlphaContext, AlphaMeta
    from v2.GetData import DataPool
    from v2.UpdateData.config import ROOT, get_zyyx_conn
    from v2.UpdateAlpha.analyst_forecast.utils import (
        _date, aggregate, latest_analyst_values,
    )



@dataclass(frozen=True)
class ScoreConfig:
    lookback_days: int = 90
    history_days: int = 365
    max_revision_gap_days: int = 180
    min_history_ratings: int = 20
    industry_field: str = "industry/industry"


class ScoreContext(AlphaContext):
    """Point-in-time standardized analyst ratings and local industries."""

    def __init__(self, root=ROOT, conn=None, config=ScoreConfig()):
        self.config = config
        self.conn = conn or get_zyyx_conn()
        self._owns_conn = conn is None
        self._cache = {}
        super().__init__(DataPool(root, asset="stock"))

    def reports(self, asof):
        asof = _date(asof)
        if self._cache.get("asof") == asof:
            return self._cache["frame"]
        start = asof - pd.Timedelta(days=self.config.history_days)
        sql = f"""
        SELECT
            f.id, f.report_id, f.stock_code, f.organ_id, ra.author_id,
            f.create_date, f.entrytime,
            CASE f.gg_rating_code
                WHEN '1' THEN 0.00
                WHEN '2' THEN 0.25
                WHEN '3' THEN 0.50
                WHEN '5' THEN 0.75
                WHEN '7' THEN 1.00
                ELSE NULL
            END AS rating_score
        FROM rpt_forecast_stk f
        JOIN rpt_report_author ra ON ra.report_id = f.report_id
        WHERE f.create_date BETWEEN '{start}' AND '{asof}'
            AND f.entrytime <= '{asof} 23:59:59'
            AND DATEDIFF(day, f.create_date, f.entrytime) BETWEEN 0 AND 7
            AND (f.reliability >= 5 OR f.reliability IS NULL)
            AND f.organ_id IS NOT NULL
            AND ra.author_id IS NOT NULL
            AND f.gg_rating_code IN ('1', '2', '3', '5', '7')
        """
        frame = (
            pl.read_database(sql, self.conn, infer_schema_length=None)
            .with_columns(
                pl.col("stock_code").cast(pl.String).str.zfill(6).alias("tick"),
                pl.col("organ_id").cast(pl.Int64, strict=False),
                pl.col("author_id").cast(pl.Int64, strict=False),
                pl.col("create_date").cast(pl.Date, strict=False),
                pl.col("entrytime").cast(pl.Datetime, strict=False),
                pl.col("rating_score").cast(pl.Float64, strict=False),
            )
            .filter(pl.col("tick").is_not_null())
            .sort([
                "tick", "author_id", "create_date", "entrytime", "report_id", "id",
            ])
            .unique(
                ["report_id", "tick", "organ_id", "author_id"],
                keep="last", maintain_order=True,
            )
        )
        self._cache = {"asof": asof, "frame": frame}
        return frame

    def current_analysts(self, asof):
        asof = _date(asof)
        recent = self.reports(asof).filter(
            pl.col("create_date") >= asof - pd.Timedelta(
                days=self.config.lookback_days
            )
        )
        return latest_analyst_values(recent, "rating_score")

    def industry(self, asof, ticks):
        values = np.asarray(self.data.read(self.config.industry_field, asof), float)
        positions = self.data.axis._tick_positions
        return np.asarray([
            values[position] if (position := positions.get(str(tick))) is not None
            else np.nan
            for tick in ticks
        ])


class _ScoreFactor(AlphaBase):
    column = ""

    def cross_section(self, asof):
        raise NotImplementedError

    def calculate(self, asof):
        return self.context.align(self.cross_section(_date(asof)), self.column)


class ScoreRevisionFactor(_ScoreFactor):
    """Latest analyst rating change versus the analyst's previous rating."""

    meta = AlphaMeta("score_revision", "analyst standardized-rating revision")
    dependencies = ("rpt_forecast_stk", "rpt_report_author")
    column = "score_revision"

    def cross_section(self, asof):
        reports = self.context.reports(asof).sort(
            ["tick", "organ_id", "author_id", "create_date", "entrytime", "report_id", "id",]
        )
        keys = ["tick", "organ_id", "author_id"]
        events = reports.with_columns(
            pl.col("rating_score").shift().over(keys).alias("prior_score"),
            pl.col("create_date").shift().over(keys).alias("prior_date"),
        ).with_columns(
            (pl.col("create_date") - pl.col("prior_date"))
            .dt.total_days().alias("gap_days"),
            (pl.col("rating_score") - pl.col("prior_score"))
            .alias(self.column),
        ).filter(
            (pl.col("create_date") >= asof - pd.Timedelta(
                days=self.context.config.lookback_days
            ))
            & pl.col("prior_score").is_not_null()
            & pl.col("gap_days").is_between(
                1, self.context.config.max_revision_gap_days
            )
        )
        return aggregate(events, self.column, alias=self.column)


class ScoreOrganBiasFactor(_ScoreFactor):
    """Current rating relative to the broker's earlier cross-stock tendency."""

    meta = AlphaMeta(
        "score_organ_bias", "rating relative to broker historical rating bias"
    )
    dependencies = ("rpt_forecast_stk", "rpt_report_author")
    column = "score_organ_bias"

    def cross_section(self, asof):
        cutoff = asof - pd.Timedelta(days=self.context.config.lookback_days)
        reports = self.context.reports(asof)
        history = (
            reports.filter(pl.col("create_date") < cutoff)
            .group_by("organ_id")
            .agg(
                pl.col("rating_score").mean().alias("organ_history_mean"),
                pl.len().alias("history_count"),
            )
            .filter(
                pl.col("history_count") >= self.context.config.min_history_ratings
            )
        )
        current = self.context.current_analysts(asof).join(
            history, on="organ_id", how="inner"
        ).with_columns(
            (pl.col("rating_score") - pl.col("organ_history_mean"))
            .alias(self.column)
        )
        return aggregate(current, self.column, alias=self.column)


class ScoreIndustryFactor(_ScoreFactor):
    """Current consensus rating relative to its industry's cross-section."""

    meta = AlphaMeta(
        "score_industry", "consensus rating minus industry mean rating"
    )
    dependencies = (
        "rpt_forecast_stk", "rpt_report_author", "industry/industry",
    )
    column = "score_industry"

    def cross_section(self, asof):
        current = aggregate(
            self.context.current_analysts(asof),
            "rating_score", alias="score",
        )
        ticks = current["tick"].to_list()
        return (
            current.with_columns(
                pl.Series("industry", self.context.industry(asof, ticks))
            )
            .filter(pl.col("industry").is_finite())
            .with_columns(
                (
                    pl.col("score") - pl.col("score").mean().over("industry")
                ).alias(self.column)
            )
            .select("tick", self.column)
        )


def _factor_classes():
    return ScoreRevisionFactor, ScoreOrganBiasFactor, ScoreIndustryFactor


def calculate_score_family(asof, root=ROOT, conn=None, config=ScoreConfig()):
    with ScoreContext(root, conn, config) as context:
        return {
            cls.meta.name: cls(context).run(asof)
            for cls in _factor_classes()
        }


def update_score_family(
    asof, root=ROOT, conn=None, config=ScoreConfig(), folder="factor_pool"
):
    with ScoreContext(root, conn, config) as context:
        return {
            cls.meta.name: cls(context).update(asof, folder)
            for cls in _factor_classes()
        }


__all__ = [
    "ScoreConfig", "ScoreContext",
    "ScoreRevisionFactor", "ScoreOrganBiasFactor", "ScoreIndustryFactor",
    "calculate_score_family", "update_score_family",
]