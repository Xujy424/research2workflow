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
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from v2.UpdateAlpha.alphabase import AlphaBase, AlphaContext, AlphaMeta
    from v2.GetData import DataPool
    from v2.UpdateData.config import ROOT, get_zyyx_conn


def _date(value):
    return pd.Timestamp(value).date()


@dataclass(frozen=True)
class ConsensusConfig:
    lookback_days: int = 90
    min_dispersion_institutions: int = 5
    market_value_field: str = "d_essentials/total_mv"
    close_field: str = "d_essentials/close_adj"


class ConsensusContext(AlphaContext):
    """Point-in-time rpt_forecast_stk data and local market fields."""

    def __init__(self, root=ROOT, conn=None, config=ConsensusConfig()):
        self.config = config
        self.conn = conn or get_zyyx_conn()
        self._owns_conn = conn is None
        self._cache = {}
        super().__init__(DataPool(root, asset="stock"))

    def reports(self, asof):
        asof = _date(asof)
        if asof in self._cache:
            return self._cache[asof]
        start = asof - pd.Timedelta(days=self.config.lookback_days)
        sql = f"""
        SELECT
            id, report_id, stock_code, organ_id, create_date, entrytime,
            report_year, report_quarter, forecast_np, gg_rating_code,
            target_price_ceiling, target_price_floor
        FROM rpt_forecast_stk
        WHERE create_date BETWEEN '{start}' AND '{asof}'
          AND entrytime <= '{asof} 23:59:59'
          AND DATEDIFF(day, create_date, entrytime) BETWEEN 0 AND 7
          AND report_year BETWEEN {asof.year - 1} AND {asof.year + 1}
          AND (reliability >= 5 OR reliability IS NULL)
          AND organ_id IS NOT NULL
        """
        frame = pl.read_database(
            sql, self.conn, infer_schema_length=None
        ).with_columns(
            pl.col("stock_code").cast(pl.String).str.zfill(6).alias("tick"),
            pl.col("organ_id").cast(pl.Int64, strict=False),
            pl.col("create_date").cast(pl.Date, strict=False),
            pl.col("entrytime").cast(pl.Datetime, strict=False),
            pl.col("report_year").cast(pl.Int32, strict=False),
            pl.col("forecast_np").cast(pl.Float64, strict=False),
            pl.col("gg_rating_code").cast(pl.Float64, strict=False),
            pl.col("target_price_ceiling").cast(pl.Float64, strict=False),
            pl.col("target_price_floor").cast(pl.Float64, strict=False),
        ).filter(
            pl.col("tick").is_not_null()
        )
        self._cache[asof] = frame
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
        """Calculate all six raw factor cross-sections for one date."""
        asof = _date(asof)
        frame = self.reports(asof)
        schema = {
            "tick": pl.String, "ep_fy1": pl.Float64, "peg": pl.Float64,
            "score": pl.Float64, "tper": pl.Float64,
            "cov": pl.Float64, "disp": pl.Float64,
        }
        if frame.is_empty():
            return pl.DataFrame(schema=schema)

        order = ["tick", "organ_id", "create_date", "entrytime", "id"]
        annual = (
            frame.filter(
                (pl.col("report_quarter") == 4)
                & pl.col("forecast_np").is_finite()
            )
            .sort([*order[:2], "report_year", *order[2:]])
            .unique(
                ["tick", "organ_id", "report_year"],
                keep="last", maintain_order=True,
            )
        )
        consensus = annual.group_by(["tick", "report_year"]).agg(
            pl.col("forecast_np").mean().alias("forecast_np")
        )
        fy0 = consensus.filter(
            pl.col("report_year") == asof.year - 1
        ).select("tick", pl.col("forecast_np").alias("np_fy0"))
        fy1 = consensus.filter(
            pl.col("report_year") == asof.year
        ).select("tick", pl.col("forecast_np").alias("np_fy1"))
        fy2 = consensus.filter(
            pl.col("report_year") == asof.year + 1
        ).select("tick", pl.col("forecast_np").alias("np_fy2"))
        result = fy1.join(fy0, on="tick", how="left").join(
            fy2, on="tick", how="left"
        )

        coverage = frame.group_by("tick").agg(
            pl.col("organ_id").n_unique().sqrt().alias("cov")
        )
        dispersion = (
            annual.filter(pl.col("report_year") == asof.year)
            .group_by("tick")
            .agg(
                pl.col("forecast_np").mean().alias("disp_mean"),
                pl.col("forecast_np").std(ddof=1).alias("disp_std"),
                pl.col("organ_id").n_unique().alias("disp_count"),
            )
            .with_columns(
                pl.when(
                    (pl.col("disp_count") >= self.config.min_dispersion_institutions)
                    & (pl.col("disp_mean").abs() > 0)
                )
                .then(pl.col("disp_std") / pl.col("disp_mean").abs())
                .otherwise(None)
                .alias("disp")
            )
            .select("tick", "disp")
        )

        rating = (
            frame.filter(
                pl.col("gg_rating_code").is_finite()
                & (pl.col("gg_rating_code") > 0)
            )
            .sort(order)
            .unique(["tick", "organ_id"], keep="last", maintain_order=True)
            .group_by("tick")
            .agg(pl.col("gg_rating_code").mean().alias("score"))
        )
        targets = (
            frame.with_columns(
                pl.mean_horizontal(
                    "target_price_ceiling", "target_price_floor"
                ).alias("target_price")
            )
            .filter(pl.col("target_price").is_finite() & (pl.col("target_price") > 0))
            .sort(order)
            .unique(["tick", "organ_id"], keep="last", maintain_order=True)
            .group_by("tick")
            .agg(pl.col("target_price").mean().alias("target_price"))
        )

        result = (
            result.join(coverage, on="tick", how="left")
            .join(dispersion, on="tick", how="left")
            .join(rating, on="tick", how="left")
            .join(targets, on="tick", how="left")
        )
        ticks = result["tick"].to_list()
        market_value = self.local_values(
            self.config.market_value_field, asof, ticks
        )
        close = self.local_values(self.config.close_field, asof, ticks)
        result = result.with_columns(
            pl.Series("market_value", market_value),
            pl.Series("close", close),
        ).with_columns(
            pl.when(
                pl.col("np_fy1").is_finite()
                & pl.col("market_value").is_finite()
                & (pl.col("market_value") > 0)
            ).then(
                pl.col("np_fy1") / pl.col("market_value")
            ).otherwise(None).alias("ep_fy1"),
            pl.when(
                (pl.col("np_fy0") > 0) & (pl.col("np_fy2") > 0)
            ).then(
                (pl.col("np_fy2") / pl.col("np_fy0")).sqrt() - 1
            ).otherwise(None).alias("growth"),
            pl.when(
                pl.col("target_price").is_finite()
                & pl.col("close").is_finite()
                & (pl.col("close") > 0)
            ).then(
                pl.col("target_price") / pl.col("close") - 1
            ).otherwise(None).alias("tper"),
        ).with_columns(
            pl.when(
                (pl.col("ep_fy1") > 0) & (pl.col("growth") > 0)
            ).then(
                (1 / pl.col("ep_fy1")) / pl.col("growth")
            ).otherwise(None).alias("peg")
        )
        return result.select(
            "tick", "ep_fy1", "peg", "score", "tper", "cov", "disp"
        )


class _ConsensusFactor(AlphaBase):
    column = ""

    def calculate(self, asof):
        return self.context.align(self.context.snapshot(asof), self.column)


class EPFY1Factor(_ConsensusFactor):
    meta = AlphaMeta("ep_fy1", "FY1 consensus net profit / market value")
    dependencies = ("rpt_forecast_stk", "d_essentials/total_mv")
    column = "ep_fy1"


class PEGFactor(_ConsensusFactor):
    meta = AlphaMeta(
        "peg", "FY1 consensus PE / FY0-to-FY2 annualized growth", direction=-1
    )
    dependencies = ("rpt_forecast_stk", "d_essentials/total_mv")
    column = "peg"


class SCOREFactor(_ConsensusFactor):
    meta = AlphaMeta("score", "consensus standardized analyst rating")
    dependencies = ("rpt_forecast_stk",)
    column = "score"


class TPERFactor(_ConsensusFactor):
    meta = AlphaMeta("tper", "consensus target price / current price - 1")
    dependencies = ("rpt_forecast_stk", "d_essentials/close")
    column = "tper"


class COVFactor(_ConsensusFactor):
    meta = AlphaMeta("cov", "square root of institutions covering in 3 months")
    dependencies = ("rpt_forecast_stk",)
    column = "cov"


class DISPFactor(_ConsensusFactor):
    meta = AlphaMeta(
        "disp", "FY1 forecast standard deviation / absolute mean", direction=-1
    )
    dependencies = ("rpt_forecast_stk",)
    column = "disp"


def _factor_classes():
    return EPFY1Factor, PEGFactor, SCOREFactor, TPERFactor, COVFactor, DISPFactor


def calculate_consensus_family(
    asof, root=ROOT, conn=None, config=ConsensusConfig()
):
    with ConsensusContext(root, conn, config) as context:
        snapshot = context.snapshot(asof)
        return {
            cls.meta.name: context.align(snapshot, cls.column)
            for cls in _factor_classes()
        }


def update_consensus_family(
    asof, root=ROOT, conn=None, config=ConsensusConfig(), folder="alpha"
):
    with ConsensusContext(root, conn, config) as context:
        snapshot = context.snapshot(asof)
        results = {}
        original = context.snapshot
        context.snapshot = lambda _: snapshot
        try:
            for cls in _factor_classes():
                results[cls.meta.name] = cls(context).update(asof, folder)
        finally:
            context.snapshot = original
        return results


__all__ = [
    "ConsensusConfig", "ConsensusContext",
    "EPFY1Factor", "PEGFactor", "SCOREFactor", "TPERFactor",
    "COVFactor", "DISPFactor",
    "calculate_consensus_family", "update_consensus_family",
]
