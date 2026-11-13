"""Alternative analyst-forecast dispersion factors, kept separate from disp.py."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl

from .disp import DISPContext, DISPConfig
from ..alphabase import AlphaBase, AlphaMeta
from .utils import _date


@dataclass(frozen=True)
class DISPNewConfig(DISPConfig):
    winsor_quantile: float = 0.05


class _DISPNewBase(AlphaBase):
    column = ""

    def calculate(self, asof):
        frame = self.cross_section(_date(asof))
        return self.context.align(frame, self.column)

    def min_analyst_count(self):
        return 3

    def cross_section(self, asof):
        cfg = self.context.config
        reports = self.context.reports(asof)
        if reports.is_empty():
            return pl.DataFrame(schema={"tick": pl.String, self.column: pl.Float64})
        return self.compute(reports, cfg)

    @staticmethod
    def center_scale(frame, value, alias):
        return frame.group_by("tick").agg(
            pl.col(value).median().alias("center"),
            pl.col(value).len().alias("n"),
            (1.4826 * (pl.col(value) - pl.col(value).median()).abs().median()).alias("mad"),
            pl.col(value).std(ddof=1).alias("std"),
        )


class DISPNewCVFactor(_DISPNewBase):
    """Winsorized coefficient of variation; robust to extreme analyst values."""
    meta = AlphaMeta("disp_new_cv", "winsorized analyst forecast coefficient of variation", direction=-1)
    column = "disp_new_cv"

    def compute(self, reports, cfg):
        q = getattr(cfg, "winsor_quantile", 0.05)
        bounded = reports.with_columns(
            pl.col("forecast_np").quantile(q).over("tick").alias("lo"),
            pl.col("forecast_np").quantile(1-q).over("tick").alias("hi"),
        ).with_columns(pl.col("forecast_np").clip(pl.col("lo"), pl.col("hi")).alias("x"))
        stats = self.center_scale(bounded, "x", self.column).filter(
            (pl.col("n") >= self.min_analyst_count()) & (pl.col("center").abs() > 0)
        )
        return stats.select("tick", (pl.col("std") / pl.col("center").abs()).alias(self.column))


class DISPNewDownsideFactor(_DISPNewBase):
    """Downside-only forecast disagreement around the cross-analyst median."""
    meta = AlphaMeta("disp_new_downside", "robust downside analyst disagreement", direction=-1)
    column = "disp_new_downside"

    def compute(self, reports, cfg):
        centered = reports.with_columns(pl.col("forecast_np").median().over("tick").alias("center"))
        downside = centered.with_columns(
            pl.when(pl.col("forecast_np") < pl.col("center"))
            .then(pl.col("center")-pl.col("forecast_np")).otherwise(0.0).alias("down")
        )
        stats = downside.group_by("tick").agg(
            pl.col("center").first(), pl.col("forecast_np").len().alias("n"),
            pl.col("down").pow(2).mean().sqrt().alias("disp"),
        ).filter((pl.col("n") >= self.min_analyst_count()) & (pl.col("center").abs() > 0))
        floor = stats.select(pl.col("center").abs().quantile(cfg.stable_floor_quantile)).item()
        if floor is None or not np.isfinite(floor):
            return pl.DataFrame(schema={"tick": pl.String, self.column: pl.Float64})
        return stats.select("tick", (pl.col("disp") / pl.max_horizontal(pl.col("center").abs(), pl.lit(float(floor)))).alias(self.column))


class DISPNewRevisionFactor(_DISPNewBase):
    """Dispersion across analysts' latest annual forecast revisions."""
    meta = AlphaMeta("disp_new_revision", "dispersion of analyst annual forecast revisions", direction=-1)
    column = "disp_new_revision"

    def cross_section(self, asof):
        cfg = self.context.config
        history = self.context.report_history(asof).sort(
            ["tick", "author_id", "create_date", "entrytime", "report_id", "id"]
        ).with_columns(
            pl.col("forecast_np").shift(1).over(["tick", "author_id"]).alias("prior")
        ).filter(pl.col("prior").is_finite() & (pl.col("prior").abs() > 0))
        history = history.with_columns(
            ((pl.col("forecast_np")-pl.col("prior"))/pl.col("prior").abs()).alias("revision")
        )
        latest = history.sort(["tick", "author_id", "create_date", "entrytime", "id"]).unique(
            ["tick", "author_id"], keep="last", maintain_order=True
        )
        stats = latest.group_by("tick").agg(
            pl.col("revision").median().alias("center"),
            pl.col("revision").len().alias("n"),
            (1.4826*(pl.col("revision")-pl.col("revision").median()).abs().median()).alias("disp"),
        ).filter(pl.col("n") >= self.min_analyst_count())
        return stats.select("tick", pl.col("disp").alias(self.column))


if __name__ == "__main__":
    from ..backtest import run_from_ide
    run_from_ide("disp_new_cv", start_date="2021-01-01", end_date="2026-09-30")


__all__ = ["DISPNewConfig", "DISPNewCVFactor", "DISPNewDownsideFactor", "DISPNewRevisionFactor"]
