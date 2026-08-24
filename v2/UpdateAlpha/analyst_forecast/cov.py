"""Point-in-time analyst coverage factors."""

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
    from .utils import _date
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from v2.UpdateAlpha.alphabase import AlphaBase, AlphaContext, AlphaMeta
    from v2.GetData import DataPool
    from v2.UpdateData.config import ROOT, get_zyyx_conn
    from v2.UpdateAlpha.analyst_forecast.utils import _date


@dataclass(frozen=True)
class COVConfig:
    lookback_days: int = 180


class COVContext(AlphaContext):
    """Current-FY1 annual forecasts used to measure active analyst coverage."""

    def __init__(self, root=ROOT, conn=None, config=COVConfig()):
        self.config = config
        self.conn = conn or get_zyyx_conn()
        self._owns_conn = conn is None
        self._cache = {}
        super().__init__(DataPool(root, asset="stock"))

    def reports(self, asof):
        """Return unique reports carrying a finite current-FY1 annual forecast."""
        asof = _date(asof)
        if self._cache.get("asof") == asof:
            return self._cache["reports"]
        start = asof - pd.Timedelta(days=self.config.lookback_days)
        sql = f"""
        SELECT
            f.id, f.report_id, f.stock_code, f.create_date, f.entrytime,
            f.forecast_np
        FROM rpt_forecast_stk f
        WHERE f.create_date BETWEEN '{start}' AND '{asof}'
            AND f.entrytime <= '{asof} 23:59:59'
            AND DATEDIFF(day, f.create_date, f.entrytime) BETWEEN 0 AND 7
            AND f.report_year = {asof.year}
            AND f.report_quarter = 4
            AND f.forecast_np IS NOT NULL
            AND (f.reliability >= 5 OR f.reliability IS NULL)

        """
        reports = (
            pl.read_database(sql, self.conn, infer_schema_length=None)
            .with_columns(
                pl.col("stock_code").cast(pl.String).str.zfill(6).alias("tick"),
                pl.col("create_date").cast(pl.Date, strict=False),
                pl.col("entrytime").cast(pl.Datetime, strict=False),
                pl.col("forecast_np").cast(pl.Float64, strict=False),
            )
            .filter(pl.col("tick").is_not_null() & pl.col("forecast_np").is_finite())
            .sort([
                "tick", "create_date", "entrytime", "report_id", "id",
            ])
            .unique(
                ["report_id", "tick"],
                keep="last", maintain_order=True,
            )
        )
        self._cache = {"asof": asof, "reports": reports}
        return reports


class COVFactor(AlphaBase):
    """Square root of unique current-FY1 reports published in six months."""

    meta = AlphaMeta(
        "cov",
        "square root of unique reports with a finite current-FY1 forecast",
    )
    dependencies = ("rpt_forecast_stk",)
    column = "cov"

    def cross_section(self, asof):
        return self.context.reports(asof).group_by("tick").agg(
            pl.col("report_id").n_unique().sqrt().alias(self.column)
        )

    def calculate(self, asof):
        values = self.context.align(self.cross_section(_date(asof)), self.column)
        return np.nan_to_num(values, nan=0.0)


__all__ = ["COVConfig", "COVContext", "COVFactor"]