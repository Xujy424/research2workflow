"""ZYYX FY1 earnings-yield and inverse-PEG factors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import polars as pl

if __package__:
    from ..alphabase import AlphaBase, AlphaContext, AlphaMeta
    from ...GetData import DataPool
    from ...UpdateData.config import ROOT, get_zyyx_conn
    from .utils import _consensus_fy1_year, _date
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from v2.UpdateAlpha.alphabase import AlphaBase, AlphaContext, AlphaMeta
    from v2.GetData import DataPool
    from v2.UpdateData.config import ROOT, get_zyyx_conn
    from v2.UpdateAlpha.analyst_forecast.utils import (
        _consensus_fy1_year,
        _date,
    )


@dataclass(frozen=True)
class EPFY1Config:
    market_value_field: str = "d_essentials/total_mv"


class EPFY1Context(AlphaContext):
    """Read point-in-time ZYYX FY1 consensus and local market value."""

    def __init__(self, root=ROOT, conn=None, config=EPFY1Config()):
        self.config = config
        self.conn = conn or get_zyyx_conn()
        self._owns_conn = conn is None
        self._cache = {}
        super().__init__(DataPool(root, asset="stock"))

    def consensus(self, asof):
        """Return the latest visible FY1 consensus row for each stock."""
        asof = _date(asof)
        if self._cache.get("asof") == asof:
            return self._cache["consensus"]
        fy1_year = _consensus_fy1_year(asof)
        sql = f"""
        WITH ranked AS (
            SELECT
                stock_code,
                con_np,
                con_npcgrate_2y,
                ROW_NUMBER() OVER (
                    PARTITION BY stock_code
                    ORDER BY entrytime DESC, id DESC
                ) AS rn
            FROM con_forecast_stk
            WHERE con_date = '{asof}'
                AND con_year = {fy1_year}
                AND entrytime <= '{asof} 23:59:59'
                AND con_np IS NOT NULL
        )
        SELECT stock_code, con_np, con_npcgrate_2y
        FROM ranked
        WHERE rn = 1
        """
        consensus = (
            pl.read_database(sql, self.conn, infer_schema_length=None)
            .with_columns(
                pl.col("stock_code").cast(pl.String).str.zfill(6).alias("tick"),
                pl.col("con_np").cast(pl.Float64, strict=False),
                pl.col("con_npcgrate_2y").cast(pl.Float64, strict=False),
            )
            .filter(pl.col("tick").is_not_null() & pl.col("con_np").is_finite())
            .select("tick", "con_np", "con_npcgrate_2y")
        )
        self._cache = {"asof": asof, "consensus": consensus}
        return consensus

    def local_values(self, field, asof, ticks):
        values = np.asarray(self.data.read(field, asof), dtype=float)
        positions = self.data.axis._tick_positions
        return np.asarray([
            values[position]
            if (position := positions.get(str(tick))) is not None
            else np.nan
            for tick in ticks
        ])


class _EPFY1Factor(AlphaBase):
    dependencies = ("con_forecast_stk", "d_essentials/total_mv")

    def inputs(self, asof):
        asof = _date(asof)
        consensus = self.context.consensus(asof)
        ticks = consensus["tick"].to_list()
        market_value = self.context.local_values(
            self.context.config.market_value_field, asof, ticks
        )
        return consensus.with_columns(pl.Series("market_value", market_value))

    def calculate(self, asof):
        return self.context.align(self.cross_section(_date(asof)), self.column)


class EPFY1Factor(_EPFY1Factor):
    """Positive FY1 consensus net profit divided by as-of market value."""

    meta = AlphaMeta("ep_fy1", "ZYYX FY1 consensus NP / as-of market value")
    column = "ep_fy1"

    def cross_section(self, asof):
        return (
            self.inputs(asof)
            .filter(
                (pl.col("con_np") > 0)
                & pl.col("market_value").is_finite()
                & (pl.col("market_value") > 0)
            )
            .with_columns(
                (pl.col("con_np") / pl.col("market_value")).alias(self.column)
            )
            .select("tick", self.column)
        )


class PEGInverseFactor(_EPFY1Factor):
    """FY1 earnings yield times ZYYX FY0-to-FY2 annualized NP growth."""

    meta = AlphaMeta(
        "peg_inverse",
        "FY1 earnings yield x ZYYX two-year annualized NP growth",
    )
    column = "peg_inverse"

    def cross_section(self, asof):
        return (
            self.inputs(asof)
            .with_columns(
                (pl.col("con_npcgrate_2y") / 100.0).alias("growth")
            )
            .filter(
                (pl.col("con_np") > 0)
                & pl.col("growth").is_finite()
                & (pl.col("growth") > 0)
                & pl.col("market_value").is_finite()
                & (pl.col("market_value") > 0)
            )
            .with_columns(
                (
                    pl.col("con_np") / pl.col("market_value")
                    * pl.col("growth")
                ).alias(self.column)
            )
            .select("tick", self.column)
        )


__all__ = [
    "EPFY1Config", "EPFY1Context", "EPFY1Factor", "PEGInverseFactor",
]