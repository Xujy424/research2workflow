"""Standardized unexpected earnings (SUE) and revenue (SUR) factors."""

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
    from ...UpdateData.config import ROOT, get_jy_conn
    from ...ResearchFlow.FactorTest.metrics import IC, rankIC, calc_group_ret
    from .utils import _date
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from v2.UpdateAlpha.alphabase import AlphaBase, AlphaContext, AlphaMeta
    from v2.GetData import DataPool
    from v2.UpdateData.config import ROOT, get_jy_conn
    from v2.ResearchFlow.FactorTest.metrics import IC, rankIC, calc_group_ret
    from v2.UpdateAlpha.analyst_forecast.utils import _date

DEFAULT_ROOT = Path("Z:/") if Path("Z:/axis/dates.npy").is_file() else ROOT


@dataclass(frozen=True)
class SUEConfig:
    """Calculation choices shared by all four seasonal-surprise factors."""

    lookback_years: int = 4
    history_quarters: int = 8
    min_quarter_observations: int = 8
    min_change_observations: int = 4


class SUEContext(AlphaContext):
    """Point-in-time financial-report data shared by SUE/SUR factors."""

    def __init__(self, root=DEFAULT_ROOT, conn=None, config=SUEConfig()):
        self.config = config
        self.conn = conn or get_jy_conn()
        self._owns_conn = conn is None
        self._cache = {}
        super().__init__(DataPool(root, asset="stock"))

    @staticmethod
    def latest_report_fields(frame):
        """Keep latest metadata and each field's latest non-null value."""
        return (
            frame.sort(["tick", "end_date", "publish_date", "id"])
            .group_by(["tick", "end_date"], maintain_order=True)
            .agg(
                pl.col("id").last(),
                pl.col("publish_date").last(),
                pl.col("net_profit_accum").drop_nulls().last(),
                pl.col("revenue_accum").drop_nulls().last(),
            )
        )

    def reports(self, asof):
        """Read every report visible at *asof*, retaining report-period keys."""
        asof = _date(asof)
        if asof in self._cache:
            return self._cache[asof]

        start = (
            pd.Timestamp(asof) - pd.DateOffset(years=self.config.lookback_years)
        ).date()
        sql = f"""
        SELECT
            f.ID AS id, 
            s.SecuCode AS tick, 
            f.EndDate AS end_date,
            f.InfoPublDate AS publish_date,
            f.NPParentCompanyOwners AS net_profit_accum,
            f.OperatingRevenue AS revenue_accum
        FROM dbo.LC_IncomeStatementAll AS f
        INNER JOIN dbo.SecuMain AS s ON f.CompanyCode = s.CompanyCode
        WHERE f.EndDate >= '{start}'
          AND f.InfoPublDate <= '{asof}'
          AND f.IfMerged = 1 AND f.IfAdjusted = 2 AND f.IfComplete = 1
          AND f.BulletinType IN (20, 30)
          AND s.SecuCategory = 1 AND s.SecuMarket IN (83, 90)
        UNION ALL
        SELECT
            f.ID AS id, 
            s.SecuCode AS tick,
            f.EndDate AS end_date,
            f.InfoPublDate AS publish_date,
            f.NPParentCompanyOwners AS net_profit_accum,
            f.OperatingRevenue AS revenue_accum
        FROM dbo.LC_STIBIncomeState AS f
        INNER JOIN dbo.SecuMain AS s ON f.CompanyCode = s.CompanyCode
        WHERE f.EndDate >= '{start}'
          AND f.InfoPublDate <= '{asof}'
          AND f.IfMerged = 1 AND f.IfAdjusted = 2 AND f.IfComplete = 1
          AND s.SecuCategory = 1 AND s.SecuMarket IN (83, 90)
        """
        frame = pl.read_database(
            sql, self.conn, infer_schema_length=None
        ).with_columns(
            pl.col("tick").cast(pl.String).str.zfill(6),
            pl.col("end_date").cast(pl.Datetime, strict=False).dt.date(),
            pl.col("publish_date").cast(pl.Datetime, strict=False).dt.date(),
            pl.col("net_profit_accum").cast(pl.Float64, strict=False),
            pl.col("revenue_accum").cast(pl.Float64, strict=False),
        )
        if not frame.is_empty():
            frame = (
                frame.filter(
                    pl.col("tick").is_not_null()
                    & pl.col("end_date").is_not_null()
                )
                .pipe(self.latest_report_fields)
            )
        self._cache[asof] = frame
        return frame

    def surprises(self, asof):
        """Return the latest point-in-time SUE0/SUE1/SUR0/SUR1 per stock."""
        frame = self.reports(asof)
        schema = {
            "tick": pl.String,
            "sue0": pl.Float64,
            "sue1": pl.Float64,
            "sur0": pl.Float64,
            "sur1": pl.Float64,
        }
        if frame.is_empty():
            return pl.DataFrame(schema=schema)

        return _quarterly_surprises(frame, self.config)


def _quarterly_surprises(frame, config):
    """Calculate all stocks with a complete quarterly axis in Polars."""
    base = frame.with_columns(
        pl.col("end_date").dt.year().alias("year"),
        (pl.col("end_date").dt.month() // 3).alias("quarter"),
    ).filter(
        pl.col("end_date").dt.month().is_in([3, 6, 9, 12])
    ).with_columns(
        (
            pl.col("year") * 4 + pl.col("quarter") - 1  # 非连续
        ).cast(pl.Int32).alias("period")
    )
    if base.is_empty():
        return pl.DataFrame(schema={
            "tick": pl.String,
            "sue0": pl.Float64,
            "sue1": pl.Float64,
            "sur0": pl.Float64,
            "sur1": pl.Float64,
        })

    previous = base.select(
        "tick",
        (pl.col("period") + 1).alias("period"),
        pl.col("net_profit_accum").alias("prior_net_profit_accum"),
        pl.col("revenue_accum").alias("prior_revenue_accum"),
    )  # 非连续

    quarterly = base.join(
        previous, on=["tick", "period"], how="left"
    ).with_columns(
        pl.when(pl.col("quarter") == 1)
        .then(pl.col("net_profit_accum"))
        .otherwise( pl.col("net_profit_accum") - pl.col("prior_net_profit_accum") ).alias("net_profit"),
        pl.when(pl.col("quarter") == 1)
        .then(pl.col("revenue_accum"))
        .otherwise( pl.col("revenue_accum") - pl.col("prior_revenue_accum") ).alias("revenue"),
    )  # 非连续

    bounds = quarterly.group_by("tick").agg(
        pl.col("period").min().alias("first_period"),
        pl.col("period").max().alias("last_period"),
    )
    grid = bounds.with_columns(
        pl.int_ranges( "first_period", pl.col("last_period") + 1 ).alias("period")
    ).explode("period").select("tick", "period")
    full = grid.join(
        quarterly, on=["tick", "period"], how="left"
    ).sort(["tick", "period"])

    full = full.with_columns(
        (pl.col("net_profit")-pl.col("net_profit").shift(4).over("tick")).alias("sue_change"),
        (pl.col("revenue")-pl.col("revenue").shift(4).over("tick")).alias("sur_change"),
    )
    expressions = []
    for source, prefix in (("net_profit", "sue"), ("revenue", "sur")):
        change = f"{prefix}_change"
        history = pl.col(change).shift(1)
        expressions.extend([
            history.rolling_mean(
                window_size=config.history_quarters,
                min_samples=config.min_change_observations,
            ).over("tick").alias(f"{prefix}_mean"),
            history.rolling_std(
                window_size=config.history_quarters,
                min_samples=config.min_change_observations,
                ddof=1,
            ).over("tick").alias(f"{prefix}_std"),
            history.pow(2).rolling_mean(
                window_size=config.history_quarters,
                min_samples=config.min_change_observations,
            ).over("tick").sqrt().alias(f"{prefix}_rms"),
            pl.col(source).is_finite().fill_null(False).cast(pl.UInt8)
            .rolling_sum(window_size=13, min_samples=1)
            .over("tick").alias(f"{prefix}_count"),
        ])
    full = full.with_columns(expressions)

    factors = []
    for prefix in ("sue", "sur"):
        available = (
            (pl.col(f"{prefix}_count") >= config.min_quarter_observations)
            & pl.col(f"{prefix}_change").is_finite()
        )
        factors.extend([
            pl.when(available & (pl.col(f"{prefix}_std") > 0))
            .then((pl.col(f"{prefix}_change") - pl.col(f"{prefix}_mean")) / pl.col(f"{prefix}_std"))
            .otherwise(None).alias(f"{prefix}0"),
            pl.when(available & (pl.col(f"{prefix}_rms") > 0))
            .then(pl.col(f"{prefix}_change") / pl.col(f"{prefix}_rms"))
            .otherwise(None).alias(f"{prefix}1"),
        ])
    return (
        full.with_columns(factors)
        .filter(pl.col("end_date").is_not_null())
        .sort(["tick", "period"])
        .unique("tick", keep="last", maintain_order=True)
        .select("tick", "sue0", "sue1", "sur0", "sur1")
    )

class _SeasonalSurpriseFactor(AlphaBase):
    column = ""
    dependencies = (
        "LC_IncomeStatementAll",
        "LC_STIBIncomeState",
    )

    def calculate(self, asof):
        frame = self.context.surprises(asof)
        return self.context.align(frame, self.column)


class SUE0Factor(_SeasonalSurpriseFactor):
    meta = AlphaMeta("sue0", "seasonal unexpected earnings with drift")
    column = "sue0"


class SUE1Factor(_SeasonalSurpriseFactor):
    meta = AlphaMeta("sue1", "seasonal unexpected earnings without drift")
    column = "sue1"


class SUR0Factor(_SeasonalSurpriseFactor):
    meta = AlphaMeta("sur0", "seasonal unexpected revenue with drift")
    column = "sur0"


class SUR1Factor(_SeasonalSurpriseFactor):
    meta = AlphaMeta("sur1", "seasonal unexpected revenue without drift")
    column = "sur1"


def _factor_classes():
    return SUE0Factor, SUE1Factor, SUR0Factor, SUR1Factor


def calculate_sue_family(asof, root=ROOT, conn=None, config=SUEConfig()):
    """Return four full-axis float32 cross-sections without writing files."""
    with SUEContext(root, conn, config) as context:
        return {
            cls.meta.name: cls(context).run(asof)
            for cls in _factor_classes()
        }


def update_sue_family(
    asof, root=ROOT, conn=None, config=SUEConfig(), folder="alpha"
):
    """Calculate and write all four factors into date-by-tick matrices."""
    with SUEContext(root, conn, config) as context:
        return {
            cls.meta.name: cls(context).update(asof, folder)
            for cls in _factor_classes()
        }


__all__ = [
    "SUEConfig",
    "SUEContext",
    "SUE0Factor",
    "SUE1Factor",
    "SUR0Factor",
    "SUR1Factor",
    "calculate_sue_family",
    "update_sue_family",
]




if __name__ == "__main__":
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # Inclusive date range. Use None to select the first/last available date.
    START_DATE = "2017-01-01"
    END_DATE = "2026-06-30"

    with SUEContext() as context:
        alpha = SUE0Factor(context)
        trade_dates = context.data["trade_dates"]

        date_index = pd.DatetimeIndex(trade_dates)
        start_date = pd.Timestamp(START_DATE) if START_DATE else date_index[0]
        end_date = pd.Timestamp(END_DATE) if END_DATE else date_index[-1]
        if start_date > end_date:
            raise ValueError("START_DATE must not be later than END_DATE")
        selected = np.flatnonzero(
            (date_index >= start_date) & (date_index <= end_date)
        )
        if selected.size == 0:
            raise ValueError("no trade dates found in the requested range")
        start_idx, end_idx = selected[0], selected[-1]
        selected_dates = trade_dates[start_idx:end_idx + 1]

        for trade_date in tqdm(selected_dates, desc="Updating SUE0"):
            alpha.update(trade_date)

        pred = context.data.load("factor_pool/sue0").copy()
        pred = pred[
            start_idx:end_idx + 1,
            :context.data.axis.tick_count,
        ]
        daily_return = context.data.read(
            "d_essentials/pct",
            start_date=0,
            end_date=context.data.axis.date_count - 1,
        ) / 100.0

        tradable = context.data.read(
            "basic/tradable",
            start_date=start_idx,
            end_date=end_idx,
        )
        pred = np.where(tradable, pred, np.nan)

        horizons = (1, 5, 10, 20)
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
        colors = plt.cm.tab10(np.linspace(0, 1, 10))

        for ax, horizon in zip(axes.flat, horizons):
            # pct[t] is the return from t - 1 to t. For a signal formed on
            # t, skip t + 1 and compound t + 2 ... t + 1 + horizon.
            windows = np.lib.stride_tricks.sliding_window_view(
                daily_return[2:], horizon, axis=0
            )
            forward_return = np.prod(1.0 + windows, axis=-1) - 1.0

            label = np.full(pred.shape, np.nan)
            range_forward_return = forward_return[start_idx:end_idx + 1]
            label[:len(range_forward_return)] = range_forward_return
            ic = IC(pred, label)
            rank_ic = rankIC(pred, label)
            group_return = calc_group_ret(pred, label, 10)
            cumulative_return = np.nancumsum(group_return, axis=1)

            for group, values in enumerate(cumulative_return, start=1):
                suffix = " (Low)" if group == 1 else " (High)" if group == 10 else ""
                ax.plot(
                    selected_dates[:len(values)],
                    values,
                    color=colors[group - 1],
                    linewidth=1.2,
                    label=f"Group {group}{suffix}",
                )

            mean_ic = np.nanmean(ic)
            mean_rank_ic = np.nanmean(rank_ic)
            ax.set_title(
                f"SUE0 {horizon}D Forward Return | "
                f"Mean IC={mean_ic:.4f}, Mean RankIC={mean_rank_ic:.4f}"
            )
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
            ax.grid(alpha=0.25)
            ax.legend(ncol=2, fontsize=8)
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(
                ax.xaxis.get_major_locator()
            ))

        fig.suptitle("SUE0 Decile Cumulative Excess Returns", fontsize=15)
        fig.supxlabel("Trade Date")
        fig.supylabel("Cumulative Group Excess Return")
        fig.tight_layout()

        range_tag = f"{date_index[start_idx]:%Y%m%d}_{date_index[end_idx]:%Y%m%d}"
        output = (
            Path(__file__).resolve().parents[1]
            / "output"
            / f"sue0_group_ret_{range_tag}.png"
        )
        # output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"saved: {output}")