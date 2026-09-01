"""AFR, PAFR, expected inertia and expected volatility factors."""

from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import polars as pl

if __package__:
    from ..alphabase import AlphaBase, AlphaContext, AlphaMeta
    from ...GetData import DataPool
    from ...UpdateData.config import ROOT, get_zyyx_conn
    from ...ResearchFlow.FactorTest.metrics import IC, rankIC, calc_group_ret
    from .utils import _date, aggregate, latest_analyst_values
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(PROJECT_ROOT))
    from v2.UpdateAlpha.alphabase import AlphaBase, AlphaContext, AlphaMeta
    from v2.GetData import DataPool
    from v2.UpdateData.config import ROOT, get_zyyx_conn
    from v2.ResearchFlow.FactorTest.metrics import IC, rankIC, calc_group_ret
    from v2.UpdateAlpha.analyst_forecast.utils import (
        _date, aggregate, latest_analyst_values,
    )



def _residual(y, x):
    """Return OLS residuals for one or more regressors without inversion."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or len(x) != len(y):
        raise ValueError("x must have shape (n_samples, n_regressors)")

    out = np.full(y.shape, np.nan)
    ok = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    if ok.sum() <= x.shape[1] + 1:
        out[ok] = y[ok]
        return out
    design = np.column_stack((np.ones(ok.sum()), x[ok]))
    coefficients = np.linalg.lstsq(design, y[ok], rcond=None)[0]
    out[ok] = y[ok] - design @ coefficients
    return out


def _zscore(x):
    x = np.asarray(x, float)
    out = np.full(x.shape, np.nan)
    ok = np.isfinite(x)
    std = np.nanstd(x[ok], ddof=1) if ok.sum() > 1 else np.nan
    if std > 0:
        out[ok] = (x[ok] - np.nanmean(x[ok])) / std
    return out




def _aggregate(context, frame, value, alias="value"):
    return aggregate(
        frame, value, context.analyst_weights, alias
    )


DEFAULT_ROOT = Path("Z:/") if Path("Z:/axis/dates.npy").is_file() else ROOT

@dataclass(frozen=True)
class AFRConfig:
    lookback_days: int = 90
    max_history_days: int = 183
    volatility_days: int = 365
    min_institutions: int = 3
    revision_limit: float = 0.25
    close_field: str = "d_essentials/close_adj"
    market_value_field: str = "d_essentials/circ_mv"
    industry_field: str = "industry/industry"


class AFRContext(AlphaContext):
    """Share point-in-time SQL and local matrix reads among factors."""

    def __init__(
        self, root=DEFAULT_ROOT, conn=None, config=AFRConfig(), analyst_weights=None
    ):
        self.config = config
        self.analyst_weights = analyst_weights
        self.conn = conn or get_zyyx_conn()
        self._owns_conn = conn is None
        super().__init__(DataPool(root, asset="stock"))
        self._cache = {}


    @staticmethod
    def analyst_values(frame, value):
        return latest_analyst_values(frame, value)
    def reports(self, asof, start):
        asof = _date(asof)
        if asof in self._cache:
            return self._cache[asof]
        sql = f"""
        SELECT
            f.id, f.report_id, f.stock_code, f.organ_id, ra.author_id,
            f.create_date, f.entrytime, f.report_year, f.forecast_np,
            f.target_price_ceiling, f.target_price_floor, f.current_price
        FROM rpt_forecast_stk f
        JOIN rpt_report_author ra ON ra.report_id = f.report_id
        WHERE f.create_date BETWEEN '{start}' AND '{asof}'
            AND f.report_quarter = 4
            AND f.report_year = YEAR(f.create_date)
            AND f.forecast_np IS NOT NULL
            AND (f.reliability >= 5 OR f.reliability IS NULL)
        """
        frame = (
            pl.read_database(sql, self.conn, infer_schema_length=None)
            .with_columns(
                pl.col("stock_code").cast(pl.String).str.zfill(6).alias("tick"),
                pl.col("create_date").str.strptime(pl.Datetime("us"), format="%Y-%m-%d").alias("create_date")
            )
            .sort([
                "tick", "author_id", "report_year", "create_date", "entrytime", "id",
            ])
            .unique(
                ["report_id", "tick", "organ_id", "author_id", "report_year"],
                keep="last", maintain_order=True,
            )
        )
        self._cache[asof] = frame
        return frame

    def empty(self):
        return np.full(self.data.axis.tick_count, np.nan, dtype=np.float32)

    def field_values(self, field, dates, ticks):
        """Read paired date/tick values at the latest available trade date."""
        axes = self.data.axis
        rows = np.searchsorted(
            axes.trade_dates,
            np.asarray(dates, dtype="datetime64[D]"),
            side="right",
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

    def price(self, dates, ticks):
        return self.field_values(self.config.close_field, dates, ticks)

    def market_price(self, dates):
        """Return the all-stock market-cap-weighted price for each date."""
        axes = self.data.axis
        requested = np.asarray(dates, dtype="datetime64[D]")
        rows = np.searchsorted(
            axes.trade_dates, requested, side="right",
        ) - 1
        result = np.full(len(rows), np.nan)
        close = self.data.load(self.config.close_field)
        market_value = self.data.load(self.config.market_value_field)

        valid_rows = (rows >= 0) & (rows < axes.date_count)
        unique_rows, inverse = np.unique(
            rows[valid_rows], return_inverse=True
        )
        prices = close[unique_rows]
        weights = market_value[unique_rows]
        valid = (
            np.isfinite(prices)
            & np.isfinite(weights)
            & (prices > 0)
            & (weights > 0)
        )
        numerator = np.sum(
            np.where(valid, prices * weights, 0.0), axis=1, dtype=np.float64
        )
        denominator = np.sum(
            np.where(valid, weights, 0.0), axis=1, dtype=np.float64
        )
        weighted_price = np.divide(
            numerator,
            denominator,
            out=np.full(len(unique_rows), np.nan),
            where=denominator > 0,
        )
        result[valid_rows] = weighted_price[inverse]
        return result

    def industry(self, asof, ticks):
        return np.asarray(
            self.data.read(self.config.industry_field, asof, ticks=ticks), float
        )


class AFRFactor(AlphaBase):
    meta = AlphaMeta("afr", "analyst forecast net-profit revision")
    dependencies = ("rpt_forecast_stk",)

    def event_values(self, asof):
        cfg = self.context.config
        start_date = asof - pd.Timedelta(days=cfg.max_history_days)
        keys = ["tick", "author_id", "report_year"]
        events = self.context.reports(asof, start_date).with_columns(
            pl.col("forecast_np").shift().over(keys).alias("prior_np"),
            pl.col("create_date").shift().over(keys).alias("prior_date"),
        ).with_columns(
            (pl.col("create_date") - pl.col("prior_date")).dt.total_days().alias("gap_days"),
            ((pl.col("forecast_np") - pl.col("prior_np")) / pl.col("prior_np").abs()).clip(-cfg.revision_limit, cfg.revision_limit).alias("afr_event"),
        )
        return events.filter(
            pl.col("prior_date").is_not_null()  # 去掉首次预测，变相保证一个分析师对某股票至少两次预测
            & (pl.col("create_date") >= _date(asof) - pd.Timedelta(days=cfg.lookback_days))  # 回看90天
            & pl.col("afr_event").is_finite()
            & (pl.col("organ_id").n_unique().over("tick") >= cfg.min_institutions)  # 剔除少于三份预测报告的股票
        )

    def calculate(self, asof):
        asof = _date(asof)
        frame = _aggregate(
            self.context, self.event_values(asof), "afr_event",
        )
        return self.context.align(frame)


class PAFRFactor(AFRFactor):
    meta = AlphaMeta("pafr", "AFR stripped of two excess-momentum effects")
    dependencies = (
        "rpt_forecast_stk",
        "d_essentials/close_adj",
        "d_essentials/circ_mv",
    )

    def excess_momentum(self, starts, ends, ticks):
        stock_momentum = np.log(
            self.context.price(ends, ticks)
            / self.context.price(starts, ticks)
        )
        market_momentum = np.log(
            self.context.market_price(ends)
            / self.context.market_price(starts)
        )
        return stock_momentum - market_momentum

    def calculate(self, asof):
        asof = _date(asof)
        events = self.event_values(asof)
        if events.is_empty():
            return self.context.empty()
        
        ticks = events["tick"].to_list()
        between = self.excess_momentum(
            events["prior_date"], events["create_date"], ticks
        )
        post = self.excess_momentum(
            events["create_date"], [asof] * len(events), ticks
        )
        events = events.with_columns(
            pl.Series("between", between),
            pl.Series("post", post),
        )
        momentum = events.select("between", "post").to_numpy()
        events = events.with_columns(
            pl.Series("pafr_event", _residual(events["afr_event"], momentum))
        )
        stock = _aggregate(
            self.context, events, "pafr_event",
        )
        return self.context.align(stock)


class ExpectedInertiaFactor(AlphaBase):
    meta = AlphaMeta("expected_inertia", "Implied valuation multiple revision")
    dependencies = ("rpt_forecast_stk",)

    def event_values(self, asof, lookback_days=None):
        cfg = self.context.config
        keys = ["tick", "author_id", "report_year"]
        start_date = asof - pd.Timedelta(days=cfg.volatility_days)
        events = self.context.reports(asof, start_date).with_columns(
            pl.mean_horizontal("target_price_ceiling", "target_price_floor").alias("target_mid")
        ).with_columns(
            pl.col("forecast_np").shift().over(keys).alias("prior_np"),
            pl.col("target_mid").shift().over(keys).alias("prior_target"),
            pl.col("create_date").shift().over(keys).alias("prior_date"),
        ).with_columns(
            (pl.col("create_date") - pl.col("prior_date")).dt.total_days().alias("gap_days"),
            (
                (pl.col("target_mid") / pl.col("prior_target")).log()
                - (pl.col("forecast_np") / pl.col("prior_np")).log()
            ).alias("inertia_event"),
        )
        days = cfg.lookback_days if lookback_days is None else lookback_days
        return events.filter(
            pl.col("prior_date").is_not_null()
            & pl.col("gap_days").is_between(1, cfg.max_history_days)
            & (
                pl.col("create_date")>= _date(asof) - pd.Timedelta(days=days)
            )
            & pl.col("inertia_event").is_finite()
        )

    def calculate(self, asof):
        cfg = self.context.config
        events = self.event_values(asof).filter(
            pl.col("organ_id").n_unique().over("tick") >= cfg.min_institutions
        )
        frame = _aggregate(
            self.context, events, "inertia_event",
        )
        return self.context.align(frame)


class ExpectedVolatilityFactor(ExpectedInertiaFactor):
    meta = AlphaMeta(
        "expected_volatility",
        "Analyst disagreement and time-series expected volatility",
    )
    dependencies = ("rpt_forecast_stk",)

    def calculate(self, asof):
        asof = _date(asof)
        cfg = self.context.config
        events = self.event_values(asof, cfg.volatility_days)
        recent = events.filter(
            pl.col("create_date") >= asof - pd.Timedelta(days=cfg.lookback_days)
        )
        if recent.is_empty():
            return self.context.empty()

        latest = self.context.analyst_values(recent, "inertia_event")
        tick_vol = latest.group_by("tick").agg(
            pl.col("inertia_event").std(ddof=1).alias("tick_vol")
        )

        analyst_vol = events.group_by(
            ["tick", "organ_id", "author_id"]
        ).agg(
            pl.col("inertia_event").std(ddof=1).alias("time_vol")
        )
        time_vol = _aggregate(
            self.context, analyst_vol, "time_vol", "time_vol",
        )

        frame = tick_vol.join(time_vol, on="tick", how="inner")
        value = (
            -_zscore(frame["tick_vol"]) - _zscore(frame["time_vol"])
        ) / 2
        frame = frame.with_columns(pl.Series("value", value))

        # Report-style industry-dispersion alternative retained for reference:
        # current = _aggregate(
        #     self.context, recent, "inertia_event", "inertia",
        # )
        # frame = current.join(time_vol, on="tick", how="left").with_columns(
        #     pl.Series("industry", self.context.industry(asof, current["tick"]))
        # ).with_columns(
        #     pl.col("inertia").std(ddof=1).over("industry").alias("industry_vol")
        # )
        # value = (
        #     _zscore(frame["industry_vol"]) - _zscore(frame["time_vol"])
        # ) / 2

        return self.context.align(frame)


def _factor_classes():
    return AFRFactor, PAFRFactor, ExpectedInertiaFactor, ExpectedVolatilityFactor


def calculate_afr_family(
    asof, root=ROOT, conn=None, config=AFRConfig(), analyst_weights=None
):
    """Return four full-axis float32 cross-sections without writing files."""
    with AFRContext(root, conn, config, analyst_weights) as context:
        return {cls.meta.name: cls(context).run(asof) for cls in _factor_classes()}


def update_afr_family(
    asof, root=ROOT, conn=None, config=AFRConfig(),
    folder="alpha", analyst_weights=None,
):
    """Calculate and write all factors into date-by-tick matrices."""
    with AFRContext(root, conn, config, analyst_weights) as context:
        return {cls.meta.name: cls(context).update(asof, folder) for cls in _factor_classes()}


__all__ = [
    "AFRConfig", "AFRContext", "AFRFactor", "PAFRFactor",
    "ExpectedInertiaFactor", "ExpectedVolatilityFactor",
    "calculate_afr_family", "update_afr_family",
]


if __name__ == "__main__":
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # Inclusive date range. Use None to select the first/last available date.
    START_DATE = "2017-01-01"
    END_DATE = "2026-06-30"

    with AFRContext() as context:
        alpha = PAFRFactor(context)
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

        for trade_date in tqdm(selected_dates, desc="Updating PARF"):
            alpha.update(trade_date)

        pred = context.data.load("factor_pool/pafr").copy()
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
                f"PARF {horizon}D Forward Return | "
                f"Mean IC={mean_ic:.4f}, Mean RankIC={mean_rank_ic:.4f}"
            )
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
            ax.grid(alpha=0.25)
            ax.legend(ncol=2, fontsize=8)
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(
                ax.xaxis.get_major_locator()
            ))

        fig.suptitle("PARF Decile Cumulative Excess Returns", fontsize=15)
        fig.supxlabel("Trade Date")
        fig.supylabel("Cumulative Group Excess Return")
        fig.tight_layout()

        range_tag = f"{date_index[start_idx]:%Y%m%d}_{date_index[end_idx]:%Y%m%d}"
        output = (
            Path(__file__).resolve().parents[1]
            / "output"
            / f"pafr_group_ret_{range_tag}.png"
        )
        # output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"saved: {output}")