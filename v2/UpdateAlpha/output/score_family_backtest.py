"""Recent-three-year grouped-return backtest for the SCORE factor family."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v2.GetData import DataPool
from v2.ResearchFlow.FactorTest.metrics import IC, rankIC, calc_group_ret
from v2.UpdateAlpha.analyst_forecast.utils import (
    aggregate, latest_analyst_values,
)
from v2.UpdateData.config import get_zyyx_conn

DATA_ROOT = Path("D:/data")
OUTPUT = Path(__file__).resolve().parent / "score"
HORIZONS = (1, 5, 10, 20)
LOOKBACK_DAYS = 90
HISTORY_DAYS = 365
MIN_HISTORY = 3
MAX_REVISION_GAP = 180


def load_reports(start, end):
    sql = f"""
    SELECT f.id, f.report_id, f.stock_code, f.organ_id, ra.author_id,
           f.create_date, f.entrytime,
           f.gg_rating_code AS rating_score
    FROM rpt_forecast_stk f
    JOIN rpt_report_author ra ON ra.report_id = f.report_id
    WHERE f.create_date BETWEEN '{start}' AND '{end}'
      AND f.entrytime <= '{end} 23:59:59'
      AND DATEDIFF(day, f.create_date, f.entrytime) BETWEEN 0 AND 7
      AND (f.reliability >= 5 OR f.reliability IS NULL)
      AND f.organ_id IS NOT NULL AND ra.author_id IS NOT NULL
      AND f.gg_rating_code IN ('1','2','3','5','7')
    """
    return (
        pl.read_database(sql, get_zyyx_conn(), infer_schema_length=None)
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
            "tick", "organ_id", "author_id", "create_date",
            "entrytime", "report_id", "id",
        ])
        .unique(
            ["report_id", "tick", "organ_id", "author_id"],
            keep="last", maintain_order=True,
        )
    )


def visible_history(reports, asof):
    start = asof - pd.Timedelta(days=HISTORY_DAYS)
    return reports.filter(
        (pl.col("create_date") >= start.date())
        & (pl.col("create_date") <= asof.date())
        & (pl.col("entrytime") < asof + pd.Timedelta(days=1))
    )


def align(frame, column, tick_positions, stock_count, fill=None):
    row = np.full(stock_count, np.nan)
    for tick, value in frame.select("tick", column).iter_rows():
        position = tick_positions.get(tick)
        if position is not None:
            row[position] = value
    if fill is not None:
        row = np.nan_to_num(row, nan=fill)
    return row


def score_level(history, asof):
    recent = history.filter(
        pl.col("create_date") >= (
            asof - pd.Timedelta(days=LOOKBACK_DAYS)
        ).date()
    )
    return aggregate(recent, "rating_score", alias="score_level")


def score_bias(history, asof):
    cutoff = (asof - pd.Timedelta(days=LOOKBACK_DAYS)).date()
    old = (
        history.filter(pl.col("create_date") < cutoff)
        .group_by(["author_id", "tick"])
        .agg(
            pl.col("rating_score").mean().alias("history_mean"),
            pl.len().alias("history_count"),
        )
        .filter(pl.col("history_count") >= MIN_HISTORY)
    )
    current = latest_analyst_values(
        history.filter(pl.col("create_date") >= cutoff),
        "rating_score",
    )
    values = current.join(
        old, on=["author_id", "tick"], how="inner"
    ).with_columns(
        (
            pl.col("rating_score") - pl.col("history_mean")
        ).alias("score_bias")
    )
    return aggregate(values, "score_bias", alias="score_bias")


def score_revision_event(history, asof):
    keys = ["tick", "organ_id", "author_id"]
    events = (
        history.sort([
            "tick", "organ_id", "author_id", "create_date",
            "entrytime", "report_id", "id",
        ])
        .with_columns(
            pl.col("rating_score").shift().over(keys).alias("prior_score"),
            pl.col("create_date").shift().over(keys).alias("prior_date"),
        )
        .with_columns(
            (
                pl.col("create_date") - pl.col("prior_date")
            ).dt.total_days().alias("gap_days"),
            (
                pl.col("rating_score") - pl.col("prior_score")
            ).alias("revision"),
        )
        .filter(
            (
                pl.col("create_date")
                >= (asof - pd.Timedelta(days=LOOKBACK_DAYS)).date()
            )
            & pl.col("prior_score").is_not_null()
            & pl.col("gap_days").is_between(1, MAX_REVISION_GAP)
        )
    )
    return aggregate(events, "revision", alias="revision")


def build_factors(reports, dates, tick_positions, stock_count):
    factors = {
        "score_level": np.full((len(dates), stock_count), np.nan),
        "score_bias": np.full((len(dates), stock_count), np.nan),
        "score_revision_event": np.zeros((len(dates), stock_count)),
    }
    for i, date in enumerate(dates):
        asof = pd.Timestamp(date)
        history = visible_history(reports, asof)
        level = score_level(history, asof)
        bias = score_bias(history, asof)
        revision = score_revision_event(history, asof)
        factors["score_level"][i] = align(
            level, "score_level", tick_positions, stock_count
        )
        factors["score_bias"][i] = align(
            bias, "score_bias", tick_positions, stock_count
        )
        factors["score_revision_event"][i] = align(
            revision, "revision", tick_positions, stock_count, fill=0.0
        )
        if i % 50 == 0:
            print(f"factor dates: {i}/{len(dates)}", flush=True)
    return factors


def forward_return(daily_return, horizon):
    output = np.full_like(daily_return, np.nan, dtype=float)
    windows = np.lib.stride_tricks.sliding_window_view(
        daily_return[2:], horizon, axis=0
    )
    output[:len(windows)] = np.prod(1 + windows, axis=-1) - 1
    return output


def evaluate(name, signal, daily_return, tradable, dates):
    signal = np.where(tradable, signal, np.nan)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    rows = []

    for ax, horizon in zip(axes.flat, HORIZONS):
        label = forward_return(daily_return, horizon)
        ic = IC(signal, label)
        rank_ic = rankIC(signal, label)
        groups = calc_group_ret(signal, label, 10)
        cumulative = np.nancumsum(groups, axis=1)
        group_mean = np.nanmean(groups, axis=1)

        for group, values in enumerate(cumulative, start=1):
            suffix = " (Low)" if group == 1 else " (High)" if group == 10 else ""
            ax.plot(
                dates[:len(values)], values,
                color=colors[group - 1], linewidth=1.1,
                label=f"G{group}{suffix}",
            )

        rows.append({
            "factor": name,
            "horizon": horizon,
            "mean_ic": np.nanmean(ic),
            "icir": np.nanmean(ic) / np.nanstd(ic) * np.sqrt(242),
            "mean_rank_ic": np.nanmean(rank_ic),
            "rank_icir": (
                np.nanmean(rank_ic) / np.nanstd(rank_ic) * np.sqrt(242)
            ),
            "mean_g10_g1": group_mean[-1] - group_mean[0],
            "monotonic_corr": np.corrcoef(
                np.arange(1, 11), group_mean
            )[0, 1],
            "adjacent_order_rate": np.mean(np.diff(group_mean) > 0),
        })
        ax.set_title(
            f"{name} {horizon}D | "
            f"IC={np.nanmean(ic):.4f}, RankIC={np.nanmean(rank_ic):.4f}"
        )
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.grid(alpha=0.25)
        ax.legend(ncol=2, fontsize=7)

    fig.suptitle(f"{name}: Decile Cumulative Excess Returns (Latest 3Y)")
    fig.supxlabel("Trade Date")
    fig.supylabel("Cumulative Group Excess Return")
    fig.tight_layout()
    fig.savefig(
        OUTPUT / f"{name}_group_returns_3y.png",
        dpi=160, bbox_inches="tight",
    )
    plt.close(fig)
    return rows


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = DataPool(DATA_ROOT, asset="stock")
    data.asset_root = DATA_ROOT
    all_dates = data.axis.trade_dates
    end_pos = data.axis.date_count - 1
    end = pd.Timestamp(all_dates[end_pos])
    start = end - pd.DateOffset(years=3)
    start_pos = int(np.searchsorted(
        all_dates, np.datetime64(start.date())
    ))
    dates = all_dates[start_pos:end_pos + 1]

    daily_return = np.asarray(
        data.read("d_essentials/pct", end_pos, start_date=start_pos),
        dtype=float,
    ) / 100
    tradable = np.asarray(
        data.read("basic/tradable", end_pos, start_date=start_pos),
        dtype=bool,
    )
    query_start = (
        start - pd.Timedelta(days=HISTORY_DAYS)
    ).date()
    print(f"query reports: {query_start} to {end.date()}", flush=True)
    reports = load_reports(query_start, end.date())
    factors = build_factors(
        reports,
        dates,
        data.axis._tick_positions,
        data.axis.tick_count,
    )

    rows = []
    for name, signal in factors.items():
        rows.extend(
            evaluate(name, signal, daily_return, tradable, dates)
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(
        OUTPUT / "score_family_summary_3y.csv", index=False
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()