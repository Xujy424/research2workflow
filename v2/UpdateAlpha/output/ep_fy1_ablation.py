"""Ablation study for forward earnings-yield construction."""

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
from v2.UpdateData.config import get_zyyx_conn

DATA_ROOT = Path("D:/data")
OUTPUT = Path(__file__).resolve().parent / "ep_fy1" / "ablation"
HORIZONS = (1, 5, 10, 20)
LOOKBACK = 180
MIN_INSTITUTIONS = 3


def load_reports(start, end):
    sql = f"""
    SELECT f.id, f.report_id, f.stock_code, f.organ_id, ra.author_id,
           f.create_date, f.entrytime, f.report_year, f.forecast_np
    FROM rpt_forecast_stk f
    JOIN rpt_report_author ra ON ra.report_id = f.report_id
    WHERE f.create_date BETWEEN '{start}' AND '{end}'
      AND f.entrytime <= '{end} 23:59:59'
      AND DATEDIFF(day, f.create_date, f.entrytime) BETWEEN 0 AND 7
      AND f.report_quarter = 4
      AND f.forecast_np IS NOT NULL
      AND (f.reliability >= 5 OR f.reliability IS NULL)
      AND f.organ_id IS NOT NULL AND ra.author_id IS NOT NULL
    """
    return (
        pl.read_database(sql, get_zyyx_conn(), infer_schema_length=None)
        .with_columns(
            pl.col("stock_code").cast(pl.String).str.zfill(6).alias("tick"),
            pl.col("organ_id").cast(pl.Int64, strict=False),
            pl.col("author_id").cast(pl.Int64, strict=False),
            pl.col("create_date").cast(pl.Date, strict=False),
            pl.col("entrytime").cast(pl.Datetime, strict=False),
            pl.col("report_year").cast(pl.Int32, strict=False),
            pl.col("forecast_np").cast(pl.Float64, strict=False),
        )
        .filter(pl.col("tick").is_not_null() & pl.col("forecast_np").is_finite())
        .sort([
            "tick", "organ_id", "author_id", "report_year",
            "create_date", "entrytime", "report_id", "id",
        ])
        .unique(
            ["report_id", "tick", "organ_id", "author_id", "report_year"],
            keep="last", maintain_order=True,
        )
    )


def visible(reports, asof):
    return reports.filter(
        (pl.col("create_date") >= (asof - pd.Timedelta(days=LOOKBACK)).date())
        & (pl.col("create_date") <= asof.date())
        & (pl.col("entrytime") < asof + pd.Timedelta(days=1))
    )


def analyst_institution_mean(frame):
    analysts = (
        frame.sort([
            "tick", "organ_id", "author_id", "report_year",
            "create_date", "entrytime", "report_id", "id",
        ])
        .unique(["tick", "author_id", "report_year"], keep="last")
    )
    institutions = analysts.group_by(
        ["tick", "organ_id", "report_year"]
    ).agg(pl.col("forecast_np").mean().alias("institution_forecast"))
    return institutions.group_by(["tick", "report_year"]).agg(
        pl.col("institution_forecast").mean().alias("consensus_np"),
        pl.col("organ_id").n_unique().alias("institution_count"),
    )


def broker_latest(frame, method="mean"):
    brokers = (
        frame.sort([
            "tick", "organ_id", "report_year",
            "create_date", "entrytime", "report_id", "id",
        ])
        .unique(["report_id", "tick", "organ_id", "report_year"], keep="last")
        .unique(["tick", "organ_id", "report_year"], keep="last")
    )
    expression = (
        pl.col("forecast_np").median()
        if method == "median"
        else pl.col("forecast_np").mean()
    )
    return brokers.group_by(["tick", "report_year"]).agg(
        expression.alias("consensus_np"),
        pl.col("organ_id").n_unique().alias("institution_count"),
    )


def weighted_median(values, weights):
    values, weights = np.asarray(values), np.asarray(weights)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return np.nan
    order = np.argsort(values[valid])
    values, weights = values[valid][order], weights[valid][order]
    return values[np.searchsorted(np.cumsum(weights), weights.sum() / 2)]


def fresh_broker_median(frame, asof, half_life=90):
    brokers = (
        frame.sort([
            "tick", "organ_id", "report_year",
            "create_date", "entrytime", "report_id", "id",
        ])
        .unique(["report_id", "tick", "organ_id", "report_year"], keep="last")
        .unique(["tick", "organ_id", "report_year"], keep="last")
        .with_columns(
            (pl.lit(asof.date()) - pl.col("create_date"))
            .dt.total_days().alias("age")
        )
        .with_columns(
            (-np.log(2) * pl.col("age") / half_life)
            .exp().alias("weight")
        )
    )
    rows = []
    for key, group in brokers.group_by(["tick", "report_year"]):
        tick, year = key
        count = group["organ_id"].n_unique()
        if count >= MIN_INSTITUTIONS:
            rows.append((
                tick, int(year),
                weighted_median(group["forecast_np"], group["weight"]),
                count,
            ))
    return pl.DataFrame(
        rows,
        schema={
            "tick": pl.String, "report_year": pl.Int32,
            "consensus_np": pl.Float64, "institution_count": pl.Int32,
        },
        orient="row",
    )


def analyst_time_median(frame, asof, half_life=60):
    weighted = (
        frame.with_columns(
            (pl.lit(asof.date()) - pl.col("create_date"))
            .dt.total_days().alias("age")
        )
        .with_columns(
            (-np.log(2) * pl.col("age") / half_life)
            .exp().alias("weight")
        )
        .with_columns(
            (pl.col("forecast_np") * pl.col("weight")).alias("weighted_np")
        )
    )
    analysts = (
        weighted.group_by(["tick", "organ_id", "author_id", "report_year"])
        .agg(
            pl.col("weighted_np").sum().alias("weighted_sum"),
            pl.col("weight").sum().alias("weight_sum"),
        )
        .with_columns(
            (pl.col("weighted_sum") / pl.col("weight_sum"))
            .alias("analyst_forecast")
        )
    )
    institutions = analysts.group_by(
        ["tick", "organ_id", "report_year"]
    ).agg(pl.col("analyst_forecast").mean().alias("institution_forecast"))
    return institutions.group_by(["tick", "report_year"]).agg(
        pl.col("institution_forecast").median().alias("consensus_np"),
        pl.col("organ_id").n_unique().alias("institution_count"),
    )


def to_dict(frame, year):
    return {
        tick: value
        for tick, value in frame.filter(
            pl.col("report_year") == year
        ).select("tick", "consensus_np").iter_rows()
    }


def make_ep(consensus, asof, mv_row, tick_positions, positive, min_count, ntm):
    result = np.full(len(mv_row), np.nan)
    if min_count:
        consensus = consensus.filter(
            pl.col("institution_count") >= MIN_INSTITUTIONS
        )
    fy1 = to_dict(consensus, asof.year)
    fy2 = to_dict(consensus, asof.year + 1) if ntm else {}
    weight = min(
        max((pd.Timestamp(asof.year, 12, 31) - asof.normalize()).days, 0)
        / 365,
        1,
    )
    for tick, np1 in fy1.items():
        if ntm:
            np2 = fy2.get(tick)
            if np2 is None:
                continue
            earnings = weight * np1 + (1 - weight) * np2
        else:
            earnings = np1
        position = tick_positions.get(tick)
        if position is None or not np.isfinite(mv_row[position]) or mv_row[position] <= 0:
            continue
        if positive and earnings <= 0:
            continue
        result[position] = earnings / mv_row[position]
    return result


def build(reports, dates, market_value, tick_positions):
    names = ["v0_raw", "v1_positive", "v2_min3", "v3_broker_mean",
             "v4_broker_median", "v5_fresh90", "v6_ntm", "v7_analyst_time_ntm"]
    factors = {name: np.full_like(market_value, np.nan, dtype=float) for name in names}
    for i, date in enumerate(dates):
        asof = pd.Timestamp(date)
        frame = visible(reports, asof)
        analyst_mean = analyst_institution_mean(frame)
        broker_mean = broker_latest(frame, "mean")
        broker_median = broker_latest(frame, "median")
        fresh90 = fresh_broker_median(frame, asof)
        analyst_time = analyst_time_median(frame, asof)
        specifications = {
            "v0_raw": (analyst_mean, False, False, False),
            "v1_positive": (analyst_mean, True, False, False),
            "v2_min3": (analyst_mean, True, True, False),
            "v3_broker_mean": (broker_mean, True, True, False),
            "v4_broker_median": (broker_median, True, True, False),
            "v5_fresh90": (fresh90, True, True, False),
            "v6_ntm": (broker_median, True, True, True),
            "v7_analyst_time_ntm": (analyst_time, True, True, True),
        }
        for name, spec in specifications.items():
            factors[name][i] = make_ep(
                spec[0], asof, market_value[i], tick_positions,
                positive=spec[1], min_count=spec[2], ntm=spec[3],
            )
        if i % 25 == 0:
            print(f"factor dates: {i}/{len(dates)}", flush=True)
    return factors


def forward_return(daily_return, horizon):
    output = np.full_like(daily_return, np.nan, dtype=float)
    windows = np.lib.stride_tricks.sliding_window_view(
        daily_return[2:], horizon, axis=0
    )
    output[:len(windows)] = np.prod(1 + windows, axis=-1) - 1
    return output


def evaluate(name, signal, returns, tradable, dates):
    signal = np.where(tradable, signal, np.nan)
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for ax, horizon in zip(axes.flat, HORIZONS):
        label = forward_return(returns, horizon)
        ic, ric = IC(signal, label), rankIC(signal, label)
        groups = calc_group_ret(signal, label, 10)
        means = np.nanmean(groups, axis=1)
        cumulative = np.nancumsum(groups, axis=1)
        for group, values in enumerate(cumulative, 1):
            ax.plot(dates[:len(values)], values, color=colors[group - 1],
                    linewidth=1.0, label=f"G{group}")
        rows.append({
            "factor": name, "horizon": horizon,
            "mean_ic": np.nanmean(ic),
            "icir": np.nanmean(ic) / np.nanstd(ic) * np.sqrt(242),
            "mean_rank_ic": np.nanmean(ric),
            "rank_icir": np.nanmean(ric) / np.nanstd(ric) * np.sqrt(242),
            "g10_g1": means[9] - means[0],
            "g10_g8": means[9] - means[7],
            "g9_g5": means[8] - means[4],
            "g10_return": means[9],
            "monotonic_corr": np.corrcoef(np.arange(1, 11), means)[0, 1],
            "adjacent_order_rate": np.mean(np.diff(means) > 0),
            "mean_coverage": np.mean(np.sum(np.isfinite(signal), axis=1)),
        })
        ax.set_title(f"{name} {horizon}D | RankIC={np.nanmean(ric):.4f}")
        ax.grid(alpha=.25)
        ax.legend(ncol=2, fontsize=7)
    fig.suptitle(f"{name}: EP Ablation Deciles (Latest 3Y)")
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{name}_group_returns_3y.png", dpi=140, bbox_inches="tight")
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
    start_pos = int(np.searchsorted(all_dates, np.datetime64(start.date())))
    dates = all_dates[start_pos:end_pos + 1]
    market_value = np.asarray(
        data.read("d_essentials/total_mv", end_pos, start_date=start_pos), dtype=float
    )
    returns = np.asarray(
        data.read("d_essentials/pct", end_pos, start_date=start_pos), dtype=float
    ) / 100
    tradable = np.asarray(
        data.read("basic/tradable", end_pos, start_date=start_pos), dtype=bool
    )
    query_start = (start - pd.Timedelta(days=LOOKBACK)).date()
    print(f"query reports: {query_start} to {end.date()}", flush=True)
    reports = load_reports(query_start, end.date())
    factors = build(reports, dates, market_value, data.axis._tick_positions)
    rows = []
    for name, signal in factors.items():
        rows.extend(evaluate(name, signal, returns, tradable, dates))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT / "ep_fy1_ablation_summary_3y.csv", index=False)
    print(summary[summary.horizon == 20].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()