"""Compare COV and excess-momentum-neutralized COV group returns."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v2.ResearchFlow.FactorTest.metrics import IC, calc_group_ret, rankIC
from v2.UpdateAlpha.analyst_forecast.cov import (
    COVContext,
    COVFactor,
    CovExMomFactor,
)


def _forward_returns(pct, horizon):
    """Match the repository convention: signal t uses t+2...t+1+h."""
    windows = np.lib.stride_tricks.sliding_window_view(
        pct[2:], horizon, axis=0,
    )
    result = np.full(pct.shape, np.nan, dtype=np.float64)
    values = np.prod(1.0 + windows, axis=-1) - 1.0
    result[:len(values)] = values
    return result


def run_backtest(
    root, years=3, horizons=(1, 5, 10, 20), output=None,
    start_date=None, end_date=None,
):
    root = Path(root)
    horizons = tuple(horizons)
    output = Path(output) if output else (
        Path(__file__).resolve().parent / "cov" / "cov_vs_cov_ex_mom.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    with COVContext(root=root) as context:
        axis = context.data.axis
        all_dates = axis.trade_dates
        latest_end = axis.date_count - max(horizons) - 1
        if end_date is None:
            end = latest_end
        else:
            end_target = np.datetime64(pd.Timestamp(end_date).date(), "D")
            end = min(
                int(np.searchsorted(all_dates, end_target, side="right")),
                latest_end,
            )
        if start_date is None:
            cutoff = all_dates[end - 1] - np.timedelta64(365 * years, "D")
            start = int(np.searchsorted(all_dates, cutoff, side="left"))
        else:
            start_target = np.datetime64(pd.Timestamp(start_date).date(), "D")
            start = int(np.searchsorted(all_dates, start_target, side="left"))
        if start >= end:
            raise ValueError("backtest start date must be before end date")
        rows = np.arange(start, end, dtype=np.int64)
        dates = all_dates[rows]

        pct = np.asarray(
            context.data.load("d_essentials/pct")[
                :axis.date_count, :axis.tick_count
            ],
            dtype=np.float64,
        ) / 100.0
        labels = {
            horizon: _forward_returns(pct, horizon)[rows]
            for horizon in horizons
        }
        tradable = np.asarray(
            context.data.load("basic/tradable")[rows, :axis.tick_count],
            dtype=bool,
        )

        factors = {
            "COV": COVFactor(context),
            "COV-ExMom": CovExMomFactor(context),
        }
        values = {
            name: np.full((len(rows), axis.tick_count), np.nan)
            for name in factors
        }

        for local_row, date in enumerate(tqdm(dates, desc="COV backtest")):
            for name, factor in factors.items():
                cross_section = factor.calculate(date)
                values[name][local_row] = np.where(
                    tradable[local_row], cross_section, np.nan,
                )

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    summary = []
    plot_paths = {}

    for name, alpha in values.items():
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
        for horizon_index, horizon in enumerate(horizons):
            ax = axes.flat[horizon_index]
            label = labels[horizon]
            group_return = calc_group_ret(alpha, label, 10)
            cumulative = np.nancumsum(group_return, axis=1)
            ic = IC(alpha, label)
            rank_ic = rankIC(alpha, label)

            for group, series in enumerate(cumulative, start=1):
                suffix = (
                    " (Low)" if group == 1
                    else " (High)" if group == 10
                    else ""
                )
                ax.plot(
                    dates,
                    series,
                    color=colors[group - 1],
                    linewidth=1.2,
                    label=f"Group {group}{suffix}",
                )

            long_short = cumulative[-1, -1] - cumulative[0, -1]
            summary.append({
                "factor": name,
                "horizon": horizon,
                "mean_ic": np.nanmean(ic),
                "mean_rank_ic": np.nanmean(rank_ic),
                "high_minus_low": long_short,
            })
            ax.set_title(
                f"{name} {horizon}D | IC={np.nanmean(ic):.4f}, "
                f"RankIC={np.nanmean(rank_ic):.4f}, H-L={long_short:.4f}"
            )
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
            ax.grid(alpha=0.25)
            ax.legend(ncol=2, fontsize=8)
            locator = mdates.AutoDateLocator()
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

        fig.suptitle(f"{name} Decile Excess Returns")
        fig.supxlabel("Trade date")
        fig.supylabel("Cumulative group excess return")
        fig.tight_layout()
        slug = name.lower().replace("-", "_")
        plot_path = output.with_name(f"{output.stem}_{slug}.png")
        fig.savefig(plot_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        plot_paths[name] = plot_path

    summary_path = output.with_suffix(".csv")
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    return plot_paths, summary_path, pd.DataFrame(summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="Z:/")
    parser.add_argument("--years", type=int, default=3)
    parser.add_argument(
        "--horizons", type=int, nargs="+", default=[1, 5, 10, 20],
    )
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--output")
    args = parser.parse_args()
    plot_paths, summary_path, summary = run_backtest(
        args.root, args.years, args.horizons, args.output,
        args.start_date, args.end_date,
    )
    print(summary.to_string(index=False))
    for name, path in plot_paths.items():
        print(f"saved {name} plot: {path}")
    print(f"saved summary: {summary_path}")


if __name__ == "__main__":
    main()
