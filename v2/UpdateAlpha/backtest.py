"""Unified calculation and grouped-return plots for registered alpha factors.

Examples
--------
List factors:
    python -m v2.UpdateAlpha.backtest --list

Calculate and plot:
    python -m v2.UpdateAlpha.backtest suef w_cut_reversal --start 2017-01-01

Plot an existing factor matrix without recalculating:
    python -m v2.UpdateAlpha.backtest suef --plot-only
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

if __package__:
    from . import FACTOR_REGISTRY, get_factor_spec
    from .alphabase import AlphaBase, AlphaContext
    from ..ResearchFlow.FactorTest.metrics import IC, rankIC, calc_group_ret
    from ..UpdateData.config import ROOT
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from v2.UpdateAlpha import FACTOR_REGISTRY, get_factor_spec
    from v2.UpdateAlpha.alphabase import AlphaBase, AlphaContext
    from v2.ResearchFlow.FactorTest.metrics import IC, rankIC, calc_group_ret
    from v2.UpdateData.config import ROOT


DEFAULT_ROOT = Path("Z:/") if Path("Z:/axis/dates.npy").is_file() else ROOT
DEFAULT_HORIZONS = (1, 5, 10, 20)


def select_period(context: AlphaContext, start_date=None, end_date=None):
    """Return inclusive trade dates and their global axis positions."""

    trade_dates = context.data["trade_dates"]
    date_index = pd.DatetimeIndex(trade_dates)
    start = pd.Timestamp(start_date) if start_date else date_index[0]
    end = pd.Timestamp(end_date) if end_date else date_index[-1]

    if start > end:
        raise ValueError("start_date must not be later than end_date")
    selected = np.flatnonzero((date_index >= start) & (date_index <= end))
    if selected.size == 0:
        raise ValueError("no trade dates found in the requested range")
    
    start_idx, end_idx = int(selected[0]), int(selected[-1])
    return trade_dates[start_idx:end_idx + 1], start_idx, end_idx


def _forward_label(
    daily_return,
    pred_shape,
    start_idx,
    end_idx,
    horizon,
    return_offset,
):
    """Build labels using pct[t + offset : t + offset + horizon]."""
    windows = np.lib.stride_tricks.sliding_window_view(
        daily_return[return_offset:], horizon, axis=0
    )
    forward_return = np.prod(1.0 + windows, axis=-1) - 1.0
    label = np.full(pred_shape, np.nan, dtype=np.float64)
    selected = forward_return[start_idx:end_idx + 1]
    label[:len(selected)] = selected
    return label


def plot_group_ret(
    name,
    context,
    start_date=None,
    end_date=None,
    *,
    factor_class=None,
    folder="factor_pool",
    horizons=DEFAULT_HORIZONS,
    num_groups=10,
    return_offset=2,
    output_dir=None,
):
    """Plot cumulative demeaned group returns for an existing factor matrix."""

    selected_dates, start_idx, end_idx = select_period(
        context, start_date, end_date
    )
    factor_path = f"{folder}/{name}"
    pred = context.data.load(factor_path)[
        start_idx:end_idx + 1, :context.data.axis.tick_count
    ].copy()
    tradable = context.data.read(
        "basic/tradable", start_date=start_idx, end_date=end_idx
    )
    pred = np.where(tradable, pred, np.nan)

    direction = factor_class.meta.direction if factor_class is not None else 1
    test_pred = pred * direction
    daily_return = context.data.read(
        "d_essentials/pct",
        start_date=0,
        end_date=context.data.axis.date_count - 1,
    ) / 100.0

    horizons = tuple(int(value) for value in horizons)
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("horizons must contain positive integers")
    if return_offset < 1:
        raise ValueError("return_offset must be at least 1")

    columns = 2
    rows = int(np.ceil(len(horizons) / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(16, 5 * rows), sharex=True, squeeze=False
    )
    colors = plt.cm.tab10(np.linspace(0, 1, num_groups))

    stats = []
    for ax, horizon in zip(axes.flat, horizons):
        label = _forward_label(
            daily_return,
            test_pred.shape,
            start_idx,
            end_idx,
            horizon,
            return_offset,
        )
        ic = IC(test_pred, label)
        rank_ic = rankIC(test_pred, label)
        group_return = calc_group_ret(test_pred, label, num_groups)
        cumulative_return = np.nancumsum(group_return, axis=1)

        for group, values in enumerate(cumulative_return, start=1):
            suffix = (
                " (Low)" if group == 1
                else " (High)" if group == num_groups
                else ""
            )
            ax.plot(
                selected_dates,
                values,
                color=colors[group - 1],
                linewidth=1.2,
                label=f"Group {group}{suffix}",
            )

        mean_ic = float(np.nanmean(ic))
        mean_rank_ic = float(np.nanmean(rank_ic))
        stats.append(
            {
                "factor": name,
                "horizon": horizon,
                "mean_ic": mean_ic,
                "mean_rank_ic": mean_rank_ic,
            }
        )
        ax.set_title(
            f"{name.upper()} {horizon}D | "
            f"Mean IC={mean_ic:.4f}, Mean RankIC={mean_rank_ic:.4f}"
        )
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.grid(alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    for ax in axes.flat[len(horizons):]:
        ax.set_visible(False)

    orientation = "meta.direction adjusted" if direction != 1 else "raw direction"
    fig.suptitle(
        f"{name.upper()} {num_groups}-Group Cumulative Excess Returns "
        f"({orientation})",
        fontsize=15,
    )
    fig.supxlabel("Trade Date")
    fig.supylabel("Cumulative Group Excess Return")
    fig.tight_layout()

    destination = (
        Path(output_dir)
        if output_dir is not None
        else Path.cwd() / "output" / name
    )
    destination.mkdir(parents=True, exist_ok=True)
    first_date = pd.Timestamp(selected_dates[0])
    last_date = pd.Timestamp(selected_dates[-1])
    range_tag = f"{first_date:%Y%m%d}_{last_date:%Y%m%d}"
    output = destination / f"{name}_group_ret_{range_tag}.png"
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {output}")
    return output, pd.DataFrame(stats)


def run_registered(
    name,
    start_date=None,
    end_date=None,
    *,
    factor_class=None,
    context_class=None,
    category="custom",
    root=DEFAULT_ROOT,
    calculate=True,
    folder="factor_pool",
    horizons=DEFAULT_HORIZONS,
    num_groups=10,
    return_offset=2,
    output_dir=None,
    context_kwargs=None,
):
    """Calculate and/or plot one registered factor."""

    spec = get_factor_spec(
        name,
        factor_class=factor_class,
        context_class=context_class,
        category=category,
    )
    kwargs = dict(context_kwargs or {})
    kwargs.setdefault("root", root)
    
    with spec.context_class(**kwargs) as context:
        factor = spec.factor_class(context)
        if calculate:
            selected_dates, _, _ = select_period(context, start_date, end_date)
            for trade_date in tqdm(
                selected_dates, desc=f"Updating {factor.meta.name}"
            ):
                factor.update(trade_date, folder=folder)

        return plot_group_ret(
            factor.meta.name,
            context,
            start_date,
            end_date,
            factor_class=spec.factor_class,
            folder=folder,
            horizons=horizons,
            num_groups=num_groups,
            return_offset=return_offset,
            output_dir=output_dir,
        )


def list_factors():
    return pd.DataFrame(
        [
            {
                "name": name,
                "category": spec.category,
                "context": spec.context_class.__name__,
                "factor": spec.factor_class.__name__,
                "direction": spec.factor_class.meta.direction,
                "description": spec.factor_class.meta.description,
            }
            for name, spec in sorted(FACTOR_REGISTRY.items())
        ]
    )





def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("factors", nargs="*", help="registered factor meta names")
    parser.add_argument("--list", action="store_true", help="list registered factors")
    parser.add_argument("--start", help="inclusive start date")
    parser.add_argument("--end", help="inclusive end date")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--folder", default="factor_pool")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="skip calculation and plot existing factor matrices",
    )
    parser.add_argument("--output", help="output directory for plots")
    parser.add_argument(
        "--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS)
    )
    parser.add_argument("--groups", type=int, default=10)
    parser.add_argument(
        "--return-offset",
        type=int,
        default=2,
        help="first pct row relative to signal t; project default is 2",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.list:
        print(list_factors().to_string(index=False))
        return 0
    if not args.factors:
        raise SystemExit("provide at least one factor name or use --list")

    for name in args.factors:
        output, stats = run_registered(
            name,
            args.start,
            args.end,
            root=Path(args.root),
            calculate=not args.plot_only,
            folder=args.folder,
            horizons=args.horizons,
            num_groups=args.groups,
            return_offset=args.return_offset,
            output_dir=args.output,
        )
        print(stats.to_string(index=False))
        print(f"completed: {output}")
    return 0


def run_from_ide(
    factors,
    *,
    start_date=None,
    end_date=None,
    root=DEFAULT_ROOT,
    calculate=True,
    folder="factor_pool",
    horizons=DEFAULT_HORIZONS,
    num_groups=10,
    return_offset=2,
    output_dir=None,
):
    """Run one or more factors directly from Python without parsing argv."""

    results = {}
    for name in factors:
        output, stats = run_registered(
            name,
            start_date,
            end_date,
            root=Path(root),
            calculate=calculate,
            folder=folder,
            horizons=horizons,
            num_groups=num_groups,
            return_offset=return_offset,
            output_dir=output_dir,
        )
        results[name] = {"output": output, "stats": stats}
        print(stats.to_string(index=False))
        print(f"completed: {output}")
    return results


if __name__ == "__main__":
    # IDE direct-run configuration. Factor names are shown by list_factors().
    FACTORS = ("apm",)
    START_DATE = "2017-01-01"
    END_DATE = "2026-06-30"

    # True: calculate factor values first and then plot.
    # False: plot an existing factor_pool matrix only.
    CALCULATE = True

    run_from_ide(
        FACTORS,
        start_date=START_DATE,
        end_date=END_DATE,
        root=DEFAULT_ROOT,
        calculate=CALCULATE,
        folder="factor_pool",
        horizons=(1, 5, 10, 20),
        num_groups=10,
        return_offset=2,
        output_dir=None,
    )
