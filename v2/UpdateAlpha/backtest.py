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

from pathlib import Path
import sys
from time import perf_counter
import argparse
import inspect
import json
from dataclasses import is_dataclass, replace

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

if __package__:
    from . import FACTOR_REGISTRY, discover_factors, get_factor_spec
    from .alphabase import AlphaBase, AlphaContext
    from ..ResearchFlow.FactorTest.metrics import IC, rankIC
    from ..ResearchFlow.matrix_math import calc_group_ret
    from ..UpdateData.config import ROOT
    from ..GetData import DataPool
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from v2.UpdateAlpha import FACTOR_REGISTRY, discover_factors, get_factor_spec
    from v2.UpdateAlpha.alphabase import AlphaBase, AlphaContext
    from v2.ResearchFlow.FactorTest.metrics import IC, rankIC
    from v2.ResearchFlow.matrix_math import calc_group_ret
    from v2.UpdateData.config import ROOT
    from v2.GetData import DataPool


DEFAULT_ROOT = Path("Z:/") if Path("Z:/axis/dates.npy").is_file() else ROOT
DEFAULT_HORIZONS = (1, 5, 10, 20)


def _universe_mask(context: AlphaContext, universe, start_idx, end_idx):
    """Return the same point-in-time universe used during calculation."""

    if universe is None or isinstance(universe, str):
        dates = context.data.axis.trade_dates[start_idx:end_idx + 1]
        return np.stack([
            context.factor_universe_mask(day) for day in dates
        ])
    values = np.asarray(universe)
    if values.ndim != 2:
        raise ValueError("universe array must be 2-D date-by-stock")
    return values.astype(bool, copy=False)


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
    label = np.full(pred_shape, np.nan, dtype=np.float64)
    if len(daily_return) - return_offset < horizon:
        return label
    windows = np.lib.stride_tricks.sliding_window_view(
        daily_return[return_offset:], horizon, axis=0
    )
    forward_return = np.prod(1.0 + windows, axis=-1) - 1.0
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
    universe="self",
):
    """Plot cumulative demeaned group returns for an existing factor matrix."""

    selected_dates, start_idx, end_idx = select_period(
        context, start_date, end_date
    )
    factor_path = f"{folder}/{name}"
    pred = context.data.load(factor_path)[
        start_idx:end_idx + 1, :context.data.axis.tick_count
    ].copy()
    
    mask = _universe_mask(context, universe, start_idx, end_idx)
    if mask is not None:
        pred = np.where(mask[:, :context.data.axis.tick_count], pred, np.nan)

    valid = np.isfinite(pred)
    count = valid.sum(axis=1, keepdims=True)
    mean = np.divide(
        np.nansum(np.where(valid, pred, 0.0), axis=1, keepdims=True),
        count,
        out=np.full((pred.shape[0], 1), np.nan),
        where=count > 0,
    )
    centered = pred - mean
    std = np.sqrt(np.divide(
        np.nansum(np.where(valid, centered * centered, 0.0), axis=1, keepdims=True),
        count,
        out=np.full((pred.shape[0], 1), np.nan),
        where=count > 1,
    ))
    pred = np.divide(centered, std, out=np.full_like(pred, np.nan), where=std > 0)
    if num_groups < 2:
        raise ValueError("num_groups must be at least 2")
    usable = np.isfinite(pred).sum(axis=1) >= num_groups
    if not usable.any():
        raise ValueError(
            f"{name}: no date has {num_groups} valid, varying factor values; "
            "check source data, warm-up history, universe and calculation dates"
        )
    pred[~usable] = np.nan

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
        paired = np.isfinite(test_pred) & np.isfinite(label)
        evaluable = paired.sum(axis=1) >= num_groups
        ic, rank_ic = np.full(len(pred), np.nan), np.full(len(pred), np.nan)
        group_return = np.full((num_groups, len(pred)), np.nan)
        if evaluable.any():
            signals = np.where(paired[evaluable], test_pred[evaluable], np.nan)
            labels = label[evaluable]
            ic[evaluable] = IC(signals, labels)
            rank_ic[evaluable] = rankIC(signals, labels)
            group_return[:, evaluable] = calc_group_ret(signals, labels, num_groups)
        else:
            ax.text(.5, .5, "No complete forward-return windows", transform=ax.transAxes,
                    ha="center", va="center")
        cumulative_return = np.nancumsum(group_return, axis=1)

        for group, values in enumerate(cumulative_return if evaluable.any() else (), start=1):
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

        mean_ic = float(np.mean(ic[np.isfinite(ic)])) if np.isfinite(ic).any() else np.nan
        mean_rank_ic = float(np.mean(rank_ic[np.isfinite(rank_ic)])) if np.isfinite(rank_ic).any() else np.nan
        stats.append(
            {
                "factor": name,
                "horizon": horizon,
                "mean_ic": mean_ic,
                "mean_rank_ic": mean_rank_ic,
                "valid_dates": int(np.sum(np.isfinite(group_return).any(axis=0))),
            }
        )
        ax.set_title(
            f"{name.upper()} {horizon}D | "
            f"Mean IC={mean_ic:.4f}, Mean RankIC={mean_rank_ic:.4f}"
        )
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.grid(alpha=0.25)
        if evaluable.any():
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
        else Path(__file__).resolve().parent / "output" / name
    )
    destination.mkdir(parents=True, exist_ok=True)
    first_date = pd.Timestamp(selected_dates[0])
    last_date = pd.Timestamp(selected_dates[-1])
    range_tag = f"{first_date:%Y%m%d}_{last_date:%Y%m%d}"
    output = destination / f"{name}_group_ret_{range_tag}.png"
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {output}")
    stats = pd.DataFrame(stats)
    stats.to_csv(destination / f"{name}_stats_{range_tag}.csv", index=False)
    return output, stats


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
    universe="self",
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
    if "universe" in kwargs:
        raise ValueError("pass universe directly, not through context_kwargs")
    kwargs.setdefault("root", root)
    kwargs["universe"] = universe
    if isinstance(kwargs.get("config"), dict):
        parameter = inspect.signature(spec.context_class).parameters.get("config")
        default = parameter.default if parameter is not None else None
        if not is_dataclass(default):
            raise TypeError(f"{spec.name} does not have a dataclass config default")
        kwargs["config"] = replace(default, **kwargs["config"])

    # Existing matrices need only local data, even for SQL-backed factors.
    context = (spec.create_context(**kwargs) if calculate else
               AlphaContext(DataPool(kwargs["root"], asset="stock"), universe=universe))
    with context:
        factor = spec.factor_class(context)
        if calculate:
            selected_dates, _, _ = select_period(context, start_date, end_date)
            interactive = sys.stdout.isatty()
            for completed, trade_date in enumerate(tqdm(
                selected_dates,
                desc=f"Updating {factor.meta.name}",
                file=sys.stdout,
                disable=not interactive,
                position=0,
            ), start=1):
                started = perf_counter()
                if not interactive:
                    print(
                        f"[{factor.meta.name}] {completed}/{len(selected_dates)} "
                        f"{pd.Timestamp(trade_date):%Y-%m-%d}: calculating",
                        flush=True,
                    )
                factor.update(trade_date, folder=folder)
                if not interactive:
                    print(
                        f"[{factor.meta.name}] {completed}/{len(selected_dates)} "
                        f"completed in {perf_counter() - started:.1f}s",
                        flush=True,
                    )

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
            universe=universe,
        )


def list_factors():
    discover_factors()
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
    universe="self",
    context_kwargs=None,
):
    """Run one or more factors directly from Python without parsing argv."""

    if isinstance(factors, str):
        factors = (factors,)
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
            universe=universe,
            context_kwargs=context_kwargs,
        )
        results[name] = {"output": output, "stats": stats}
        print(stats.to_string(index=False))
        print(f"completed: {output}")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("factors", nargs="*")
    parser.add_argument("--list", action="store_true", help="list valid factor names")
    parser.add_argument("--start", dest="start_date")
    parser.add_argument("--end", dest="end_date")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--folder", default="factor_pool")
    parser.add_argument("--horizons", nargs="+", type=int, default=DEFAULT_HORIZONS)
    parser.add_argument("--num-groups", type=int, default=10)
    parser.add_argument("--return-offset", type=int, default=2)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--universe", default="self")
    parser.add_argument("--config", type=json.loads, help="factor config overrides as a JSON object")
    parser.add_argument("--l2-root", type=Path)
    args = parser.parse_args(argv)
    if args.list:
        print(list_factors().to_string(index=False))
        return
    if not args.factors:
        parser.error("provide a factor name or --list")
    kwargs = {}
    if args.config is not None:
        if not isinstance(args.config, dict):
            parser.error("--config must be a JSON object")
        kwargs["config"] = args.config
    if args.l2_root is not None:
        kwargs["l2_root"] = args.l2_root
    return run_from_ide(
        args.factors, start_date=args.start_date, end_date=args.end_date,
        root=args.root, calculate=not args.plot_only, folder=args.folder,
        horizons=args.horizons, num_groups=args.num_groups,
        return_offset=args.return_offset, output_dir=args.output_dir,
        universe=args.universe, context_kwargs=kwargs,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
        raise SystemExit(0)
    # IDE direct-run configuration. Factor names are shown by list_factors().
    FACTORS = ("bayeshighvol_onset",)  # cov, cov_current, cov_decay_optimized, extra_large_attention, main_force_control, sentiment_instability, vm_diff
    # CONTEXT_KWARGS = {
    #     "config": {"extreme_minutes": None},  # Example for tgd.
    # }
    CONTEXT_KWARGS = None

    START_DATE = "2023-01-01"
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
        universe=None,
        context_kwargs=CONTEXT_KWARGS,
    )







