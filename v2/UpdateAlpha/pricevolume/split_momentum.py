"""Intraday/overnight cut momentum from Kaiyuan microstructure series (4)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__:
    from ..alphabase import AlphaBase, AlphaContext, AlphaMeta
    from ...GetData import DataPool
    from ...UpdateData.config import ROOT
    from ...ResearchFlow.FactorTest.metrics import IC, rankIC, calc_group_ret
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from v2.UpdateAlpha.alphabase import AlphaBase, AlphaContext, AlphaMeta
    from v2.ResearchFlow.FactorTest.metrics import IC, rankIC, calc_group_ret
    from v2.GetData import DataPool
    from v2.UpdateData.config import ROOT

DEFAULT_ROOT = Path("Z:/") if Path("Z:/axis/dates.npy").is_file() else ROOT


@dataclass(frozen=True)
class IntradayOvernightMomentumConfig:
    lookback_days: int = 20
    min_valid_days: int = 20


class IntradayOvernightMomentumContext(AlphaContext):
    def __init__(self, root=DEFAULT_ROOT, config=IntradayOvernightMomentumConfig()):
        self.config = config
        super().__init__(DataPool(root, asset="stock"))


def _ratio_return(numerator, denominator):
    numerator = np.asarray(numerator, float)
    denominator = np.asarray(denominator, float)
    return np.divide(
        numerator, denominator, out=np.full_like(numerator, np.nan),
        where=(np.isfinite(numerator) & np.isfinite(denominator)
               & (numerator > 0) & (denominator > 0)),
    ) - 1.0


class IntradayOvernightMomentumFactor(AlphaBase):
    """M0-M1: 20D intraday momentum minus overnight reversal."""

    meta = AlphaMeta(
        "intraday_overnight_momentum",
        "20D intraday momentum minus overnight reversal",
        direction=1,
    )
    dependencies = (
        "d_essentials/open_adj",
        "d_essentials/close_adj",
        "basic/tradable",
    )

    def calculate(self, asof):
        cfg, data = self.context.config, self.context.data
        axis = data.axis
        end = axis.date_position(pd.Timestamp(asof).date())
        start = end - cfg.lookback_days + 1
        previous = start - 1
        if previous < 0:
            return np.full(axis.tick_count, np.nan, np.float32)

        open_adj = np.asarray(data.read("d_essentials/open_adj", end, previous), float)
        close_adj = np.asarray(data.read("d_essentials/close_adj", end, previous), float)
        tradable = np.asarray(data.read("basic/tradable", end, start), bool)
        intraday = _ratio_return(close_adj[1:], open_adj[1:])
        overnight = _ratio_return(open_adj[1:], close_adj[:-1])
        valid = tradable & np.isfinite(intraday) & np.isfinite(overnight)
        result = (
            np.sum(np.where(valid, intraday, 0), axis=0)
            - np.sum(np.where(valid, overnight, 0), axis=0)
        )
        result[valid.sum(axis=0) < cfg.min_valid_days] = np.nan
        return result.astype(np.float32)


__all__ = [
    "IntradayOvernightMomentumConfig",
    "IntradayOvernightMomentumContext",
    "IntradayOvernightMomentumFactor",
]



if __name__ == "__main__":
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # Inclusive date range. Use None to select the first/last available date.
    START_DATE = "2025-01-01"
    END_DATE = "2026-06-30"

    with IntradayOvernightMomentumContext() as context:
        alpha = IntradayOvernightMomentumFactor(context)
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

        for trade_date in tqdm(selected_dates, desc="Updating SplitMom"):
            alpha.update(trade_date)

        pred = context.data.load("factor_pool/intraday_overnight_momentum").copy()
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
                f"SplitMom {horizon}D Forward Return | "
                f"Mean IC={mean_ic:.4f}, Mean RankIC={mean_rank_ic:.4f}"
            )
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
            ax.grid(alpha=0.25)
            ax.legend(ncol=2, fontsize=8)
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(
                ax.xaxis.get_major_locator()
            ))

        fig.suptitle("SplitMom Decile Cumulative Excess Returns", fontsize=15)
        fig.supxlabel("Trade Date")
        fig.supylabel("Cumulative Group Excess Return")
        fig.tight_layout()

        range_tag = f"{date_index[start_idx]:%Y%m%d}_{date_index[end_idx]:%Y%m%d}"
        output = (
            Path(__file__).resolve().parents[1]
            / "output"
            / f"splitmom_group_ret_{range_tag}.png"
        )
        # output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"saved: {output}")
