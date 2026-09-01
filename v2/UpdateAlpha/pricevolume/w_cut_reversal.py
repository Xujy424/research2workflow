"""W-cut reversal factor from Kaiyuan Securities microstructure series (1)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__:
    from ..alphabase import AlphaBase, AlphaContext, AlphaMeta
    from ...GetData import DataPool
    from ...ResearchFlow.FactorTest.metrics import IC, rankIC, calc_group_ret
    from ...UpdateData.config import ROOT
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
class WCutConfig:
    lookback_days: int = 20
    high_days: int = 10


class WCutContext(AlphaContext):
    def __init__(self, root=DEFAULT_ROOT, config=WCutConfig()):
        self.config = config
        super().__init__(DataPool(root, asset="stock"))


class WCutReversalFactor(AlphaBase):
    """Return sum on the 10 high average-trade-size days of the past 20 days.

    Average trade size is daily amount divided by daily transaction count. This
    is the report's original, daily-data W-cut. The recommended 13/16
    transaction-level variant cannot be reconstructed from daily/minute bars.
    """

    meta = AlphaMeta(
        "w_cut_reversal",
        "20D W-cut high-average-trade-size return sum",
        direction=-1,
    )
    dependencies = ("d_essentials/amount", "d_essentials/cjbs", "d_essentials/pct")

    def calculate(self, asof):
        cfg = self.context.config
        axis = self.context.data.axis
        end = axis.date_position(pd.Timestamp(asof).date())
        start = end - cfg.lookback_days + 1
        if start < 0:
            return np.full(axis.tick_count, np.nan, dtype=np.float32)

        amount = np.asarray(self.context.data.read("d_essentials/amount", end, start), float)
        count = np.asarray(self.context.data.read("d_essentials/cjbs", end, start), float)
        returns = np.asarray(self.context.data.read("d_essentials/pct", end, start), float) / 100.0
        average_trade = np.divide(
            amount, count,
            out=np.full_like(amount, np.nan),
            where=np.isfinite(amount) & np.isfinite(count) & (count > 0),
        )
        valid = np.isfinite(average_trade) & np.isfinite(returns)
        enough = valid.sum(axis=0) == cfg.lookback_days
        score = np.where(valid, average_trade, -np.inf)
        high_rows = np.argpartition(
            score, -cfg.high_days, axis=0
        )[-cfg.high_days:]
        selected_returns = np.take_along_axis(returns, high_rows, axis=0)
        result = np.sum(selected_returns, axis=0)
        result[~enough] = np.nan
        return result.astype(np.float32)


__all__ = ["WCutConfig", "WCutContext", "WCutReversalFactor", "calculate_w_cut"]




if __name__ == "__main__":
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # Inclusive date range. Use None to select the first/last available date.
    START_DATE = "2022-01-01"
    END_DATE = "2024-12-31"

    with WCutContext() as context:
        alpha = WCutReversalFactor(context)
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

        # for trade_date in tqdm(selected_dates, desc="Updating WCut"):
        #     alpha.update(trade_date)

        pred = context.data.load("factor_pool/w_cut_reversal").copy()
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
                f"WCut {horizon}D Forward Return | "
                f"Mean IC={mean_ic:.4f}, Mean RankIC={mean_rank_ic:.4f}"
            )
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
            ax.grid(alpha=0.25)
            ax.legend(ncol=2, fontsize=8)
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(
                ax.xaxis.get_major_locator()
            ))

        fig.suptitle("WCut Decile Cumulative Excess Returns", fontsize=15)
        fig.supxlabel("Trade Date")
        fig.supylabel("Cumulative Group Excess Return")
        fig.tight_layout()

        range_tag = f"{date_index[start_idx]:%Y%m%d}_{date_index[end_idx]:%Y%m%d}"
        output = (
            Path(__file__).resolve().parents[1]
            / "output"
            / f"wcut_group_ret_{range_tag}.png"
        )
        # output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"saved: {output}")
