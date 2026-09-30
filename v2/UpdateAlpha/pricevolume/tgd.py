"""TGD factor from Kaiyuan microstructure series (19)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__:
    from ..alphabase import AlphaBase, AlphaContext, AlphaMeta
    from ..operators import cross_sectional_residual, safe_ratio_return
    from ...GetData import DataPool
    from ...UpdateData.config import ROOT
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from v2.UpdateAlpha.alphabase import AlphaBase, AlphaContext, AlphaMeta
    from v2.UpdateAlpha.operators import cross_sectional_residual, safe_ratio_return
    from v2.GetData import DataPool
    from v2.UpdateData.config import ROOT


DEFAULT_ROOT = Path("Z:/") if Path("Z:/axis/dates.npy").is_file() else ROOT


@dataclass(frozen=True)
class TGDConfig:
    lookback_days: int = 20
    min_valid_days: int = 12
    min_cross_section_observations: int = 30
    # None uses every same-sign minute; an integer uses the largest N moves.
    extreme_minutes: int | None = 17
    morning_first_end: int = 30
    morning_second_end: int = 60

    def __post_init__(self):
        if self.lookback_days < 1:
            raise ValueError("lookback_days must be positive")
        if not 1 <= self.min_valid_days <= self.lookback_days:
            raise ValueError("min_valid_days must be in [1, lookback_days]")
        if self.extreme_minutes is not None and (
            isinstance(self.extreme_minutes, bool)
            or not isinstance(self.extreme_minutes, (int, np.integer))
            or self.extreme_minutes < 1
        ):
            raise ValueError("extreme_minutes must be positive or None")
        if not 0 < self.morning_first_end < self.morning_second_end:
            raise ValueError("invalid morning segment endpoints")


class TGDContext(AlphaContext):
    def __init__(self, root=DEFAULT_ROOT, config=TGDConfig(), universe="self"):
        self.config = config
        super().__init__(DataPool(root, asset="stock"), universe=universe)

    def read_window(self, asof):
        """Read the complete raw-data window needed for one TGD cross-section."""
        axis = self.data.axis
        end = axis.date_position(pd.Timestamp(asof).date())
        start = end - self.config.lookback_days + 1
        previous = start - 1
        if previous < 0:
            return None

        return {
            "minute_close": np.asarray(
                self.data.read(
                    "m_essentials/close", end_date=end, start_date=start
                ),
                dtype=np.float64,
            ),
            "open_adj": np.asarray(
                self.data.read(
                    "d_essentials/open_adj",
                    end_date=end,
                    start_date=previous,
                ),
                dtype=np.float64,
            ),
            "close_adj": np.asarray(
                self.data.read(
                    "d_essentials/close_adj",
                    end_date=end,
                    start_date=previous,
                ),
                dtype=np.float64,
            ),
            "tradable": np.asarray(
                self.data.read(
                    "basic/tradable", end_date=end, start_date=start
                ),
                dtype=bool,
            ),
        }


class TGDFactor(AlphaBase):
    """20D mean of residualized down-time gravity deviation."""

    meta = AlphaMeta(
        "tgd",
        "20D time-gravity deviation of intraday minute returns",
        direction=1,
    )
    dependencies = (
        "m_essentials/close",
        "d_essentials/open_adj",
        "d_essentials/close_adj",
        "basic/tradable",
    )

    def calculate(self, asof):
        cfg = self.context.config
        tick_count = self.context.data.axis.tick_count
        raw = self.context.read_window(asof)
        if raw is None:
            return np.full(tick_count, np.nan, dtype=np.float32)

        minute_close = raw["minute_close"]
        minute_return = safe_ratio_return(
            minute_close[:, 1:], minute_close[:, :-1]
        )
        if cfg.morning_second_end > minute_return.shape[1]:
            raise ValueError(
                "morning_second_end exceeds available minute return bars "
                f"({minute_return.shape[1]})"
            )
        if (
            cfg.extreme_minutes is not None
            and cfg.extreme_minutes > minute_return.shape[1]
        ):
            raise ValueError(
                "extreme_minutes exceeds available minute return bars "
                f"({minute_return.shape[1]})"
            )

        stamp = np.arange(
            1, minute_return.shape[1] + 1, dtype=np.float64
        )[None, :, None]
        gravity = []
        average_return = []
        for positive in (True, False):
            sign_mask = minute_return > 0 if positive else minute_return < 0
            selected = np.where(sign_mask, minute_return, np.nan)
            weight = np.abs(selected)
            weight_sum = np.nansum(weight, axis=1)
            gravity.append(np.divide(
                np.nansum(weight * stamp, axis=1),
                weight_sum,
                out=np.full_like(weight_sum, np.nan),
                where=weight_sum > 0,
            ))

            available_count = np.isfinite(selected).sum(axis=1)
            if cfg.extreme_minutes is None:
                values = selected
                enough = available_count > 0
            else:
                score = np.where(np.isfinite(selected), weight, -np.inf)
                rows = np.argpartition(
                    score, -cfg.extreme_minutes, axis=1
                )[:, -cfg.extreme_minutes:, :]
                values = np.take_along_axis(selected, rows, axis=1)
                enough = available_count >= cfg.extreme_minutes

            value_count = np.isfinite(values).sum(axis=1)
            mean = np.divide(
                np.nansum(values, axis=1),
                value_count,
                out=np.full_like(value_count, np.nan, dtype=np.float64),
                where=value_count > 0,
            )
            average_return.append(np.where(enough, mean, np.nan))

        segment_returns = []
        for start, end in (
            (0, cfg.morning_first_end),
            (cfg.morning_first_end, cfg.morning_second_end),
        ):
            segment = minute_return[:, start:end, :]
            valid = np.isfinite(segment)
            compounded = (
                np.prod(1.0 + np.where(valid, segment, 0.0), axis=1) - 1.0
            )
            compounded[valid.sum(axis=1) == 0] = np.nan
            segment_returns.append(compounded)

        open_adj = raw["open_adj"]
        prior_close = raw["close_adj"][:-1]
        overnight = safe_ratio_return(open_adj[1:], prior_close)

        daily = np.full((cfg.lookback_days, tick_count), np.nan, dtype=float)
        for i in range(cfg.lookback_days):
            mask = raw["tradable"][i]
            common = [segment_returns[0][i], segment_returns[1][i], overnight[i]]
            first_stage = []
            for dependent, mean_return in (
                (gravity[0][i], average_return[0][i]),
                (gravity[1][i], average_return[1][i]),
            ):
                y = np.where(mask, dependent, np.nan)
                x = np.column_stack([mean_return, *common])
                first_stage.append(cross_sectional_residual(
                    y, x, cfg.min_cross_section_observations
                ))

            y = first_stage[1]
            x = first_stage[0]
            daily[i] = cross_sectional_residual(
                y, x, cfg.min_cross_section_observations
            )

        valid = np.isfinite(daily)
        count = valid.sum(axis=0)
        result = np.divide(
            np.where(valid, daily, 0.0).sum(axis=0),
            count,
            out=np.full(tick_count, np.nan),
            where=count >= cfg.min_valid_days,
        )
        result = np.where(raw["tradable"][-1], result, np.nan)
        return result.astype(np.float32)


__all__ = ["TGDConfig", "TGDContext", "TGDFactor"]
