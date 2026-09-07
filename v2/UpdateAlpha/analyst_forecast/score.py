"""Point-in-time analyst-consensus rating upgrade/downgrade events.

The local con_rating_strength series is the vendor-maintained consensus
rating based on ratings available in the trailing 180 calendar days. A rise
is an upgrade event and a fall is a downgrade event. Factors are event
impulses: zero means no event, positive means upgrade, and negative means
downgrade. Backtests must trade with at least a one-day lag.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..alphabase import AlphaBase, AlphaContext, AlphaMeta
from ...GetData import DataPool
from ...UpdateData.config import ROOT


DEFAULT_ROOT = Path("Z:/") if Path("Z:/axis/dates.npy").is_file() else ROOT


@dataclass(frozen=True)
class ScoreEventConfig:
    """Configuration for consensus-rating event detection."""

    rating_field: str = "zyyx/con_rating/con_rating_strength"
    consensus_window_days: int = 180
    min_abs_change: float = 1e-4
    use_change_magnitude: bool = False

    def __post_init__(self):
        if self.consensus_window_days < 1:
            raise ValueError("consensus_window_days must be positive")
        if self.min_abs_change < 0:
            raise ValueError("min_abs_change must be non-negative")


class ScoreEventContext(AlphaContext):
    """Read-only local environment for point-in-time rating events."""

    def __init__(self, root=DEFAULT_ROOT, config=ScoreEventConfig()):
        self.config = config
        super().__init__(DataPool(root, asset="stock"))

    def rating_change(self, asof) -> tuple[np.ndarray, np.ndarray]:
        """Return daily change and the finite current/prior observation mask."""
        axis = self.data.axis
        row = axis.date_position(asof)
        change = np.full(axis.tick_count, np.nan, dtype=np.float64)
        valid = np.zeros(axis.tick_count, dtype=bool)
        if row == 0:
            return change, valid

        ratings = self.data.load(self.config.rating_field)
        current = np.asarray(ratings[row, :axis.tick_count], dtype=np.float64)
        previous = np.asarray(
            ratings[row - 1, :axis.tick_count], dtype=np.float64
        )
        valid = np.isfinite(current) & np.isfinite(previous)
        change[valid] = current[valid] - previous[valid]
        return change, valid


class _ScoreEventFactor(AlphaBase):
    """Base class for sparse consensus-rating event impulses."""

    event_direction = 0
    dependencies = ("zyyx/con_rating/con_rating_strength",)

    def calculate(self, asof):
        change, valid = self.context.rating_change(pd.Timestamp(asof).date())
        threshold = self.context.config.min_abs_change
        out = np.full(change.shape, np.nan, dtype=np.float32)
        out[valid] = 0.0

        if self.event_direction > 0:
            event = valid & (change > threshold)
        elif self.event_direction < 0:
            event = valid & (change < -threshold)
        else:
            event = valid & (np.abs(change) > threshold)

        if self.context.config.use_change_magnitude:
            out[event] = change[event]
        elif self.event_direction > 0:
            out[event] = 1.0
        elif self.event_direction < 0:
            out[event] = -1.0
        else:
            out[event] = np.sign(change[event])
        return out


class ScoreEventFactor(_ScoreEventFactor):
    """Combined event signal: upgrade +1, downgrade -1."""

    meta = AlphaMeta(
        "score_event",
        "180D consensus-rating upgrade (+1) and downgrade (-1) events",
    )


class ScoreUpgradeEventFactor(_ScoreEventFactor):
    """Positive impulse only when consensus rating is upgraded."""

    meta = AlphaMeta(
        "score_upgrade_event",
        "180D consensus-rating upgrade event",
    )
    event_direction = 1


class ScoreDowngradeEventFactor(_ScoreEventFactor):
    """Negative impulse only when consensus rating is downgraded."""

    meta = AlphaMeta(
        "score_downgrade_event",
        "180D consensus-rating downgrade event",
    )
    event_direction = -1


__all__ = [
    "ScoreEventConfig",
    "ScoreEventContext",
    "ScoreEventFactor",
    "ScoreUpgradeEventFactor",
    "ScoreDowngradeEventFactor",
]
