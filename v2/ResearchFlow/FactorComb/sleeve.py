"""Branch A: convert family signals into sleeves, then merge sleeves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import SleeveConfig
from ..matrix_math import calc_group_weights
from .allocation import AllocationParams, CapitalAllocator


@dataclass(frozen=True)
class SleeveResult:
    sleeve_stock_weights: np.ndarray
    sleeve_capital_weights: np.ndarray
    merged_score: np.ndarray
    sleeve_returns: np.ndarray | None


class SleevePath:
    """Build per-family sleeves and allocate capital across them."""

    def __init__(self, config: SleeveConfig | None = None) -> None:
        self.config = config or SleeveConfig()
        self.allocator = CapitalAllocator(
            AllocationParams(
                method=self.config.allocation_method,
                lookback=self.config.lookback,
                min_periods=self.config.min_periods,
                max_weight=self.config.max_sleeve_weight,
                smoothing=self.config.smoothing,
                return_shrinkage=self.config.return_shrinkage,
                covariance_shrinkage=self.config.covariance_shrinkage,
                turnover_penalty=self.config.turnover_penalty,
                risk_aversion=self.config.risk_aversion,
            )
        )

    def run(
        self,
        family_scores: np.ndarray,
        *,
        labels: np.ndarray | None = None,
        tradable: np.ndarray | None = None,
    ) -> SleeveResult:
        if family_scores.ndim != 3:
            raise ValueError("family_scores must have shape T x N x F")
        mask = np.ones(family_scores.shape[:2], dtype=bool) if tradable is None else tradable.astype(bool)
        sleeve_weights = self._build_sleeves(family_scores, mask)
        sleeve_returns = self._sleeve_returns(sleeve_weights, labels) if labels is not None else None
        capital = self.allocator.allocate(sleeve_returns, family_scores.shape[0], family_scores.shape[2])
        merged = np.sum(sleeve_weights * capital[:, None, :], axis=2)
        return SleeveResult(
            sleeve_stock_weights=sleeve_weights,
            sleeve_capital_weights=capital,
            merged_score=merged,
            sleeve_returns=sleeve_returns,
        )

    def _build_sleeves(self, family_scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
        n_groups = max(int(round(1.0 / self.config.quantile)), 2)
        out = np.zeros_like(family_scores, dtype=float)
        for k in range(family_scores.shape[2]):
            score = np.where(mask, family_scores[:, :, k], np.nan)
            out[:, :, k] = calc_group_weights(score, num_group=n_groups, long_only=self.config.long_only)
        return out

    @staticmethod
    def _sleeve_returns(sleeve_weights: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return np.nansum(sleeve_weights * labels[:, :, None], axis=1)
