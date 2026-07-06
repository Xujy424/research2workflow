"""Branch A: convert family signals into sleeves, then merge sleeves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import SleeveConfig
from ..matrix_math import cap_and_renormalize


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
        capital = self._capital_weights(sleeve_returns, family_scores.shape[0], family_scores.shape[2])
        merged = np.sum(sleeve_weights * capital[:, None, :], axis=2)
        return SleeveResult(
            sleeve_stock_weights=sleeve_weights,
            sleeve_capital_weights=capital,
            merged_score=merged,
            sleeve_returns=sleeve_returns,
        )

    def _build_sleeves(self, family_scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
        t_count, n_stocks, n_families = family_scores.shape
        out = np.zeros((t_count, n_stocks, n_families), dtype=float)
        q = self.config.quantile
        for t in range(t_count):
            for k in range(n_families):
                score = np.where(mask[t], family_scores[t, :, k], np.nan)
                valid = np.isfinite(score)
                if valid.sum() < 10:
                    continue
                upper = np.nanquantile(score, 1.0 - q)
                long = valid & (score >= upper)
                if self.config.long_only:
                    if long.any():
                        out[t, long, k] = 1.0 / long.sum()
                    continue
                lower = np.nanquantile(score, q)
                short = valid & (score <= lower)
                if long.any():
                    out[t, long, k] = 0.5 / long.sum()
                if short.any():
                    out[t, short, k] = -0.5 / short.sum()
        return out

    @staticmethod
    def _sleeve_returns(sleeve_weights: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return np.nansum(sleeve_weights * labels[:, :, None], axis=1)

    def _capital_weights(self, sleeve_returns: np.ndarray | None, n_dates: int, n_sleeves: int) -> np.ndarray:
        if sleeve_returns is None or self.config.allocation_method == "equal":
            return np.full((n_dates, n_sleeves), 1.0 / n_sleeves, dtype=float)
        if self.config.allocation_method != "icir":
            raise ValueError(f"unsupported sleeve allocation method: {self.config.allocation_method}")
        out = np.full((n_dates, n_sleeves), 1.0 / n_sleeves, dtype=float)
        prev = out[0]
        for t in range(n_dates):
            history = sleeve_returns[max(0, t - self.config.lookback):t]
            if len(history) < self.config.min_periods:
                out[t] = prev
                continue
            mean = np.nanmean(history, axis=0)
            std = np.nanstd(history, axis=0)
            score = np.divide(mean, std, out=np.zeros(n_sleeves), where=std > 1e-12)
            score = np.maximum(score, 0.0)
            if score.sum() <= 1e-12:
                target = prev
            else:
                target = cap_and_renormalize(score / score.sum(), max_weight=self.config.max_sleeve_weight)
            out[t] = self.config.smoothing * prev + (1.0 - self.config.smoothing) * target
            out[t] = cap_and_renormalize(out[t], max_weight=self.config.max_sleeve_weight)
            prev = out[t]
        return out

