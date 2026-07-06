"""Final stock-weight projection and execution-aware constraints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import OptimizerConfig
from ..matrix_math import cap_and_renormalize


@dataclass(frozen=True)
class PortfolioProjectionResult:
    weights: np.ndarray
    turnover: np.ndarray
    diagnostics: dict[str, float]


class StockWeightProjector:
    """Fast long-only projection used after either branch produces a raw score."""

    def __init__(self, config: OptimizerConfig | None = None) -> None:
        self.config = config or OptimizerConfig()

    def project(
        self,
        score: np.ndarray,
        *,
        tradable: np.ndarray,
        current_weight: np.ndarray | None = None,
        benchmark_weight: np.ndarray | None = None,
        adv: np.ndarray | None = None,
        industry: np.ndarray | None = None,
    ) -> PortfolioProjectionResult:
        raw = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        raw = np.where(tradable, np.maximum(raw, 0.0), 0.0)
        if benchmark_weight is not None and self.config.benchmark_blend > 0:
            raw = (1.0 - self.config.benchmark_blend) * raw + self.config.benchmark_blend * np.maximum(benchmark_weight, 0.0)
        weights = np.zeros_like(raw)
        turnover = np.zeros(raw.shape[0], dtype=float)
        prev = np.zeros(raw.shape[1], dtype=float) if current_weight is None else np.nan_to_num(current_weight[0], nan=0.0)
        for t in range(raw.shape[0]):
            row = raw[t]
            if row.sum() <= 1e-12:
                eligible = tradable[t].astype(float)
                row = eligible
            w = row / row.sum() if row.sum() > 1e-12 else row
            if industry is not None and self.config.industry_upper is not None:
                w = self._limit_industry(w, industry[t])
            w = cap_and_renormalize(w, max_weight=self.config.max_stock_weight)
            if adv is not None and self.config.max_adv_participation is not None:
                w = self._limit_adv_trade(w, prev, adv[t])
            if self.config.max_turnover is not None:
                w = self._limit_turnover(w, prev, self.config.max_turnover)
            weights[t] = w
            turnover[t] = float(np.abs(w - prev).sum())
            prev = w
        return PortfolioProjectionResult(
            weights=weights,
            turnover=turnover,
            diagnostics={
                "avg_turnover": float(np.nanmean(turnover)),
                "max_turnover": float(np.nanmax(turnover)),
                "avg_holding_count": float(np.nanmean((weights > 1e-12).sum(axis=1))),
            },
        )

    def _limit_industry(self, weights: np.ndarray, industry: np.ndarray) -> np.ndarray:
        upper = self.config.industry_upper
        if upper is None:
            return weights
        out = weights.copy()
        for code in np.unique(industry[np.isfinite(industry)]):
            m = industry == code
            total = out[m].sum()
            if total > upper and total > 0:
                excess = total - upper
                out[m] *= upper / total
                free = ~m & (out > 0)
                if free.any():
                    out[free] += excess * out[free] / out[free].sum()
        return out / out.sum() if out.sum() > 1e-12 else out

    @staticmethod
    def _limit_turnover(target: np.ndarray, current: np.ndarray, max_turnover: float) -> np.ndarray:
        trade = target - current
        gross = np.abs(trade).sum()
        if gross <= max_turnover or gross <= 1e-12:
            return target
        return current + trade * (max_turnover / gross)

    def _limit_adv_trade(self, target: np.ndarray, current: np.ndarray, adv: np.ndarray) -> np.ndarray:
        capacity = np.nan_to_num(adv, nan=0.0, posinf=0.0, neginf=0.0)
        if capacity.max(initial=0.0) > 1.0:
            capacity = capacity / max(capacity.sum(), 1e-12)
        cap = capacity * float(self.config.max_adv_participation or 0.0)
        trade = np.clip(target - current, -cap, cap)
        out = np.maximum(current + trade, 0.0)
        return out / out.sum() if out.sum() > 1e-12 else out


