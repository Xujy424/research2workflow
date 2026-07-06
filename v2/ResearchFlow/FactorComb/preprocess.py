"""Factor-pool preprocessing before family selection."""

from __future__ import annotations

import numpy as np

from ..matrix_math import cross_sectional_zscore, industry_size_neutralize, winsorize


class FactorPoolPreprocessor:
    """Apply the same cross-sectional treatment to each factor matrix."""

    def __init__(
        self,
        *,
        winsor_method: str = "mad",
        standardize: bool = True,
        neutralize: bool = True,
    ) -> None:
        self.winsor_method = winsor_method
        self.standardize = standardize
        self.neutralize = neutralize

    def transform(
        self,
        factors: np.ndarray,
        *,
        tradable: np.ndarray,
        industry: np.ndarray | None = None,
        market_cap: np.ndarray | None = None,
    ) -> np.ndarray:
        if factors.ndim != 3:
            raise ValueError("factors must have shape T x N x K")
        out = np.full_like(factors, np.nan, dtype=float)
        can_neutralize = (
            self.neutralize
            and industry is not None
            and market_cap is not None
            and np.isfinite(industry).any()
            and np.isfinite(market_cap).any()
        )
        for k in range(factors.shape[2]):
            x = winsorize(factors[:, :, k], method=self.winsor_method, mask=tradable)
            if self.standardize:
                x = cross_sectional_zscore(x, mask=tradable)
            if can_neutralize:
                x = industry_size_neutralize(x, industry, market_cap, mask=tradable, standardize=self.standardize)
            out[:, :, k] = x
        return out

