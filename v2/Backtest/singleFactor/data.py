"""Explicit, aligned input contract for every strategy."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorData:
    signal: pd.DataFrame
    returns: pd.DataFrame
    tradable: pd.DataFrame | None = None
    industry: pd.DataFrame | None = None
    benchmark_weight: pd.DataFrame | None = None
    market_weight: pd.DataFrame | None = None

    def aligned(self) -> "FactorData":
        if not self.signal.index.is_monotonic_increasing:
            raise ValueError("signal dates must be increasing")
        if self.signal.index.has_duplicates or self.signal.columns.has_duplicates:
            raise ValueError("signal axes must be unique")
        optional = (self.tradable, self.industry, self.benchmark_weight,
                    self.market_weight)
        for frame in (self.returns, *[x for x in optional if x is not None]):
            if not self.signal.index.equals(frame.index):
                raise ValueError("all inputs must have identical date axes")
            if not self.signal.columns.equals(frame.columns):
                raise ValueError("all inputs must have identical instrument axes")
        tradable = self.tradable
        if tradable is None:
            tradable = pd.DataFrame(True, self.signal.index, self.signal.columns)
        return FactorData(self.signal.astype(float), self.returns.astype(float),
                          tradable.astype(bool), self.industry,
                          self.benchmark_weight, self.market_weight)

    @property
    def shape(self):
        return self.signal.shape
