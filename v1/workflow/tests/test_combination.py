"""Tests that factor-combination weights are stable and free of look-ahead."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from quant_shared.combination import FactorCombiner
from quant_shared.config import CompositeConfig


# 中文说明：`CombinationTest` 验证该场景的预期行为。
class CombinationTest(unittest.TestCase):
    # 中文说明：`test_icir_weights_are_lagged` 验证该场景的预期行为。
    def test_icir_weights_are_lagged(self) -> None:
        dates = pd.bdate_range("2025-01-01", periods=8)
        ic = pd.DataFrame(
            {"good": np.linspace(0.01, 0.08, 8), "bad": -np.linspace(0.01, 0.08, 8)},
            index=dates,
        )
        assets = ["A", "B"]
        index = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])
        factors = pd.DataFrame(
            {"good": np.tile([1.0, -1.0], 8), "bad": np.tile([-1.0, 1.0], 8)},
            index=index,
        )
        _, weights = FactorCombiner(
            CompositeConfig(lookback=3, min_periods=2, weight_smoothing=0.0)
        ).combine(factors, ic)
        self.assertTrue(weights.iloc[0].isna().all())
        self.assertGreater(float(weights.loc[dates[-1], "good"]), 0.99)
        self.assertAlmostEqual(float(weights.loc[dates[-1], "bad"]), 0.0, places=10)


if __name__ == "__main__":
    unittest.main()
