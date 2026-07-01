"""Tests cross-sectional clipping, standardisation, and preprocessing contracts."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from quant_shared.config import PreprocessConfig
from quant_shared.preprocessing import CrossSectionalPreprocessor


# 中文说明：`PreprocessingTest` 验证该场景的预期行为。
class PreprocessingTest(unittest.TestCase):
    # 中文说明：`test_cross_section_is_standardized_and_outlier_is_clipped` 验证该场景的预期行为。
    def test_cross_section_is_standardized_and_outlier_is_clipped(self) -> None:
        index = pd.MultiIndex.from_product(
            [[pd.Timestamp("2025-01-02")], list("ABCDE")],
            names=["date", "asset"],
        )
        factors = pd.DataFrame({"factor": [1.0, 2.0, 3.0, 4.0, 1000.0]}, index=index)
        result = CrossSectionalPreprocessor(
            PreprocessConfig(neutralize=False, winsor_limit=3.0)
        ).transform(factors)
        self.assertAlmostEqual(float(result["factor"].mean()), 0.0, places=10)
        self.assertAlmostEqual(float(result["factor"].std(ddof=0)), 1.0, places=10)
        self.assertLess(float(result["factor"].max()), 2.0)


if __name__ == "__main__":
    unittest.main()
