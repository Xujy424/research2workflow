"""Tests capacity/crowding monitors and compatibility research adapters."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from examples.basic_workflow import make_demo_data
from researchflow.adapters import LegacyAnalyzerAdapter, ResearchScorecard
from quant_workflow.monitoring import CapacityAnalyzer, CrowdingMonitor
from researchflow.research import FactorResearchEngine


# 中文说明：`MonitoringAndAdapterTest` 验证该场景的预期行为。
class MonitoringAndAdapterTest(unittest.TestCase):
    # 中文说明：`test_capacity_crowding_scorecard_and_legacy_tables` 验证该场景的预期行为。
    def test_capacity_crowding_scorecard_and_legacy_tables(self) -> None:
        panel = make_demo_data(seed=9)
        report = FactorResearchEngine().analyze(panel)
        scorecard = ResearchScorecard().score(report)
        self.assertEqual(set(scorecard["factor"]), set(panel.factors.columns))

        adapter = LegacyAnalyzerAdapter(
            report, panel.factors, panel.forward_returns, "value"
        )
        self.assertIn("Overall", adapter.table_IC_annual_stats().index)
        self.assertFalse(adapter.table_group_ret_stats().empty)
        self.assertFalse(adapter.table_alpha_annual_stats().empty)

        assets = panel.assets
        weights = pd.Series(1.0 / len(assets), index=assets)
        adv = pd.Series(5_000_000.0, index=assets)
        alpha = pd.Series(0.01, index=assets)
        capacity = CapacityAnalyzer().analyze(
            weights, adv, alpha, [1_000_000, 10_000_000, 100_000_000]
        )
        self.assertEqual(len(capacity.capital_grid), 3)
        crowding = CrowdingMonitor().score(
            weights,
            ownership=pd.Series(np.linspace(0, 1, len(assets)), index=assets),
        )
        self.assertTrue(crowding["crowding_score"].is_monotonic_decreasing)


if __name__ == "__main__":
    unittest.main()
