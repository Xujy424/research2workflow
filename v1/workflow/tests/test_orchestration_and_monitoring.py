"""Tests corrected DAG arrows and the production monitoring feedback gate."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from quant_workflow import (
    DailyProductionTracker,
    DailyProductionGraph,
    ProductionMonitoringLoop,
    TradingRunState,
)


# 中文说明：`OrchestrationAndMonitoringTest` 验证该场景的预期行为。
class OrchestrationAndMonitoringTest(unittest.TestCase):
    # 中文说明：`test_corrected_combination_dependencies_are_acyclic` 验证该场景的预期行为。
    def test_corrected_combination_dependencies_are_acyclic(self) -> None:
        graph = DailyProductionGraph()
        upstream = graph.upstream("combination_ready")
        self.assertIn("alpha_ready", upstream)
        self.assertIn("risk_bank_update", upstream)
        self.assertNotIn("combination_ready", graph.upstream("alpha_ready"))
        order = graph.topological_order()
        self.assertLess(order.index("alpha_ready"), order.index("comb_bank"))
        self.assertLess(
            order.index("factor_combination_update"),
            order.index("factor_comb_bank"),
        )

    # 中文说明：`test_reconciliation_failure_blocks_trading` 验证该场景的预期行为。
    def test_reconciliation_failure_blocks_trading(self) -> None:
        decision = ProductionMonitoringLoop().decide(
            reconciliation_passed=False,
            drift_warning=True,
        )
        self.assertEqual(decision.state, TradingRunState.BLOCKED)

    # 中文说明：`test_monitor_warnings_reduce_without_mutating_research` 验证该场景的预期行为。
    def test_monitor_warnings_reduce_without_mutating_research(self) -> None:
        decision = ProductionMonitoringLoop().decide(
            reconciliation_passed=True,
            drift_warning=True,
            risk_calibration_multiplier=1.4,
            recommended_capital=50.0,
            current_capital=100.0,
            maximum_crowding_score=0.9,
        )
        self.assertEqual(decision.state, TradingRunState.REDUCED)
        self.assertEqual(decision.factor_action, "request_downweight_or_pause")
        self.assertEqual(decision.risk_action, "request_risk_recalibration")
        self.assertEqual(decision.optimizer_action, "tighten_position_limits")

    # 中文说明：`test_daily_tracker_builds_a_complete_monitoring_snapshot` 验证监控闭环汇总。
    def test_daily_tracker_builds_a_complete_monitoring_snapshot(self) -> None:
        dates = pd.bdate_range("2025-01-01", periods=80)
        assets = pd.Index(["A", "B", "C"])
        snapshot = DailyProductionTracker().evaluate(
            reconciliation_passed=True,
            live_ic=pd.Series(-0.08, index=dates),
            reference_ic_mean=0.03,
            reference_ic_std=0.02,
            predicted_volatility=pd.Series(0.01, index=dates),
            realized_returns=pd.Series(
                np.sin(np.arange(len(dates))) * 0.02,
                index=dates,
            ),
            target_weights=pd.Series([0.4, 0.35, 0.25], index=assets),
            adv_currency=pd.Series([1e9, 8e8, 6e8], index=assets),
            expected_alpha=pd.Series([0.03, 0.02, 0.01], index=assets),
            capitals=[1e7, 5e7, 1e8],
            current_capital=1e8,
        )
        self.assertFalse(snapshot.drift.empty)
        self.assertFalse(snapshot.risk_calibration.empty)
        self.assertFalse(snapshot.capacity.capital_grid.empty)
        self.assertFalse(snapshot.crowding.empty)
        self.assertIn(
            snapshot.decision.state,
            {TradingRunState.ALLOWED, TradingRunState.REDUCED},
        )


if __name__ == "__main__":
    unittest.main()
