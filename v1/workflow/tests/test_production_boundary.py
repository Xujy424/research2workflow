"""Tests the explicit research-artifact to daily-production boundary."""

from __future__ import annotations

import unittest

from examples.basic_workflow import make_demo_data
from quant_workflow import (
    AlphaConfig,
    CompositeConfig,
    DailyProductionWorkflow,
    OptimizerConfig,
    PreprocessConfig,
    ResearchArtifact,
    RiskConfig,
    TransformConfig,
    StrategyType,
)
import pandas as pd


# 中文说明：`ProductionBoundaryTest` 验证该场景的预期行为。
class ProductionBoundaryTest(unittest.TestCase):
    # 中文说明：`test_daily_line_consumes_approved_factors_without_clustering` 验证该场景的预期行为。
    def test_daily_line_consumes_approved_factors_without_clustering(self) -> None:
        panel = make_demo_data(seed=23)
        artifact = ResearchArtifact(
            artifact_id="approved-v1",
            created_at=pd.Timestamp("2026-06-14", tz="UTC"),
            selected_factors=tuple(panel.factors.columns),
            preprocess=PreprocessConfig(neutralize=True, min_observations=15),
            transform=TransformConfig(method="none"),
            composite=CompositeConfig(method="icir", lookback=30, min_periods=10),
            alpha=AlphaConfig(lookback=60, min_periods=20, ridge=5.0),
            risk=RiskConfig(factor_halflife=30, specific_halflife=30),
            optimizer=OptimizerConfig(max_weight=0.08, max_turnover=None),
        )
        result = DailyProductionWorkflow(artifact).run(panel, panel.dates[-1])

        self.assertEqual(result.artifact_id, "approved-v1")
        self.assertFalse(result.diagnostics["research_steps_executed"])
        self.assertFalse(result.diagnostics["clustering_executed"])
        self.assertAlmostEqual(float(result.portfolio.weights.sum()), 1.0, places=6)

    # 中文说明：`test_missing_approved_factor_stops_production` 验证该场景的预期行为。
    def test_missing_approved_factor_stops_production(self) -> None:
        panel = make_demo_data(seed=29)
        artifact = ResearchArtifact(
            artifact_id="approved-v2",
            created_at=pd.Timestamp("2026-06-14", tz="UTC"),
            selected_factors=("missing_factor",),
            preprocess=PreprocessConfig(min_observations=15),
            transform=TransformConfig(),
            composite=CompositeConfig(),
            alpha=AlphaConfig(),
            risk=RiskConfig(),
            optimizer=OptimizerConfig(),
        )
        with self.assertRaisesRegex(ValueError, "missing_factor"):
            DailyProductionWorkflow(artifact).run(panel, panel.dates[-1])

    # 中文说明：`test_future_research_artifact_cannot_run_in_history` 验证时点一致性。
    def test_future_research_artifact_cannot_run_in_history(self) -> None:
        panel = make_demo_data(seed=30)
        artifact = ResearchArtifact(
            artifact_id="future-artifact",
            created_at=pd.Timestamp(panel.dates[-1], tz="UTC") + pd.Timedelta(days=1),
            selected_factors=tuple(panel.factors.columns),
            preprocess=PreprocessConfig(min_observations=15),
            transform=TransformConfig(),
            composite=CompositeConfig(),
            alpha=AlphaConfig(),
            risk=RiskConfig(),
            optimizer=OptimizerConfig(),
            effective_from=pd.Timestamp(panel.dates[-1], tz="UTC")
            + pd.Timedelta(days=1),
        )
        with self.assertRaisesRegex(ValueError, "not effective"):
            DailyProductionWorkflow(artifact).run(panel, panel.dates[-1])

    # 中文说明：`test_daily_line_reuses_one_snapshot_for_three_strategy_books` 验证该场景的预期行为。
    def test_daily_line_reuses_one_snapshot_for_three_strategy_books(self) -> None:
        panel = make_demo_data(seed=37)
        artifact = ResearchArtifact(
            artifact_id="approved-multi-v1",
            created_at=pd.Timestamp("2026-06-14", tz="UTC"),
            selected_factors=tuple(panel.factors.columns),
            preprocess=PreprocessConfig(neutralize=True, min_observations=15),
            transform=TransformConfig(),
            composite=CompositeConfig(lookback=30, min_periods=10),
            alpha=AlphaConfig(lookback=60, min_periods=20),
            risk=RiskConfig(factor_halflife=30, specific_halflife=30),
            optimizer=OptimizerConfig(),
        )
        configs = {
            StrategyType.LONG_ONLY: OptimizerConfig(
                strategy=StrategyType.LONG_ONLY,
                max_weight=0.08,
                max_turnover=None,
            ),
            StrategyType.INDEX_ENHANCED: OptimizerConfig(
                strategy=StrategyType.INDEX_ENHANCED,
                max_weight=0.08,
                max_active_weight=0.03,
                max_turnover=None,
            ),
            StrategyType.MARKET_NEUTRAL: OptimizerConfig(
                strategy=StrategyType.MARKET_NEUTRAL,
                max_weight=0.05,
                gross_exposure=1.0,
                net_exposure=0.0,
                max_turnover=None,
            ),
        }
        benchmark = pd.Series(1.0 / len(panel.assets), index=panel.assets)

        result = DailyProductionWorkflow(artifact).run_strategies(
            panel,
            panel.dates[-1],
            configs,
            benchmark_weights=benchmark,
        )

        self.assertEqual(set(result.portfolios), set(configs))
        self.assertAlmostEqual(
            float(result.portfolios[StrategyType.LONG_ONLY].weights.sum()),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(result.portfolios[StrategyType.INDEX_ENHANCED].active_weights.sum()),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(result.portfolios[StrategyType.MARKET_NEUTRAL].weights.sum()),
            0.0,
            places=6,
        )
        for portfolio in result.portfolios.values():
            self.assertIn("linear_cost", portfolio.diagnostics)
            self.assertIn("impact_cost", portfolio.diagnostics)
            self.assertGreater(portfolio.predicted_volatility, 0.0)


if __name__ == "__main__":
    unittest.main()
