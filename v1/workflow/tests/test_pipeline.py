"""End-to-end tests from factor panels to single- and multi-strategy weights."""

from __future__ import annotations

import unittest

import pandas as pd

from quant_workflow import (
    AlphaConfig,
    CompositeConfig,
    DailyProductionWorkflow,
    OptimizerConfig,
    PreprocessConfig,
    ResearchArtifact,
    RiskConfig,
    StrategyType,
    TransformConfig,
)

from examples.basic_workflow import make_demo_data


# 中文说明：`PipelineTest` 验证该场景的预期行为。
class PipelineTest(unittest.TestCase):
    # 中文说明：`test_end_to_end` 验证该场景的预期行为。
    def test_end_to_end(self) -> None:
        panel = make_demo_data(seed=11)
        artifact = ResearchArtifact(
            artifact_id="pipeline-single",
            created_at=pd.Timestamp("2026-06-14", tz="UTC"),
            selected_factors=tuple(panel.factors.columns),
            preprocess=PreprocessConfig(neutralize=True, min_observations=15),
            transform=TransformConfig(),
            composite=CompositeConfig(lookback=30, min_periods=10),
            alpha=AlphaConfig(lookback=60, min_periods=20, ridge=5.0),
            risk=RiskConfig(factor_halflife=30, specific_halflife=30),
            optimizer=OptimizerConfig(max_weight=0.08, max_turnover=None),
        )
        workflow = DailyProductionWorkflow(artifact)
        result = workflow.run(panel, panel.dates[-1])
        self.assertAlmostEqual(float(result.portfolio.weights.sum()), 1.0, places=6)
        self.assertFalse(result.risk_attribution.empty)
        self.assertGreater(result.portfolio.predicted_volatility, 0.0)

    # 中文说明：`test_one_model_snapshot_generates_three_strategy_books` 验证该场景的预期行为。
    def test_one_model_snapshot_generates_three_strategy_books(self) -> None:
        panel = make_demo_data(seed=17)
        artifact = ResearchArtifact(
            artifact_id="pipeline-multi",
            created_at=pd.Timestamp("2026-06-14", tz="UTC"),
            selected_factors=tuple(panel.factors.columns),
            preprocess=PreprocessConfig(neutralize=True, min_observations=15),
            transform=TransformConfig(),
            composite=CompositeConfig(lookback=30, min_periods=10),
            alpha=AlphaConfig(lookback=60, min_periods=20, ridge=5.0),
            risk=RiskConfig(factor_halflife=30, specific_halflife=30),
            optimizer=OptimizerConfig(),
        )
        workflow = DailyProductionWorkflow(artifact)
        assets = panel.assets
        benchmark = pd.Series(1.0 / len(assets), index=assets)
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

        result = workflow.run_strategies(
            panel,
            panel.dates[-1],
            strategy_configs=configs,
            benchmark_weights=benchmark,
        )

        self.assertEqual(set(result.portfolios), set(configs))
        self.assertAlmostEqual(
            float(result.portfolios[StrategyType.LONG_ONLY].weights.sum()),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(result.portfolios[StrategyType.INDEX_ENHANCED].weights.sum()),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(result.portfolios[StrategyType.MARKET_NEUTRAL].weights.sum()),
            0.0,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
