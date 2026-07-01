"""Tests research ordering and artifact publication."""

from __future__ import annotations

import unittest

from examples.basic_workflow import make_demo_data
from quant_workflow import (
    AlphaConfig,
    CompositeConfig,
    OptimizerConfig,
    PreprocessConfig,
    RiskConfig,
    TransformConfig,
)
from researchflow import FactorResearchWorkflow, ResearchFlowConfig


# 中文说明：`ResearchFlowTest` 验证该场景的预期行为。
class ResearchFlowTest(unittest.TestCase):
    # 中文说明：`test_clustering_precedes_optional_orthogonalization` 验证该场景的预期行为。
    def test_clustering_precedes_optional_orthogonalization(self) -> None:
        panel = make_demo_data(seed=31)
        config = ResearchFlowConfig(
            preprocess=PreprocessConfig(neutralize=True, min_observations=15),
            transform=TransformConfig(
                method="orthogonal", orthogonalization="symmetric"
            ),
            composite=CompositeConfig(lookback=30, min_periods=10),
            alpha=AlphaConfig(lookback=60, min_periods=20),
            risk=RiskConfig(factor_halflife=30, specific_halflife=30),
            optimizer=OptimizerConfig(max_weight=0.08, max_turnover=None),
            run_robustness=True,
        )
        result = FactorResearchWorkflow(config).run(
            panel, created_at="2026-06-14T00:00:00Z"
        )

        self.assertLess(
            result.stage_order.index("correlation_clustering"),
            result.stage_order.index("optional_model_transform"),
        )
        self.assertEqual(
            list(result.single_factor_report.factor_correlations.columns),
            list(result.processed_factors.columns),
        )
        self.assertEqual(
            result.artifact.diagnostics["clustering_input"],
            "uniformly_preprocessed_factor_correlations",
        )
        self.assertEqual(
            set(result.model_transform.values.columns),
            set(result.selected_factors),
        )
        self.assertEqual(
            set(result.incremental_ic.columns),
            set(result.selected_factors),
        )
        self.assertGreater(result.incremental_ic.notna().sum().sum(), 0)
        self.assertIsNotNone(result.robustness)


if __name__ == "__main__":
    unittest.main()
