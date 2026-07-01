"""Tests positive-semidefinite risk estimates and portfolio constraints."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from quant_shared.config import OptimizerConfig, RiskConfig, StrategyType
from quant_workflow.portfolio import PortfolioOptimizer
from quant_workflow.risk import EquityFactorRiskModel


# 中文说明：`RiskAndPortfolioTest` 验证该场景的预期行为。
class RiskAndPortfolioTest(unittest.TestCase):
    # 中文说明：`setUp` 验证该场景的预期行为。
    def setUp(self) -> None:
        rng = np.random.default_rng(1)
        self.dates = pd.bdate_range("2024-01-01", periods=80)
        self.assets = pd.Index([f"S{i:02d}" for i in range(30)], name="asset")
        factor_names = ["market", "size", "value"]
        static_x = rng.normal(size=(len(self.assets), len(factor_names)))
        index = pd.MultiIndex.from_product(
            [self.dates, self.assets], names=["date", "asset"]
        )
        self.exposures = pd.DataFrame(
            np.tile(static_x, (len(self.dates), 1)),
            index=index,
            columns=factor_names,
        )
        factor_returns = rng.normal(scale=0.01, size=(len(self.dates), len(factor_names)))
        returns = factor_returns @ static_x.T + rng.normal(
            scale=0.02, size=(len(self.dates), len(self.assets))
        )
        self.returns = pd.DataFrame(returns, index=self.dates, columns=self.assets)

    # 中文说明：`test_covariance_is_psd_and_optimizer_respects_budget` 验证该场景的预期行为。
    def test_covariance_is_psd_and_optimizer_respects_budget(self) -> None:
        risk = EquityFactorRiskModel(RiskConfig()).fit(
            self.returns.iloc[:-1],
            self.exposures.loc[self.dates[:-1]],
            self.exposures.xs(self.dates[-1], level=0),
        )
        minimum_eigenvalue = np.linalg.eigvalsh(risk.stock_covariance).min()
        self.assertGreaterEqual(minimum_eigenvalue, -1e-10)
        alpha = pd.Series(np.linspace(-0.01, 0.02, len(self.assets)), index=self.assets)
        result = PortfolioOptimizer(
            OptimizerConfig(max_weight=0.10, max_turnover=None, risk_aversion=2.0)
        ).optimize(alpha, risk)
        self.assertAlmostEqual(float(result.weights.sum()), 1.0, places=6)
        self.assertGreaterEqual(float(result.weights.min()), -1e-8)
        self.assertLessEqual(float(result.weights.max()), 0.100001)

    # 中文说明：`test_index_enhancement_uses_negative_alpha_as_underweight` 验证该场景的预期行为。
    def test_index_enhancement_uses_negative_alpha_as_underweight(self) -> None:
        risk = EquityFactorRiskModel(RiskConfig()).fit(
            self.returns.iloc[:-1],
            self.exposures.loc[self.dates[:-1]],
            self.exposures.xs(self.dates[-1], level=0),
        )
        constituents = self.assets[:20]
        benchmark = pd.Series(1.0 / len(constituents), index=constituents)
        alpha = pd.Series(-0.02, index=self.assets)
        alpha.loc[constituents[:10]] = -0.03
        alpha.loc[constituents[10:]] = 0.03
        alpha.loc[self.assets[20:]] = 0.50  # Must not enter a constituent-only book.

        result = PortfolioOptimizer(
            OptimizerConfig(
                strategy=StrategyType.INDEX_ENHANCED,
                risk_aversion=0.01,
                linear_cost_penalty=0.0,
                impact_cost_penalty=0.0,
                max_weight=0.10,
                max_active_weight=0.05,
                max_turnover=None,
                tracking_error_limit=None,
                benchmark_constituents_only=True,
            )
        ).optimize(alpha, risk, benchmark_weights=benchmark)

        self.assertEqual(result.weights.index.tolist(), constituents.tolist())
        self.assertGreaterEqual(float(result.weights.min()), -1e-8)
        self.assertAlmostEqual(float(result.weights.sum()), 1.0, places=7)
        self.assertAlmostEqual(float(result.active_weights.sum()), 0.0, places=7)
        self.assertTrue(
            np.allclose(
                result.weights.to_numpy(),
                (
                    result.benchmark_weights + result.active_weights
                ).to_numpy(),
            )
        )
        self.assertLess(
            float(result.active_weights.loc[constituents[:10]].mean()),
            0.0,
        )
        self.assertGreater(
            float(result.active_weights.loc[constituents[10:]].mean()),
            0.0,
        )
        self.assertTrue(
            result.active_weights.loc[constituents[:10]].lt(0.0).all()
        )
        self.assertTrue(
            result.active_weights.loc[constituents[10:]].gt(0.0).all()
        )

    # 中文说明：`test_index_enhancement_requires_a_complete_benchmark` 验证该场景的预期行为。
    def test_index_enhancement_requires_a_complete_benchmark(self) -> None:
        risk = EquityFactorRiskModel(RiskConfig()).fit(
            self.returns.iloc[:-1],
            self.exposures.loc[self.dates[:-1]],
            self.exposures.xs(self.dates[-1], level=0),
        )
        alpha = pd.Series(0.01, index=self.assets)
        optimizer = PortfolioOptimizer(
            OptimizerConfig(strategy=StrategyType.INDEX_ENHANCED)
        )
        with self.assertRaisesRegex(ValueError, "benchmark_weights are required"):
            optimizer.optimize(alpha, risk)
        with self.assertRaisesRegex(ValueError, "must sum to 1"):
            optimizer.optimize(
                alpha,
                risk,
                benchmark_weights=pd.Series(0.01, index=self.assets),
            )


if __name__ == "__main__":
    unittest.main()
