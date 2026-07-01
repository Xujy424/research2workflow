"""Tests optional transforms, alpha models, sleeves, regimes, and execution tools."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from quant_shared.alpha import (
    DynamicLinearAlpha,
    FamaMacBethAlpha,
    WalkForwardSklearnAlpha,
)
from researchflow.clustering import FactorClusterer, HierarchicalFactorComposite
from quant_shared.config import (
    AlphaConfig,
    RegimeConfig,
    SleeveConfig,
    TransformConfig,
)
from quant_workflow.execution import (
    FillSimulator,
    OrderGenerator,
    ParticipationScheduler,
    PositionPostProcessor,
)
from researchflow.regime import MixtureOfExperts, ObservableRegimeModel
from researchflow.sleeves import SleeveAllocator
from quant_shared.transforms import FactorTransformer
from researchflow.validation import WalkForwardSplitter, incremental_value


# 中文说明：`ExtendedMethodsTest` 验证该场景的预期行为。
class ExtendedMethodsTest(unittest.TestCase):
    # 中文说明：`setUp` 验证该场景的预期行为。
    def setUp(self) -> None:
        rng = np.random.default_rng(4)
        self.dates = pd.bdate_range("2024-01-02", periods=50)
        self.assets = pd.Index([f"S{i:02d}" for i in range(25)], name="asset")
        self.index = pd.MultiIndex.from_product(
            [self.dates, self.assets], names=["date", "asset"]
        )
        base = rng.normal(size=(len(self.index), 3))
        base[:, 2] = 0.8 * base[:, 0] + 0.2 * base[:, 2]
        self.factors = pd.DataFrame(base, index=self.index, columns=["a", "b", "c"])
        self.returns = pd.Series(
            base @ np.array([0.01, -0.006, 0.004])
            + rng.normal(scale=0.02, size=len(self.index)),
            index=self.index,
        )

    # 中文说明：`test_orthogonal_pca_pls_and_clustering` 验证该场景的预期行为。
    def test_orthogonal_pca_pls_and_clustering(self) -> None:
        orth = FactorTransformer(
            TransformConfig(method="orthogonal", orthogonalization="symmetric")
        ).transform(self.factors)
        latest_corr = orth.values.xs(self.dates[-1], level=0).corr()
        off_diagonal = latest_corr.to_numpy() - np.eye(3)
        self.assertLess(np.abs(off_diagonal).max(), 1e-8)

        pca = FactorTransformer(
            TransformConfig(method="pca", n_components=2, lookback=20, min_periods=10)
        ).transform(self.factors)
        self.assertEqual(pca.values.shape[1], 2)
        self.assertGreater(len(pca.values), 0)

        pls = FactorTransformer(
            TransformConfig(method="pls", n_components=2, lookback=20, min_periods=10)
        ).transform(self.factors, self.returns)
        self.assertEqual(pls.values.shape[1], 2)

        correlation = self.factors.groupby(level=0).corr().groupby(level=1).mean()
        quality = pd.Series({"a": 1.0, "b": 0.5, "c": 0.8})
        clustered = FactorClusterer().cluster(correlation, quality, threshold=0.35)
        self.assertIn("a", clustered.representatives)

        family_frame, composite = HierarchicalFactorComposite().combine(
            self.factors, {"a": "price", "b": "quality", "c": "price"}
        )
        self.assertEqual(set(family_frame.columns), {"price", "quality"})
        self.assertEqual(len(composite), len(self.factors))

    # 中文说明：`test_extended_alpha_models_are_walk_forward` 验证该场景的预期行为。
    def test_extended_alpha_models_are_walk_forward(self) -> None:
        config = AlphaConfig(
            method="fama_macbeth", lookback=15, min_periods=8, ridge=0.1
        )
        prediction, coefficients = FamaMacBethAlpha(config).fit_predict(
            self.factors, self.returns
        )
        self.assertGreater(len(prediction), 0)
        self.assertTrue(coefficients.iloc[:8].isna().all(axis=None))

        elastic, importance = WalkForwardSklearnAlpha(
            AlphaConfig(
                method="elastic_net",
                lookback=15,
                min_periods=8,
                ridge=0.001,
                max_iter=1000,
            )
        ).fit_predict(self.factors, self.returns)
        self.assertGreater(len(elastic), 0)
        self.assertEqual(importance.shape[1], 3)
        bayesian, _ = WalkForwardSklearnAlpha(
            AlphaConfig(
                method="bayesian_ridge",
                lookback=15,
                min_periods=8,
                max_iter=50,
            )
        ).fit_predict(self.factors, self.returns)
        self.assertGreater(len(bayesian), 0)
        dynamic, dynamic_beta = DynamicLinearAlpha(
            AlphaConfig(method="dynamic_linear", min_periods=8)
        ).fit_predict(self.factors, self.returns)
        self.assertGreater(len(dynamic), 0)
        self.assertEqual(dynamic_beta.shape[1], 3)

    # 中文说明：`test_sleeve_regime_validation_and_execution` 验证该场景的预期行为。
    def test_sleeve_regime_validation_and_execution(self) -> None:
        rng = np.random.default_rng(8)
        sleeve_returns = pd.DataFrame(
            rng.normal(0.0005, 0.01, size=(80, 3)),
            index=pd.bdate_range("2024-01-01", periods=80),
            columns=["value", "quality", "momentum"],
        )
        allocation = SleeveAllocator(
            SleeveConfig(
                method="risk_parity",
                lookback=30,
                min_periods=15,
                max_weight=0.60,
                weight_smoothing=0.5,
            )
        ).allocate(sleeve_returns)
        self.assertTrue(
            np.allclose(allocation.weights.sum(axis=1).to_numpy(), 1.0, atol=1e-6)
        )

        market = pd.Series(
            rng.normal(0.0002, 0.01, 100),
            index=pd.bdate_range("2024-01-01", periods=100),
        )
        regime = ObservableRegimeModel(
            RegimeConfig(lookback=20, transition_smoothing=0.5)
        ).fit_predict(market)
        self.assertTrue(
            np.allclose(regime.probabilities.sum(axis=1).tail(50), 1.0, atol=1e-6)
        )
        forecast_index = pd.MultiIndex.from_product(
            [market.index, ["A", "B"]], names=["date", "asset"]
        )
        experts = {
            name: pd.Series(float(i + 1), index=forecast_index)
            for i, name in enumerate(regime.probabilities.columns)
        }
        combined = MixtureOfExperts.combine(experts, regime.probabilities)
        self.assertEqual(len(combined), len(forecast_index))

        splits = WalkForwardSplitter().split(
            self.dates, train_size=20, test_size=5, purge=2
        )
        self.assertGreater(len(splits), 0)
        self.assertLess(splits[0][0].max(), splits[0][1].min())
        incremental = incremental_value(
            self.factors["a"], self.factors["c"], self.returns
        )
        self.assertEqual(len(incremental), len(self.dates))

        processor = PositionPostProcessor()
        target = pd.Series({"A": 0.6, "B": 0.4})
        current = pd.Series({"A": 0.5, "B": 0.5})
        adjusted = processor.apply_tradability(
            target,
            current,
            can_buy=pd.Series({"A": False, "B": True}),
            can_sell=pd.Series({"A": True, "B": True}),
        )
        self.assertAlmostEqual(float(adjusted["A"]), 0.5)
        shares, _ = processor.round_lots(
            adjusted, pd.Series({"A": 10.0, "B": 20.0}), 100_000, 100
        )
        book = OrderGenerator().generate(
            shares,
            pd.Series({"A": 0, "B": 0}),
            pd.Series({"A": 10.0, "B": 20.0}),
        )
        curve = pd.Series({"open": 0.4, "mid": 0.3, "close": 0.3})
        schedule = ParticipationScheduler().schedule(book.orders, curve)
        volume = pd.DataFrame(
            1_000_000.0, index=curve.index, columns=["A", "B"]
        )
        mids = pd.DataFrame(
            {"A": [10.0] * 3, "B": [20.0] * 3}, index=curve.index
        )
        fills = FillSimulator().simulate(schedule, volume, mids)
        self.assertGreater(float(fills["fill_quantity"].sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
