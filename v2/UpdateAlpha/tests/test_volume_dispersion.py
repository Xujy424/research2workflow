"""Numerical and time-alignment checks; no live data or database required."""

import itertools
import unittest
import tempfile
from pathlib import Path
from datetime import time
from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import polars as pl

from v2.UpdateAlpha import get_factor_spec
from v2.UpdateAlpha.operators import linear_quantile_residual as quantile_residual
from v2.UpdateAlpha.operators import time_series_corr
from v2.UpdateAlpha.pricevolume.highVol_onset import onset_counts
from v2.UpdateAlpha.pricevolume.bayesHighVol_onset import posterior_probability
from v2.UpdateAlpha.pricevolume.sr_dispersion_stdnumtrades import (
    sparse_gamma, l2_minute_trade_counts,
)
from v2.UpdateAlpha.pricevolume.vr_quantile_resid_corr import (
    VRQuantileResidCorrConfig as VolumeDispersionConfig,
    VRQuantileResidCorrFactor,
)


NAMES = (
    "vr_quantile_resid_corr", "vr_q80_ret_corr", "lr_score_mor",
    "lr_beta2_aft_minus_mor", "sr_dispersion_stdnumtrades",
    "sr_dispersion_mean_vwap", "sr_dispersion_volume_absret_corr",
    "highVol_onset", "lowVol_onset", "bayesHighVol_onset", "bayesLowVol_onset",
)


class VolumeDispersionTests(unittest.TestCase):
    def test_discovery(self):
        for name in NAMES:
            spec = get_factor_spec(name)
            self.assertEqual(spec.name, name.lower())
            self.assertEqual(spec.category, "pricevolume")
            self.assertEqual(spec.context_class.__module__, spec.factor_class.__module__)

    def test_quantile_intercept(self):
        y = np.arange(11, dtype=float)
        for q, expected in ((0.2, 2), (0.8, 8)):
            residual = quantile_residual(y, np.ones((len(y), 1)), q)
            np.testing.assert_allclose(y - residual, expected)

    def test_generic_correlation_shapes_and_missing_samples(self):
        x = np.array([1., 2., np.nan, 4.])
        y = np.array([2., 4., 8., 8.])
        self.assertAlmostEqual(time_series_corr(x, y, 3), 1)
        np.testing.assert_allclose(time_series_corr(x[:, None], y[:, None], 3), [1])
        self.assertTrue(np.isnan(time_series_corr(x, y, 4)))
        self.assertTrue(np.isnan(time_series_corr(np.ones(4), y, 3)))

    def test_sparse_global_optimum_against_support_enumeration(self):
        rng = np.random.default_rng(134)
        for repeat in range(8):
            market = rng.normal(0, 0.001, 9)
            market[0] = 0
            y = np.abs(rng.normal(0, 0.003, 9))
            for s in (0, 1, 3):
                gamma = sparse_gamma(y, market, s)
                self.assertLessEqual(np.count_nonzero(gamma), s)
                x = np.abs(market)
                remainder = y - market ** 2 * gamma
                beta = np.dot(x, remainder) / np.dot(x, x)
                actual = np.sum((remainder - beta * x) ** 2)
                best = np.inf
                for support in itertools.combinations(range(1, 9), s):
                    keep = np.ones(9, dtype=bool)
                    keep[list(support)] = False
                    b = np.dot(x[keep], y[keep]) / np.dot(x[keep], x[keep])
                    best = min(best, np.sum((y[keep] - b * x[keep]) ** 2))
                self.assertAlmostEqual(actual, best, places=14)

    def test_onset_excludes_current_dispersion_and_incomplete_future(self):
        gamma = np.array([0, 0, 1, 0, 0, np.nan, 0, 0], float)
        low, high = np.ones(8), np.ones(8)
        result = onset_counts(gamma, low, high, 2)
        self.assertEqual(result[0], (2, 2, 0, 0))
        # Last two positions are not base minutes, and missing minute 5 is
        # neither compressed away nor counted as a non-dispersion state.
        self.assertEqual(result[1], (0, 0, 2, 2))

    def test_l2_execution_counts_and_minute_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "proc" / "20261023"
            folder.mkdir(parents=True)
            pl.DataFrame({
                "SecurityID": [600000] * 12,
                "TransactTime": [time(9, 25), time(9, 30), time(9, 30, 59),
                                 time(9, 31), time(11, 29), time(11, 30),
                                 time(12), time(13), time(14, 59), time(15),
                                 time(10), time(10, 1)],
                "Price": [10.] * 10 + [0., 10.],
                "OrderQty": [100] * 11 + [0],
            }).write_parquet(folder / "shcj.pq")
            counts = l2_minute_trade_counts(tmp, "2026-10-23", ["600000", "000001"])
            self.assertEqual(counts[0, 0], 2)
            self.assertEqual(counts[1, 0], 1)
            self.assertEqual(counts[119, 0], 2)
            self.assertEqual(counts[120, 0], 1)
            self.assertEqual(counts[239, 0], 2)
            self.assertEqual(counts[:, 0].sum(), 8)
            self.assertEqual(counts[30, 0], 0)
            self.assertTrue(np.isnan(counts[:, 1]).all())
            with self.assertRaises(FileNotFoundError):
                l2_minute_trade_counts(tmp, "2026-10-22", ["600000"])

    def test_bayes_formula_and_universe(self):
        n = np.array([4., 40., 10., 10000.])
        h = np.array([0., 0., 5., 10000.])
        mask = np.array([True, True, True, False])
        p = posterior_probability(n, h, mask)
        total = n[:3].sum()
        mean = (h[:3].sum() + .5) / (total + 1)
        observed = np.sum(n[:3] * (h[:3] / n[:3] - mean) ** 2) / total
        noise = mean * (1 - mean) / (total / 3)
        k = np.clip(mean * (1 - mean) / max(observed - noise, 1e-8) - 1, 5, 200)
        np.testing.assert_allclose(p, (h + mean * k) / (n + k))
        self.assertGreater(p[0], p[1])
        self.assertTrue(np.all((p > 0) & (p < 1)))

    def test_lag_and_history(self):
        calls = []
        def daily(day):
            calls.append(day)
            return {"vr_quantile_resid_corr": np.array([day, np.nan])}
        context = SimpleNamespace(
            config=VolumeDispersionConfig(), daily_features=daily,
            data=SimpleNamespace(axis=SimpleNamespace(tick_count=2, date_position=lambda _: 15)),
        )
        value = VRQuantileResidCorrFactor(context).calculate("2026-10-26")
        self.assertEqual(calls, list(range(5, 15)))
        self.assertEqual(value[0], 9.5)
        self.assertTrue(np.isnan(value[1]))
        self.assertEqual(value.dtype, np.float32)
        calls.clear()
        context.data.axis.date_position = lambda _: 9
        self.assertTrue(np.isnan(VRQuantileResidCorrFactor(context).calculate("2026-10-26")).all())
        self.assertEqual(calls, [])

    def test_daily_families_and_missing_numtrades(self):
        rng = np.random.default_rng(156)
        returns = rng.normal(0, 0.001, (240, 3))
        close = np.vstack((np.full(3, 10.), 10 * np.cumprod(1 + returns, axis=0)))
        volume = rng.lognormal(5, 0.5, (241, 3))
        fields = {"m_essentials/close": close, "m_essentials/volume": volume,
                  "m_essentials/amount": close * volume,
                  "m_essentials/numtrades": rng.poisson(100, (241, 3)).astype(float),
                  "basic/tradable": np.ones(3)}
        def read(field, day):
            if field not in fields:
                raise FileNotFoundError(field)
            return fields[field]
        for name in NAMES:
            spec = get_factor_spec(name)
            context = spec.context_class.__new__(spec.context_class)
            import inspect
            config = inspect.signature(spec.context_class).parameters["config"].default
            kwargs = {}
            if "sparse_minutes" in config.__dataclass_fields__:
                kwargs["sparse_minutes"] = 12
            if "numtrades_field" in config.__dataclass_fields__:
                kwargs["numtrades_field"] = "m_essentials/numtrades"
            context.config = replace(config, **kwargs)
            context._daily_cache = {}
            context.data = SimpleNamespace(axis=SimpleNamespace(tick_count=3), read=read)
            result = context.daily_features(0)
            self.assertEqual(list(result), [name.lower()])
            for values in result.values():
                self.assertEqual(values.shape, (3,))
                self.assertTrue(np.isfinite(values).all())
            self.assertIs(result, context.daily_features(0))
            if "numtrades_field" in kwargs:
                saved = fields.pop("m_essentials/numtrades")
                with self.assertRaisesRegex(FileNotFoundError, "numtrades"):
                    context.daily_features(1)
                fields["m_essentials/numtrades"] = saved


if __name__ == "__main__":
    unittest.main()
