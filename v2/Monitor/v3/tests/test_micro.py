"""Micro指标接口、共用盘口读取和日期轴bin写入；不依赖真实数据。"""

from collections import OrderedDict
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd

from v2.Monitor.v3 import micro


class FakeContext:
    def __init__(self, root):
        self.root = Path(root)
        self.dates = pd.bdate_range("2026-01-01", periods=30)
        self.ticks = np.array(["600000", "000001", "000002"])
        self.data = SimpleNamespace(axis=SimpleNamespace(
            full_dates=np.r_[self.dates.values.astype("datetime64[D]"),
                             np.full(2, np.datetime64("NaT", "D"))]))
        self._feature_cache = OrderedDict()
        self.snapshot_calls, self.closed = 0, False
        self.weights = np.tile([.2, .3, .5], (len(self.dates), 1))
        self.fields = {
            "d_essentials/pct": np.tile([1., -2., 3.], (len(self.dates), 1)),
            "d_essentials/amount": (1000 + np.arange(len(self.dates))[:, None] ** 2)
                                  * np.array([[1., 2., 3.]]),
        }
        self.quotes = []
        rng = np.random.default_rng(72)
        for day in range(len(self.dates)):
            ret = rng.normal(0, .0001, (72, 3))
            ret[35] += .01
            ret[51] += .025
            ret[65] -= .06
            mid = 10 * np.exp(np.cumsum(ret, axis=0))
            bid_px = np.stack([mid - .01 * level for level in range(1, 6)])
            ask_px = np.stack([mid + .01 * level for level in range(1, 6)])
            bid_qty = np.full((5, 72, 3), 1000. + day)
            ask_qty = bid_qty * 1.2
            ofi = rng.normal(0, 30, (72, 3))
            ofi[0] = np.nan
            self.quotes.append({
                "mid": mid, "spread": (ask_px[0] - bid_px[0]) / mid * 10000,
                "bid_px": bid_px, "ask_px": ask_px,
                "bid_qty": bid_qty, "ask_qty": ask_qty, "ofi": ofi,
                "best_imbalance": (bid_qty[0] - ask_qty[0]) / (bid_qty[0] + ask_qty[0]),
            })
        self.quotes[-1] = None

    def get_position(self, date):
        return int(self.dates.searchsorted(pd.Timestamp(date), side="right") - 1)

    def get_weight(self, domain, name, end):
        return self.weights[:end + 1]

    def read(self, source, key, start, end, *, kind):
        if key.startswith("m_essentials/"):
            raise AssertionError("日频成交额指标不应展开全历史分钟矩阵")
        return self.fields[key][start:end + 1]

    def aligned_snapshot(self, date):
        self.snapshot_calls += 1
        return self.quotes[self.get_position(date)]

    def l2_path(self, date, exchange, suffix):
        return self.root / "missing_l2" / f"{exchange}{suffix}.pq"

    def close(self):
        self.closed = True


class MicroTests(unittest.TestCase):
    def test_rolling_prior_standardization_matches_previous_formula(self):
        from v2.Monitor.v3.common import rolling_zscore

        values = np.random.default_rng(17).normal(size=(60, 5))
        values[::3, 1] = np.nan
        values[:, 2] = 0
        for window, minimum in ((20, 10), (5, 3)):
            expected = np.full_like(values, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                for i in range(window, len(values)):
                    history = values[i - window:i]
                    std = np.nanstd(history, axis=0)
                    expected[i] = np.divide(values[i] - np.nanmean(history, axis=0), std,
                        out=np.full(values.shape[1], np.nan),
                        where=(np.isfinite(history).sum(axis=0) >= minimum) & (std > 0))
            actual = rolling_zscore(values, window, minimum,
                                    ddof=0, exclude_current=True).to_numpy()
            actual[:window] = np.nan
            np.testing.assert_allclose(actual, expected, atol=1e-12, equal_nan=True)

    def test_rolling_zscore_default_preserves_series_behavior(self):
        from v2.Monitor.v3.common import rolling_zscore

        series = pd.Series(np.arange(10.), name='test')
        history = series.rolling(4, min_periods=3)
        expected = (series - history.mean()) / history.std(ddof=1)
        pd.testing.assert_series_equal(rolling_zscore(series, 4, 3), expected)

    def test_batched_acd_matches_original_stock_recursion(self):
        with tempfile.TemporaryDirectory() as root:
            context = FakeContext(root)
            context.ticks = np.arange(100)
            rng = np.random.default_rng(19)
            jump = rng.random((240, 100)) < .04
            jump[:, 0] = False
            jump[:, 1] = False
            jump[20, 1] = True
            context.quotes[0] = {"mid": np.ones((241, 100))}
            expected = np.full(100, np.nan)
            expected[jump.sum(axis=0) == 0] = 0
            for stock in np.flatnonzero(jump.sum(axis=0) >= 2):
                waits = np.diff(np.flatnonzero(jump[:, stock])).astype(float)
                omega = (1 - .1 - .8) * waits.mean()
                psi = max(waits.mean(), 1.)
                for wait in waits:
                    psi = omega + .1 * wait + .8 * psi
                expected[stock] = 1 / psi
            with patch.object(micro, "significant_jump_events",
                              return_value=(jump, np.ones(100, dtype=bool))):
                actual = micro.acd_jump_intensity(
                    context, context.dates[0], "index", "hs300", stock_level=True)
            np.testing.assert_allclose(actual.iloc[0], expected, rtol=1e-13, equal_nan=True)

    def test_acd_uses_significant_events_and_trading_minute_durations(self):
        from v2.Monitor.v3.common import significant_jump_events

        with tempfile.TemporaryDirectory() as root:
            context = FakeContext(root)
            ret = np.tile([.001, -.001], 120)
            ret[[50, 180]] = [.08, -.1]
            returns = np.column_stack([ret, np.tile([.001, -.001], 120), ret])
            returns[20, 2] = np.nan
            events, valid = significant_jump_events(returns)
            np.testing.assert_array_equal(np.flatnonzero(events[:, 0]), [50, 180])
            np.testing.assert_array_equal(valid, [True, True, False])
            self.assertFalse(events[:, 1:].any())
            mid = 10 * np.exp(np.vstack([np.zeros(3), returns.cumsum(axis=0)]))
            context.quotes[0] = {"mid": mid}
            raw = micro.acd_jump_intensity(
                context, context.dates[0], "index", "hs300", stock_level=True).iloc[0]
            self.assertAlmostEqual(raw.iloc[0], 1 / 130)
            self.assertEqual(raw.iloc[1], 0)
            self.assertTrue(np.isnan(raw.iloc[2]))
            # 仅一次显著事件不能估计ACD间隔。
            returns[180, 0] = .001
            context.quotes[0]["mid"] = 10 * np.exp(
                np.vstack([np.zeros(3), returns.cumsum(axis=0)]))
            single = micro.acd_jump_intensity(
                context, context.dates[0], "index", "hs300", stock_level=True).iloc[0]
            self.assertTrue(np.isnan(single.iloc[0]))

    def test_bns_ratio_matches_scalar_formula_and_handles_invalid_stocks(self):
        from math import gamma

        with tempfile.TemporaryDirectory() as root:
            context = FakeContext(root)
            # 相同幅度的连续收益不应被识别成跳跃；单次大跳跃应通过检验。
            r = np.tile([.001, -.001], 120)
            jumped = r.copy()
            jumped[100] = .08
            returns = np.column_stack([r, jumped, np.zeros_like(r)])
            mid = 10 * np.exp(np.vstack([np.zeros(3), returns.cumsum(axis=0)]))
            context.quotes[0] = {"mid": mid}
            result = micro.significant_jump_variance_ratio(
                context, context.dates[0], "index", "hs300", stock_level=True).iloc[0]
            n = len(jumped)
            rv = sum(jumped ** 2)
            bv = np.pi / 2 * n / (n - 1) * sum(abs(jumped[1:] * jumped[:-1]))
            mu = 2 ** (2 / 3) * gamma(7 / 6) / np.sqrt(np.pi)
            tq = n * n / (n - 2) / mu ** 3 * sum(
                abs(jumped[2:] * jumped[1:-1] * jumped[:-2]) ** (4 / 3))
            z = (1 - bv / rv) / np.sqrt(
                ((np.pi / 2) ** 2 + np.pi - 5) / n * max(1, tq / bv ** 2))
            self.assertGreater(z, 2.3263478740408408)
            self.assertEqual(result.iloc[0], 0)
            self.assertAlmostEqual(result.iloc[1], 1 - bv / rv)
            self.assertTrue(np.isnan(result.iloc[2]))
            # 缺失不能当作零收益；价格缩放不能改变无量纲结果。
            context.quotes[0]["mid"] = mid * 100
            scaled = micro.significant_jump_variance_ratio(
                context, context.dates[0], "index", "hs300", stock_level=True).iloc[0]
            np.testing.assert_allclose(result, scaled, atol=1e-12, equal_nan=True)
            context.quotes[0]["mid"][50, 1] = np.nan
            missing = micro.significant_jump_variance_ratio(
                context, context.dates[0], "index", "hs300", stock_level=True).iloc[0]
            self.assertTrue(np.isnan(missing.iloc[1]))

    def test_bns_excludes_lunch_price_change(self):
        with tempfile.TemporaryDirectory() as root:
            context = FakeContext(root)
            times = pd.date_range("2026-01-01 09:30", periods=61, freq="min").append(
                pd.date_range("2026-01-01 13:00", periods=61, freq="min"))
            ret = np.tile([.001, -.001], 61)[:121]
            ret[60] = .2
            mid = 10 * np.exp(np.r_[0, ret.cumsum()])
            context.quotes[0] = {
                "times": times.strftime("%H:%M:%S").to_numpy(),
                "mid": np.tile(mid[:, None], (1, 3)),
            }
            result = micro.significant_jump_variance_ratio(
                context, context.dates[0], "index", "hs300")
            self.assertEqual(result.iloc[0], 0)

    def test_intraday_means_include_brief_stress_and_ignore_missing(self):
        with tempfile.TemporaryDirectory() as root:
            context = FakeContext(root)
            context.quotes[0] = {
                "spread": np.array([[1., 2., np.nan], [1., np.nan, np.nan],
                                    [10., 8., np.nan]]),
                "bid_px": np.ones((5, 3, 3)),
                "ask_px": np.ones((5, 3, 3)),
                "bid_qty": np.broadcast_to(
                    np.array([[10., 20., np.nan], [10., np.nan, np.nan],
                              [1., 2., np.nan]]), (5, 3, 3)).copy(),
                "ask_qty": np.zeros((5, 3, 3)),
            }
            context.quotes[0]["ask_qty"][:, :, 2] = np.nan
            context.quotes[0]["ask_qty"][:, 1, 1] = np.nan
            spread = micro.effective_spread(
                context, context.dates[0], "index", "hs300", stock_level=True)
            depth = micro.five_level_depth_notional(
                context, context.dates[0], "index", "hs300", stock_level=True,
                normalize=False)
            np.testing.assert_allclose(spread.iloc[0], [4., 5., np.nan], equal_nan=True)
            np.testing.assert_allclose(depth.iloc[0], [35., 55., np.nan], equal_nan=True)
            aggregated = micro.effective_spread(
                context, context.dates[0], "index", "hs300")
            self.assertAlmostEqual(aggregated.iloc[0], 4.6)

    def test_depth_normalization_uses_only_previous_twenty_days(self):
        with tempfile.TemporaryDirectory() as root:
            context = FakeContext(root)
            date = context.dates[20]
            raw = micro.five_level_depth_notional(
                context, date, "index", "hs300", stock_level=True, normalize=False)
            normalized = micro.five_level_depth_notional(
                context, date, "index", "hs300", stock_level=True)
            self.assertTrue(normalized.iloc[:20].isna().all().all())
            expected = raw.iloc[20].to_numpy() / context.fields[
                "d_essentials/amount"][:20].mean(axis=0)
            np.testing.assert_allclose(normalized.iloc[20], expected)
            context.fields["d_essentials/amount"][20] *= 1000
            unchanged = micro.five_level_depth_notional(
                context, date, "index", "hs300", stock_level=True)
            np.testing.assert_allclose(unchanged.iloc[20], expected)
            aggregate = micro.five_level_depth_notional(context, date, "index", "hs300")
            self.assertAlmostEqual(aggregate.iloc[20], expected @ context.weights[20])
            capacity = micro.executable_capacity(context, date, "index", "hs300")
            expected_capacity = np.minimum(raw.iloc[20], .1 * context.fields[
                "d_essentials/amount"][20]) @ context.weights[20]
            self.assertAlmostEqual(capacity.iloc[20], expected_capacity)
            context.fields["d_essentials/amount"][:20, 0] = 0
            context.fields["d_essentials/amount"][0, 1] = np.nan
            invalid = micro.five_level_depth_notional(
                context, date, "index", "hs300", stock_level=True)
            self.assertTrue(invalid.iloc[20, :2].isna().all())

    def test_all_indicators_write_matching_float32_series(self):
        with tempfile.TemporaryDirectory() as root, warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            reference = FakeContext(root)
            expected = {
                name: calculate(reference, reference.dates[-1], "index", "hs300")
                for name, calculate in micro.MICRO_INDICATORS.items()
            }
            self.assertEqual(len(expected), 17)
            for values in expected.values():
                self.assertIsInstance(values, pd.Series)
                pd.testing.assert_index_equal(values.index, reference.dates)
            context = FakeContext(root)
            with patch.object(micro, "MonitorContext", return_value=context), redirect_stdout(StringIO()):
                outputs = micro.update_micro(root=root)
            self.assertGreaterEqual(context.snapshot_calls, len(context.dates))
            self.assertTrue(context.closed)
            self.assertFalse(hasattr(context, "_micro_matrices"))
            self.assertEqual(set(outputs), set(expected))
            for name, path in outputs.items():
                self.assertEqual(path.parent, Path(root) / "median_variables/monitor/micro")
                stored = np.fromfile(path, dtype=np.float32)
                self.assertEqual(stored.shape, (32,))
                np.testing.assert_allclose(stored[:30], expected[name].to_numpy(),
                                           rtol=1e-6, atol=1e-7, equal_nan=True)
                self.assertTrue(np.isnan(stored[30:]).all())

    def test_daily_only_selection_does_not_read_snapshots(self):
        with tempfile.TemporaryDirectory() as root:
            context = FakeContext(root)
            with patch.object(micro, "MonitorContext", return_value=context), redirect_stdout(StringIO()):
                outputs = micro.update_micro(root=root, indicators="amihud_illiquidity")
            self.assertEqual(context.snapshot_calls, 0)
            self.assertEqual(set(outputs), {"amihud_illiquidity"})
            expected = np.sum(np.abs(context.fields["d_essentials/pct"] / 100)
                              / context.fields["d_essentials/amount"] * context.weights, axis=1)
            np.testing.assert_allclose(np.fromfile(outputs["amihud_illiquidity"], np.float32)[:30],
                                       expected, rtol=1e-6)

    def test_invalid_selection_fails_before_opening_context(self):
        with patch.object(micro, "MonitorContext") as factory:
            with self.assertRaisesRegex(ValueError, "未知micro指标"):
                micro.update_micro(indicators=("not_an_indicator",))
            self.assertEqual(micro.update_micro(indicators=[]), {})
            factory.assert_not_called()

    def test_calculation_failure_closes_context(self):
        with tempfile.TemporaryDirectory() as root:
            context = FakeContext(root)
            def fail(*args, **kwargs):
                raise ValueError("calculation failed")

            with patch.object(micro, "MonitorContext", return_value=context), \
                 patch.dict(micro.MICRO_INDICATORS, {"five_level_depth_notional": fail}), \
                 redirect_stdout(StringIO()):
                with self.assertRaisesRegex(ValueError, "calculation failed"):
                    micro.update_micro(root=root, indicators=["five_level_depth_notional"])
            self.assertTrue(context.closed)
            self.assertFalse(hasattr(context, "_micro_matrices"))

    def test_no_private_dispatch_or_feature_cache(self):
        import inspect

        source = inspect.getsource(micro)
        for removed in ("_feature_matrix", "_daily_matrix", "_domain_feature",
                        "_micro_matrices", "_feature_cache", "_prepare_snapshots"):
            self.assertNotIn(removed, source)
        with tempfile.TemporaryDirectory() as root:
            context = FakeContext(root)
            matrix = micro.effective_spread(context, context.dates[-1], "index", "hs300",
                                            stock_level=True)
            self.assertIsInstance(matrix, pd.DataFrame)
            self.assertEqual(matrix.shape, (30, 3))
            self.assertTrue(matrix.iloc[-1].isna().all())


if __name__ == "__main__":
    unittest.main()
