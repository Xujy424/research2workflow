"""Only the eleven Xingye volume/dispersion factors and their backtest path."""

import inspect
import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import replace
from datetime import time

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import polars as pl

from v2.UpdateAlpha import get_factor_spec
from v2.UpdateAlpha import backtest
from v2.UpdateAlpha.tests.test_volume_dispersion import NAMES


class NewFactorWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        (cls.root / "axis").mkdir()
        cls.dates = pd.bdate_range("2024-01-01", periods=14).values.astype("datetime64[D]")
        cls.ticks = np.array([f"{600000+i:06d}" for i in range(16)])
        np.save(cls.root / "axis/dates.npy", np.r_[cls.dates, np.datetime64("NaT", "D")])
        np.save(cls.root / "axis/stock_ticks.npy", np.concatenate((cls.ticks, [""])))
        rng = np.random.default_rng(415)
        d, n = len(cls.dates), len(cls.ticks)
        close = 10 * np.cumprod(1 + rng.normal(0, .02, (d, n)), axis=0)
        values = {
            "basic/tradable": np.ones((d, n)),
            "industry/industry": np.tile(np.arange(n) % 2, (d, 1)),
            "d_essentials/pct": rng.normal(0, 2, (d, n)),
            "d_essentials/total_mv": rng.lognormal(12, 1, (d, n)),
        }
        minute_close = close[:, None, :] * np.cumprod(1 + rng.normal(0, .001, (d, 241, n)), axis=1)
        values["m_essentials/close"] = minute_close
        values["m_essentials/volume"] = rng.lognormal(6, 1, (d, 241, n))
        values["m_essentials/amount"] = minute_close * values["m_essentials/volume"]
        for field, array in values.items():
            path = cls.root / "stock" / f"{field}.bin"
            path.parent.mkdir(parents=True, exist_ok=True)
            full = np.full((d + 1, *array.shape[1:-1], n + 1), np.nan, dtype=np.float32)
            full[:d, ..., :n] = array
            full.tofile(path)
        # Factor date 10 uses date 9's L2 executions because lag_days=1.
        folder = cls.root / "stock/l2/proc" / pd.Timestamp(cls.dates[9]).strftime("%Y%m%d")
        folder.mkdir(parents=True)
        tick_rows, clock_rows = [], []
        for tick in cls.ticks:
            for minute in range(240):
                clock = 570 + minute if minute < 120 else 780 + minute - 120
                for _ in range(int(rng.integers(1, 6))):
                    tick_rows.append(tick)
                    clock_rows.append(time(clock // 60, clock % 60, 5))
        pl.DataFrame({"SecurityID": tick_rows, "TransactTime": clock_rows,
                      "Price": np.full(len(tick_rows), 10.),
                      "OrderQty": np.full(len(tick_rows), 100)}).write_parquet(folder / "shcj.pq")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_new_factor_interfaces_and_default_config(self):
        for name in NAMES:
            with self.subTest(factor=name):
                spec = get_factor_spec(name)
                self.assertEqual(spec.name, name.lower())
                inspect.signature(spec.context_class).bind(root=self.root, universe="self")
                inspect.signature(spec.factor_class).bind(None)
                with spec.create_context(root=self.root) as context:
                    self.assertEqual(context.config.lookback_days, 10)
                    self.assertEqual(context.config.lag_days, 1)
                    self.assertEqual(context.config.min_valid_days, 10)

    def test_factor_scripts_are_independent(self):
        for name in NAMES:
            spec = get_factor_spec(name)
            source = Path(inspect.getfile(spec.factor_class)).read_text(encoding="utf-8")
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn("pricevolume", node.module or "")
            self.assertEqual(spec.context_class.__bases__, (backtest.AlphaContext,))
            self.assertEqual(spec.factor_class.__bases__, (backtest.AlphaBase,))

    def test_process_workers_match_serial_and_close(self):
        for name in ("vr_quantile_resid_corr", "bayesHighVol_onset", "sr_dispersion_stdnumtrades"):
            spec = get_factor_spec(name)
            config = inspect.signature(spec.context_class).parameters["config"].default
            config = replace(config, show_progress=False, workers=1)
            with spec.create_context(root=self.root, config=config) as context:
                expected = context.daily_features(9)[spec.name].copy()
            with spec.create_context(root=self.root, config=replace(config, workers=2)) as context:
                actual = context.daily_features(9)[spec.name].copy()
                self.assertIsNotNone(context._executor)
            self.assertIsNone(context._executor)
            np.testing.assert_array_equal(actual, expected)

    def test_all_eleven_calculate_store_and_plot_by_name(self):
        for name in NAMES:
            with self.subTest(factor=name):
                result = backtest.run_from_ide(
                    name, root=self.root, start_date=str(self.dates[10]),
                    end_date=str(self.dates[10]), horizons=(1, 100), return_offset=1,
                    output_dir=self.root / "plots", num_groups=2,
                    context_kwargs={"config": {"lookback_days": 1, "min_valid_days": 1,
                                                "show_progress": False}},
                )
                output, stats = result[name]["output"], result[name]["stats"]
                self.assertTrue(output.is_file())
                self.assertGreater(output.stat().st_size, 1000)
                self.assertEqual(stats.iloc[0]["valid_dates"], 1)
                self.assertEqual(stats.iloc[-1]["valid_dates"], 0)
                self.assertTrue(list(output.parent.glob(f"{name.lower()}_stats_*.csv")))
                matrix = self.root / "stock/factor_pool" / f"{name.lower()}.bin"
                stored = np.fromfile(matrix, dtype=np.float32).reshape(15, 17)
                self.assertTrue(np.isnan(stored[10, -1]))
                self.assertTrue(np.isfinite(stored[10, :16]).all())

    def test_plot_only_does_not_recompute(self):
        name = "vr_q80_ret_corr"
        spec = get_factor_spec(name)
        stored = self.root / "stock/factor_pool" / f"{name}.bin"
        stored.parent.mkdir(parents=True, exist_ok=True)
        np.random.default_rng(32).normal(size=(15, 17)).astype(np.float32).tofile(stored)
        with patch.object(spec.context_class, "__init__", side_effect=AssertionError("recomputed")):
            output, _ = backtest.run_registered(
                name, root=self.root, calculate=False,
                start_date=str(self.dates[10]), end_date=str(self.dates[11]),
                horizons=(1,), output_dir=self.root / "plots",
            )
        self.assertTrue(output.is_file())

    def test_short_forward_labels_and_cli(self):
        returns = np.arange(12, dtype=float).reshape(6, 2) / 100
        actual = backtest._forward_label(returns, (4, 2), 2, 5, 2, 1)
        np.testing.assert_allclose(actual[0], (1 + returns[3]) * (1 + returns[4]) - 1)
        self.assertTrue(np.isnan(actual[2:]).all())
        self.assertTrue(np.isnan(backtest._forward_label(returns, (4, 2), 2, 5, 20, 1)).all())
        with patch.object(backtest, "run_from_ide", return_value={}) as run:
            backtest.main(["highVol_onset", "--start", "2024-01-01", "--config", '{"sparse_minutes": 12}'])
            self.assertEqual(run.call_args.kwargs["context_kwargs"], {"config": {"sparse_minutes": 12}})


if __name__ == "__main__":
    unittest.main()
