import tempfile
import unittest
from unittest.mock import Mock, patch
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from .update_iv30 import calculate_iv30, maturity_variance, update_iv30, resolve_rate
from ..axis.reset_axis import ensure_axis_capacity, insert_matrix_date_rows
from ..axis.delete import delete_before


class IV30Tests(unittest.TestCase):
    def test_edb_rate_resolution(self):
        day = pd.Timestamp("2026-10-26")
        edb = Mock(return_value=pd.Series([2.0], index=[day]))
        self.assertEqual(resolve_rate(day, 0.03, edb=edb), 0.03)
        edb.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            with patch("v2.UpdateData.option.update_iv30.calculate_iv30", return_value=25.0) as calculate:
                update_iv30(day, np.array([day], dtype="datetime64[D]"), self.reader,
                            directory, rate=None, edb=edb, rate_code="test")
                self.assertAlmostEqual(calculate.call_args.kwargs["rate"], np.log1p(0.02))
        edb.assert_called_once_with("test", "2026-10-26", "2026-10-26")
        for days in (20, 50, 180):
            rate = resolve_rate(day, None, edb=edb)
            self.assertAlmostEqual(np.exp(rate * days / 365), 1.02**(days / 365))
        self.assertEqual(resolve_rate(day, None, edb=edb, rate_compounding="continuous"), .02)
        edb.return_value = pd.Series([2.0], index=[day - pd.Timedelta(days=1)])
        with self.assertRaisesRegex(ValueError, "missing or invalid rate"):
            resolve_rate(day, None, edb=edb, rate_code="test")
        with self.assertRaisesRegex(ValueError, "requires edb"):
            resolve_rate(day, None)

    def test_edb_and_explicit_continuous_rate_equivalent(self):
        day = pd.Timestamp("2026-10-26")
        edb = Mock(return_value=pd.Series([2.0], index=[day]))
        dates = np.array([day], dtype="datetime64[D]")
        with tempfile.TemporaryDirectory() as directory:
            for prefix in ("IO", "MO"):
                explicit = update_iv30(day, dates, self.reader, directory,
                                       prefix=prefix, rate=np.log1p(.02))
                fetched = update_iv30(day, dates, self.reader, directory,
                                      prefix=prefix, rate=None, edb=edb)
                self.assertAlmostEqual(explicit, fetched)

    def reader(self, code, field, start, end):
        self.calls += 1
        if code in ("000300.SH", "000852.SH"):
            spot = 4000 if code == "000300.SH" else 6000
            value = spot
        else:
            month, side, strike = code.removesuffix(".CFE").split("-")
            spot = 4000 if month.startswith("IO") else 6000
            strike = int(strike)
            value = 20 + max(spot - strike if side == "C" else strike - spot, 0)
        return pd.Series([value], index=[pd.Timestamp(start)])

    def setUp(self):
        self.calls = 0

    def test_formula_and_zero_bid_cutoff(self):
        k = np.arange(60, 141, 10, dtype=float)
        prices = np.column_stack((2 + np.maximum(100 - k, 0), 2 + np.maximum(k - 100, 0)))
        expected = 2 / 0.1 * np.sum(10 * 2 / k**2)
        self.assertAlmostEqual(maturity_variance(k, prices, .1, 0), expected)
        grid = np.arange(60, 141, 5, dtype=float)
        sparse = np.full((len(grid), 2), np.nan)
        sparse[::2] = prices
        self.assertAlmostEqual(maturity_variance(grid, sparse, .1, 0), expected)
        bids = prices - .1
        bids[5:7, 0] = 0
        self.assertTrue(np.isnan(maturity_variance(k, prices, .1, 0, bids=bids, min_strikes=3)))

    def test_both_indices_binary_and_request_count(self):
        dates = np.array(["2026-10-26", "NaT", "NaT"], dtype="datetime64[D]")
        with tempfile.TemporaryDirectory() as directory:
            for prefix, name, expected_calls in [("IO", "HS300_IV30", 229), ("MO", "CSI1000_IV30", 341)]:
                self.calls = 0
                value = update_iv30("2026-10-26", dates, self.reader, directory, prefix=prefix)
                self.assertTrue(np.isfinite(value))
                self.assertEqual(self.calls, expected_calls)
                path = Path(directory) / (name + ".bin")
                data = np.fromfile(path, dtype=np.float32).reshape(3, 1)
                self.assertAlmostEqual(float(data[0, 0]), value, places=4)
                self.assertTrue(np.isnan(data[1:]).all())
                update_iv30("2026-10-26", dates, self.reader, directory, prefix=prefix)
                self.assertEqual(path.stat().st_size, 12)
            self.assertEqual(len(list(Path(directory).iterdir())), 2)

    def test_no_extrapolation(self):
        # Just after expiry: the next listed month is more than 30 days away.
        self.assertTrue(np.isnan(calculate_iv30("2026-07-18", self.reader, "IO")))

    def test_axis_growth_insertion_and_pruning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "axis").mkdir()
            dates = np.array(["2026-10-26", "2026-10-28", "NaT"], dtype="datetime64[D]")
            np.save(root / "axis/dates.npy", dates)
            np.save(root / "axis/stock_ticks.npy", np.array(["000001", ""], dtype="<U6"))
            (root / "stock/d_essentials").mkdir(parents=True)
            np.array([[1, np.nan], [2, np.nan], [np.nan, np.nan]], dtype=np.float32).tofile(root / "stock/d_essentials/close.bin")
            for prefix in ("IO", "MO"):
                update_iv30("2026-10-26", dates, self.reader, root / "option", prefix=prefix)
            paths = list((root / "option").glob("*.bin"))
            for path in paths:
                data = np.memmap(path, dtype=np.float32, mode="r+", shape=(3, 1))
                data[1, 0] = 12.5
                data.flush()
                del data
            values = [np.fromfile(p, dtype=np.float32)[0] for p in paths]
            result = ensure_axis_capacity(root, min_date_free=5, min_tick_free=5)
            for path, value in zip(paths, values):
                data = np.fromfile(path, dtype=np.float32)
                self.assertEqual(len(data), result.new_date_len)
                self.assertEqual(data[0], value)
            dates = np.load(root / "axis/dates.npy")
            insert_matrix_date_rows(root, 1, 2, len(dates), result.new_tick_len)
            dates[2], dates[1] = dates[1], np.datetime64("2026-10-27")
            np.save(root / "axis/dates.npy", dates)
            for path in paths:
                data = np.fromfile(path, dtype=np.float32)
                self.assertTrue(np.isnan(data[1]))
                self.assertEqual(data[2], 12.5)
            delete_before("2026-10-27", root, date_reserve=2, tick_reserve=1)
            for path in paths:
                self.assertEqual(path.stat().st_size, 4 * 4)
                data = np.fromfile(path, dtype=np.float32)
                self.assertTrue(np.isnan(data[0]))
                self.assertEqual(data[1], 12.5)
                self.assertTrue(np.isnan(data[2:]).all())


if __name__ == "__main__":
    unittest.main()
