from pathlib import Path
from tempfile import TemporaryDirectory
import importlib
import unittest
from unittest.mock import patch

import numpy as np

from v2.UpdateData import daily_update
from v2.UpdateData.utils import ensure_memmap

axis_reset = importlib.import_module("v2.UpdateData.axis.reset_axis")
date_axis = importlib.import_module("v2.UpdateData.axis.update_date")
tick_axis = importlib.import_module("v2.UpdateData.axis.update_stockticks")


class AxisFlowTest(unittest.TestCase):
    def test_resize_insert_append_and_l2_exclusion(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            stock_root = root / "stock"
            dates_path, ticks_path = axis_reset.init_axis(
                root, date_reserve=3, tick_reserve=3
            )
            dates = np.load(dates_path, mmap_mode="r+")
            dates[:2] = np.asarray(
                ["2024-01-02", "2024-01-04"], dtype="datetime64[D]"
            )
            dates.flush()
            del dates
            ticks = np.load(ticks_path, mmap_mode="r+")
            ticks[:2] = ["000001", "600000"]
            ticks.flush()
            del ticks

            shape = (3, 3)
            flag = ensure_memmap(
                stock_root / "basic" / "flag.bin",
                shape,
                dtype=np.bool_,
            )
            value32 = ensure_memmap(
                stock_root / "d_essentials" / "close.bin",
                shape,
                dtype=np.float32,
            )
            value64 = ensure_memmap(
                stock_root / "barra" / "beta.bin",
                shape,
                dtype=np.float64,
            )
            minute = ensure_memmap(
                stock_root / "m_essentials" / "close.bin",
                (3, 241, 3),
                dtype=np.float32,
            )
            forecast = ensure_memmap(
                stock_root / "zyyx" / "con_forecast" / "eps.bin",
                (3, 4, 3),
                dtype=np.float64,
            )
            flag[0, 0], flag[1, 0] = True, True
            value32[0, 0], value32[1, 0] = 10, 20
            value64[0, 0], value64[1, 0] = 100, 200
            minute[0, 0, 0], minute[1, 0, 0] = 1, 2
            forecast[0, 0, 0], forecast[1, 0, 0] = 3, 4
            for array in (flag, value32, value64, minute, forecast):
                array.flush()
            del array, flag, value32, value64, minute, forecast

            l2_file = stock_root / "l2" / "proc" / "raw.bin"
            l2_file.parent.mkdir(parents=True)
            l2_file.write_bytes(b"not-a-matrix")

            date_axis.update_date("2024-01-03", root)
            dates = np.load(dates_path)
            np.testing.assert_array_equal(
                dates[:3].astype("datetime64[D]"),
                np.asarray(
                    ["2024-01-02", "2024-01-03", "2024-01-04"],
                    dtype="datetime64[D]",
                ),
            )
            value32 = np.memmap(
                stock_root / "d_essentials" / "close.bin",
                dtype=np.float32,
                mode="r",
                shape=shape,
            )
            self.assertEqual(value32[0, 0], 10)
            self.assertTrue(np.isnan(value32[1, 0]))
            self.assertEqual(value32[2, 0], 20)
            del value32

            with patch.object(axis_reset, "DATE_RESERVE", 2), patch.object(
                axis_reset, "TICK_RESERVE", 2
            ):
                resize = axis_reset.ensure_axis_capacity(
                    root, min_date_free=2, min_tick_free=2
                )
                self.assertTrue(resize.changed)
                with patch.object(
                    tick_axis,
                    "get_all_ticks",
                    return_value=np.asarray(
                        ["000001", "600000", "001234", "688001"],
                        dtype="<U6",
                    ),
                ):
                    added = tick_axis.update_stockticks(
                        "2024-01-04", root
                    )

            self.assertEqual(added, ["001234", "688001"])
            stock_ticks = np.load(ticks_path)
            self.assertEqual(
                stock_ticks[:4].tolist(),
                ["000001", "600000", "001234", "688001"],
            )
            self.assertEqual(l2_file.read_bytes(), b"not-a-matrix")

            new_dates, new_ticks = len(np.load(dates_path)), len(stock_ticks)
            self.assertEqual(
                (stock_root / "basic" / "flag.bin").stat().st_size,
                new_dates * new_ticks,
            )
            self.assertEqual(
                (stock_root / "m_essentials" / "close.bin").stat().st_size,
                new_dates * 241 * new_ticks * 4,
            )
            self.assertEqual(
                (
                    stock_root / "zyyx" / "con_forecast" / "eps.bin"
                ).stat().st_size,
                new_dates * 4 * new_ticks * 8,
            )

    def test_calendar_and_non_trading_day_short_circuit(self):
        self.assertTrue(date_axis.is_tradedate("2024-06-14"))
        self.assertFalse(date_axis.is_tradedate("2024-06-15"))
        self.assertTrue(
            date_axis.is_last_tradedate_of_year("2024-12-31")
        )
        with TemporaryDirectory() as folder:
            result = daily_update.update_data(
                Path(folder), "2024-06-15"
            )
            self.assertEqual(result["status"], "skipped")
            self.assertTrue(
                (Path(folder) / "axis" / "stock_ticks.npy").exists()
            )

    def test_daily_update_uses_one_axis_for_all_updaters(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            calls = []

            def recorder(name):
                def update(date, dates, ticks, *args, **kwargs):
                    calls.append((
                        name, date, len(dates), len(ticks), Path(args[-1])
                    ))
                    return {}
                return update

            functions = {
                name: recorder(name)
                for name in (
                    "update_d_essentials", "update_m_essentials",
                    "update_basic", "update_tradable",
                    "update_industry", "update_sector", "update_index",
                    "update_zyyx", "update_fundamental",
                    "update_dividend", "update_barra",
                )
            }
            with patch.object(
                daily_update, "update_stockticks", return_value=[]
            ), patch.multiple(daily_update, **functions):
                result = daily_update.update_data(
                    root,
                    "2024-06-14",
                    jy_conn=object(),
                    zyyx_conn=object(),
                    str_conn=object(),
                    update_level2=False,
                )

            self.assertEqual(result["status"], "updated")
            self.assertEqual(len(calls), len(functions))
            self.assertEqual({call[2:4] for call in calls}, {(500, 5000)})
            self.assertEqual(
                {call[4] for call in calls if call[0] != "update_zyyx"},
                {root / "stock"},
            )
            self.assertEqual(
                {call[4] for call in calls if call[0] == "update_zyyx"},
                {root / "stock" / "zyyx"},
            )


if __name__ == "__main__":
    unittest.main()
