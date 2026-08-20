from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from v2.GetData import DataPool


class DataPoolTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        axis = self.root / "axis"
        axis.mkdir(parents=True)

        dates = np.full(4, np.datetime64("NaT"), dtype="datetime64[D]")
        dates[:3] = np.asarray(
            ["2024-01-02", "2024-01-03", "2024-01-04"],
            dtype="datetime64[D]",
        )
        ticks = np.full(5, "", dtype="<U6")
        ticks[:3] = ["000001", "600000", "688001"]
        np.save(axis / "dates.npy", dates, allow_pickle=False)
        np.save(axis / "stock_ticks.npy", ticks, allow_pickle=False)
        fund_ticks = np.asarray(["510300", "159919", ""], dtype="<U6")
        np.save(axis / "fund_ticks.npy", fund_ticks, allow_pickle=False)

        stock = self.root / "stock"
        self.close = np.arange(20, dtype=np.float32).reshape(4, 5)
        self.flag = np.zeros((4, 5), dtype=np.bool_)
        self.flag[1, 1] = True
        self.forecast = np.arange(
            4 * 4 * 5, dtype=np.float32
        ).reshape(4, 4, 5)
        self.minute = np.arange(
            4 * 241 * 5, dtype=np.float32
        ).reshape(4, 241, 5)

        self._write(stock / "d_essentials" / "close.bin", self.close)
        fund_nav = np.arange(12, dtype=np.float32).reshape(4, 3)
        self._write(self.root / "fund" / "nav" / "unit.bin", fund_nav)
        self._write(stock / "basic" / "listed.bin", self.flag)
        self._write(
            stock / "zyyx" / "con_forecast" / "eps.bin",
            self.forecast,
        )
        self._write(
            stock / "m_essentials" / "close.bin",
            self.minute,
        )
        l2 = stock / "l2" / "proc" / "not_matrix.bin"
        l2.parent.mkdir(parents=True)
        l2.write_bytes(b"raw")
        self.pool = DataPool(self.root)

    def tearDown(self):
        self.pool.close()
        self.temp.cleanup()

    @staticmethod
    def _write(path, values):
        path.parent.mkdir(parents=True, exist_ok=True)
        values.tofile(path)

    def test_axes_discovery_and_specs(self):
        pool = self.pool
        self.assertEqual(len(pool["trade_dates"]), 3)
        self.assertEqual(
            pool["stock_ticks"].tolist(),
            ["000001", "600000", "688001"],
        )
        self.assertNotIn("l2/proc/not_matrix", pool.fields())

        close = pool.field_spec("d_essentials/close")
        forecast = pool.field_spec(
            "stock/zyyx/con_forecast/eps.bin"
        )
        minute = pool.field_spec("m_essentials/close")
        flag = pool.field_spec("basic/listed")
        self.assertEqual(close.shape, (4, 5))
        self.assertEqual(close.dtype, np.dtype(np.float32))
        self.assertEqual(forecast.shape, (4, 4, 5))
        self.assertEqual(minute.shape, (4, 241, 5))
        self.assertEqual(flag.dtype, np.dtype(np.bool_))

    def test_date_and_tick_selection(self):
        pool = self.pool
        row = pool.read(
            "d_essentials/close",
            "2024-01-03",
            ticks=["688001", "000001"],
        )
        np.testing.assert_array_equal(row, self.close[1, [2, 0]])

        history = pool.read(
            "d_essentials/close",
            "2024-01-04",
            "2024-01-02",
        )
        np.testing.assert_array_equal(history, self.close[:3, :3])


    def test_two_and_three_dimensional_interfaces(self):
        pool = self.pool
        frame = pool.get_field("d_essentials/close")
        self.assertEqual(frame.shape, (3, 3))
        self.assertEqual(frame.index[0], np.datetime64("2024-01-02"))
        self.assertEqual(frame.columns.tolist()[1], "600000")

        forecast = pool.read(
            "zyyx/con_forecast/eps",
            "2024-01-03",
        )
        np.testing.assert_array_equal(forecast, self.forecast[1, :, :3])

        minute = pool.read(
            "m_essentials/close",
            "2024-01-03",
            ticks=["600000"],
        )
        np.testing.assert_array_equal(
            minute,
            self.minute[1, :, [1]].T,
        )

        raw = pool["d_essentials/close"]
        self.assertIsInstance(raw, np.memmap)
        self.assertEqual(raw.shape, (4, 5))

    def test_another_asset_module(self):
        with DataPool(self.root, asset="fund") as fund:
            self.assertEqual(
                fund["ticks"].tolist(),
                ["510300", "159919"],
            )
            nav = fund.read(
                "nav/unit",
                "2024-01-03",
                ticks=["159919"],
            )
            np.testing.assert_array_equal(
                nav,
                np.asarray([4], dtype=np.float32),
            )
    def test_invalid_requests(self):
        pool = self.pool
        with self.assertRaises(KeyError):
            pool.read(
                "d_essentials/close",
                "2024-01-06",
            )
        with self.assertRaises(KeyError):
            pool.read(
                "d_essentials/close",
                "2024-01-03",
                ticks=["999999"],
            )
        with self.assertRaises(ValueError):
            pool.get_field("zyyx/con_forecast/eps")


if __name__ == "__main__":
    unittest.main()