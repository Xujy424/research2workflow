"""Offline checks for event timing and AOG formulas."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from v2.UpdateAlpha.analyst_forecast.aog import (
    AOGContext, AOGConfig, AOGDemaxFactor, _event_statistics,
)


class Axis:
    trade_dates = pd.bdate_range("2024-01-01", periods=30).to_numpy(dtype="datetime64[D]")
    tick_count = 2
    _tick_positions = {"000001": 0, "000002": 1}

    def date_position(self, value):
        return int(np.flatnonzero(self.trade_dates == np.datetime64(value, "D"))[0])


class Pool:
    axis = Axis()

    def __init__(self, *args, **kwargs):
        pass

    def read(self, field, end_date, *args, **kwargs):
        if field.endswith("open_adj") or field.endswith("low_adj"):
            return np.array([101., 99.]) if end_date == 22 else np.array([99., 101.])
        if field.endswith("close_adj"):
            return np.array([100., 100.])
        return np.ones(2)

    def close(self):
        pass


class AOGTests(unittest.TestCase):
    def context(self, **kwargs):
        dates = Axis.trade_dates
        events = pd.DataFrame({
            "tick": ["000001", "000001", "000002"],
            "end_date": ["2023-12-31", "2023-12-31", "2023-12-31"],
            "publish_date": [dates[21], dates[24], dates[28]],
        })
        with patch("v2.UpdateAlpha.analyst_forecast.aog.DataPool", Pool):
            return AOGContext(announcements=events, **kwargs)

    def test_pre_event_window(self):
        current = np.array([1., .5])
        history = np.tile([.5, 1.], (20, 1))
        demax, quantile = _event_statistics(current, history, AOGConfig())
        np.testing.assert_allclose(demax, [.5, -.5])
        np.testing.assert_allclose(quantile, [.5, 1.])

    def test_timing_carry_and_future(self):
        with self.context() as ctx:
            d = Axis.trade_dates
            self.assertTrue(np.isnan(ctx.values(d[21])["rank"]).all())
            first = ctx.values(d[22])
            self.assertEqual(first["demax"][0], .5)
            self.assertEqual(first["quantile"][0], .5)
            self.assertTrue(np.isnan(first["rank"][1]))
            # A correction to the same period must not reset the event.
            np.testing.assert_allclose(ctx.values(d[25])["rank"], first["rank"])
            ctx.values(d[29])
            np.testing.assert_allclose(ctx.values(d[22])["rank"], first["rank"])
            self.assertEqual(AOGDemaxFactor(ctx).calculate(d[22]).dtype, np.float32)

    def test_expiry(self):
        with self.context(config=AOGConfig(max_event_age=1)) as ctx:
            self.assertTrue(np.isnan(ctx.values(Axis.trade_dates[24])["rank"]).all())

    def test_same_day_option(self):
        with self.context(config=AOGConfig(announcement_same_day=True)) as ctx:
            self.assertEqual(ctx.values(Axis.trade_dates[21])["rank"][0], .5)


if __name__ == "__main__":
    unittest.main()
