import tempfile
from dataclasses import replace
import unittest
import json
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
import numpy as np
import pandas as pd
from ..schema import MarketData
from ..preprocess import preprocess
from ..bars import build_bars, minute_returns
from ..features.extended import lifecycle
from ..monitor import detect_regime
from ..history import detect_migration
from ..quality import ensure_history_identity, quality_features
from ..data.daily_loader import load_v2_daily
from ..statistics.migration import compare_previous
from ..examples.example_usage import make_demo_data
from ..statistics import compare_universes, compare_with_history
from ..features import compute_features
from ..features.minute_features import build_feature_minutes, dynamic_intraday_shape
from ..statistics.security_history import security_history
from ..statistics.memberships import group_minute_history, membership_history
from ..features.depth_recovery import depth_recovery
from ..features.context import compute_event_context
from ..statistics.aggregation import aggregate_features


class Contracts(unittest.TestCase):
    def test_prefetch_keeps_history_writes_chronological(self):
        from ..examples import run_l2_range as runner
        with tempfile.TemporaryDirectory() as directory:
            root, output = Path(directory)/"l2", Path(directory)/"reports"
            dates = ["20260722", "20260723", "20260724"]
            for date in dates:
                (root/"proc"/date).mkdir(parents=True)
            argv = ["run_l2_range", "--root", str(root), "--start", dates[0], "--end", dates[-1], "--output", str(output), "--prefetch", "3"]
            with patch("sys.argv", argv), patch.object(runner, "load_proc_day", side_effect=lambda root, date, symbols, **kwargs: date), patch.object(runner, "run_daily_monitor", return_value=SimpleNamespace(regime={}, dashboard_path=output/"monitor.html")) as monitor:
                runner.main()
            self.assertEqual([call.args[0] for call in monitor.call_args_list], dates)
            self.assertEqual([call.args[1] for call in monitor.call_args_list], dates)

    def test_prefiltered_l2_reader_preserves_selected_events(self):
        import polars as pl
        from ..examples.run_l2_day import load_proc_day
        with tempfile.TemporaryDirectory() as directory:
            day = Path(directory)/"proc"/"20260724"
            day.mkdir(parents=True)
            for exchange, symbols in (("sh", [600000, 600001]), ("sz", [1, 2])):
                for kind in ("wt", "cj", "cancel"):
                    frame = pl.DataFrame({"TransactTime": ["09:30:01", "09:30:02"], "SecurityID": symbols, "Price": [10., 11.], "OrderQty": [100, 200], "Side": [1, -1], "ChannelNo": [1, 1]})
                    if kind == "wt":
                        frame = frame.with_columns(pl.Series("ApplSeqNum", [1, 2]))
                    else:
                        frame = frame.with_columns(pl.Series("BidApplSeqNum", [1, 2]), pl.Series("OfferApplSeqNum", [3, 4]))
                    frame.write_parquet(day/f"{exchange}{kind}.pq", row_group_size=1)
            automatic = load_proc_day(directory, "20260724", ["000001.SZ", "600000.SH"], cache_dir=None)
            filtered = load_proc_day(directory, "20260724", ["000001.SZ", "600000.SH"], cache_dir=None, read_strategy="prefiltered")
            for name in ("orders", "trades", "quotes"):
                pd.testing.assert_frame_equal(getattr(automatic, name), getattr(filtered, name))

    def test_real_range_resume_and_final_recompute(self):
        from ..examples import run_l2_range as runner
        with tempfile.TemporaryDirectory() as directory:
            root, output = Path(directory)/"l2", Path(directory)/"reports"
            for date in ("20260723", "20260724"):
                (root/"proc"/date).mkdir(parents=True)
                (output/date).mkdir(parents=True)
                for name in ("features.csv", "baseline.csv", "memberships.json", "bars_feature_1min.csv"):
                    (output/date/name).write_text("", encoding="utf-8")
            ensure_history_identity(output, ["000001.SZ", "600000.SH"], "selected:000001.SZ,600000.SH")
            pd.DataFrame({"metric": [1., 2.]}, index=["2026-07-23", "2026-07-24"]).to_csv(output/"history.csv")
            argv = ["run_l2_range", "--root", str(root), "--start", "20260723", "--end", "20260724", "--output", str(output), "--resume"]
            with patch("sys.argv", argv), patch.object(runner, "load_proc_day", return_value=object()) as loader, patch.object(runner, "run_daily_monitor", return_value=SimpleNamespace(regime={}, dashboard_path=output/"monitor.html")) as monitor:
                runner.main()
            self.assertEqual(loader.call_count, 1)
            self.assertEqual(loader.call_args.args[1], "20260724")
            self.assertEqual(monitor.call_args.kwargs["data_source"], "v2_proc")
            status = json.loads((output/"run_status.json").read_text())
            self.assertEqual(status["state"], "complete")
            self.assertEqual(status["completed_dates"], ["20260723", "20260724"])

    def test_real_range_records_read_failure(self):
        from ..examples import run_l2_range as runner
        with tempfile.TemporaryDirectory() as directory:
            root, output = Path(directory)/"l2", Path(directory)/"reports"
            (root/"proc"/"20260724").mkdir(parents=True)
            argv = ["run_l2_range", "--root", str(root), "--start", "20260724", "--end", "20260724", "--output", str(output)]
            with patch("sys.argv", argv), patch.object(runner, "load_proc_day", side_effect=OSError("read failed")):
                with self.assertRaises(OSError):
                    runner.main()
            status = json.loads((output/"run_status.json").read_text())
            self.assertEqual(status["state"], "failed")
            self.assertEqual(status["completed_dates"], [])
            self.assertIn("read failed", status["error"])

    def test_partial_and_invalid_depth(self):
        original = make_demo_data()
        for missing in ("ask_volume1", "bid_volume1"):
            data = preprocess(replace(original, quotes=original.quotes.drop(columns=missing)), "2026-09-29")
            bars = build_bars(data)
            liquidity = compute_features(data, bars)["liquidity"]
            self.assertTrue(liquidity["ask_depth" if missing == "ask_volume1" else "bid_depth"].isna().all())
            self.assertTrue(liquidity["bid_depth" if missing == "ask_volume1" else "ask_depth"].notna().all())
            self.assertTrue(liquidity.total_depth_valid_quotes.eq(0).all())
        quotes = original.quotes.copy()
        quotes["bid_volume1"] = -100.
        data = preprocess(replace(original, quotes=quotes), "2026-09-29")
        self.assertTrue(data.quotes.bid_volume1.isna().all())
        self.assertTrue(compute_features(data, build_bars(data))["liquidity"].bid_depth.isna().all())

    def test_group_paired_depth_and_event_weights(self):
        frame = pd.DataFrame({"liquidity.bid_depth": [1., 100.], "liquidity.ask_depth": [np.nan, 100.], "discovery.large_ret_5m": [1., 0.], "discovery.large_samples_5m": [1., 9.]})
        group = compare_universes(frame).loc["all"]
        self.assertEqual(group["liquidity.depth_imbalance"], 0.)
        self.assertEqual(group["liquidity.depth_pair_samples"], 1)
        self.assertAlmostEqual(group["discovery.large_ret_5m"], .1)
        self.assertEqual(group["discovery.large_samples_5m"], 10)

    def test_depth_pair_means_share_snapshot_population(self):
        original = make_demo_data()
        quotes = original.quotes.copy()
        quotes.loc[0, ["bid_volume1", "ask_volume1"]] = [1e8, np.nan]
        data = preprocess(replace(original, quotes=quotes), "2026-09-29")
        bars = build_bars(data)
        from ..schema import flatten_feature_frames
        result = compare_universes(flatten_feature_frames(compute_features(data, bars))).loc["all"]
        self.assertAlmostEqual(result["liquidity.depth_imbalance"], (10000-12000)/(10000+12000))

    def test_security_history_preserves_numeric_identifier(self):
        with tempfile.TemporaryDirectory() as directory:
            day = Path(directory)/"20260101"
            day.mkdir()
            pd.DataFrame({"trade.amount": [10.]}, index=["000001"]).to_csv(day/"features.csv")
            _, baseline = security_history(directory, pd.DataFrame({"trade.amount": [20.]}, index=["000001"]), "2026-01-02")
            self.assertEqual(baseline.loc[("000001", "trade.amount"), "sample_count"], 1)

    def test_packed_report_table_roundtrip(self):
        from ..explorer import records
        frame = pd.DataFrame({"value": [1., np.nan]}, index=pd.Index(["000001", "000002"], name="symbol"))
        packed = records(frame)
        self.assertEqual(packed["columns"], ["symbol", "value"])
        self.assertEqual(packed["data"], [["000001", 1.], ["000002", None]])

    def test_group_curves_recompute_agreement_and_correlation(self):
        rows = []
        for time, oi, ti in zip(pd.date_range("2026-01-02 09:30", periods=4, freq="min"), [-.8, -.2, .2, .8], [-.9, -.1, .1, .9]):
            for symbol, small, xl in (("X", 10., -1.), ("Y", -1., 10.)):
                rows.append({"symbol": symbol, "time": time, "flow.small_flow": small, "flow.xl_flow": xl, "order.buy_amount": 1+oi, "order.sell_amount": 1-oi, "flow.small_buy_amount": 1+ti, "flow.small_sell_amount": 1-ti})
        minute = pd.DataFrame(rows)
        daily = pd.DataFrame({"flow.small_xl_agreement": [0., 0.], "flow.oi_ti_corr": [-1., -1.]}, index=["X", "Y"])
        result = compare_universes(daily, minute=minute).loc["all"]
        self.assertEqual(result["flow.small_xl_agreement"], 1.)
        self.assertGreater(result["flow.oi_ti_corr"], .9)

    def test_no_shock_minutes_keep_recovery_schema(self):
        data = preprocess(make_demo_data(), "2026-09-29")
        minute = build_feature_minutes(data, build_bars(data))
        for side in ("bid", "ask", "total"):
            self.assertTrue(minute[f"liquidity.{side}_depth_shock_count"].eq(0).all())
            self.assertTrue(minute[f"liquidity.{side}_depth_recovery_seconds"].isna().all())
        quotes = data.quotes.drop(columns=["bid_volume1", "ask_volume1"])
        minute = build_feature_minutes(replace(data, quotes=quotes), build_bars(replace(data, quotes=quotes)))
        self.assertTrue(minute["liquidity.total_depth_shock_count"].isna().all())

    def test_curve_distance_covers_spread_and_signed_flow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            day = root/"20260101"
            day.mkdir()
            features = pd.DataFrame({"trade.amount": [1.]}, index=["000001"])
            features.to_csv(day/"features.csv")
            previous = pd.DataFrame({"symbol": ["000001"]*2, "time": pd.to_datetime(["2026-01-01 09:30", "2026-01-01 09:31"]), "liquidity.spread": [1., 2.], "flow.small_flow": [-1., 1.]})
            previous.to_csv(day/"bars_feature_1min.csv", index=False)
            current = previous.copy()
            current["time"] += pd.Timedelta(days=1)
            current["flow.small_flow"] *= -1
            result = compare_previous(features, current, root, "2026-01-02")
            self.assertEqual(result.loc["liquidity.spread", "curve_l1"], 0.)
            self.assertEqual(result.loc["flow.small_flow", "curve_l1"], 2.)
            self.assertEqual(result.loc["flow.small_flow", "curve_common_minutes"], 2)

    def test_shared_event_context_computes_once(self):
        from ..features import context as module
        data = preprocess(make_demo_data(), "2026-09-29")
        bars = build_bars(data)
        with patch.object(module, "lifecycle", wraps=module.lifecycle) as life, patch.object(module, "event_returns", wraps=module.event_returns) as returns, patch.object(module, "depth_recovery", wraps=module.depth_recovery) as recovery:
            context = compute_event_context(data)
            compute_features(data, bars, context)
            build_feature_minutes(data, bars, context)
            self.assertEqual([life.call_count, returns.call_count, recovery.call_count], [1, 1, 1])

    def test_batch_aggregation_matches_daily_and_missing_weights(self):
        data = preprocess(make_demo_data(), "2026-09-29")
        minute = build_feature_minutes(data, build_bars(data))
        minute["clock"] = minute.time.dt.strftime("%H:%M")
        batch = aggregate_features(minute, by="clock")
        for clock in ("09:30", "13:00", "14:56"):
            daily = compare_universes(minute.loc[minute.clock == clock].set_index("symbol")).loc["all"]
            pd.testing.assert_series_equal(batch.loc[clock], daily, check_names=False)
        frame = pd.DataFrame({"order.amount": [100., 900.], "order.large_ratio": [np.nan, 1.]})
        self.assertEqual(compare_universes(frame).loc["all", "order.large_ratio"], 1.)
        frame["order.amount"] = 0.
        self.assertTrue(pd.isna(compare_universes(frame).loc["all", "order.large_ratio"]))

    def test_invalid_side_depth_does_not_create_total_observation(self):
        quotes = pd.DataFrame({"symbol": ["X"], "time": pd.to_datetime(["2026-01-02 09:30"]), "bid_volume1": [-1.], "ask_volume1": [100.]})
        summary, _ = depth_recovery(quotes)
        self.assertEqual(summary.loc["X", "total_depth_valid_quotes"], 0)

    def test_group_order_ratios(self):
        frame = pd.DataFrame({"order.amount": [100., 900.], "order.large_ratio": [0., 1.], "order.xl_ratio": [0., 1.]}, index=["X", "Y"])
        result = compare_universes(frame).loc["all"]
        self.assertEqual(result["order.large_ratio"], .9)
        self.assertEqual(result["order.xl_ratio"], .9)

    def test_historical_member_switch_and_unknown_date(self):
        rows = []
        for date in ("2026-01-01", "2026-01-02", "2026-01-03"):
            for symbol, amount, ratio in (("X", 100., 0.), ("Y", 900., 1.)):
                rows.append({"date": date, "clock": "09:30", "symbol": symbol, "order.amount": amount, "order.large_ratio": ratio, "trade.amount": amount, "trade.large_ratio": ratio})
        memberships = {"2026-01-01": {"all": ["X", "Y"], "G": ["X"]}, "2026-01-02": {"all": ["X", "Y"], "G": ["Y"]}}
        scopes = {"universe.G": {"kind": "group", "name": "G"}, "universe.all": {"kind": "group", "name": "all"}}
        result = group_minute_history(pd.DataFrame(rows), memberships, scopes)
        self.assertEqual([r["order.large_ratio"] for r in result["universe.G"]], [0., 1.])
        self.assertEqual([r["trade.large_ratio"] for r in result["universe.all"]], [.9, .9])
        self.assertEqual(len(result["universe.G"]), 2)

    def test_membership_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            membership_history(directory, "2026-01-01", {"G": ["X"]})
            history = membership_history(directory, "2026-01-02", {"G": ["Y"]})
            self.assertEqual(history["2026-01-01"]["G"], ["X"])
            self.assertEqual(history["2026-01-02"]["G"], ["Y"])

    def test_depth_recovery_and_censoring(self):
        times = pd.date_range("2026-01-02 09:30:00", periods=4, freq="3s")
        quotes = pd.DataFrame({"symbol": "X", "time": times, "bid_volume1": [1000., 300., 600., 950.], "ask_volume1": [1000., 1000., 1000., 1000.]})
        summary, events = depth_recovery(quotes)
        self.assertEqual(summary.loc["X", "bid_depth_recovery_seconds"], 6.)
        self.assertEqual(summary.loc["X", "bid_depth_recovered_count"], 1)
        quotes.loc[3, "bid_volume1"] = 700
        summary, events = depth_recovery(quotes)
        self.assertTrue(pd.isna(summary.loc["X", "bid_depth_recovery_seconds"]))
        self.assertEqual(summary.loc["X", "bid_depth_censored_count"], 1)
        self.assertEqual(events.loc[events.side == "bid", "censored_reason"].iloc[0], "end_of_data")
        quotes.loc[3, "time"] += pd.Timedelta("10min")
        _, events = depth_recovery(quotes)
        self.assertEqual(events.loc[events.side == "bid", "censored_reason"].iloc[0], "observation_gap")
    def test_group_lifecycle_weights(self):
        frame = pd.DataFrame({"lifecycle.context_order_count": [1., 9.], "lifecycle.fill_ratio_sum": [1., 0.], "lifecycle.submitted_volume": [10., 90.], "lifecycle.cancelled_volume": [0., 90.], "lifecycle.lifetime_seconds_sum": [1., 99.], "lifecycle.lifetime_samples": [1., 9.]}, index=["X", "Y"])
        result = compare_universes(frame).loc["all"]
        self.assertAlmostEqual(result["lifecycle.fill_rate"], .1)
        self.assertAlmostEqual(result["lifecycle.cancel_rate"], .9)
        self.assertAlmostEqual(result["lifecycle.lifetime_seconds"], 10.)
    def test_funding_pair_agreement(self):
        original = make_demo_data()
        trades = original.trades.iloc[:4].copy()
        trades["time"] = pd.Timestamp("2026-09-29 09:30:05")
        trades["price"] = 10.
        trades["volume"] = [100, 3000, 20000, 100000]
        trades["direction"] = ["buy", "sell", "buy", "sell"]
        data = preprocess(MarketData(original.orders, trades, original.quotes), "2026-09-29")
        frame = compute_features(data, build_bars(data))["flow"]
        symbol = trades.symbol.iloc[0]
        self.assertEqual(frame.loc[symbol, "small_large_agreement"], 1)
        self.assertEqual(frame.loc[symbol, "small_xl_divergence"], 1)
        self.assertEqual(frame.loc[symbol, "medium_xl_agreement"], 1)

    def test_annual_percentile_requires_full_history(self):
        dates = pd.bdate_range(end="2026-01-01", periods=252)
        past = pd.DataFrame({"x": np.arange(252, dtype=float)}, index=dates)
        today = pd.Series({"x": 120.})
        full = compare_with_history(today, past, "2026-01-02")
        self.assertAlmostEqual(full.loc["x", "pct_1y"], 121/252)
        short = compare_with_history(today, past.tail(60), "2026-01-02")
        self.assertTrue(pd.isna(short.loc["x", "pct_1y"]))
        self.assertTrue(pd.notna(short.loc["x", "pct_history"]))
    def test_individual_stock_baselines(self):
        with tempfile.TemporaryDirectory() as directory:
            day = Path(directory)/"20260101"
            day.mkdir()
            pd.DataFrame({"trade.amount": [10., 100.]}, index=["X", "Y"]).to_csv(day/"features.csv")
            current = pd.DataFrame({"trade.amount": [20., 300.]}, index=["X", "Y"])
            history, baseline = security_history(directory, current, "2026-01-02")
            self.assertEqual(baseline.loc[("X", "trade.amount"), "sample_count"], 1)
            self.assertEqual(baseline.loc[("Y", "trade.amount"), "raw"], 300)
            self.assertEqual(history.loc[(pd.Timestamp("2026-01-01"), "X"), "trade.amount"], 10)

    def test_generic_intraday_metric(self):
        minute = pd.DataFrame({"symbol": ["X", "X"], "time": pd.to_datetime(["2026-01-02 09:31", "2026-01-02 14:40"]), "custom": [1., 3.]})
        shape = dynamic_intraday_shape(minute)
        self.assertEqual(shape.loc["X", "custom_close_minus_open"], 2)
        self.assertEqual(shape.loc["X", "custom_argmax_time"], "14:40")

    def test_minute_size_ratios_and_event_horizons(self):
        data = preprocess(make_demo_data(), "2026-09-29")
        minute = build_feature_minutes(data, build_bars(data))
        expected = (minute.large_amount/minute.amount.replace(0, np.nan)).to_numpy()
        np.testing.assert_allclose(minute["trade.large_ratio"], expected, equal_nan=True)
        for horizon in (1, 5, 10, 30):
            self.assertIn(f"discovery.large_ret_{horizon}m", minute)
        self.assertIn("lifecycle.lifetime_seconds", minute)
        self.assertIn("flow.small_large_gap", minute)

    def test_ratios_at_every_frequency(self):
        data = preprocess(make_demo_data(), "2026-09-29")
        bars = build_bars(data)
        for name in ("1min", "5min", "daily", "period_summary"):
            frame = bars[name]
            for field, buy, sell in (("ti", frame.buy_amount, frame.sell_amount), ("oi", frame.buy_order_amount, frame.sell_order_amount)):
                expected = (buy-sell)/(buy+sell).replace(0, np.nan)
                pd.testing.assert_series_equal(frame[field], expected, check_names=False)
            self.assertAlmostEqual(frame.amount.sum(), data.trades.amount.sum())
            self.assertEqual(frame.buy_count.sum()+frame.sell_count.sum(), len(data.trades))

    def test_lunch_and_missing_minutes(self):
        frame = pd.DataFrame({"symbol": ["X"]*4, "time": pd.to_datetime(["2026-09-29 11:29", "2026-09-29 13:00", "2026-09-29 13:02", "2026-09-29 13:03"]), "close": [10., 20., 30., 33.]})
        result = minute_returns(frame)
        self.assertTrue(result.iloc[:3].isna().all())
        self.assertAlmostEqual(result.iloc[3], .1)

    def test_absolute_direction(self):
        baseline = pd.DataFrame({"raw": [-.4, 100., 40.], "z20": [2., 0., 3.]}, index=["flow.imbalance", "intraday.amount_open_mean", "intraday.amount_close_mean"])
        state, _ = detect_regime(baseline, {})
        self.assertEqual(state["orderflow_regime"], "sell")
        self.assertEqual(state["intraday_regime"], "opening_dominant")

    def test_group_flow_reweights_amount(self):
        features = pd.DataFrame({"flow.small_buy_amount": [100., 0.], "flow.small_sell_amount": [0., 1.], "flow.imbalance": [1., -1.]}, index=["X", "Y"])
        grouped = compare_universes(features)
        self.assertAlmostEqual(grouped.loc["all", "flow.imbalance"], 99/101)

    def test_quote_impact_decomposition_and_sample_counts(self):
        data = preprocess(make_demo_data(), "2026-09-29")
        frames = compute_features(data, build_bars(data))
        discovery = frames["discovery"]
        np.testing.assert_allclose(discovery.effective_spread_5m_cohort, discovery.realized_spread_5m+discovery.signed_price_impact_5m, atol=1e-12)
        self.assertTrue(discovery.large_samples_30m.gt(0).all())
        self.assertTrue(discovery.quote_match_ratio.eq(1).all())
        self.assertTrue(frames["intraday"].amount_argmax_time.str.match(r"\d{2}:\d{2}").all())

    def test_preopen_context_and_auction_output(self):
        orders = pd.DataFrame([["09:20:00", "X", "b", 10., 100, "buy"], ["09:30:00", "X", "s", 10., 100, "sell"]], columns=["time", "symbol", "order_id", "price", "volume", "side"])
        trades = pd.DataFrame([["09:31:00", "X", 10., 100, "b", "s", "buy"]], columns=["time", "symbol", "price", "volume", "buy_order_id", "sell_order_id", "direction"])
        quotes = pd.DataFrame(columns=["time", "symbol", "bid_price1", "ask_price1"])
        data = preprocess(MarketData(orders, trades, quotes), "2026-09-29")
        life, submitted = lifecycle(data)
        self.assertEqual(len(data.orders), 1)
        self.assertEqual(len(data.auctions["orders"]), 1)
        self.assertEqual(len(submitted), 1)
        self.assertEqual(life.loc["X", "completed_ratio"], 1.)
        self.assertEqual(life.loc["X", "lifetime_seconds"], 360.)
        self.assertEqual(quality_features(data).loc["X", "buy_order_match_ratio"], 1.)

    def test_universe_version_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            ensure_history_identity(directory, ["X"])
            with self.assertRaises(ValueError):
                ensure_history_identity(directory, ["Y"])

    def test_structural_and_group_signals(self):
        history = pd.DataFrame({"structural.ks.trade.median": [.5]*5, "structural.curve.amount": [.6]*5}, index=pd.bdate_range("2026-01-01", periods=5))
        self.assertEqual(detect_migration(history).overall.iloc[-1], "NEW_REGIME")
        rng = np.random.default_rng(8)
        history = pd.DataFrame({"gap.csi1000-hs300.liquidity.spread": rng.normal(0, .01, 30), "flow.imbalance": rng.normal(0, .01, 30)}, index=pd.bdate_range("2026-01-01", periods=30))
        history.iloc[-1] = [1, 1]
        self.assertEqual(detect_migration(history).overall.iloc[-1], "TRANSITION")

    def test_v2_daily_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"axis").mkdir()
            np.save(root/"axis/dates.npy", np.array(["2026-01-05"], dtype="datetime64[D]"))
            np.save(root/"axis/stock_ticks.npy", np.array(["000001", "600000"]))
            path = root/"stock/index/weight/hs300_weight.bin"
            path.parent.mkdir(parents=True)
            np.array([[1., np.nan]], dtype=np.float32).tofile(path)
            daily = load_v2_daily(root, "2026-01-05", ["000001.SZ", "600000.SH"])
            self.assertEqual(daily.hs300.tolist(), [True, False])

    def test_ks_insufficient_sample_is_not_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            day = root/"20260101"
            day.mkdir()
            pd.DataFrame({"trade.median": [1, 2]}, index=["X", "Y"]).to_csv(day/"features.csv")
            current = pd.DataFrame({"trade.median": [10, 20]}, index=["X", "Y"])
            result = compare_previous(current, pd.DataFrame(), root, "2026-01-02")
            self.assertTrue(pd.isna(result.loc["trade.median", "ks_distance"]))


if __name__ == "__main__":
    unittest.main()
