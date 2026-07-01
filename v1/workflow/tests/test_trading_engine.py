"""Tests four-table replay, queue matching, T+1, persistence, and paper trading."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd

from quant_workflow.trading import (
    ChinaEquityAccount,
    AccountReconciler,
    AtomicStateStore,
    CanonicalL2Gateway,
    CanonicalL2Preprocessor,
    DailyL2Bundle,
    Exchange,
    HistoricalReplayEngine,
    L2ColumnMap,
    L2DataQualityValidator,
    L2TableGateway,
    OrderRequest,
    OrderType,
    PaperTradingEngine,
    PreTradeRiskConfig,
    PreTradeRiskEngine,
    Side,
    TargetWeightExecutionStrategy,
    TradingJournal,
    PortfolioTradingBridge,
)
from quant_shared.contracts import OptimizationResult
from quant_workflow.trading.events import L2OrderEvent, L2TradeEvent


# 中文说明：`TradingEngineTest` 验证该场景的预期行为。
class TradingEngineTest(unittest.TestCase):
    # 中文说明：`setUp` 验证该场景的预期行为。
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.trading_date = date(2025, 1, 2)
        self.bundle = self._write_bundle()
        mapping = L2ColumnMap(
            symbol="symbol",
            timestamp="timestamp",
            sequence="sequence",
            order_id="order_id",
            trade_id="trade_id",
            side="side",
            price="price",
            quantity="quantity",
            action="action",
            buy_order_id="buy_order_id",
            sell_order_id="sell_order_id",
        )
        self.gateway = L2TableGateway(
            order_columns={Exchange.SSE: mapping, Exchange.SZSE: mapping},
            trade_columns={Exchange.SSE: mapping, Exchange.SZSE: mapping},
            chunksize=2,
        )

    # 中文说明：`tearDown` 验证该场景的预期行为。
    def tearDown(self) -> None:
        self.temp.cleanup()

    # 中文说明：`test_four_table_merge_quality_and_backtest` 验证该场景的预期行为。
    def test_four_table_merge_quality_and_backtest(self) -> None:
        events = list(self.gateway.stream(self.bundle))
        self.assertEqual(len(events), 8)
        self.assertEqual(events, sorted(events, key=self.gateway._sort_key))
        quality = L2DataQualityValidator().validate(events)
        self.assertTrue(quality.passed)

        account = ChinaEquityAccount(100_000.0)
        risk = PreTradeRiskEngine(
            PreTradeRiskConfig(
                max_symbol_weight=0.50,
                max_order_notional=100_000.0,
                max_daily_turnover=2.0,
            )
        )
        engine = HistoricalReplayEngine(account, risk, self.gateway)
        strategy = TargetWeightExecutionStrategy(
            pd.Series({"600000": 0.10, "000001": 0.10}),
            start_time=time(9, 30),
            order_type=OrderType.LIMIT,
        )
        engine.add_strategy(strategy)
        result = engine.run_bundle(self.bundle)

        self.assertGreaterEqual(len(result.trades), 2)
        self.assertEqual(set(result.positions["symbol"]), {"600000", "000001"})
        self.assertGreater(result.statistics["commission"], 0.0)
        self.assertAlmostEqual(
            float(result.positions["quantity"].sum()), 2000.0, delta=200.0
        )

        journal = TradingJournal(self.root / "audit.jsonl")
        state = AtomicStateStore(self.root / "state.json")
        persisted = HistoricalReplayEngine(
            ChinaEquityAccount(100_000.0),
            PreTradeRiskEngine(
                PreTradeRiskConfig(
                    max_symbol_weight=0.50,
                    max_daily_turnover=2.0,
                )
            ),
            self.gateway,
            journal=journal,
            state_store=state,
        )
        persisted.add_strategy(
            TargetWeightExecutionStrategy(
                pd.Series({"600000": 0.10}), start_time=time(9, 30)
            )
        )
        persisted.run_bundle(self.bundle)
        self.assertGreater(len(journal.read()), 0)
        self.assertIn("cash", state.load())
        reconciliation = AccountReconciler().reconcile(
            persisted.account,
            persisted.account.cash,
            pd.Series(
                {
                    symbol: position.total_quantity
                    for symbol, position in persisted.account.positions.items()
                }
            ),
        )
        self.assertTrue(reconciliation.passed)

    # 中文说明：`test_t_plus_one_and_paper_engine` 验证该场景的预期行为。
    def test_t_plus_one_and_paper_engine(self) -> None:
        account = ChinaEquityAccount(100_000.0)
        account.start_session(self.trading_date)
        commission, _ = account.apply_fill(
            Exchange.SSE, Side.BUY, "600000", 10.0, 1000, self.trading_date
        )
        self.assertGreater(commission, 0)
        self.assertEqual(account.get_position("600000").sellable_quantity, 0)

        risk = PreTradeRiskEngine(
            PreTradeRiskConfig(max_symbol_weight=1.0, max_daily_turnover=5.0)
        )
        engine = HistoricalReplayEngine(account, risk)
        engine.current_time = datetime(2025, 1, 2, 10, 0)
        engine.oms.matcher.get_book("600000").bid_levels[10.0] = 10_000
        order_id = engine.oms.send_order(
            OrderRequest(
                "600000",
                Exchange.SSE,
                Side.SELL,
                1000,
                OrderType.LIMIT,
                10.0,
            ),
            engine.current_time,
        )
        self.assertEqual(engine.oms.orders[order_id].status.value, "REJECTED")

        account.start_session(date(2025, 1, 3))
        self.assertEqual(account.get_position("600000").sellable_quantity, 1000)

        paper = PaperTradingEngine(
            ChinaEquityAccount(100_000.0),
            PreTradeRiskEngine(
                PreTradeRiskConfig(
                    max_symbol_weight=0.5,
                    max_daily_turnover=2.0,
                )
            ),
            self.gateway,
            speed=1_000_000.0,
            max_sleep=0.0,
        )
        paper.add_strategy(
            TargetWeightExecutionStrategy(
                pd.Series({"600000": 0.10}),
                start_time=time(9, 30),
            )
        )
        result = paper.run_bundle(self.bundle)
        self.assertGreater(len(result.orders), 0)

    # 中文说明：`test_pretrade_gross_exposure_and_cash_only_short_guard` 验证敞口与卖空边界。
    def test_pretrade_gross_exposure_and_cash_only_short_guard(self) -> None:
        account = ChinaEquityAccount(100_000.0)
        account.start_session(self.trading_date)
        account.seed_position("600000", 5_000, 10.0, 10.0)
        risk = PreTradeRiskEngine(
            PreTradeRiskConfig(
                max_symbol_weight=1.0,
                max_gross_exposure=0.40,
                max_daily_turnover=5.0,
            )
        )
        book = HistoricalReplayEngine(account, risk).oms.matcher.get_book("000001")
        book.ask_levels[10.0] = 10_000
        passed, reason = risk.check(
            OrderRequest(
                "000001",
                Exchange.SZSE,
                Side.BUY,
                2_000,
                OrderType.LIMIT,
                10.0,
            ),
            account,
            book,
            0,
            datetime(2025, 1, 2, 10, 0),
        )
        self.assertFalse(passed)
        self.assertIn("gross-exposure", reason)

        short_result = OptimizationResult(
            weights=pd.Series({"600000": -0.1, "000001": 0.1}),
            trades=pd.Series({"600000": -0.1, "000001": 0.1}),
            status="optimal",
            expected_return=0.0,
            predicted_volatility=0.0,
            turnover=0.2,
            expected_cost=0.0,
            exposures=pd.Series(dtype=float),
            constraint_usage={},
        )
        with self.assertRaisesRegex(ValueError, "securities-lending"):
            PortfolioTradingBridge().backtest(short_result, [])

    # 中文说明：`test_passive_queue_fill` 验证该场景的预期行为。
    def test_passive_queue_fill(self) -> None:
        account = ChinaEquityAccount(100_000.0)
        account.start_session(self.trading_date)
        risk = PreTradeRiskEngine(
            PreTradeRiskConfig(max_symbol_weight=1.0, max_daily_turnover=5.0)
        )
        engine = HistoricalReplayEngine(account, risk)
        add = L2OrderEvent(
            Exchange.SSE,
            "600000",
            datetime(2025, 1, 2, 9, 30),
            1,
            "B1",
            Side.BUY,
            9.99,
            500,
        )
        engine.current_time = add.timestamp
        engine.oms.on_market_event(add)
        order_id = engine.oms.send_order(
            OrderRequest(
                "600000",
                Exchange.SSE,
                Side.BUY,
                100,
                OrderType.LIMIT,
                9.99,
            ),
            add.timestamp,
        )
        self.assertEqual(engine.oms.orders[order_id].queue_ahead, 500)
        trade = L2TradeEvent(
            Exchange.SSE,
            "600000",
            datetime(2025, 1, 2, 9, 31),
            2,
            "T1",
            9.99,
            600,
            Side.SELL,
            buy_order_id="B1",
        )
        engine.current_time = trade.timestamp
        engine.oms.on_market_event(trade)
        self.assertEqual(engine.oms.orders[order_id].filled_quantity, 100)

    # 中文说明：`test_latency_rechecks_take_liquidity_at_exchange_arrival` 验证该场景的预期行为。
    def test_latency_rechecks_take_liquidity_at_exchange_arrival(self) -> None:
        account = ChinaEquityAccount(100_000.0)
        account.start_session(self.trading_date)
        engine = HistoricalReplayEngine(
            account,
            PreTradeRiskEngine(
                PreTradeRiskConfig(max_symbol_weight=1.0, max_daily_turnover=5.0)
            ),
            order_latency=timedelta(seconds=1),
        )
        start = datetime(2025, 1, 2, 9, 30)
        ask = L2OrderEvent(
            Exchange.SSE, "600000", start, 1, "S1", Side.SELL, 10.0, 100
        )
        engine.current_time = start
        engine.oms.on_market_event(ask)
        order_id = engine.oms.send_order(
            OrderRequest(
                "600000",
                Exchange.SSE,
                Side.BUY,
                100,
                OrderType.LIMIT,
                10.0,
            ),
            start,
        )
        self.assertEqual(engine.oms.orders[order_id].status.value, "SUBMITTING")

        cancel_time = start + timedelta(milliseconds=500)
        cancel = L2OrderEvent(
            Exchange.SSE,
            "600000",
            cancel_time,
            2,
            "S1",
            Side.SELL,
            10.0,
            100,
            action="CANCEL",
        )
        engine.oms.advance_time(cancel_time)
        engine.oms.on_market_event(cancel)
        arrival = start + timedelta(seconds=1)
        engine.oms.advance_time(arrival)

        order = engine.oms.orders[order_id]
        self.assertEqual(order.arrived_at, arrival)
        self.assertEqual(order.filled_quantity, 0)
        self.assertEqual(order.status.value, "ACTIVE")

    # 中文说明：`test_consecutive_take_orders_cannot_reuse_consumed_depth` 验证该场景的预期行为。
    def test_consecutive_take_orders_cannot_reuse_consumed_depth(self) -> None:
        account = ChinaEquityAccount(100_000.0)
        account.start_session(self.trading_date)
        engine = HistoricalReplayEngine(
            account,
            PreTradeRiskEngine(
                PreTradeRiskConfig(max_symbol_weight=1.0, max_daily_turnover=5.0)
            ),
        )
        timestamp = datetime(2025, 1, 2, 9, 30)
        engine.current_time = timestamp
        engine.oms.on_market_event(
            L2OrderEvent(
                Exchange.SSE,
                "600000",
                timestamp,
                1,
                "S1",
                Side.SELL,
                10.0,
                100,
            )
        )
        first = engine.oms.send_order(
            OrderRequest(
                "600000", Exchange.SSE, Side.BUY, 100, OrderType.LIMIT, 10.0
            ),
            timestamp,
        )
        second = engine.oms.send_order(
            OrderRequest(
                "600000", Exchange.SSE, Side.BUY, 100, OrderType.LIMIT, 10.0
            ),
            timestamp,
        )
        self.assertEqual(engine.oms.orders[first].filled_quantity, 100)
        self.assertEqual(engine.oms.orders[second].filled_quantity, 0)
        self.assertIsNone(engine.oms.matcher.get_book("600000").best_ask)

    # 中文说明：`test_latency_make_joins_arrival_queue_then_future_trade_fills` 验证该场景的预期行为。
    def test_latency_make_joins_arrival_queue_then_future_trade_fills(self) -> None:
        account = ChinaEquityAccount(100_000.0)
        account.start_session(self.trading_date)
        engine = HistoricalReplayEngine(
            account,
            PreTradeRiskEngine(
                PreTradeRiskConfig(max_symbol_weight=1.0, max_daily_turnover=5.0)
            ),
            order_latency=timedelta(seconds=1),
        )
        start = datetime(2025, 1, 2, 9, 30)
        first_bid = L2OrderEvent(
            Exchange.SSE, "600000", start, 1, "B1", Side.BUY, 9.99, 500
        )
        engine.current_time = start
        engine.oms.on_market_event(first_bid)
        order_id = engine.oms.send_order(
            OrderRequest(
                "600000",
                Exchange.SSE,
                Side.BUY,
                100,
                OrderType.LIMIT,
                9.99,
            ),
            start,
        )

        later_bid = L2OrderEvent(
            Exchange.SSE,
            "600000",
            start + timedelta(milliseconds=500),
            2,
            "B2",
            Side.BUY,
            9.99,
            200,
        )
        engine.oms.advance_time(later_bid.timestamp)
        engine.oms.on_market_event(later_bid)
        trade = L2TradeEvent(
            Exchange.SSE,
            "600000",
            start + timedelta(seconds=1),
            3,
            "T1",
            9.99,
            800,
            Side.SELL,
            buy_order_id="B1",
        )
        engine.oms.advance_time(trade.timestamp)
        self.assertEqual(engine.oms.orders[order_id].queue_ahead, 700)
        engine.oms.on_market_event(trade)
        self.assertEqual(engine.oms.orders[order_id].filled_quantity, 100)

    # 中文说明：`test_four_tables_preprocess_to_two_reusable_exchange_streams` 验证该场景的预期行为。
    def test_four_tables_preprocess_to_two_reusable_exchange_streams(self) -> None:
        processor = CanonicalL2Preprocessor(self.gateway, batch_rows=2)
        canonical, report = processor.preprocess(
            self.bundle,
            self.root / "canonical",
        )
        self.assertEqual(report.sse_rows, 4)
        self.assertEqual(report.szse_rows, 4)
        self.assertGreater(report.rows_per_second, 0.0)
        self.assertTrue(canonical.manifest.exists())

        raw_events = list(self.gateway.stream(self.bundle))
        canonical_events = list(CanonicalL2Gateway(batch_rows=2).stream(canonical))
        raw_keys = [
            (type(event), event.exchange, event.timestamp, event.sequence)
            for event in raw_events
        ]
        canonical_keys = [
            (type(event), event.exchange, event.timestamp, event.sequence)
            for event in canonical_events
        ]
        self.assertEqual(canonical_keys, raw_keys)
        self.assertEqual(
            [(event.price, event.quantity) for event in canonical_events],
            [(event.price, event.quantity) for event in raw_events],
        )

    # 中文说明：`_write_bundle` 验证该场景的预期行为。
    def _write_bundle(self) -> DailyL2Bundle:
        sse_orders = pd.DataFrame(
            [
                ["600000", 93000000, 1, "S1", "S", 10.00, 2000, "ADD"],
                ["600000", 93000010, 2, "B1", "B", 9.99, 2000, "ADD"],
            ],
            columns=[
                "symbol",
                "timestamp",
                "sequence",
                "order_id",
                "side",
                "price",
                "quantity",
                "action",
            ],
        )
        sse_trades = pd.DataFrame(
            [
                ["600000", 93001000, 3, "T1", "B", 10.00, 100, "B2", "S1"],
                ["600000", 93100000, 4, "T2", "B", 10.00, 1000, "B3", "S1"],
            ],
            columns=[
                "symbol",
                "timestamp",
                "sequence",
                "trade_id",
                "side",
                "price",
                "quantity",
                "buy_order_id",
                "sell_order_id",
            ],
        )
        szse_orders = pd.DataFrame(
            [
                ["000001", 93000005, 1, "S2", "S", 10.00, 2000, "ADD"],
                ["000001", 93000015, 2, "B2", "B", 9.99, 2000, "ADD"],
            ],
            columns=sse_orders.columns,
        )
        szse_trades = pd.DataFrame(
            [
                ["000001", 93001005, 3, "T3", "B", 10.00, 100, "B4", "S2"],
                ["000001", 93100005, 4, "T4", "B", 10.00, 1000, "B5", "S2"],
            ],
            columns=sse_trades.columns,
        )
        paths = {}
        for name, frame in {
            "sse_orders": sse_orders,
            "sse_trades": sse_trades,
            "szse_orders": szse_orders,
            "szse_trades": szse_trades,
        }.items():
            path = self.root / f"{name}.csv"
            frame.to_csv(path, index=False)
            paths[name] = path
        return DailyL2Bundle(self.trading_date, **paths)


if __name__ == "__main__":
    unittest.main()
