from __future__ import annotations

from datetime import date

import pandas as pd

from quant_workflow.trading import DailyL2Bundle, L2OrderEvent, L2TradeEvent
from quant_workflow.trading import tonglian_l2_gateway


def test_tonglian_gateway_maps_four_tick_tables(tmp_path):
    order = pd.DataFrame(
        {
            "SecurityID": ["600000", "000001"],
            "TickTime": ["09:30:00.001", "09:30:00.002"],
            "BizIndex": [1, 1],
            "Side": ["B", "S"],
            "Price": [10.1, 20.2],
            "Volume": [1000, 2000],
            "MDStreamID": ["A", "A"],
            "ChannelNo": [1, 2],
        }
    )
    trade = pd.DataFrame(
        {
            "SecurityID": ["600000", "000001"],
            "TradeTime": ["09:30:00.003", "09:30:00.004"],
            "TradeIndex": [2, 2],
            "TradePrice": [10.2, 20.1],
            "TradeQty": [500, 800],
            "Side": ["B", "S"],
            "BuyOrderNo": ["11", "21"],
            "SellOrderNo": ["12", "22"],
            "TradeChannel": [1, 2],
        }
    )
    paths = {}
    for name, frame in {
        "sse_orders": order.iloc[[0]],
        "sse_trades": trade.iloc[[0]],
        "szse_orders": order.iloc[[1]],
        "szse_trades": trade.iloc[[1]],
    }.items():
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path

    bundle = DailyL2Bundle(date(2026, 1, 5), **paths)
    events = list(tonglian_l2_gateway(use_polars=False).stream(bundle))

    assert len(events) == 4
    orders = [event for event in events if isinstance(event, L2OrderEvent)]
    trades = [event for event in events if isinstance(event, L2TradeEvent)]
    assert [event.symbol for event in orders] == ["600000", "000001"]
    assert orders[0].side.value == "BUY"
    assert orders[0].quantity == 1000
    assert trades[0].buy_order_id == "11"
    assert trades[1].sell_order_id == "22"