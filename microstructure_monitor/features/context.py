"""One per-run event calculation shared by daily features and minute series."""
from dataclasses import dataclass
import pandas as pd
from .price_impact import event_returns
from .depth_recovery import depth_recovery
from ..preprocess.match_order_trade import lifecycle


@dataclass
class EventContext:
    lifecycle: pd.DataFrame
    submissions: pd.DataFrame
    orders: pd.DataFrame
    discovery: pd.DataFrame
    trade_events: pd.DataFrame
    recovery: pd.DataFrame
    recovery_events: pd.DataFrame


def compute_event_context(data):
    life, submissions, orders = lifecycle(data, with_orders=True)
    discovery, trade_events = event_returns(data, with_events=True)
    recovery, recovery_events = depth_recovery(data.quotes)
    return EventContext(life, submissions, orders, discovery, trade_events, recovery, recovery_events)
