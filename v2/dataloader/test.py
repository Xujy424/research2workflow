from pathlib import Path
import datetime as dt
import sys

import polars as pl


# =============================================================================
# 1. 路径及函数导入
# =============================================================================

PROJECT_ROOT = Path(
    "/home/xujiayi/PycharmProjects/infrastructure/research2workflow"
)

DATALOADER_DIR = PROJECT_ROOT / "v2" / "dataloader"

if str(DATALOADER_DIR) not in sys.path:
    sys.path.insert(0, str(DATALOADER_DIR))

from generate_bar_snapshot import (
    prepare_events,
    _prepare_bar_actions,
    generate_bar_snapshots,
)


ROOT = Path("/data/xujiayi/xjy/l2")
DATE = "20260624"
PROC = ROOT / "proc" / DATE

SECURITY = 688797

PROBLEM_BID = 460.41
PROBLEM_ASK = 445.79

EPS = 1e-8


# =============================================================================
# 2. 只读取目标证券，避免全市场占用内存
# =============================================================================

orders = (
    pl.scan_parquet(PROC / "shwt.pq")
    .filter(pl.col("SecurityID") == SECURITY)
    .collect()
)

trades = (
    pl.scan_parquet(PROC / "shcj.pq")
    .filter(pl.col("SecurityID") == SECURITY)
    .collect()
)

cancels = (
    pl.scan_parquet(PROC / "shcancel.pq")
    .filter(pl.col("SecurityID") == SECURITY)
    .collect()
)

print("orders:", orders.shape)
print("trades:", trades.shape)
print("cancels:", cancels.shape)


# =============================================================================
# 3. 生成标准事件表
# =============================================================================

events = prepare_events(
    orders,
    trades,
    cancels,
    sort_events=False,
)

print("\nevents:", events.shape)
print("events columns:", events.columns)


# =============================================================================
# 4. 筛选 460.41 和 445.79 的问题事件
# =============================================================================

problem_events = (
    events
    .filter(
        (pl.col("SecurityID") == SECURITY)
        & (
            (
                (pl.col("Price") - PROBLEM_BID).abs()
                < EPS
            )
            | (
                (pl.col("Price") - PROBLEM_ASK).abs()
                < EPS
            )
        )
        & (pl.col("EventTime") <= dt.time(9, 31))
    )
    .sort([
        "EventTime",
        "SortNo",
        "EventType",
        "ChannelNo",
        "ApplSeqNum",
    ])
)

print("\n" + "=" * 80)
print("problem_events 明细")
print("=" * 80)

print(
    problem_events.select([
        c for c in [
            "EventTime",
            "SortNo",
            "EventType",
            "ChannelNo",
            "SecurityID",
            "Side",
            "ApplSeqNum",
            "Price",
            "QtyDelta",
            "OrdType",
            "OrderStatus",
        ]
        if c in problem_events.columns
    ])
)


# =============================================================================
# 5. 检查 problem_events 的新增、减少与净变化
# =============================================================================

print("\n" + "=" * 80)
print("problem_events 按 EventType 汇总")
print("=" * 80)

problem_event_type_summary = (
    problem_events
    .group_by(
        "Side",
        "Price",
        "EventType",
    )
    .agg(
        pl.col("QtyDelta").sum().alias("QtyDelta"),
        pl.len().alias("Rows"),
    )
    .sort([
        "Side",
        "Price",
        "EventType",
    ])
)

print(problem_event_type_summary)


print("\n" + "=" * 80)
print("problem_events 最终净变化")
print("=" * 80)

problem_event_net_summary = (
    problem_events
    .group_by(
        "Side",
        "Price",
    )
    .agg(
        pl.col("QtyDelta").sum().alias("NetQtyDelta"),
        pl.len().alias("Rows"),
    )
    .sort([
        "Side",
        "Price",
    ])
)

print(problem_event_net_summary)


# =============================================================================
# 6. 构造 bar_actions
# =============================================================================

bars = [
    dt.time(9, 30),
    dt.time(9, 31),
    dt.time(9, 32),
]

bar_actions, dynamic_events = _prepare_bar_actions(
    events,
    bars,
)

print("\nbar_actions:", bar_actions.shape)
print("bar_actions columns:", bar_actions.columns)

print("dynamic_events:", dynamic_events.shape)
print("dynamic_events columns:", dynamic_events.columns)


# =============================================================================
# 7. 筛选问题价格对应的 bar_actions
# =============================================================================

problem_actions = (
    bar_actions
    .filter(
        (pl.col("SecurityID") == SECURITY)
        & (
            (
                (pl.col("Price") - PROBLEM_BID).abs()
                < EPS
            )
            | (
                (pl.col("Price") - PROBLEM_ASK).abs()
                < EPS
            )
        )
    )
    .sort([
        c for c in [
            "EventTime",
            "BarTime",
            "Side",
            "Price",
        ]
        if c in bar_actions.columns
    ])
)

print("\n" + "=" * 80)
print("problem_actions")
print("=" * 80)

print(problem_actions)


# =============================================================================
# 8. 查看 dynamic_events 中是否存在问题事件
# =============================================================================

dynamic_problem_events = (
    dynamic_events
    .filter(
        (pl.col("SecurityID") == SECURITY)
        & (
            (
                (pl.col("Price") - PROBLEM_BID).abs()
                < EPS
            )
            | (
                (pl.col("Price") - PROBLEM_ASK).abs()
                < EPS
            )
        )
    )
)

print("\n" + "=" * 80)
print("dynamic_problem_events")
print("=" * 80)

print(dynamic_problem_events)


# =============================================================================
# 9. 单独重新生成 688797 的三个 bar
# =============================================================================

check_snapshot = generate_bar_snapshots(
    orders=orders,
    trades=trades,
    cancels=cancels,
    bar_times=bars,
    topn=10,
    securities=[SECURITY],
    wide=True,
)

print("\n" + "=" * 80)
print("check_snapshot")
print("=" * 80)

print(
    check_snapshot.select(
        "BarTime",
        "SecurityID",
        "BidPrice1",
        "BidQty1",
        "AskPrice1",
        "AskQty1",
        "BidPrice2",
        "BidQty2",
        "AskPrice2",
        "AskQty2",
    )
)


# =============================================================================
# 10. 自动给出初步分层判断
# =============================================================================

bid_event_net = (
    problem_event_net_summary
    .filter(
        (pl.col("Side") == 1)
        & (
            (pl.col("Price") - PROBLEM_BID).abs()
            < EPS
        )
    )
    .get_column("NetQtyDelta")
    .to_list()
)

bid_action_net = (
    problem_actions
    .filter(
        (pl.col("Side") == 1)
        & (
            (pl.col("Price") - PROBLEM_BID).abs()
            < EPS
        )
    )
    .get_column("QtyDelta")
    .sum()
)

bid_event_net = bid_event_net[0] if bid_event_net else None

print("\n" + "=" * 80)
print("自动判断")
print("=" * 80)

print("460.41 problem_events 净变化：", bid_event_net)
print("460.41 problem_actions 净变化：", bid_action_net)

if bid_event_net not in (None, 0):
    print(
        "判断：prepare_events 层已经不平，"
        "需要检查成交/撤单是否进入事件表。"
    )

elif bid_action_net not in (None, 0):
    print(
        "判断：problem_events 正确，但 bar_actions 不正确，"
        "问题位于 _prepare_bar_actions。"
    )

else:
    generated_0931 = check_snapshot.filter(
        pl.col("BarTime") == dt.time(9, 31)
    )

    if generated_0931.height:
        generated_bid = generated_0931.item(
            0,
            "BidPrice1",
        )

        print("09:31 生成的 BidPrice1：", generated_bid)

        if (
            generated_bid is not None
            and abs(float(generated_bid) - PROBLEM_BID) < EPS
        ):
            print(
                "判断：events 和 bar_actions 净额都正确，"
                "但主回放后仍残留 460.41，"
                "问题位于 levels 更新或事件合并顺序。"
            )
        else:
            print(
                "判断：单股重算已经不再保留 460.41。"
                "此前全量 shshot 或 warning 可能来自旧版本数据，"
                "或者全量状态事件改变了该证券的处理路径。"
            )