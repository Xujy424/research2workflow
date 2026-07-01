"""Append-only audit journal, atomic snapshots, and account reconciliation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
from enum import Enum
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Mapping

import pandas as pd

from .account import ChinaEquityAccount


# 中文说明：`_json_default`：内部辅助步骤，不作为稳定公共接口。
def _json_default(value: object) -> object:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


# 中文说明：定义 `TradingJournal`，封装本模块对应的数据、配置与行为。
class TradingJournal:
    # 中文说明：`__init__`：初始化对象及其运行依赖。
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sequence = self._last_sequence()

    # 中文说明：`append`：执行该名称对应的业务计算，并返回调用方所需结果。
    def append(self, event_type: str, payload: object, timestamp: datetime) -> int:
        self.sequence += 1
        record = {
            "sequence": self.sequence,
            "timestamp": timestamp.isoformat(),
            "event_type": event_type,
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=_json_default))
            stream.write("\n")
            stream.flush()
        return self.sequence

    # 中文说明：`read`：读取持久化数据。
    def read(self) -> list[dict[str, object]]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]

    # 中文说明：`_last_sequence`：内部辅助步骤，不作为稳定公共接口。
    def _last_sequence(self) -> int:
        if not self.path.exists():
            return 0
        last = 0
        with self.path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    last = int(json.loads(line)["sequence"])
        return last


# 中文说明：定义 `AtomicStateStore`，封装本模块对应的数据、配置与行为。
class AtomicStateStore:
    # 中文说明：`__init__`：初始化对象及其运行依赖。
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # 中文说明：`save_account`：持久化当前状态。
    def save_account(self, account: ChinaEquityAccount) -> None:
        state = {
            "account_id": account.account_id,
            "cash": account.cash,
            "frozen_cash": account.frozen_cash,
            "realized_pnl": account.realized_pnl,
            "total_commission": account.total_commission,
            "total_tax": account.total_tax,
            "current_date": account.current_date,
            "positions": {
                symbol: {
                    "total_quantity": position.total_quantity,
                    "sellable_quantity": position.sellable_quantity,
                    "average_cost": position.average_cost,
                    "last_price": position.last_price,
                    "today_buys": position.today_buys,
                    "lots": position.lots,
                }
                for symbol, position in account.positions.items()
            },
        }
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self.path.parent,
            delete=False,
            suffix=".tmp",
        ) as stream:
            json.dump(state, stream, ensure_ascii=False, default=_json_default)
            temporary = Path(stream.name)
        temporary.replace(self.path)

    # 中文说明：`load`：读取并规范化外部数据。
    def load(self) -> dict[str, object]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))


# 中文说明：定义 `ReconciliationReport`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class ReconciliationReport:
    cash_difference: float
    position_differences: pd.DataFrame
    passed: bool


# 中文说明：定义 `AccountReconciler`，封装本模块对应的数据、配置与行为。
class AccountReconciler:
    # 中文说明：`reconcile`：执行账户或状态对账。
    def reconcile(
        self,
        account: ChinaEquityAccount,
        external_cash: float,
        external_positions: pd.Series,
        cash_tolerance: float = 0.01,
        quantity_tolerance: int = 0,
    ) -> ReconciliationReport:
        internal = pd.Series(
            {
                symbol: position.total_quantity
                for symbol, position in account.positions.items()
            },
            dtype=float,
        )
        assets = internal.index.union(external_positions.index)
        frame = pd.DataFrame(
            {
                "internal": internal.reindex(assets).fillna(0.0),
                "external": external_positions.reindex(assets).fillna(0.0),
            }
        )
        frame["difference"] = frame["internal"] - frame["external"]
        cash_difference = account.cash - external_cash
        passed = (
            abs(cash_difference) <= cash_tolerance
            and frame["difference"].abs().max() <= quantity_tolerance
        )
        return ReconciliationReport(cash_difference, frame, bool(passed))
