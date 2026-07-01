"""Corrected daily DAG distilled from the Airflow and production diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from graphlib import TopologicalSorter
from typing import Mapping


# 中文说明：定义 `TaskNode`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class TaskNode:
    name: str
    dependencies: tuple[str, ...] = ()
    layer: str = "production"
    external: bool = False


# 中文说明：定义 `DailyProductionGraph`，封装本模块对应的数据、配置与行为。
class DailyProductionGraph:
    """Scheduler-neutral dependency graph with corrected arrow directions."""

    # 中文说明：`__init__`：初始化对象及其运行依赖。
    def __init__(self, nodes: Mapping[str, TaskNode] | None = None) -> None:
        self.nodes = dict(nodes or self.default_nodes())
        self.validate()

    # 中文说明：`default_nodes`：执行该名称对应的业务计算，并返回调用方所需结果。
    @staticmethod
    def default_nodes() -> dict[str, TaskNode]:
        nodes = [
            TaskNode("is_trade_date", external=True),
            TaskNode("market_data_update", ("is_trade_date",), external=True),
            TaskNode("fundamental_data_update", ("is_trade_date",), external=True),
            TaskNode("index_data_update", ("is_trade_date",), external=True),
            TaskNode("l2_data_update", ("is_trade_date",), external=True),
            TaskNode(
                "data_ready",
                (
                    "market_data_update",
                    "fundamental_data_update",
                    "index_data_update",
                    "l2_data_update",
                ),
            ),
            TaskNode("risk_bank_update", ("data_ready",)),
            TaskNode("approved_factor_update", ("data_ready",)),
            TaskNode(
                "alpha_ready",
                ("risk_bank_update", "approved_factor_update"),
            ),
            TaskNode("factor_combination_update", ("alpha_ready",)),
            TaskNode("comb_bank", ("factor_combination_update",)),
            TaskNode("factor_comb_bank", ("factor_combination_update",)),
            TaskNode("model_snapshot", ("factor_combination_update",)),
            TaskNode(
                "combination_ready",
                ("comb_bank", "factor_comb_bank", "model_snapshot"),
            ),
            TaskNode("portfolio_long_only", ("combination_ready",)),
            TaskNode("portfolio_index_enhanced", ("combination_ready",)),
            TaskNode("portfolio_market_neutral", ("combination_ready",)),
            TaskNode(
                "portfolio_ready",
                (
                    "portfolio_long_only",
                    "portfolio_index_enhanced",
                    "portfolio_market_neutral",
                ),
            ),
            TaskNode("publish_target_weights", ("portfolio_ready",)),
            TaskNode("canonical_l2_preprocess", ("l2_data_update",)),
            TaskNode("execution_or_replay", ("publish_target_weights", "canonical_l2_preprocess")),
            TaskNode("account_reconciliation", ("execution_or_replay",)),
            TaskNode("monitoring_feedback", ("account_reconciliation",)),
        ]
        return {node.name: node for node in nodes}

    # 中文说明：`validate`：校验输入数据和业务约束。
    def validate(self) -> "DailyProductionGraph":
        missing = {
            dependency
            for node in self.nodes.values()
            for dependency in node.dependencies
            if dependency not in self.nodes
        }
        if missing:
            raise ValueError(f"unknown task dependencies: {sorted(missing)}")
        TopologicalSorter(
            {
                name: set(node.dependencies)
                for name, node in self.nodes.items()
            }
        ).prepare()
        return self

    # 中文说明：`topological_order`：执行该名称对应的业务计算，并返回调用方所需结果。
    def topological_order(self) -> tuple[str, ...]:
        return tuple(
            TopologicalSorter(
                {
                    name: set(node.dependencies)
                    for name, node in self.nodes.items()
                }
            ).static_order()
        )

    # 中文说明：`upstream`：执行该名称对应的业务计算，并返回调用方所需结果。
    def upstream(self, task: str) -> set[str]:
        if task not in self.nodes:
            raise KeyError(task)
        result: set[str] = set()
        pending = list(self.nodes[task].dependencies)
        while pending:
            name = pending.pop()
            if name in result:
                continue
            result.add(name)
            pending.extend(self.nodes[name].dependencies)
        return result
