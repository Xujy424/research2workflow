"""Compatibility adapters for legacy analyzer/score-oriented research workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from quant_shared.contracts import FactorResearchReport


# 中文说明：定义 `AcceptanceRule`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class AcceptanceRule:
    metric: str
    minimum: float | None = None
    maximum: float | None = None
    hard_gate: bool = False
    weight: float = 1.0


# 中文说明：定义 `ResearchScorecard`，封装本模块对应的数据、配置与行为。
class ResearchScorecard:
    """Transparent, template-independent factor acceptance scorecard."""

    DEFAULT_RULES = (
        AcceptanceRule("ic_mean", minimum=0.02, hard_gate=True, weight=0.25),
        AcceptanceRule("icir", minimum=0.30, hard_gate=False, weight=0.25),
        AcceptanceRule("ic_positive_ratio", minimum=0.52, hard_gate=False, weight=0.15),
        AcceptanceRule("long_short_sharpe", minimum=0.80, hard_gate=True, weight=0.25),
        AcceptanceRule("coverage", minimum=0.80, hard_gate=False, weight=0.10),
    )

    # 中文说明：`score`：计算评分或监控指标。
    def score(
        self,
        report: FactorResearchReport,
        rules: tuple[AcceptanceRule, ...] | None = None,
    ) -> pd.DataFrame:
        rules = rules or self.DEFAULT_RULES
        rows: list[dict[str, object]] = []
        for factor, metrics in report.summary.iterrows():
            weighted = 0.0
            weight_sum = 0.0
            hard_fail = False
            for rule in rules:
                value = float(metrics.get(rule.metric, np.nan))
                passed = np.isfinite(value)
                if rule.minimum is not None:
                    passed = passed and value >= rule.minimum
                    item_score = np.clip(value / rule.minimum, 0.0, 1.5) / 1.5 * 100
                elif rule.maximum is not None:
                    passed = passed and value <= rule.maximum
                    item_score = np.clip(rule.maximum / max(value, 1e-12), 0.0, 1.5) / 1.5 * 100
                else:
                    item_score = 50.0
                hard_fail |= rule.hard_gate and not passed
                weighted += item_score * rule.weight
                weight_sum += rule.weight
                rows.append(
                    {
                        "factor": factor,
                        "metric": rule.metric,
                        "actual": value,
                        "passed": passed,
                        "hard_gate": rule.hard_gate,
                        "item_score": item_score,
                        "weight": rule.weight,
                    }
                )
            total = 0.0 if hard_fail else weighted / max(weight_sum, 1e-12)
            for row in rows:
                if row["factor"] == factor:
                    row["factor_score"] = total
                    row["conclusion"] = (
                        "Reject"
                        if hard_fail
                        else "Production"
                        if total >= 80
                        else "Shadow"
                        if total >= 65
                        else "Research"
                    )
        return pd.DataFrame(rows)


# 中文说明：定义 `LegacyAnalyzerAdapter`，封装本模块对应的数据、配置与行为。
class LegacyAnalyzerAdapter:
    """Expose common legacy ``FactorAnalyzer`` tables from a new report."""

    # 中文说明：`__init__`：初始化对象及其运行依赖。
    def __init__(
        self,
        report: FactorResearchReport,
        factors: pd.DataFrame,
        forward_returns: pd.Series,
        factor_name: str,
    ) -> None:
        if factor_name not in factors.columns:
            raise KeyError(f"unknown factor: {factor_name}")
        self.report = report
        self.factor_name = factor_name
        self.alpha_df = factors[[factor_name]]
        self.forward_returns = forward_returns
        self.ics_df = report.ic[factor_name]
        self.rankics_df = self.ics_df
        self.groupret_df = report.quantile_returns[factor_name]
        self.cache = {
            "alpha_df": self.alpha_df,
            "ics_df": self.ics_df,
            "rankics_df": self.rankics_df,
            "groupret_df": self.groupret_df,
        }

    # 中文说明：`table_IC_annual_stats`：生成诊断表格。
    def table_IC_annual_stats(self) -> pd.DataFrame:
        series = self.ics_df.dropna()
        rows: list[dict[str, object]] = []
        for year, values in series.groupby(series.index.year):
            rows.append(self._ic_row(values, year))
        rows.append(self._ic_row(series, "Overall"))
        return pd.DataFrame(rows).set_index("year")

    # 中文说明：`table_group_ret_stats`：生成诊断表格。
    def table_group_ret_stats(self) -> pd.DataFrame:
        annualized = self.groupret_df.mean() * 252.0
        volatility = self.groupret_df.std(ddof=1) * np.sqrt(252.0)
        return pd.DataFrame(
            {
                "annual_return": annualized,
                "annual_volatility": volatility,
                "sharpe": annualized / volatility.replace(0.0, np.nan),
            }
        )

    # 中文说明：`table_alpha_annual_stats`：生成诊断表格。
    def table_alpha_annual_stats(self) -> pd.DataFrame:
        signal = self.alpha_df[self.factor_name]
        rows: list[dict[str, object]] = []
        for year, values in signal.groupby(signal.index.get_level_values(0).year):
            rows.append(self._distribution_row(values, year))
        rows.append(self._distribution_row(signal, "Overall"))
        return pd.DataFrame(rows).set_index("year")

    # 中文说明：`_ic_row`：内部辅助步骤，不作为稳定公共接口。
    @staticmethod
    def _ic_row(values: pd.Series, year: object) -> dict[str, object]:
        std = values.std(ddof=1)
        return {
            "year": year,
            "avg_ic": values.mean(),
            "ic_ir": values.mean() / std if std > 0 else np.nan,
            "positive_ratio": (values > 0).mean(),
            "n": len(values),
        }

    # 中文说明：`_distribution_row`：内部辅助步骤，不作为稳定公共接口。
    @staticmethod
    def _distribution_row(values: pd.Series, year: object) -> dict[str, object]:
        clean = values.dropna()
        return {
            "year": year,
            "min": clean.min(),
            "max": clean.max(),
            "mean": clean.mean(),
            "std": clean.std(ddof=1),
            "skew": clean.skew(),
            "kurtosis": clean.kurtosis(),
        }
