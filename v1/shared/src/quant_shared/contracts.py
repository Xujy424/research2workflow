"""Validated data contracts shared by research, risk, and portfolio layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd


# 中文说明：`_require_panel_index`：内部辅助步骤，不作为稳定公共接口。
def _require_panel_index(index: pd.Index, name: str) -> None:
    if not isinstance(index, pd.MultiIndex) or index.nlevels != 2:
        raise ValueError(f"{name} must use a two-level MultiIndex: (date, asset)")
    dates = pd.to_datetime(index.get_level_values(0), errors="coerce")
    if dates.isna().any():
        raise ValueError(f"{name} contains invalid dates")
    if index.has_duplicates:
        raise ValueError(f"{name} contains duplicate (date, asset) rows")


# 中文说明：定义 `PanelData`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class PanelData:
    """Point-in-time research panel.

    ``factors`` is indexed by ``(date, asset)`` and stores one factor per column.
    ``forward_returns`` must already be aligned to the signal date. The workflow
    never shifts labels implicitly, which makes the information timing explicit.
    """

    factors: pd.DataFrame
    forward_returns: pd.Series
    exposures: pd.DataFrame | None = None
    market_caps: pd.Series | None = None
    tradable: pd.Series | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    # 中文说明：`validate`：校验输入数据和业务约束。
    def validate(self) -> "PanelData":
        _require_panel_index(self.factors.index, "factors")
        _require_panel_index(self.forward_returns.index, "forward_returns")
        if not self.factors.index.equals(self.forward_returns.index):
            raise ValueError("factors and forward_returns must have identical indices")
        if self.factors.columns.has_duplicates:
            raise ValueError("factor names must be unique")
        for name, value in (
            ("exposures", self.exposures),
            ("market_caps", self.market_caps),
            ("tradable", self.tradable),
        ):
            if value is not None and not value.index.equals(self.factors.index):
                raise ValueError(f"{name} index must match factors")
        return self

    # 中文说明：`dates`：执行该名称对应的业务计算，并返回调用方所需结果。
    @property
    def dates(self) -> pd.DatetimeIndex:
        values = pd.to_datetime(self.factors.index.get_level_values(0).unique())
        return pd.DatetimeIndex(values).sort_values()

    # 中文说明：`assets`：执行该名称对应的业务计算，并返回调用方所需结果。
    @property
    def assets(self) -> pd.Index:
        return self.factors.index.get_level_values(1).unique()


# 中文说明：定义 `FactorResearchReport`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class FactorResearchReport:
    summary: pd.DataFrame
    ic: pd.DataFrame
    quantile_returns: Mapping[str, pd.DataFrame]
    factor_correlations: pd.DataFrame
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


# 中文说明：定义 `RiskModelOutput`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class RiskModelOutput:
    assets: pd.Index
    exposures: pd.DataFrame
    factor_covariance: pd.DataFrame
    specific_variance: pd.Series
    stock_covariance: pd.DataFrame
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    # 中文说明：`validate`：校验输入数据和业务约束。
    def validate(self) -> "RiskModelOutput":
        sigma = self.stock_covariance.loc[self.assets, self.assets].to_numpy(float)
        if not np.allclose(sigma, sigma.T, atol=1e-10):
            raise ValueError("stock covariance is not symmetric")
        if np.linalg.eigvalsh(sigma).min() < -1e-8:
            raise ValueError("stock covariance is not positive semidefinite")
        return self


# 中文说明：定义 `OptimizationResult`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class OptimizationResult:
    weights: pd.Series
    trades: pd.Series
    status: str
    expected_return: float
    predicted_volatility: float
    turnover: float
    expected_cost: float
    exposures: pd.Series
    constraint_usage: Mapping[str, float]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    benchmark_weights: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    active_weights: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
