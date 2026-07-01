"""Factor-sleeve construction and allocation at the strategy-return layer."""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_shared.config import SleeveConfig
from quant_shared.math_utils import nearest_psd, normalize_weights



@dataclass(frozen=True)
class SleeveAllocationResult:
    weights: pd.DataFrame
    combined_returns: pd.Series
    diagnostics: dict[str, object]


class FactorSleeveBuilder:
    """Convert each factor signal into a self-financing simulated portfolio."""
    def build_weights(
        self,
        factors: pd.DataFrame,
        quantile: float = 0.20,
        demean: bool = True,
    ) -> dict[str, pd.DataFrame]:
        sleeves: dict[str, pd.DataFrame] = {}
        for factor in factors.columns:
            rows: list[pd.Series] = []
            for date, signal_group in factors[factor].groupby(level=0, sort=True):
                signal = signal_group.droplevel(0).dropna()
                weight = pd.Series(0.0, index=signal_group.droplevel(0).index)
                if len(signal) >= 10:
                    lower = signal.quantile(quantile)
                    upper = signal.quantile(1.0 - quantile)
                    long = signal.index[signal >= upper]
                    short = signal.index[signal <= lower]
                    if len(long):
                        weight.loc[long] = 0.5 / len(long)
                    if len(short):
                        weight.loc[short] = -0.5 / len(short)
                    if not demean:
                        weight = weight.clip(lower=0.0)
                        if weight.sum():
                            weight /= weight.sum()
                weight.name = date
                rows.append(weight)
            sleeves[factor] = pd.DataFrame(rows).fillna(0.0)
        return sleeves

    # 中文说明：`returns_from_weights`：执行该名称对应的业务计算，并返回调用方所需结果。
    @staticmethod
    def returns_from_weights(
        sleeve_weights: dict[str, pd.DataFrame],
        asset_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        result = {}
        for name, weights in sleeve_weights.items():
            aligned_weights = weights.reindex(
                index=asset_returns.index, columns=asset_returns.columns
            ).fillna(0.0)
            result[name] = (aligned_weights * asset_returns).sum(axis=1)
        return pd.DataFrame(result)


# 中文说明：定义 `SleeveAllocator`，封装本模块对应的数据、配置与行为。
class SleeveAllocator:
    """
    各因子组合历史收益
            ↓
    滚动截取过去 lookback 天
            ↓
    估计因子组合均值和协方差
            ↓
    使用 equal / ICIR / 最小方差 / 均值方差 / 风险平价等方法
            ↓
    得到各因子组合目标权重
            ↓
    与上一期权重平滑
            ↓
    限制单个因子最大权重
            ↓
    得到每日动态因子权重和组合收益
    """

    # 中文说明：`__init__`：初始化对象及其运行依赖。
    def __init__(self, config: SleeveConfig | None = None) -> None:
        self.config = config or SleeveConfig()

    # 中文说明：`allocate`：执行该名称对应的业务计算，并返回调用方所需结果。
    def allocate(self, sleeve_returns: pd.DataFrame) -> SleeveAllocationResult:
        rows: list[pd.Series] = []
        dates = sleeve_returns.index
        previous = np.full(sleeve_returns.shape[1], 1.0 / sleeve_returns.shape[1])
        solver_status: dict[pd.Timestamp, str] = {}
        for position, date in enumerate(dates):
            history = sleeve_returns.iloc[max(0, position - self.config.lookback) : position]
            history = history.dropna(how="any")
            if len(history) < self.config.min_periods:
                weight = previous.copy()
                status = "warmup"
            else:
                weight, status = self._allocate_window(history, previous)
            weight = (
                self.config.weight_smoothing * previous
                + (1.0 - self.config.weight_smoothing) * weight
            )
            weight = normalize_weights(weight, False, self.config.max_weight)
            previous = weight
            rows.append(pd.Series(weight, index=sleeve_returns.columns, name=date))
            solver_status[pd.Timestamp(date)] = status
        weights = pd.DataFrame(rows)
        combined = (weights * sleeve_returns).sum(axis=1).rename("combined_sleeve_return")
        return SleeveAllocationResult(
            weights,
            combined,
            {"method": self.config.method, "status": solver_status},
        )

    # 中文说明：`_allocate_window`：内部辅助步骤，不作为稳定公共接口。
    def _allocate_window(
        self,
        returns: pd.DataFrame,
        previous: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        covariance = returns.cov().to_numpy(float)
        diagonal = np.diag(np.diag(covariance))
        covariance = (
            (1.0 - self.config.covariance_shrinkage) * covariance
            + self.config.covariance_shrinkage * diagonal
        )
        covariance = nearest_psd(covariance, 1e-10)
        mean = returns.mean().to_numpy(float) * (1.0 - self.config.return_shrinkage)
        method = self.config.method.lower()
        if method == "equal":
            return np.full(len(mean), 1.0 / len(mean)), "analytic"
        if method == "icir":
            volatility = returns.std(ddof=1).replace(0.0, np.nan).to_numpy(float)
            score = np.divide(mean, volatility, out=np.zeros_like(mean), where=np.isfinite(volatility))
            return normalize_weights(np.maximum(score, 0.0), False, self.config.max_weight), "analytic"
        if method == "correlation_adjusted":
            inverse = np.linalg.pinv(covariance)
            score = inverse @ np.maximum(mean, 0.0)
            return normalize_weights(np.maximum(score, 0.0), False, self.config.max_weight), "analytic"
        if method == "minimum_variance":
            return self._convex_allocate(np.zeros_like(mean), covariance, previous)
        if method == "mean_variance":
            return self._convex_allocate(mean, covariance, previous)
        if method == "risk_parity":
            return self._risk_parity(covariance, previous), "scipy"
        raise ValueError(f"unsupported sleeve allocation method: {self.config.method}")

    # 中文说明：`_convex_allocate`：内部辅助步骤，不作为稳定公共接口。
    def _convex_allocate(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        previous: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        weight = cp.Variable(len(mean))
        objective = cp.Maximize(
            mean @ weight
            - 0.5 * self.config.risk_aversion * cp.quad_form(
                weight, cp.psd_wrap(covariance)
            )
            - self.config.turnover_penalty * cp.sum_squares(weight - previous)
        )
        problem = cp.Problem(
            objective,
            [
                cp.sum(weight) == 1.0,
                weight >= 0.0,
                weight <= self.config.max_weight,
            ],
        )
        for solver in ("CLARABEL", "SCS"):
            if solver not in cp.installed_solvers():
                continue
            try:
                problem.solve(solver=solver)
                if weight.value is not None:
                    return np.asarray(weight.value), solver
            except cp.error.SolverError:
                continue
        return previous, "fallback"

    # 中文说明：`_risk_parity`：内部辅助步骤，不作为稳定公共接口。
    def _risk_parity(self, covariance: np.ndarray, previous: np.ndarray) -> np.ndarray:
        n_assets = len(previous)

        # 中文说明：`objective`：执行该名称对应的业务计算，并返回调用方所需结果。
        def objective(weight: np.ndarray) -> float:
            marginal = covariance @ weight
            contributions = weight * marginal
            total = contributions.sum()
            target = total / n_assets
            return float(np.square(contributions - target).sum())

        bounds = [(1e-8, self.config.max_weight) for _ in range(n_assets)]
        result = minimize(
            objective,
            previous,
            method="SLSQP",
            bounds=bounds,
            constraints={"type": "eq", "fun": lambda weight: weight.sum() - 1.0},
            options={"maxiter": 500, "ftol": 1e-12},
        )
        return result.x if result.success else previous
