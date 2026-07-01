"""Live model monitoring, capacity, crowding, and risk forecast calibration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import numpy as np
import pandas as pd


# 中文说明：定义 `CapacityReport`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class CapacityReport:
    capital_grid: pd.DataFrame
    recommended_capital: float
    diagnostics: Mapping[str, float]


# 中文说明：定义 `TradingRunState`，封装本模块对应的数据、配置与行为。
class TradingRunState(str, Enum):
    ALLOWED = "allowed"
    REDUCED = "reduced"
    BLOCKED = "blocked"


# 中文说明：定义 `MonitoringDecision`，封装本模块对应的数据、配置与行为。
@dataclass(frozen=True)
class MonitoringDecision:
    state: TradingRunState
    factor_action: str
    risk_action: str
    optimizer_action: str
    reasons: tuple[str, ...]


# 中文说明：定义 `DailyTrackingSnapshot`，汇总每日实盘或模拟盘的监控结果。
@dataclass(frozen=True)
class DailyTrackingSnapshot:
    decision: MonitoringDecision
    drift: pd.DataFrame
    risk_calibration: pd.DataFrame
    capacity: CapacityReport
    crowding: pd.DataFrame
    diagnostics: Mapping[str, object]


# 中文说明：定义 `ProductionMonitoringLoop`，封装本模块对应的数据、配置与行为。
class ProductionMonitoringLoop:
    """Translate production observations into explicit governance actions."""

    # 中文说明：`decide`：执行该名称对应的业务计算，并返回调用方所需结果。
    def decide(
        self,
        *,
        reconciliation_passed: bool,
        drift_warning: bool = False,
        risk_calibration_multiplier: float | None = None,
        recommended_capital: float | None = None,
        current_capital: float | None = None,
        maximum_crowding_score: float | None = None,
        crowding_limit: float = 0.85,
    ) -> MonitoringDecision:
        reasons: list[str] = []
        if not reconciliation_passed:
            return MonitoringDecision(
                TradingRunState.BLOCKED,
                "hold",
                "hold",
                "hold",
                ("account_reconciliation_failed",),
            )

        factor_action = "none"
        risk_action = "none"
        optimizer_action = "none"
        state = TradingRunState.ALLOWED
        if drift_warning:
            factor_action = "request_downweight_or_pause"
            state = TradingRunState.REDUCED
            reasons.append("factor_drift_warning")
        if (
            risk_calibration_multiplier is not None
            and np.isfinite(risk_calibration_multiplier)
            and abs(risk_calibration_multiplier - 1.0) > 0.20
        ):
            risk_action = "request_risk_recalibration"
            state = TradingRunState.REDUCED
            reasons.append("risk_forecast_miscalibrated")
        if (
            recommended_capital is not None
            and current_capital is not None
            and current_capital > recommended_capital
        ):
            optimizer_action = "cap_strategy_capital"
            state = TradingRunState.REDUCED
            reasons.append("capacity_limit_exceeded")
        if (
            maximum_crowding_score is not None
            and maximum_crowding_score > crowding_limit
        ):
            optimizer_action = "tighten_position_limits"
            state = TradingRunState.REDUCED
            reasons.append("crowding_limit_exceeded")
        return MonitoringDecision(
            state,
            factor_action,
            risk_action,
            optimizer_action,
            tuple(reasons),
        )


# 中文说明：定义 `CapacityAnalyzer`，封装本模块对应的数据、配置与行为。
class CapacityAnalyzer:
    # 中文说明：`analyze`：分析输入并生成诊断结果。
    def analyze(
        self,
        target_weights: pd.Series,
        adv_currency: pd.Series,
        expected_alpha: pd.Series,
        capitals: list[float],
        max_participation: float = 0.10,
        impact_coefficient: float = 0.10,
    ) -> CapacityReport:
        weights = target_weights.reindex(adv_currency.index).fillna(0.0)
        alpha = expected_alpha.reindex(adv_currency.index).fillna(0.0)
        rows: list[dict[str, float]] = []
        for capital in capitals:
            notional = weights.abs() * capital
            participation = notional / adv_currency.clip(lower=1.0)
            impact = impact_coefficient * np.sqrt(participation.clip(lower=0.0))
            gross_alpha = float((weights * alpha).sum())
            cost = float((weights.abs() * impact).sum())
            rows.append(
                {
                    "capital": float(capital),
                    "gross_alpha": gross_alpha,
                    "impact_cost": cost,
                    "net_alpha": gross_alpha - cost,
                    "max_participation": float(participation.max()),
                    "tradable_ratio": float((participation <= max_participation).mean()),
                }
            )
        grid = pd.DataFrame(rows).set_index("capital")
        feasible = grid[
            (grid["max_participation"] <= max_participation) & (grid["net_alpha"] > 0)
        ]
        recommended = float(feasible.index.max()) if len(feasible) else 0.0
        return CapacityReport(
            grid,
            recommended,
            {
                "max_participation_limit": max_participation,
                "alpha_half_life_not_modelled": 1.0,
            },
        )


# 中文说明：定义 `CrowdingMonitor`，封装本模块对应的数据、配置与行为。
class CrowdingMonitor:
    # 中文说明：`score`：计算评分或监控指标。
    def score(
        self,
        weights: pd.Series,
        ownership: pd.Series | None = None,
        short_interest: pd.Series | None = None,
        turnover_percentile: pd.Series | None = None,
    ) -> pd.DataFrame:
        frame = pd.DataFrame(index=weights.index)
        frame["position_concentration"] = weights.abs() / max(weights.abs().sum(), 1e-12)
        inputs = {
            "ownership": ownership,
            "short_interest": short_interest,
            "turnover_percentile": turnover_percentile,
        }
        for name, values in inputs.items():
            frame[name] = (
                values.reindex(weights.index).rank(pct=True)
                if values is not None
                else 0.0
            )
        frame["crowding_score"] = (
            0.35 * frame["position_concentration"].rank(pct=True)
            + 0.25 * frame["ownership"]
            + 0.20 * frame["short_interest"]
            + 0.20 * frame["turnover_percentile"]
        )
        return frame.sort_values("crowding_score", ascending=False)


# 中文说明：定义 `RiskForecastMonitor`，封装本模块对应的数据、配置与行为。
class RiskForecastMonitor:
    # 中文说明：`calibrate`：执行该名称对应的业务计算，并返回调用方所需结果。
    def calibrate(
        self,
        predicted_volatility: pd.Series,
        realized_returns: pd.Series,
        window: int = 20,
    ) -> pd.DataFrame:
        realized = realized_returns.rolling(window, min_periods=window // 2).std()
        predicted = predicted_volatility.reindex(realized.index)
        ratio = realized / predicted.replace(0.0, np.nan)
        return pd.DataFrame(
            {
                "predicted_volatility": predicted,
                "realized_volatility": realized,
                "realized_to_predicted": ratio,
                "calibration_multiplier": ratio.ewm(span=60, min_periods=20).mean(),
            }
        )


# 中文说明：定义 `LiveDriftMonitor`，封装本模块对应的数据、配置与行为。
class LiveDriftMonitor:
    # 中文说明：`monitor_ic`：监控运行状态并生成治理信号。
    def monitor_ic(
        self,
        live_ic: pd.Series,
        reference_mean: float,
        reference_std: float,
        warning_z: float = -2.0,
    ) -> pd.DataFrame:
        rolling_mean = live_ic.rolling(20, min_periods=10).mean()
        z_score = (rolling_mean - reference_mean) / max(reference_std, 1e-12)
        return pd.DataFrame(
            {
                "live_ic": live_ic,
                "rolling_ic": rolling_mean,
                "z_score": z_score,
                "warning": z_score < warning_z,
            }
        )


# 中文说明：定义 `DailyProductionTracker`，统一执行对账后的每日监控闭环。
class DailyProductionTracker:
    # 中文说明：`evaluate`：汇总漂移、风险、容量、拥挤并生成运行决策。
    def evaluate(
        self,
        *,
        reconciliation_passed: bool,
        live_ic: pd.Series,
        reference_ic_mean: float,
        reference_ic_std: float,
        predicted_volatility: pd.Series,
        realized_returns: pd.Series,
        target_weights: pd.Series,
        adv_currency: pd.Series,
        expected_alpha: pd.Series,
        capitals: list[float],
        current_capital: float,
        ownership: pd.Series | None = None,
        short_interest: pd.Series | None = None,
        turnover_percentile: pd.Series | None = None,
    ) -> DailyTrackingSnapshot:
        drift = LiveDriftMonitor().monitor_ic(
            live_ic,
            reference_ic_mean,
            reference_ic_std,
        )
        calibration = RiskForecastMonitor().calibrate(
            predicted_volatility,
            realized_returns,
        )
        capacity = CapacityAnalyzer().analyze(
            target_weights,
            adv_currency,
            expected_alpha,
            capitals,
        )
        crowding = CrowdingMonitor().score(
            target_weights,
            ownership,
            short_interest,
            turnover_percentile,
        )
        drift_warning = bool(drift["warning"].fillna(False).iloc[-1]) if len(drift) else False
        multiplier = (
            float(calibration["calibration_multiplier"].dropna().iloc[-1])
            if calibration["calibration_multiplier"].notna().any()
            else None
        )
        maximum_crowding = (
            float(crowding["crowding_score"].max()) if len(crowding) else None
        )
        decision = ProductionMonitoringLoop().decide(
            reconciliation_passed=reconciliation_passed,
            drift_warning=drift_warning,
            risk_calibration_multiplier=multiplier,
            recommended_capital=capacity.recommended_capital,
            current_capital=current_capital,
            maximum_crowding_score=maximum_crowding,
        )
        return DailyTrackingSnapshot(
            decision=decision,
            drift=drift,
            risk_calibration=calibration,
            capacity=capacity,
            crowding=crowding,
            diagnostics={
                "reconciliation_passed": reconciliation_passed,
                "current_capital": current_capital,
            },
        )
