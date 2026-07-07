"""Factor monitoring utilities for lifecycle governance.

The functions in this module work on production-style matrices (T x N). They
produce monitor snapshots and status suggestions, but they do not mutate the
registry by themselves. Human approval stays in FactorRegistry.promote/retire.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..matrix_math import (
    IC,
    rankIC,
    calc_group_ret,
    calc_maxdrawdown,
    robust_extreme_ratio_by_row,
)
from .registry import FactorStatus


@dataclass(frozen=True)
class FactorMonitorConfig:
    n_groups: int = 10
    top_quantile: float = 0.1
    bottom_quantile: float = 0.1
    min_obs: int = 50
    rolling_windows: tuple[int, ...] = (20, 60, 120, 250)
    extreme_mad_multiple: float = 5.0

    min_coverage: float = 0.80
    max_nan_ratio: float = 0.20
    max_extreme_ratio: float = 0.05
    max_corr_with_production: float = 0.80

    research_min_icir_60: float = 0.0
    research_min_hit_rate_60: float = 0.50
    candidate_min_icir_60: float = 0.15
    candidate_min_hit_rate_60: float = 0.52
    candidate_min_monotonicity_60: float = 0.20

    production_pause_icir_20: float = -0.30
    production_pause_hit_rate_20: float = 0.40
    production_pause_ls_sharpe_20: float = -0.30
    shadow_retire_icir_120: float = -0.10
    shadow_retire_hit_rate_120: float = 0.45


@dataclass(frozen=True)
class LifecycleDecision:
    factor_id: str
    version: str
    current_status: FactorStatus
    suggested_status: FactorStatus
    action: str
    reason: str
    metrics_snapshot: Mapping[str, Any] = field(default_factory=dict)


class FactorMonitor:
    """Calculate matrix-based diagnostics and lifecycle suggestions."""

    def __init__(self, config: FactorMonitorConfig | None = None) -> None:
        self.config = config or FactorMonitorConfig()

    def daily_performance(
        self,
        factor_values: np.ndarray,
        forward_returns: np.ndarray,
        *,
        dates: Sequence[Any] | None = None,
        mask: np.ndarray | None = None,
    ) -> pd.DataFrame:
        x = np.asarray(factor_values, dtype=float)
        y = np.asarray(forward_returns, dtype=float)
        if x.shape != y.shape:
            raise ValueError(f"factor_values and forward_returns shape mismatch: {x.shape} vs {y.shape}")
        if mask is None:
            valid_mask = np.ones_like(x, dtype=bool)
        else:
            valid_mask = np.asarray(mask, dtype=bool)
            if valid_mask.shape != x.shape:
                raise ValueError(f"mask shape mismatch: {valid_mask.shape} vs {x.shape}")

        factor_valid = valid_mask & np.isfinite(x)
        valid = factor_valid & np.isfinite(y)
        n_total = valid_mask.sum(axis=1)
        n_valid = valid.sum(axis=1)
        date_index = list(dates) if dates is not None else list(range(x.shape[0]))

        ic = IC(np.where(valid_mask, x, np.nan), y)
        rank_ic = rankIC(np.where(valid_mask, x, np.nan), y)
        alpha_df = pd.DataFrame(np.where(valid_mask, x, np.nan), index=date_index)
        group_ret_df = calc_group_ret(alpha_df, y, num_group=self.config.n_groups)
        group_ret = group_ret_df.to_numpy(dtype=float)
        group_axis = np.broadcast_to(group_ret_df.columns.to_numpy(dtype=float), group_ret.shape)
        monotonicity = IC(group_axis, group_ret)
        bottom_ret = group_ret_df[10].to_numpy(dtype=float)
        top_ret = group_ret_df[1].to_numpy(dtype=float)
        ls_ret = top_ret - bottom_ret
        group_spread = ls_ret.copy()

        out = pd.DataFrame({
            "date": date_index,
            "n_total": n_total.astype(int),
            "n_valid": n_valid.astype(int),
            "coverage": np.divide(n_valid, n_total, out=np.full(x.shape[0], np.nan, dtype=float), where=n_total > 0),
            "nan_ratio": 1.0 - np.divide(factor_valid.sum(axis=1), n_total, out=np.full(x.shape[0], np.nan, dtype=float), where=n_total > 0),
            "extreme_ratio": robust_extreme_ratio_by_row(x, mad_multiple=self.config.extreme_mad_multiple, mask=valid_mask),
            "ic": ic,
            "rank_ic": rank_ic,
            "long_short_return": ls_ret,
            "top_return": top_ret,
            "bottom_return": bottom_ret,
            "group_monotonicity": monotonicity,
            "group_spread_return": group_spread,
        })
        insufficient = n_valid < self.config.min_obs
        metric_cols = [
            "ic",
            "rank_ic",
            "long_short_return",
            "top_return",
            "bottom_return",
            "group_monotonicity",
            "group_spread_return",
        ]
        out.loc[insufficient, metric_cols] = np.nan
        return out.sort_values("date").reset_index(drop=True)

    def add_rolling_metrics(self, daily_perf: pd.DataFrame) -> pd.DataFrame:
        out = daily_perf.sort_values("date").copy()
        ric = out["rank_ic"].astype(float)
        ls_ret = out["long_short_return"].astype(float)
        mono = out["group_monotonicity"].astype(float)
        for window in self.config.rolling_windows:
            ic_mean = ric.rolling(window).mean()
            ic_std = ric.rolling(window).std()
            ls_mean = ls_ret.rolling(window).mean()
            ls_std = ls_ret.rolling(window).std()
            out[f"rank_ic_mean_{window}"] = ic_mean
            out[f"rank_ic_std_{window}"] = ic_std
            out[f"rank_ic_ir_{window}"] = ic_mean / ic_std.replace(0.0, np.nan)
            out[f"ic_hit_rate_{window}"] = ric.gt(0.0).rolling(window).mean()
            out[f"ls_ret_mean_{window}"] = ls_mean
            out[f"ls_ret_sharpe_{window}"] = ls_mean / ls_std.replace(0.0, np.nan)
            out[f"ls_max_drawdown_{window}"] = ls_ret.rolling(window).apply(calc_maxdrawdown, raw=False)
            out[f"monotonicity_mean_{window}"] = mono.rolling(window).mean()
        return out

    def latest_summary(self, rolling_perf: pd.DataFrame, factor_id: str, version: str) -> dict[str, Any]:
        if rolling_perf.empty:
            return {"factor_id": factor_id, "version": version}
        latest = rolling_perf.sort_values("date").iloc[-1]
        keys = [
            "date",
            "coverage",
            "nan_ratio",
            "extreme_ratio",
            "ic",
            "rank_ic",
            "long_short_return",
            "rank_ic_ir_20",
            "rank_ic_ir_60",
            "rank_ic_ir_120",
            "ic_hit_rate_20",
            "ic_hit_rate_60",
            "ic_hit_rate_120",
            "ls_ret_sharpe_20",
            "ls_ret_sharpe_60",
            "monotonicity_mean_60",
        ]
        summary = {"factor_id": factor_id, "version": version}
        summary.update({key: _json_value(latest.get(key)) for key in keys if key in latest.index})
        return summary

    def decide(
        self,
        factor_id: str,
        version: str,
        current_status: FactorStatus | str,
        metrics: Mapping[str, Any] | pd.Series,
        *,
        max_corr_with_production: float | None = None,
    ) -> LifecycleDecision:
        status = FactorStatus(current_status)
        snapshot = metrics.to_dict() if isinstance(metrics, pd.Series) else dict(metrics)
        cfg = self.config

        coverage = _get(snapshot, "coverage", 1.0)
        nan_ratio = _get(snapshot, "nan_ratio", 0.0)
        extreme_ratio = _get(snapshot, "extreme_ratio", 0.0)
        data_bad = (
            _finite_lt(coverage, cfg.min_coverage)
            or _finite_gt(nan_ratio, cfg.max_nan_ratio)
            or _finite_gt(extreme_ratio, cfg.max_extreme_ratio)
        )
        if data_bad and status == FactorStatus.PRODUCTION:
            return LifecycleDecision(factor_id, version, status, FactorStatus.SHADOW, "pause", "data quality below production threshold", snapshot)
        if data_bad:
            return LifecycleDecision(factor_id, version, status, status, "observe", "data quality below threshold", snapshot)

        if max_corr_with_production is not None and np.isfinite(max_corr_with_production):
            if max_corr_with_production > cfg.max_corr_with_production and status in {FactorStatus.RESEARCH, FactorStatus.CANDIDATE}:
                return LifecycleDecision(factor_id, version, status, FactorStatus.CANDIDATE, "observe", "high correlation with production factors", snapshot)

        if status == FactorStatus.RESEARCH:
            ok = _get(snapshot, "rank_ic_ir_60", -np.inf) >= cfg.research_min_icir_60 and _get(snapshot, "ic_hit_rate_60", 0.0) >= cfg.research_min_hit_rate_60
            if ok:
                return LifecycleDecision(factor_id, version, status, FactorStatus.CANDIDATE, "promote", "research factor passed candidate watch criteria", snapshot)

        if status == FactorStatus.CANDIDATE:
            ok = (
                _get(snapshot, "rank_ic_ir_60", -np.inf) >= cfg.candidate_min_icir_60
                and _get(snapshot, "ic_hit_rate_60", 0.0) >= cfg.candidate_min_hit_rate_60
                and _get(snapshot, "monotonicity_mean_60", -np.inf) >= cfg.candidate_min_monotonicity_60
            )
            if ok:
                return LifecycleDecision(factor_id, version, status, FactorStatus.SHADOW, "promote", "candidate passed shadow observation criteria", snapshot)

        if status == FactorStatus.SHADOW:
            ok = (
                _get(snapshot, "rank_ic_ir_60", -np.inf) >= cfg.candidate_min_icir_60
                and _get(snapshot, "ic_hit_rate_60", 0.0) >= cfg.candidate_min_hit_rate_60
                and _get(snapshot, "monotonicity_mean_60", -np.inf) >= cfg.candidate_min_monotonicity_60
            )
            bad_mid = _get(snapshot, "rank_ic_ir_120", 0.0) <= cfg.shadow_retire_icir_120 and _get(snapshot, "ic_hit_rate_120", 1.0) <= cfg.shadow_retire_hit_rate_120
            if ok:
                return LifecycleDecision(factor_id, version, status, FactorStatus.PRODUCTION, "promote", "shadow factor passed production criteria", snapshot)
            if bad_mid:
                return LifecycleDecision(factor_id, version, status, FactorStatus.RETIRED, "retire", "shadow factor failed to recover", snapshot)

        if status == FactorStatus.PRODUCTION:
            bad_recent = (
                _get(snapshot, "rank_ic_ir_20", 0.0) <= cfg.production_pause_icir_20
                or _get(snapshot, "ic_hit_rate_20", 1.0) <= cfg.production_pause_hit_rate_20
                or _get(snapshot, "ls_ret_sharpe_20", 0.0) <= cfg.production_pause_ls_sharpe_20
            )
            if bad_recent:
                return LifecycleDecision(factor_id, version, status, FactorStatus.SHADOW, "pause", "recent production performance deterioration", snapshot)

        return LifecycleDecision(factor_id, version, status, status, "keep", "keep current status", snapshot)

    def evaluate(
        self,
        factor_id: str,
        version: str,
        current_status: FactorStatus | str,
        factor_values: np.ndarray,
        forward_returns: np.ndarray,
        *,
        dates: Sequence[Any] | None = None,
        mask: np.ndarray | None = None,
        max_corr_with_production: float | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any], LifecycleDecision]:
        daily = self.daily_performance(factor_values, forward_returns, dates=dates, mask=mask)
        rolling = self.add_rolling_metrics(daily)
        summary = self.latest_summary(rolling, factor_id, version)
        decision = self.decide(
            factor_id,
            version,
            current_status,
            summary,
            max_corr_with_production=max_corr_with_production,
        )
        return rolling, summary, decision


def _get(metrics: Mapping[str, Any], key: str, default: float) -> float:
    value = metrics.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _finite_lt(value: float, threshold: float) -> bool:
    return np.isfinite(value) and value < threshold


def _finite_gt(value: float, threshold: float) -> bool:
    return np.isfinite(value) and value > threshold


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value
