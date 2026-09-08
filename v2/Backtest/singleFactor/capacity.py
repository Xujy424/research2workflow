"""Execution-capacity overlay, deliberately separate from alpha discovery."""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from .config import CapacityConfig
from .metrics import performance


@dataclass(frozen=True)
class CapacityResult:
    summary: pd.DataFrame
    equity: dict[float, pd.Series]
    fill_ratio: dict[float, pd.Series]
    returns: dict[float, pd.Series] = field(default_factory=dict)
    turnover: dict[float, pd.Series] = field(default_factory=dict)
    commission_cost_ratio: dict[float, pd.Series] = field(default_factory=dict)
    impact_cost_ratio: dict[float, pd.Series] = field(default_factory=dict)
    actual_weight: dict[float, pd.DataFrame] = field(default_factory=dict)
    weight_deviation: dict[float, pd.Series] = field(default_factory=dict)
    net_exposure_deviation: dict[float, pd.Series] = field(default_factory=dict)
    group_exposure_deviation: dict[float, pd.Series] = field(default_factory=dict)

    def report(self, print_summary=True, plot=True, show=True,
               figsize=(14, 10)):
        """Print the statistics table and optionally build diagnostic plots."""
        return report_capacity_result(
            self,
            print_summary=print_summary,
            plot=plot,
            show=show,
            figsize=figsize,
        )


class CapacitySimulator:
    """Simulate target weights with lots, participation, costs and impact."""

    def __init__(self, config=CapacityConfig()):
        self.config = config

    def _costs(self, shares, price, amount):
        traded_value = np.abs(shares * price)
        gross_value = traded_value.sum()
        participation = np.divide(
            traded_value,
            amount,
            out=np.zeros_like(traded_value),
            where=amount > 0,
        )
        commission = gross_value * self.config.commission_bps / 1e4
        impact = np.sum(
            traded_value
            * (self.config.impact_coefficient / 100.0)
            * np.sqrt(np.clip(participation, 0.0, None))
        )
        return float(gross_value), float(commission), float(impact)

    @staticmethod
    def _repair_bucket_exposure(requested, candidate, price, groups=None):
        """Match buy/sell completion rates inside each exposure bucket."""
        repaired = candidate.copy()
        requested_value = requested * price
        candidate_value = candidate * price
        if groups is None:
            bucket_masks = [np.ones(len(requested), dtype=bool)]
        else:
            finite = np.isfinite(groups)
            bucket_masks = [
                groups == code for code in np.unique(groups[finite])
            ]
            if (~finite).any():
                bucket_masks.append(~finite)

        for bucket in bucket_masks:
            buy = bucket & (requested_value > 0)
            sell = bucket & (requested_value < 0)
            requested_buy = requested_value[buy].sum()
            requested_sell = -requested_value[sell].sum()
            if requested_buy <= 0 or requested_sell <= 0:
                continue

            candidate_buy = candidate_value[buy].sum()
            candidate_sell = -candidate_value[sell].sum()
            buy_fill = candidate_buy / requested_buy
            sell_fill = candidate_sell / requested_sell
            common_fill = min(buy_fill, sell_fill)
            if candidate_buy > 0:
                repaired[buy] *= common_fill / buy_fill
            if candidate_sell > 0:
                repaired[sell] *= common_fill / sell_fill
        return repaired

    def _buy_cash_required(self, candidate, scale, price, amount):
        gross, commission, impact = self._costs(
            candidate * scale, price, amount
        )
        return gross + commission + impact

    def _cash_limited_buys(self, candidate, price, amount, cash):
        """Apply only the minimum buy haircut needed to pay all costs."""
        if cash <= 0 or not np.any(candidate > 0):
            return np.zeros_like(candidate)

        full_requirement = self._buy_cash_required(
            candidate, 1.0, price, amount
        )
        if full_requirement <= cash:
            return candidate

        low, high = 0.0, 1.0
        for _ in range(50):
            middle = 0.5 * (low + high)
            requirement = self._buy_cash_required(
                candidate, middle, price, amount
            )
            if requirement <= cash:
                low = middle
            else:
                high = middle
        return candidate * low

    def run(
        self, 
        target_weight: pd.DataFrame, 
        execution_price: pd.DataFrame,
        traded_amount: pd.DataFrame,
        exposure_group: pd.DataFrame | None = None,
    ) -> CapacityResult:
        '''
            允许出现由“现金和交易成本约束”产生的小幅实际敞口偏差
            但不允许 ADV 流动性差异直接造成大的多空或行业失衡
        '''
        optional = () if exposure_group is None else (exposure_group,)
        for frame in (execution_price, traded_amount, *optional):
            if not target_weight.index.equals(frame.index) or not target_weight.columns.equals(frame.columns):
                raise ValueError("capacity inputs must have identical axes")
            
        price, amount = execution_price.to_numpy(float), traded_amount.to_numpy(float)
        target = target_weight.to_numpy(float)
        groups = (
            None if exposure_group is None else exposure_group.to_numpy(float)
        )

        summaries, curves, fills = [], {}, {}
        return_curves, turnovers = {}, {}
        commission_curves, impact_curves = {}, {}
        actual_weights, weight_deviations = {}, {}
        net_deviations, group_deviations = {}, {}
        for initial in self.config.capital:
            cash, shares = float(initial), np.zeros(target.shape[1])
            equity_curve = np.zeros(len(target))
            fill_curve = np.full(len(target), np.nan)
            turnover_curve = np.zeros(len(target))
            commission_curve = np.zeros(len(target))
            impact_curve = np.zeros(len(target))
            actual_weight_curve = np.zeros_like(target, dtype=float)
            weight_deviation_curve = np.zeros(len(target))
            net_deviation_curve = np.zeros(len(target))
            group_deviation_curve = np.full(len(target), np.nan)
            last_price = np.zeros(target.shape[1], dtype=float)
            for t in range(len(target)):
                valid_price = np.isfinite(price[t]) & (price[t] > 0)
                last_price[valid_price] = price[t, valid_price]
                equity = cash + np.sum(shares * last_price)  # 持仓*更新价格

                # An unavailable price means no trade, not liquidation at zero.
                desired = shares.copy()
                valid_target = valid_price & np.isfinite(target[t])
                desired[valid_target] = np.trunc(
                    equity * target[t, valid_target] / price[t, valid_target]
                    / self.config.lot_size) * self.config.lot_size
                requested = desired - shares
                amount_t = np.nan_to_num(
                    amount[t], nan=0.0, posinf=0.0, neginf=0.0
                ).clip(min=0.0)
                capacity_value = (
                    amount_t * self.config.max_participation
                )

                execution_mark = np.where(valid_price, price[t], 0.0)
                requested_value = np.abs(requested * execution_mark)
                fill_limit = np.divide(
                    capacity_value,
                    requested_value,
                    out=np.ones_like(requested_value),
                    where=requested_value > 0,
                ).clip(0.0, 1.0)
                candidate = requested * fill_limit
                candidate = self._repair_bucket_exposure(
                    requested,
                    candidate,
                    execution_mark,
                    None if groups is None else groups[t],
                )

                # Repair the material ADV-driven exposure mismatch first.
                # Then sell before buying so only the minimum cash-related buy
                # haircut is allowed to create a residual implementation gap.
                sold = np.maximum(-candidate, 0.0)
                sell_gross, sell_commission, sell_impact = self._costs(
                    sold, execution_mark, amount_t
                )
                shares -= sold
                cash += sell_gross - sell_commission - sell_impact

                candidate_buy = np.maximum(candidate, 0.0)
                bought = self._cash_limited_buys(
                    candidate_buy, execution_mark, amount_t, cash
                )
                buy_gross, buy_commission, buy_impact = self._costs(
                    bought, execution_mark, amount_t
                )
                shares += bought
                cash -= buy_gross + buy_commission + buy_impact

                total_commission = sell_commission + buy_commission
                total_impact = sell_impact + buy_impact
                equity_curve[t] = cash + np.sum(shares * last_price)
                base_equity = max(abs(equity_curve[t]), 1e-12)
                turnover_curve[t] = (
                    0.5 * (sell_gross + buy_gross) / base_equity
                )
                commission_curve[t] = total_commission / base_equity
                impact_curve[t] = total_impact / base_equity

                requested_gross = requested_value.sum()
                if requested_gross > 0:
                    actual_gross = np.sum(
                        (sold + bought) * execution_mark
                    )
                    fill_curve[t] = np.clip(
                        actual_gross / requested_gross, 0.0, 1.0
                    )

                actual_weight_curve[t] = np.divide(
                    shares * last_price,
                    equity_curve[t],
                    out=np.zeros_like(shares),
                    where=equity_curve[t] != 0,
                )
                valid_comparison = np.isfinite(target[t])
                deviation = np.where(
                    valid_comparison,
                    actual_weight_curve[t] - target[t],
                    0.0,
                )
                weight_deviation_curve[t] = np.abs(deviation).sum()  # 股票平均权重偏差
                net_deviation_curve[t] = deviation.sum()             # 净敞口偏差
                if groups is not None:
                    group_t = groups[t]
                    finite_group = np.isfinite(group_t)
                    group_values = [
                        abs(deviation[group_t == code].sum())
                        for code in np.unique(group_t[finite_group])
                    ]
                    if group_values:
                        group_deviation_curve[t] = max(group_values)  # 最大行业通敞口偏差

            equity_series = pd.Series(equity_curve, target_weight.index, name="equity")
            previous_equity = np.r_[initial, equity_curve[:-1]]
            daily_return = np.divide(
                equity_curve,
                previous_equity,
                out=np.ones_like(equity_curve),
                where=previous_equity != 0,
            ) - 1.0
            return_series = pd.Series(
                daily_return, target_weight.index, name="return"
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            stat = performance(return_series)
            stat["capital"] = initial
            stat["average_fill_ratio"] = (
                np.nanmean(fill_curve)
                if np.isfinite(fill_curve).any() else np.nan
            )
            stat["average_turnover"] = turnover_curve.mean()
            stat["average_commission_cost"] = commission_curve.mean()
            stat["average_impact_cost"] = impact_curve.mean()
            stat["average_weight_deviation"] = weight_deviation_curve.mean()
            stat["average_abs_net_exposure_deviation"] = np.mean(
                np.abs(net_deviation_curve)
            )
            stat["average_max_group_exposure_deviation"] = (
                np.nanmean(group_deviation_curve)
                if groups is not None else np.nan
            )
            stat["online_decision"] = (
                "Pass"
                if (
                    stat["average_fill_ratio"] > self.config.min_fill_ratio
                    and stat.get("sharpe", np.nan) > self.config.min_sharpe
                )
                else "Fail"
            )
            summaries.append(stat)
            curves[initial] = equity_series
            fills[initial] = pd.Series(fill_curve, target_weight.index, name="fill_ratio")
            return_curves[initial] = return_series
            turnovers[initial] = pd.Series(
                turnover_curve, target_weight.index, name="turnover"
            )
            commission_curves[initial] = pd.Series(
                commission_curve,
                target_weight.index,
                name="commission_cost_ratio",
            )
            impact_curves[initial] = pd.Series(
                impact_curve,
                target_weight.index,
                name="impact_cost_ratio",
            )
            actual_weights[initial] = pd.DataFrame(
                actual_weight_curve,
                target_weight.index,
                target_weight.columns,
            )
            weight_deviations[initial] = pd.Series(
                weight_deviation_curve,
                target_weight.index,
                name="weight_deviation",
            )
            net_deviations[initial] = pd.Series(
                net_deviation_curve,
                target_weight.index,
                name="net_exposure_deviation",
            )
            group_deviations[initial] = pd.Series(
                group_deviation_curve,
                target_weight.index,
                name="max_group_exposure_deviation",
            )
        return CapacityResult(
            summary=pd.DataFrame(summaries).set_index("capital"),
            equity=curves,
            fill_ratio=fills,
            returns=return_curves,
            turnover=turnovers,
            commission_cost_ratio=commission_curves,
            impact_cost_ratio=impact_curves,
            actual_weight=actual_weights,
            weight_deviation=weight_deviations,
            net_exposure_deviation=net_deviations,
            group_exposure_deviation=group_deviations,
        )


def _capital_label(capital):
    if capital >= 1e8:
        return f"{capital / 1e8:g}e8"
    if capital >= 1e4:
        return f"{capital / 1e4:g}e4"
    return f"{capital:g}"


def report_capacity_result(
    result: CapacityResult,
    print_summary=True,
    plot=True,
    show=True,
    figsize=(14, 10),
):
    """Print the capacity table and plot NAV plus implementation deviation.

    Returns a summary and figure tuple. The figure is None when plot is false.
    Detailed time series remain available on CapacityResult; the compact plot
    focuses on realized NAV and requested-value-weighted fill ratio.
    """
    summary = result.summary.copy()
    if print_summary:
        print(summary.to_string())
    if not plot:
        return summary, None

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required when plot=True"
        ) from exc

    figure, axes = plt.subplots(
        2, 1, figsize=figsize, constrained_layout=True,
        gridspec_kw={"height_ratios": (1.15, 1.0)},
    )
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plot_colors = [
        "#f2b6d2" if color.lower() == "#d62728" else color
        for color in default_colors
    ]
    for axis in axes:
        axis.set_prop_cycle(color=plot_colors)

    equity_items = list(result.equity.items())
    for capital, series in equity_items:
        axes[0].plot(
            series.index,
            series / float(capital),
            label=_capital_label(capital),
            alpha=0.78,
        )
    axes[0].set_title("Normalized equity by initial capital")
    axes[0].set_ylabel("NAV")
    axes[0].grid(alpha=0.2)
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend(fontsize="small")

    fill_items = list(result.fill_ratio.items())
    if fill_items:
        for capital, series in fill_items:
            observed = series.dropna()
            if observed.empty:
                continue
            axes[1].plot(
                observed.index,
                observed,
                alpha=0.78,
                label=(
                    f"{_capital_label(capital)} "
                    f"(mean={observed.mean():.1%})"
                ),
            )
        axes[1].axhline(
            1.0, color="grey", linestyle="--", linewidth=0.8,
            label="full fill",
        )
    else:
        axes[1].text(
            0.5, 0.5, "No fill-ratio series",
            ha="center", va="center", transform=axes[1].transAxes,
        )
    axes[1].set_title(
        "Fill ratio on trade-request dates"
    )
    axes[1].set_xlabel("Target date")
    axes[1].set_ylabel("Executed value / requested value")
    axes[1].grid(alpha=0.2)
    axes[1].tick_params(axis="x", rotation=30)
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(fontsize="small")

    if show:
        plt.show()
    return summary, figure
