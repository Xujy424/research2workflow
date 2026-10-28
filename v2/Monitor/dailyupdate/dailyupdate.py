"""每日监控编排、统一长表及日期轴二进制更新入口。"""
from __future__ import annotations
from pathlib import Path
from functools import reduce
import warnings
import numpy as np
import pandas as pd

if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = 'v2.Monitor.dailyupdate'

from .config import *
from .context import MonitorContext, _daily_return
from .utils import *
from .utils import _aggregated_percentile, _clean, _correlation_series, _history_percentile, _lag_correlation_columns, _limit_down_series, _matrix_from_indexed, _mean_pair_correlation, _nanmean_rows, _p, _pca_series, _pct, _percentile, _rolling_mean, _rv_metrics, _series_percentile, _sum_matrices, _sum_preserve_nan, _summarize, _time_key, _weighted_rows
from .vals.acd_jump_intensity import acd_jump_intensity
from .vals.active_buy_ratio import active_buy_ratio
from .vals.active_trade_imbalance import active_trade_imbalance
from .vals.amihud_illiquidity import amihud_illiquidity
from .vals.amount_change import amount_change
from .vals.amount_percentile import amount_percentile
from .vals.amount_share_percentile import amount_share_percentile
from .vals.average_correlation import average_correlation
from .vals.balance_change import balance_change
from .vals.best_level_imbalance import best_level_imbalance
from .vals.beta_spread_percentile import beta_spread_percentile
from .utils import cancellation_rate
from .utils import credit_impulse
from .vals.cross_section_dispersion import cross_section_dispersion
from .vals.daily_cancellation_rate import daily_cancellation_rate
from .vals.downside_correlation import downside_correlation
from .vals.downside_resonance import downside_resonance
from .vals.earnings_bond_spread import earnings_bond_spread
from .vals.effective_spread import effective_spread
from .vals.estimate_kyle_lambda import estimate_kyle_lambda
from .vals.executable_capacity import executable_capacity
from .utils import five_level_depth
from .vals.five_level_depth_notional import five_level_depth_notional
from .vals.growth_value_spread_percentile import growth_value_spread_percentile
from .utils import index_eps_bond_spread
from .utils import index_pe_bond_spread
from .vals.industry_relative_return import industry_relative_return
from .vals.jump_frequency_30m import jump_frequency_30m
from .vals.jump_volume_divergence import jump_volume_divergence
from .vals.level_percentile import level_percentile
from .vals.limit_down_ratio import limit_down_ratio
from .utils import liquidity_coverage
from .vals.liquidity_pressure_percentile import liquidity_pressure_percentile
from .vals.new_high_low_share import new_high_low_share
from .vals.order_flow_imbalance import order_flow_imbalance
from .vals.order_impact_cost import order_impact_cost
from .utils import pb_spread_percentile
from .vals.pca_explained_ratio import pca_explained_ratio
from .utils import pe_spread_percentile
from .vals.relative_return import relative_return
from .vals.rotation_speed_percentile import rotation_speed_percentile
from .vals.size_spread_percentile import size_spread_percentile
from .vals.standardized_acd_intensity import standardized_acd_intensity
from .vals.top_share import top_share
from .vals.turnover import turnover
from .utils import volatility_risk_premium
from .vals.worst_rolling_loss import worst_rolling_loss


# 兼容旧面板名称；所有变体绑定到注册表中的同一个参数化函数。
impact_path_10m = INDICATORS['impact_path_10m'].load()
impact_path_30m = INDICATORS['impact_path_30m'].load()
impact_path_60m = INDICATORS['impact_path_60m'].load()
impact_path_120m = INDICATORS['impact_path_120m'].load()
impact_decay_1m = INDICATORS['impact_decay_1m'].load()
impact_decay_5m = INDICATORS['impact_decay_5m'].load()
impact_decay_15m = INDICATORS['impact_decay_15m'].load()
impact_decay_30m = INDICATORS['impact_decay_30m'].load()
order_flow_autocorrelation_lag1 = INDICATORS['order_flow_autocorrelation_lag1'].load()
order_flow_autocorrelation_lag5 = INDICATORS['order_flow_autocorrelation_lag5'].load()
pe_percentile = INDICATORS['pe_percentile'].load()
pb_percentile = INDICATORS['pb_percentile'].load()


def _metric_value(metric, member, period, amount, float_mv, buy, sell,
                  margin, monitor, start, end, history):
    a = np.where(member, amount, np.nan)
    if metric == "成交额变化": return INDICATORS['amount_change'].load()(a, history=history, period=period)
    if metric == "top10%成交额占比": return INDICATORS['top_share'].load()(a, history=history, period=period)
    if metric == "自由流通换手率": return turnover(a, np.where(member, float_mv, np.nan), history, period)
    if metric == "主动买入占比": return active_buy_ratio(np.where(member, buy, np.nan), np.where(member, sell, np.nan), history, period)
    if metric in {"大单成交占比", "中单成交占比", "小单成交占比"}:
        key = {"大单成交占比": "large", "中单成交占比": "medium", "小单成交占比": "small"}[metric]
        name = f'{key}_trade_share'
        return INDICATORS[name].load()(a, np.isfinite(monitor._matrix(f"{key}_buy", start, end)), history, period)
    if metric == "融资余额变化率": return balance_change(margin, history, period)
    raise ValueError(f"未注册资金流指标: {metric}")

class MacroMonitor:
    """计算宏观指标当前值、变化和历史分位。"""

    def __init__(self, context):
        self.context = context

    def _series(self, name, start, end):
        return self.context.series(name, start, end)

    def snapshot(self, asof, history=252, periods=MACRO_PERIODS, metrics=None):
        end = pd.Timestamp(asof)
        start = end - pd.DateOffset(years=3)
        c = self.context
        specs = [spec for spec in INDICATORS.values() if spec.module == 'macro'
                 and (metrics is None or spec.label in metrics)]
        selected = tuple(spec.label for spec in specs)
        needed = set().union(*(MACRO_INPUTS[spec.name] for spec in specs)) if specs else set()
        raw = {name: self._series(name, start, end) for name in needed}
        trading_index = pd.DatetimeIndex(c.dates)
        trading_index = trading_index[(trading_index >= start.normalize()) & (trading_index <= end.normalize())]
        raw = {name: _clean(value).reindex(_clean(value).index.union(trading_index).sort_values())
               .ffill().reindex(trading_index) for name, value in raw.items()}
        series = {spec.label: spec.load()(raw) for spec in specs}
        if '信用脉冲' in series:
            series['信用脉冲'] = series['信用脉冲'].reindex(trading_index).ffill()
        rows = []
        for name in selected:
            x = series.get(name, pd.Series(dtype=float))
            value = level(x) if len(x) else np.nan
            if name in RATE_METRICS:
                unit = "bp" if name in SPREAD_METRICS else "%"
            elif name in {"M2同比", "社融存量同比"}:
                unit = "%"
            elif name == "信用脉冲":
                unit = "%GDP"
            else:
                unit = "比例"
            for label, lag in (("当前值", 0), ("5日变化", 5), ("20日变化", 20)):
                raw_value = value if lag == 0 else change(x, lag)
                hist = percentile(x, history) if lag == 0 else rolling_change_percentile(x, lag, history)
                display_value = raw_value
                if name in SPREAD_METRICS and np.isfinite(display_value):
                    display_value *= 100.0
                elif name in RATE_METRICS and lag > 0 and np.isfinite(display_value):
                    display_value *= 100.0
                elif name == "信用脉冲" and np.isfinite(display_value):
                    display_value *= 100.0
                display_unit = unit if lag == 0 else ("bp" if name in RATE_METRICS else unit)
                rows.append({"指标": name, "周期": label, "数值": display_value,
                             "单位": display_unit, "历史分位": hist})
        return pd.DataFrame(rows).set_index(["指标", "周期"]) if rows else pd.DataFrame()

    @staticmethod
    def table(result, decimals=6):
        return result.round(decimals)

class MicroMonitor:
    """调用独立的个股指标函数，再按当前域市值权重计算历史分位表。"""

    METRIC_FUNCTIONS = {INDICATORS[name].label: INDICATORS[name].load() for name in MICRO_SNAPSHOT_METRICS}

    DOMAIN_POLICY = MICRO_DOMAIN_POLICY

    def __init__(self, context):
        self.context = context

    def _daily_individual_factors(self, start, end, metrics=None, show_progress=True):
        c = self.context
        all_names = tuple(self.METRIC_FUNCTIONS) + (
            "主动成交不平衡", "订单流自相关-lag1", "订单流自相关-lag5",
            "Amihud非流动性", "跳跃-成交量偏离", "标准化跳跃强度",
            "订单冲击成本", "可成交量", "撤单率",
        )
        requested = None if metrics is None else set(metrics)
        names = all_names if requested is None else tuple(name for name in all_names if name in requested)
        selected = set(names)
        needs_acd = bool(selected.intersection(("ACD预期跳跃强度", "标准化跳跃强度", "跳跃-成交量偏离")))
        needs_amount = bool(selected.intersection(("Amihud非流动性", "跳跃-成交量偏离", "订单冲击成本", "可成交量")))
        needs_quotes = needs_acd or bool(selected.intersection(self.METRIC_FUNCTIONS)) or bool(
            selected.intersection(("订单冲击成本", "可成交量"))
        )
        factors = {name: np.full((end - start + 1, len(c.ticks)), np.nan) for name in names}
        acd_history = []
        amount_history = []
        date_indices = range(start, end + 1)
        total_days = end - start + 1
        for row, idx in enumerate(date_indices):
            date = c.dates[idx]
            quotes = c.aligned_snapshot(date) if needs_quotes else None
            amount = None
            if needs_amount:
                amount = np.asarray(c.row("m_essentials/amount", idx), float)
                if amount.ndim > 1:
                    amount = np.nansum(amount, axis=0)
            daily_values = {
                name: fn(quotes) for name, fn in self.METRIC_FUNCTIONS.items()
                if name in selected and name != "ACD预期跳跃强度"
            }
            if "ACD预期跳跃强度" in selected or needs_acd:
                acd = acd_jump_intensity(quotes)
                acd_history.append(acd)
                if "ACD预期跳跃强度" in selected:
                    daily_values["ACD预期跳跃强度"] = acd
            if "主动成交不平衡" in selected:
                daily_values["主动成交不平衡"] = active_trade_imbalance(c, date)
            if "撤单率" in selected:
                daily_values["撤单率"] = daily_cancellation_rate(c, date)
            if "Amihud非流动性" in selected:
                daily_values["Amihud非流动性"] = amihud_illiquidity(_daily_return(c, idx), amount)
            if "订单流自相关-lag1" in selected:
                daily_values["订单流自相关-lag1"] = order_flow_autocorrelation_lag1(quotes)
            if "订单流自相关-lag5" in selected:
                daily_values["订单流自相关-lag5"] = order_flow_autocorrelation_lag5(quotes)
            if "可成交量" in selected:
                daily_values["可成交量"] = executable_capacity(quotes, amount)
            if "订单冲击成本" in selected:
                kyle = daily_values.get("KyleLambda")
                if kyle is None:
                    kyle = estimate_kyle_lambda(quotes)
                daily_values["订单冲击成本"] = order_impact_cost(quotes, 0.10 * amount, kyle)
            for name, values in daily_values.items():
                if name in factors and values is not None:
                    factors[name][row] = values
            if needs_acd and len(acd_history) > 20:
                acd_matrix = np.asarray(acd_history, float)
                current_acd_z = standardized_acd_intensity(acd_matrix[-1], acd_matrix[:-1])
                if "标准化跳跃强度" in selected:
                    factors["标准化跳跃强度"][row] = current_acd_z
                if "跳跃-成交量偏离" in selected:
                    amount_history.append(amount)
                    factors["跳跃-成交量偏离"][row] = jump_volume_divergence(
                        current_acd_z, np.asarray(amount_history[-5:], float),
                    )
            elif "跳跃-成交量偏离" in selected:
                amount_history.append(amount)
            if show_progress:
                date_text = pd.Timestamp(date).strftime("%Y-%m-%d")
                print(f"[{row + 1}/{total_days}] {date_text} completed", flush=True)
        return factors

    def _domain_series(self, values, start, end, domain):
        c = self.context
        if domain.startswith("barra"):
            factor = list(c.style_masks)[int(domain[5:]) - 1]
            mask = c.matrix(c.style_masks[factor], start, end)
            weights = c.stock_weight_matrix(start, end)
            high = _weighted_rows(values, np.where(mask > 0, weights, np.nan))
            low = _weighted_rows(values, np.where(mask < 0, weights, np.nan))
            return high - low
        return _weighted_rows(values, c.domain_weights(start, end, domain))

    def snapshot(self, asof, domains=("全市场",), history=252, periods=PERIODS, metrics=None, expand_domains=True, show_progress=True):
        c = self.context
        end = c.position(asof)
        if end < 0:
            return pd.DataFrame()
        warmup = max(history + max(periods) + 30, 252 + max(periods) + 30)
        start = max(0, end - warmup)
        daily = self._daily_individual_factors(start, end, metrics, show_progress)
        selected_metrics = tuple(daily)
        all_domains = ("全市场",) + INDEX_DOMAINS + STYLE_DOMAINS + tuple(c.industry_map.keys())
        requested = None if expand_domains and tuple(domains) == ("全市场",) else set(domains)
        rows = []
        for metric in selected_metrics:
            if metric not in daily:
                continue
            scope = self.DOMAIN_POLICY.get(metric, "all")
            candidates = all_domains if scope == "all" else (
                ("全市场",) + INDEX_DOMAINS if scope == "index" else
                ("全市场",) + INDEX_DOMAINS + STYLE_DOMAINS
            )
            if requested is not None:
                candidates = tuple(domain for domain in candidates if domain in requested)
            for domain in candidates:
                series = self._domain_series(daily[metric], start, end, domain)
                for period in periods:
                    rows.append({"指标": metric, "域": domain, "周期": period,
                                 "历史分位": _history_percentile(series, history, period)})
        if not rows:
            return pd.DataFrame()
        result = pd.DataFrame(rows).pivot(index=["指标", "域"], columns="周期", values="历史分位")
        result.columns = [f"{int(period)}日历史分位" for period in result.columns]
        return result

    @staticmethod
    def table(result, decimals=4):
        return result.round(decimals)

class MoneyFlowMonitor:
    """资金交易指标编排器，返回指标×域长表。"""

    DEFAULT_FIELDS = MONEYFLOW_FIELDS

    def __init__(self, context, field_map=None):
        self.context = context
        self.fields = {**self.DEFAULT_FIELDS, **(field_map or {})}

    def _matrix(self, name, start, end):
        if hasattr(self, '_needed_fields') and name not in self._needed_fields:
            return np.full((end-start+1, len(self.context.ticks)), np.nan)
        return self.context.matrix(self.fields[name], start, end)

    def snapshot(self, asof, domains=("全市场",), history=252, periods=PERIODS,
                 metrics=("主动买入占比", "大单成交占比", "中单成交占比", "小单成交占比",
                          "成交额变化", "top10%成交额占比", "自由流通换手率", "融资余额变化率"), expand_domains=True):
        c = self.context
        end = c.position(asof)
        start = max(0, end - history - max(periods) - 30)
        self._needed_fields = set()
        if set(metrics) - {'融资余额变化率'}:
            self._needed_fields.update(f'{size}_{side}' for size in ('large', 'medium', 'small', 'extra') for side in ('buy', 'sell'))
        if '自由流通换手率' in metrics:
            self._needed_fields.add('float_mv')
        amount = _sum_matrices(
            [self._matrix(f"{size}_buy", start, end)
             + self._matrix(f"{size}_sell", start, end)
             for size in ("large", "medium", "small", "extra")],
        )
        float_mv = self._matrix("float_mv", start, end)
        margin = c.edb(
            c.edb_codes.get("margin_balance", "M0075990"),
            c.dates[start], c.dates[end],
        ).reindex(c.dates[start:end + 1]).ffill().to_numpy(dtype=float) if "融资余额变化率" in metrics else np.full(end-start+1, np.nan)
        buy = _sum_matrices(
            [self._matrix(f"{size}_buy", start, end)
             for size in ("large", "medium", "small", "extra")],
        )
        sell = _sum_matrices(
            [self._matrix(f"{size}_sell", start, end)
             for size in ("large", "medium", "small", "extra")],
        )
        selected = {
            "主动买入占比": ("全市场",) + INDEX_DOMAINS + STYLE_DOMAINS + tuple(c.industry_map.keys()),
            "大单成交占比": ("全市场",) + INDEX_DOMAINS,
            "中单成交占比": ("全市场",) + INDEX_DOMAINS,
            "小单成交占比": ("全市场",) + INDEX_DOMAINS,
            "成交额变化": ("全市场",) + INDEX_DOMAINS + tuple(c.industry_map.keys()),
            "top10%成交额占比": ("全市场",),
            "自由流通换手率": ("全市场",) + INDEX_DOMAINS + STYLE_DOMAINS + tuple(c.industry_map.keys()),
            "融资余额变化率": ("全市场",),
        }
        requested = None if expand_domains and tuple(domains) == ("全市场",) else set(domains)
        rows = []
        for metric in metrics:
            metric_domains = selected.get(metric, ("全市场",))
            if requested is not None:
                metric_domains = tuple(d for d in metric_domains if d in requested)
            for domain in metric_domains:
                for period in periods:
                    if domain.startswith("barra"):
                        factor = list(c.style_masks)[int(domain[5:]) - 1]
                        mask = c.matrix(c.style_masks[factor], start, end)
                        top = _metric_value(metric, mask > 0, period, amount, float_mv, buy, sell, margin, self, start, end, history)
                        bottom = _metric_value(metric, mask < 0, period, amount, float_mv, buy, sell, margin, self, start, end, history)
                        value = top - bottom if np.isfinite(top) and np.isfinite(bottom) else np.nan
                    else:
                        value = _metric_value(metric, np.isfinite(c.domain_weights(start, end, domain)), period, amount, float_mv, buy, sell, margin, self, start, end, history)
                    rows.append({"指标": metric, "域": domain, "周期": period,
                                 "历史分位": value})
        raw = pd.DataFrame(rows)
        if raw.empty:
            return raw
        table = raw.pivot(index=["指标", "域"], columns="周期", values="历史分位")
        table.columns = [f"{int(x)}日历史分位" for x in table.columns]
        return table

class StyleMonitor:
    """薄编排层：Context数据读取 -> 指标函数 -> 表格/绘图；不负责数据获取实现。"""
    UNIVERSES = STYLE_UNIVERSES
    PERIODS = (1, 5, 10, 20)

    def __init__(self, context):
        self.context = context

    def snapshot(self, asof, periods=PERIODS, universes=UNIVERSES, industries=None, benchmark='zzfull', history=252, high_lookback=60, rotation_window=5, metrics=None, level_history=252):
        """返回Excel首Sheet的 market/industry/style 三张DataFrame。"""
        c = self.context
        end = c.position(asof)
        start = max(0, end - history - high_lookback - max(periods) - rotation_window - 260)
        dates = c.dates[start:end + 1]
        selected = {spec.label for spec in INDICATORS.values() if spec.module == 'style'} if metrics is None else set(metrics)
        market_labels = {spec.label for spec in INDICATORS.values() if spec.section == 'market'}
        style_labels = {spec.label for spec in INDICATORS.values() if spec.section == 'style'}
        need_returns = bool(selected.intersection({'相对收益', '新高新低差占比历史分位', '行业相对收益'} | style_labels))
        r = c.matrix('d_essentials/pct', start, end) / 100 if need_returns else None
        amount = c.matrix('d_essentials/amount', start, end) if selected.intersection({'成交额历史分位', '成交额占比历史分位'}) else None
        pe = c.matrix('fundamental/valuation/pe_ttm', start, end) if selected.intersection({'PE_TTM历史分位', 'PE收益差分位', '盈利-10年国债收益率'}) else None
        pb = c.matrix('fundamental/valuation/pb_lf', start, end) if selected.intersection({'PB_LF历史分位', 'PB收益差分位'}) else None
        industry_codes = c.matrix('industry/industry', start, end) if '行业轮动速度分位' in selected else None
        needed_weights = set(universes) if selected.intersection(market_labels) else set()
        if selected.intersection(style_labels | {'行业相对收益'}):
            needed_weights.add(None)
        if selected.intersection({'相对收益', '行业相对收益'}):
            needed_weights.add(benchmark)
        weights = {u: c.stock_weight_matrix(start, end, u) for u in needed_weights}
        benchmark_daily = (np.array([weighted_mean(r[t], weights[benchmark][t - 1]) if t else np.nan for t in range(len(dates))])
                           if selected.intersection({'相对收益', '行业相对收益'}) else None)
        bond = c.bond_10y(dates[0], dates[-1]).reindex(dates).ffill().to_numpy(dtype=float) if metrics is None or '盈利-10年国债收益率' in metrics else np.full(len(dates), np.nan)
        if np.isfinite(bond).any() and np.nanmedian(np.abs(bond)) > 1:
            bond = bond / 100.0
        columns = pd.MultiIndex.from_product([universes, periods], names=['universe', 'period'])
        market = pd.DataFrame(index=['点位252历史分位', '相对收益', '新高新低差占比历史分位', 'PE_TTM历史分位', 'PB_LF历史分位', '盈利-10年国债收益率', '成交额历史分位', '成交额占比历史分位'], columns=columns, dtype=float)
        for u in universes if selected.intersection(market_labels) else ():
            w = weights[u]
            members = np.isfinite(w) & (w > 0)
            close = c.index_close(u, dates[0], dates[-1]).reindex(dates).ffill().to_numpy(dtype=float) if metrics is None or '点位252历史分位' in metrics else None
            for h in periods:
                if (metrics is None or '点位252历史分位' in metrics) and h < len(dates):
                    market.loc['点位252历史分位', (u, h)] = level_percentile(close, level_history, h)
                if (metrics is None or '相对收益' in metrics) and h < len(dates):
                    market.loc['相对收益', (u, h)] = relative_return(r, w[-h - 1], benchmark_daily, h)
                if (metrics is None or '新高新低差占比历史分位' in metrics) and h < len(dates):
                    market.loc['新高新低差占比历史分位', (u, h)] = new_high_low_share(r, members, high_lookback, h, history)
                if (metrics is None or 'PE_TTM历史分位' in metrics) and h < len(dates):
                    market.loc['PE_TTM历史分位', (u, h)] = pe_percentile(pe, w, history, h)
                if (metrics is None or 'PB_LF历史分位' in metrics) and h < len(dates):
                    market.loc['PB_LF历史分位', (u, h)] = pb_percentile(pb, w, history, h)
                if (metrics is None or '盈利-10年国债收益率' in metrics) and h < len(dates):
                    market.loc['盈利-10年国债收益率', (u, h)] = earnings_bond_spread(pe, w, bond, h)
                if (metrics is None or '成交额历史分位' in metrics) and h < len(dates):
                    market.loc['成交额历史分位', (u, h)] = amount_percentile(amount, members, history, h)
                if (metrics is None or '成交额占比历史分位' in metrics) and h < len(dates):
                    market.loc['成交额占比历史分位', (u, h)] = amount_share_percentile(amount, members, history, h)
        if industries is None:
            industries = list(c.industry_map)
        industry = pd.DataFrame(index=industries, columns=periods, dtype=float)
        for ind in industries if metrics is None or '行业相对收益' in metrics else ():
            for h in periods:
                if h >= len(dates):
                    continue
                w = c.stock_weights(end - h, None, ind)
                industry.loc[ind, h] = industry_relative_return(r, w, benchmark_daily, h)
        all_w = weights.get(None)
        mask_names = {'大小盘收益差分位': 'size', '成长价值收益差分位': 'growth_value', '高低Beta收益差分位': 'beta'}
        masks = {name: c.matrix(c.style_masks[name], start, end) for label, name in mask_names.items() if label in selected}
        style = pd.DataFrame(index=['大小盘收益差分位', '成长价值收益差分位', 'PE收益差分位', 'PB收益差分位', '高低Beta收益差分位', '行业轮动速度分位'], columns=periods, dtype=float)
        industry_ids = [c.industry_map.get(x, x) for x in industries]
        sector_daily = np.full((len(dates), len(industry_ids)), np.nan)
        for j, ind in enumerate(industry_ids) if metrics is None or '行业轮动速度分位' in metrics else ():
            for t in range(1, len(dates)):
                w = np.where(industry_codes[t - 1] == ind, all_w[t - 1], np.nan)
                sector_daily[t, j] = weighted_mean(r[t], w)
        for h in periods:
            if (metrics is None or '大小盘收益差分位' in metrics) and h < len(dates):
                style.loc['大小盘收益差分位', h] = size_spread_percentile(masks['size'], r, all_w, h, history)
            if (metrics is None or '成长价值收益差分位' in metrics) and h < len(dates):
                style.loc['成长价值收益差分位', h] = growth_value_spread_percentile(masks['growth_value'], r, all_w, h, history)
            if (metrics is None or 'PE收益差分位' in metrics) and h < len(dates):
                style.loc['PE收益差分位', h] = INDICATORS['pe_return_spread'].load()(pe, r, all_w, h, history)
            if (metrics is None or 'PB收益差分位' in metrics) and h < len(dates):
                style.loc['PB收益差分位', h] = INDICATORS['pb_return_spread'].load()(pb, r, all_w, h, history)
            if (metrics is None or '高低Beta收益差分位' in metrics) and h < len(dates):
                style.loc['高低Beta收益差分位', h] = beta_spread_percentile(masks['beta'], r, all_w, h, history)
            if (metrics is None or '行业轮动速度分位' in metrics) and h < len(dates):
                style.loc['行业轮动速度分位', h] = rotation_speed_percentile(sector_daily, rotation_window, history, h)
        return {'market': market, 'industry': industry, 'style': style}

    @staticmethod
    def table(result, section='market', decimals=4):
        """交互界面的直接表数据，原始结果仍可用于进一步前端渲染。"""
        return result[section].round(decimals)

    @staticmethod
    def plot(result, section='style', figsize=(10, 5)):
        """轻量绘图预览，不引入Dashboard/Web依赖。"""
        import matplotlib.pyplot as plt
        data = result[section]
        fig, ax = plt.subplots(figsize=figsize)
        if section == 'market':
            data.T.plot(ax=ax, marker='o')
        else:
            data.T.plot(ax=ax, marker='o')
        ax.set_title(section)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        return (fig, ax)

class RiskMonitor:
    DEFAULT_FIELDS = RISK_FIELDS
    def __init__(self, context, field_map=None):
        self.context = context
        self.fields = {**self.DEFAULT_FIELDS, **(field_map or {})}

    def option_volatility(self, asof, start, end, index_codes=("000300.SH", "000852.SH"), *,
                          surface_field="品种隐含波动率曲面(%)",
                          surface_options="Period=D;期限={tenor}天;价值状态={moneyness}%",
                          surface_scale=0.01, points=((30,100),)):
        """固定期限/价值状态曲面，列为(code, tenor, moneyness)，值为年化小数。

        价值状态方向须由 Wind 元数据确认；这里原样传入，不推断看涨/看跌。
        缺失数据保留 NaN，不跨日期前向填充。asof 限制查询截止日期。
        字段与参数可按 Wind 生成的 API 调用覆盖；surface_scale 将百分数转为小数。
        """
        end = min(pd.Timestamp(asof), pd.Timestamp(end))
        out = {}
        for code in index_codes:
            for tenor, moneyness in points:
                options = surface_options.format(tenor=tenor, moneyness=moneyness)
                x = self.context.wsd(code, surface_field, start, end, options=options)
                x = pd.to_numeric(x, errors='coerce') * surface_scale
                x = x.where(np.isfinite(x) & (x >= 0))
                out[(code,tenor,moneyness)] = x
        return pd.DataFrame(out)

    @staticmethod
    def _volatility_metrics(surface, close):
        """IV30 与方差风险溢价代理；RV5/20/60 在域指标中统一计算。

        RV 按交易日、简单收益样本标准差计算，统一年化小数。
        代理的 RV30 为30个交易日，与 IV30 的日历期限并非严格匹配。
        """
        atm = surface[(30,100)]
        return {
            '指数隐含波动率IV30': INDICATORS['index_iv30'].load()(atm),
            '波动率风险溢价代理IV30²-RV30²': INDICATORS['variance_risk_premium'].load()(atm, close),
        }
    def snapshot(self,asof,history=252,periods=PERIODS,domains=("全市场",), micro_result=None, metrics=None, option_settings=None):
        c=self.context; 
        end=c.position(asof); 
        start=max(0,end-history-260); 
        r=c.matrix(self.fields["returns"],start,end)
        need_limits = metrics is None or bool(set(metrics).intersection(('市场下行共振', '跌停扩散比例')))
        floor=c.matrix(self.fields["hit_floor"],start,end) if need_limits else np.full_like(r, np.nan)
        trade=c.matrix(self.fields["has_trade"],start,end)
        price_floor=c.matrix(self.fields["price_floor"],start,end) if need_limits else np.full_like(r, np.nan)
        eligible = np.isfinite(price_floor) & (price_floor > 0)
        dates = pd.DatetimeIndex(c.dates[start:end+1])
        rows=[]
        for domain in domains:
            w=c.domain_weights(start,end,domain)
            member = np.isfinite(w) & (w > 0)
            active = member & np.isfinite(trade) & (trade > 0)
            dr = np.where(active & np.isfinite(r), r, np.nan)
            domain_trade = np.where(member, trade, 0)
            # Domain benchmark uses current returns with prior-day weights.
            ir = np.full(len(r), np.nan)
            ir[1:] = [weighted_mean(ret, weight)
                      for ret, weight in zip(_pct(dr[1:]), w[:-1])]
            correlation = _correlation_series(dr, INDICATORS['average_correlation'].parameters.get('window', 60)) if metrics is None or "平均收益相关性" in metrics else np.array([])
            downside = _correlation_series(dr, INDICATORS['downside_correlation'].parameters.get('window', 120), downside=True, market_returns=ir) if metrics is None or "下跌条件相关性" in metrics else np.array([])
            pca = _pca_series(dr, INDICATORS['pca_explained_ratio'].parameters.get('window', 60)) if metrics is None or "首主成分解释率" in metrics else np.array([])
            for p in periods:
                for n in ('横截面收益离散度', '平均收益相关性', '下跌条件相关性', '首主成分解释率',
                          '市场下行共振', '跌停扩散比例', '最差滚动5日收益'):
                    if metrics is not None and n not in metrics:
                        continue
                    if n == '横截面收益离散度':
                        v, h = cross_section_dispersion(dr, p, history)
                    elif n in ('平均收益相关性', '下跌条件相关性', '首主成分解释率'):
                        series = {'平均收益相关性': correlation, '下跌条件相关性': downside, '首主成分解释率': pca}[n]
                        v, h = _summarize(series, p, history)
                    elif n == '市场下行共振':
                        v, h = downside_resonance(dr, floor, domain_trade, p, history, eligible)
                    elif n == '跌停扩散比例':
                        v, h = limit_down_ratio(floor, domain_trade, p, history, eligible)
                    else:
                        v, h = worst_rolling_loss(ir * 100, history, 5, history)
                    rows.append({'指标': n, '域': domain, '周期': p, '数值': v, '历史分位': h})
            for key in ('rv5', 'rv20', 'rv60', 'rv_ratio_5_20', 'rv_ratio_20_60'):
                spec = INDICATORS[key]
                if metrics is not None and spec.label not in metrics:
                    continue
                value, rank = spec.load()(ir if 'ratio' in key else ir * 100, history=history)
                rows.append({"指标":spec.label,"域":domain,"周期":1,
                             "数值":value,"历史分位":rank})
        # Use constant-tenor surface points and official index close returns.
        for code, label in (
            ("000300.SH", "沪深300"),
            ("000852.SH", "中证1000"),
        ) if metrics is None or set(metrics).intersection(("指数隐含波动率IV30", "波动率风险溢价代理IV30²-RV30²")) else ():
            vol = self.option_volatility(asof, c.dates[start], c.dates[end], (code,), **(option_settings or {}))
            surface = vol[code].reindex(dates)
            close = c.wsd(code, 'close', c.dates[max(0, start-65)], c.dates[end])
            volatility_metrics = self._volatility_metrics(surface, close)
            for name, series in volatility_metrics.items():
                if metrics is not None and name not in metrics:
                    continue
                x = series.reindex(dates)
                # Missing as-of observations stay unavailable, rather than
                # presenting an older quote as today's value.
                rows.append({"指标": name, "域": label, "周期": 1,
                             "数值": float(x.iloc[-1]),
                             "历史分位": _p(x.to_numpy(), history)})
        if micro_result is not None and (metrics is None or "流动性压力历史分位" in metrics):
            required = ("相对价差分位", "深度下降分位", "Amihud分位")
            v, h = np.nan, np.nan
            if all(k in micro_result for k in required):
                v,h=liquidity_pressure_percentile(*(micro_result[k] for k in required), history)
            rows.append({"指标":"流动性压力历史分位","域":"全市场","周期":1,"数值":v,"历史分位":h})
        elif metrics is not None and "流动性压力历史分位" in metrics:
            # 未注入三个压力分量时显式保留缺失，不将旧值误当成今日计算结果。
            rows.append({"指标":"流动性压力历史分位","域":"全市场","周期":1,"数值":np.nan,"历史分位":np.nan})
        return pd.DataFrame(rows).set_index(["指标","域","周期"]) if rows else pd.DataFrame()

    @staticmethod
    def table(result, decimals=6):
        return result.round(decimals)


def normalize_snapshot(module, result, specs):
    """将原五类面板统一为 name/domain/period/value/percentile 长表。"""
    rows = []
    for spec in specs:
        if module == 'style':
            table = result[spec.section]
            if spec.section == 'market':
                for domain, period in table.columns:
                    value = table.loc[spec.label, (domain, period)]
                    field = 'percentile' if '分位' in spec.label else 'value'
                    rows.append(dict(name=spec.name, module=module, domain=domain, period=int(period), **{field: value}))
            elif spec.section == 'industry':
                for domain in table.index:
                    for period in table.columns:
                        rows.append(dict(name=spec.name, module=module, domain=domain, period=int(period), value=table.loc[domain, period]))
            else:
                for period in table.columns:
                    rows.append(dict(name=spec.name, module=module, domain='全市场', period=int(period), percentile=table.loc[spec.label, period]))
        elif result.empty or spec.label not in result.index.get_level_values(0):
            continue
        elif module in ('micro', 'moneyflow'):
            for domain, values in result.loc[spec.label].iterrows():
                for column, value in values.items():
                    period = int(column.removesuffix('日历史分位'))
                    rows.append(dict(name=spec.name, module=module, domain=domain, period=period, percentile=value))
        elif module == 'macro':
            for period, values in result.loc[spec.label].iterrows():
                lag = {'当前值': 0, '5日变化': 5, '20日变化': 20}[period]
                rows.append(dict(name=spec.name, module=module, domain='全市场', period=lag,
                                 value=values['数值'], percentile=values['历史分位']))
        else:
            for (domain, period), values in result.loc[spec.label].iterrows():
                rows.append(dict(name=spec.name, module=module, domain=domain, period=int(period),
                                 value=values['数值'], percentile=values['历史分位']))
    frame = pd.DataFrame(rows).reindex(columns=['name', 'module', 'domain', 'period', 'value', 'percentile'])
    if not frame.empty and frame.duplicated(['name', 'domain', 'period']).any():
        raise ValueError('监控结果存在重复的指标/域/周期')
    return frame


def calculate_daily(date, context, *, names=None, modules=None, config=None):
    """按注册表分组计算当日指标，每个模块一次编排，公式来自独立 vals 脚本。"""
    config = config or MonitorConfig()
    specs = select_indicators(names, modules)
    context.begin_day(date)
    frames = []
    for module in dict.fromkeys(spec.module for spec in specs):
        selected = tuple(spec for spec in specs if spec.module == module)
        labels = tuple(spec.label for spec in selected)
        options = dict(config.module_options.get(module, {}))
        common = dict(history=config.history, periods=config.periods, metrics=labels)
        if module == 'macro':
            result = MacroMonitor(context).snapshot(date, **{**common, **options})
        elif module == 'micro':
            result = MicroMonitor(context).snapshot(date, **{**common, 'domains': config.domains,
                                                            'expand_domains': False, 'show_progress': False, **options})
        elif module == 'moneyflow':
            result = MoneyFlowMonitor(context).snapshot(date, **{**common, 'domains': config.domains,
                                                                'expand_domains': False, **options})
        elif module == 'style':
            result = StyleMonitor(context).snapshot(date, **{**common, 'universes': config.universes,
                                                            'industries': config.industries, 'benchmark': config.benchmark,
                                                            'high_lookback': config.high_lookback,
                                                            'level_history': config.level_history,
                                                            'rotation_window': config.rotation_window, **options})
        else:
            result = RiskMonitor(context).snapshot(date, **{**common, 'domains': config.domains, **options})
        frames.append(normalize_snapshot(module, result, selected))
    if not frames:
        return pd.DataFrame(columns=['name', 'module', 'domain', 'period', 'value', 'percentile'])
    return pd.concat(frames, ignore_index=True)


def metric_path(root, name, domain, period, field, config=None):
    """配置作为列定义；域名编码为安全目录，每个数值文件固定一列。"""
    from urllib.parse import quote
    config = config or MonitorConfig()
    spec = INDICATORS[name]
    if field not in ('value', 'percentile') or int(period) < 0:
        raise ValueError('invalid field or period')
    domain = str(domain)
    if not domain or domain in ('.', '..'):
        raise ValueError('invalid domain')
    relative = Path(config.output_folder)
    if relative.is_absolute() or '..' in relative.parts or relative.parts[:2] != ('median_variables', 'monitor'):
        raise ValueError('output_folder 必须位于 median_variables/monitor 下，以保持轴维护兼容')
    return Path(root) / relative / spec.module / name / quote(domain, safe='-_') / f'p{int(period)}' / f'{field}.bin'


def write_daily(date, result, root, dates, *, config=None):
    """按共享日期轴覆写目标行，预留行填NaN；不自动改变或重排日期轴。"""
    from v2.UpdateData.utils import date_index, ensure_memmap
    row = date_index(date, dates)
    if np.isinf(result[['value', 'percentile']].to_numpy(dtype=float)).any():
        raise ValueError('监控结果含无穷值，未写入文件')
    for record in result.itertuples(index=False):
        # 只保存该类指标实际提供的字段；缺失观测仍写NaN，覆盖之前的旧值。
        percentile_only = record.module in ('micro', 'moneyflow') or (
            record.module == 'style' and ('分位' in INDICATORS[record.name].label))
        fields = ('percentile',) if percentile_only else (
            ('value', 'percentile') if record.module in ('macro', 'risk') else ('value',))
        for field in fields:
            value = float(getattr(record, field))
            if np.isinf(value):
                raise ValueError(f'{record.name} 输出无穷值')
            path = metric_path(root, record.name, record.domain, record.period, field, config)
            array = ensure_memmap(path, (len(dates), 1), dtype=np.float32)
            array[row, 0] = value
            array.flush()
            del array


def update_daily(date, root, *, context=None, config=None, names=None, modules=None):
    """每日统一更新入口；可注入同一个Context供历史循环复用。"""
    own = context is None
    context = context or MonitorContext(root=root)
    try:
        # 调用前先验证日期与轴，避免计算完才发现日期不在轴上。
        from v2.UpdateData.utils import date_index
        dates = context.data.axis.full_dates
        date_index(date, dates)
        result = calculate_daily(date, context, names=names, modules=modules, config=config)
        write_daily(date, result, root, dates, config=config)
        return result
    finally:
        if own:
            context.close()


def read_indicator(root, name, domain='全市场', period=1, field='percentile', *, config=None):
    """按本地共享日期轴读取单列监控序列，不暴露预留NaT行。"""
    dates = np.load(Path(root) / 'axis' / 'dates.npy', allow_pickle=False)
    path = metric_path(root, name, domain, period, field, config)
    if path.stat().st_size != len(dates) * np.dtype(np.float32).itemsize:
        raise ValueError(f'文件大小与日期轴不一致: {path}')
    array = np.memmap(path, dtype=np.float32, mode='r', shape=(len(dates), 1))
    valid = ~np.isnat(dates)
    result = pd.Series(np.array(array[valid, 0]), index=pd.DatetimeIndex(dates[valid]), name=name)
    del array
    return result


if __name__ == '__main__':
    from v2.UpdateData.config import ROOT

    # 在IDE中直接修改参数。设为 None 可更新全部已注册指标（需要全部数据源）。
    names = ('growth_value_spread_percentile',)
    start_date = '2018-01-01'
    end_date = '2026-09-30'
    config = MonitorConfig(history=252, periods=(1, 5, 10, 20))
    context = MonitorContext(root=ROOT)
    try:
        for date in context.dates:
            if pd.Timestamp(start_date) <= date <= pd.Timestamp(end_date):
                result = update_daily(date, ROOT, context=context, config=config, names=names)
                print(f'[OK] {date.date()}: {len(result)} 条监控数值', flush=True)
    finally:
        context.close()
