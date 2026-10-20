"""资金交易监控指标。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import polars as pl

try:
    from .utils import historical_percentile, safe_divide
    from .context import MarketContext
except ImportError:  # 直接以脚本路径运行
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from v2.Monitor.utils import historical_percentile, safe_divide
    from v2.Monitor.context import MarketContext

PERIODS = (1, 5, 10, 20)
INDEX_DOMAINS = ("沪深300", "中证500", "中证1000", "中证2000")
STYLE_DOMAINS = tuple(f"barra{i}" for i in range(1, 11))


def _sum_matrices(matrices):
    """Sum matrices elementwise while retaining all-NaN cells as NaN."""
    values = np.asarray(matrices, dtype=float)
    result = np.nansum(values, axis=0)
    result[np.all(~np.isfinite(values), axis=0)] = np.nan
    return result


def _metric_value(metric, member, period, amount, float_mv, buy, sell,
                  margin, monitor, start, end, history):
    a = np.where(member, amount, np.nan)
    if metric == "成交额变化": return amount_change(a, history, period)
    if metric == "top10%成交额占比": return top_share(a, history, period)
    if metric == "自由流通换手率": return turnover(a, np.where(member, float_mv, np.nan), history, period)
    if metric == "主动买入占比": return active_buy_ratio(np.where(member, buy, np.nan), np.where(member, sell, np.nan), history, period)
    if metric in {"大单成交占比", "中单成交占比", "小单成交占比"}:
        key = {"大单成交占比": "large", "中单成交占比": "medium", "小单成交占比": "small"}[metric]
        return category_share(a, np.isfinite(monitor._matrix(f"{key}_buy", start, end)), history, period)
    if metric == "融资余额变化率": return balance_change(margin, history, period)
    raise ValueError(f"未注册资金流指标: {metric}")


class MoneyFlowContext(MarketContext):
    """资金模块取数上下文；主买主卖从预处理 L2 成交表实时聚合。"""

    def __init__(self, *args, l2_root=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.l2_root = Path(l2_root or Path(self.data.asset_root) / "l2")


def _percentile(values, history=252, period=1):
    values = np.asarray(values, dtype=float)
    smooth = np.array([np.nanmean(values[i-period+1:i+1]) if i >= period-1 else np.nan for i in range(len(values))])
    return historical_percentile(smooth[-history:])


def amount_change(amount, history=252, period=1, baseline=20):
    """近 h 日均成交额相对紧邻前 20 日基准均额的历史分位。"""
    amount = np.nansum(np.asarray(amount, float), axis=1)
    values = []
    for i in range(len(amount)):
        if i < baseline + period - 1:
            values.append(np.nan)
            continue
        current = np.nanmean(amount[i-period+1:i+1])
        base = np.nanmean(amount[i-period-baseline+1:i-period+1])
        values.append(current / base - 1 if base > 0 else np.nan)
    return _percentile(values, history)


def top_share(amount, history=252, period=1, fraction=0.1):
    """每日成交额最高 fraction 股票的成交额占比历史分位。"""
    values = []
    for row in np.asarray(amount, float):
        valid = row[np.isfinite(row) & (row >= 0)]
        n = max(1, int(np.ceil(len(valid) * fraction))) if len(valid) else 0
        values.append(np.sum(np.sort(valid)[-n:]) / np.sum(valid) if n and np.sum(valid) > 0 else np.nan)
    return _percentile(values, history, period)


def turnover(amount, float_mv, history=252, period=1):
    """组成交额/前一日自由流通市值的历史分位。"""
    amount = np.nansum(np.asarray(amount, float), axis=1)
    float_mv = np.nansum(np.asarray(float_mv, float), axis=1)
    return _percentile(np.r_[np.nan, safe_divide(amount[1:], float_mv[:-1])], history, period)


def active_buy_ratio(buy, sell, history=252, period=1):
    buy, sell = np.asarray(buy, float), np.asarray(sell, float)
    return _percentile(safe_divide(np.nansum(buy, axis=1), np.nansum(buy + sell, axis=1)), history, period)


def balance_change(balance, history=252, period=1):
    balance = np.asarray(balance, float)
    if balance.ndim > 1:
        balance = np.nansum(balance, axis=1)
    return _percentile(np.r_[np.nan, safe_divide(balance[period:], balance[:-period]) - 1], history, 1)


def category_share(amount, category, history=252, period=1):
    amount, category = np.asarray(amount, float), np.asarray(category, bool)
    values = safe_divide(np.nansum(np.where(category, amount, np.nan), axis=1), np.nansum(amount, axis=1))
    values[~category.any(axis=1)] = np.nan
    return _percentile(values, history, period)


def threshold_conclusion(value, thresholds=(0.1, 0.9), labels=("偏低", "偏高", "中性")):
    """将历史分位映射为简短结论。"""
    if not np.isfinite(value):
        return "数据不足"
    low, high = thresholds
    return labels[0] if value <= low else labels[1] if value >= high else labels[2]


def trend_conclusion(values):
    """根据1/5/10/20日分位给出域内时间趋势结论。"""
    x = np.asarray(values, dtype=float)
    if not np.isfinite(x).any():
        return "数据不足"
    if np.isfinite(x).sum() < 2:
        return "数据不足"
    slope = np.polyfit(np.arange(len(x))[np.isfinite(x)], x[np.isfinite(x)], 1)[0]
    if slope >= 0.08:
        return "短期走强"
    if slope <= -0.08:
        return "短期走弱"
    return "时间趋势平稳"


DEFAULT_THRESHOLDS = {
    "成交额变化": (0.1, 0.9), "top10%成交额占比": (0.1, 0.9),
    "自由流通换手率": (0.1, 0.9),
}


class MoneyFlowMonitor:
    """资金交易指标编排器，返回指标×域长表。"""

    DEFAULT_FIELDS = {
        # Turnover amount is stored intraday in m_essentials; MarketContext
        # aggregates its minute axis to one daily row when reading it.
        "amount": "m_essentials/amount", "float_mv": "d_essentials/circ_mv",
        "large_buy": "d_moneyflow/buy_lg_amount", "large_sell": "d_moneyflow/sell_lg_amount",
        "medium_buy": "d_moneyflow/buy_mid_amount", "medium_sell": "d_moneyflow/sell_mid_amount",
        "small_buy": "d_moneyflow/buy_sm_amount", "small_sell": "d_moneyflow/sell_sm_amount",
        "extra_buy": "d_moneyflow/buy_elg_amount", "extra_sell": "d_moneyflow/sell_elg_amount",
        "margin_balance": "d_moneyflow/margin_balance",
    }

    def __init__(self, context, field_map=None):
        self.context = context
        self.fields = {**self.DEFAULT_FIELDS, **(field_map or {})}

    def _matrix(self, name, start, end):
        return self.context.matrix(self.fields[name], start, end)

    def snapshot(self, asof, domains=("全市场",), history=252, periods=PERIODS,
                 metrics=("主动买入占比", "大单成交占比", "中单成交占比", "小单成交占比",
                          "成交额变化", "top10%成交额占比", "自由流通换手率", "融资余额变化率")):
        c = self.context
        end = c.position(asof)
        start = max(0, end - history - max(periods) - 30)
        amount = _sum_matrices(
            [self._matrix(f"{size}_buy", start, end)
             + self._matrix(f"{size}_sell", start, end)
             for size in ("large", "medium", "small", "extra")],
        )
        float_mv = self._matrix("float_mv", start, end)
        margin = c.edb(
            c.edb_codes.get("margin_balance", "M0075990"),
            c.dates[start], c.dates[end],
        ).reindex(c.dates[start:end + 1]).ffill().to_numpy(dtype=float)
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
        requested = set(domains) if domains != ("全市场",) else None
        rows = []
        for metric in metrics:
            metric_domains = selected.get(metric, ("全市场",))
            if requested is not None:
                metric_domains = tuple(d for d in metric_domains if d in requested)
            for domain in metric_domains:
                for period in periods:
                    if domain.startswith("barra"):
                        factor = list(c.style_masks)[int(domain[5:]) - 1]
                        mask = self._matrix(c.style_masks[factor], start, end)
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


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    context = MoneyFlowContext()
    try:
        result = MoneyFlowMonitor(context).snapshot("2026-09-30")
        print(result.to_string(index=False))
    finally:
        context.close()
