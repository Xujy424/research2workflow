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


def _sum_matrices(matrices):
    """Sum matrices elementwise while retaining all-NaN cells as NaN."""
    values = np.asarray(matrices, dtype=float)
    result = np.nansum(values, axis=0)
    result[np.all(~np.isfinite(values), axis=0)] = np.nan
    return result


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
    balance = np.nansum(np.asarray(balance, float), axis=1)
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
            axis=0,
        )
        float_mv = self._matrix("float_mv", start, end)
        buy = _sum_matrices(
            [self._matrix(f"{size}_buy", start, end)
             for size in ("large", "medium", "small", "extra")],
            axis=0,
        )
        sell = _sum_matrices(
            [self._matrix(f"{size}_sell", start, end)
             for size in ("large", "medium", "small", "extra")],
            axis=0,
        )
        rows = []
        for domain in domains:
            weights = c.stock_weight_matrix(start, end, None if domain == "全市场" else domain)
            member = np.isfinite(weights)
            for metric in metrics:
                for period in periods:
                    a = np.where(member, amount, np.nan)
                    if metric == "成交额变化":
                        value = amount_change(a, history, period)
                    elif metric == "top10%成交额占比":
                        value = top_share(a, history, period)
                    elif metric == "自由流通换手率":
                        value = turnover(a, np.where(member, float_mv, np.nan), history, period)
                    elif metric == "主动买入占比":
                        value = active_buy_ratio(np.where(member, buy, np.nan), np.where(member, sell, np.nan), history, period)
                    elif metric in {"大单成交占比", "中单成交占比", "小单成交占比"}:
                        key = {"大单成交占比": "large", "中单成交占比": "medium", "小单成交占比": "small"}[metric]
                        value = category_share(a, np.isfinite(self._matrix(f"{key}_buy", start, end)), history, period)
                    elif metric == "融资余额变化率":
                        value = balance_change(np.where(member, self._matrix("margin_balance", start, end), np.nan), history, period)
                    else:
                        raise ValueError(f"未注册资金流指标: {metric}")
                    threshold = DEFAULT_THRESHOLDS.get(metric, (0.1, 0.9))
                    rows.append({"指标": metric, "域": domain, "周期": period,
                                 "历史分位": value, "阈值": threshold})
        raw = pd.DataFrame(rows)
        if raw.empty:
            return raw
        table = raw.pivot(index=["指标", "域"], columns="周期", values="历史分位")
        table.columns = [f"{int(x)}日历史分位" for x in table.columns]
        table["阈值"] = raw.drop_duplicates(["指标", "域"]).set_index(["指标", "域"])["阈值"]
        percentile_columns = [f"{p}日历史分位" for p in periods]
        table["时间趋势结论"] = table[percentile_columns].apply(trend_conclusion, axis=1)
        table["截面结构结论"] = ""
        for metric, group in table.groupby(level="指标", sort=False):
            latest = group["1日历史分位"]
            valid = latest.dropna()
            if len(valid) < 2:
                conclusion = "域数据不足"
            else:
                high, low = valid.idxmax()[1], valid.idxmin()[1]
                conclusion = f"高位域：{high}；低位域：{low}"
            table.loc[(metric,), "截面结构结论"] = conclusion
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
