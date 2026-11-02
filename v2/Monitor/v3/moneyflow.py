"""资金流指标；每个函数接收一个 market/index/industry/barra 域并返回日序列。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import DOMAIN_MAP, get_weight, select_domain_series


BUY_FIELDS = {
    "large": "d_moneyflow/buy_lg_amount", "medium": "d_moneyflow/buy_mid_amount",
    "small": "d_moneyflow/buy_sm_amount", "extra": "d_moneyflow/buy_elg_amount",
}
SELL_FIELDS = {
    "large": "d_moneyflow/sell_lg_amount", "medium": "d_moneyflow/sell_mid_amount",
    "small": "d_moneyflow/sell_sm_amount", "extra": "d_moneyflow/sell_elg_amount",
}


def _amount(context, end):
    parts = []
    for size in BUY_FIELDS:
        buy = context.read("local", BUY_FIELDS[size], 0, end, kind="matrix")
        sell = context.read("local", SELL_FIELDS[size], 0, end, kind="matrix")
        parts.append(buy + sell)
    stacked = np.stack(parts)
    valid = np.isfinite(stacked)
    result = np.sum(np.where(valid, stacked, 0), axis=0)
    result[np.all(~valid, axis=0)] = np.nan
    return result


def _members(context, domain, end):
    return get_weight(context, *DOMAIN_MAP[domain], end) > 0, context.dates[:end + 1]


def amount_change(context, date, domain, baseline=20, period=1):
    """域内近 period 日成交额均值相对前置 baseline 日均额的变化率。"""
    end = context.get_position(date)
    masks, dates = _members(context, domain, end)
    amount = _amount(context, end)
    results = []
    for mask in masks:
        group = select_domain_series(amount, (mask,), dates)
        current = group.rolling(period, min_periods=period).mean()
        previous = group.rolling(baseline, min_periods=baseline).mean().shift(period)
        results.append(current.div(previous.where(previous > 0)) - 1)
    series = results[0] if len(results) == 1 else results[0] - results[1]
    return series.rename("成交额变化")


def top_share(context, date, domain, fraction=0.1):
    """每日成交额最高 fraction 股票的成交额占域内总成交额比例。"""
    end = context.get_position(date)
    masks, dates = _members(context, domain, end)
    amount = _amount(context, end)
    outputs = []
    for mask in masks:
        values = np.where(mask & np.isfinite(amount) & (amount >= 0), amount, np.nan)
        result = np.full(len(dates), np.nan)
        for i, row in enumerate(values):
            valid = row[np.isfinite(row)]
            if valid.size:
                count = max(1, int(np.ceil(valid.size * fraction)))
                total = valid.sum()
                if total > 0:
                    result[i] = np.sort(valid)[-count:].sum() / total
        outputs.append(pd.Series(result, index=dates))
    series = outputs[0] if len(outputs) == 1 else outputs[0] - outputs[1]
    return series.rename("头部成交额占比")


def turnover(context, date, domain):
    """域内成交额除以前一交易日域内自由流通市值。"""
    end = context.get_position(date)
    masks, dates = _members(context, domain, end)
    amount = _amount(context, end)
    cap = context.read("local", "d_essentials/circ_mv", 0, end, kind="matrix")
    outputs = []
    for mask in masks:
        daily_amount = np.nansum(np.where(mask, amount, np.nan), axis=1)
        daily_cap = np.nansum(np.where(mask, cap, np.nan), axis=1)
        daily_amount[np.all(~np.isfinite(np.where(mask, amount, np.nan)), axis=1)] = np.nan
        daily_cap[np.all(~np.isfinite(np.where(mask, cap, np.nan)), axis=1)] = np.nan
        outputs.append(pd.Series(daily_amount, index=dates).div(pd.Series(daily_cap, index=dates).shift(1)))
    series = outputs[0] if len(outputs) == 1 else outputs[0] - outputs[1]
    return series.rename("自由流通换手率")


def active_buy_ratio(context, date, domain):
    """域内各类主动买入金额占主动买卖总金额的比例。"""
    end = context.get_position(date)
    masks, dates = _members(context, domain, end)
    buy_fields = np.stack([context.read("local", field, 0, end, kind="matrix") for field in BUY_FIELDS.values()])
    sell_fields = np.stack([context.read("local", field, 0, end, kind="matrix") for field in SELL_FIELDS.values()])
    buy = np.nansum(buy_fields, axis=0)
    sell = np.nansum(sell_fields, axis=0)
    buy[np.all(~np.isfinite(buy_fields), axis=0)] = np.nan
    sell[np.all(~np.isfinite(sell_fields), axis=0)] = np.nan
    outputs = []
    for mask in masks:
        buy_sum = np.nansum(np.where(mask, buy, np.nan), axis=1)
        total = np.nansum(np.where(mask, buy + sell, np.nan), axis=1)
        ratio = np.divide(buy_sum, total, out=np.full(len(dates), np.nan), where=total > 0)
        outputs.append(pd.Series(ratio, index=dates))
    series = outputs[0] if len(outputs) == 1 else outputs[0] - outputs[1]
    return series.rename("主动买入占比")


def balance_change(context, date, domain=None, period=1, code="M0075990"):
    """融资余额 period 日变化率；EDB序列按交易日向前填充。"""
    end = pd.Timestamp(date).normalize()
    start = end - pd.DateOffset(years=3)
    raw = context.read("edb", code, start, end, kind="series")
    dates = context.dates[(context.dates >= start) & (context.dates <= end)]
    if raw.empty or dates.empty:
        return pd.Series(index=dates, dtype=float, name="融资余额变化率")
    index = raw.index.union(dates).sort_values()
    values = raw.reindex(index).ffill().reindex(dates)
    return (values.div(values.shift(period)) - 1).rename("融资余额变化率")


def category_share(context, date, domain, size="large"):
    """指定成交额类别可用股票的金额占域内总成交额比例。"""
    if size not in BUY_FIELDS:
        raise ValueError(f"未知成交额类别: {size}")
    end = context.get_position(date)
    masks, dates = _members(context, domain, end)
    amount = _amount(context, end)
    category_available = np.isfinite(context.read("local", BUY_FIELDS[size], 0, end, kind="matrix"))
    outputs = []
    for mask in masks:
        total_values = np.where(mask, amount, np.nan)
        category_values = np.where(mask & category_available, amount, np.nan)
        total = np.nansum(total_values, axis=1)
        selected = np.nansum(category_values, axis=1)
        valid_total = np.any(np.isfinite(total_values), axis=1)
        valid_category = np.any(np.isfinite(category_values), axis=1)
        ratio = np.divide(selected, total, out=np.full(len(dates), np.nan), where=total > 0)
        ratio[~valid_total | ~valid_category] = np.nan
        outputs.append(pd.Series(ratio, index=dates))
    series = outputs[0] if len(outputs) == 1 else outputs[0] - outputs[1]
    return series.rename(f"{size}单成交额占比")


def large_trade_share(context, date, domain):
    """大单成交额占域内成交额比例。"""
    return category_share(context, date, domain, "large")


def medium_trade_share(context, date, domain):
    """中单成交额占域内成交额比例。"""
    return category_share(context, date, domain, "medium")


def small_trade_share(context, date, domain):
    """小单成交额占域内成交额比例。"""
    return category_share(context, date, domain, "small")
