"""资金流指标：每个函数直接从context读数，返回域内日度Series。"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

if __package__:
    from .context import MonitorContext
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from v2.Monitor.v3.context import MonitorContext
from v2.UpdateData.config import ROOT
from v2.Monitor.v3.common import weighted_series


def amount_change(context, date, domain, name, baseline=20, period=1):
    """近period日域内成交额均值/此前baseline日均值-1；风格取正组减负组。"""
    end = context.get_position(date)
    dates = context.dates[:end + 1]
    amount = context.read('local', 'd_essentials/amount', 0, end, kind='matrix')
    weights = context.get_weight(domain, name, end)
    outputs = []
    for mask in ((weights > 0, weights < 0) if domain == 'style' else (weights > 0,)):
        valid = mask & np.isfinite(amount)
        total = pd.Series(np.where(valid, amount, 0).sum(axis=1), index=dates)
        total[~valid.any(axis=1)] = np.nan
        previous = total.shift(period).rolling(baseline).mean()
        outputs.append(total.rolling(period).mean() / previous.where(previous > 0) - 1)
    return (outputs[0] if len(outputs) == 1 else outputs[0] - outputs[1]).rename('成交额变化')


def top_share(context, date, domain, name, fraction=0.1):
    """域内成交额最高fraction股票的成交额占比，不计算风格正负组差值。"""
    end = context.get_position(date)
    dates = context.dates[:end + 1]
    amount = context.read('local', 'd_essentials/amount', 0, end, kind='matrix')
    weights = context.get_weight(domain, name, end)
    members = np.isfinite(weights) & (weights != 0)
    result = np.full(len(dates), np.nan)
    for i, row in enumerate(amount):
        values = row[members[i] & np.isfinite(row) & (row >= 0)]
        count = max(1, int(np.ceil(len(values) * fraction)))
        total = values.sum()
        if values.size and total > 0:
            result[i] = np.partition(values, -count)[-count:].sum() / total
    return pd.Series(result, index=dates, name='头部成交额占比')


def turnover(context, date, domain, name):
    """域内成交额/前日流通市值，小数比例；当前成员用于分子分母，保持样本一致。"""
    end = context.get_position(date)
    dates = context.dates[:end + 1]
    amount = context.read('local', 'd_essentials/amount', 0, end, kind='matrix')
    cap = context.read('local', 'd_essentials/circ_mv', 0, end, kind='matrix')
    cap = np.vstack((np.full_like(cap[:1], np.nan), cap[:-1]))
    weights = context.get_weight(domain, name, end)
    outputs = []
    for mask in ((weights > 0, weights < 0) if domain == 'style' else (weights > 0,)):
        valid = mask & np.isfinite(amount) & np.isfinite(cap) & (cap > 0)
        ratio = np.divide(amount, cap, out=np.full_like(amount, np.nan), where=valid)
        outputs.append(weighted_series(ratio, np.where(valid, cap, 0), dates))
    return (outputs[0] if len(outputs) == 1 else outputs[0] - outputs[1]).rename('流通市值换手率')


def balance_change(context, date, domain, name, period=1, code='M0075990'):
    """原始日频融资余额相隔period条观测的变化率，不填充、不细分域。"""
    end = context.get_position(date)
    dates = context.dates[:end + 1]
    raw = context.read('edb', code, dates[0] - pd.DateOffset(days=30), dates[-1], kind='series')
    return (raw / raw.shift(period) - 1).loc[dates[0]:dates[-1]].rename('融资余额变化率')


def active_buy_ratio(context, date, domain, name):
    """四档主动买入金额/主动买卖金额，先汇总金额再求比值。"""
    end = context.get_position(date)
    dates = context.dates[:end + 1]
    buy = np.zeros((len(dates), len(context.ticks)))
    sell = np.zeros_like(buy)
    observed = np.zeros_like(buy, dtype=bool)
    for size in ('sm', 'mid', 'lg', 'elg'):
        b = context.read('local', f'd_moneyflow/buy_{size}_amount', 0, end, kind='matrix')
        s = context.read('local', f'd_moneyflow/sell_{size}_amount', 0, end, kind='matrix')
        buy += np.where(np.isfinite(b), b, 0)
        sell += np.where(np.isfinite(s), s, 0)
        observed |= np.isfinite(b) | np.isfinite(s)
    weights = context.get_weight(domain, name, end)
    total = buy + sell
    ratio = np.divide(buy, total, out=np.full_like(buy, np.nan), where=total > 0)
    outputs = []
    for mask in ((weights > 0, weights < 0) if domain == 'style' else (weights > 0,)):
        outputs.append(weighted_series(ratio, np.where(mask & observed, total, 0), dates))
    return (outputs[0] if len(outputs) == 1 else outputs[0] - outputs[1]).rename('主动买入占比')


def category_share(context, date, domain, name, size='large'):
    """size取large/medium/small；分类买卖金额/域内成交额，两侧全缺保留缺失。"""
    end = context.get_position(date)
    dates = context.dates[:end + 1]
    field = {'large': 'lg', 'medium': 'mid', 'small': 'sm'}[size]
    buy = context.read('local', f'd_moneyflow/buy_{field}_amount', 0, end, kind='matrix')
    sell = context.read('local', f'd_moneyflow/sell_{field}_amount', 0, end, kind='matrix')
    category = np.where(np.isfinite(buy) | np.isfinite(sell),
                        np.fmax(buy, 0) + np.fmax(sell, 0), np.nan)
    amount = context.read('local', 'd_essentials/amount', 0, end, kind='matrix')
    weights = context.get_weight(domain, name, end)
    ratio = np.divide(np.where(np.isfinite(category), category, 0), amount,
                      out=np.full_like(amount, np.nan), where=amount > 0)
    outputs = []
    for mask in ((weights > 0, weights < 0) if domain == 'style' else (weights > 0,)):
        result = weighted_series(ratio, np.where(mask, amount, 0), dates)
        result[~(mask & np.isfinite(category) & np.isfinite(amount)).any(axis=1)] = np.nan
        outputs.append(result)
    return (outputs[0] if len(outputs) == 1 else outputs[0] - outputs[1]).rename(f'{size}单成交额占比')




def etf_net_subscription_ratio(context, date, domain='market', name='all', *,
                               period=1, fund_codes=None):
    """境内股票ETF近period日估算净申赎金额/期初AUM，输出小数比例。

    SharesChange为期末份额减期初份额，乘当日单位净值估算净申赎金额；
    份额和净值按InfoPublDate映射到当日或随后首个交易日，规模基准沿用最近披露值。
    净申赎份额只在披露对应日计入，不向后填充；分子分母使用有效ETF样本。
    market默认全体境内股票ETF；其他域须传入明确的ETF InnerCode列表。
    """
    if domain != 'market' and fund_codes is None:
        raise ValueError('ETF分组尚无跟踪指数映射，请提供fund_codes（ETF InnerCode）')
    if period < 1:
        raise ValueError('period须为正整数')
    end = context.get_position(date)
    dates = context.dates[:end + 1]
    result = pd.Series(np.nan, index=dates, name='ETF净申赎率')
    condition = 'a.Type=4 AND a.FundNature=1 AND a.FundTypeCode=1101'
    params = [dates[0].to_pydatetime(), dates[-1].to_pydatetime()]
    if fund_codes:
        condition += ' AND a.InnerCode IN (' + ','.join(['%s'] * len(fund_codes)) + ')'
        params.extend(int(code) for code in fund_codes)
    shares = context.jy(f"""
        SELECT s.InnerCode, s.EndDate, s.InfoPublDate, s.FloatShares, s.SharesChange
        FROM MF_SharesChange s
        JOIN MF_FundArchives a ON a.InnerCode=s.InnerCode
        WHERE s.StatPeriod='996' AND s.EndDate BETWEEN %s AND %s AND {condition}
        """, tuple(params))
    nav = context.jy(f"""
        SELECT n.InnerCode, n.EndDate, n.InfoPublDate, n.UnitNV
        FROM MF_NetValue n
        JOIN MF_FundArchives a ON a.InnerCode=n.InnerCode
        WHERE n.EndDate BETWEEN %s AND %s AND {condition}
        """, tuple(params))
    if shares.empty or nav.empty:
        return result
    for frame in (shares, nav):
        frame['EndDate'] = pd.to_datetime(frame['EndDate']).dt.normalize()
        frame['InfoPublDate'] = pd.to_datetime(frame['InfoPublDate']).dt.normalize()
        # 各截止日使用首次披露值，再按发布日进入交易日轴。
        frame.sort_values(['InnerCode', 'EndDate', 'InfoPublDate'], kind='stable', inplace=True)
        frame.drop_duplicates(['InnerCode', 'EndDate'], keep='first', inplace=True)
    codes = sorted(set(shares['InnerCode']) & set(nav['InnerCode']))

    def daily_matrix(frame, fields):
        frame = frame.loc[frame['InnerCode'].isin(codes)].copy()
        position = dates.searchsorted(frame['InfoPublDate'], side='left')
        frame = frame.loc[position < len(dates)].copy()
        frame['date'] = dates[position[position < len(dates)]].to_numpy()
        pivot = frame.pivot_table(index='date', columns='InnerCode', values=fields,
                                  aggfunc='last')
        return {field: pivot[field].reindex(index=dates, columns=codes)
                for field in fields}

    shares_daily = daily_matrix(shares, ['FloatShares', 'SharesChange'])
    nav_daily = daily_matrix(nav, ['UnitNV'])
    s = shares_daily['FloatShares'].ffill()
    shares_change = shares_daily['SharesChange']
    nv = nav_daily['UnitNV'].ffill()
    s = s.astype(float).where(np.isfinite(s) & (s >= 0))
    shares_change = shares_change.astype(float)
    nv = nv.astype(float).where(np.isfinite(nv) & (nv > 0))
    flow = shares_change * nv
    flow = flow.rolling(period, min_periods=period).sum()
    base = s.shift(period) * nv.shift(period)
    valid = np.isfinite(flow) & np.isfinite(base) & (base > 0)
    numerator = flow.where(valid).sum(axis=1, min_count=1)
    denominator = base.where(valid).sum(axis=1, min_count=1)
    return (numerator / denominator.where(denominator > 0)).rename('ETF净申赎率')


def equity_fund_new_issue_scale(context, date, domain='market', name='all', *,
                                period=20, fund_type_codes=(1101, 1103)):
    """按发行结束日统计近period个交易日权益基金新发规模，单位元。
    单只基金规模=ShareIssued×UnitIssuePrice；1101为股票型，1103为混合型。
    这是按发行结束日回溯的统计值，不代表该规模在发行结束日已披露。
    分类取MF_FundArchives.FundTypeCode，不使用MF_FundType分类体系。
    """
    if period < 1:
        raise ValueError('period须为正整数')
    end = context.get_position(date)
    dates = context.dates[:end + 1]
    codes = tuple(int(code) for code in fund_type_codes)
    if not codes:
        return pd.Series(np.nan, index=dates, name='权益基金新发规模')
    placeholders = ','.join(['%s'] * len(codes))
    frame = context.jy(f"""
        SELECT i.InnerCode, i.IssueEndDate, i.ShareIssued, i.UnitIssuePrice
        FROM MF_IssueAndListing i
        JOIN MF_FundArchives a ON a.InnerCode=i.InnerCode
        WHERE i.IssueEndDate BETWEEN %s AND %s
          AND a.FundTypeCode IN ({placeholders})
        """, (dates[0].to_pydatetime(), dates[-1].to_pydatetime(), *codes))
    if frame.empty:
        return pd.Series(np.nan, index=dates, name='权益基金新发规模')

    frame['IssueEndDate'] = pd.to_datetime(frame['IssueEndDate']).dt.normalize()
    position = dates.searchsorted(frame['IssueEndDate'], side='left')
    valid = position < len(dates)
    frame = frame.loc[valid].copy()
    frame['date'] = dates[position[valid]].to_numpy()
    frame['scale'] = (pd.to_numeric(frame['ShareIssued'], errors='coerce') * pd.to_numeric(frame['UnitIssuePrice'], errors='coerce'))
    daily = frame.groupby('date')['scale'].sum(min_count=1).reindex(dates)
    issue_count = frame.groupby('date').size().reindex(dates, fill_value=0)
    valid_count = frame.groupby('date')['scale'].count().reindex(dates, fill_value=0)
    daily.loc[issue_count.eq(0)] = 0
    daily.loc[valid_count.lt(issue_count)] = np.nan
    return daily.rolling(period, min_periods=period).sum().rename('权益基金新发规模')


def fund_holdings_similarity(context, date, domain='market', name='all', *,
                              fund_type_codes=(1101, 1103)):
    """以持仓占净值比例估算权益基金每日重仓股加权Jaccard相似度。

    报告按InfoPublDate生效；报告间持仓用股票pct和基金UnitNV收益漂移估算。
    InvestType=1取综合投资；以基金InnerCode为样本单位，不合并关联份额。
    """
    end = context.get_position(date)
    dates = context.dates[:end + 1]
    ticks = pd.Index(context.ticks)
    result = pd.Series(np.nan, index=dates, name='基金重仓股相似度')
    codes = tuple(int(code) for code in fund_type_codes)
    if not codes:
        return result
    placeholders = ','.join(['%s'] * len(codes))
    params = (dates[-1].to_pydatetime(), *codes)
    holdings = context.jy(f"""
        SELECT p.InnerCode, p.ReportDate, p.InfoPublDate, s.SecuCode AS StockCode,
               p.RatioInNV
        FROM MF_StockPortfolioDetail p
        JOIN MF_FundArchives a ON a.InnerCode=p.InnerCode
        JOIN SecuMain s ON s.InnerCode=p.StockInnerCode
        WHERE p.InfoPublDate<=%s AND p.InvestType=1
          AND a.FundTypeCode IN ({placeholders})
          AND s.SecuMarket IN (83,90) AND s.SecuCategory=1
        """, params)
    if holdings.empty:
        return result

    holdings['InfoPublDate'] = pd.to_datetime(holdings['InfoPublDate']).dt.normalize()
    holdings['ReportDate'] = pd.to_datetime(holdings['ReportDate']).dt.normalize()
    available = dates.searchsorted(holdings['InfoPublDate'], side='left')
    holdings = holdings.loc[available < len(dates)].copy()
    holdings['available'] = available[available < len(dates)]
    holdings['report_pos'] = np.maximum(
        dates.searchsorted(holdings['ReportDate'], side='right') - 1, 0)
    holdings['stock_pos'] = ticks.get_indexer(holdings['StockCode'].astype(str).str.zfill(6))
    holdings['weight'] = pd.to_numeric(holdings['RatioInNV'], errors='coerce')
    holdings = holdings.loc[(holdings['stock_pos'] >= 0) & np.isfinite(holdings['weight'])
                            & (holdings['weight'] > 0)]
    if holdings.empty:
        return result

    nav = context.jy(f"""
        SELECT n.InnerCode, n.EndDate, n.InfoPublDate, n.UnitNV
        FROM MF_NetValue n
        JOIN MF_FundArchives a ON a.InnerCode=n.InnerCode
        WHERE n.InfoPublDate<=%s AND a.FundTypeCode IN ({placeholders})
        """, params)
    if nav.empty:
        return result
    nav['EndDate'] = pd.to_datetime(nav['EndDate']).dt.normalize()
    nav['InfoPublDate'] = pd.to_datetime(nav['InfoPublDate']).dt.normalize()
    nav = nav.loc[nav['InnerCode'].isin(holdings['InnerCode'])].copy()
    nav['available'] = dates.searchsorted(nav['InfoPublDate'], side='left')
    nav = nav.loc[nav['available'] < len(dates)].copy()
    nav['end_pos'] = np.maximum(dates.searchsorted(nav['EndDate'], side='right') - 1, 0)
    nav['UnitNV'] = pd.to_numeric(nav['UnitNV'], errors='coerce')
    nav = nav.loc[np.isfinite(nav['UnitNV']) & (nav['UnitNV'] > 0)]
    nav.sort_values(['InnerCode', 'EndDate', 'InfoPublDate'], kind='stable', inplace=True)
    nav.drop_duplicates(['InnerCode', 'EndDate'], keep='first', inplace=True)

    events = {}
    for (available_pos, fund), rows in holdings.groupby(['available', 'InnerCode'], sort=False):
        report_date = rows['ReportDate'].max()
        rows = rows.loc[rows['ReportDate'].eq(report_date)]
        rows = rows.groupby('stock_pos', sort=False)['weight'].sum()
        report_pos = int(dates.searchsorted(report_date, side='right') - 1)
        events.setdefault(int(available_pos), []).append(
            (fund, max(report_pos, 0), rows.index.to_numpy(dtype=int), rows.to_numpy(dtype=float)))
    nav_events = {}
    for (available_pos, fund), rows in nav.groupby(['available', 'InnerCode'], sort=False):
        row = rows.loc[rows['EndDate'].idxmax()]
        nav_events.setdefault(int(available_pos), []).append(
            (fund, int(row['end_pos']), float(row['UnitNV'])))

    returns = np.nan_to_num(
        context.read('local', 'd_essentials/pct', 0, end, kind='matrix') / 100,
        nan=0.0)
    price_index = np.vstack((np.ones((1, returns.shape[1])), np.cumprod(1 + returns, axis=0)))
    active, latest_report, current_nav, latest_nav_end, nav_history = {}, {}, {}, {}, {}

    def mean_similarity(portfolios):
        funds = [(fund, stock, weight) for fund, (stock, weight) in portfolios.items()
                 if len(stock) and np.isfinite(weight).all() and weight.sum() > 0]
        count = len(funds)
        if count < 2:
            return np.nan
        totals = {fund: weight.sum() for fund, _, weight in funds}
        holders = {}
        for fund, stocks, weights in funds:
            for stock, weight in zip(stocks, weights):
                holders.setdefault(stock, []).append((fund, weight))
        overlap = {}
        for stock_holders in holders.values():
            for left in range(len(stock_holders) - 1):
                fund_i, weight_i = stock_holders[left]
                for right in range(left + 1, len(stock_holders)):
                    fund_j, weight_j = stock_holders[right]
                    key = (fund_i, fund_j)
                    overlap[key] = overlap.get(key, 0.0) + min(weight_i, weight_j)
        score = sum(value / (totals[i] + totals[j] - value)
                    for (i, j), value in overlap.items())
        return score / (count * (count - 1) / 2)

    for i in range(len(dates)):
        nav_return = {}
        for fund, report_pos, unit_nav in nav_events.get(i, ()):
            if report_pos < latest_nav_end.get(fund, -1):
                continue
            previous = current_nav.get(fund)
            if previous is not None:
                nav_return[fund] = unit_nav / previous - 1
            current_nav[fund] = unit_nav
            latest_nav_end[fund] = report_pos
            nav_history.setdefault(fund, []).append((report_pos, unit_nav))

        for fund, (stocks, weight) in active.items():
            weight *= (1 + returns[i, stocks]) / (1 + nav_return.get(fund, 0.0))
            active[fund] = (stocks, weight)

        for fund, report_pos, stocks, weight in events.get(i, ()):
            if report_pos < latest_report.get(fund, -1):
                continue
            base_nav = next((value for pos, value in reversed(nav_history.get(fund, ()))
                             if pos <= report_pos), None)
            nav_factor = current_nav.get(fund, base_nav) / base_nav if base_nav else 1.0
            price_factor = price_index[i + 1, stocks] / price_index[report_pos + 1, stocks]
            active[fund] = (stocks, weight * price_factor / nav_factor)
            latest_report[fund] = report_pos
        result.iloc[i] = mean_similarity(active)
    return result


MONEYFLOW_INDICATORS = {
    function.__name__: function for function in (
        amount_change, top_share, turnover, active_buy_ratio, balance_change,
        category_share, etf_net_subscription_ratio, equity_fund_new_issue_scale,
        fund_holdings_similarity,
    )
}


def update_moneyflow(root=ROOT, domain='index', name='hs300', indicators=None, *, indicator_params=None):
    """统一写入单列bin；category_share默认计算大中小三档，size可指定单档。"""
    from v2.UpdateData.utils import ensure_memmap

    selected = (tuple(MONEYFLOW_INDICATORS) if indicators is None else
                (indicators,) if isinstance(indicators, str) else tuple(indicators))
    unknown = set(selected) - MONEYFLOW_INDICATORS.keys()
    if unknown:
        raise ValueError(f'未知moneyflow指标: {sorted(unknown)}')
    if not selected:
        return {}
    context = MonitorContext(root=root)
    try:
        axis_dates = context.data.axis.full_dates
        if not len(context.dates):
            raise ValueError('日期轴没有有效交易日')
        start, end = context.dates[0], context.dates[-1]
        output_root = Path(root) / 'median_variables' / 'monitor' / 'moneyflow'
        outputs = {}
        for indicator in selected:
            params = dict((indicator_params or {}).get(indicator, {}))
            if indicator == 'etf_net_subscription_ratio' and domain != 'market' and params.get('fund_codes') is None:
                print(f'[SKIP] {indicator}: 分组需提供fund_codes', flush=True)
                continue
            sizes = ((params['size'],) if 'size' in params else ('large', 'medium', 'small')) if indicator == 'category_share' else (None,)
            for size in sizes:
                if size is not None:
                    params['size'] = size
                values = MONEYFLOW_INDICATORS[indicator](context, end, domain, name,
                    **params).reindex(context.dates)
                output_name = f'{size}_trade_share' if size is not None else indicator
                path = output_root / f'{output_name}.bin'
                array = ensure_memmap(path, (len(axis_dates), 1), dtype=np.float32)
                try:
                    array[:] = np.nan
                    array[:len(context.dates), 0] = values.to_numpy(dtype=np.float32)
                    array.flush()
                finally:
                    del array
                outputs[output_name] = path
                print(f'[OK] {output_name}: {start.date()} - {end.date()}', flush=True)
        return outputs
    finally:
        context.close()


__all__ = [*MONEYFLOW_INDICATORS, 'MONEYFLOW_INDICATORS', 'update_moneyflow']


if __name__ == '__main__':
    DOMAIN = 'index'
    NAME = 'hs300'
    INDICATORS = None
    update_moneyflow(root=ROOT, domain=DOMAIN, name=NAME, indicators=INDICATORS)
