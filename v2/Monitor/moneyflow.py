"""资金流指标：每个函数直接从context读数，返回域内日度Series。"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "v2.Monitor"

from .context import MonitorContext
from v2.UpdateData.config import ROOT
from .common import weighted_series


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
    if fund_codes is not None:
        fund_codes = tuple(dict.fromkeys(int(code) for code in fund_codes))
    if domain != 'market' and not fund_codes:
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
        SELECT s.InnerCode, s.EndDate, s.InfoPublDate, s.FloatShares,
               a.LastOperationDate
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
    split_filter = ''
    split_params = [dates[-1].to_pydatetime(), dates[0].to_pydatetime(),
                    dates[-1].to_pydatetime()]
    if fund_codes:
        split_filter = 'AND a.InnerCode IN (' + ','.join(['%s'] * len(fund_codes)) + ')'
        split_params.extend(fund_codes)
    splits = context.jy(f"""
        SELECT s.InnerCode, s.InfoPublDate, s.SplitDay, s.ActualSplitDay,
               s.SplitDayEX, s.SplitRatio, s.SharesAfter
        FROM MF_SharesSplit s
        JOIN MF_FundArchives a ON a.InnerCode=s.InnerCode
        WHERE s.InfoPublDate<=%s AND a.Type=4 AND a.FundNature=1
          AND a.FundTypeCode=1101
          AND COALESCE(s.SplitDayEX,s.ActualSplitDay,s.SplitDay) BETWEEN %s AND %s
          {split_filter}
        """, tuple(split_params))
    if shares.empty or nav.empty:
        return result
    for frame in (shares, nav):
        frame['EndDate'] = pd.to_datetime(frame['EndDate']).dt.normalize()
        frame['InfoPublDate'] = pd.to_datetime(frame['InfoPublDate']).dt.normalize()
        # 各截止日使用首次披露值，再按发布日进入交易日轴。
        frame.sort_values(['InnerCode', 'EndDate', 'InfoPublDate'], kind='stable', inplace=True)
        frame.drop_duplicates(['InnerCode', 'EndDate'], keep='first', inplace=True)
    codes = sorted(set(shares['InnerCode']) & set(nav['InnerCode']))
    last_operation = (shares[['InnerCode', 'LastOperationDate']]
        .drop_duplicates('InnerCode').set_index('InnerCode')['LastOperationDate']
        .reindex(codes))
    last_operation = pd.to_datetime(last_operation, errors='coerce').dt.normalize()
    operation_pos = np.full(len(codes), -1, dtype=np.int64)
    known_operation = last_operation.notna().to_numpy()
    operation_pos[known_operation] = dates.searchsorted(last_operation[known_operation], side='right') - 1   # operation_pos[i] 是第 i 只基金的最后运作日对应的交易日索引
    nav_available = dates.searchsorted(nav['InfoPublDate'], side='left')
    nav_available = pd.Series(nav_available, index=nav['InnerCode'])
    last_seen_pos = nav_available[nav_available < len(dates)].groupby(level=0).max()
    last_seen_pos = last_seen_pos.reindex(codes).fillna(-1).to_numpy(dtype=np.int64)
    last_pos = np.where(known_operation, operation_pos, last_seen_pos)
    alive = np.arange(len(dates))[:, None] <= last_pos[None, :]

    def daily_matrix(frame, fields):
        frame = frame.loc[frame['InnerCode'].isin(codes)].copy()
        position = dates.searchsorted(frame['InfoPublDate'], side='left')
        frame = frame.loc[position < len(dates)].copy()
        frame['date'] = dates[position[position < len(dates)]].to_numpy()
        return {
            field: frame.pivot_table(index='date', columns='InnerCode', values=field, aggfunc='last').reindex(index=dates, columns=codes)
            for field in fields
        }

    shares_daily = daily_matrix(shares, ['FloatShares'])
    nav_daily = daily_matrix(nav, ['UnitNV'])
    float_shares = shares_daily['FloatShares'].apply(pd.to_numeric, errors='coerce')
    float_shares = float_shares.where(np.isfinite(float_shares) & (float_shares >= 0))
    shares_state = float_shares.ffill().where(alive)
    shares_change = shares_state.diff().where(float_shares.notna())
    if not splits.empty:
        # 拆分比例只在信息已披露、且拆后份额首次出现在日频数据时生效。
        for column in ('InfoPublDate', 'SplitDay', 'ActualSplitDay', 'SplitDayEX'):
            splits[column] = pd.to_datetime(splits[column], errors='coerce').dt.normalize()
        splits['SplitRatio'] = pd.to_numeric(splits['SplitRatio'], errors='coerce')
        splits['SharesAfter'] = pd.to_numeric(splits['SharesAfter'], errors='coerce')
        splits['action_date'] = splits['SplitDayEX'].fillna(
            splits['ActualSplitDay']).fillna(splits['SplitDay'])
        splits = splits.loc[
            np.isfinite(splits['SplitRatio']) & (splits['SplitRatio'] > 0)
            & np.isfinite(splits['SharesAfter']) & (splits['SharesAfter'] > 0)
            & splits['action_date'].notna()
        ].copy()
        ratio = np.ones((len(dates), len(codes)), dtype=float)
        code_pos = {code: i for i, code in enumerate(codes)}
        share_events = shares[['InnerCode', 'InfoPublDate', 'FloatShares']].copy()
        share_events['FloatShares'] = pd.to_numeric(share_events['FloatShares'], errors='coerce')
        share_events['day'] = dates.searchsorted(share_events['InfoPublDate'], side='left')
        share_events = share_events.loc[share_events['day'] < len(dates)]
        by_fund = {fund: frame.sort_values('day') for fund, frame in share_events.groupby('InnerCode')}
        for split in splits.itertuples(index=False):
            fund = split.InnerCode
            observations = by_fund.get(fund)
            if observations is None or fund not in code_pos:
                continue
            first_day = max(dates.searchsorted(split.InfoPublDate, side='left'),
                            dates.searchsorted(split.action_date, side='left'))
            observed = observations.loc[observations['day'] >= first_day]
            matched = observed.loc[np.isclose(
                observed['FloatShares'], split.SharesAfter, rtol=1e-6, atol=1.0)]
            if not matched.empty:
                split_day = int(matched.iloc[0]['day'])
                ratio[split_day, code_pos[fund]] *= split.SplitRatio
        # 把前一期份额换算到拆后单位，再计算真实份额差。
        shares_change = (shares_state - shares_state.shift(1) * ratio).where(float_shares.notna())
    s = shares_state
    nv = nav_daily['UnitNV'].apply(pd.to_numeric, errors='coerce').ffill().where(alive)
    nv = nv.where(np.isfinite(nv) & (nv > 0))
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


def fund_holdings_similarity(
    context, date, domain='market', name='all', *,
    fund_type_codes=(1101, 1103)
):
    """已披露A股持仓的基金两两加权Jaccard均值（日频）。
    持仓与净值分别按照披露日生效，并独立向前填充。
    持仓报告整体替换；净值更新独立进行。
    持仓权重按股票收益和最新已披露净值漂移，不归一化。
    存续截止日后不再参与计算。
    """
    end = context.get_position(date)
    dates = pd.DatetimeIndex(context.dates[:end + 1]).normalize()
    T = len(dates)
    result = pd.Series(np.nan, index=dates, name='基金重仓股相似度')

    codes = tuple(map(int, fund_type_codes))
    if not codes:
        return result

    placeholders = ','.join(['%s'] * len(codes))
    params = (dates[-1].to_pydatetime(), *codes)

    # ==========================================================
    # 1. 原始数据：不进行无意义的唯一键去重
    # ==========================================================
    holdings = context.jy(f"""
        SELECT
            p.InnerCode,
            p.ReportDate,
            p.InfoPublDate,
            s.SecuCode AS StockCode,
            p.RatioInNV,
            a.LastOperationDate
        FROM MF_StockPortfolioDetail p
        JOIN MF_FundArchives a
            ON a.InnerCode = p.InnerCode
        JOIN SecuMain s
            ON s.InnerCode = p.StockInnerCode
        WHERE p.InfoPublDate <= %s
          AND p.InvestType = 1
          AND a.FundTypeCode IN ({placeholders})
          AND s.SecuMarket IN (83, 90)
          AND s.SecuCategory = 1
    """, params)

    nav = context.jy(f"""
        SELECT
            n.InnerCode,
            n.EndDate,
            n.InfoPublDate,
            n.UnitNV
        FROM MF_NetValue n
        JOIN MF_FundArchives a
            ON a.InnerCode = n.InnerCode
        WHERE n.InfoPublDate <= %s
          AND a.FundTypeCode IN ({placeholders})
    """, params)

    if holdings.empty or nav.empty:
        return result

    for frame, report_col in ((holdings, 'ReportDate'), (nav, 'EndDate')):
        frame[report_col] = pd.to_datetime(frame[report_col]).dt.normalize()
        frame['InfoPublDate'] = pd.to_datetime(frame['InfoPublDate']).dt.normalize()
    nav['UnitNV'] = pd.to_numeric(nav['UnitNV'], errors='coerce')
    nav = nav.loc[np.isfinite(nav['UnitNV']) & (nav['UnitNV'] > 0)].copy()
    if nav.empty:
        return result

    # ==========================================================
    # 2. 基金、股票映射与存续边界
    # ==========================================================
    funds = pd.Index(holdings['InnerCode'].unique())
    F = len(funds)

    holdings['fund'] = funds.get_indexer(holdings['InnerCode'])
    nav = nav.loc[nav['InnerCode'].isin(funds)].copy()
    nav['fund'] = funds.get_indexer(nav['InnerCode'])

    tick_index = pd.Index(pd.Series(context.ticks).astype(str).str.zfill(6))
    holdings['stock'] = tick_index.get_indexer(holdings['StockCode'].astype(str).str.zfill(6))
    holdings['weight'] = pd.to_numeric(holdings['RatioInNV'], errors='coerce')
    holdings = holdings.loc[(holdings['stock'] >= 0) & np.isfinite(holdings['weight']) & (holdings['weight'] > 0)].copy()
    if holdings.empty:
        return result

    # 不存在截止日的基金默认持续至当前计算日。
    last_operation = pd.to_datetime(
        holdings.groupby('fund')['LastOperationDate'].first()
        .reindex(range(F)),
        errors='coerce'
    ).dt.normalize()
    last_pos = np.full(F, T - 1, dtype=int)
    known = last_operation.notna().to_numpy()
    last_pos[known] = dates.searchsorted(last_operation.loc[known], side='right') - 1

    # ==========================================================
    # 3. 净值事件：映射到有效交易日
    # ==========================================================
    nav['day'] = dates.searchsorted(nav[['EndDate', 'InfoPublDate']].max(axis=1),side='left')
    nav = nav.loc[(nav['day'] < T) & (nav['day'] >= 0)].copy()
    # 同一交易日同一基金只采用最新报告期的净值。
    # 这里是事件冲突处理，不是原始记录去重。
    nav = nav.sort_values(['fund', 'day', 'EndDate', 'InfoPublDate']).drop_duplicates(['fund', 'day'], keep='last')
    # 防止迟到的旧报告期净值覆盖新报告期净值。
    nav = nav.loc[nav['EndDate'].eq(nav.groupby('fund')['EndDate'].cummax())].copy()

    # 直接构造“交易日×基金”矩阵，列内ffill替代逐日更新净值字典。
    nav_daily = (
        nav.pivot(index='day', columns='fund', values='UnitNV')
        .reindex(index=range(T), columns=range(F))
        .ffill()
        .to_numpy(dtype=float, copy=True)
    )
    nav_daily[np.arange(T)[:, None] > last_pos[None, :]] = np.nan

    # ==========================================================
    # 4. 报告期基准净值：以首次披露版本匹配
    # ==========================================================
    # 此处仅为每个报告期选择一个基准净值，
    # 不是对原始nav表做全局去重。
    base_nav = (
        nav.sort_values(['InfoPublDate', 'EndDate'])
        .drop_duplicates(['fund', 'EndDate'], keep='first')
        [['fund', 'EndDate', 'InfoPublDate', 'UnitNV']]
        .rename(columns={
            'EndDate': 'ReportDate',
            'InfoPublDate': 'NavPublDate',
            'UnitNV': 'base_nav'
        })
    )

    # 每个基金、报告期、披露日是一份持仓报告事件。
    report_keys = ['fund', 'ReportDate', 'InfoPublDate']
    reports = (
        holdings.groupby(report_keys, as_index=False)
        .size()
        .drop(columns='size')
        .merge(
            base_nav,
            on=['fund', 'ReportDate'],
            how='inner'
        )
    )

    # 持仓和报告期净值均可获得后才生效。
    effective_date = reports[['ReportDate', 'InfoPublDate', 'NavPublDate']].max(axis=1)
    reports['day'] = dates.searchsorted(effective_date, side='left')
    reports['report_pos'] = (dates.searchsorted(reports['ReportDate'], side='right') - 1)
    reports = reports.loc[(reports['day'] < T) & (reports['report_pos'] >= 0)].copy()
    if reports.empty:
        return result

    # ==========================================================
    # 5. 股票累计收益指数
    # ==========================================================
    returns = context.read('local', 'd_essentials/pct', 0, end, kind='matrix')
    returns = np.nan_to_num(np.asarray(returns, dtype=float) / 100, nan=0.0, posinf=0.0, neginf=0.0)
    price = np.cumprod(1 + returns, axis=0)

    # ==========================================================
    # 6. 预计算每次披露报告的持仓系数
    # ==========================================================
    # w(t) = coefficient * price(t) / nav_available(t)
    # coefficient = report_weight * base_nav / price(report_pos)

    # 关联报告参数后一次性计算所有持仓系数，避免逐份报告切表。
    reports = reports.sort_values(['fund', 'day', 'ReportDate', 'InfoPublDate']).copy()
    reports['event'] = np.arange(len(reports))
    positions = holdings.merge(reports, on=report_keys, how='inner')
    report_price = price[positions['report_pos'].to_numpy(), positions['stock'].to_numpy()]
    positions['coefficient'] = positions['weight'].to_numpy() * positions['base_nav'].to_numpy() / report_price
    positions['valid'] = (np.isfinite(report_price) & (report_price > 0) &
                          np.isfinite(positions['coefficient']))
    valid_events = positions.groupby('event')['valid'].all()
    reports = reports.loc[reports['event'].isin(valid_events.index[valid_events])]

    # 每日每只基金采用最新报告；迟到的旧报告不能覆盖新报告。
    reports = reports.drop_duplicates(['fund', 'day'], keep='last')
    reports = reports.loc[reports['ReportDate'].eq(reports.groupby('fund')['ReportDate'].cummax())].copy()
    # [day, stop)等价于在下一报告或存续截止日之前fill整份报告。
    reports['stop'] = np.minimum(
        reports.groupby('fund')['day'].shift(-1).fillna(T).to_numpy(dtype=int),
        last_pos[reports['fund'].to_numpy()] + 1)
    intervals = pl.from_pandas(reports.loc[reports['day'] < reports['stop'], ['event', 'day', 'stop']])
    table = (pl.from_pandas(positions[['event', 'fund', 'stock', 'coefficient']])
        .join(intervals, on='event')
        .group_by(['fund', 'stock', 'day', 'stop']).agg(pl.col('coefficient').sum()))

    # 7. 每块展开20个交易日，批量漂移持仓并按日计算横截面相似度。
    # 保留分块循环控制内存；无逐日、逐基金、逐报告的Python循环。
    for start in tqdm(range(0, T, 20)):
        stop = min(start + 20, T)
        block = (
            table.filter((pl.col('day') < stop) & (pl.col('stop') > start))
            .with_columns(
                pl.int_ranges(
                    pl.max_horizontal('day', pl.lit(start)),
                    pl.min_horizontal('stop', pl.lit(stop))
                ).alias('day')
            )
            .explode('day')
        )
        if block.is_empty():
            continue
        day, fund, stock = (block[col].to_numpy().astype(int) for col in ('day', 'fund', 'stock'))
        weight = block['coefficient'].to_numpy() * price[day, stock] / nav_daily[day, fund]
        rows = (block.select('day', 'fund', 'stock').with_columns(pl.Series('weight', weight))
            .filter((pl.col('weight').is_finite() & (pl.col('weight') >= 0)).all().over(['day', 'fund'])))
        # 整只基金的保留持仓都有效且总权重大于0，才参与当日比较。
        totals = (rows.group_by(['day', 'fund']).agg(pl.col('weight').sum().alias('total'))
            .filter(pl.col('total') > 0))
        rows = rows.join(totals, on=['day', 'fund'])
        counts = totals.group_by('day').agg(pl.len().alias('count'))
        # 只关联同日共同持仓，fund<fund_j排除自身和反向重复基金对。
        scores = (rows.lazy().join(rows.lazy(), on=['day', 'stock'], suffix='_j')
            .filter(pl.col('fund') < pl.col('fund_j'))
            .group_by(['day', 'fund', 'fund_j'])
            .agg(pl.min_horizontal('weight', 'weight_j').sum().alias('overlap'),
                 pl.col('total').first(), pl.col('total_j').first())
            .with_columns((pl.col('overlap') /
                (pl.col('total') + pl.col('total_j') - pl.col('overlap'))).alias('similarity'))
            .group_by('day').agg(pl.col('similarity').sum()).collect(engine='streaming'))
        # 无共同股票的基金对贡献0；不足两只基金的日期保留NaN。
        daily = counts.join(scores, on='day', how='left').filter(pl.col('count') >= 2)
        n = daily['count'].to_numpy().astype(float)
        result.iloc[daily['day'].to_numpy()] = daily['similarity'].fill_null(0).to_numpy() / (n * (n - 1) / 2)

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
            sizes = ((params['size'],) if 'size' in params else ('large', 'medium', 'small')) if indicator == 'category_share' else (None,)
            for size in sizes:
                if size is not None:
                    params['size'] = size
                calc_domain, calc_name = domain, name
                if indicator == 'etf_net_subscription_ratio' and not params.get('fund_codes'):
                    calc_domain, calc_name = 'market', 'all'
                values = MONEYFLOW_INDICATORS[indicator](context, end, calc_domain,
                    calc_name, **params).reindex(context.dates)
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
    DOMAIN = 'market'
    NAME = 'all'
    INDICATORS = None
    update_moneyflow(root=ROOT, domain=DOMAIN, name=NAME, indicators=INDICATORS)
