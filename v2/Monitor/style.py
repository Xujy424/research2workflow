"""广度风格指标。

约定：所有计算函数只接受数据数组/Series，返回数值，不依赖Context。
收益和分位采用小数；输入日收益为小数(非百分数)。
末尾 StyleMonitor 负责调用 Context、生成三张表和绘图。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
try:
    from .utils import (as_float, safe_divide, weighted_mean, compound_return,
                        historical_percentile, rank_rotation)
    from .context import MarketContext
except ImportError:  # 直接以脚本路径运行时没有 package parent
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from v2.Monitor.utils import (as_float, safe_divide, weighted_mean,
                                  compound_return, historical_percentile,
                                  rank_rotation)
    from v2.Monitor.context import MarketContext


def level_percentile(index_close, history=252, period=1):
    """指数点位先做 period 日平滑，再计算 history 日历史分位。"""
    return _series_percentile(index_close, history, period)


def relative_return(stock_returns, start_weights, benchmark_returns, period=1):
    """按窗口起始时点成分权重计算区间收益，再减同区间基准收益。"""
    r = compound_return(as_float(stock_returns)[-period:])
    portfolio = weighted_mean(r, start_weights)
    benchmark = float(compound_return(as_float(benchmark_returns)[-period:]))
    return portfolio - benchmark


def new_high_low_share(daily_returns, membership, lookback=252, period=1, history=252):
    """先计算period日平滑值，再计算其history日历史分位。"""
    r, masks = as_float(daily_returns), np.asarray(membership, dtype=bool)
    values = []
    for end in range(lookback, len(r)):
        segment = r[end-lookback:end+1]
        if len(segment) != lookback+1:
            values.append(np.nan)
            continue
        path = np.vstack([np.ones(segment.shape[1]), np.cumprod(1+segment, axis=0)])
        valid = np.isfinite(segment).all(0) & masks[end]
        high = path[-1] > np.max(path[:-1], axis=0)
        low = path[-1] < np.min(path[:-1], axis=0)
        values.append(float((high[valid].sum()-low[valid].sum())/valid.sum()) if valid.any() else np.nan)
    return _series_percentile(values, history, period)


def harmonic_valuation(multiple, weights):
    """先以市值/指数权重加权盈利或账面收益率，再取倒数，避免算术平均PE/PB失真。"""
    m = as_float(multiple)
    inv = np.where(m > 0, safe_divide(1, m), np.nan)
    return float(safe_divide(1, weighted_mean(inv, weights)))


def _aggregated_percentile(values, weights, aggregator, history, period):
    daily = [aggregator(v, w) for v, w in zip(values, weights)]
    return _series_percentile(daily, history, period)


def pe_percentile(pe_history, weight_history, history=252, period=1):
    """逐日计算股票池调和PE，对最近period日均值序列求history日分位。"""
    return _aggregated_percentile(pe_history, weight_history, harmonic_valuation, history, period)


def pb_percentile(pb_history, weight_history, history=252, period=1):
    """逐日计算股票池调和PB，对最近period日均值序列求history日分位。"""
    return _aggregated_percentile(pb_history, weight_history, harmonic_valuation, history, period)


def earnings_bond_spread(pe_history, weight_history, bond_yield, period=1):
    """股债收益差：股票池盈利收益率(加权1/PE)减同期10年国债收益率，最近period日取均值。"""
    earnings = np.array([weighted_mean(safe_divide(1, p), w) for p, w in zip(pe_history, weight_history)])
    return float(np.nanmean((earnings - as_float(bond_yield))[-period:]))


def index_eps_bond_spread(eps, market_value, bond_yield, period=1):
    """Forecast earnings divided by market value, minus bond yield."""
    earnings_yield = safe_divide(as_float(eps), as_float(market_value))
    return float(np.nanmean((earnings_yield - as_float(bond_yield))[-period:]))


def index_pe_bond_spread(pe, bond_yield, period=1):
    """Index PE-TTM reciprocal minus the 10-year bond yield."""
    earnings_yield = safe_divide(1.0, as_float(pe))
    return float(np.nanmean((earnings_yield - as_float(bond_yield))[-period:]))


def amount_percentile(amount_history, membership_history, history=252, period=1):
    """股票池当日成交金额先求和，再比较同period平滑口径历史分位。"""
    a = np.where(membership_history, as_float(amount_history), np.nan)
    values = np.nansum(a, axis=1)
    return _series_percentile(values, history, period)


def amount_share_percentile(amount_history, membership_history, history=252, period=1):
    """组内成交金额/全A成交金额；先按日计算份额，再对同period历史序列取分位。"""
    a = as_float(amount_history)
    part = np.nansum(np.where(membership_history, a, np.nan), axis=1)
    total = np.nansum(a, axis=1)
    return _series_percentile(safe_divide(part, total), history, period)


def industry_relative_return(stock_returns, industry_weight, benchmark_returns, period=5):
    """以期初行业成分权重求行业区间收益，减去同期市场基准区间收益。"""
    return relative_return(stock_returns, industry_weight, benchmark_returns, period)


def mask_spread_percentile(mask, returns, weights, period=5, history=252):
    """期初读取风格mask：1为高组、-1为低组、其他不参与；各组按期初权重加权，计算高减低的区间收益，再取历史分位。"""
    values = []
    for t in range(period, len(returns)):
        m, w = mask[t-period], weights[t-period]
        r = compound_return(returns[t-period+1:t+1])
        high, low = (m == 1), (m == -1)
        values.append(weighted_mean(r[high], w[high]) - weighted_mean(r[low], w[low]))
    return historical_percentile(np.asarray(values)[-history:])


def size_spread_percentile(mask, returns, weights, period=5, history=252):
    """直接使用大小盘mask，高市值组(1)收益减低市值组(-1)收益的历史分位。"""
    return mask_spread_percentile(mask, returns, weights, period, history)


def growth_value_spread_percentile(mask, returns, weights, period=5, history=252):
    """直接使用成长/价值mask：成长组(1)减价值组(-1)，避免临时重建未经公告日处理的盈利增长。"""
    return mask_spread_percentile(mask, returns, weights, period, history)


def pe_spread_percentile(mask, returns, weights, period=5, history=252):
    """PE风格mask正组(1)减负组(-1)的区间收益差历史分位。"""
    return mask_spread_percentile(mask, returns, weights, period, history)


def pb_spread_percentile(mask, returns, weights, period=5, history=252):
    """PB风格mask正组(1)减负组(-1)的区间收益差历史分位。"""
    return mask_spread_percentile(mask, returns, weights, period, history)


def valuation_spread_percentile(values, returns, weights, period=5, history=252):
    """市值加权计算估值最低 10% 与最高 10% 的收益差历史分位。"""
    spreads = []
    for t in range(period, len(values)):
        v = as_float(values[t - period])
        w = as_float(weights[t - period])
        r = compound_return(returns[t - period + 1:t + 1], axis=0)
        valid = np.isfinite(v) & np.isfinite(w) & (w > 0) & np.isfinite(r)
        if valid.sum() < 10:
            spreads.append(np.nan)
            continue
        cut = max(1, int(valid.sum() * 0.1))
        order = np.argsort(v[valid])
        idx = np.flatnonzero(valid)
        low, high = idx[order[:cut]], idx[order[-cut:]]
        spreads.append(weighted_mean(r[low], w[low]) - weighted_mean(r[high], w[high]))
    return _series_percentile(spreads, history, 1)


def beta_spread_percentile(mask, returns, weights, period=5, history=252):
    """使用已存储的Beta高低分组mask；Beta暴露本身直接从本地barra/beta读取而不在此估计。"""
    return mask_spread_percentile(mask, returns, weights, period, history)


def rotation_speed_percentile(industry_returns, rank_period=20, history=252, period=1):
    """以相邻日行业过去rank_period日收益排名Spearman相关性定义轮动速度=1-rho。"""
    r = as_float(industry_returns)
    speeds = []
    for t in range(rank_period, len(r)):
        previous = compound_return(r[t-rank_period:t], axis=0)
        current = compound_return(r[t-rank_period+1:t+1], axis=0)
        speeds.append(rank_rotation(previous, current))
    return _series_percentile(speeds, history, period)


def _series_percentile(series, history=252, period=1):
    x = as_float(series)
    smoothed = np.array([np.nanmean(x[i-period+1:i+1]) if i >= period-1 else np.nan for i in range(len(x))])
    return historical_percentile(smoothed[-history:])





class StyleMonitor:
    """薄编排层：Context数据读取 -> 指标函数 -> 表格/绘图；不负责数据获取实现。"""

    UNIVERSES = ('hs300', 'zzfull')
    PERIODS = (1, 5, 10, 20)

    def __init__(self, context):
        self.context = context

    def snapshot(self, asof, periods=PERIODS, universes=UNIVERSES,
                 industries=None, benchmark='zzfull', history=252,
                 high_lookback=60, rotation_window=5):
        """返回Excel首Sheet的 market/industry/style 三张DataFrame。"""
        c = self.context
        end = c.position(asof)
        # 多窗口指标、历史分位以及增长率的252日回看需要预留历史。
        start = max(0, end - history - high_lookback - max(periods) - rotation_window - 260)
        dates = c.dates[start:end+1]
        r = c.matrix('d_essentials/pct', start, end) / 100
        amount = c.matrix('d_essentials/amount', start, end)
        pe = c.matrix('fundamental/valuation/pe_ttm', start, end)
        pb = c.matrix('fundamental/valuation/pb_lf', start, end)
        industry_codes = c.matrix('industry/industry', start, end)
        weights = {u: c.stock_weight_matrix(start, end, u) for u in (*universes, benchmark, None)}
        benchmark_daily = np.array([weighted_mean(r[t], weights[benchmark][t-1]) if t else np.nan for t in range(len(dates))])
        bond = c.bond_10y(dates[0], dates[-1]).reindex(dates).ffill().to_numpy(dtype=float)
        # EDB commonly returns yields in percentage points (e.g. 1.6),
        # while PE earnings yield is a decimal (e.g. 0.06).
        if np.isfinite(bond).any() and np.nanmedian(np.abs(bond)) > 1:
            bond = bond / 100.0
        columns = pd.MultiIndex.from_product([universes, periods], names=['universe','period'])
        market = pd.DataFrame(index=['点位252历史分位','相对收益','新高新低差占比历史分位','PE_TTM历史分位',
                                     'PB_LF历史分位','盈利-10年国债收益率','成交额历史分位','成交额占比历史分位'],
                              columns=columns, dtype=float)
        for u in universes:
            w = weights[u]
            members = np.isfinite(w) & (w > 0)
            close = c.index_close(u, dates[0], dates[-1]).reindex(dates).ffill().to_numpy(dtype=float)
            index_code = {'hs300': '000300.SH', 'zzfull': '000985.SH'}.get(u)
            index_pe = c.wsd(index_code, 'pe_ttm', dates[0], dates[-1])
            index_pe = index_pe.reindex(dates).ffill().to_numpy(dtype=float)
            for h in periods:
                market.loc['点位252历史分位',(u,h)] = level_percentile(close, 252, h)
                market.loc['相对收益',(u,h)] = relative_return(r, w[-h-1], benchmark_daily, h)
                market.loc['新高新低差占比历史分位',(u,h)] = new_high_low_share(
                    r, members, high_lookback, h, history
                )
                market.loc['PE_TTM历史分位',(u,h)] = pe_percentile(pe, w, history, h)
                market.loc['PB_LF历史分位',(u,h)] = pb_percentile(pb, w, history, h)
                market.loc['盈利-10年国债收益率',(u,h)] = index_pe_bond_spread(index_pe, bond, h)
                market.loc['成交额历史分位',(u,h)] = amount_percentile(amount, members, history, h)
                market.loc['成交额占比历史分位',(u,h)] = amount_share_percentile(amount, members, history, h)

        if industries is None:
            industries = list(c.industry_map)
        industry = pd.DataFrame(index=industries, columns=periods, dtype=float)
        for ind in industries:
            for h in periods:
                w = c.stock_weights(end-h, None, ind)
                industry.loc[ind,h] = industry_relative_return(r, w, benchmark_daily, h)

        # 风格mask由上游按其明确的高/低10%规则生成；这里直接读取，避免重复排序。
        # Beta值本地可直接读 barra/beta；此面板只需要已经由Beta构建的分组mask。
        all_w = weights[None]
        masks = {name: c.matrix(field, start, end) for name, field in c.style_masks.items()}
        style = pd.DataFrame(index=['大小盘收益差分位','成长价值收益差分位','PE收益差分位',
                                    'PB收益差分位','高低Beta收益差分位','行业轮动速度分位'],
                             columns=periods, dtype=float)
        industry_ids = [c.industry_map.get(x, x) for x in industries]
        # 行业收益使用每日前一交易日行业成员和流通市值权重。
        sector_daily = np.full((len(dates), len(industry_ids)), np.nan)
        for j, ind in enumerate(industry_ids):
            for t in range(1, len(dates)):
                w = np.where(industry_codes[t-1] == ind, all_w[t-1], np.nan)
                sector_daily[t,j] = weighted_mean(r[t], w)
        for h in periods:
            style.loc['大小盘收益差分位',h] = size_spread_percentile(masks['size'], r, all_w, h, history)
            style.loc['成长价值收益差分位',h] = growth_value_spread_percentile(masks['growth_value'], r, all_w, h, history)
            style.loc['PE收益差分位',h] = valuation_spread_percentile(pe, r, all_w, h, history)
            style.loc['PB收益差分位',h] = valuation_spread_percentile(pb, r, all_w, h, history)
            style.loc['高低Beta收益差分位',h] = beta_spread_percentile(masks['beta'], r, all_w, h, history)
            style.loc['行业轮动速度分位',h] = rotation_speed_percentile(sector_daily, rotation_window, history, h)
        return {'market':market, 'industry':industry, 'style':style}

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
        return fig, ax


if __name__ == '__main__':
    # 直接运行本文件时，打印指定交易日的三张监控表。
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    context = MarketContext()
    try:
        result = StyleMonitor(context).snapshot('2026-09-30')
        for section in ('market', 'industry', 'style'):
            print(f'\n===== {section} =====')
            print(StyleMonitor.table(result, section))
    finally:
        context.close()
