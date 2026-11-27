"""按因子分组构造目标持仓权重。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from .maths import normalize


def estimate_hedge_ratio(base, long_w, short_w, benchmark, construction, window=60, min_periods=40):
    """用历史收益估计对冲比例，T日仅使用截至T-1日的数据。"""
    dates = base.dates
    returns = base.pct.to_numpy(dtype=float, copy=False)

    # 持仓滞后一天计算收益，并在持仓股票收益缺失时剔除该日。
    def portfolio_returns(weights):
        previous = pd.DataFrame(weights, index=dates).shift(1).to_numpy()
        valid = np.isfinite(returns)
        values = (np.nan_to_num(previous) * np.where(valid, returns, 0)).sum(axis=1)
        missing = ((np.abs(previous) > 0) & ~valid).any(axis=1)
        return pd.Series(values, index=dates).mask(missing)

    benchmark_returns = portfolio_returns(benchmark)
    variance = benchmark_returns.rolling(window, min_periods=min_periods).var()

    # 计算组合相对基准的滚动 beta。
    def beta(weights):
        leg_returns = portfolio_returns(weights)
        covariance = leg_returns.rolling(window, min_periods=min_periods).cov(benchmark_returns)
        return covariance / variance.where(variance > 1e-12)

    if construction == "longshort":
        long_beta, short_beta = beta(long_w), beta(short_w)
        hedge = long_beta / short_beta.where(short_beta.abs() > 1e-12)
    else:
        hedge = beta(long_w if construction == "long" else short_w)

    # 滞后一日使用已知的 hedge ratio，并向前填充有限窗口。
    hedge = hedge.shift(1).ffill(limit=window)
    valid = hedge.notna().to_numpy()
    return hedge.fillna(0).to_numpy(), valid


def build_target(
    base,
    strategy="enhanced",
    construction="longshort",
    num_groups=10,
    group_numbers=(10, 1),
    normalization="factor",
    domain="universe",
    name=None,
    industry_neutral=True,
    active_gross=1.0,
    hedge_method="dollar",
    hedge_window=60,
    hedge_min_periods=40,
):
    """按选定分组和策略生成目标权重，不修改 base.alpha。"""
    # 1. 统一参数，并复制因子值，后续筛选只作用于副本。
    alpha = base.alpha.copy(deep=True)
    alpha = alpha.where(np.isfinite(alpha))
    dates, ticks = alpha.index, alpha.columns
    market_cap = base.fields["market_cap"].reindex(index=dates, columns=ticks)

    # 2. 按 domain/name 读取已计算的基准权重。
    if not hasattr(base, "benchmark") or "weight" not in base.benchmark:
        base.cal_benchmark()
    name = "market_cap" if domain == "universe" else name
    key = str(int(name)) if domain == "industry" else name
    benchmark_frame = base.benchmark["weight"][domain][key].reindex(
        index=dates, columns=ticks
    ).fillna(0)
    benchmark = benchmark_frame.to_numpy(dtype=float, copy=False)

    # 3. 按 construction 和 group_numbers 保留因子值，其余设为 NaN。
    base.cal_groups(num_groups=num_groups, domain=domain, name=None if domain == "universe" else name)
    codes = base.group_codes.reindex(index=dates, columns=ticks)
    groups = (num_groups, 1) if group_numbers is None else (
        (group_numbers,) if np.isscalar(group_numbers) else tuple(group_numbers)
    )
    if construction == "longshort":
        if len(groups) < 2:
            raise ValueError("longshort需要多头和空头组号")
        long_groups, short_groups = groups[0:1], groups[1:2]
    elif construction == "long":
        long_groups, short_groups = groups[0:1], ()
    else:
        long_groups, short_groups = (), groups[-1:]
    eligible = benchmark_frame.gt(0)
    long_alpha = alpha.where(codes.isin(long_groups) & eligible)
    short_alpha = alpha.where(codes.isin(short_groups) & eligible)

    # 4. 按等权、市值或因子值分数分别归一化多头腿和空头腿。
    if normalization == "equal":
        long_raw = long_alpha.notna().to_numpy(dtype=float)
        short_raw = short_alpha.notna().to_numpy(dtype=float)
    elif normalization == "market_cap":
        valid_cap = np.isfinite(market_cap) & market_cap.gt(0)
        long_raw = market_cap.where(long_alpha.notna() & valid_cap, 0).to_numpy(dtype=float)
        short_raw = market_cap.where(short_alpha.notna() & valid_cap, 0).to_numpy(dtype=float)
    else:
        long_raw = long_alpha.to_numpy(dtype=float)
        short_raw = -short_alpha.to_numpy(dtype=float)
    long_w, short_w = normalize(long_raw), normalize(short_raw)
    benchmark_valid = benchmark.sum(axis=1) > 1e-12

    # 5. 指数增强可将多空腿分别缩放到各自行业的基准预算；无候选股票时不回填。
    if strategy == "enhanced" and industry_neutral:
        industry_values = base.fields["industry"].reindex(index=dates, columns=ticks).to_numpy(dtype=float)
        valid_industry = np.isfinite(industry_values) & (industry_values >= 0) & (industry_values < 31)
        industry = np.where(valid_industry, industry_values, -1).astype(int)
        budget = np.zeros((len(dates), 31))
        long_total, short_total = np.zeros_like(budget), np.zeros_like(budget)
        for code in range(31):
            mask = valid_industry & (industry == code)
            budget[:, code] = (benchmark * mask).sum(axis=1)
            long_total[:, code] = (long_w * mask).sum(axis=1)
            short_total[:, code] = (short_w * mask).sum(axis=1)
        rows = np.arange(len(dates))[:, None]
        stock_budget = budget[rows, industry]
        long_budget = long_total[rows, industry]
        short_budget = short_total[rows, industry]
        long_w = np.divide(long_w * stock_budget, long_budget, out=np.zeros_like(long_w), where=valid_industry & (long_budget > 1e-12))
        short_w = np.divide(short_w * stock_budget, short_budget, out=np.zeros_like(short_w), where=valid_industry & (short_budget > 1e-12))

    # 6. 按策略生成目标权重：absolute直接持有，enhanced叠加主动权重。
    if strategy == "absolute":
        target = long_w - short_w
        active = target.copy()
        hedge, hedge_valid = np.zeros(len(dates)), np.zeros(len(dates), dtype=bool)
    elif strategy == "enhanced":
        active = long_w - short_w
        active *= active_gross
        active[~benchmark_valid] = 0
        target = benchmark + active
        hedge, hedge_valid = np.zeros(len(dates)), np.zeros(len(dates), dtype=bool)
    else:
        # 市场中性不做行业缩放，可用美元或历史 beta 估计对冲比例。
        if hedge_method == "dollar":
            hedge, hedge_valid = np.ones(len(dates)), np.ones(len(dates), dtype=bool)
        elif hedge_method == "beta":
            hedge, hedge_valid = estimate_hedge_ratio(
                base, long_w, short_w, benchmark, construction,
                hedge_window, hedge_min_periods,
            )
        else:
            raise ValueError("hedge_method须为dollar或beta")
        if construction == "long":
            target = long_w - hedge[:, None] * benchmark
        elif construction == "short":
            target = hedge[:, None] * benchmark - short_w
        else:
            target = long_w - hedge[:, None] * short_w
        hedge_valid &= benchmark_valid
        target[~hedge_valid] = 0
        active = target.copy()

    # 7. 保存各腿、主动权重、目标权重和对冲比例。
    base.long_target = pd.DataFrame(long_w, index=dates, columns=ticks)
    base.short_target = pd.DataFrame(-short_w, index=dates, columns=ticks)
    base.benchmark_weight = pd.DataFrame(benchmark, index=dates, columns=ticks)
    base.active_target = pd.DataFrame(active, index=dates, columns=ticks)
    base.target_holding = pd.DataFrame(target, index=dates, columns=ticks)
    base.hedge_ratio = pd.Series(
        np.where(hedge_valid, hedge, np.nan) if strategy == "market_neutral" else np.nan,
        index=dates, name="hedge_ratio",
    )
    base.realized_active_gross = pd.Series(np.abs(active).sum(axis=1), index=dates)
    base.target_spec = (
        strategy, construction, group_numbers, normalization, domain, name,
        industry_neutral, active_gross, num_groups, hedge_method,
        hedge_window, hedge_min_periods,
    )
    return base.target_holding
