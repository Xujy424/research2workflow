"""将因子分组映射为目标持仓；不计算交易成本、不改变实际成交。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from .maths import normalize



def estimate_hedge_ratio(
        base, 
        long_w, 
        short_w, 
        benchmark, 
        construction,
        window=60, min_periods=40
    ):
    """滚动回归：T日h仅使用截至T-1日已实现的历史组合收益。"""
    exposure = (
        long_w if construction == "long" else
        short_w if construction == "short" else
        0.5 * (long_w - short_w)
    )

    pct = base.pct.to_numpy(dtype=float, copy=False)
    dates = base.dates

    # T-1日持仓计算T日收益
    b_prev = pd.DataFrame(benchmark, index=dates).shift(1).to_numpy()
    leg_prev = pd.DataFrame(exposure, index=dates).shift(1).to_numpy()

    valid_pct = np.isfinite(pct)

    bm_ret = np.sum(
        np.nan_to_num(b_prev) * np.where(valid_pct, pct, 0.),
        axis=1
    )
    leg_ret = np.sum(
        np.nan_to_num(leg_prev) * np.where(valid_pct, pct, 0.),
        axis=1
    )

    # 持仓股票收益缺失时，当日组合收益无效
    missing_b = ((np.abs(b_prev) > 0) & ~valid_pct).any(axis=1)
    missing_leg = ((np.abs(leg_prev) > 0) & ~valid_pct).any(axis=1)

    rb = pd.Series(bm_ret, index=dates).mask(missing_b)
    rl = pd.Series(leg_ret, index=dates).mask(missing_leg)

    # 首日无前一日持仓
    rb.iloc[0] = np.nan
    rl.iloc[0] = np.nan

    # 截至当日的滚动Beta
    cov = rl.rolling(window, min_periods=min_periods).cov(rb)
    var = rb.rolling(window, min_periods=min_periods).var()
    h = cov / var.where(var > 1e-12)

    # T日对冲比例仅使用截至T-1日的数据
    h = h.shift(1)

    hedge_valid = h.notna().to_numpy()
    hedge = np.where(hedge_valid, h.to_numpy(), 0.)

    return hedge, hedge_valid


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
    hedge_min_periods=40
):
    """将T日Alpha分组映射为T日目标权重，不处理成交。"""

    # --------------------------------------------------
    # 0. 参数处理
    # --------------------------------------------------
    strategy = str(strategy).lower()
    construction = str(construction or base.construction).lower()
    normalization = str(normalization or base.normalization).lower()
    num_groups = base.num_groups if num_groups is None else num_groups
    active_gross = (
        0.1 if strategy == "enhanced" else 1.0
    ) if active_gross is None else float(active_gross)

    # --------------------------------------------------
    # 1. 获取Benchmark归一化权重
    # --------------------------------------------------
    if domain == "universe":
        name = "market_cap"
        mv_df = base.fields["market_cap"]
        eligible = (base.pct.notna() & np.isfinite(mv_df) & (mv_df > 0))
        b_frame = mv_df.where(eligible, 0).fillna(0)
        b_frame = b_frame.div(
            b_frame.sum(axis=1).replace(0, np.nan),
            axis=0
        ).fillna(0)
    else:
        # 指数、行业基准：读取已有权重
        key = str(int(name)) if domain == "industry" else name
        b_frame = base.domain_weights[domain][key]

    b_w = b_frame.to_numpy(dtype=float, copy=False)
    valid_b = b_w.sum(axis=1) > 1e-12

    # --------------------------------------------------
    # 2. 计算域内分组
    # --------------------------------------------------
    base.cal_groups(
        num_groups=num_groups,
        domain=domain,
        name=None if domain == "universe" else name
    )
    codes = base.group_codes.to_numpy(dtype=float, copy=False)

    # --------------------------------------------------
    # 3. 确定域内多空分组
    # --------------------------------------------------
    if group_numbers is None:
        group_numbers = (num_groups, 1)
    if construction == "longshort":
        lg, sg = group_numbers
    elif construction == "long":
        lg, sg = group_numbers[0], ()
    else:
        lg, sg = (), group_numbers[1]

    # --------------------------------------------------
    # 4. 构建多头和空头原始权重
    # --------------------------------------------------
    mv = base.fields["market_cap"].to_numpy(dtype=float, copy=False)
    alpha = base.alpha.to_numpy(dtype=float, copy=False)
    # cal_groups已经完成域内分组， benchmark > 0进一步保证股票具有有效基准权重
    in_long = np.isin(codes, lg) & (b_w > 0)
    in_short = np.isin(codes, sg) & (b_w > 0)

    if normalization == "equal":
        raw_l = in_long.astype(float)
        raw_s = in_short.astype(float)
    elif normalization == "market_cap":
        valid_mv = np.isfinite(mv) & (mv > 0)
        raw_l = np.where(in_long & valid_mv, mv, 0.)
        raw_s = np.where(in_short & valid_mv, mv, 0.)
    else:
        valid_alpha = np.isfinite(alpha)
        # 多头：因子值越高，权重越大
        alpha_l = np.where(in_long & valid_alpha, alpha, np.nan)
        min_l = np.nanmin(alpha_l, axis=1, keepdims=True)
        min_l = np.where(np.isfinite(min_l), min_l, 0.)
        raw_l = np.where(in_long & valid_alpha, alpha-min_l, 0.)
        # 空头：因子值越低，权重越大
        alpha_s = np.where(in_short & valid_alpha, alpha, np.nan)
        max_s = np.nanmax(alpha_s, axis=1, keepdims=True)
        max_s = np.where(np.isfinite(max_s), max_s, 0.)
        raw_s = np.where(in_short & valid_alpha, max_s-alpha, 0.)

    # --------------------------------------------------
    # 5. 构建完整的多头和空头组合
    # --------------------------------------------------
    wl = np.zeros_like(b_w)
    ws = np.zeros_like(b_w)
    if strategy=='absolute' or not industry_neutral:   # 非行业中性：分别在整个组合内归一化
        sum_l = raw_l.sum(axis=1, keepdims=True)
        sum_s = raw_s.sum(axis=1, keepdims=True)
        wl = np.divide(raw_l, sum_l, out=np.zeros_like(raw_l), where=sum_l > 1e-12)
        ws = np.divide(raw_s, sum_s, out=np.zeros_like(raw_s), where=sum_s > 1e-12)
    else:
        # 行业中性：按Benchmark行业权重分配预算
        industry = base.fields["industry"].to_numpy(dtype=int, copy=False)
        groups = np.arange(31)
        # 股票所属行业掩码
        industry_mask = industry[:, :, None] == groups[None, None, :]
        # 各行业Benchmark权重及候选股票原始权重
        b_ind = (industry_mask * b_w[:, :, None]).sum(axis=1)
        l_ind = (industry_mask * raw_l[:, :, None]).sum(axis=1)
        s_ind = (industry_mask * raw_s[:, :, None]).sum(axis=1)
        if len(lg) > 0:
            l_total = np.take_along_axis(l_ind, industry, axis=1)
            l_budget = np.take_along_axis(b_ind, industry, axis=1)
            wl = np.divide(
                raw_l * l_budget,
                l_total,
                out=np.zeros_like(raw_l),
                where=l_total > 1e-12
            )
        if len(sg) > 0:
            s_total = np.take_along_axis(s_ind, industry, axis=1)
            s_budget = np.take_along_axis(b_ind, industry, axis=1)
            ws = np.divide(
                raw_s * s_budget,
                s_total,
                out=np.zeros_like(raw_s),
                where=s_total > 1e-12
            )

    # --------------------------------------------------
    # 6. 根据策略类型构建初始主动权重
    # --------------------------------------------------
    if strategy == "absolute":
        if construction == "long":
            init = wl.copy()
        elif construction == "short":
            init = -ws.copy()
        else:
            init = 0.5 * (wl - ws)

    elif strategy == "enhanced":
        if construction == "long":
            init = wl - b_w
        elif construction == "short":
            init = b_w - ws
        else:
            init = 0.5 * (wl - ws)

    elif strategy == "market_neutral":
        hedge_valid = np.ones(len(base.dates), dtype=bool)
        if hedge_method == "dollar":
            hedge = np.ones(len(base.dates))
        elif hedge_method == "beta":
            hedge, hedge_valid = estimate_hedge_ratio(
                base,
                wl,
                ws,
                b_w,
                construction,
                hedge_window,
                hedge_min_periods
            )
        if construction == "long":
            init = wl - hedge[:, None] * b_w
        elif construction == "short":
            init = hedge[:, None] * b_w - ws
        else:  # longshort
            init = wl - hedge[:, None] * ws

        valid = valid_b & hedge_valid

    init[~valid] = 0.

    # --------------------------------------------------
    # 9. 按目标主动敞口缩放
    # --------------------------------------------------
    gross = np.abs(init).sum(axis=1)
    scale = np.divide(active_gross, gross, out=np.zeros_like(gross), where=gross > 1e-12)

    if strategy == "enhanced":
        # 取所有股票允许的最大缩放系数中的最小值，保证所有股票都满足非负约束;
        # 保证Benchmark + Active之后的个股权重不为负
        feasible = np.divide(
            b_w,
            -init,
            out=np.full_like(b_w, np.inf),
            where=init < -1e-12
        )
        scale = np.minimum(scale, feasible.min(axis=1))

    active = init * scale[:, None]
    target = (
        b_w + active
        if strategy == "enhanced"
        else active
    )

    # --------------------------------------------------
    # 10. 保存结果
    # --------------------------------------------------
    base.long_target = pd.DataFrame(
        wl,
        index=base.dates,
        columns=base.ticks
    )

    base.short_target = pd.DataFrame(
        ws,
        index=base.dates,
        columns=base.ticks
    )

    base.benchmark_weight = pd.DataFrame(
        b_w,
        index=base.dates,
        columns=base.ticks
    )

    base.active_target = pd.DataFrame(
        active,
        index=base.dates,
        columns=base.ticks
    )

    base.target_holding = pd.DataFrame(
        target,
        index=base.dates,
        columns=base.ticks
    )

    base.hedge_ratio = pd.Series(
        np.where(
            hedge_valid & valid_b,
            hedge,
            np.nan
        ) if strategy == "market_neutral"
        else np.full(len(base.dates), np.nan),
        index=base.dates,
        name="hedge_ratio"
    )

    base.realized_active_gross = pd.Series(
        np.abs(active).sum(axis=1),
        index=base.dates
    )

    base.target_spec = (
        strategy,
        construction,
        group_numbers,
        normalization,
        domain,
        name,
        industry_neutral,
        active_gross,
        num_groups,
        hedge_method,
        hedge_window,
        hedge_min_periods
    )

    return base.target_holding
