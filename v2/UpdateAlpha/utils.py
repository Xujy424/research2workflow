def inverse_u_transform_panel(
    frame: pl.DataFrame,
    column: str,
    date_col: str = "date",
    center: float = 0.70,
    method: str = "quadratic",
    alias: str = "factor_transformed",
):
    frame = frame.with_columns(
        (
            (
                pl.col(column)
                .rank(method="average")
                .over(date_col)
                - 0.5
            )
            /
            pl.col(column)
            .is_finite()
            .sum()
            .over(date_col)
        ).alias("_rank_pct")
    )

    if method == "quadratic":
        expr = -(pl.col("_rank_pct") - center).pow(2)
    elif method == "absolute":
        expr = -(pl.col("_rank_pct") - center).abs()
    else:
        raise ValueError(method)

    return (
        frame
        .with_columns(expr.alias(alias))
        .drop("_rank_pct")
    )



import numpy as np
import polars as pl


def fit_piecewise_mapping(
    frame: pl.DataFrame,
    factor_col: str,
    return_col: str,
    date_col: str = "date",
    n_bins: int = 20,
    smooth: bool = True,
):
    """
    用历史训练数据拟合：
        factor percentile -> expected future return

    Returns
    -------
    q_nodes:
        percentile 节点
    alpha_nodes:
        对应 expected return
    """

    df = (
        frame
        .filter(
            pl.col(factor_col).is_finite()
            & pl.col(return_col).is_finite()
        )
        .with_columns(
            (
                (
                    pl.col(factor_col)
                    .rank(method="average")
                    .over(date_col)
                    - 0.5
                )
                /
                pl.len().over(date_col)
            ).alias("_q")
        )
        .with_columns(
            (
                pl.col("_q") * n_bins
            )
            .floor()
            .clip(0, n_bins - 1)
            .cast(pl.Int32)
            .alias("_bin")
        )
    )

    stats = (
        df.group_by("_bin")
        .agg(
            pl.col(return_col).mean().alias("mean_ret"),
            pl.len().alias("count"),
        )
        .sort("_bin")
    )

    bins = stats["_bin"].to_numpy()
    mean_ret = stats["mean_ret"].to_numpy()

    # 理论中心点：
    # bin 0 -> 2.5%
    # bin 1 -> 7.5%
    # ...
    q_nodes = (bins + 0.5) / n_bins

    if smooth and len(mean_ret) >= 3:
        # 简单3点移动平均，减少单bin噪声
        smoothed = mean_ret.copy()

        smoothed[1:-1] = (
            mean_ret[:-2]
            + mean_ret[1:-1]
            + mean_ret[2:]
        ) / 3

        mean_ret = smoothed

    return q_nodes, mean_ret



