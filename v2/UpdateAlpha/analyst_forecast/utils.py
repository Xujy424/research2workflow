"""Shared analyst-level aggregation helpers."""

from __future__ import annotations

import pandas as pd
import polars as pl


def _date(value):
    return pd.Timestamp(value).date()


def latest_analyst_values(frame, value=None, extra_keys=()):
    """Keep each analyst's latest finite value for every grouping key."""
    keys = ["tick", "author_id", *extra_keys]
    order = [
        column
        for column in [*keys, "create_date", "entrytime", "report_id", "id"]
        if column in frame.columns
    ]
    if value is not None:
        frame = frame.filter(pl.col(value).is_finite())
    return (
        frame.sort(order)
        .unique(keys, keep="last", maintain_order=True)
    )


def institution_values(frame, value, alias=None, extra_keys=()):
    """Equal-weight analysts within each institution."""
    alias = alias or value
    authors = latest_analyst_values(frame, value, extra_keys)
    return authors.group_by(
        ["tick", "organ_id", *extra_keys]
    ).agg(pl.col(value).mean().alias(alias))


def equal_weight(frame, value, alias="value", extra_keys=()):
    """Equal-weight analysts within institutions, then institutions."""
    institutions = institution_values(frame, value, value, extra_keys)
    return institutions.group_by(["tick", *extra_keys]).agg(
        pl.col(value).mean().alias(alias)
    )


def analyst_weight(frame, value, weights, alias="value", extra_keys=()):
    """Weight analysts within institutions, then equal-weight institutions."""
    if not isinstance(weights, pl.DataFrame):
        weights = pl.from_pandas(weights)
    missing = {"author_id", "weight"}.difference(weights.columns)
    if missing:
        raise ValueError(f"analyst weights missing columns: {sorted(missing)}")

    weights = weights.select(
        pl.col("author_id").cast(pl.Int64, strict=False),
        pl.col("weight").cast(pl.Float64, strict=False),
    ).unique("author_id", keep="last")
    authors = latest_analyst_values(frame, value, extra_keys).join(
        weights, on="author_id", how="left"
    ).with_columns(
        pl.when(pl.col("weight").is_finite() & (pl.col("weight") > 0))
        .then(pl.col("weight"))
        .otherwise(0.0)
        .alias("weight")
    )
    institutions = authors.group_by(
        ["tick", "organ_id", *extra_keys]
    ).agg(
        (pl.col(value) * pl.col("weight")).sum().alias("weighted_sum"),
        pl.col("weight").sum().alias("weight_sum"),
        pl.col(value).mean().alias("equal_value"),
    ).with_columns(
        pl.when(pl.col("weight_sum") > 0)
        .then(pl.col("weighted_sum") / pl.col("weight_sum"))
        .otherwise(pl.col("equal_value"))
        .alias(value)
    )
    return institutions.group_by(["tick", *extra_keys]).agg(
        pl.col(value).mean().alias(alias)
    )


def aggregate(frame, value, weights=None, alias="value", extra_keys=()):
    """Apply optional analyst weights and equal-weight institutions."""
    if weights is None:
        return equal_weight(frame, value, alias, extra_keys)
    return analyst_weight(frame, value, weights, alias, extra_keys)


__all__ = [
    "_date", "latest_analyst_values", "institution_values",
    "equal_weight", "analyst_weight", "aggregate",
]