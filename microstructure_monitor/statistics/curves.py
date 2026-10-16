"""Scope agreement and correlations recomputed from pooled minute flows."""
import numpy as np
import pandas as pd
from ..features.imbalance import safe_correlation

PAIRS = (("small", "medium"), ("small", "large"), ("small", "xl"),
         ("medium", "large"), ("medium", "xl"), ("large", "xl"))


def curve_statistics(curve):
    result = curve.copy().sort_index()
    times = pd.Series(pd.to_datetime(result.index.astype(str), format="%H:%M"), index=result.index)
    adjacent = (times.shift(-1)-times).eq(pd.Timedelta("1min"))
    daily = {}
    for first, second in PAIRS:
        a, b = "flow."+first+"_flow", "flow."+second+"_flow"
        if a not in result or b not in result:
            continue
        product = result[a]*result[b]
        valid = product.notna() & product.ne(0)
        for name, condition in (("agreement", product.gt(0)), ("divergence", product.lt(0))):
            column = f"flow.{first}_{second}_{name}"
            result[column] = condition.astype(float).where(valid)
            daily[column] = result[column].mean()
    for alias, original in (("direction_agreement_rate", "small_xl_agreement"), ("direction_divergence_rate", "small_xl_divergence")):
        key = "flow."+original
        if key in daily:
            daily["flow."+alias] = daily[key]
    if {"flow.small_normalized", "flow.xl_normalized"}.issubset(result.columns):
        product = result["flow.small_normalized"]*result["flow.xl_normalized"]
        result["flow.small_xl_aligned"] = product.gt(0).astype(float).where(product.notna())
    if {"order.imbalance", "flow.imbalance"}.issubset(result.columns):
        oi, ti = result["order.imbalance"], result["flow.imbalance"]
        comparisons = {"oi_ti_corr": (oi, ti), "oi_ti_lead_corr": (oi, ti.shift(-1).where(adjacent))}
        if "return" in result:
            comparisons["flow_return_lead_corr"] = (ti, result["return"].shift(-1).where(adjacent))
        daily["flow.oi_ti_samples"] = int((oi.notna() & ti.notna()).sum())
        # Rolling estimates restart at missing minutes as well as session boundaries.
        segments = times.diff().ne(pd.Timedelta("1min")).cumsum()
        for name, (a, b) in comparisons.items():
            daily["flow."+name] = safe_correlation(a, b)
            values = pd.Series(np.nan, index=result.index)
            for _, indices in segments.groupby(segments).groups.items():
                values.loc[indices] = a.loc[indices].rolling(20, min_periods=10).corr(b.loc[indices])
            result["flow."+name] = values
    return result, pd.Series(daily, dtype=float)
