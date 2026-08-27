"""Explicit pair portfolios; never mine pairs on the evaluation sample."""

from __future__ import annotations
import numpy as np
import pandas as pd
from .config import PairDefinition


def explicit_pair_weights(pair_signal: pd.DataFrame,
                          pairs: tuple[PairDefinition, ...], holding_days=5):
    """Convert one spread signal column per `left|right` pair to stock weights."""
    if holding_days < 1 or not pairs:
        raise ValueError("positive holding_days and explicit pairs are required")
    assets = sorted({x for pair in pairs for x in (pair.left, pair.right)})
    out = pd.DataFrame(0.0, index=pair_signal.index, columns=assets)
    active = []
    for t in range(len(pair_signal)):
        for pair in pairs:
            key = f"{pair.left}|{pair.right}"
            if key not in pair_signal:
                raise KeyError(f"missing pair signal: {key}")
            value = pair_signal.iloc[t][key]
            if np.isfinite(value) and value != 0:
                active.append((t + holding_days, pair, np.sign(value)))
        active = [item for item in active if item[0] > t]
        for _, pair, direction in active:
            scale = 1 / (1 + pair.hedge_ratio)
            out.iloc[t, out.columns.get_loc(pair.left)] += direction * scale
            out.iloc[t, out.columns.get_loc(pair.right)] -= (
                direction * pair.hedge_ratio * scale)
        gross = out.iloc[t].abs().sum()
        if gross > 0:
            out.iloc[t] /= gross
    return out
