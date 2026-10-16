"""Correlations with explicit pair-sample sufficiency."""
import numpy as np
import pandas as pd


def safe_correlation(first, second, minimum=3):
    pairs = pd.concat([first.rename("first"), second.rename("second")], axis=1).dropna()
    if len(pairs) < minimum or pairs["first"].std() == 0 or pairs["second"].std() == 0:
        return np.nan
    return pairs["first"].corr(pairs["second"])
