"""Cross-sectional distribution summaries and universe differences."""
import pandas as pd


def summarize_distribution(features):
    numeric = features.select_dtypes("number")
    result = pd.DataFrame({"mean": numeric.mean(), "median": numeric.median(), "std": numeric.std(), "skew": numeric.skew(), "kurt": numeric.kurt(), "max": numeric.max()})
    for p in (.5, .9, .95, .99):
        result[f"p{int(p*100)}"] = numeric.quantile(p)
    result["tail_ratio"] = result.p99 / result.p50.replace(0, float("nan"))
    return result


def universe_differences(cross_section):
    rows = {}
    names = list(cross_section.index)
    for i, first in enumerate(names):
        for second in names[i+1:]:
            rows[f"{second}-{first}"] = cross_section.loc[second] - cross_section.loc[first]
    return pd.DataFrame.from_dict(rows, orient="index")
