"""Cross-sectional distribution summaries and universe differences."""
import pandas as pd


def summarize_distribution(features):
    numeric = features.select_dtypes("number")
    result = pd.DataFrame({"mean": numeric.mean(), "median": numeric.median(), "std": numeric.std(), "skew": numeric.skew(), "kurt": numeric.kurt(), "max": numeric.max()})
    for p in (.5, .9, .95, .99):
        result[f"p{int(p*100)}"] = numeric.quantile(p)
    result["tail_ratio"] = result.p99 / result.p50.replace(0, float("nan"))
    return result


def universe_differences(cross_section, reference=None):
    rows = {}
    names = list(cross_section.index)
    reference = reference or ("hs300" if "hs300" in names else "all")
    if reference not in names:
        return pd.DataFrame()
    for name in names:
        if name != reference:
            rows[f"{name}-{reference}"] = cross_section.loc[name] - cross_section.loc[reference]
    return pd.DataFrame.from_dict(rows, orient="index")
