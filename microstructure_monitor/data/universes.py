"""Point-in-time group memberships from the caller's daily security table."""


def build_universes(daily):
    result = {}
    if daily is None or daily.empty:
        return result
    table = daily.set_index("symbol")
    for index in ("hs300", "csi500", "csi1000"):
        if index in table:
            result[index] = table.index[table[index].fillna(False).astype(bool)].tolist()
    if "industry" in table:
        for industry, group in table.groupby("industry"):
            result[f"industry:{industry}"] = group.index.tolist()
    for column in ("mcap", "liquidity", "volatility"):
        if column in table:
            valid = table[column].dropna().sort_values()
            count = len(valid)
            if count >= 3:
                for name, begin, end in (("low", 0, count//3), ("mid", count//3, 2*count//3), ("high", 2*count//3, count)):
                    result[f"{column}:{name}"] = valid.index[begin:end].tolist()
    return result
