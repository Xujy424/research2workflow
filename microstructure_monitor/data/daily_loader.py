"""Read point-in-time metadata from v2 axis-aligned arrays, without databases."""
from pathlib import Path
import numpy as np
import pandas as pd


def load_v2_daily(root, date, symbols):
    root = Path(root)
    dates = np.load(root / "axis" / "dates.npy", allow_pickle=False).astype("datetime64[D]")
    ticks = np.load(root / "axis" / "stock_ticks.npy", allow_pickle=False).astype(str)
    positions = np.flatnonzero(dates == np.datetime64(pd.Timestamp(date).date()))
    if len(positions) != 1:
        raise ValueError("Trading date missing or duplicated in v2 axis")
    row = int(positions[0])
    tick_map = {tick.split(".")[0].zfill(6): i for i, tick in enumerate(ticks) if tick}
    requested = [tick_map.get(symbol.split(".")[0]) for symbol in symbols]
    result = pd.DataFrame({"symbol": symbols})
    shape = (len(dates), len(ticks))
    stock = root / "stock"
    def field(path, dtype=np.float32):
        if not path.exists():
            return None
        array = np.memmap(path, dtype=dtype, mode="r", shape=shape)
        values = np.array([array[row, i] if i is not None else np.nan for i in requested])
        del array
        return values
    for name, source in (("hs300", "hs300"), ("csi500", "zz500"), ("csi1000", "zz1000")):
        # Weight arrays preserve missing observations; masks alone cannot distinguish missing dates.
        values = field(stock / "index" / "weight" / f"{source}_weight.bin")
        if values is not None:
            result[name] = np.isfinite(values) & (values > 0)
    for name, relative in (("mcap", "d_essentials/total_mv.bin"), ("liquidity", "d_essentials/amount.bin"), ("industry", "industry/industry.bin"), ("open", "d_essentials/open.bin"), ("high", "d_essentials/high.bin"), ("low", "d_essentials/low.bin"), ("close", "d_essentials/close.bin")):
        values = field(stock / relative)
        if values is not None:
            result[name] = values
    if {"high", "low", "close"}.issubset(result):
        result["volatility"] = (result.high-result.low)/result.close.replace(0, np.nan)
    return result
