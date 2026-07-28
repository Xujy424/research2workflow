import os
from pathlib import Path
import numpy as np
import pandas as pd
import bisect
import bottleneck as bn


class FactorBase:
    ROOT = Path(os.environ.get("FACTOR_ROOT", "/data/xujiayi/xjy"))
    DATE_RESERVE = 500
    TICK_RESERVE = 1000

    def __init__(self, name, root=None):
        self.name = name
        self.root = Path(root) if root is not None else self.ROOT
        self.ticks = np.load(self.root / "axis/ticks.npy", allow_pickle=True)
        self.dates = np.load(self.root / "axis/dates.npy", allow_pickle=True)
        self.T = len(self.dates) + self.DATE_RESERVE
        self.N = len(self.ticks) + self.TICK_RESERVE
        self.path = self.root / "research_factors" / f"{name}.bin"

        if self.path.exists():
            self.arr = np.memmap(
                self.path, shape=(self.T, self.N),
                dtype=np.float32, mode="r+",
            )
        else:
            self.arr = np.full((self.T, self.N), np.nan, dtype=np.float32)

    def calc(self):
        """Return one day's factor as a Series indexed by security."""
        raise NotImplementedError("subclass must implement calc()")

    def add(self, dates):
        if np.isscalar(dates):
            dates = [dates]
        for date in dates:
            vals = self.calc(date)
            vals = vals.reindex(self.ticks).values.astype(np.float32).flatten()
            dt = bisect.bisect_left(self.dates, pd.to_datetime(date))
            self.arr[dt,:len(vals)] = vals
        if isinstance(self.arr, np.memmap):
            self.arr.flush()

    def get(self, dates):
        if np.isscalar(dates):
            dates = [dates]
        res = np.asarray(self.arr)[[bisect.bisect_left(self.dates, date) for date in dates]]
        return res

    def reset_index(self):
        """Extend the matrix only when its unused reserve is insufficient."""
        arr = np.asarray(self.arr)
        valid_dates = int(np.count_nonzero(~np.all(np.isnan(arr), axis=1)))
        valid_ticks = int(np.count_nonzero(~np.all(np.isnan(arr), axis=0)))

        new_T = max(arr.shape[0], valid_dates + self.DATE_RESERVE)
        new_N = max(arr.shape[1], valid_ticks + self.TICK_RESERVE)
        if (new_T, new_N) == arr.shape:
            return

        new_arr = np.full((new_T, new_N), np.nan, dtype=np.float32)
        new_arr[:arr.shape[0], :arr.shape[1]] = arr
        if isinstance(self.arr, np.memmap):
            self.arr.flush()

        self.arr = new_arr
        self.T, self.N = new_arr.shape

    def save(self):
        """Save the matrix as a float32 binary file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.asarray(self.arr, dtype=np.float32).tofile(self.path)

