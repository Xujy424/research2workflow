import numpy as np

from data_loader.data_loader import DataLoader


class StockFundLoader(DataLoader):
    def load(self, field):
        dtype = float
        if field in self.data:
            return None

        data_shape = (
            len(self.axis_loader["quarter_dates"]),
            2,
            len(self.axis_loader["stock_tick"]),
        )

        if not hasattr(self, "ann_di"):
            self.ann_di = np.fromfile(self.data_path / self.sheet_path / "ann_di.bin", dtype=dtype).reshape(data_shape)

        arr = np.fromfile(self.data_path / self.sheet_path / (field + ".bin"), dtype=dtype).reshape(data_shape)
        self.data[field] = FundFieldLoader(arr, self.ann_di)

    def read(self, field, end_di, start_di=None):
        if field not in self.data:
            self.load(field)
        if start_di is not None:
            raise ValueError("Don't support start di parameter in fund data")
        return self.data[field][end_di]

    def __getitem__(self, field):
        if field not in self.data:
            self.load(field)
        return self.data[field]


class FundFieldLoader:
    def __init__(self, data, ann_di):   # data=arr
        self.data = data
        self.ann_di = ann_di

    def __getitem__(self, di):
        select_ix = self.ann_di <= di
        remove_ix = select_ix.any(axis=1).any(axis=1)

        f_data = self.data[remove_ix]    # data=arr

        select_ix = ~select_ix[remove_ix]
        adjust_ix = (~np.isnan(f_data[:, 1])) & (~select_ix[:, 1])

        result = f_data[:, 0].copy()
        result[select_ix[:, 0]] = np.nan
        result[adjust_ix] = f_data[:, 1][adjust_ix]
        return result
