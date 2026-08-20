import numpy as np

from data_loader.data_loader import DataLoader


class StockMinLoader(DataLoader):
    def load(self, field):
        if field in self.data:
            return None
        data_shape = (
            len(self.axis_loader["trade_dates"]),
            241,
            len(self.axis_loader["stock_tick"]),
        )
        dtype = np.float32
        self.data[field] = np.memmap(
            self.data_path / self.sheet_path / (field + ".bin"),
            mode="r",
            dtype=dtype,
            shape=data_shape,
        )

    def read(self, field, end_di, start_di=None):
        if field not in self.data:
            self.load(field)
        if start_di is None:
            return self.data[field][end_di]
        else:
            return self.data[field][start_di : end_di + 1]

    def __getitem__(self, field):
        if field not in self.data:
            self.load(field)
        return self.data[field]
