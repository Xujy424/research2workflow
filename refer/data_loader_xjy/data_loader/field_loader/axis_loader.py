import numpy as np

from data_loader.data_loader import DataLoader


class AxisLoader(DataLoader):
    def __init__(self, data_path):
        # 如果子类不显式调用父类的__init__，父类的构造函数不会自动执行
        # 没有调用 super().__init__()，所以 DataLoader.__init__() 不会执行
        self.data_path = data_path
        self.data = dict()
        self.load()

    def load(self):
        self.data["trade_dates"] = np.load(self.data_path / "axis" / "dates.npy", allow_pickle=True)
        self.data["quarter_dates"] = np.load(self.data_path / "axis" / "quarter_dates.npy", allow_pickle=True)

        self.data["stock_tick"] = np.load(self.data_path / "axis" / "stock_tick.npy")

    def read(self, field, end_di=None, start_di=None):
        return self.data[field]

    def __getitem__(self, field):
        return self.data[field]

    def __contains__(self, field):
        return field in self.data




