import pandas as pd


class DataLoader:
    def __init__(self, data_path, sheet_path, axis_loader):
        self.data_path = data_path
        self.sheet_path = sheet_path
        self.axis_loader = axis_loader
        self.data = dict()

    def load(self, field):
        raise NotImplementedError

    def read(self, field, end_di, start_di=None):
        raise NotImplementedError

    def __getitem__(self, field):
        raise NotImplementedError

    def keys(self):
        return self.data.keys()

    def get_field(self, field):
        arr = self.__getitem__(field)
        df = pd.DataFrame(
            arr,
            columns=self.axis_loader["stock_tick"],
            index=self.axis_loader["trade_dates"],
        )
        return df.dropna(how="all").dropna(how="all", axis=1)
