from collections import defaultdict
from pathlib import Path

from data_loader.constants import SHEET_LOADER_DICT
from data_loader.field_loader import *


class DataPool(object):
    def __init__(self, data_path, sheet_loader_dict=SHEET_LOADER_DICT):
        if isinstance(data_path, str):
            self.data_path = Path(data_path)
        elif isinstance(data_path, Path):
            self.data_path = data_path
        else:
            raise ValueError("Please input a string or a pathlib object for data_path")
        self.loader_dict = sheet_loader_dict
        self.axis_loader = AxisLoader(self.data_path)
        self.pool = {}

    def load(self, asset):
        if asset not in self.pool:
            self.pool[asset] = AssetPool(self.data_path, self.loader_dict, self.axis_loader, asset)

    def read_data(self, asset, sheet, field, end_di, start_di=None):
        try:
            return self.pool[asset][sheet].read(field, end_di, start_di)
        except KeyError:
            self.load(asset)
            return self.pool[asset][sheet].read(field, end_di, start_di)

    def __getitem__(self, asset):
        if asset in self.axis_loader:
            return self.axis_loader[asset]
        if asset not in self.pool:
            self.pool[asset] = AssetPool(self.data_path, self.loader_dict, self.axis_loader, asset)
        return self.pool[asset]

    def get_field(self, asset, sheet, field):
        return self.pool[asset][sheet].get_field(field)


class AssetPool(object):
    def __init__(self, data_path, sheet_loader_dict, axis_loader, asset):
        self.data_path = data_path
        self.loader_dict = sheet_loader_dict
        self.axis_loader = axis_loader
        self.asset = asset
        self.pool = defaultdict(dict)

    def load(self, sheet):
        self.pool[sheet] = eval(self.loader_dict[self.asset][sheet])(
            self.data_path, Path(self.asset)/sheet, self.axis_loader
        )

    def __getitem__(self, sheet):
        if sheet not in self.pool:
            self.load(sheet)
        return self.pool[sheet]
