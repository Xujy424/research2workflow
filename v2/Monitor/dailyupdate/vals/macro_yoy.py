"""宏观同比：优先已有同比，缺失时使用余额同比。"""
from ..utils import yoy_series


def calculate(raw, yoy_name, balance_name):
    # 保留旧版选择规则，输入频率和发布时间口径需由数据源提供。
    return raw[yoy_name] if len(raw[yoy_name]) else yoy_series(raw[balance_name])
