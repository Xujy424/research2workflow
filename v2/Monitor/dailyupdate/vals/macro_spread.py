"""宏观利差：左右序列均为参数。"""
from ..utils import aligned_spread


def calculate(raw, left, right):
    # 先对齐日期，仅向前填充已有观测，再计算左侧减右侧利率。
    return aligned_spread(raw[left], raw[right])
