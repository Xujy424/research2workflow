"""信用脉冲：宏观输入序列派生。"""
from ..utils import credit_impulse


def calculate(raw):
    # 将已按日期对齐的原始序列按本指标口径组合；不在指标中访问数据服务。
    return credit_impulse(raw["新增社融"], raw["名义GDP"])
