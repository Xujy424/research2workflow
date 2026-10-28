"""调和估值的历史分位：适用于PE和PB。"""
from ..utils import _aggregated_percentile, harmonic_valuation


def calculate(multiple_history, weight_history, history=252, period=1):
    # 每日按市值权重聚合倒数估值，再对period日均值求history日历史分位。
    return _aggregated_percentile(multiple_history, weight_history, harmonic_valuation, history, period)
