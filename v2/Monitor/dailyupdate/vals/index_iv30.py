"""指数隐含波动率IV30; Context entry for fetching and calculating this indicator."""
from ..config import MonitorConfig
from ..utils import metric_result
import numpy as np
from ..utils import _p


def _compute(atm):
    # 使用上下文提供的固定期限平值曲面；缺失当日报价不前填。
    return atm


def calculate(context, date, config=None):
    """从统一Context读取本指标输入，计算当日各域/周期的数值。"""
    config = config or MonitorConfig()
    start, end = context.window(date, config.history + 260)
    settings = config.module_options.get('risk', {}).get('option_settings', {})
    field = settings.get('surface_field', '品种隐含波动率曲面(%)')
    options = settings.get('surface_options', 'Period=D;期限={tenor}天;价值状态={moneyness}%').format(tenor=30, moneyness=100)
    scale = settings.get('surface_scale', 0.01)
    rows = []
    for code, domain in (('000300.SH', '沪深300'), ('000852.SH', '中证1000')):
        # 固定30天、平值曲面，不向前填补当日缺失报价。
        atm = context.wsd(code, field, context.dates[start], context.dates[end], options=options)*scale
        atm = atm.where(np.isfinite(atm) & (atm >= 0)).reindex(context.dates[start:end+1])
        values = _compute(atm)
        rows.append(dict(domain=domain, period=1, value=float(values.iloc[-1]),
                         percentile=_p(values.to_numpy(),config.history)))
    return metric_result(rows)
