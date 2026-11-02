"""波动率风险溢价代理IV30²-RV30²; Context entry for fetching and calculating this indicator."""
import numpy as np
from ..config import MonitorConfig
from ..utils import metric_result
from ..utils import _p


def _compute(atm, close):
    # IV与RV统一使用年化小数；RV窗口沿用30个交易日，非严格的30日历日匹配。
    close = close.where(np.isfinite(close) & (close > 0)).sort_index()
    rv = close.pct_change(fill_method=None).rolling(30).std(ddof=1) * np.sqrt(252)
    return atm**2 - rv**2


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
        close = context.wsd(code, 'close', context.dates[max(0,start-65)], context.dates[end])
        values = _compute(atm, close).reindex(context.dates[start:end+1])
        rows.append(dict(domain=domain, period=1, value=float(values.iloc[-1]),
                         percentile=_p(values.to_numpy(),config.history)))
    return metric_result(rows)
