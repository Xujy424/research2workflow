"""监控指标注册、计算和日期轴二进制更新接口。"""
from .config import INDICATORS, IndicatorSpec, MonitorConfig, select_indicators
from .context import MonitorContext


def __getattr__(name):
    # 延迟导入编排器，避免 python -m 执行入口时产生重复导入警告。
    if name in {'calculate_daily', 'update_daily', 'metric_path', 'read_indicator'}:
        from importlib import import_module
        return getattr(import_module('.dailyupdate', __name__), name)
    raise AttributeError(name)


__all__ = ['INDICATORS', 'IndicatorSpec', 'MonitorConfig', 'select_indicators',
           'MonitorContext', 'calculate_daily', 'update_daily', 'metric_path', 'read_indicator']
