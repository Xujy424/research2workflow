"""兼容旧导入；新增调用请使用 config 注册表及 dailyupdate.update_daily。"""
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = 'v2.Monitor.dailyupdate'
from .config import *
PERIODS = MACRO_PERIODS
from .utils import _clean
from .utils import level
from .utils import change
from .utils import change_months
from .utils import yoy
from .utils import yoy_series
from .utils import spread
from .utils import aligned_spread
from .utils import percentile
from .utils import rolling_change_percentile
from .utils import credit_impulse
from .context import MonitorContext as MacroContext
from .dailyupdate import MacroMonitor
