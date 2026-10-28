"""兼容旧导入；新增调用请使用 config 注册表及 dailyupdate.update_daily。"""
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = 'v2.Monitor.dailyupdate'
from .config import *
from .utils import _sum_matrices
from .dailyupdate import _metric_value
from .context import MonitorContext as MoneyFlowContext
from .utils import _percentile
from .vals.amount_change import amount_change
from .vals.top_share import top_share
from .vals.turnover import turnover
from .vals.active_buy_ratio import active_buy_ratio
from .vals.balance_change import balance_change
from .vals.category_share import calculate as category_share
from .utils import threshold_conclusion
from .utils import trend_conclusion
from .dailyupdate import MoneyFlowMonitor
