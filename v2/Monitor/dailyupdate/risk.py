"""兼容旧导入；新增调用请使用 config 注册表及 dailyupdate.update_daily。"""
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = 'v2.Monitor.dailyupdate'
from .config import *
from .utils import _pct
from .utils import _p
from .vals.cross_section_dispersion import cross_section_dispersion
from .utils import _summarize
from .utils import _mean_pair_correlation
from .utils import _correlation_series
from .vals.average_correlation import average_correlation
from .vals.downside_correlation import downside_correlation
from .utils import _pca_series
from .vals.pca_explained_ratio import pca_explained_ratio
from .vals.downside_resonance import downside_resonance
from .vals.worst_rolling_loss import worst_rolling_loss
from .utils import _limit_down_series
from .vals.limit_down_ratio import limit_down_ratio
from .utils import volatility_risk_premium
from .vals.liquidity_pressure_percentile import liquidity_pressure_percentile
from .vals.realized_volatility import calculate as realized_volatility
from .utils import _rv_metrics
from .context import MonitorContext as RiskContext
from .dailyupdate import RiskMonitor
