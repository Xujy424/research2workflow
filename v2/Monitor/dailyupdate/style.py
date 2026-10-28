"""兼容旧导入；新增调用请使用 config 注册表及 dailyupdate.update_daily。"""
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = 'v2.Monitor.dailyupdate'
from .config import *
from .vals.level_percentile import level_percentile
from .vals.relative_return import relative_return
from .vals.new_high_low_share import new_high_low_share
from .utils import harmonic_valuation
from .utils import _aggregated_percentile
from .dailyupdate import pe_percentile
from .dailyupdate import pb_percentile
from .vals.earnings_bond_spread import earnings_bond_spread
from .utils import index_eps_bond_spread
from .utils import index_pe_bond_spread
from .vals.amount_percentile import amount_percentile
from .vals.amount_share_percentile import amount_share_percentile
from .vals.industry_relative_return import industry_relative_return
from .utils import mask_spread_percentile
from .vals.size_spread_percentile import size_spread_percentile
from .vals.growth_value_spread_percentile import growth_value_spread_percentile
from .utils import pe_spread_percentile
from .utils import pb_spread_percentile
from .vals.valuation_spread_percentile import calculate as valuation_spread_percentile
from .vals.beta_spread_percentile import beta_spread_percentile
from .vals.rotation_speed_percentile import rotation_speed_percentile
from .utils import _series_percentile
from .dailyupdate import StyleMonitor
