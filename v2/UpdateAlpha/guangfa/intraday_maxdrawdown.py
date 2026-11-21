from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from ..alphabase import AlphaContext
from ...GetData import DataPool
from ...UpdateData.config import ROOT

@dataclass(frozen=True)
class MinuteConfig:
    min_minutes: int = 120

    def __post_init__(self):
        if self.min_minutes < 2:
            raise ValueError('min_minutes must be >= 2')

def sample_variance(values, minimum):
    valid = np.isfinite(values)
    count = valid.sum(axis=0)
    mean = np.divide(np.where(valid, values, 0).sum(axis=0), count, out=np.zeros(count.shape, dtype=float), where=count > 0)
    ss = np.where(valid, values - mean, 0) ** 2
    return np.divide(ss.sum(axis=0), count - 1, out=np.full(count.shape, np.nan), where=count >= minimum)

def ratio(numerator, denominator):
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=np.isfinite(denominator) & (denominator > 0))

def moment(values, order):
    valid = np.isfinite(values)
    count = valid.sum(axis=0)
    mean = ratio(np.where(valid, values, 0).sum(axis=0), count)
    centered = np.where(valid, values - mean, 0)
    central_moment = ratio((centered ** order).sum(axis=0), count)
    return ratio(central_moment, sample_variance(values, 2) ** (order / 2))
'Guangfa 2023-03-10 pp8-9: signed intraday maximum drawdown.'
import numpy as np
from ..alphabase import AlphaBase, AlphaMeta

class IntradayMaxDrawdownContext(AlphaContext):

    def __init__(self, root=Path('Z:/') if Path('Z:/axis/dates.npy').is_file() else ROOT, config=MinuteConfig(), universe='self'):
        self.config = config
        super().__init__(DataPool(root, asset='stock'), universe=universe)

    def prices(self, asof):
        day = self.data.axis.date_position(pd.Timestamp(asof).date())
        values = np.asarray(self.data.read('m_essentials/close', day), dtype=float)
        if values.ndim == 3 and values.shape[0] == 1:
            values = values[0]
        if values.ndim != 2:
            raise ValueError('minute close must have shape (minutes, stocks)')
        return np.where(np.isfinite(values) & (values > 0), values, np.nan)

    def returns(self, asof):
        prices = self.prices(asof)
        return prices[1:] / prices[:-1] - 1

class IntradayMaxDrawdownFactor(AlphaBase):
    meta = AlphaMeta('gf_intraday_maxdrawdown', 'Signed minute-price maximum drawdown', direction=1)
    dependencies = ('m_essentials/close',)

    def calculate(self, asof):
        prices = self.context.prices(asof)
        valid = np.isfinite(prices)
        peak = np.maximum.accumulate(np.where(valid, prices, -np.inf), axis=0)
        drawdown = np.where(valid, prices / peak - 1, np.inf)
        result = drawdown.min(axis=0)
        enough = valid.sum(axis=0) >= self.context.config.min_minutes
        return np.where(enough & np.isfinite(result), result, np.nan).astype('float32')
