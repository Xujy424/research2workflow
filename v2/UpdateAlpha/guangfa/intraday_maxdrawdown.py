"""Guangfa 2023-03-10 pp8-9: signed intraday maximum drawdown."""
import numpy as np
from ..alphabase import AlphaBase, AlphaMeta
from ._minute import MinuteContext

class IntradayMaxDrawdownContext(MinuteContext):
    pass

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
