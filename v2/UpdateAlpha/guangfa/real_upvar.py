"""Guangfa 2023-03-10, pp7-9: conditional positive-return variance."""
import numpy as np
from ..alphabase import AlphaBase, AlphaMeta
from ._minute import MinuteContext, sample_variance

class RealUpVarContext(MinuteContext):
    pass

class RealUpVarFactor(AlphaBase):
    meta = AlphaMeta('gf_real_upvar', 'Daily positive minute-return sample variance', direction=-1)
    dependencies = ('m_essentials/close',)

    def calculate(self, asof):
        returns = self.context.returns(asof)
        enough = np.isfinite(returns).sum(axis=0) >= self.context.config.min_minutes
        value = sample_variance(np.where(returns > 0, returns, np.nan), 2)
        return np.where(enough, value, np.nan).astype('float32')
