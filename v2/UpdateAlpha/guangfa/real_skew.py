"""Guangfa 2023-03-10 pp7-9; raw daily statistic, report direction."""
import numpy as np
from ..alphabase import AlphaBase, AlphaMeta
from ._minute import MinuteContext, sample_variance, moment, ratio

class RealSkewContext(MinuteContext):
    pass

class RealSkewFactor(AlphaBase):
    meta = AlphaMeta('gf_real_skew', 'real_skew: daily minute-return statistic', direction=-1)
    dependencies = ('m_essentials/close',)

    def calculate(self, asof):
        r = self.context.returns(asof)
        value = moment(r, 3)
        enough = np.isfinite(r).sum(axis=0) >= self.context.config.min_minutes
        return np.where(enough, value, np.nan).astype('float32')
