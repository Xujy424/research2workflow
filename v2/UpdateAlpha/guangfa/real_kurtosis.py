"""Guangfa 2023-03-10 pp7-9; raw daily statistic, report direction."""
import numpy as np
from ..alphabase import AlphaBase, AlphaMeta
from ._minute import MinuteContext, sample_variance, moment, ratio

class RealKurtosisContext(MinuteContext):
    pass

class RealKurtosisFactor(AlphaBase):
    meta = AlphaMeta('gf_real_kurtosis', 'real_kurtosis: daily minute-return statistic', direction=-1)
    dependencies = ('m_essentials/close',)

    def calculate(self, asof):
        r = self.context.returns(asof)
        value = moment(r, 4)
        enough = np.isfinite(r).sum(axis=0) >= self.context.config.min_minutes
        return np.where(enough, value, np.nan).astype('float32')
