"""Guangfa 2023-03-10, p7 sample variance; p9 negative RankIC."""
from ..alphabase import AlphaBase, AlphaMeta
from ._minute import MinuteContext, sample_variance

class RealVarContext(MinuteContext):
    pass

class RealVarFactor(AlphaBase):
    meta = AlphaMeta('gf_real_var', 'Daily minute-return sample variance', direction=-1)
    dependencies = ('m_essentials/close',)

    def calculate(self, asof):
        return sample_variance(self.context.returns(asof), self.context.config.min_minutes).astype('float32')
