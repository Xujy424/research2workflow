"""Guangfa 2026-05-11 p8, daily L2 active-side imbalance."""
from ..alphabase import AlphaBase, AlphaMeta
from ._informed import InformedContext

class VPINVolContext(InformedContext):
    pass

class VPINVolFactor(AlphaBase):
    meta = AlphaMeta('gf_vpin_vol', 'Daily absolute active buy/sell vol imbalance', direction=1)
    dependencies = ('l2/proc/shcj.pq', 'l2/proc/szcj.pq')

    def calculate(self, asof):
        return self.context.value(asof, 'vol')
