"""Guangfa 2026-05-11 p8, daily L2 active-side imbalance."""
from ..alphabase import AlphaBase, AlphaMeta
from ._informed import InformedContext

class VPINAmtContext(InformedContext):
    pass

class VPINAmtFactor(AlphaBase):
    meta = AlphaMeta('gf_vpin_amt', 'Daily absolute active buy/sell amt imbalance', direction=1)
    dependencies = ('l2/proc/shcj.pq', 'l2/proc/szcj.pq')

    def calculate(self, asof):
        return self.context.value(asof, 'amt')
