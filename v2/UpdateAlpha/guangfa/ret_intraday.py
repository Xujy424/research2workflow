"""Guangfa 2023-03-10 pp8-9, one of the twelve selected factors (p1)."""
import numpy as np
import pandas as pd
from ..alphabase import AlphaBase, AlphaMeta
from ._minute import MinuteContext


class RetIntradayContext(MinuteContext):
    def daily_prices(self, asof):
        day = self.data.axis.date_position(pd.Timestamp(asof).date())
        return tuple(np.asarray(self.data.read('d_essentials/' + field, day),
                                dtype=float).reshape(-1)
                     for field in ('open', 'close'))


class RetIntradayFactor(AlphaBase):
    meta = AlphaMeta('gf_ret_intraday', 'Daily close / open minus one', direction=-1)
    dependencies = ('d_essentials/open', 'd_essentials/close')

    def calculate(self, asof):
        opening, closing = self.context.daily_prices(asof)
        valid = np.isfinite(opening) & np.isfinite(closing) & (opening > 0) & (closing > 0)
        result = np.divide(closing, opening, out=np.full_like(opening, np.nan), where=valid) - 1
        return result.astype(np.float32)
