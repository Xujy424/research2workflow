"""Factor-calendar implementation generated from its formula family."""

import numpy as np
from ..alphabase import AlphaMeta

from ._intraday_common import (
    CalendarIntradayContext, _CalendarIntradayFactor, _bar_returns, _daily_mean, _jump_components, _moment, _realized
)


class JumpIntensityContext(CalendarIntradayContext):
    pass


class JumpIntensityFactor(_CalendarIntradayFactor):
    meta = AlphaMeta("calendar_jump_intensity", "20D average RV-minus-BPV jump intensity")
    dependencies = ("m_essentials/close",)

    def calculate(self, asof):
        returns = self.returns(asof)
        if returns is None:
            return self.empty()
        rv, up, down = _jump_components(returns)
        return self.aggregate(np.divide(up + down, rv, out=np.full_like(rv, np.nan), where=rv > 0))


__all__ = ["JumpIntensityContext", "JumpIntensityFactor"]
