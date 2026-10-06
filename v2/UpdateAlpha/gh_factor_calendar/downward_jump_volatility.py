"""Factor-calendar implementation generated from its formula family."""

import numpy as np
from ..alphabase import AlphaMeta

from ._intraday_common import (
    CalendarIntradayContext, _CalendarIntradayFactor, _bar_returns, _daily_mean, _jump_components, _moment, _realized
)


class DownwardJumpVolatilityContext(CalendarIntradayContext):
    pass


class DownwardJumpVolatilityFactor(_CalendarIntradayFactor):
    meta = AlphaMeta("calendar_downward_jump_volatility", "20D average downward jump variance")
    dependencies = ("m_essentials/close",)

    def calculate(self, asof):
        returns = self.returns(asof)
        if returns is None:
            return self.empty()
        return self.aggregate(_jump_components(returns)[2])


__all__ = ["DownwardJumpVolatilityContext", "DownwardJumpVolatilityFactor"]
