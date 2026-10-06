"""Factor-calendar implementation generated from its formula family."""

import numpy as np
from ..alphabase import AlphaMeta

from ._intraday_common import (
    CalendarIntradayContext, _CalendarIntradayFactor, _bar_returns, _daily_mean, _jump_components, _moment, _realized
)


class RealizedVolatilityContext(CalendarIntradayContext):
    pass


class RealizedVolatilityFactor(_CalendarIntradayFactor):
    meta = AlphaMeta("calendar_realized_volatility", "20D average intraday realized volatility")
    dependencies = ("m_essentials/close",)

    def calculate(self, asof):
        returns = self.returns(asof)
        return self.empty() if returns is None else self.aggregate(
            np.sqrt(_realized(returns))
        )


__all__ = ["RealizedVolatilityContext", "RealizedVolatilityFactor"]
