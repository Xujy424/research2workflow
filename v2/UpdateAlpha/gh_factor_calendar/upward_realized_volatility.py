"""Factor-calendar implementation generated from its formula family."""

import numpy as np
from ..alphabase import AlphaMeta

from ._intraday_common import (
    CalendarIntradayContext, _CalendarIntradayFactor, _bar_returns, _daily_mean, _jump_components, _moment, _realized
)


class UpwardRealizedVolatilityContext(CalendarIntradayContext):
    pass


class UpwardRealizedVolatilityFactor(_CalendarIntradayFactor):
    meta = AlphaMeta("calendar_upward_realized_volatility", "20D average upward realized semivariance")
    dependencies = ("m_essentials/close",)

    def calculate(self, asof):
        returns = self.returns(asof)
        if returns is None:
            return self.empty()
        return self.aggregate(np.sqrt(
            _realized(returns, lambda values: values > 0)
        ))


__all__ = ["UpwardRealizedVolatilityContext", "UpwardRealizedVolatilityFactor"]
