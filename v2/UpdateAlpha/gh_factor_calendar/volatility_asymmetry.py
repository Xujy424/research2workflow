"""Factor-calendar implementation generated from its formula family."""

import numpy as np
from ..alphabase import AlphaMeta

from ._intraday_common import (
    CalendarIntradayContext, _CalendarIntradayFactor, _bar_returns, _daily_mean, _jump_components, _moment, _realized
)


class VolatilityAsymmetryContext(CalendarIntradayContext):
    pass


class VolatilityAsymmetryFactor(_CalendarIntradayFactor):
    meta = AlphaMeta("calendar_volatility_asymmetry", "Upward minus downward realized semivariance")
    dependencies = ("m_essentials/close",)

    def calculate(self, asof):
        returns = self.returns(asof)
        if returns is None:
            return self.empty()
        upward = _realized(returns, lambda values: values > 0)
        downward = _realized(returns, lambda values: values < 0)
        daily = np.divide(
            upward - downward,
            upward + downward,
            out=np.full_like(upward, np.nan),
            where=(upward + downward) > 0,
        )
        return self.aggregate(daily)


__all__ = ["VolatilityAsymmetryContext", "VolatilityAsymmetryFactor"]
