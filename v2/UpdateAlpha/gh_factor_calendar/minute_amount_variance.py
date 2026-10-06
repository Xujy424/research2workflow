"""Factor-calendar implementation generated from its formula family."""

import numpy as np
from ..alphabase import AlphaMeta

from ._intraday_common import (
    CalendarIntradayContext, _CalendarIntradayFactor, _bar_returns, _daily_mean, _jump_components, _moment, _realized
)


class MinuteAmountVarianceContext(CalendarIntradayContext):
    pass


class MinuteAmountVarianceFactor(_CalendarIntradayFactor):
    meta = AlphaMeta("calendar_minute_amount_variance", "20D mean variance of log minute amount")
    dependencies = ("m_essentials/amount",)

    def calculate(self, asof):
        raw = self.context.window(asof, self.dependencies)
        if raw is None:
            return self.empty()
        amount = raw[self.dependencies[0]]
        logged = np.where(amount > 0, np.log1p(amount), np.nan)
        return self.aggregate(np.nanvar(logged, axis=1, ddof=1))


__all__ = ["MinuteAmountVarianceContext", "MinuteAmountVarianceFactor"]
