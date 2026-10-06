"""Factor-calendar implementation generated from its formula family."""

import numpy as np
from ..alphabase import AlphaMeta

from ._intraday_common import (
    CalendarIntradayContext, _CalendarIntradayFactor, _bar_returns, _daily_mean, _jump_components, _moment, _realized
)


class IntradayMaximumDrawdownContext(CalendarIntradayContext):
    pass


class IntradayMaximumDrawdownFactor(_CalendarIntradayFactor):
    meta = AlphaMeta("calendar_intraday_max_drawdown", "20D average intraday maximum drawdown")
    dependencies = ("m_essentials/close",)

    def calculate(self, asof):
        raw = self.context.window(asof, self.dependencies)
        if raw is None:
            return self.empty()
        price = raw[self.dependencies[0]]
        running_high = np.fmax.accumulate(np.where(np.isfinite(price), price, -np.inf), axis=1)
        drawdown = np.divide(price, running_high, out=np.full_like(price, np.nan), where=running_high > 0) - 1.0
        daily = np.min(np.where(np.isfinite(drawdown), drawdown, np.inf), axis=1)
        daily[~np.isfinite(daily)] = np.nan
        return self.aggregate(daily)


__all__ = ["IntradayMaximumDrawdownContext", "IntradayMaximumDrawdownFactor"]
