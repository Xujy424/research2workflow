"""Factor-calendar implementation generated from its formula family."""

import numpy as np
from ..alphabase import AlphaMeta

from ._intraday_common import (
    CalendarIntradayContext, _CalendarIntradayFactor, _bar_returns, _daily_mean, _jump_components, _moment, _realized
)


class TimeWeightedRelativePriceContext(CalendarIntradayContext):
    pass


class TimeWeightedRelativePriceFactor(_CalendarIntradayFactor):
    meta = AlphaMeta("calendar_time_weighted_relative_price", "20D mean intraday relative price position")
    dependencies = ("m_essentials/close",)

    def calculate(self, asof):
        raw = self.context.window(asof, self.dependencies)
        if raw is None:
            return self.empty()
        price = raw[self.dependencies[0]]
        high = np.fmax.accumulate(np.where(np.isfinite(price), price, -np.inf), axis=1)
        low = np.fmin.accumulate(np.where(np.isfinite(price), price, np.inf), axis=1)
        spread = high - low
        position = np.divide(price - low, spread, out=np.full_like(price, np.nan), where=spread > 0)
        return self.aggregate(np.nanmean(position, axis=1))


__all__ = ["TimeWeightedRelativePriceContext", "TimeWeightedRelativePriceFactor"]
