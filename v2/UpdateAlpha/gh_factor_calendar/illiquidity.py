"""Factor-calendar implementation generated from its formula family."""

import numpy as np
from ..alphabase import AlphaMeta

from ._daily_common import (
    CalendarDailyContext, _CalendarDailyFactor, _mean, _sum
)


class IlliquidityContext(CalendarDailyContext):
    pass


class IlliquidityFactor(_CalendarDailyFactor):
    meta = AlphaMeta("calendar_illiquidity", "20D Amihud absolute return per traded amount")
    dependencies = ("d_essentials/pct", "d_essentials/amount")

    def calculate(self, asof):
        raw = self.context.window(asof, self.dependencies)
        if raw is None:
            return self.empty()
        amount = raw[self.dependencies[1]]
        daily = np.divide(
            np.abs(raw[self.dependencies[0]]) / 100.0,
            amount,
            out=np.full_like(amount, np.nan),
            where=np.isfinite(amount) & (amount > 0),
        )
        return self.finish(_mean(daily, self.context.config.min_valid_days))


__all__ = ["IlliquidityContext", "IlliquidityFactor"]
