"""Factor-calendar implementation generated from its formula family."""

import numpy as np
from ..alphabase import AlphaMeta

from ._daily_common import (
    CalendarDailyContext, _CalendarDailyFactor, _mean, _sum
)


class TrendClarityContext(CalendarDailyContext):
    pass


class TrendClarityFactor(_CalendarDailyFactor):
    meta = AlphaMeta("calendar_trend_clarity", "Absolute net return divided by total path variation")
    dependencies = ("d_essentials/pct",)

    def calculate(self, asof):
        raw = self.context.window(asof, self.dependencies)
        if raw is None:
            return self.empty()
        returns = raw[self.dependencies[0]] / 100.0
        valid = np.isfinite(returns)
        numerator = np.abs(np.where(valid, returns, 0.0).sum(axis=0))
        denominator = np.where(valid, np.abs(returns), 0.0).sum(axis=0)
        result = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
        result[valid.sum(axis=0) < self.context.config.min_valid_days] = np.nan
        return self.finish(result)


__all__ = ["TrendClarityContext", "TrendClarityFactor"]
