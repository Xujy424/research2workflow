"""Factor-calendar implementation generated from its formula family."""

import numpy as np
from ..alphabase import AlphaMeta

from ._daily_common import (
    CalendarDailyContext, _CalendarDailyFactor, _mean, _sum
)


class MaximumGainContext(CalendarDailyContext):
    pass


class MaximumGainFactor(_CalendarDailyFactor):
    meta = AlphaMeta("calendar_maximum_gain", "Maximum daily return over 20D")
    dependencies = ("d_essentials/pct",)

    def calculate(self, asof):
        raw = self.context.window(asof, self.dependencies)
        if raw is None:
            return self.empty()
        values = raw[self.dependencies[0]]
        enough = np.isfinite(values).sum(axis=0) >= self.context.config.min_valid_days
        result = np.max(np.where(np.isfinite(values), values, -np.inf), axis=0)
        result[~enough] = np.nan
        return self.finish(result)


__all__ = ["MaximumGainContext", "MaximumGainFactor"]
