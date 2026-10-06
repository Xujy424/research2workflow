"""Factor-calendar implementation generated from its formula family."""

import numpy as np
from ..alphabase import AlphaMeta

from ._daily_common import (
    CalendarDailyContext, _CalendarDailyFactor, _mean, _sum
)


class AverageTurnoverContext(CalendarDailyContext):
    pass


class AverageTurnoverFactor(_CalendarDailyFactor):
    meta = AlphaMeta("calendar_average_turnover", "20D average turnover")
    dependencies = ("d_essentials/turnover",)

    def calculate(self, asof):
        raw = self.context.window(asof, self.dependencies)
        return self.empty() if raw is None else self.finish(
            _mean(raw[self.dependencies[0]], self.context.config.min_valid_days)
        )


__all__ = ["AverageTurnoverContext", "AverageTurnoverFactor"]
