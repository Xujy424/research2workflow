"""Factor-calendar implementation generated from its formula family."""

import numpy as np
from ..alphabase import AlphaMeta

from ._daily_common import (
    CalendarDailyContext, _CalendarDailyFactor, _mean, _sum
)
from ..operators import safe_ratio_return


class OvernightReturnContext(CalendarDailyContext):
    pass


class OvernightReturnFactor(_CalendarDailyFactor):
    meta = AlphaMeta("calendar_overnight_return", "20D cumulative overnight return")
    dependencies = ("d_essentials/open_adj", "d_essentials/close_adj")

    def calculate(self, asof):
        raw = self.context.window(asof, self.dependencies, previous=True)
        if raw is None:
            return self.empty()
        values = safe_ratio_return(raw[self.dependencies[0]][1:], raw[self.dependencies[1]][:-1])
        return self.finish(_sum(values, self.context.config.min_valid_days))


__all__ = ["OvernightReturnContext", "OvernightReturnFactor"]
