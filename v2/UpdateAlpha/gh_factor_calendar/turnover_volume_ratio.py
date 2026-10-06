"""Factor-calendar implementation generated from its formula family."""

import numpy as np
from ..alphabase import AlphaMeta

from ._daily_common import (
    CalendarDailyContext, _CalendarDailyFactor, _mean, _sum
)


class TurnoverVolumeRatioContext(CalendarDailyContext):
    pass


class TurnoverVolumeRatioFactor(_CalendarDailyFactor):
    meta = AlphaMeta("calendar_volume_ratio", "Latest volume relative to its 20D mean")
    dependencies = ("d_essentials/volume",)

    def calculate(self, asof):
        raw = self.context.window(asof, self.dependencies)
        if raw is None:
            return self.empty()
        values = raw[self.dependencies[0]]
        baseline = _mean(values, self.context.config.min_valid_days)
        return self.finish(np.divide(values[-1], baseline, out=np.full_like(baseline, np.nan), where=baseline > 0))


__all__ = ["TurnoverVolumeRatioContext", "TurnoverVolumeRatioFactor"]
