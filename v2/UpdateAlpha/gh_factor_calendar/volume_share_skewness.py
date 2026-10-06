"""Factor-calendar implementation generated from its formula family."""

import numpy as np
from ..alphabase import AlphaMeta

from ._intraday_common import (
    CalendarIntradayContext, _CalendarIntradayFactor, _bar_returns, _daily_mean, _jump_components, _moment, _realized
)


class VolumeShareSkewnessContext(CalendarIntradayContext):
    pass


class VolumeShareSkewnessFactor(_CalendarIntradayFactor):
    meta = AlphaMeta("calendar_volume_share_skewness", "20D skewness of intraday volume shares")
    dependencies = ("m_essentials/volume",)

    def calculate(self, asof):
        raw = self.context.window(asof, self.dependencies)
        if raw is None:
            return self.empty()
        volume = raw[self.dependencies[0]]
        total = np.nansum(volume, axis=1, keepdims=True)
        share = np.divide(volume, total, out=np.full_like(volume, np.nan), where=total > 0)
        return self.aggregate(_moment(share, 3))


__all__ = ["VolumeShareSkewnessContext", "VolumeShareSkewnessFactor"]
