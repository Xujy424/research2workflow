"""Factor-calendar implementation generated from its formula family."""

import numpy as np
from ..alphabase import AlphaMeta

from ._daily_common import (
    CalendarDailyContext, _CalendarDailyFactor, _mean, _sum
)


class PriceElasticityContext(CalendarDailyContext):
    pass


class PriceElasticityFactor(_CalendarDailyFactor):
    meta = AlphaMeta("calendar_price_elasticity", "20D price range per unit turnover")
    dependencies = (
        "d_essentials/high_adj", "d_essentials/low_adj",
        "d_essentials/close_adj", "d_essentials/turnover",
    )

    def calculate(self, asof):
        raw = self.context.window(asof, self.dependencies, previous=True)
        if raw is None:
            return self.empty()
        high, low = raw[self.dependencies[0]][1:], raw[self.dependencies[1]][1:]
        previous_close = raw[self.dependencies[2]][:-1]
        turnover = raw[self.dependencies[3]][1:]
        amplitude = np.divide(
            high - low, previous_close, out=np.full_like(high, np.nan),
            where=np.isfinite(previous_close) & (previous_close > 0),
        )
        elasticity = np.divide(
            amplitude, turnover, out=np.full_like(amplitude, np.nan),
            where=np.isfinite(turnover) & (turnover > 0),
        )
        return self.finish(_mean(elasticity, self.context.config.min_valid_days))


__all__ = ["PriceElasticityContext", "PriceElasticityFactor"]
