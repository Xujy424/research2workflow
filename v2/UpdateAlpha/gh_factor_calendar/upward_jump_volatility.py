"""Upward jump volatility from five-minute returns."""

import numpy as np

from ._intraday_common import CalendarIntradayContext, _CalendarIntradayFactor, _jump_components
from ..alphabase import AlphaMeta


class UpwardJumpVolatilityContext(CalendarIntradayContext):
    pass


class UpwardJumpVolatilityFactor(_CalendarIntradayFactor):
    meta = AlphaMeta("calendar_upward_jump_volatility", "20D average upward jump variance")
    dependencies = ("m_essentials/close",)

    def calculate(self, asof):
        returns = self.returns(asof)
        if returns is None:
            return self.empty()
        return self.aggregate(_jump_components(returns)[1])


__all__ = ["UpwardJumpVolatilityContext", "UpwardJumpVolatilityFactor"]
