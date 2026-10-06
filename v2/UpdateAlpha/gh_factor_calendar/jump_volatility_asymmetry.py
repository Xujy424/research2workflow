"""Normalized upward/downward jump-volatility asymmetry."""

import numpy as np

from ._intraday_common import CalendarIntradayContext, _CalendarIntradayFactor, _jump_components
from ..alphabase import AlphaMeta


class JumpVolatilityAsymmetryContext(CalendarIntradayContext):
    pass


class JumpVolatilityAsymmetryFactor(_CalendarIntradayFactor):
    meta = AlphaMeta(
        "calendar_jump_volatility_asymmetry",
        "20D mean normalized upward-minus-downward jump variance",
    )
    dependencies = ("m_essentials/close",)

    def calculate(self, asof):
        returns = self.returns(asof)
        if returns is None:
            return self.empty()
        _, upward, downward = _jump_components(returns)
        daily = np.divide(
            upward - downward,
            upward + downward,
            out=np.full_like(upward, np.nan),
            where=(upward + downward) > 0,
        )
        return self.aggregate(daily)


__all__ = ["JumpVolatilityAsymmetryContext", "JumpVolatilityAsymmetryFactor"]
