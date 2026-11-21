"""作者：xujiayi | 创建日期：2026-11-11
单因子研究与日频影子成交，不包含事件驱动及配对交易。
"""
from .factorbase import FactorBase, factorbase
from .backtest import PortfolioResult
from . import maths

__all__ = ["FactorBase", "factorbase", "PortfolioResult", "maths"]
