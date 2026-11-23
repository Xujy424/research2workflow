"""Run the full SingleFactorTest FactorBase pipeline on smart_money.

From the repository root:
    python -m v2.SingleFactorTest.run
"""

from __future__ import annotations

if __package__:
    from .factorbase import FactorBase
else:
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from v2.SingleFactorTest.base.factorbase import FactorBase


FACTOR_NAME = "smart_money"
CONSTRUCTION="longshort"  # 构建：long / short / longshort
NORMALIZATION="factor"    # 归一化：equal / market_cap / factor
TRANSFORM="asinh"       # 线性：softsign / piecewise / rank_gaussian / yeo_johnson / none
NEURALIZE=True          # 行业市值中性化
STANDARDIZE=True        # 标准化
NUM_GROUPS=10           # 分组数
DIRECTION=-1             # 因子方向
START = None                 # e.g. "2020-01-01"
END = None                   # defaults to latest local trading date
HISTORY_YEARS = 6

HORIZONS = (1, 3, 5, 10, 20)

STRATEGY="enhanced"
G_IDS=(10, 1)
DOMAIN="universe"
DOMAIN_NAME=None

ACTIVE_GROSS=1.0
HEDGE_METHOD="dollar"
HEDGE_WINDOW=60
HEDGE_MIN_PERIOD=40

COMMISSION=0.0003
STAMP_TAX=0.0005
EXECUTION_PRICE="vwap"
CAPITAL=1e8
MAX_PARTICIPATION=0.1
IMPACT_COEFF=0.001
LOT_SIZE=100
BORROW_RATE=0.0
REBALANCE_DAYS=5



def create_base():
    base = FactorBase(
        factor_name=FACTOR_NAME,
        construction=CONSTRUCTION,  # long / short / longshort
        normalization=NORMALIZATION,    # equal / market_cap / factor
        transform=TRANSFORM,        # softsign / piecewise / rank_gaussian / yeo_johnson / none
        neutralize=NEURALIZE,
        standardize=STANDARDIZE,
        num_groups=NUM_GROUPS,
        direction=DIRECTION,
        start=START, end=END, history_years=HISTORY_YEARS,
    )

    try:
        base.load()
        print("Local data loaded.")
        base.process_alpha()
        print("Alpha processed.")
        base.cal_labels()
        base.cal_ics()
        base.cal_benchmark()
        base.cal_barra_ret()
        print("Relative data calculated.")
        base.cal_groups(
            num_groups=NUM_GROUPS, 
            domain=DOMAIN, 
            name=DOMAIN_NAME, 
            horizon=HORIZONS
        )
        print("Groups calculated.")
        base.cal_target_holding(
            strategy=STRATEGY,
            construction=CONSTRUCTION,
            num_groups=NUM_GROUPS,
            group_numbers=G_IDS,
            normalization=NORMALIZATION,
            domain=DOMAIN,
            name=DOMAIN_NAME,
            industry_neutral=NEURALIZE,
            active_gross=ACTIVE_GROSS,
            hedge_method=HEDGE_METHOD,
            hedge_window=HEDGE_WINDOW,
            hedge_min_periods=HEDGE_MIN_PERIOD,
        )
        print("Target holdings calculated.")
        base.cal_portfolio(
            strategy=STRATEGY,
            construction=CONSTRUCTION,
            num_groups=NUM_GROUPS,
            group_numbers=G_IDS,
            normalization=NORMALIZATION,
            domain=DOMAIN,
            name=DOMAIN_NAME,
            industry_neutral=NEURALIZE,
            active_gross=ACTIVE_GROSS,
            hedge_method=HEDGE_METHOD,
            hedge_window=HEDGE_WINDOW,
            hedge_min_periods=HEDGE_MIN_PERIOD,
            commission=COMMISSION,
            stamp_tax=STAMP_TAX,
            execution_price=EXECUTION_PRICE,
            capital=CAPITAL,
            max_participation=MAX_PARTICIPATION,
            impact_coefficient=IMPACT_COEFF,
            lot_size=LOT_SIZE,
            borrow_rate=BORROW_RATE,
            rebalance_days=REBALANCE_DAYS,
        )
        print("FactorBase pipeline completed.")
    finally:
        data = getattr(base, "data", None)
        if data is not None and hasattr(data, "close"):
            data.close()
    
    return base


if __name__ == "__main__":
    base = create_base()
    nav = base.portfolio.daily['nav']
    import matplotlib.pyplot as plt
    plt.plot(nav)
    plt.show()


