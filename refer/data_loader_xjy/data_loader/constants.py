SHEET_LOADER_DICT = {
    "stock": {
        "m_essentials": "StockMinLoader",
        "d_essentials": "StockDailyLoader",
        "d_margin": "StockDailyLoader",
        "d_moneyflow": "StockDailyLoader",
        "d_industry": "StockDailyLoader",
        "d_basic": "StockDailyLoader",
        "f_balancesheet_merge": "StockFundLoader",
        "f_balancesheet": "StockFundLoader",
        "f_income_merge": "StockFundLoader",
        "f_income": "StockFundLoader",
        "f_cashflow_merge": "StockFundLoader",
        "f_cashflow": "StockFundLoader",
        "f_income_merge_season": "StockFundLoader",
        "f_income_season": "StockFundLoader",
        "f_cashflow_merge_season": "StockFundLoader",
        "f_cashflow_season": "StockFundLoader",
    }
}

DAILY_BOOL_LIST = [
    "tradable",
    "ceil_only",
    "floor_only",
    "has_trade",
    "is_new",
    "is_ST",
    "is_suspend",
]
