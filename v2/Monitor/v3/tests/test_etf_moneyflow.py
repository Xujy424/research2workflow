"""ETF净申赎：披露日对齐、最新可知状态与金额汇总。"""
import unittest

import numpy as np
import pandas as pd

from v2.Monitor.v3.moneyflow import etf_net_subscription_ratio


class ETFContext:
    def __init__(self):
        self.dates = pd.bdate_range('2026-01-01', periods=7)
        shares, nav = [], []
        for code, values, price in [(1, [100, 110, 110, 120, 120, 120], 2.),
                                     (2, [200, 200, 180, 180, 180, 180], 1.)]:
            changes = np.diff(values, prepend=values[0])
            for i, (value, change) in enumerate(zip(values, changes)):
                shares.append((code, self.dates[i + 1], self.dates[i + 1], value, change, pd.NaT))
                nav.append((code, self.dates[i], self.dates[i + 1], price, pd.NaT))
        self.shares = pd.DataFrame(shares, columns=[
            'InnerCode', 'EndDate', 'InfoPublDate', 'FloatShares', 'SharesChange',
            'LastOperationDate'])
        self.nav = pd.DataFrame(nav, columns=[
            'InnerCode', 'EndDate', 'InfoPublDate', 'UnitNV', 'LastOperationDate'])
        self.queries = []

    def get_position(self, date):
        return self.dates.searchsorted(date, side='right') - 1

    def jy(self, sql, params):
        self.queries.append((sql, params))
        data = self.shares if 'MF_SharesChange' in sql else self.nav
        if len(params) > 2:
            data = data[data.InnerCode.isin(params[2:])]
        return data.copy()


class ETFTests(unittest.TestCase):
    def test_later_revisions_do_not_replace_first_disclosure(self):
        c = ETFContext()
        expected = etf_net_subscription_ratio(c, c.dates[-1])
        revision = c.shares.iloc[[1]].copy()
        revision['InfoPublDate'] = c.dates[-1]
        revision['FloatShares'] = 999
        revision['SharesChange'] = 999
        c.shares = pd.concat([revision, c.shares], ignore_index=True)
        revision = c.nav.iloc[[1]].copy()
        revision['InfoPublDate'] = c.dates[-1]
        revision['UnitNV'] = 999
        c.nav = pd.concat([c.nav, revision], ignore_index=True)
        actual = etf_net_subscription_ratio(c, c.dates[-1])
        pd.testing.assert_series_equal(actual, expected)

    def test_amount_weighting_and_multi_day_initial_aum(self):
        c = ETFContext()
        result = etf_net_subscription_ratio(c, c.dates[-1])
        self.assertTrue(result.iloc[:2].isna().all())
        self.assertAlmostEqual(result.iloc[2], 20 / 400)
        self.assertAlmostEqual(result.iloc[3], -20 / 420)
        multi = etf_net_subscription_ratio(c, c.dates[-1], period=2)
        self.assertAlmostEqual(multi.iloc[3], 0)
        self.assertAlmostEqual(multi.iloc[4], 0)
        self.assertIn("s.StatPeriod='996'", c.queries[0][0])
        self.assertNotIn('s.SharesChange', c.queries[0][0])

    def test_nav_gains_are_not_subscriptions_and_group_is_explicit(self):
        c = ETFContext()
        c.shares.loc[:, 'SharesChange'] = 0
        c.shares['FloatShares'] = c.shares.groupby('InnerCode')['FloatShares'].transform('first')
        c.nav['UnitNV'] = np.arange(len(c.nav)) + 1.
        result = etf_net_subscription_ratio(c, c.dates[-1], 'index', 'hs300', fund_codes=[1])
        np.testing.assert_allclose(result.dropna(), 0)
        self.assertEqual(c.queries[0][1][-1], 1)
        with self.assertRaisesRegex(ValueError, 'fund_codes'):
            etf_net_subscription_ratio(c, c.dates[-1], 'index', 'hs300')

    def test_shares_change_is_used_and_late_older_nav_does_not_overwrite_latest(self):
        c = ETFContext()
        c.shares.loc[(c.shares.InnerCode == 1) & (c.shares.EndDate == c.dates[2]), 'SharesChange'] = 999
        result = etf_net_subscription_ratio(c, c.dates[-1], fund_codes=[1])
        self.assertAlmostEqual(result.iloc[2], .1)
        c = ETFContext()
        c.nav.loc[(c.nav.InnerCode == 1) & (c.nav.EndDate == c.dates[1]), 'InfoPublDate'] = c.dates[4]
        c.nav.loc[(c.nav.InnerCode == 1) & (c.nav.EndDate == c.dates[1]), 'UnitNV'] = 999
        result = etf_net_subscription_ratio(c, c.dates[-1], fund_codes=[1])
        self.assertAlmostEqual(result.iloc[2], .1)
        self.assertEqual(result.iloc[3], 0)
        self.assertAlmostEqual(result.iloc[4], 10 / 110)

    def test_missing_share_observation_moves_difference_to_next_disclosure(self):
        c = ETFContext()
        c.shares = c.shares[~((c.shares.InnerCode == 1) & (c.shares.EndDate == c.dates[2]))]
        result = etf_net_subscription_ratio(c, c.dates[-1], fund_codes=[1])
        self.assertTrue(np.isnan(result.iloc[2]))
        self.assertAlmostEqual(result.iloc[3], .1)

    def test_float_shares_delta_is_used_when_share_change_field_is_empty(self):
        c = ETFContext()
        c.shares['SharesChange'] = np.nan
        result = etf_net_subscription_ratio(c, c.dates[-1])
        self.assertAlmostEqual(result.iloc[2], 20 / 400)
        self.assertAlmostEqual(result.iloc[3], -20 / 420)

    def test_terminated_fund_aum_is_not_filled_forward(self):
        c = ETFContext()
        c.shares.loc[c.shares.InnerCode == 1, 'LastOperationDate'] = c.dates[3]
        c.nav.loc[c.nav.InnerCode == 1, 'LastOperationDate'] = c.dates[3]
        c.shares = c.shares.loc[(c.shares.InnerCode != 1) |
                                (c.shares.EndDate <= c.dates[3])].copy()
        c.nav = c.nav.loc[(c.nav.InnerCode != 1) |
                          (c.nav.EndDate <= c.dates[3])].copy()
        c.shares.loc[(c.shares.InnerCode == 2) & (c.shares.EndDate == c.dates[5]),
                     'FloatShares'] = 189
        result = etf_net_subscription_ratio(c, c.dates[-1])
        self.assertAlmostEqual(result.iloc[5], 9 / 180)

    def test_weekend_publication_maps_to_next_trading_date(self):
        c = ETFContext()
        # 2026-01-03为周六，公告只能从周一(交易轴第3日)进入矩阵。
        row = (c.shares.InnerCode == 1) & (c.shares.EndDate == c.dates[2])
        c.shares.loc[row, 'InfoPublDate'] = pd.Timestamp('2026-01-03')
        result = etf_net_subscription_ratio(c, c.dates[-1], fund_codes=[1])
        self.assertTrue(np.isnan(result.iloc[0]))
        self.assertTrue(np.isnan(result.iloc[1]))
        self.assertAlmostEqual(result.iloc[2], .1)


if __name__ == '__main__':
    unittest.main()
