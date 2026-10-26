import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from .update_iv30 import maturity_variance, update_iv30


class IV30Tests(unittest.TestCase):
    def test_wsd_discovery_without_contract_file(self):
        day = pd.Timestamp('2026-10-26')
        dates = np.array(['2026-10-26'], dtype='datetime64[D]')
        def reader(code, field, start, end):
            if code in ('000300.SH', '000852.SH'):
                value = 4000
            else:
                _, side, strike = code.removesuffix('.CFE').split('-')
                strike = int(strike)
                value = 20 + max(4000 - strike if side == 'C' else strike - 4000, 0)
            return pd.Series([value], index=[day])
        with tempfile.TemporaryDirectory() as tmp:
            result, metadata = update_iv30(day, dates, reader, tmp,
                                           flat_rate=0, price_field='close')
            self.assertTrue(np.isfinite(result.HS300_IV30.iloc[0]))
            self.assertEqual(metadata['contract_source'], 'wsd_probed_grid_proxy')
            self.assertEqual(metadata['expiry_method'], 'unverified_third_friday')

    def chain(self):
        rows = []
        for strike in range(60, 141, 10):
            for side in ('C', 'P'):
                price = 2 + max((100 - strike) if side == 'C' else (strike - 100), 0)
                rows.append(dict(strike=strike, cp=side, q=price, bid=price - 0.1))
        return pd.DataFrame(rows)

    def test_formula(self):
        result = maturity_variance(self.chain(), 0.1, 0)
        k = np.arange(60, 141, 10)
        expected = 2 / 0.1 * np.sum(10 * 2 / k**2)
        self.assertAlmostEqual(result['variance'], expected)
        self.assertEqual(result['k0'], 100)

    def test_two_zero_bids_stop_wing(self):
        chain = self.chain()
        chain.loc[(chain.cp == 'C') & chain.strike.isin([110, 120]), 'bid'] = 0
        with self.assertRaisesRegex(ValueError, 'insufficient_otm'):
            maturity_variance(chain, 0.1, 0, min_strikes=3)

    def test_update_and_no_extrapolation(self):
        day = pd.Timestamp('2026-10-26')
        dates = np.array(['2026-10-26', 'NaT'], dtype='datetime64[D]')
        rows = []
        quotes = {}
        for prefix in ('IO', 'MO'):
            for days in (20, 40):
                for i, row in self.chain().iterrows():
                    code = f'{prefix}-{days}-{i}'
                    quotes[code] = row.q
                    rows.append(dict(code=code, prefix=prefix, cp=row.cp, strike=row.strike,
                                     listed='2026-01-01', expiry=day + pd.Timedelta(days=days, hours=15)))
        contracts = pd.DataFrame(rows)
        def reader(code, field, start, end):
            return pd.Series([quotes[code]], index=[day])
        with tempfile.TemporaryDirectory() as tmp:
            kwargs = dict(contracts=contracts, flat_rate=0, price_field='close')
            result, meta = update_iv30(day, dates, reader, tmp, **kwargs)
            self.assertTrue(np.isfinite(result.HS300_IV30.iloc[0]))
            self.assertEqual(meta['method'], 'close_proxy')
            update_iv30(day, dates, reader, tmp, **kwargs)
            self.assertEqual(len(pd.read_csv(Path(tmp) / 'index_iv30.csv')), 1)
            kwargs['contracts'] = contracts.loc[contracts.expiry > day + pd.Timedelta(days=30)]
            result, meta = update_iv30(day, dates, reader, tmp, **kwargs)
            self.assertTrue(pd.isna(result.HS300_IV30.iloc[0]))
            self.assertEqual(meta['diagnostics'][0]['status'], 'no_bracketing_expiries')


if __name__ == '__main__':
    unittest.main()
