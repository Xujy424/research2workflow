from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import warnings
import tempfile
import unittest

import numpy as np
import pandas as pd

from v2.Monitor.dailyupdate.config import INDICATORS, MonitorConfig
from v2.Monitor.dailyupdate.context import MonitorContext, align_snapshot
from v2.Monitor.dailyupdate.dailyupdate import calculate_daily, update_daily, metric_path, normalize_snapshot
from v2.Monitor.dailyupdate.vals.growth_value_spread_percentile import growth_value_spread_percentile
from v2.UpdateData.axis.reset_axis import ensure_axis_capacity, insert_matrix_date_rows
from v2.UpdateData.axis.delete import delete_before


class MemoryPool:
    """Deterministic source fixture; attempts to read unknown fields fail."""
    def __init__(self, root, n=100, m=20):
        self.root = Path(root)
        self.asset_root = self.root / 'stock'
        dates = pd.bdate_range('2020-01-01', periods=n).to_numpy(dtype='datetime64[D]')
        ticks = np.array([f'{i:06d}' for i in range(m)])
        self.axis = SimpleNamespace(trade_dates=dates, ticks=ticks, tick_count=m,
                                    full_dates=np.r_[dates, np.full(3, np.datetime64('NaT'))], full_ticks=ticks)
        rng = np.random.default_rng(412)
        r = rng.normal(0, 1, (n, m))
        w = rng.uniform(1, 100, (n, m))
        masks = np.tile(np.r_[-np.ones(m//2), np.ones(m-m//2)], (n, 1))
        self.fields = {'d_essentials/pct': r, 'd_essentials/circ_mv': w,
                       'd_essentials/amount': w * 100, 'barra/growth1_mask': masks,
                       'basic/hastrade': np.ones_like(r), 'basic/hit_floor': (r < -1).astype(float),
                       'basic/price_floor': np.ones_like(r), 'industry/industry': np.tile(np.arange(m) % 2, (n, 1)),
                       'fundamental/valuation/pe_ttm': w, 'fundamental/valuation/pb_lf': w / 10}
        for name in ('hs300', 'zz500', 'zz1000', 'zz2000', 'zzfull'):
            self.fields[f'index/weight/{name}_weight'] = w
        for name in ('size', 'beta', 'btop', 'growth1', 'nonlinear_size', 'momentom', 'residual_vol', 'liquidity', 'leverage', 'earnings_yield'):
            self.fields[f'barra/{name}_mask'] = masks
        self.fields['valuation/pe_mask'] = masks
        self.fields['valuation/pb_mask'] = masks
        for size in ('lg', 'mid', 'sm', 'elg'):
            for side in ('buy', 'sell'):
                self.fields[f'd_moneyflow/{side}_{size}_amount'] = w * (2 if side == 'buy' else 1)
        self.read_calls = []

    def position(self, day):
        if isinstance(day, (int, np.integer)):
            return int(day)
        return int(np.flatnonzero(self.axis.trade_dates == np.datetime64(pd.Timestamp(day).date()))[0])

    def read(self, field, end, start=None, **kwargs):
        self.read_calls.append(field)
        end = self.position(end)
        return self.fields[field][end].copy() if start is None else self.fields[field][self.position(start):end+1].copy()

    def close(self):
        pass


def make_context(root):
    pool = MemoryPool(root)
    forbidden = Mock(side_effect=AssertionError('unexpected external request'))
    return MonitorContext(root=root, data=pool, wsd_reader=forbidden, edb_reader=forbidden, industry_map={'a': 0, 'b': 1})


class WindFixture:
    def __call__(self, **kwargs):
        dates = pd.bdate_range(kwargs['startDate'], kwargs['endDate'])
        level = 100 if kwargs['indicator'] == 'close' else 25
        return pd.Series(level + np.arange(len(dates)) / 100, index=dates)


def make_full_context(root):
    """All source interfaces are represented, without remote requests."""
    c = make_context(root)
    c.data.fields['m_essentials/amount'] = c.data.fields['d_essentials/amount']
    c.wsd_reader = WindFixture()
    c.edb_reader = Mock(return_value=pd.Series(2 + np.arange(len(c.dates)) / 100, index=c.dates))
    rng = np.random.default_rng(8)
    mid = np.exp(np.cumsum(rng.normal(0, .001, (150, len(c.ticks))), axis=0)) * 100
    bid_qty, ask_qty = rng.uniform(1, 100, (2, 5, *mid.shape))
    quotes = dict(mid=mid, spread=np.full_like(mid, 2), ofi=rng.normal(0, 10, mid.shape),
                  bid_px=np.tile(mid[None] - .01, (5, 1, 1)), ask_px=np.tile(mid[None] + .01, (5, 1, 1)),
                  bid_qty=bid_qty, ask_qty=ask_qty, depth=bid_qty.sum(0) + ask_qty.sum(0),
                  best_imbalance=(bid_qty[0] - ask_qty[0]) / (bid_qty[0] + ask_qty[0]))
    c.aligned_snapshot = Mock(return_value=quotes)
    return c


class WorkflowTests(unittest.TestCase):
    def test_incomplete_liquidity_inputs_preserve_missing_output(self):
        with tempfile.TemporaryDirectory() as root:
            c = make_context(root)
            cfg = MonitorConfig(module_options={'risk': {'micro_result': {'相对价差分位': [.1, .2]}}})
            result = calculate_daily(c.dates[-1], c, names='liquidity_pressure_percentile', config=cfg)
            self.assertEqual(set(result.name), {'liquidity_pressure_percentile'})
            self.assertTrue(result[['value', 'percentile']].isna().all().all())
            c.close()

    def test_every_registered_indicator_calculates_and_writes(self):
        from v2.Monitor.dailyupdate.dailyupdate import write_daily
        with tempfile.TemporaryDirectory() as root:
            c = make_full_context(root)
            cfg = MonitorConfig(history=20, industries=('a', 'b'))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                result = calculate_daily(c.dates[-1], c, config=cfg)
            self.assertEqual(set(result['name']), set(INDICATORS))
            self.assertFalse(result.duplicated(['name', 'domain', 'period']).any())
            write_daily(c.dates[-1], result, root, c.data.axis.full_dates, config=cfg)
            files = list((Path(root) / cfg.output_folder).rglob('*.bin'))
            self.assertGreater(len(files), len(INDICATORS))
            self.assertTrue(all(path.stat().st_size == 4 * len(c.data.axis.full_dates) for path in files))
            for name in INDICATORS:
                row = result.loc[result.name.eq(name)].iloc[0]
                field = 'percentile' if np.isnan(row.value) else 'value'
                path = metric_path(root, name, row.domain, row.period, field, cfg)
                self.assertTrue(path.is_file(), f'{name}: no binary output')
            c.close()

    def test_every_indicator_can_run_on_its_own(self):
        with tempfile.TemporaryDirectory() as root:
            c = make_full_context(root)
            cfg = MonitorConfig(history=20, industries=('a', 'b'))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                for name in INDICATORS:
                    with self.subTest(name=name):
                        result = calculate_daily(c.dates[-1], c, names=name, config=cfg)
                        self.assertEqual(set(result['name']), {name})
            c.close()

    def test_window_variants_share_one_function_and_one_context(self):
        import inspect
        from v2.Monitor.dailyupdate import context as context_module
        for names in (('impact_path_10m', 'impact_path_120m'), ('impact_decay_1m', 'impact_decay_30m'),
                      ('rv5', 'rv60'), ('rv_ratio_5_20', 'rv_ratio_20_60'),
                      ('order_flow_autocorrelation_lag1', 'order_flow_autocorrelation_lag5')):
            left, right = (INDICATORS[name].load() for name in names)
            self.assertIs(left.func, right.func)
            self.assertNotEqual(left.keywords, right.keywords)
        classes = {item for item in vars(context_module).values()
                   if inspect.isclass(item) and item.__module__ == context_module.__name__}
        self.assertEqual(classes, {MonitorContext})

    def test_registry_and_single_indicator_reads(self):
        self.assertEqual(len(INDICATORS), 73)
        for spec in INDICATORS.values():
            self.assertTrue(callable(spec.load()))
            self.assertTrue((Path(__file__).parents[1] / 'vals' / (spec.function.split(':')[0] + '.py')).is_file())
        with tempfile.TemporaryDirectory() as root:
            c = make_context(root)
            date = c.dates[-1]
            cfg = MonitorConfig(history=20)
            result = calculate_daily(date, c, names=('growth_value_spread_percentile',), config=cfg)
            self.assertEqual(set(c.data.read_calls), {'d_essentials/pct', 'd_essentials/circ_mv', 'barra/growth1_mask'})
            for row in result.itertuples():
                expected = growth_value_spread_percentile(c.data.fields['barra/growth1_mask'], c.data.fields['d_essentials/pct']/100,
                                                          c.data.fields['d_essentials/circ_mv'], row.period, 20)
                self.assertEqual(row.percentile, expected)
            c.close()

    def test_binary_rerun_and_missing_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            c = make_context(root)
            date = c.dates[-1]
            result = update_daily(date, root, context=c, names='growth_value_spread_percentile')
            path = metric_path(root, result.iloc[0]['name'], '全市场', 1, 'percentile')
            values = np.fromfile(path, dtype=np.float32)
            self.assertEqual(len(values), len(c.data.axis.full_dates))
            self.assertTrue(np.isnan(values[:-4]).all())
            self.assertTrue(np.isnan(values[-3:]).all())
            c.data.fields['barra/growth1_mask'][:] = 0
            c.row.cache_clear()
            update_daily(date, root, context=c, names='growth_value_spread_percentile')
            self.assertTrue(np.isnan(np.fromfile(path, dtype=np.float32)).all())
            with self.assertRaises(ValueError):
                update_daily('2030-01-01', root, context=c, names='growth_value_spread_percentile')
            c.close()

    def test_axis_resize_insert_and_prune(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root/'axis').mkdir()
            dates = np.array(['2020-01-01', '2020-01-03', 'NaT'], dtype='datetime64[D]')
            np.save(root/'axis/dates.npy', dates)
            np.save(root/'axis/stock_ticks.npy', np.array(['000001', ''], dtype='<U6'))
            path = metric_path(root, 'growth_value_spread_percentile', '全市场', 1, 'percentile')
            path.parent.mkdir(parents=True)
            np.array([.2, .8, np.nan], dtype=np.float32).tofile(path)
            close = root/'stock/d_essentials/close.bin'
            close.parent.mkdir(parents=True)
            np.array([[1,np.nan],[2,np.nan],[np.nan,np.nan]], dtype=np.float32).tofile(close)
            size = ensure_axis_capacity(root, min_date_free=5, min_tick_free=5)
            self.assertEqual(path.stat().st_size, size.new_date_len*4)
            insert_matrix_date_rows(root, 1, 2, size.new_date_len, size.new_tick_len)
            axis = np.load(root/'axis/dates.npy')
            axis[2], axis[1] = axis[1], np.datetime64('2020-01-02')
            np.save(root/'axis/dates.npy', axis)
            values = np.fromfile(path,dtype=np.float32)
            np.testing.assert_allclose(values[:3], [.2,np.nan,.8], equal_nan=True)
            delete_before('2020-01-02',root,date_reserve=2,tick_reserve=1)
            values = np.fromfile(path,dtype=np.float32)
            self.assertEqual(len(values),4)
            np.testing.assert_allclose(values[:2], [np.nan,.8],equal_nan=True)

    def test_external_cache_is_truncated(self):
        with tempfile.TemporaryDirectory() as root:
            c = make_context(root)
            reader = Mock(return_value=pd.Series([1.,2.,3.],index=pd.to_datetime(['2020-01-01','2020-01-02','2020-01-03'])))
            c.edb_reader = reader
            early = c.edb('test','2020-01-01','2020-01-02')
            late = c.edb('test','2020-01-01','2020-01-03')
            self.assertEqual(early.tolist(), [1.,2.])
            self.assertEqual(late.tolist(), [1.,2.,3.])
            reader.assert_called_once()
            c.close()

    def test_macro_only_selected_source(self):
        with tempfile.TemporaryDirectory() as root:
            c = make_context(root)
            c.edb_reader = Mock(return_value=pd.Series(np.arange(len(c.dates))/100+2,index=c.dates))
            result = calculate_daily(c.dates[-1],c,names='macro_dr007')
            c.edb_reader.assert_called_once_with('M1006337',IDname='value')
            self.assertEqual(set(result.period), {0,5,20})
            self.assertEqual(result.loc[result.period.eq(0),'value'].iloc[0],2.99)
            c.close()

    def test_rv_only_does_not_request_option_surface(self):
        with tempfile.TemporaryDirectory() as root:
            c = make_context(root)
            result = calculate_daily(c.dates[-1],c,names='rv20')
            self.assertEqual(len(result),1)
            self.assertEqual(result.iloc[0]['period'],1)
            self.assertTrue(np.isfinite(result.iloc[0]['value']))
            c.wsd_reader.assert_not_called()
            c.close()

    def test_standardized_acd_alone_loads_quotes(self):
        with tempfile.TemporaryDirectory() as root:
            c = make_context(root)
            c.aligned_snapshot = Mock(return_value={'mid': np.ones((150, len(c.ticks))) * 100})
            result = calculate_daily(c.dates[-1], c, names='standardized_acd_intensity')
            self.assertEqual(len(result), 4)
            self.assertEqual(c.aligned_snapshot.call_count, len(c.dates))
            self.assertTrue(result.percentile.isna().all())
            c.close()


if __name__ == '__main__':
    unittest.main()
