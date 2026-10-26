"""Check that algebraic shortcuts preserve the original factor definitions."""

from dataclasses import replace
import importlib
import inspect
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from v2.UpdateAlpha import get_factor_spec
from v2.UpdateAlpha.operators import linear_quantile_residual
from v2.UpdateAlpha.tests._event_reference import onset_counts as reference_counts


class FormulaOptimizationTests(unittest.TestCase):
    def test_one_tail_lp_matches_original_two_tail_events(self):
        rng = np.random.default_rng(451)
        for name in ("highVol_onset", "lowVol_onset", "bayesHighVol_onset", "bayesLowVol_onset"):
            spec = get_factor_spec(name)
            module = importlib.import_module(spec.factor_class.__module__)
            cfg = replace(inspect.signature(spec.context_class).parameters["config"].default,
                          show_progress=False)
            for case in range(5):
                r, market = rng.normal(0, 0.001, (2, 240))
                v = rng.lognormal(6, 1, 240)
                if case % 2 == 0:
                    r[50:53] = np.nan
                    v[50:53] = 0
                    market[120] = np.nan
                good = np.isfinite(r) & np.isfinite(market)
                relative = v / np.mean(v[np.isfinite(r)])
                lag = np.r_[np.nan, relative[:-1]]
                qgood = good & np.isfinite(lag) & (lag > 0)
                t = np.arange(240)
                design = np.column_stack((np.ones(240), r, lag, t < 30, t >= 210))
                low, high = np.full(240, np.nan), np.full(240, np.nan)
                low[qgood] = linear_quantile_residual(relative[qgood], design[qgood], .2)
                high[qgood] = linear_quantile_residual(relative[qgood], design[qgood], .8)
                gamma = np.full(240, np.nan)
                gamma[good] = module.sparse_gamma(np.abs(r[good] - market[good]), market[good], cfg.sparse_minutes)
                group = 1 if "lowvol" in name.lower() else 0
                expected = reference_counts(gamma, low, high, cfg.horizon)[group]
                with patch.object(module, "linear_quantile_residual", wraps=linear_quantile_residual) as solve:
                    _, actual = module._stock_feature((0, r, v, market, cfg, None))
                self.assertEqual(solve.call_count, 1)
                np.testing.assert_array_equal(actual, expected)

    def test_batched_csad_matches_stockwise_lstsq_with_missing_samples(self):
        rng = np.random.default_rng(551)
        for name in ("lr_score_mor", "lr_beta2_aft_minus_mor"):
            spec = get_factor_spec(name)
            cfg = replace(inspect.signature(spec.context_class).parameters["config"].default,
                          show_progress=False)
            for case in range(3):
                n = 37
                close = 10 * np.cumprod(1 + rng.normal(0, .002, (241, n)), axis=0)
                volume = rng.lognormal(5, 1, (241, n))
                if case == 1:
                    close[5:12, :3] = np.nan
                    volume[90:93, 5] = 0
                    close[:26, 6] = np.nan
                if case == 2:
                    close[:] = 10  # singular, all-zero market-return design
                tradable = np.ones(n)
                tradable[-1] = 0
                fields = {"m_essentials/close": close, "m_essentials/volume": volume,
                          "basic/tradable": tradable}
                context = spec.context_class.__new__(spec.context_class)
                context.config, context._daily_cache = cfg, {}
                context.data = SimpleNamespace(axis=SimpleNamespace(tick_count=n),
                    read=lambda field, day, **kw: fields[field])
                actual = context.daily_features(0)[name]

                ret = close[1:] / close[:-1] - 1
                ret[volume[1:] <= 0] = np.nan
                valid = np.isfinite(ret) & (tradable == 1)
                count = valid.sum(1)
                market = np.divide(np.where(valid, ret, 0).sum(1), count,
                                   out=np.full(240, np.nan), where=count > 0)
                expected = np.full(n, np.nan)
                for stock in range(n - 1):
                    paired = np.isfinite(ret[:, stock]) & np.isfinite(market)
                    if paired.sum() < cfg.min_minutes:
                        continue
                    betas, score = [], np.nan
                    for group, (segment, threshold) in enumerate((
                            (np.arange(240) < 30, cfg.min_morning_minutes),
                            (np.arange(240) >= 30, cfg.min_minutes - 30))):
                        mask = paired & segment
                        if mask.sum() < max(3, threshold):
                            betas.append(np.nan)
                            continue
                        x, y = np.abs(market[mask]), np.abs(ret[mask, stock] - market[mask])
                        scale = x.max()
                        if scale <= 0:
                            betas.append(np.nan)
                            continue
                        design = np.column_stack((np.ones(len(x)), x / scale, (x / scale) ** 2))
                        coefficient, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
                        if rank < 3:
                            betas.append(np.nan)
                            continue
                        betas.append(coefficient[2] / scale ** 2)
                        if group == 0:
                            variance = np.sum((y - y.mean()) ** 2)
                            score = 1 - np.sum((y - design @ coefficient) ** 2) / variance if variance > 0 else np.nan
                    expected[stock] = score if name == "lr_score_mor" else betas[1] - betas[0]
                np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-7, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
