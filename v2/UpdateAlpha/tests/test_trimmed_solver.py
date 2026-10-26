"""Independent exhaustive and old-implementation checks of the fast solver."""

import itertools
import unittest

import numpy as np

from v2.UpdateAlpha.operators import least_trimmed_squares_slope
from v2.UpdateAlpha.pricevolume.bayesHighVol_onset import sparse_gamma
from v2.UpdateAlpha.tests._sparse_reference import sparse_gamma as reference_gamma


class TrimmedSolverTests(unittest.TestCase):
    def test_signed_predictors_and_zero_rows_against_exhaustive_subsets(self):
        rng = np.random.default_rng(112)
        for _ in range(30):
            x, y = rng.normal(size=(2, 9))
            x[:2] = 0
            for trim in (0, 1, 3, 7):
                beta = least_trimmed_squares_slope(x, y, trim)
                actual = np.sort((y - beta * x) ** 2)[:len(x) - trim].sum()
                best = np.inf
                for keep in itertools.combinations(range(9), 9 - trim):
                    a, b = x[list(keep)], y[list(keep)]
                    denominator = np.dot(a, a)
                    slope = np.dot(a, b) / denominator if denominator > 0 else 0
                    best = min(best, np.sum((b - slope * a) ** 2))
                np.testing.assert_allclose(actual, best, rtol=1e-10, atol=1e-12)

    def test_minute_model_matches_old_global_enumeration(self):
        rng = np.random.default_rng(987)
        for case in range(24):
            market = rng.normal(0, 0.001, 240)
            target = np.abs(rng.normal(0, 0.003, 240))
            if case % 3 == 0:
                market[::11] = 0
            if case % 4 == 0:
                target[::7] = 0
            actual, expected = sparse_gamma(target, market, 24), reference_gamma(target, market, 24)
            np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-6)
            np.testing.assert_array_equal(actual > 0, expected > 0)
            np.testing.assert_array_equal(actual < 0, expected < 0)

    def test_degenerate_rows_and_work_limit(self):
        self.assertEqual(least_trimmed_squares_slope(np.zeros(5), np.arange(5), 2), 0)
        self.assertEqual(least_trimmed_squares_slope(np.arange(5), np.zeros(5), 2), 0)
        self.assertEqual(least_trimmed_squares_slope(np.ones(5), np.full(5, 3.), 2), 3)
        rng = np.random.default_rng(16)
        with self.assertRaisesRegex(RuntimeError, "uncertified"):
            least_trimmed_squares_slope(rng.normal(size=240), rng.normal(size=240), 24,
                                       max_intervals=1)


if __name__ == "__main__":
    unittest.main()
