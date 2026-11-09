"""Run with python -m v2.Monitor.tests.test_holdings_similarity."""
from time import perf_counter

import numpy as np
import polars as pl

from v2.Monitor.moneyflow import _holdings_similarity


def old_similarity(rows):
    n = rows['fund'].n_unique()
    if n < 2:
        return np.nan
    rows = rows.join(
        rows.group_by('fund').agg(pl.col('weight').sum().alias('total')),
        on='fund',
    )
    score = (
        rows.lazy().join(rows.lazy(), on='stock', suffix='_j')
        .filter(pl.col('fund') < pl.col('fund_j'))
        .group_by('fund', 'fund_j')
        .agg(pl.min_horizontal('weight', 'weight_j').sum().alias('overlap'),
             pl.col('total').first(), pl.col('total_j').first())
        .select((pl.col('overlap') /
                 (pl.col('total') + pl.col('total_j') - pl.col('overlap'))).sum())
        .collect().item()
    )
    return (score or 0.0) / (n * (n - 1) / 2)


def make_rows(weights):
    fund, stock = np.nonzero(weights)
    return pl.DataFrame({'fund': fund * 3 + 7, 'stock': stock,
                         'weight': weights[fund, stock]})


def test_exact_similarity():
    rng = np.random.default_rng(42)
    examples = [np.eye(5), np.ones((5, 4)), np.ones((1, 3)),
                np.zeros((0, 3)), np.array([[1., 2., 0.], [0., 2., 3.]])]
    for _ in range(20):
        w = rng.uniform(0.1, 5, (30, 40))
        w[rng.random(w.shape) < 0.8] = 0
        examples.append(w)
    for w in examples:
        rows = make_rows(w)
        expected = old_similarity(rows)
        for budget in (128, 32 * 1024 * 1024):
            np.testing.assert_allclose(_holdings_similarity(rows, budget),
                                       expected, atol=1e-14, equal_nan=True)


def benchmark():
    rng = np.random.default_rng(42)
    for n in (300, 1000, 2000, 5000):
        w = np.zeros((n, 400))
        probabilities = np.arange(400, 0, -1, dtype=float)
        probabilities /= probabilities.sum()
        for i in range(n):
            stocks = rng.choice(400, 30, replace=False, p=probabilities)
            w[i, stocks] = rng.uniform(0.1, 5, 30)
        rows = make_rows(w)
        start = perf_counter()
        expected = old_similarity(rows)
        old_time = perf_counter() - start
        start = perf_counter()
        actual = _holdings_similarity(rows)
        new_time = perf_counter() - start
        np.testing.assert_allclose(actual, expected, atol=1e-14)
        join_rows = sum(int(x) ** 2 for x in np.count_nonzero(w, axis=0))
        print(f'{n} funds: join={old_time:.3f}s new={new_time:.3f}s '
              f'speedup={old_time / new_time:.2f}x join_rows={join_rows:,}', flush=True)


if __name__ == '__main__':
    test_exact_similarity()
    print('Equivalence checks passed (including one-row blocks).', flush=True)
    benchmark()
