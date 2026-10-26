"""Slow independent pre-optimization reference for tests and benchmarks only."""
import numpy as np

def sparse_gamma(y, market, sparsity):
    """Global l0 fit of y = beta*abs(market) + diag(market**2)*gamma.

    A selected nonzero-market row can be fit exactly. Thus minimize the sum
    of the T-s smallest squared residuals over scalar beta (trimmed LS).
    Residual ordering changes only at pairwise intersections of squared
    residual lines. Enumerate those intervals, solve their quadratic minima,
    and choose the global minimum. No l1 surrogate or local sparse solver.
    """
    x = np.abs(np.asarray(market, float))
    y = np.asarray(y, float)
    gamma = np.zeros_like(y)
    eligible = x > 0
    a, b = (x[eligible], y[eligible])
    s = min(sparsity, max(0, len(a) - 1))
    if not len(a):
        return gamma
    keep = len(a) - s
    if s == 0:
        return gamma
    scale = a.max()
    a = a / scale
    lo, hi = (np.min(b / a), np.max(b / a))
    i, j = np.triu_indices(len(a), 1)
    roots = []
    for numerator, denominator in ((b[i] - b[j], a[i] - a[j]), (b[i] + b[j], a[i] + a[j])):
        nonzero = denominator != 0
        values = numerator[nonzero] / denominator[nonzero]
        roots.append(values[(values > lo) & (values < hi)])
    edges = np.unique(np.r_[lo, *roots, hi])
    if len(edges) == 1:
        beta = edges[0]
    else:
        best_loss, beta = (np.inf, lo)
        for start in range(0, len(edges) - 1, 256):
            stop = min(start + 256, len(edges) - 1)
            left, right = (edges[start:stop], edges[start + 1:stop + 1])
            mid = left + (right - left) / 2
            squared = (b[None, :] - mid[:, None] * a) ** 2
            selected = np.argpartition(squared, keep - 1, axis=1)[:, :keep]
            ax, by = (a[selected], b[selected])
            candidate = np.clip((ax * by).sum(1) / (ax * ax).sum(1), left, right)
            error = (b[None, :] - candidate[:, None] * a) ** 2
            loss = np.partition(error, keep - 1, axis=1)[:, :keep].sum(1)
            k = np.argmin(loss)
            if loss[k] < best_loss:
                best_loss, beta = (loss[k], candidate[k])
    residual = b - beta * a
    selected = np.argsort(residual ** 2, kind='stable')[-s:]
    positions = np.flatnonzero(eligible)[selected]
    gamma[positions] = residual[selected] / x[positions] ** 2
    gamma[positions[np.abs(residual[selected]) <= 1e-12 * max(1, np.max(np.abs(y)))]] = 0
    return gamma
