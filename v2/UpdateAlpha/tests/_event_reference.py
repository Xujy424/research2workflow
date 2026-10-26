"""Pre-optimization two-tail event-count reference (tests only)."""
import numpy as np

def onset_counts(gamma, low, high, horizon):
    valid = np.isfinite(gamma) & np.isfinite(low) & np.isfinite(high)
    dispersion = gamma > 0
    base = valid[:-horizon] & ~dispersion[:-horizon]
    hit = np.zeros_like(base)
    for k in range(1, horizon + 1):
        base &= valid[k:len(valid) - horizon + k]
        hit |= dispersion[k:len(valid) - horizon + k]
    result = []
    for event in (high[:-horizon] > 0, low[:-horizon] < 0):
        e, c = (base & event, base & ~event)
        result.append((e.sum(), (e & hit).sum(), c.sum(), (c & hit).sum()))
    return result
