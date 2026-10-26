"""Daily IO/MO IV30 proxies, stored as date-axis x 1 float32 matrices.

WSD-only discovery uses a bounded strike grid and third-Friday expiries.
Close/settlement prices and a constant rate are explicit approximations.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd

if __package__:
    from ..utils import asof, date_index, ensure_memmap
else:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from v2.UpdateData.utils import asof, date_index, ensure_memmap

INDEX_CODES = {"IO": "000300.SH", "MO": "000852.SH"}
OUTPUT_NAMES = {"IO": "HS300_IV30", "MO": "CSI1000_IV30"}
TARGET = 30 / 365


def quote(wsd, code, field, day):
    """Read one dated value without accepting stale or duplicate observations."""
    result = wsd(code, field, str(day.date()), str(day.date()))
    if isinstance(result, pd.DataFrame):
        if "date" in result:
            result = result.set_index("date")
        result = result[field] if field in result else result.squeeze("columns")
    if not isinstance(result, pd.Series):
        raise ValueError("WSD must return a dated Series or single-field DataFrame")
    mask = pd.to_datetime(result.index).tz_localize(None).normalize() == day
    values = result.loc[mask]
    if len(values) > 1:
        raise ValueError(f"duplicate WSD observations: {code}")
    return float(values.iloc[0]) if len(values) else np.nan


def expiry_candidates(day):
    """Near three months plus next three quarters; holiday shifts unverified."""
    months = []
    for offset in range(18):
        first = (day.to_period("M") + offset).to_timestamp()
        expiry = first + pd.Timedelta(days=(4 - first.weekday()) % 7 + 14)
        if expiry > day:
            months.append(expiry)
    near = months[:3]
    return sorted(near + [e for e in months if e.month in (3, 6, 9, 12) and e not in near][:3])


def maturity_variance(strikes, prices, term, rate, *, bids=None, min_strikes=8):
    """Pure NumPy VIX variance; prices/bids have columns [call, put]."""
    strikes = np.asarray(strikes, dtype=float)
    prices = np.asarray(prices, dtype=float)
    if (term <= 0 or not np.isfinite(rate) or len(strikes) < 3
            or prices.shape != (len(strikes), 2)
            or not np.isfinite(strikes).all() or np.any(strikes <= 0)
            or np.any(np.diff(strikes) <= 0)):
        return np.nan
    # The discovery grid also contains unlisted strikes (especially for MO).
    # Remove rows with no quote before locating K0 or measuring strike gaps.
    available = np.isfinite(prices).any(axis=1)
    strikes, prices = strikes[available], prices[available]
    if bids is not None:
        bids = np.asarray(bids)[available]
    if len(strikes) < 3:
        return np.nan
    paired = np.flatnonzero(np.isfinite(prices).all(axis=1))
    if not len(paired):
        return np.nan
    star = paired[np.argmin(np.abs(prices[paired, 0] - prices[paired, 1]))]
    growth = np.exp(rate * term)
    forward = strikes[star] + growth * (prices[star, 0] - prices[star, 1])
    base = np.searchsorted(strikes, forward, side="right") - 1
    if base < 0 or base >= len(strikes) - 1 or not np.isfinite(prices[base]).all():
        return np.nan
    selected = [(base, prices[base].mean())]
    for indices, side in [(range(base - 1, -1, -1), 1), (range(base + 1, len(strikes)), 0)]:
        zeros, count = 0, 0
        for i in indices:
            q = prices[i, side]
            if not np.isfinite(q):
                zeros = 0
                continue
            if bids is not None and bids[i, side] == 0:
                zeros += 1
                if zeros == 2:
                    break
                continue
            zeros = 0
            if q > 0:
                selected.append((i, q))
                count += 1
        if count == 0:
            return np.nan
    if len(selected) < max(3, min_strikes):
        return np.nan
    selected.sort()
    k = strikes[[i for i, _ in selected]]
    q = np.array([q for _, q in selected])
    delta = np.r_[k[1] - k[0], (k[2:] - k[:-2]) / 2, k[-1] - k[-2]]
    variance = (2 * growth * np.sum(delta * q / k**2) - (forward / strikes[base] - 1)**2) / term
    return float(variance) if np.isfinite(variance) and variance > 0 else np.nan


def fetch_maturity_variance(day, wsd, prefix, expiry, strikes, *, rate,
                            price_field, bid_field, ask_field, min_strikes):
    """Fetch one option maturity and return its term and implied variance."""
    midpoint = price_field is None
    prices = np.full((len(strikes), 2), np.nan)
    bids = np.full_like(prices, np.nan) if midpoint else None
    month = expiry.strftime("%y%m")
    for i, strike in enumerate(strikes):
        for side, cp in enumerate(("C", "P")):
            code = f"{prefix}{month}-{cp}-{int(strike)}.CFE"
            if midpoint:
                bid = quote(wsd, code, bid_field, day)
                ask = quote(wsd, code, ask_field, day)
                if np.isfinite([bid, ask]).all() and bid >= 0 and ask > 0 and ask >= bid:
                    prices[i, side], bids[i, side] = (bid + ask) / 2, bid
            else:
                value = quote(wsd, code, price_field, day)
                if np.isfinite(value) and value > 0:
                    prices[i, side] = value
    term = (expiry - day).days / 365
    return term, maturity_variance(strikes, prices, term, rate, bids=bids, min_strikes=min_strikes)


def calculate_iv30(day, wsd, prefix, *, rate=0.02, price_field="close",
                   bid_field=None, ask_field=None, strike_step=50,
                   strike_bounds=(0.65, 1.35), min_strikes=8):
    """Fetch only needed expiries; never extrapolate beyond available terms.

    Optional bid/ask mode applies the consecutive-zero-bid rule. WSD failures
    propagate rather than being silently converted into successful updates.
    Invalid candidate codes should be returned as empty/NaN by the reader.
    """
    day = asof(day)
    prefix = prefix.upper()
    if prefix not in INDEX_CODES:
        raise ValueError("prefix must be IO or MO")
    if (not np.isfinite(rate) or strike_step <= 0 or int(strike_step) != strike_step
            or not 0 < strike_bounds[0] < 1 < strike_bounds[1]):
        raise ValueError("invalid rate or strike grid")
    if price_field is None:
        if not bid_field or not ask_field:
            raise ValueError("supply both bid_field and ask_field")
    elif bid_field or ask_field:
        raise ValueError("choose price_field or both bid_field/ask_field")
    spot = quote(wsd, INDEX_CODES[prefix], "close", day)
    if not np.isfinite(spot) or spot <= 0:
        return np.nan
    step = int(strike_step)
    lower = max(step, int(np.floor(round(spot * strike_bounds[0] / step, 10))) * step)
    upper = int(np.ceil(round(spot * strike_bounds[1] / step, 10))) * step
    strikes = np.arange(lower, upper + 1, step, dtype=float)

    expiries = expiry_candidates(day)
    near = [e for e in expiries if (e - day).days <= 30]
    far = [e for e in expiries if (e - day).days >= 30]
    # Search closest terms first, falling back only when quotes are insufficient.
    terms = []
    for candidates in (reversed(near), far):
        for expiry in candidates:
            term, var = fetch_maturity_variance(
                day, wsd, prefix, expiry, strikes, rate=rate, price_field=price_field,
                bid_field=bid_field, ask_field=ask_field, min_strikes=min_strikes,
            )
            if np.isfinite(var):
                if term == TARGET:
                    return float(100 * np.sqrt(var))
                terms.append((term, var))
                break
        else:
            return np.nan
    (t1, v1), (t2, v2) = terms
    weight = (TARGET - t1) / (t2 - t1)
    total = (1 - weight) * t1 * v1 + weight * t2 * v2
    return float(100 * np.sqrt(total / TARGET))


def update_iv30(date, dates, wsd, root, *, prefix="IO", **kwargs):
    """Write one IV to option-root/<index>_IV30.bin, shape (len(dates), 1)."""
    prefix = prefix.upper()
    if prefix not in OUTPUT_NAMES:
        raise ValueError("prefix must be IO or MO")
    row = date_index(date, dates)
    value = calculate_iv30(date, wsd, prefix, **kwargs)
    array = ensure_memmap(Path(root) / f"{OUTPUT_NAMES[prefix]}.bin", (len(dates), 1))
    array[row, 0] = value
    array.flush()
    del array
    return value


if __name__ == "__main__":
    from v2.Monitor.context import MarketContext
    from v2.UpdateData.config import ROOT

    date = "2026-07-24"
    dates = np.load(Path(ROOT) / "axis" / "dates.npy", allow_pickle=False)
    context = MarketContext(root=ROOT)
    try:
        for prefix in ("IO", "MO"):
            started = perf_counter()
            value = update_iv30(
                date, dates, context.wsd, Path(ROOT) / "option",
                prefix=prefix,
                price_field="close",
                rate=0.02,
            )
            print(f"{date} {prefix} IV30={value:.4f}, elapsed={perf_counter() - started:.3f}s",
                  flush=True)
    finally:
        context.close()
