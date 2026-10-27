"""Daily IO/MO IV30 proxies, stored as date-axis x 1 float32 matrices.

WSD-only discovery uses a bounded strike grid and third-Friday expiries.
Close/settlement prices and a constant rate are explicit approximations.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor
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
RATE_CODE = "M1001102"  # ChinaBond government spot yield, 1 year, percent


def quote(wsd, code, field, day):
    """Read one dated value without accepting stale or duplicate observations."""
    result = wsd(code, field, str(day.date()), str(day.date()))
    return dated_value(result, day, field)


def dated_value(result, day, field):
    """Extract a single same-day observation from a dated provider response."""
    if isinstance(result, pd.DataFrame):
        if "date" in result:
            result = result.set_index("date")
        result = result[field] if field in result else result.squeeze("columns")
    if not isinstance(result, pd.Series):
        raise ValueError("reader must return a dated Series or single-field DataFrame")
    if result.empty:
        return np.nan
    if len(result) == 1:
        timestamp = pd.Timestamp(result.index[0]).tz_localize(None).normalize()
        return float(result.iloc[0]) if timestamp == day else np.nan
    mask = pd.to_datetime(result.index).tz_localize(None).normalize() == day
    values = result.loc[mask]
    if len(values) > 1:
        raise ValueError(f"duplicate observations for {day.date()}: {field}")
    return float(values.iloc[0]) if len(values) else np.nan


def resolve_rate(day, rate, *, edb=None, rate_code=RATE_CODE, rate_scale=0.01,
                 rate_compounding="annual"):
    """Return continuous annual decimal rate; explicit rates are already continuous.

    For EDB, annual means a one-year effective return: r=log(1+y).
    A flat continuous zero curve is assumed for other maturities. The screenshot
    identifies the series and units, not its compounding; confirm with metadata
    and use continuous if the published yield is already continuously compounded.
    """
    if rate is None:
        if edb is None or not rate_code:
            raise ValueError("rate=None requires edb and a verified rate_code")
        if not np.isfinite(rate_scale) or rate_scale <= 0:
            raise ValueError("rate_scale must be positive and finite")
        date = str(day.date())
        rate = dated_value(edb(rate_code, date, date), day, "value") * rate_scale
        if rate_compounding == "annual":
            if rate <= -1:
                raise ValueError("annual effective yield must exceed -100%")
            rate = np.log1p(rate)
        elif rate_compounding != "continuous":
            raise ValueError("rate_compounding must be annual or continuous")
    rate = float(rate)
    if not np.isfinite(rate):
        raise ValueError(f"missing or invalid rate for {day.date()}")
    return rate


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
    # Flat continuous zero curve: growth(T) = exp(r*T) = (1+y_1y)**T.
    # Do not multiply rate by term before passing it to this function.
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
                            price_field, bid_field, ask_field, min_strikes, workers=1):
    """Fetch one option maturity and return its term and implied variance."""
    midpoint = price_field is None
    month = expiry.strftime("%y%m")
    codes = [f"{prefix}{month}-{cp}-{int(strike)}.CFE" for strike in strikes for cp in ("C", "P")]
    fields = (bid_field, ask_field) if midpoint else (price_field,)
    if workers == 1:
        values = [quote(wsd, code, field, day) for code in codes for field in fields]
    else:
        # Preserve request order; worker failures propagate and prevent writes.
        # The initial index quote initializes MarketContext's client beforehand.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(quote, wsd, code, field, day) for code in codes for field in fields]
            values = [future.result() for future in futures]
    values = np.asarray(values).reshape(len(strikes), 2, len(fields))
    bids = None
    if midpoint:
        bids, asks = values[:, :, 0], values[:, :, 1]
        valid = np.isfinite(values).all(axis=2) & (bids >= 0) & (asks > 0) & (asks >= bids)
        prices = np.where(valid, (bids + asks) / 2, np.nan)
    else:
        prices = values[:, :, 0]
        prices = np.where(np.isfinite(prices) & (prices > 0), prices, np.nan)
    term = (expiry - day).days / 365
    return term, maturity_variance(strikes, prices, term, rate, bids=bids, min_strikes=min_strikes)


def calculate_iv30(day, wsd, prefix, *, rate=0.02, price_field="close",
                   bid_field=None, ask_field=None, strike_step=50,
                   strike_bounds=(0.65, 1.35), min_strikes=8, workers=1):
    """Fetch only needed expiries; never extrapolate beyond available terms.

    Optional bid/ask mode applies the consecutive-zero-bid rule. WSD failures
    propagate rather than being silently converted into successful updates.
    Invalid candidate codes should be returned as empty/NaN by the reader.
    """
    day = asof(day)
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
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
                workers=workers,
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


def update_iv30(date, dates, wsd, root, *, prefix="IO", rate=0.02,
                edb=None, rate_code=RATE_CODE, rate_scale=0.01,
                rate_compounding="annual", **kwargs):
    """Write IV to a date x 1 matrix; rate=None reads EDB once per update."""
    prefix = prefix.upper()
    if prefix not in OUTPUT_NAMES:
        raise ValueError("prefix must be IO or MO")
    row = date_index(date, dates)
    day = asof(date)
    rate = resolve_rate(day, rate, edb=edb, rate_code=rate_code,
                        rate_scale=rate_scale, rate_compounding=rate_compounding)
    value = calculate_iv30(day, wsd, prefix, rate=rate, **kwargs)
    array = ensure_memmap(Path(root) / f"{OUTPUT_NAMES[prefix]}.bin", (len(dates), 1))
    array[row, 0] = value
    array.flush()
    del array
    return value


if __name__ == "__main__":
    from v2.Monitor.context import MarketContext
    from v2.UpdateData.config import ROOT
    from tqdm import tqdm

    # date = "2026-07-24"
    # dates = np.load(Path(ROOT) / "axis" / "dates.npy", allow_pickle=False)
    # context = MarketContext(root=ROOT)
    # try:
    #     for prefix in ("IO", "MO"):
    #         started = perf_counter()
    #         value = update_iv30(
    #             date, dates, context.wsd, Path(ROOT) / "option",
    #             prefix=prefix,
    #             price_field="close",
    #             rate=None,
    #             edb=context.edb,
    #             rate_code=RATE_CODE,
    #             rate_compounding="annual",  # Confirm the published convention.
    #             workers=4,  # Use 1 if the injected WSD reader is not thread-safe.
    #             # To reproduce the previous fixed-rate run, use rate=0.02.
    #         )
    #         print(f"{date} {prefix} IV30={value:.4f}, elapsed={perf_counter() - started:.3f}s",
    #               flush=True)
    # finally:
    #     context.close()



    dates = np.load(Path(ROOT) / "axis" / "dates.npy", allow_pickle=False)
    valid_dates = dates[~np.isnat(dates)]
    for prefix in ("IO", "MO"): 
        for date in tqdm(valid_dates):
            value = update_iv30(
                date, dates, MarketContext(root=ROOT).wsd, Path(ROOT) / "option",
                prefix=prefix,
                price_field="close",
                rate=None,
                edb=MarketContext(root=ROOT).edb,
                rate_code=RATE_CODE,
                rate_compounding="annual",  # Confirm the published convention.
                workers=4,  # Use 1 if the injected WSD reader is not thread-safe.
                # To reproduce the previous fixed-rate run, use rate=0.02.
            )




