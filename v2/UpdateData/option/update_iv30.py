"""IO/MO model-free IV30; all provider access is through an injected WSD reader."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from ..utils import asof, date_index
else:
    import sys

    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from v2.UpdateData.utils import asof, date_index

OUTPUTS = {"IO": "HS300_IV30", "MO": "CSI1000_IV30"}
YEAR = pd.Timedelta(days=365)
TARGET = 30 / 365


def _atomic_csv(frame, path, *, index=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=index)
    temp.replace(path)


def _quote(wsd, code, field, day):
    """Accept DataContext.wsd series or GetData's date/value DataFrame."""
    result = wsd(code, field, str(day.date()), str(day.date()))
    if isinstance(result, pd.DataFrame):
        if "date" in result:
            result = result.set_index("date")
        if field in result:
            result = result[field]
        elif result.shape[1] == 1:
            result = result.iloc[:, 0]
        else:
            raise ValueError("ambiguous WSD columns")
    if not isinstance(result, pd.Series):
        raise ValueError("WSD must return a dated Series/DataFrame")
    result = result.copy()
    result.index = pd.to_datetime(result.index).tz_localize(None).normalize()
    values = result.loc[result.index == day]
    if len(values) > 1:
        raise ValueError("duplicate WSD dates")
    return float(values.iloc[0]) if len(values) else np.nan


def prepare_contracts(contracts, day):
    """Require actual listing dates and settlement timestamps, never infer expiry."""
    frame = contracts.copy()
    required = {"code", "prefix", "cp", "strike", "listed", "expiry"}
    if not required.issubset(frame):
        raise ValueError(f"contracts require {sorted(required)}")
    frame["listed"] = pd.to_datetime(frame.listed)
    frame["expiry"] = pd.to_datetime(frame.expiry)
    if frame.listed.dt.tz is not None or frame.expiry.dt.tz is not None:
        raise ValueError("contract timestamps must be timezone-naive Shanghai local time")
    if (frame.expiry == frame.expiry.dt.normalize()).any():
        raise ValueError("expiry must include the actual settlement time")
    frame["strike"] = pd.to_numeric(frame.strike, errors="raise")
    if (frame.strike <= 0).any() or not np.isfinite(frame.strike).all():
        raise ValueError("invalid strike")
    if not frame.cp.isin(["C", "P"]).all():
        raise ValueError("cp must be C or P")
    frame = frame.loc[frame.prefix.isin(OUTPUTS) & (frame.listed <= day + pd.Timedelta(hours=15))
                      & (frame.expiry > day + pd.Timedelta(hours=15))].copy()
    if frame.code.duplicated().any() or frame.duplicated(["prefix", "expiry", "strike", "cp"]).any():
        raise ValueError("duplicate contract or strike/side")
    return frame


def curve_rate(curve, day, term):
    """Same-day curve, tenor in days, continuous annual rates in decimal units."""
    curve = curve.copy()
    curve["date"] = pd.to_datetime(curve.date).dt.normalize()
    nodes = curve.loc[curve.date == day, ["tenor_days", "rate"]].astype(float).sort_values("tenor_days")
    if (len(nodes) < 2 or nodes.tenor_days.duplicated().any()
            or not np.isfinite(nodes.to_numpy()).all() or (nodes.tenor_days <= 0).any()):
        raise ValueError("invalid_or_missing_rate_curve")
    days = term * 365
    if not nodes.tenor_days.min() <= days <= nodes.tenor_days.max():
        raise ValueError("rate_curve_does_not_cover_term")
    return float(np.interp(days, nodes.tenor_days, nodes.rate))


def maturity_variance(chain, term, rate, *, midpoint=True, min_strikes=8):
    if term <= 0 or not np.isfinite(rate):
        raise ValueError("invalid_term_or_rate")
    prices = chain.pivot(index="strike", columns="cp", values="q").sort_index()
    if not {"C", "P"}.issubset(prices):
        raise ValueError("missing_call_put_pairs")
    paired = prices.dropna(subset=["C", "P"])
    if paired.empty:
        raise ValueError("missing_call_put_pairs")
    star = (paired.C - paired.P).abs().idxmin()
    forward = float(star + np.exp(rate * term) * (paired.loc[star, "C"] - paired.loc[star, "P"]))
    below = prices.index[prices.index <= forward]
    if not len(below) or forward >= prices.index.max():
        raise ValueError("forward_outside_strike_coverage")
    k0 = float(below.max())
    if k0 not in paired.index:
        raise ValueError("missing_pair_at_k0")
    selected = [(k0, float(paired.loc[k0, ["C", "P"]].mean()))]
    bids = chain.pivot(index="strike", columns="cp", values="bid") if midpoint else None
    wings = []
    for side, strikes in [("P", prices.index[prices.index < k0][::-1]),
                          ("C", prices.index[prices.index > k0])]:
        zeros, count = 0, 0
        for strike in strikes:
            q = prices.loc[strike, side]
            if not np.isfinite(q):
                zeros = 0
                continue
            if midpoint and bids.loc[strike, side] == 0:
                zeros += 1
                if zeros == 2:
                    break
                continue
            zeros = 0
            if q > 0:
                selected.append((float(strike), float(q)))
                count += 1
        wings.append(count)
    if len(selected) < max(3, min_strikes) or min(wings) == 0:
        raise ValueError("insufficient_otm_strikes")
    strikes, quotes = np.array(sorted(selected)).T
    delta = np.r_[strikes[1] - strikes[0], (strikes[2:] - strikes[:-2]) / 2,
                  strikes[-1] - strikes[-2]]
    variance = (2 * np.exp(rate * term) / term * np.sum(delta * quotes / strikes**2)
                - (forward / k0 - 1)**2 / term)
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("invalid_variance")
    return dict(term=term, rate=rate, variance=float(variance), forward=forward,
                k0=k0, strikes=len(selected))


def candidate_contracts(day, wsd):
    """Propose WSD codes, not an authoritative historical contract master.

    Third-Friday expiry and strike coverage are explicit approximations.
    Every candidate is subsequently queried; absent quotes remain missing.
    """
    rows, discovery = [], []
    for prefix, index in [("IO", "000300.SH"), ("MO", "000852.SH")]:
        try:
            spot = _quote(wsd, index, "close", day)
            if not np.isfinite(spot) or spot <= 0:
                raise ValueError("missing_index_close")
        except Exception as exc:
            discovery.append(dict(prefix=prefix, status=f"index_error:{exc}"))
            continue
        months = []
        for offset in range(18):
            month = day.to_period("M") + offset
            first = month.to_timestamp()
            expiry = first + pd.Timedelta(days=(4 - first.weekday()) % 7 + 14, hours=15)
            if expiry > day + pd.Timedelta(hours=15):
                months.append((month, expiry))
        near = months[:3]
        months = near + [item for item in months if item[0].month in (3, 6, 9, 12)
                         and item not in near][:3]
        # A 50-point discovery grid includes the coarser 100/200-point strikes.
        lower = max(50, int(np.floor(spot * 0.65 / 50)) * 50)
        upper = int(np.ceil(spot * 1.35 / 50)) * 50
        for month, expiry in months:
            for strike in range(lower, upper + 1, 50):
                for side in ("C", "P"):
                    rows.append(dict(code=f"{prefix}{month.strftime('%y%m')}-{side}-{strike}.CFE",
                                     prefix=prefix, cp=side, strike=strike,
                                     listed=day, expiry=expiry))
        discovery.append(dict(prefix=prefix, status="candidate_grid", spot=spot,
                              strike_lower=lower, strike_upper=upper, strike_step=50))
    return pd.DataFrame(rows, columns=["code", "prefix", "cp", "strike", "listed", "expiry"]), discovery


def update_iv30(date, dates, wsd, root, *, contracts=None, rate_curve=None,
                flat_rate=None, bid_field=None, ask_field=None, price_field=None,
                min_strikes=8):
    """Update one trading date under option root; reruns replace that date.

    Supply bid_field/ask_field verified for your WSD service, OR explicitly opt
    into a close/settlement proxy using price_field. No silent fallback occurs.
    contracts and rate_curve may be DataFrames or callables accepting date.
    Reader errors are retained per quote; missing results are persisted as NaN.
    """
    day = asof(date)
    date_index(day, dates)
    midpoint = price_field is None
    if midpoint and (not bid_field or not ask_field):
        raise ValueError("supply verified bid_field/ask_field or explicit price_field")
    if not midpoint and (bid_field or ask_field):
        raise ValueError("choose midpoint or price proxy")
    if (rate_curve is None) == (flat_rate is None):
        raise ValueError("supply exactly one of rate_curve and explicit flat_rate")
    if flat_rate is not None and not np.isfinite(flat_rate):
        raise ValueError("invalid flat_rate")
    discovery = []
    if contracts is None:
        frame, discovery = candidate_contracts(day, wsd)
    else:
        frame = prepare_contracts(contracts(day) if callable(contracts) else contracts, day)
    curve = rate_curve(day) if callable(rate_curve) else rate_curve
    frame["bid"], frame["ask"], frame["q"] = np.nan, np.nan, np.nan
    frame["quote_status"] = "missing_quote"
    for idx, row in frame.iterrows():
        try:
            if midpoint:
                bid, ask = _quote(wsd, row.code, bid_field, day), _quote(wsd, row.code, ask_field, day)
                frame.loc[idx, ["bid", "ask"]] = [bid, ask]
                valid = np.isfinite([bid, ask]).all() and bid >= 0 and ask > 0 and ask >= bid
                q = (bid + ask) / 2 if valid else np.nan
            else:
                q = _quote(wsd, row.code, price_field, day)
                valid = np.isfinite(q) and q > 0
            if valid:
                frame.loc[idx, ["q", "quote_status"]] = [q, "ok"]
        except Exception as exc:
            frame.loc[idx, "quote_status"] = f"reader_error:{type(exc).__name__}:{exc}"
    records, diagnostics = {"date": str(day.date())}, []
    for prefix, output in OUTPUTS.items():
        terms, details = [], []
        for expiry, chain in frame.loc[frame.prefix == prefix].groupby("expiry"):
            term = (expiry - (day + pd.Timedelta(hours=15))) / YEAR
            detail = {"expiry": str(expiry)}
            try:
                rate = float(flat_rate) if flat_rate is not None else curve_rate(curve, day, term)
                result = maturity_variance(chain, term, rate, midpoint=midpoint, min_strikes=min_strikes)
                terms.append(result)
                detail.update(result, status="ok")
            except ValueError as exc:
                detail["status"] = str(exc)
            details.append(detail)
        near = [t for t in terms if t["term"] <= TARGET]
        far = [t for t in terms if t["term"] >= TARGET]
        records[output] = np.nan
        selected = []
        if near and far:
            a, b = max(near, key=lambda t: t["term"]), min(far, key=lambda t: t["term"])
            selected = [a["term"], b["term"]]
            if a["term"] == b["term"]:
                variance = a["variance"]
            else:
                weight = (TARGET - a["term"]) / (b["term"] - a["term"])
                variance = ((1 - weight) * a["term"] * a["variance"]
                            + weight * b["term"] * b["variance"]) / TARGET
            records[output] = 100 * np.sqrt(variance)
        diagnostics.append(dict(prefix=prefix, status="ok" if selected else "no_bracketing_expiries",
                                selected_terms=selected, maturities=details))
    root = Path(root)
    daily = root / "daily" / str(day.date())
    _atomic_csv(frame, daily / "quotes.csv")
    metadata = dict(date=str(day.date()), valuation_time="15:00 Asia/Shanghai",
                    contract_source="wsd_probed_grid_proxy" if contracts is None else "supplied_contracts",
                    expiry_method="unverified_third_friday" if contracts is None else "supplied_settlement_time",
                    discovery=discovery,
                    method="bid_ask_midpoint" if midpoint else f"{price_field}_proxy",
                    rate_method="flat_rate_proxy" if flat_rate is not None else "continuous_curve",
                    bid_field=bid_field, ask_field=ask_field, price_field=price_field,
                    flat_rate=flat_rate, diagnostics=diagnostics)
    temp = daily / "diagnostics.json.tmp"
    temp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(daily / "diagnostics.json")
    today = pd.DataFrame([records]).set_index("date")
    path = root / "index_iv30.csv"
    old = pd.read_csv(path, index_col="date") if path.exists() else today.iloc[:0]
    history = pd.concat([old.drop(index=today.index, errors="ignore"), today]).sort_index()
    # Reindex to the complete supplied trading axis so missing days never compress lags.
    valid_dates = pd.to_datetime(np.asarray(dates)[~np.isnat(np.asarray(dates))])
    axis = valid_dates[valid_dates <= day].strftime("%Y-%m-%d")
    axis = sorted(set(axis).union(history.index))
    history = history.reindex(axis)
    history.index.name = "date"
    history["IV_spread"] = history.CSI1000_IV30 - history.HS300_IV30
    for col in OUTPUTS.values():
        history[col + "_chg5"] = history[col].diff(5)
        history[col + "_pct3y"] = history[col].rolling(756, min_periods=126).apply(
            lambda x: np.mean(x[np.isfinite(x)] <= x[-1]) if np.isfinite(x[-1]) else np.nan, raw=True)
    _atomic_csv(history, path, index=True)
    return history.loc[[str(day.date())]], metadata


def update_history(start, end, dates, wsd, root, **kwargs):
    """Replay only dates on the shared trading axis, in chronological order."""
    if asof(start) > asof(end):
        raise ValueError("start must not exceed end")
    for day in pd.to_datetime(dates):
        if pd.notna(day) and asof(start) <= day <= asof(end):
            update_iv30(day, dates, wsd, root, **kwargs)


def main(root, *, contracts=None, date=None, start=None, end=None,
         rate_curve=None, flat_rate=None, bid_field=None,
         ask_field=None, price_field=None):
    """Run an update using explicit Python configuration, as in daily_update."""
    if date is not None:
        if start is not None or end is not None:
            raise ValueError("choose date or start/end")
    elif start is None or end is None:
        raise ValueError("supply date or both start and end")

    from v2.Monitor.context import MarketContext

    root = Path(root)
    dates = np.load(root / "axis" / "dates.npy", allow_pickle=False)
    kwargs = dict(
        contracts=contracts,
        rate_curve=rate_curve,
        flat_rate=flat_rate,
        bid_field=bid_field,
        ask_field=ask_field,
        price_field=price_field,
    )
    context = MarketContext(root=root)
    try:
        if date is not None:
            result, metadata = update_iv30(date, dates, context.wsd, root / "option", **kwargs)
            print(result.to_string(), flush=True)
            for diagnostic in metadata["diagnostics"]:
                print(f"[{diagnostic['status']}] {diagnostic['prefix']}", flush=True)
            return result, metadata
        update_history(start, end, dates, context.wsd, root / "option", **kwargs)
        print(f"[DONE] IV30 history: {start} to {end}", flush=True)
    finally:
        context.close()


if __name__ == "__main__":
    from v2.UpdateData.config import ROOT

    # Edit these parameters, then run this file directly from the IDE.
    main(
        root=ROOT,
        date="2026-10-26",
        # No external input files: candidate contracts are probed through WSD.
        # Explicit close/flat-rate proxy; use verified bid/ask fields and
        # an in-memory rate_curve instead when historical midpoint data is available.
        price_field="close",
        flat_rate=0.02,
        # For historical replay, replace date above with:
        # start="2026-01-01",
        # end="2026-10-26",
    )
