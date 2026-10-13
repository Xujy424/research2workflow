# Microstructure Monitor: Requirements And Runtime Audit

Reference: the user attachment e66e0016-d2c6-43dd-a6a1-54a332720dfc.
Scope: inspect the current implementation against the attachment, run it, and
explain which conclusions the evidence supports. This is an assessment, not a
claim that every requested function has been implemented.

## Conclusion

Not functionally complete. The research pipeline runs and computes many useful
daily features, but aggregation/state semantics and lifecycle coverage have
verified defects. Historical persistence works in controlled examples; a real
multi-month, full-market monitoring conclusion has not been demonstrated.

## Requirement Coverage

| Attachment requirement | Evidence / status |
| --- | --- |
| Order, trade, quote, daily inputs | Standard contracts and loaders exist; v2 adapter reads normalized proc orders/trades/cancels and optional snapshots. No automatic daily metadata connection in the real-data runner. |
| Cleaning, sorting, anomalies, order/trade linkage | Basic numeric/session cleaning and lifecycle joins exist. No comprehensive quality report. Removing preopen submissions breaks linkage with later fills. |
| Auctions versus continuous trading | Continuous trades are retained; auction records are discarded rather than exposed separately. |
| Direction and size / submit-fill-cancel | Trade direction and fixed amount buckets exist, as do lifecycle joins. No distinct passive-side event output or complete order-size classification contract. |
| Tick/minute/five-minute/period/daily | Aggregates and CSV outputs exist. Trade amount/volume conservation is checked. OI/TI are incorrectly averaged at coarser frequencies. Buy/sell/large metrics retain amounts, not separate counts and volumes for every category. |
| A: trade structure | Counts, amounts, mean/median/P90/P95/P99, large/XL and active ratios implemented. |
| B: order structure | Distribution, buy/sell amounts, large/XL ratios and lifecycle metrics implemented. Preopen coverage and data-quality checks remain incomplete. |
| C: money flows | Four size classes, net/normalized flows, gaps, products, minute agreement/divergence implemented. |
| D: OI/TI and lead-lag | Daily feature formulas and minute correlations implemented. Coarser bars and handling of gaps need correction. |
| E: liquidity | Spread/depth/effective/realized spread and recovery functions exist. The named impact metric duplicates unsigned return/amount (Amihud-style), not a distinct signed impact measure. Recovery is an observed-minute proxy. Quote coverage is not reported. |
| F: price discovery | Event return horizons and continuation/reversal functions exist, plus a one-bar lead correlation. Real results require coverage counts and clock-time gap checks. |
| Distribution layer | Per-security trade/order distributions and cross-security feature summaries exist. The average of stock P99 values is not the pooled-market trade P99. KS only compares adjacent available cross-sections; no historical-distribution significance test. |
| Intraday shape | Period means, amount shares, extrema and peak minutes exist for selected metrics. Not every metric has shares; no historical intraday overlay / peak-time dashboard. |
| Historical baseline | MA5/20/60, STD/Z20/60, up-to-252-observation percentile with warmup, automatic history storage. Controlled no-lookahead checks passed; genuine one-year market history has not been run. |
| Universe comparisons | Caller-provided point-in-time index/industry/cap/liquidity/volatility groups and pair differences exist. Real L2 CLI does not populate memberships automatically. All summaries are equal-weight stock means. |
| Level/distribution/time resonance | Persistent independent-family Z rules exist. KS, curve distances, peak times and universe gaps are not inputs to the state classifier. No explicit MA-gap detection. |
| Six states and overall migration | States and NORMAL/TRANSITION/NEW_REGIME implemented as heuristics. Orderflow and intraday labels confuse deviation from history with absolute direction / opening dominance. Institution state is only a large-trade proxy. |
| Modular project layout | Logical packages exist, but most features are concentrated in __init__.py and extended.py rather than the attachment's separate feature modules. |
| Reports / dashboard | CSV, detailed Markdown and embedded static trend plots exist. Limited fixed metrics, no interactive selectors, universe-gap charts or intraday overlays. Dashboard MA includes today's value; baseline MA uses prior observations, so these need explicit distinction. |

## Verified Defects

1. bars/__init__.py:37 averages minute OI/TI during aggregation. In the
   deterministic demo, 000001.SZ daily TI is reported as -0.021097 versus the
   ratio of total buy/sell amounts -0.044886. For 600000.SH it is 0.105485
   versus 0.117359. These should be rebuilt from aggregated numerators and
   denominators. Five-minute bars have the same defect.
2. monitor/__init__.py:18 uses Z20 sign for the buy/sell state. A raw
   flow imbalance of -0.4 with Z20=+2 is labelled buy even though net active
   selling remains dominant. Relative improvement and absolute direction need
   distinct fields. The opening/closing label has a similar semantic problem.
3. preprocess/__init__.py:17 drops preopen submissions before lifecycle
   matching. Moving one synthetic order to 09:29 gives zero retained original
   orders while its 09:30 fill remains. Lifecycle matching must use full-day
   order context while feature counting uses the intended trading session.
4. features/extended.py:121 and bars/__init__.py:35 compute close returns
   across lunch; shifted one-bar correlations can also bridge missing minutes.
   Equal row offsets are not necessarily equal elapsed trading intervals.
5. history/state logic does not validate history universe/metric version.
   Reusing one output directory for different securities can create apparent
   changes from composition alone. Separate directories are documented but
   there is no enforced identity/coverage manifest.

## Controlled Runs

- python -m compileall -q microstructure_monitor: passed.
- python -m microstructure_monitor.examples.check_pipeline: passed.
- python -m microstructure_monitor.examples.check_extended: passed.
- Extended checks cover one simple submit/fill/cancel example, empirical KS,
  same-date replacement, a deliberately shifted synthetic 65-day history,
  two-day generated reports, and a history sample count.
- They do not prove exchange reconstruction correctness, full-market scale,
  quote matching, index memberships, absolute state semantics, or real
  persistent regime changes. Synthetic NEW_REGIME is a rule test only.

## Next Actions

Correct coarser-bar ratios and state semantics; separate full-day order context
from continuous feature samples; add matching/quote/invalid-data coverage and
history identity checks. Then run a fixed security universe over at least 60
real trading days, connect actual point-in-time groups and snapshots, and compare
raw values, historical baselines, distributions, intraday profiles and universe
gaps. Full-market conclusions require full-market coverage, not two stocks.
