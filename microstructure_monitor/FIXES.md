# Version 2 Completion And Validation

The earlier AUDIT.md describes the pre-fix implementation. This document records
the fixes and implemented functionality; it does not claim empirical validation
on a full-market, multi-year dataset.

## Resolved Audit Findings

| Finding | Resolution | Evidence |
| --- | --- | --- |
| Averaged coarse OI/TI | Additive numerators/denominators and per-period recomputation | All-frequency ratio and amount conservation test; real daily TI agrees with feature TI to 1e-16 |
| Direction labels follow historical Z | Current signed flow determines buy/sell; current period mean activity determines opening/closing | Negative-flow/positive-Z and early/late dominance regression |
| Preopen orders removed before matching | Full-day context with a continuous cohort; auctions exported separately | Preopen lifecycle regression; real buy/sell/cancel matching ratios all 1 |
| Lunch and missing-minute returns | Exact elapsed one-minute adjacency for returns and lead correlations | Lunch/gap regression |
| Mixed universe/version history | Persisted manifest, explicit rejection of legacy/incompatible history | Universe identity regression |
| Impact duplicates Amihud | Quote-based signed 5-minute midpoint impact; Amihud and trade-only proxy have distinct fields | Matched-cohort effective=realized+impact regression |

## Added Requirements

- Count/volume/amount for buy/sell/large/XL, orders/cancels, active/passive sides
  at all aggregation levels. Cleaned and auction tables are exported.
- Order sizes, lifecycle coverage, overaccounting and censoring flags.
- Per-stock distributions and separately named pooled trade distributions/totals.
- Historical KS via scipy with minimum-sample checks and exploratory p-values;
  normalized intraday curve distances against recent historical profiles.
- Distribution, curve, peak-time and universe-gap signals enter persistent
  migration alongside level and standardized MA-gap signals.
- Point-in-time v2 daily/index/industry metadata reader; universe ratios use
  additive flows and activity-weighted proportions rather than averaging ratios.
- Every selected intraday metric has means, extrema, ranges and readable peak
  times; unsigned shares and explicitly named absolute OI/TI shares are available.
- Cleaning/order/quote/duplicate quality reports and separated missing-data vs
  missing-history states. True 1-year percentile requires 252 observations;
  shorter-history percentile and sample counts have separate fields.
- Offline chart selector, prior-data MA/Z plots, distribution/peak/group/gap
  views, historical intraday overlays and quality tables. Compact daily report.
- Validated local Parquet input cache for repeat real-data runs.

## Verification

10 regression tests pass, along with the original pipeline and extended checks.
Compilation passes. Controlled tests cover all coarse ratios, session/gap
boundaries, direction semantics, preopen linkage, identity guards, structural
signal resonance, metadata alignment, sample sufficiency and impact decomposition.

Real 20260724 run: 000001.SZ and 600000.SH, reading six proc order/trade/cancel
tables and point-in-time Z metadata. Both buy/sell/cancel match ratios are 1;
overaccounted and negative-lifetime counts are 0. Actual groups: all, HS300,
CSI500, CSI1000 and industry 25. Both selected stocks are in HS300; the other
index groups are empty and remain missing, not reported as zero-valued metrics.
The run has no quotes or prior history, so liquidity/quote impact are unavailable
and overall regime remains INSUFFICIENT_HISTORY. Current signed flow correctly
indicates selling; current activity is opening-dominant.

Output: output/l2_v2/20260724 and output/l2_v2/trends.html. Synthetic historical
visualization is under output/synthetic_trends_v2 and is explicitly simulated.
Embedded figures were visually inspected. No browser connection was available,
so actual browser interaction with the chart selector was not verified.

## Measurement And Deployment Boundaries

Completed lifecycle is measured in wall-clock seconds. Sparse snapshot data
limits effective-spread coverage and spread recovery remains a measured proxy.
The KS p-value has independence assumptions that repeated/correlated market
observations may violate; distance signals are heuristic. Families can correlate.
Institutional identity cannot be inferred from trade size. Data availability is
separate from feature implementation: absent snapshots do not become synthetic
liquidity. A fixed-universe 60+ day real run and broader market calibration remain
empirical validation work, not claims made by the supplied tests.
