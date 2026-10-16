# Microstructure Monitor

Daily and historical research pipeline next to v2. Dependencies: pandas >= 2,
numpy, scipy, matplotlib; the L2 adapter additionally uses polars and pyarrow.

Version 5 requires Plotly (see requirements.txt). The primary
interactive artifact is monitor.html; trends.html is an optional static companion
(run_daily_monitor(..., static_dashboard=True)).
Use new output directories: demo_v5, l2_v5, l2_range_v5, synthetic_trends_v5.
Prior-version history is preserved and rejected rather than silently mixed.

## Run

```powershell
python -m microstructure_monitor.examples.example_usage
python -m microstructure_monitor.examples.small_sample
python -m microstructure_monitor.examples.run_l2_day --date 20260724 --metadata-root Z:/
python -m microstructure_monitor.examples.run_l2_range --start 20260601 --end 20260724 --symbols 000001.SZ 600000.SH --metadata-root Z:/ --output microstructure_monitor/output/l2_range_v5
python -m microstructure_monitor.examples.demo_trends
```

Real Z-drive 61-day selected-stock run (2026-04-27 through 2026-07-24):

```powershell
python -m microstructure_monitor.examples.run_l2_range --start 20260427 --end 20260724 --symbols 000001.SZ 600000.SH --metadata-root Z:/ --output microstructure_monitor/output/l2_61d_july_v5 --resume --prefetch 3 --read-strategy prefiltered
```

run_status.json records requested/completed dates, the active date, failure
details and final completion. --resume reuses a completed chronological prefix
and recomputes the final date so its baselines include the entire prior history.
This remains a two-stock study, not a full-market report. Missing processed
quotes are reported as unavailable, never replaced with synthetic prices/depth.
First remote reads can be slow; filtered local caches validate source file
size/time and selected securities before reuse. Source files are read-only.
--prefetch bounds concurrent input reads; history writes remain chronological.
--read-strategy prefiltered uses the installed Polars late-materialization
strategy and has an event-equivalence regression test. The default remains
auto; choose auto if another Polars version does not support prefiltered.

The range runner processes available dates chronologically. Default L2 root:
Z:/stock/l2. --metadata-root reads v2 index weights, industry and daily essentials
on the exact date axis, without database access. Missing fields remain absent.
Filtered raw inputs have a source-size/time/symbol/version validated local cache.
Existing output history without a matching version/universe manifest is rejected.
Use a fresh directory to regenerate legacy metrics; old results are not deleted.

## Inputs

Orders: time, symbol, order_id, price, volume, side; optional status.
Trades: time, symbol, price, volume, buy_order_id, sell_order_id; optional direction.
Quotes: time, symbol, bid_price1, ask_price1; optional bid_volume1, ask_volume1.
Daily: symbol; optional hs300/csi500/csi1000 boolean memberships, industry,
mcap, liquidity, volatility. v2 metadata uses daily amount as a liquidity proxy
and daily high-low/close as a volatility proxy.

Use consistent identifiers, currency-unit prices and share volumes. Timestamps
or HH:MM:SS strings are supported. Order IDs must include channel where needed.
Submit keys (symbol, side, order_id) must be unique; cancel events refer to the
original key and cancelled volume. Unpriced market orders remain in counts and
lifecycles; their unknown amounts are not fabricated. Zero cancellation prices
are inferred from submitted orders where possible.

Full-day order/trade context preserves preopen submissions and earlier fills.
Lifecycle cohorts include continuous-session submissions and orders referenced
by continuous fills/cancels. Completed lifetimes are wall-clock seconds; open
orders are censored. Fill rate is mean per-order volume fraction; cancel rate is
volume weighted, so they must not be added together. Overaccounting, missing
matches and negative lifetimes are reported.

The adapter reads wt/cj/cancel proc files and existing wide/long *_shot_1m.pq.
Missing snapshots are not reconstructed. No Z files are written or downloaded.

## Implemented

- Minute/five-minute/period/daily count, volume and amount aggregates for buy,
  sell, large, XL, orders and cancellations; OI/TI are recomputed per period.
- Trade/order distributions, active and passive side labels, size classes,
  net/normalized size flows, gaps, alignment, OI/TI and return lead-lag.
- Order linkage, cancellation asymmetry and completed-order lifetime.
- Spread/depth, effective/realized spread, signed midpoint impact, Amihud and
  observed-minute spread recovery and observed best-level depth recovery events.
  Trade-only signed impact is a separate proxy.
- Event returns at 1/5/10/30 minutes with samples and continuation/reversal.
- Intraday means/shares, extrema/ranges and numeric/readable peak times.
- Per-stock and pooled trade distributions, cross-sectional summaries, historical
  KS and normalized intraday distances, point-in-time groups and reference gaps.
- Automatic rolling history, identity validation and persistent migration.
- Coverage/cleaning quality tables, separate auctions and cleaned input tables.
- CSV, compact Markdown and selectable offline HTML plots for trends, Z scores,
  peaks, groups, gaps, historical intraday overlays and data quality.

## History And State Semantics

Pass MarketData(orders, trades, quotes, daily) to run_daily_monitor. Optional
universe_map overrides automatic groups. universe_id can name a dynamic
point-in-time population; without it, exact observed symbols define identity.

Root metrics are equal-weight stock means. market.* are additive totals and
pooled_trade.* are actual pooled trade statistics. universe.* sums activity/flow
amounts, recomputes amount-based imbalance and weighted trade ratios, and uses
stock means for nonadditive features. Order size ratios are amount weighted too.
Daily/minute group calculations share one vectorized aggregation implementation.
memberships.json records each day's members; historical group plots never reuse
today's membership when an old snapshot is absent.
gap.* compares groups against HS300 if
available, otherwise all. It avoids quadratic all-pairs industry comparisons.

history.csv replaces same-date observations. Prior observations only enter
MA5/20/60, STD/Z20/60 and dashboard moving averages. Full windows are required.
pct_history requires 60 valid samples; pct_1y requires 252, with sample counts.
In a dynamic universe, changes in coverage/composition must also be examined.

Migration families: liquidity, size, flow, intraday/peak-time, impact,
distribution, curve and universe gaps. Signals combine Z20 and standardized
MA5-MA20 differences. KS/L1 distances also have absolute configured thresholds.
Two families trigger TRANSITION; two persistent same-direction families on
3 of 5 observations trigger NEW_REGIME. Insufficient families produce
INSUFFICIENT_HISTORY. Families may be correlated; these are research heuristics.

Orderflow buy/sell reflects current signed flow, not deviation from its historical
mean. Opening/closing dominance compares current period mean activity. Missing
raw inputs yield unavailable; missing reference history yields insufficient_history.
Institution states remain large-trade activity proxies, not investor identities.

KS compares current common securities with up to 20 prior daily cross-sections
and requires 20 valid samples on each side. Exploratory p-values assume
independent observations and must not be treated as calibrated market tests.
Intraday L1 compares normalized current curves to prior observed-day mean curves.

## Outputs

output/YYYYMMDD contains features, baselines, groups/gaps, distribution, anomalies,
migration, structural_change, quality/cleaning_quality, bars, clean_* and auction_*.
monitor.html embeds Plotly and data; opens directly without a server.
depth_recovery_events.csv exposes recovered and censored depth episodes.
The synthetic trend demo is under output/synthetic_trends_v5 and is not real L2.
The three-day, two-stock sample is under output/small_sample_july_v5/monitor.html
(2026-07-22 through 2026-07-24). Earlier September outputs remain archived.
Its input changes are intentional; it is a software demonstration, not real L2.

## Measurement Limits

Auction records are separated; continuous feature windows exclude auctions.
Observed bars do not fill inactivity. Exact minute returns exclude lunch/gaps.
Future event prices use the first trade within 30 seconds after the target;
missing horizons stay NaN. Sparse quote matching coverage is reported.
Spread recovery uses observed snapshots and within-day thresholds; it is not
causal order-book resilience. Large trades and small samples cannot establish
institutional behavior or full-market regime changes. Full-market and long-history
empirical validation remains separate from implementation tests.

Optional single-side quote depth is retained; negative/infinite quantities become
missing. Scope depth imbalance uses bid/ask means from the same valid snapshots.
Scope flow agreement and OI/TI correlations are recomputed from pooled minute
curves, not averaged stock coefficients. Event-return scope means use samples.
Historical curve distances cover every available numeric minute column, use
shared observed clocks, preserve signs and normalize each prior day separately.
inspect curve_common_minutes and curve_history_days when interpreting them.
No shock is zero count when valid depth exists; no depth is missing count.

Depth recovery measures bid/ask/total best-level quantity falling to <=50% of
the previous valid observation and returning to >=90% within 300 seconds.
Gaps over 60 seconds, lunch, day end and horizon expiry censor the episode.
The mean duration uses recovered events only; always inspect censored counts.
This is observed snapshot recovery, not causal full-book replenishment.

## Checks

The explorer includes security_baseline.csv for every stock/metric, full numeric
feature selectors, six size-flow pair comparisons, feature_1min.csv-derived
intraday shapes, 1/5/10/30-minute event charts, Z20/Z60/one-year percentile
panels and actual order/trade amount histograms with historical CDF comparison.
The offline page embeds Plotly and all data: no CDN or server is needed.
Minute correlation curves are retrospective rolling-20-observation estimates
(at least 10 samples); event horizons and lifecycle cohort statistics can use
later events and must not be presented as live trading signals.
Intraday shares of signed metrics use absolute contributions, named abs_share.
security_baseline.csv and feature shape columns can be large on wide universes.

Optional --offline-cache points to a previously generated validated-cache folder;
it checks date and securities but explicitly does not refresh source timestamps.
It must not be used to claim freshly downloaded data.

```powershell
python -m compileall -q microstructure_monitor
python -m unittest discover -s microstructure_monitor/tests -t . -v
python -m microstructure_monitor.examples.check_pipeline
python -m microstructure_monitor.examples.check_extended
```
