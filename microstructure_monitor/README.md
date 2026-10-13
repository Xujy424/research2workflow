# Microstructure Monitor

Daily and historical research pipeline next to v2. Dependencies: pandas >= 2,
numpy, matplotlib; the L2 adapter additionally uses polars and pyarrow.

## Run

```powershell
python -m microstructure_monitor.examples.example_usage
python -m microstructure_monitor.examples.run_l2_day --date 20260724
python -m microstructure_monitor.examples.run_l2_range --start 20260601 --end 20260724 --symbols 000001.SZ 600000.SH --output microstructure_monitor/output/l2_range
python -m microstructure_monitor.examples.demo_trends
```

The range runner processes available directories chronologically. Default root:
Z:/stock/l2. Use --root to change it. Trend demo data is explicitly synthetic.

## Inputs

Orders: time, symbol, order_id, price, volume, side; optional status.
Trades: time, symbol, price, volume, buy_order_id, sell_order_id; optional direction.
Quotes: time, symbol, bid_price1, ask_price1; optional bid_volume1, ask_volume1.
Daily: symbol; optional hs300/csi500/csi1000 boolean memberships, industry,
mcap, liquidity, volatility for point-in-time group assignments.

Use consistent identifiers, currency-unit prices and share volumes. Time supports
timestamps or HH:MM:SS. Order IDs must include channel where appropriate.
Orders include submit and cancel rows: cancel events reference the original ID,
side and cancelled volume. Zero cancellation prices are permitted. Submitted
keys (symbol, side, order_id) must be unique. Incomplete order lifetimes are
censored. Fill rate is the mean per-order volume fill fraction; cancellation
rate is total cancelled volume / submitted volume.

The v2 adapter reads wt/cj/cancel proc tables and existing sh_shot_1m.pq /
sz_shot_1m.pq (wide/long) snapshots. Missing quote metrics stay NaN.
It does not reconstruct missing snapshots, download, or modify Z data.

## Implemented

- Trade and order amount distributions, XL and active buy/sell ratios.
- Four size-class net/normalized flows, gaps, alignment and OI/TI correlations.
- Order fill/cancel accounting and completed-order lifetime.
- Relative spread, bid/ask depth, effective/realized spread, minute Amihud,
  and observed-minute spread recovery proxy.
- Event returns at 1/5/10/30 minutes and continuation/reversal proportions.
- Intraday period means, amount shares, extrema, ranges and peak times.
- Cross-sectional distributions, automatic daily-table group memberships,
  universe pair differences, empirical KS and normalized curve L1 distances.
- Automatic historical storage, rolling baselines and persistent migration.
- Detailed CSV/Markdown outputs and standalone trends.html with embedded plots.

Pass MarketData(orders, trades, quotes, daily) to run_daily_monitor. Optional
universe_map overrides automatic groups. Keep a separate output_dir for each
security universe and metric version. Historical data can also be passed via
history (DatetimeIndex, one market observation per date).

## Historical Monitoring

history.csv is updated automatically; rerunning a date replaces its observation.
Baselines use strictly earlier observations. MA5/20/60 and STD/Z20/60 require
full windows. pct_1y uses at most 252 observations and requires 60 valid samples.
Universe differences and distribution summaries are saved in history too.

Two independent families (liquidity, size, flow, intraday, impact) with |Z20| >= 2
trigger TRANSITION. Two families showing same-direction exceptions on 3 of the
last 5 observations trigger NEW_REGIME. Fewer than two usable families produce
INSUFFICIENT_HISTORY. These are research heuristics, not validated signals.

structural_change.csv compares the previous available day on common securities:
empirical KS statistic without a p-value and normalized amount-curve L1 distance.
These comparisons are descriptive and are not yet inputs to the classifier.

output/YYYYMMDD contains features, baselines, distribution, cross_section,
universe_gaps, anomalies, migration, structural_change, bars and daily_report.md.
output/trends.html shows daily metrics, MA5/20/60 and migration history.
Regenerate reports chronologically after changing historical observations.

## Interpretation Limits

Market metrics are equal-weight security means, not pooled market quantiles.
Size classes have fixed 20k/100k/500k amount boundaries. Large trades are activity
proxies, not investor identities. Auctions are excluded; observed bars do not
fill inactive minutes. Event prices use the first trade within 30 seconds of the
target and do not bridge lunch. Sparse quotes limit effective-spread coverage.
Spread recovery uses within-day quantile thresholds and is not a causal resilience
estimate. Passive-side return analysis, auction features and live interactive
dashboards remain extensions.

## Checks

```powershell
python -m compileall -q microstructure_monitor
python -m microstructure_monitor.examples.check_pipeline
python -m microstructure_monitor.examples.check_extended
```
