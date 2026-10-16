# Version 5 Requirement Coverage

Reference: the user's Desktop pasted text. This records implemented calculations
and views, not a claim of full-market empirical calibration.

| Requirement / previously missing item | Implementation / view |
| --- | --- |
| Trade count, amount, mean/median/P90/P95/P99 and active ratios | Every numeric feature is selectable under trade; per-stock and aggregate histories |
| Order distribution, cancellation/lifetime/asymmetry | Order/lifecycle selectors; real-event order amount histogram and cohort minute series |
| Four size-class normalized flows and all pair agreements | flow namespace; six pair agreement/divergence combinations |
| OI/TI/gap and same/lead correlations | stock estimates plus scope correlations recomputed from pooled minute OI/TI; scope rolling windows restart at gaps and lunch |
| Spread/depth/effective/realized/Amihud/impact/recovery | liquidity/discovery selectors; best-level bid/ask/total depth recovery events with gap/session/horizon censoring; spread recovery remains a separate proxy |
| All event horizons and continuation/reversal | Event-horizon view, matched sample counts, stock/group selection |
| Mean/median/std/skew/kurt and tails, pooled vs stock distributions | Numeric trend selectors, stock-feature histograms/CDF and original-event amount histograms/CDF |
| Intraday shape for derived economic minute metrics | Generic shape engine traverses all numeric feature-minute columns; extrema/time/means/absolute shares |
| Per-security historical baselines | security_history.py, security_baseline.csv, stock selector and MA5/20/60 + Z20/Z60 + percentile views |
| Index/cap/industry/liquidity/volatility groups and differences | Dated membership snapshots; shared daily/minute weighted aggregation; group/difference curves and both groups' distributions; unknown historical members are not backfilled |
| Historical distributions and curve/peak changes | Previous event histograms persisted; numerical curve distances across all available numeric minute metrics, not only three amounts; shared observed clocks and signed L1 normalization |
| Conclusions and availability | Chinese narrative and current-state table; current sample/history/quote scope stated |

Artifacts: monitor.html (embedded Plotly/data), optional static trends.html,
security_baseline.csv, bars_feature_1min.csv, event_distributions.json and the
existing feature, group, baseline, quality, anomaly and migration outputs,
memberships.json and depth_recovery_events.csv.

Verification: 35 Python regression tests cover prior fixes plus per-security
history, arbitrary intraday metric extraction and minute amount/event-horizon
contracts, weighted daily/minute parity, missing weights, single-side/invalid
depth, matched depth snapshot means, numeric stock IDs, pooled scope agreements,
curve distance coverage, no-shock recovery schema, compact report roundtrip and
single event-context computation. Pipeline and extended checks cover report
generation and two-day history. The generated v5 HTML passes JavaScript syntax
and six-view logic checks for market, stock, group and gap scopes (86 plot calls).
These checks use a mocked DOM/Plotly and do not verify visual rendering.

Efficiency: lifecycle matching, event returns and depth recovery are computed
once and reused. Minute scopes aggregate entire days. Historical metrics are
assembled in a dict, then converted once to a Series, avoiding quadratic inserts.
Minute shape raw/history/distribution tables remain available; history does not
add a second layer of distribution statistics on top of derived minute shapes.
HTML stores column names once and decodes tables in the page. On the same two-
stock/three-scope one-day fixture, runtime fell from 45.245 s to 3.701 s; HTML
fell from 32.34 MB to 13.22 MB including the final paired-depth fields. These are
small synthetic measurements, not full-market capacity claims. Migration uses
only the detector's actual feature families when rebuilding prior baselines.
Static Matplotlib
reports remain opt-in via static_dashboard=True.

Small sample: python -m microstructure_monitor.examples.small_sample generates
2026-07-22, 2026-07-23 and 2026-07-24 with two stocks and HS300/CSI1000 sample
scopes. Output: output/small_sample_july_v5/monitor.html. Inputs deliberately increase
late-day trade sizes and small-cap spread, and contain recovered/end-of-day
censored depth shocks. The HTML describes changes versus the prior observation
and shows histories, intraday curves, scope differences, distributions and event
returns. Three days do not support Z20/Z60 or a one-year percentile; these remain
missing rather than fabricated. No investment or market-wide inference is made.
Previous three-day benchmark: 17.998 s, standalone HTML 19.20 MB; pooled P99/P50 rises
1.9245 to 2.8190, stock-average late-day amount share rises 12.27% to 17.03%.
Each security has one observed bid-depth recovery (120 seconds) and one
end-of-day censored episode. These changes are deliberately constructed.

Known empirical boundaries: snapshot scarcity, investor identity and market
calibration cannot be solved by plotting more columns. Fresh Z-drive two-stock
data for 2026-07-24 has now passed the full pipeline in output/l2_real_july_v5.
The requested 61-day batch (2026-04-27 through 2026-07-24) uses independent
output/l2_61d_july_v5; its run_status.json is the completion authority. The
July 24 proc directory has no processed quote snapshots, so quote-dependent
features are unavailable. Synthetic example directories remain explicitly
synthetic. Browser rendering must be verified separately
when a browser connection is available; syntax/logic checks do not establish
visual correctness. Minute rolling correlations are not equivalent to the
full-day coefficient; retrospective horizons are not live predictive signals.
