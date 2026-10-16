# Microstructure monitor: 2026-07-22

Market metrics are equal-weight security averages.

- liquidity_regime: insufficient_history
- trade_size_regime: insufficient_history
- orderflow_regime: neutral
- impact_regime: insufficient_history
- institution_regime: insufficient_history
- institution_proxy_regime: insufficient_history
- intraday_regime: balanced
- overall: INSUFFICIENT_HISTORY

## Core Features

```text
                                      000001.SZ     600000.SH
trade.count                        2.370000e+02  2.370000e+02
trade.amount                       9.919743e+07  8.954479e+07
trade.mean                         4.185546e+05  3.778261e+05
trade.median                       4.251006e+05  3.797033e+05
trade.std                          2.295046e+05  2.342347e+05
trade.skew                        -9.910000e-02  8.220000e-02
trade.maximum                      7.953180e+05  7.932381e+05
trade.p90                          7.335524e+05  7.007093e+05
trade.p95                          7.557982e+05  7.434852e+05
trade.p99                          7.863437e+05  7.875775e+05
trade.tail_ratio                   1.849800e+00  2.074200e+00
trade.classified_ratio             1.000000e+00  1.000000e+00
trade.large_ratio                  9.853000e-01  9.821000e-01
trade.active_buy_ratio             4.776000e-01  5.587000e-01
trade.active_sell_ratio            5.224000e-01  4.413000e-01
trade.xl_ratio                     6.158000e-01  5.694000e-01
trade.kurt                        -1.228400e+00 -1.226400e+00
lifecycle.fill_rate                4.895000e-01  5.527000e-01
lifecycle.completed_ratio          4.895000e-01  5.527000e-01
lifecycle.lifetime_seconds         0.000000e+00  0.000000e+00
lifecycle.overaccounted_orders     0.000000e+00  0.000000e+00
lifecycle.context_order_count      2.370000e+02  2.370000e+02
lifecycle.submitted_volume         9.940424e+06  8.999290e+06
lifecycle.filled_volume            4.747621e+06  5.029227e+06
lifecycle.cancelled_volume         0.000000e+00  0.000000e+00
lifecycle.fill_ratio_sum           1.160000e+02  1.310000e+02
lifecycle.lifetime_seconds_sum     0.000000e+00  0.000000e+00
lifecycle.lifetime_samples         1.160000e+02  1.310000e+02
lifecycle.negative_lifetime_count  0.000000e+00  0.000000e+00
lifecycle.cancel_rate              0.000000e+00  0.000000e+00
lifecycle.buy_submitted_volume     4.747621e+06  5.029227e+06
lifecycle.buy_cancelled_volume     0.000000e+00  0.000000e+00
lifecycle.buy_cancel_rate          0.000000e+00  0.000000e+00
lifecycle.sell_submitted_volume    5.192803e+06  3.970063e+06
lifecycle.sell_cancelled_volume    0.000000e+00  0.000000e+00
lifecycle.sell_cancel_rate         0.000000e+00  0.000000e+00
lifecycle.cancel_side_gap          0.000000e+00  0.000000e+00
quality.buy_order_match_ratio      4.895000e-01  5.527000e-01
quality.sell_order_match_ratio     0.000000e+00  0.000000e+00
quality.direction_known_ratio      1.000000e+00  1.000000e+00
quality.priced_order_ratio         1.000000e+00  1.000000e+00
quality.cancel_order_match_ratio            NaN           NaN
quality.quotes_count               2.370000e+02  2.370000e+02
quality.duplicate_trade_rows       0.000000e+00  0.000000e+00
```

## Core Baselines

```text
                                                 raw  sample_count  ma5  ma20  std20  z20  ma60  std60  z60  pct_1y  pct_history  pct_1y_samples
trade.count                             2.370000e+02             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.amount                            9.437111e+07             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.mean                              3.981903e+05             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.median                            4.024020e+05             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.std                               2.318696e+05             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.skew                             -8.500000e-03             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.maximum                           7.942781e+05             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.p90                               7.171308e+05             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.p95                               7.496417e+05             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.p99                               7.869606e+05             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.tail_ratio                        1.962000e+00             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.classified_ratio                  1.000000e+00             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.large_ratio                       9.837000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.active_buy_ratio                  5.181000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.active_sell_ratio                 4.819000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.xl_ratio                          5.926000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.kurt                             -1.227400e+00             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_flow                         1.933111e+04             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.medium_flow                       -1.094984e+05             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.large_flow                         2.688953e+06             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.xl_flow                            4.293606e+05             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.imbalance                          3.620000e-02             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_classified_amount            6.368394e+04             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_buy_amount                   4.150753e+04             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_sell_amount                  2.217641e+04             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_normalized                   3.964000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.medium_classified_amount           1.468057e+06             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.medium_buy_amount                  6.792794e+05             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.medium_sell_amount                 7.887778e+05             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.medium_normalized                 -7.620000e-02             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.large_classified_amount            3.680216e+07             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.large_buy_amount                   1.974556e+07             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.large_sell_amount                  1.705661e+07             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.large_normalized                   7.270000e-02             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.xl_classified_amount               5.603721e+07             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.xl_buy_amount                      2.823328e+07             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.xl_sell_amount                     2.780392e+07             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.xl_normalized                      1.510000e-02             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_large_gap                    3.236000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.large_xl_gap                       5.770000e-02             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_xl_product                  -9.300000e-03             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_xl_aligned                   5.000000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.oi_ti_gap                         -0.000000e+00             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_medium_agreement                      NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_medium_divergence                     NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_large_agreement                       NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_large_divergence                      NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_xl_agreement                          NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_xl_divergence                         NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.medium_large_agreement                      NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.medium_large_divergence                     NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.medium_xl_agreement                         NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.medium_xl_divergence                        NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.large_xl_agreement                          NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.large_xl_divergence                         NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.direction_agreement_rate                    NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.direction_divergence_rate                   NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.oi_ti_corr                         1.000000e+00             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.oi_ti_samples                      2.370000e+02             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.oi_ti_lead_corr                    8.000000e-02             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.flow_return_lead_corr              4.810000e-02             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.spread                        2.000000e-03             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.depth                         2.192194e+04             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.depth_imbalance              -9.630000e-02             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.paired_bid_depth              9.921941e+03             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.paired_ask_depth              1.200000e+04             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.amihud                        0.000000e+00             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.impact                        2.000000e-04             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.trade_impact_proxy            1.000000e-04             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.bid_depth_recovery_seconds    1.200000e+02             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.bid_depth_shock_count         2.000000e+00             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.bid_depth_recovered_count     1.000000e+00             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.bid_depth_censored_count      1.000000e+00             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.bid_depth_valid_quotes        2.370000e+02             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.ask_depth_recovery_seconds             NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.ask_depth_shock_count         0.000000e+00             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.ask_depth_recovered_count     0.000000e+00             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.ask_depth_censored_count      0.000000e+00             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.ask_depth_valid_quotes        2.370000e+02             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.total_depth_recovery_seconds           NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.total_depth_shock_count       0.000000e+00             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.total_depth_recovered_count   0.000000e+00             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.total_depth_censored_count    0.000000e+00             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.total_depth_valid_quotes      2.370000e+02             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.bid_depth                     9.921941e+03             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.ask_depth                     1.200000e+04             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.spread_recovery_seconds       1.293333e+03             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.recovery_samples              4.500000e+00             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
pooled_trade.mean                       3.981903e+05             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
pooled_trade.median                     4.093711e+05             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
pooled_trade.std                        2.325318e+05             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
pooled_trade.skew                      -1.090000e-02             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
pooled_trade.kurt                      -1.239000e+00             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
pooled_trade.max                        7.953180e+05             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
pooled_trade.p50                        4.093711e+05             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
pooled_trade.p90                        7.163751e+05             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
pooled_trade.p95                        7.472950e+05             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
pooled_trade.p99                        7.878275e+05             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
pooled_trade.tail_ratio                 1.924500e+00             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
```

Full tables: features.csv, baseline.csv, quality.csv.
Interactive report: [monitor.html](../monitor.html)

Securities: 2
Anomalies: 0

Large trade size is a proxy, not an identification of institutional activity.
Continuous session: 09:30-11:30 and 13:00-14:57; auctions excluded.