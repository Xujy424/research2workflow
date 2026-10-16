# Microstructure monitor: 2026-07-24

Market metrics are equal-weight security averages.

- liquidity_regime: insufficient_history
- trade_size_regime: insufficient_history
- orderflow_regime: buy
- impact_regime: insufficient_history
- institution_regime: insufficient_history
- institution_proxy_regime: insufficient_history
- intraday_regime: closing_dominant
- overall: INSUFFICIENT_HISTORY

## Core Features

```text
                                      000001.SZ     600000.SH
trade.count                        2.370000e+02  2.370000e+02
trade.amount                       1.142802e+08  1.032894e+08
trade.mean                         4.821951e+05  4.358201e+05
trade.median                       4.758378e+05  4.034203e+05
trade.std                          2.985904e+05  2.965065e+05
trade.skew                         4.814000e-01  5.307000e-01
trade.maximum                      1.272119e+06  1.247222e+06
trade.p90                          7.940638e+05  7.878894e+05
trade.p95                          1.033203e+06  9.688717e+05
trade.p99                          1.220503e+06  1.194408e+06
trade.tail_ratio                   2.565000e+00  2.960700e+00
trade.classified_ratio             1.000000e+00  1.000000e+00
trade.large_ratio                  9.896000e-01  9.844000e-01
trade.active_buy_ratio             4.862000e-01  5.653000e-01
trade.active_sell_ratio            5.138000e-01  4.347000e-01
trade.xl_ratio                     7.297000e-01  6.749000e-01
trade.kurt                        -3.013000e-01 -3.505000e-01
lifecycle.fill_rate                4.895000e-01  5.527000e-01
lifecycle.completed_ratio          4.895000e-01  5.527000e-01
lifecycle.lifetime_seconds         0.000000e+00  0.000000e+00
lifecycle.overaccounted_orders     0.000000e+00  0.000000e+00
lifecycle.context_order_count      2.370000e+02  2.370000e+02
lifecycle.submitted_volume         1.145389e+07  1.038613e+07
lifecycle.filled_volume            5.569731e+06  5.873538e+06
lifecycle.cancelled_volume         0.000000e+00  0.000000e+00
lifecycle.fill_ratio_sum           1.160000e+02  1.310000e+02
lifecycle.lifetime_seconds_sum     0.000000e+00  0.000000e+00
lifecycle.lifetime_samples         1.160000e+02  1.310000e+02
lifecycle.negative_lifetime_count  0.000000e+00  0.000000e+00
lifecycle.cancel_rate              0.000000e+00  0.000000e+00
lifecycle.buy_submitted_volume     5.569731e+06  5.873538e+06
lifecycle.buy_cancelled_volume     0.000000e+00  0.000000e+00
lifecycle.buy_cancel_rate          0.000000e+00  0.000000e+00
lifecycle.sell_submitted_volume    5.884157e+06  4.512593e+06
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
trade.count                             2.370000e+02             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
trade.amount                            1.087848e+08             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
trade.mean                              4.590076e+05             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
trade.median                            4.396291e+05             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
trade.std                               2.975485e+05             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
trade.skew                              5.061000e-01             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
trade.maximum                           1.259671e+06             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
trade.p90                               7.909766e+05             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
trade.p95                               1.001037e+06             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
trade.p99                               1.207455e+06             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
trade.tail_ratio                        2.762800e+00             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
trade.classified_ratio                  1.000000e+00             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
trade.large_ratio                       9.870000e-01             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
trade.active_buy_ratio                  5.258000e-01             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
trade.active_sell_ratio                 4.742000e-01             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
trade.xl_ratio                          7.023000e-01             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
trade.kurt                             -3.259000e-01             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
flow.small_flow                         1.706518e+04             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
flow.medium_flow                       -8.492206e+04             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
flow.large_flow                         5.116777e+05             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
flow.xl_flow                            4.729478e+06             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
flow.imbalance                          5.160000e-02             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
flow.small_classified_amount            6.141801e+04             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
flow.small_buy_amount                   3.924159e+04             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
flow.small_sell_amount                  2.217641e+04             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
flow.small_normalized                   3.811000e-01             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
flow.medium_classified_amount           1.341405e+06             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
flow.medium_buy_amount                  6.282412e+05             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
flow.medium_sell_amount                 7.131633e+05             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
flow.medium_normalized                 -6.300000e-02             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
flow.large_classified_amount            3.083281e+07             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
flow.large_buy_amount                   1.567225e+07             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
flow.large_sell_amount                  1.516057e+07             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
flow.large_normalized                   1.250000e-02             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
flow.xl_classified_amount               7.654917e+07             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
flow.xl_buy_amount                      4.063932e+07             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
flow.xl_sell_amount                     3.590985e+07             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
flow.xl_normalized                      6.800000e-02             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
flow.small_large_gap                    3.686000e-01             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
flow.large_xl_gap                      -5.550000e-02             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
flow.small_xl_product                   8.400000e-03             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
flow.small_xl_aligned                   5.000000e-01             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
flow.oi_ti_gap                         -0.000000e+00             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
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
flow.oi_ti_corr                         1.000000e+00             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
flow.oi_ti_samples                      2.370000e+02             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
flow.oi_ti_lead_corr                    8.000000e-02             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
flow.flow_return_lead_corr              4.810000e-02             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
liquidity.spread                        2.400000e-03             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
liquidity.depth                         2.192194e+04             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
liquidity.depth_imbalance              -9.630000e-02             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
liquidity.paired_bid_depth              9.921941e+03             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
liquidity.paired_ask_depth              1.200000e+04             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
liquidity.amihud                        0.000000e+00             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
liquidity.impact                        2.000000e-04             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
liquidity.trade_impact_proxy            1.000000e-04             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
liquidity.bid_depth_recovery_seconds    1.200000e+02             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
liquidity.bid_depth_shock_count         2.000000e+00             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
liquidity.bid_depth_recovered_count     1.000000e+00             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
liquidity.bid_depth_censored_count      1.000000e+00             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
liquidity.bid_depth_valid_quotes        2.370000e+02             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
liquidity.ask_depth_recovery_seconds             NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.ask_depth_shock_count         0.000000e+00             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
liquidity.ask_depth_recovered_count     0.000000e+00             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
liquidity.ask_depth_censored_count      0.000000e+00             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
liquidity.ask_depth_valid_quotes        2.370000e+02             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
liquidity.total_depth_recovery_seconds           NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.total_depth_shock_count       0.000000e+00             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
liquidity.total_depth_recovered_count   0.000000e+00             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
liquidity.total_depth_censored_count    0.000000e+00             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
liquidity.total_depth_valid_quotes      2.370000e+02             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
liquidity.bid_depth                     9.921941e+03             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
liquidity.ask_depth                     1.200000e+04             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
liquidity.spread_recovery_seconds       1.293333e+03             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
liquidity.recovery_samples              4.500000e+00             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
pooled_trade.mean                       4.590076e+05             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
pooled_trade.median                     4.317520e+05             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
pooled_trade.std                        2.981406e+05             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
pooled_trade.skew                       5.013000e-01             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
pooled_trade.kurt                      -3.366000e-01             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
pooled_trade.max                        1.272119e+06             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
pooled_trade.p50                        4.317520e+05             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
pooled_trade.p90                        7.900800e+05             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
pooled_trade.p95                        1.017040e+06             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
pooled_trade.p99                        1.217128e+06             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
pooled_trade.tail_ratio                 2.819000e+00             2  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               2
```

Full tables: features.csv, baseline.csv, quality.csv.
Interactive report: [monitor.html](../monitor.html)

Securities: 2
Anomalies: 0

Large trade size is a proxy, not an identification of institutional activity.
Continuous session: 09:30-11:30 and 13:00-14:57; auctions excluded.