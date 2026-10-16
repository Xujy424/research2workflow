# Microstructure monitor: 2026-09-28

Market metrics are equal-weight security averages.

- liquidity_regime: insufficient_history
- trade_size_regime: insufficient_history
- orderflow_regime: neutral
- impact_regime: insufficient_history
- institution_regime: insufficient_history
- institution_proxy_regime: insufficient_history
- intraday_regime: closing_dominant
- overall: INSUFFICIENT_HISTORY

## Core Features

```text
                                      000001.SZ     600000.SH
trade.count                        2.370000e+02  2.370000e+02
trade.amount                       1.067388e+08  9.641709e+07
trade.mean                         4.503747e+05  4.068232e+05
trade.median                       4.633887e+05  3.951853e+05
trade.std                          2.578858e+05  2.599901e+05
trade.skew                         1.010000e-01  2.171000e-01
trade.maximum                      1.033596e+06  1.013373e+06
trade.p90                          7.613850e+05  7.455097e+05
trade.p95                          8.394788e+05  7.937127e+05
trade.p99                          9.916575e+05  9.704591e+05
trade.tail_ratio                   2.140000e+00  2.455700e+00
trade.classified_ratio             1.000000e+00  1.000000e+00
trade.large_ratio                  9.854000e-01  9.827000e-01
trade.active_buy_ratio             4.822000e-01  5.622000e-01
trade.active_sell_ratio            5.178000e-01  4.378000e-01
trade.xl_ratio                     6.976000e-01  6.378000e-01
trade.kurt                        -9.598000e-01 -1.000800e+00
lifecycle.fill_rate                4.895000e-01  5.527000e-01
lifecycle.completed_ratio          4.895000e-01  5.527000e-01
lifecycle.lifetime_seconds         0.000000e+00  0.000000e+00
lifecycle.overaccounted_orders     0.000000e+00  0.000000e+00
lifecycle.context_order_count      2.370000e+02  2.370000e+02
lifecycle.submitted_volume         1.069715e+07  9.692712e+06
lifecycle.filled_volume            5.158674e+06  5.451384e+06
lifecycle.cancelled_volume         0.000000e+00  0.000000e+00
lifecycle.fill_ratio_sum           1.160000e+02  1.310000e+02
lifecycle.lifetime_seconds_sum     0.000000e+00  0.000000e+00
lifecycle.lifetime_samples         1.160000e+02  1.310000e+02
lifecycle.negative_lifetime_count  0.000000e+00  0.000000e+00
lifecycle.cancel_rate              0.000000e+00  0.000000e+00
lifecycle.buy_submitted_volume     5.158674e+06  5.451384e+06
lifecycle.buy_cancelled_volume     0.000000e+00  0.000000e+00
lifecycle.buy_cancel_rate          0.000000e+00  0.000000e+00
lifecycle.sell_submitted_volume    5.538479e+06  4.241328e+06
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
trade.count                             2.370000e+02             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
trade.amount                            1.015780e+08             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
trade.mean                              4.285989e+05             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
trade.median                            4.292870e+05             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
trade.std                               2.589380e+05             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
trade.skew                              1.590000e-01             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
trade.maximum                           1.023484e+06             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
trade.p90                               7.534474e+05             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
trade.p95                               8.165958e+05             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
trade.p99                               9.810583e+05             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
trade.tail_ratio                        2.297900e+00             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
trade.classified_ratio                  1.000000e+00             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
trade.large_ratio                       9.840000e-01             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
trade.active_buy_ratio                  5.222000e-01             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
trade.active_sell_ratio                 4.778000e-01             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
trade.xl_ratio                          6.677000e-01             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
trade.kurt                             -9.803000e-01             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.small_flow                         1.434886e+04             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.medium_flow                       -1.196413e+05             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.large_flow                         1.294751e+06             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.xl_flow                            2.911266e+06             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.imbalance                          4.440000e-02             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.small_classified_amount            5.870169e+04             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.small_buy_amount                   3.652527e+04             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.small_sell_amount                  2.217641e+04             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.small_normalized                   3.523000e-01             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.medium_classified_amount           1.556466e+06             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.medium_buy_amount                  7.184122e+05             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.medium_sell_amount                 8.380535e+05             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.medium_normalized                 -7.850000e-02             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.large_classified_amount            3.198654e+07             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.large_buy_amount                   1.664065e+07             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.large_sell_amount                  1.534589e+07             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.large_normalized                   3.560000e-02             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.xl_classified_amount               6.797624e+07             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.xl_buy_amount                      3.544376e+07             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.xl_sell_amount                     3.253249e+07             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.xl_normalized                      4.860000e-02             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.small_large_gap                    3.167000e-01             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.large_xl_gap                      -1.300000e-02             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.small_xl_product                   1.700000e-03             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.small_xl_aligned                   5.000000e-01             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.oi_ti_gap                         -0.000000e+00             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
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
flow.oi_ti_corr                         1.000000e+00             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.oi_ti_samples                      2.370000e+02             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.oi_ti_lead_corr                    8.000000e-02             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.flow_return_lead_corr              4.810000e-02             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
liquidity.spread                        2.200000e-03             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
liquidity.depth                         2.192194e+04             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
liquidity.depth_imbalance              -9.630000e-02             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
liquidity.paired_bid_depth              9.921941e+03             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
liquidity.paired_ask_depth              1.200000e+04             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
liquidity.amihud                        0.000000e+00             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
liquidity.impact                        2.000000e-04             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
liquidity.trade_impact_proxy            1.000000e-04             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
liquidity.bid_depth_recovery_seconds    1.200000e+02             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
liquidity.bid_depth_shock_count         2.000000e+00             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
liquidity.bid_depth_recovered_count     1.000000e+00             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
liquidity.bid_depth_censored_count      1.000000e+00             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
liquidity.bid_depth_valid_quotes        2.370000e+02             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
liquidity.ask_depth_recovery_seconds             NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.ask_depth_shock_count         0.000000e+00             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
liquidity.ask_depth_recovered_count     0.000000e+00             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
liquidity.ask_depth_censored_count      0.000000e+00             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
liquidity.ask_depth_valid_quotes        2.370000e+02             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
liquidity.total_depth_recovery_seconds           NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.total_depth_shock_count       0.000000e+00             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
liquidity.total_depth_recovered_count   0.000000e+00             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
liquidity.total_depth_censored_count    0.000000e+00             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
liquidity.total_depth_valid_quotes      2.370000e+02             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
liquidity.bid_depth                     9.921941e+03             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
liquidity.ask_depth                     1.200000e+04             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
liquidity.spread_recovery_seconds       1.293333e+03             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
liquidity.recovery_samples              4.500000e+00             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
pooled_trade.mean                       4.285989e+05             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
pooled_trade.median                     4.253355e+05             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
pooled_trade.std                        2.595831e+05             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
pooled_trade.skew                       1.555000e-01             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
pooled_trade.kurt                      -9.890000e-01             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
pooled_trade.max                        1.033596e+06             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
pooled_trade.p50                        4.253355e+05             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
pooled_trade.p90                        7.549420e+05             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
pooled_trade.p95                        8.263515e+05             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
pooled_trade.p99                        9.889136e+05             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
pooled_trade.tail_ratio                 2.325000e+00             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
```

Full tables: features.csv, baseline.csv, quality.csv.
Interactive report: [monitor.html](../monitor.html)

Securities: 2
Anomalies: 0

Large trade size is a proxy, not an identification of institutional activity.
Continuous session: 09:30-11:30 and 13:00-14:57; auctions excluded.