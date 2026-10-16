# Microstructure monitor: 2026-07-24

Market metrics are equal-weight security averages.

- liquidity_regime: unavailable
- trade_size_regime: insufficient_history
- orderflow_regime: sell
- impact_regime: unavailable
- institution_regime: insufficient_history
- institution_proxy_regime: insufficient_history
- intraday_regime: opening_dominant
- overall: INSUFFICIENT_HISTORY

## Core Features

```text
                                      000001.SZ     600000.SH
trade.count                        9.039100e+04  5.021300e+04
trade.amount                       1.244562e+09  4.506545e+08
trade.mean                         1.376865e+04  8.974857e+03
trade.median                       4.444000e+03  2.715000e+03
trade.std                          3.995442e+04  3.151002e+04
trade.skew                         1.902140e+01  3.112810e+01
trade.maximum                      2.503137e+06  2.565792e+06
trade.p90                          3.221900e+04  1.814000e+04
trade.p95                          5.453700e+04  3.352200e+04
trade.p99                          1.526880e+05  9.966000e+04
trade.tail_ratio                   3.435820e+01  3.670720e+01
trade.classified_ratio             1.000000e+00  1.000000e+00
trade.large_ratio                  3.018000e-01  2.322000e-01
trade.active_buy_ratio             4.175000e-01  4.970000e-01
trade.active_sell_ratio            5.825000e-01  5.030000e-01
trade.xl_ratio                     5.280000e-02  4.910000e-02
trade.kurt                         7.658993e+02  1.891906e+03
lifecycle.fill_rate                7.297000e-01  8.086000e-01
lifecycle.completed_ratio          9.062000e-01  9.620000e-01
lifecycle.lifetime_seconds         5.072482e+02  2.701832e+02
lifecycle.overaccounted_orders     0.000000e+00  0.000000e+00
lifecycle.context_order_count      1.270960e+05  9.565200e+04
lifecycle.submitted_volume         3.754000e+08  1.525266e+08
lifecycle.filled_volume            2.245150e+08  9.951336e+07
lifecycle.cancelled_volume         1.094511e+08  3.914161e+07
lifecycle.fill_ratio_sum           9.274009e+04  7.734605e+04
lifecycle.lifetime_seconds_sum     5.841926e+07  2.486117e+07
lifecycle.lifetime_samples         1.151690e+05  9.201600e+04
lifecycle.negative_lifetime_count  0.000000e+00  0.000000e+00
lifecycle.cancel_rate              2.916000e-01  2.566000e-01
lifecycle.buy_submitted_volume     1.722233e+08  7.178390e+07
lifecycle.buy_cancelled_volume     5.573278e+07  1.978108e+07
lifecycle.buy_cancel_rate          3.236000e-01  2.756000e-01
lifecycle.sell_submitted_volume    2.031767e+08  8.074271e+07
lifecycle.sell_cancelled_volume    5.371829e+07  1.936053e+07
lifecycle.sell_cancel_rate         2.644000e-01  2.398000e-01
lifecycle.cancel_side_gap          5.920000e-02  3.580000e-02
quality.buy_order_match_ratio      1.000000e+00  1.000000e+00
quality.sell_order_match_ratio     1.000000e+00  1.000000e+00
quality.direction_known_ratio      1.000000e+00  1.000000e+00
quality.priced_order_ratio         9.949000e-01  1.000000e+00
quality.cancel_order_match_ratio   1.000000e+00  1.000000e+00
quality.quotes_count               0.000000e+00  0.000000e+00
quality.duplicate_trade_rows       0.000000e+00  0.000000e+00
```

## Core Baselines

```text
                                                 raw  sample_count  ma5  ma20  std20  z20  ma60  std60  z60  pct_1y  pct_history  pct_1y_samples
trade.count                             7.030200e+04             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.amount                            8.476083e+08             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.mean                              1.137175e+04             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.median                            3.579500e+03             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.std                               3.573222e+04             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.skew                              2.507480e+01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.maximum                           2.534464e+06             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.p90                               2.517950e+04             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.p95                               4.402950e+04             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.p99                               1.261740e+05             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.tail_ratio                        3.553270e+01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.classified_ratio                  1.000000e+00             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.large_ratio                       2.670000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.active_buy_ratio                  4.572000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.active_sell_ratio                 5.428000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.xl_ratio                          5.100000e-02             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.kurt                              1.328903e+03             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_flow                         6.469136e+06             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.medium_flow                       -6.546503e+07             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.large_flow                        -4.403961e+07             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.xl_flow                           -9.912820e+05             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.imbalance                         -8.550000e-02             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_classified_amount            2.770755e+08             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_buy_amount                   1.417723e+08             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_sell_amount                  1.353032e+08             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_normalized                   2.790000e-02             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.medium_classified_amount           3.304064e+08             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.medium_buy_amount                  1.324707e+08             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.medium_sell_amount                 1.979357e+08             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.medium_normalized                 -1.266000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.large_classified_amount            1.961988e+08             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.large_buy_amount                   7.607958e+07             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.large_sell_amount                  1.201192e+08             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.large_normalized                  -1.909000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.xl_classified_amount               4.392768e+07             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.xl_buy_amount                      2.146820e+07             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.xl_sell_amount                     2.245948e+07             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.xl_normalized                     -3.120000e-02             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_large_gap                    2.188000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.large_xl_gap                      -1.597000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_xl_product                  -1.100000e-03             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_xl_aligned                   0.000000e+00             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.oi_ti_gap                          1.500000e-02             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_medium_agreement             7.353000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_medium_divergence            2.647000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_large_agreement              6.366000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_large_divergence             3.634000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_xl_agreement                 6.197000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_xl_divergence                3.803000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.medium_large_agreement             7.673000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.medium_large_divergence            2.327000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.medium_xl_agreement                6.607000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.medium_xl_divergence               3.393000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.large_xl_agreement                 6.019000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.large_xl_divergence                3.981000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.direction_agreement_rate           6.197000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.direction_divergence_rate          3.803000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.oi_ti_corr                         6.335000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.oi_ti_samples                      2.370000e+02             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.oi_ti_lead_corr                    1.110000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.flow_return_lead_corr             -2.500000e-03             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.spread                                 NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.depth                                  NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.depth_imbalance                        NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.paired_bid_depth                       NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.paired_ask_depth                       NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.amihud                        0.000000e+00             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.impact                                 NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.trade_impact_proxy           -3.000000e-04             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.bid_depth_recovery_seconds             NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.bid_depth_shock_count                  NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.bid_depth_recovered_count              NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.bid_depth_censored_count               NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.bid_depth_valid_quotes                 NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.ask_depth_recovery_seconds             NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.ask_depth_shock_count                  NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.ask_depth_recovered_count              NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.ask_depth_censored_count               NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.ask_depth_valid_quotes                 NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.total_depth_recovery_seconds           NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.total_depth_shock_count                NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.total_depth_recovered_count            NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.total_depth_censored_count             NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.total_depth_valid_quotes               NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.bid_depth                              NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.ask_depth                              NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.spread_recovery_seconds                NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.recovery_samples              0.000000e+00             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
pooled_trade.mean                       1.205667e+04             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
pooled_trade.median                     3.339000e+03             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
pooled_trade.std                        3.723048e+04             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
pooled_trade.skew                       2.189110e+01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
pooled_trade.kurt                       1.000376e+03             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
pooled_trade.max                        2.565792e+06             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
pooled_trade.p50                        3.339000e+03             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
pooled_trade.p90                        2.634967e+04             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
pooled_trade.p95                        4.785900e+04             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
pooled_trade.p99                        1.346692e+05             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
pooled_trade.tail_ratio                 4.033220e+01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
```

Full tables: features.csv, baseline.csv, quality.csv.
Interactive report: [monitor.html](../monitor.html)

Securities: 2
Anomalies: 0

Large trade size is a proxy, not an identification of institutional activity.
Continuous session: 09:30-11:30 and 13:00-14:57; auctions excluded.