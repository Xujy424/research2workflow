# Microstructure monitor: 2026-04-27

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
trade.count                        1.959140e+05  5.795000e+04
trade.amount                       3.761954e+09  8.135482e+08
trade.mean                         1.920207e+04  1.403880e+04
trade.median                       5.740000e+03  3.780000e+03
trade.std                          6.617208e+04  6.393157e+04
trade.skew                         3.265630e+01  2.485050e+01
trade.maximum                      9.226218e+06  3.612805e+06
trade.p90                          3.914381e+04  2.457000e+04
trade.p95                          6.996700e+04  4.705000e+04
trade.p99                          2.274000e+05  1.683880e+05
trade.tail_ratio                   3.961670e+01  4.454710e+01
trade.classified_ratio             1.000000e+00  1.000000e+00
trade.large_ratio                  4.185000e-01  3.616000e-01
trade.active_buy_ratio             4.653000e-01  2.918000e-01
trade.active_sell_ratio            5.347000e-01  7.082000e-01
trade.xl_ratio                     1.174000e-01  1.556000e-01
trade.kurt                         2.791287e+03  8.979485e+02
lifecycle.fill_rate                7.596000e-01  8.001000e-01
lifecycle.completed_ratio          9.186000e-01  9.654000e-01
lifecycle.lifetime_seconds         4.021188e+02  5.189723e+02
lifecycle.overaccounted_orders     0.000000e+00  0.000000e+00
lifecycle.context_order_count      2.613350e+05  1.148510e+05
lifecycle.submitted_volume         9.329072e+08  2.451553e+08
lifecycle.filled_volume            6.574069e+08  1.731271e+08
lifecycle.cancelled_volume         1.981425e+08  5.598848e+07
lifecycle.fill_ratio_sum           1.985159e+05  9.189110e+04
lifecycle.lifetime_seconds_sum     9.653746e+07  5.754105e+07
lifecycle.lifetime_samples         2.400720e+05  1.108750e+05
lifecycle.negative_lifetime_count  0.000000e+00  0.000000e+00
lifecycle.cancel_rate              2.124000e-01  2.284000e-01
lifecycle.buy_submitted_volume     4.341770e+08  1.235924e+08
lifecycle.buy_cancelled_volume     9.945383e+07  2.942418e+07
lifecycle.buy_cancel_rate          2.291000e-01  2.381000e-01
lifecycle.sell_submitted_volume    4.987302e+08  1.215629e+08
lifecycle.sell_cancelled_volume    9.868871e+07  2.656431e+07
lifecycle.sell_cancel_rate         1.979000e-01  2.185000e-01
lifecycle.cancel_side_gap          3.120000e-02  1.960000e-02
quality.buy_order_match_ratio      1.000000e+00  1.000000e+00
quality.sell_order_match_ratio     1.000000e+00  1.000000e+00
quality.direction_known_ratio      1.000000e+00  1.000000e+00
quality.priced_order_ratio         9.966000e-01  1.000000e+00
quality.cancel_order_match_ratio   1.000000e+00  1.000000e+00
quality.quotes_count               0.000000e+00  0.000000e+00
quality.duplicate_trade_rows       0.000000e+00  0.000000e+00
```

## Core Baselines

```text
                                                 raw  sample_count  ma5  ma20  std20  z20  ma60  std60  z60  pct_1y  pct_history  pct_1y_samples
trade.count                             1.269320e+05             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.amount                            2.287751e+09             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.mean                              1.662043e+04             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.median                            4.760000e+03             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.std                               6.505183e+04             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.skew                              2.875340e+01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.maximum                           6.419512e+06             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.p90                               3.185690e+04             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.p95                               5.850850e+04             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.p99                               1.978940e+05             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.tail_ratio                        4.208190e+01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.classified_ratio                  1.000000e+00             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.large_ratio                       3.901000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.active_buy_ratio                  3.785000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.active_sell_ratio                 6.215000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.xl_ratio                          1.365000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
trade.kurt                              1.844618e+03             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_flow                        -8.853632e+07             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.medium_flow                       -7.087753e+07             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.large_flow                        -8.197717e+07             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.xl_flow                           -5.863204e+07             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.imbalance                         -2.430000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_classified_amount            5.721712e+08             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_buy_amount                   2.418174e+08             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_sell_amount                  3.303538e+08             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_normalized                  -1.575000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.medium_classified_amount           7.812497e+08             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.medium_buy_amount                  3.551861e+08             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.medium_sell_amount                 4.260636e+08             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.medium_normalized                 -1.917000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.large_classified_amount            6.502939e+08             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.large_buy_amount                   2.841584e+08             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.large_sell_amount                  3.661356e+08             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.large_normalized                  -3.119000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.xl_classified_amount               2.840364e+08             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.xl_buy_amount                      1.127022e+08             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.xl_sell_amount                     1.713342e+08             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.xl_normalized                     -4.436000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_large_gap                    1.544000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.large_xl_gap                       1.318000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_xl_product                   7.190000e-02             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_xl_aligned                   1.000000e+00             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.oi_ti_gap                          2.093000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_medium_agreement             8.304000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_medium_divergence            1.696000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_large_agreement              7.882000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_large_divergence             2.118000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_xl_agreement                 7.377000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.small_xl_divergence                2.623000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.medium_large_agreement             8.297000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.medium_large_divergence            1.703000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.medium_xl_agreement                8.013000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.medium_xl_divergence               1.987000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.large_xl_agreement                 8.144000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.large_xl_divergence                1.856000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.direction_agreement_rate           7.377000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.direction_divergence_rate          2.623000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.oi_ti_corr                         5.718000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.oi_ti_samples                      2.370000e+02             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.oi_ti_lead_corr                    1.340000e-01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
flow.flow_return_lead_corr              7.760000e-02             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.spread                                 NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.depth                                  NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.depth_imbalance                        NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.paired_bid_depth                       NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.paired_ask_depth                       NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.amihud                        0.000000e+00             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.impact                                 NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.trade_impact_proxy           -2.000000e-04             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
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
pooled_trade.mean                       1.802344e+04             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
pooled_trade.median                     5.700000e+03             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
pooled_trade.std                        6.570299e+04             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
pooled_trade.skew                       3.097400e+01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
pooled_trade.kurt                       2.400589e+03             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
pooled_trade.max                        9.226218e+06             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
pooled_trade.p50                        5.700000e+03             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
pooled_trade.p90                        3.537100e+04             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
pooled_trade.p95                        6.445600e+04             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
pooled_trade.p99                        2.160270e+05             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
pooled_trade.tail_ratio                 3.789950e+01             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
```

Full tables: features.csv, baseline.csv, quality.csv.
Interactive report: [monitor.html](../monitor.html)

Securities: 2
Anomalies: 0

Large trade size is a proxy, not an identification of institutional activity.
Continuous session: 09:30-11:30 and 13:00-14:57; auctions excluded.