# Microstructure monitor: 2026-04-28

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
trade.count                        1.199580e+05  4.570800e+04
trade.amount                       1.873702e+09  5.439889e+08
trade.mean                         1.561965e+04  1.190139e+04
trade.median                       4.560000e+03  3.744000e+03
trade.std                          6.387205e+04  4.656909e+04
trade.skew                         4.679950e+01  3.383380e+01
trade.maximum                      9.202233e+06  3.195828e+06
trade.p90                          3.086100e+04  2.322500e+04
trade.p95                          5.690000e+04  4.356900e+04
trade.p99                          1.902848e+05  1.207682e+05
trade.tail_ratio                   4.172910e+01  3.225650e+01
trade.classified_ratio             1.000000e+00  1.000000e+00
trade.large_ratio                  3.944000e-01  2.647000e-01
trade.active_buy_ratio             4.641000e-01  4.311000e-01
trade.active_sell_ratio            5.359000e-01  5.689000e-01
trade.xl_ratio                     1.224000e-01  7.310000e-02
trade.kurt                         4.830834e+03  1.765805e+03
lifecycle.fill_rate                7.729000e-01  6.966000e-01
lifecycle.completed_ratio          9.590000e-01  9.554000e-01
lifecycle.lifetime_seconds         5.665936e+02  5.569918e+02
lifecycle.overaccounted_orders     0.000000e+00  0.000000e+00
lifecycle.context_order_count      1.588260e+05  9.872500e+04
lifecycle.submitted_volume         4.821318e+08  2.077907e+08
lifecycle.filled_volume            3.292912e+08  1.170808e+08
lifecycle.cancelled_volume         1.272114e+08  7.705025e+07
lifecycle.fill_ratio_sum           1.227607e+05  6.876928e+04
lifecycle.lifetime_seconds_sum     8.629673e+07  5.253770e+07
lifecycle.lifetime_samples         1.523080e+05  9.432400e+04
lifecycle.negative_lifetime_count  0.000000e+00  0.000000e+00
lifecycle.cancel_rate              2.639000e-01  3.708000e-01
lifecycle.buy_submitted_volume     2.331250e+08  1.142893e+08
lifecycle.buy_cancelled_volume     6.031842e+07  4.871190e+07
lifecycle.buy_cancel_rate          2.587000e-01  4.262000e-01
lifecycle.sell_submitted_volume    2.490068e+08  9.350139e+07
lifecycle.sell_cancelled_volume    6.689295e+07  2.833835e+07
lifecycle.sell_cancel_rate         2.686000e-01  3.031000e-01
lifecycle.cancel_side_gap         -9.900000e-03  1.231000e-01
quality.buy_order_match_ratio      1.000000e+00  1.000000e+00
quality.sell_order_match_ratio     1.000000e+00  1.000000e+00
quality.direction_known_ratio      1.000000e+00  1.000000e+00
quality.priced_order_ratio         9.955000e-01  1.000000e+00
quality.cancel_order_match_ratio   1.000000e+00  1.000000e+00
quality.quotes_count               0.000000e+00  0.000000e+00
quality.duplicate_trade_rows       0.000000e+00  0.000000e+00
```

## Core Baselines

```text
                                                 raw  sample_count  ma5  ma20  std20  z20  ma60  std60  z60  pct_1y  pct_history  pct_1y_samples
trade.count                             8.283300e+04             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
trade.amount                            1.208846e+09             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
trade.mean                              1.376052e+04             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
trade.median                            4.152000e+03             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
trade.std                               5.522057e+04             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
trade.skew                              4.031670e+01             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
trade.maximum                           6.199031e+06             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
trade.p90                               2.704300e+04             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
trade.p95                               5.023450e+04             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
trade.p99                               1.555265e+05             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
trade.tail_ratio                        3.699280e+01             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
trade.classified_ratio                  1.000000e+00             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
trade.large_ratio                       3.295000e-01             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
trade.active_buy_ratio                  4.476000e-01             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
trade.active_sell_ratio                 5.524000e-01             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
trade.xl_ratio                          9.780000e-02             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
trade.kurt                              3.298319e+03             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.small_flow                        -4.091341e+07             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.medium_flow                       -1.941461e+07             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.large_flow                        -3.919723e+07             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.xl_flow                           -5.263017e+06             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.imbalance                         -1.048000e-01             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.small_classified_amount            3.500371e+08             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.small_buy_amount                   1.545618e+08             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.small_sell_amount                  1.954752e+08             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.small_normalized                  -9.140000e-02             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.medium_classified_amount           4.173523e+08             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.medium_buy_amount                  1.989688e+08             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.medium_sell_amount                 2.183834e+08             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.medium_normalized                 -6.160000e-02             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.large_classified_amount            3.068596e+08             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.large_buy_amount                   1.338312e+08             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.large_sell_amount                  1.730284e+08             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.large_normalized                  -1.875000e-01             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.xl_classified_amount               1.345966e+08             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.xl_buy_amount                      6.466680e+07             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.xl_sell_amount                     6.992982e+07             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.xl_normalized                     -2.366000e-01             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.small_large_gap                    9.610000e-02             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.large_xl_gap                       4.910000e-02             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.small_xl_product                   5.600000e-03             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.small_xl_aligned                   5.000000e-01             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.oi_ti_gap                          1.348000e-01             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.small_medium_agreement             7.689000e-01             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.small_medium_divergence            2.311000e-01             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.small_large_agreement              7.215000e-01             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.small_large_divergence             2.785000e-01             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.small_xl_agreement                 7.155000e-01             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.small_xl_divergence                2.845000e-01             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.medium_large_agreement             7.406000e-01             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.medium_large_divergence            2.594000e-01             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.medium_xl_agreement                7.300000e-01             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.medium_xl_divergence               2.700000e-01             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.large_xl_agreement                 8.284000e-01             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.large_xl_divergence                1.716000e-01             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.direction_agreement_rate           7.155000e-01             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.direction_divergence_rate          2.845000e-01             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.oi_ti_corr                         5.823000e-01             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.oi_ti_samples                      2.370000e+02             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.oi_ti_lead_corr                    1.634000e-01             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
flow.flow_return_lead_corr              1.247000e-01             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
liquidity.spread                                 NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.depth                                  NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.depth_imbalance                        NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.paired_bid_depth                       NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.paired_ask_depth                       NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.amihud                        0.000000e+00             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
liquidity.impact                                 NaN             0  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               0
liquidity.trade_impact_proxy           -0.000000e+00             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
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
liquidity.recovery_samples              0.000000e+00             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
pooled_trade.mean                       1.459377e+04             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
pooled_trade.median                     4.556000e+03             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
pooled_trade.std                        5.962501e+04             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
pooled_trade.skew                       4.612370e+01             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
pooled_trade.kurt                       4.789681e+03             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
pooled_trade.max                        9.202233e+06             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
pooled_trade.p50                        4.556000e+03             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
pooled_trade.p90                        2.857500e+04             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
pooled_trade.p95                        5.138745e+04             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
pooled_trade.p99                        1.716525e+05             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
pooled_trade.tail_ratio                 3.767610e+01             1  NaN   NaN    NaN  NaN   NaN    NaN  NaN     NaN          NaN               1
```

Full tables: features.csv, baseline.csv, quality.csv.
Interactive report: [monitor.html](../monitor.html)

Securities: 2
Anomalies: 0

Large trade size is a proxy, not an identification of institutional activity.
Continuous session: 09:30-11:30 and 13:00-14:57; auctions excluded.