# Calibrated Abstention Broad Data Purchase Signal

This evaluator uses only the existing January-March pilot data. The model is trained on January, score calibration is fitted on the first half of February, abstention/risk controls are selected on the second half of February, and March remains the untouched holdout. No new paid data is used or purchased.

Recommendation: **no_broad_purchase_yet_continue_existing_data_iteration**

## Gate

| Criterion | Value |
|---|---:|
| min_test_profit_factor_median | 1.2 |
| min_test_pnl_median | 5000.0 |
| min_positive_seed_fraction | 0.7 |
| min_positive_day_fraction_median | 0.55 |
| max_drawdown_floor_median | -6000.0 |
| min_test_trades_median | 40.0 |
| max_top_day_profit_share_median | 0.45 |

## Policy Robustness

| Policy | Runs | Filters | Scores | Max Trades | Daily Stops | Median PnL | Median PF | Median DD | Trades | Positive Seeds | Positive Days | Top-Day Share | Pass |
|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| ask_to_bid_stop35_target60_hold10m | 10 | {'all_times': 9, 'skip_first_30': 1} | {'directional_margin': 4, 'edge_plus_directional_margin': 1, 'top_score': 5} | {'2': 2, '3': 1, '4': 3, '99': 4} | {'-1000.0': 4, 'None': 6} | 56 | 1.010 | -2508 | 45 | 0.50 | 0.47 | 0.30 | False |
| ask_to_bid_stop50_target100_hold25m | 10 | {'all_times': 9, 'late_afternoon_only': 1} | {'directional_margin': 4, 'edge_vs_no_trade': 2, 'top_score': 4} | {'2': 2, '3': 1, '4': 4, '99': 3} | {'-1000.0': 3, 'None': 7} | 1679 | 1.113 | -3720 | 46 | 0.70 | 0.42 | 0.27 | False |
| ask_to_bid_stop65_target150_hold45m | 10 | {'all_times': 4, 'late_afternoon_only': 6} | {'directional_margin': 5, 'edge_vs_no_trade': 1, 'top_score': 4} | {'2': 7, '4': 2, '99': 1} | {'None': 10} | -274 | 1.001 | -4874 | 44 | 0.50 | 0.45 | 0.28 | False |

## Interpretation

Calibration is fitted on the first half of February only. Abstention and risk controls are selected on the second half of February only. March is held out for the purchase decision. A fail here means the project should continue improving on the existing pilot before buying a broad history.
