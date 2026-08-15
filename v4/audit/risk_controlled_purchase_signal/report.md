# Risk-Controlled Broad Data Purchase Signal

This evaluator uses only the existing January-March pilot data. It selects the action threshold, time window, max trades per day, and daily loss stop from February validation, then scores March holdout across random seeds. No new paid data is used or purchased.

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

| Policy | Runs | Filters | Max Trades | Daily Stops | Median PnL | Median PF | Median DD | Trades | Positive Seeds | Positive Days | Top-Day Share | Pass |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| ask_to_bid_stop35_target60_hold10m | 10 | {'all_times': 9, 'late_afternoon_only': 1} | {'3': 2, '4': 3, '6': 2, '99': 3} | {'None': 2, '-1500.0': 1, '-1000.0': 2, '-500.0': 5} | 2554 | 1.187 | -2996 | 61 | 0.80 | 0.48 | 0.27 | False |
| ask_to_bid_stop50_target100_hold25m | 10 | {'all_times': 9, 'late_afternoon_only': 1} | {'2': 3, '3': 1, '4': 2, '6': 2, '99': 2} | {'-2500.0': 1, '-1000.0': 2, '-500.0': 7} | 2537 | 1.194 | -5517 | 49 | 0.70 | 0.42 | 0.24 | False |
| ask_to_bid_stop65_target150_hold45m | 10 | {'all_times': 8, 'late_afternoon_only': 2} | {'2': 3, '3': 2, '4': 3, '6': 1, '99': 1} | {'None': 3, '-1500.0': 3, '-1000.0': 2, '-500.0': 2} | -1318 | 0.944 | -11937 | 60 | 0.40 | 0.41 | 0.29 | False |

## Interpretation

A pass means the current pilot produced a broad historical-data purchase signal after validation-only selection of model threshold, time window, and basic daily risk controls. A fail does not reject the project; it means the next work should happen on the existing data before buying a much larger history.
