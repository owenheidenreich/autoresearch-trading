# Broad Data Purchase Signal

Broad data purchase requires repeated evidence on the existing pilot, with validation-only threshold/filter selection and March holdout scoring. This does not use or purchase new data.

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

| Policy | Runs | Filters | Median PnL | Median PF | Median DD | Positive Seeds | Positive Days | Top-Day Share | Pass |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| ask_to_bid_stop35_target60_hold10m | 10 | {'all_times': 2, 'late_afternoon_only': 6, 'post_open_only': 2} | 2593 | 1.586 | -1439 | 0.80 | 0.50 | 0.41 | False |
| ask_to_bid_stop50_target100_hold25m | 10 | {'all_times': 3, 'late_afternoon_only': 4, 'post_open_and_late': 1, 'skip_first_30': 2} | 2448 | 1.490 | -2544 | 0.90 | 0.55 | 0.37 | False |
| ask_to_bid_stop65_target150_hold45m | 10 | {'all_times': 3, 'late_afternoon_only': 5, 'post_open_only': 1, 'skip_first_30': 1} | 3505 | 1.154 | -6246 | 0.80 | 0.50 | 0.27 | False |
