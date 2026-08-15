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
| ask_to_bid_stop35_target60_hold10m | 10 | {'all_times': 6, 'late_afternoon_only': 2, 'post_open_only': 2} | 2170 | 1.107 | -2527 | 0.70 | 0.50 | 0.28 | False |
| ask_to_bid_stop50_target100_hold25m | 10 | {'all_times': 3, 'late_afternoon_only': 6, 'post_open_only': 1} | 1310 | 1.047 | -3553 | 0.60 | 0.49 | 0.27 | False |
| ask_to_bid_stop65_target150_hold45m | 10 | {'all_times': 2, 'late_afternoon_only': 6, 'post_open_only': 2} | 2806 | 1.175 | -3674 | 0.90 | 0.45 | 0.23 | False |
