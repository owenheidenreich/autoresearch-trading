# Fixed Time-Filter Robustness

This summary reuses the broad purchase run and scores each fixed time window across seeds on March holdout. No thresholds or filters are reselected here.

| Policy | Filter | Runs | Trades | Median PnL | Median PF | Median DD | Positive Seeds | Positive Days | Top-Day Share | Pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ask_to_bid_stop35_target60_hold10m | all_times | 10 | 106 | -30 | 1.003 | -4919 | 0.50 | 0.49 | 0.28 | False |
| ask_to_bid_stop35_target60_hold10m | late_afternoon_only | 10 | 34 | 2306 | 1.593 | -1506 | 0.80 | 0.50 | 0.56 | False |
| ask_to_bid_stop35_target60_hold10m | post_open_and_late | 10 | 46 | 4284 | 1.597 | -1646 | 0.90 | 0.60 | 0.36 | False |
| ask_to_bid_stop35_target60_hold10m | post_open_only | 10 | 10 | 1773 | 1.769 | -814 | 0.80 | 0.63 | 0.41 | False |
| ask_to_bid_stop35_target60_hold10m | skip_first_30 | 10 | 98 | 1100 | 1.086 | -3989 | 0.60 | 0.53 | 0.28 | False |
| ask_to_bid_stop50_target100_hold25m | all_times | 10 | 85 | 3502 | 1.141 | -3618 | 0.90 | 0.55 | 0.23 | False |
| ask_to_bid_stop50_target100_hold25m | late_afternoon_only | 10 | 31 | 1667 | 1.183 | -2086 | 0.60 | 0.49 | 0.43 | False |
| ask_to_bid_stop50_target100_hold25m | post_open_and_late | 10 | 50 | 4006 | 1.518 | -2798 | 0.90 | 0.54 | 0.29 | False |
| ask_to_bid_stop50_target100_hold25m | post_open_only | 10 | 18 | 2289 | 1.754 | -1609 | 0.70 | 0.45 | 0.51 | False |
| ask_to_bid_stop50_target100_hold25m | skip_first_30 | 10 | 85 | 4658 | 1.203 | -3426 | 0.90 | 0.59 | 0.23 | False |
| ask_to_bid_stop65_target150_hold45m | all_times | 10 | 176 | 2258 | 1.034 | -11305 | 0.70 | 0.48 | 0.22 | False |
| ask_to_bid_stop65_target150_hold45m | late_afternoon_only | 10 | 44 | 4074 | 1.379 | -3315 | 0.90 | 0.50 | 0.27 | False |
| ask_to_bid_stop65_target150_hold45m | post_open_and_late | 10 | 88 | 8814 | 1.332 | -6789 | 0.90 | 0.50 | 0.26 | False |
| ask_to_bid_stop65_target150_hold45m | post_open_only | 10 | 44 | 4029 | 1.316 | -5948 | 0.70 | 0.52 | 0.22 | False |
| ask_to_bid_stop65_target150_hold45m | skip_first_30 | 10 | 154 | 6082 | 1.166 | -8557 | 0.80 | 0.57 | 0.24 | False |
