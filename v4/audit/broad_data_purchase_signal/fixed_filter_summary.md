# Fixed Time-Filter Robustness

This summary reuses the broad purchase run and scores each fixed time window across seeds on March holdout. No thresholds or filters are reselected here.

| Policy | Filter | Runs | Trades | Median PnL | Median PF | Median DD | Positive Seeds | Positive Days | Top-Day Share | Pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ask_to_bid_stop35_target60_hold10m | all_times | 10 | 87 | 3002 | 1.249 | -2698 | 0.80 | 0.49 | 0.30 | False |
| ask_to_bid_stop35_target60_hold10m | late_afternoon_only | 10 | 20 | 12 | 0.999 | -1483 | 0.50 | 0.50 | 0.42 | False |
| ask_to_bid_stop35_target60_hold10m | post_open_and_late | 10 | 47 | 1746 | 1.596 | -1686 | 0.90 | 0.56 | 0.27 | False |
| ask_to_bid_stop35_target60_hold10m | post_open_only | 10 | 10 | 1561 | 3.179 | -363 | 0.90 | 0.66 | 0.32 | False |
| ask_to_bid_stop35_target60_hold10m | skip_first_30 | 10 | 84 | 2946 | 1.289 | -2500 | 0.80 | 0.50 | 0.30 | False |
| ask_to_bid_stop50_target100_hold25m | all_times | 10 | 234 | 3497 | 1.059 | -7829 | 0.70 | 0.55 | 0.22 | False |
| ask_to_bid_stop50_target100_hold25m | late_afternoon_only | 10 | 86 | 490 | 1.025 | -3331 | 0.50 | 0.49 | 0.28 | False |
| ask_to_bid_stop50_target100_hold25m | post_open_and_late | 10 | 136 | 5106 | 1.169 | -5767 | 0.70 | 0.50 | 0.25 | False |
| ask_to_bid_stop50_target100_hold25m | post_open_only | 10 | 54 | 3252 | 1.308 | -4257 | 0.70 | 0.49 | 0.24 | False |
| ask_to_bid_stop50_target100_hold25m | skip_first_30 | 10 | 228 | 4870 | 1.090 | -7878 | 0.70 | 0.52 | 0.20 | False |
| ask_to_bid_stop65_target150_hold45m | all_times | 10 | 174 | 772 | 1.018 | -11290 | 0.60 | 0.45 | 0.23 | False |
| ask_to_bid_stop65_target150_hold45m | late_afternoon_only | 10 | 46 | 4584 | 1.357 | -2757 | 1.00 | 0.48 | 0.30 | False |
| ask_to_bid_stop65_target150_hold45m | post_open_and_late | 10 | 88 | 3237 | 1.120 | -5326 | 0.90 | 0.55 | 0.23 | False |
| ask_to_bid_stop65_target150_hold45m | post_open_only | 10 | 42 | 229 | 1.013 | -7426 | 0.50 | 0.45 | 0.23 | False |
| ask_to_bid_stop65_target150_hold45m | skip_first_30 | 10 | 154 | 6318 | 1.127 | -6904 | 0.80 | 0.59 | 0.19 | False |
