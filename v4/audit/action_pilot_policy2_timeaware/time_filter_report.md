# Time-Aware Action Filter Report

These are user-guided time filters evaluated on the existing pilot data. They are not proof of broad-market generalization, but they test whether the model can use time-of-day structure that appears repeatedly in the data.

| Filter | Train PnL | Train PF | Val PnL | Val PF | Test Trades | Test PnL | Test PF | Test DD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all_times | 22340 | 1.767 | 15536 | 1.364 | 176 | 10473 | 1.168 | -8867 |
| skip_first_30 | 24490 | 2.168 | 11464 | 1.344 | 154 | 17427 | 1.372 | -5611 |
| post_open_and_late | 14350 | 2.167 | 9378 | 1.496 | 89 | 15107 | 1.559 | -4051 |
| late_afternoon_only | 7960 | 3.025 | 6144 | 1.860 | 45 | 8545 | 1.832 | -1880 |
| post_open_only | 6390 | 1.764 | 3234 | 1.275 | 44 | 6562 | 1.392 | -6426 |
