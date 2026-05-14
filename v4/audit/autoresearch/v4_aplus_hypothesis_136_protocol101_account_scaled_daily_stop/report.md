# Protocol 136: Account-Scaled Daily Stop

No paid data was downloaded. No broker endpoint was called. No orders were placed. Protocol101 remains frozen.

- Decision: `pass_account_scaled_daily_stop_candidate_not_live`
- Stopped after 3 failed hypotheses: `False`

## Experiments

| experiment | pass | total_inc_vs_current | min_inc | risk_ok |
| --- | --- | ---: | ---: | --- |
| daily_stop_0.25%_equity | `True` | $9,860 | $0 | `True` |
| daily_stop_0.50%_equity | `True` | $108,930 | $4,680 | `True` |
| daily_stop_0.75%_equity | `False` | $163,010 | $7,400 | `False` |
| daily_stop_1.00%_equity | `False` | $197,920 | $7,890 | `False` |

## Accepted

`daily_stop_0.25%_equity, daily_stop_0.50%_equity`

## Outputs

- Daily-stop summary: `v4/audit/autoresearch/v4_aplus_hypothesis_136_protocol101_account_scaled_daily_stop/daily_stop_summary.csv`
- Experiment summary: `v4/audit/autoresearch/v4_aplus_hypothesis_136_protocol101_account_scaled_daily_stop/experiment_summary.csv`

## Next Gate

Fold the accepted daily-stop rule into the single offline sizing candidate and rerun attribution/visualization.
