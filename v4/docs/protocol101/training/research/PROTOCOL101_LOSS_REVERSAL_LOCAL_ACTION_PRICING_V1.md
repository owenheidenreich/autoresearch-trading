# AUDIT_PROTOCOL101_LOSS_REVERSAL_LOCAL_ACTION_PRICING_V1

What is this: local action pricing for `PROTOCOL101_LOSS_REVERSAL_EXIT_GATE_V1`
Does it change the paper-trading default: no
Paid data downloaded: no
Broker endpoint called: no
Model training: no
Untouched holdout scored: no
Decision: `protocol101_loss_reversal_local_action_pricing_complete_full_serial_replay_required_training_blocked`

## Bottom Line

The loss-reversal idea survives the first local action-pricing check, especially in the priority-1 bucket. This is still not a trading rule or neural label because the effects are local and not full serial-account replay.

- Priced rows: `441`
- Exit-now minus keep total: `$153,980`
- Switch minus keep total: `$511,500`
- Switch minus keep under `$0.10` two-side stress: `$502,680`
- Switch minus keep under `$0.25` two-side stress: `$489,450`
- Priority-1 rows: `145`
- Priority-1 switch minus keep under `$0.25` stress: `$223,120`

## Pricing Summary

| bucket | rows | exit-now delta | switch delta | switch delta $0.10 | switch delta $0.25 | switch positive $0.25 | best switch rows $0.25 | keep rows $0.25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all_matched_loss_reversal_rows | 441 | $153,980 | $511,500 | $502,680 | $489,450 | 0.887 | 377 | 50 |
| live_test:priority_1_current_loss_opposite_side_negative_next_and_exit_deteriorates | 145 | $86,960 | $230,370 | $227,470 | $223,120 | 1.000 | 145 | 0 |
| live_test:priority_2_current_loss_opposite_side | 20 | $-150 | $6,650 | $6,250 | $5,650 | 0.900 | 16 | 2 |
| live_test:priority_3_current_loss_same_side | 128 | $-30,610 | $30,700 | $28,140 | $24,300 | 0.625 | 78 | 48 |
| live_test:priority_4_not_yet_loss_but_negative_next_and_exit_deteriorates | 135 | $94,320 | $237,320 | $234,620 | $230,570 | 1.000 | 129 | 0 |
| live_test:priority_5_final_loss_but_live_state_not_stale_enough | 13 | $3,460 | $6,460 | $6,200 | $5,810 | 1.000 | 9 | 0 |
| split:march_2026 | 81 | $38,310 | $137,330 | $135,710 | $133,280 | 0.926 | 73 | 6 |
| split:q1_2026 | 158 | $60,700 | $207,460 | $204,300 | $199,560 | 0.880 | 130 | 19 |
| split:q3_2025 | 60 | $20,830 | $53,090 | $51,890 | $50,090 | 0.867 | 51 | 8 |
| split:q4_2025 | 142 | $34,140 | $113,620 | $110,780 | $106,520 | 0.880 | 123 | 17 |
| relation:opposite_side | 299 | $180,890 | $473,980 | $468,000 | $459,030 | 0.993 | 289 | 2 |
| relation:same_contract | 47 | $-11,010 | $4,820 | $3,880 | $2,470 | 0.532 | 23 | 22 |
| relation:same_side_different_contract | 95 | $-15,900 | $32,700 | $30,800 | $27,950 | 0.726 | 65 | 26 |

## Stopping Point

This packet prices the local decision and gives us a viable first playbook candidate. The next step must be full serial replay, because switching out of one trade changes all later account state and may block or create future Protocol101 opportunities.

No neural model should be trained until `PROTOCOL101_LOSS_REVERSAL_FULL_SERIAL_REPLAY_V1` defines the label under one account, one contract, no overlaps, affordability, bid/ask execution, and slippage/fill stress.

## Outputs

- Summary: `v4/audit/autoresearch/protocol101_loss_reversal_local_action_pricing_v1/summary.json`
- Local action rows: `v4/audit/autoresearch/protocol101_loss_reversal_local_action_pricing_v1/local_action_pricing_rows.csv`
- Bucket summary: `v4/audit/autoresearch/protocol101_loss_reversal_local_action_pricing_v1/local_action_bucket_summary.csv`
- Priority review rows: `v4/audit/autoresearch/protocol101_loss_reversal_local_action_pricing_v1/priority_action_review_rows.csv`
- Report: `v4/audit/autoresearch/protocol101_loss_reversal_local_action_pricing_v1/report.md`
