# AUDIT_PROTOCOL101_LOSS_REVERSAL_CAUSAL_STATE_ATTACHMENT_V1

What is this: direct quote-path causal-state attachment for `PROTOCOL101_LOSS_REVERSAL_EXIT_GATE_V1`
Does it change the paper-trading default: no
Paid data downloaded: no
Broker endpoint called: no
Model training: no
Untouched holdout scored: no
Decision: `protocol101_loss_reversal_causal_state_attachment_partial_quote_coverage_replay_scaffold_ready_training_blocked`

## Bottom Line

Direct quote-path attachment substantially improves the selected hypothesis from a final-PnL story into a live-state diagnostic. It still does not authorize neural training.

- Loss-reversal candidates: `485`
- Direct quote state matched: `441`
- Match rate: `0.909`
- Matched positive slot-cost: `$590,500`
- Current-loss rows at blocked signal: `293`
- Current-loss plus opposite-side rows: `165`
- Priority-1 rows: `145`
- Priority-1 positive slot-cost: `$268,080`

The next valid experiment is a mutually exclusive replay, not a neural net. For each matched row, compare the actual keep-holding path against exit-now and exit/switch-to-later-signal, charging spread, latency, fill uncertainty, and single-slot opportunity cost.

## Direct State Summary

| bucket | rows | slot-cost total | matched | match rate | current loss | current positive | one-step negative | hold-to-exit worse | median current PnL | median exit-current |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all_loss_reversal_candidates | 485 | $635,240 | 441 | 0.909 | 293 | 148 | 300 | 317 | $-120 | $-250 |
| subcase:final_loss_plus_opposite_side_signal | 325 | $517,980 | 299 | 0.920 | 165 | 134 | 293 | 285 | $-60 | $-420 |
| subcase:final_loss_plus_same_side_signal | 160 | $117,260 | 142 | 0.887 | 128 | 14 | 7 | 32 | $-450 | $190 |
| live_test:blocked_no_readable_session_quotes | 44 | $44,740 | 0 | 0.000 | 0 | 0 | 0 | 0 | $0 | $0 |
| live_test:priority_1_current_loss_opposite_side_negative_next_and_exit_deteriorates | 145 | $268,080 | 145 | 1.000 | 145 | 0 | 145 | 145 | $-180 | $-480 |
| live_test:priority_2_current_loss_opposite_side | 20 | $9,870 | 20 | 1.000 | 20 | 0 | 14 | 6 | $-80 | $20 |
| live_test:priority_3_current_loss_same_side | 128 | $99,390 | 128 | 1.000 | 128 | 0 | 6 | 18 | $-480 | $200 |
| live_test:priority_4_not_yet_loss_but_negative_next_and_exit_deteriorates | 135 | $208,580 | 135 | 1.000 | 0 | 135 | 135 | 135 | $160 | $-560 |
| live_test:priority_5_final_loss_but_live_state_not_stale_enough | 13 | $4,580 | 13 | 1.000 | 0 | 13 | 0 | 13 | $120 | $-150 |
| split:march_2026 | 89 | $163,500 | 81 | 0.910 | 51 | 30 | 58 | 59 | $-120 | $-390 |
| split:q1_2026 | 176 | $258,570 | 158 | 0.898 | 103 | 55 | 107 | 117 | $-120 | $-320 |
| split:q3_2025 | 64 | $72,230 | 60 | 0.938 | 45 | 15 | 33 | 42 | $-170 | $-240 |
| split:q4_2025 | 156 | $140,940 | 142 | 0.910 | 94 | 48 | 102 | 99 | $-115 | $-165 |

## Stopping Point

This packet reaches the next Track A stopping point. We now know which rows are live-state candidates, but we have not priced the action. The next artifact must be `PROTOCOL101_LOSS_REVERSAL_MUTUALLY_EXCLUSIVE_REPLAY_V1`.

## Outputs

- Summary: `v4/audit/autoresearch/protocol101_loss_reversal_causal_state_attachment_v1/summary.json`
- Direct state join: `v4/audit/autoresearch/protocol101_loss_reversal_causal_state_attachment_v1/direct_causal_state_join.csv`
- Direct state summary: `v4/audit/autoresearch/protocol101_loss_reversal_causal_state_attachment_v1/direct_causal_state_summary.csv`
- Manual review rows: `v4/audit/autoresearch/protocol101_loss_reversal_causal_state_attachment_v1/manual_review_priority_rows.csv`
- Quote read issues: `v4/audit/autoresearch/protocol101_loss_reversal_causal_state_attachment_v1/quote_read_issues.csv`
- Report: `v4/audit/autoresearch/protocol101_loss_reversal_causal_state_attachment_v1/report.md`
