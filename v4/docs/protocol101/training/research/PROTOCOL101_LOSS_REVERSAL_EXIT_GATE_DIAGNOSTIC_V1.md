# AUDIT_PROTOCOL101_LOSS_REVERSAL_EXIT_GATE_DIAGNOSTIC_V1

What is this: no-training diagnostic for `PROTOCOL101_LOSS_REVERSAL_EXIT_GATE_V1`
Does it change the paper-trading default: no
Paid data downloaded: no
Broker endpoint called: no
Model training: no
Untouched holdout scored: no
Decision: `protocol101_loss_reversal_exit_gate_diagnostic_partial_causal_state_coverage_training_blocked`

## Bottom Line

The loss-reversal hypothesis is the right first question, but it is not ready for neural training. The slot-cost evidence used final open-trade PnL; a live rule needs the current holding state at the later blocked signal time.

- Loss-reversal candidates: `485`
- Positive slot-cost total: `$635,240`
- Causal state matched at blocked signal: `65` / `485`
- Current-loss rows among matched: `46`
- Current-positive rows among matched: `19`
- Current-loss plus opposite-side rows: `21`
- One-step hold-negative rows among matched: `39`

This is the important truth: some final losers were not actually losing at the later signal time. A model trained on final-loss slot-cost rows would learn a hindsight label unless the row is redefined around live state.

## Causal State Summary

| bucket | rows | slot-cost total | matched | match rate | current loss | current positive | exit advantage + | hold advantage + | one-step hold negative | median current PnL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all_loss_reversal_candidates | 485 | $635,240 | 65 | 0.134 | 46 | 19 | 8 | 56 | 39 | $-180 |
| subcase:final_loss_plus_opposite_side_signal | 325 | $517,980 | 38 | 0.117 | 21 | 17 | 8 | 29 | 37 | $-65 |
| subcase:final_loss_plus_same_side_signal | 160 | $117,260 | 27 | 0.169 | 25 | 2 | 0 | 27 | 2 | $-450 |
| live_test:blocked_missing_causal_state_at_signal | 420 | $566,570 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0 | $0 |
| live_test:priority_1_current_loss_opposite_side_one_step_negative | 20 | $32,790 | 20 | 1.000 | 20 | 0 | 6 | 14 | 20 | $-210 |
| live_test:priority_2_current_loss_opposite_side | 1 | $280 | 1 | 1.000 | 1 | 0 | 0 | 1 | 0 | $-60 |
| live_test:priority_3_current_loss_same_side | 25 | $17,570 | 25 | 1.000 | 25 | 0 | 0 | 25 | 2 | $-450 |
| live_test:priority_4_hindsight_exit_advantage_positive | 2 | $3,290 | 2 | 1.000 | 0 | 2 | 2 | 0 | 2 | $70 |
| live_test:priority_5_final_loss_but_not_current_loss | 17 | $14,740 | 17 | 1.000 | 0 | 17 | 0 | 16 | 15 | $160 |
| split:march_2026 | 89 | $163,500 | 11 | 0.124 | 9 | 2 | 1 | 10 | 6 | $-390 |
| split:q1_2026 | 176 | $258,570 | 24 | 0.136 | 18 | 6 | 2 | 22 | 13 | $-345 |
| split:q3_2025 | 64 | $72,230 | 8 | 0.125 | 6 | 2 | 1 | 7 | 3 | $-135 |
| split:q4_2025 | 156 | $140,940 | 22 | 0.141 | 13 | 9 | 4 | 17 | 17 | $-80 |

## Stopping Point

Do not train a loss-reversal neural gate from the current slot-cost rows. First build the full causal state attachment for every candidate row, then run a mutually exclusive keep-hold versus exit/switch replay. Only rows where the live state supports loss/staleness should become labels.

## Outputs

- Summary: `v4/audit/autoresearch/protocol101_loss_reversal_exit_gate_diagnostic_v1/summary.json`
- Loss-reversal candidates: `v4/audit/autoresearch/protocol101_loss_reversal_exit_gate_diagnostic_v1/loss_reversal_candidates.csv`
- Causal state join: `v4/audit/autoresearch/protocol101_loss_reversal_exit_gate_diagnostic_v1/causal_state_join.csv`
- Causal state summary: `v4/audit/autoresearch/protocol101_loss_reversal_exit_gate_diagnostic_v1/causal_state_summary.csv`
- Manual review priority rows: `v4/audit/autoresearch/protocol101_loss_reversal_exit_gate_diagnostic_v1/manual_review_priority_rows.csv`
- Diagnostic gate checklist: `v4/audit/autoresearch/protocol101_loss_reversal_exit_gate_diagnostic_v1/diagnostic_gate_checklist.csv`
- Report: `v4/audit/autoresearch/protocol101_loss_reversal_exit_gate_diagnostic_v1/report.md`
