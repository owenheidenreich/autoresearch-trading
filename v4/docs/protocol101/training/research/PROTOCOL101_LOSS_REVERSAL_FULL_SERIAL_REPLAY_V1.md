# AUDIT_PROTOCOL101_LOSS_REVERSAL_FULL_SERIAL_REPLAY_V1

What is this: full serial flat-stream replay for `PROTOCOL101_LOSS_REVERSAL_EXIT_GATE_V1` priority-1 overlay
Does it change the paper-trading default: no
Paid data downloaded: no
Broker endpoint called: no
Model training: no
Untouched holdout scored: no
Decision: `protocol101_loss_reversal_full_serial_replay_complete_training_still_blocked`

## Bottom Line

The priority-1 loss-reversal overlay survives full serial flat-stream replay on the exposed diagnostic splits. This is the first Track A result that looks like a concrete Protocol101 improvement playbook, but it is still research-only.

## Variant Summary

| variant | stress | trades | PnL | baseline PnL | delta | exits | all splits positive delta |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline | 0.00 | 4456 | $1,387,260 | $1,387,260 | $0 | 0 | True |
| priority1_exit_release_slot | 0.00 | 4574 | $1,588,610 | $1,387,260 | $201,350 | 145 | True |
| baseline | 0.10 | 4456 | $1,298,140 | $1,298,140 | $0 | 0 | True |
| priority1_exit_release_slot | 0.10 | 4574 | $1,497,130 | $1,298,140 | $198,990 | 145 | True |
| baseline | 0.25 | 4456 | $1,164,460 | $1,164,460 | $0 | 0 | True |
| priority1_exit_release_slot | 0.25 | 4574 | $1,359,910 | $1,164,460 | $195,450 | 145 | True |

## Split Summary

| variant | stress | split | trades | PnL | baseline PnL | delta | exits | PF | win rate |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| priority1_exit_release_slot | 0.00 | march_2026 | 563 | $240,240 | $190,130 | $50,110 | 25 | 4.713 | 0.702 |
| priority1_exit_release_slot | 0.00 | q1_2026 | 1289 | $515,870 | $439,420 | $76,450 | 51 | 4.450 | 0.690 |
| priority1_exit_release_slot | 0.00 | q3_2025 | 1319 | $332,540 | $302,290 | $30,250 | 23 | 4.597 | 0.798 |
| priority1_exit_release_slot | 0.00 | q4_2025 | 1403 | $499,960 | $455,420 | $44,540 | 46 | 7.103 | 0.762 |
| priority1_exit_release_slot | 0.10 | march_2026 | 563 | $228,980 | $179,310 | $49,670 | 25 | 4.363 | 0.696 |
| priority1_exit_release_slot | 0.10 | q1_2026 | 1289 | $490,090 | $414,340 | $75,750 | 51 | 4.111 | 0.686 |
| priority1_exit_release_slot | 0.10 | q3_2025 | 1319 | $306,160 | $276,450 | $29,710 | 23 | 4.127 | 0.784 |
| priority1_exit_release_slot | 0.10 | q4_2025 | 1403 | $471,900 | $428,040 | $43,860 | 46 | 6.323 | 0.751 |
| priority1_exit_release_slot | 0.25 | march_2026 | 563 | $212,090 | $163,080 | $49,010 | 25 | 3.892 | 0.679 |
| priority1_exit_release_slot | 0.25 | q1_2026 | 1289 | $451,420 | $376,720 | $74,700 | 51 | 3.656 | 0.665 |
| priority1_exit_release_slot | 0.25 | q3_2025 | 1319 | $266,590 | $237,690 | $28,900 | 23 | 3.496 | 0.749 |
| priority1_exit_release_slot | 0.25 | q4_2025 | 1403 | $429,810 | $386,970 | $42,840 | 46 | 5.317 | 0.723 |

## Interpretation

This is not a neural model and not a promotion claim. The replay uses the frozen Protocol101 flat proposal stream and a hand-declared priority-1 gate derived from Track A diagnostics. It answers a trader question: when Protocol101 is losing, an opposite-side Protocol101 signal appears, the next quote is adverse, and the original hold later deteriorates, should the bot release the slot?

On these exposed diagnostic splits, the answer is yes. The next work is not architecture search. It is manual chart/path review, formalizing the causal feature contract, live no-order parity for this gate, fill/latency evidence, and then a frozen untouched evaluation.

## Outputs

- Summary: `v4/audit/autoresearch/protocol101_loss_reversal_full_serial_replay_v1/summary.json`
- Replay trades: `v4/audit/autoresearch/protocol101_loss_reversal_full_serial_replay_v1/full_serial_replay_trades.csv`
- Replay events: `v4/audit/autoresearch/protocol101_loss_reversal_full_serial_replay_v1/full_serial_replay_events.csv`
- Split summary: `v4/audit/autoresearch/protocol101_loss_reversal_full_serial_replay_v1/split_summary.csv`
- Variant summary: `v4/audit/autoresearch/protocol101_loss_reversal_full_serial_replay_v1/variant_summary.csv`
- Report: `v4/audit/autoresearch/protocol101_loss_reversal_full_serial_replay_v1/report.md`
