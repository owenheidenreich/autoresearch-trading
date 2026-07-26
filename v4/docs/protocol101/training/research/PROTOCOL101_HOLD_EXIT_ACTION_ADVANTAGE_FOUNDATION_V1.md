# FOUNDATION_PROTOCOL101_HOLD_EXIT_ACTION_ADVANTAGE_V1

What is this: Protocol101 hold/exit action-advantage foundation
Does it change the paper-trading default: no
Paid data downloaded: no
Broker endpoint called: no
Model training: no
Untouched holdout scored: no
Decision: `protocol101_hold_exit_action_advantage_foundation_complete_training_blocked`

## Bottom Line

This packet implements the engineer-response lifecycle framing: hold/exit should be an action-advantage problem, not a blanket hold-longer rule. The dataset labels `A_hold = Q(hold) - Q(exit now at bid)` for Protocol101 selected trades, while explicitly blocking training because slot opportunity cost and fill realism are not yet included.

## Coverage

- Source trades: `1028`
- Covered trades: `709`
- Labeled holding-state rows: `198261`
- Path skips: `319`

## Exit-State Audit

- Protocol101 exit states audited: `709`
- Oracle-best hold fraction at Protocol101 exit: `0.917`
- One-step hold fraction at Protocol101 exit: `0.258`
- Median `A_hold` at Protocol101 exit: `1160.00`

## Interpretation

The high oracle-best hold fraction is not a recommendation to hold every trade longer. It says many Protocol101 exits had a later better bid somewhere before forced flat, using hindsight labels. The one-step diagnostic and the earlier exact-runner audit are the counterweight: continuation must be learned as a causal state decision with giveback, switching cost, fill uncertainty, and slot opportunity cost, not imposed as a blanket duration rule.

The actionable question remains: when does `A_hold` stay positive after subtracting the cost of keeping the only trade slot occupied and the cost of exiting/re-entering under live execution uncertainty?

## Engineer-Response Alignment

Implemented in this packet:

- `A_hold = Q(hold) - Q(exit now at bid)` label skeleton.
- Current executable bid is used for `Q(exit)`; no midpoint exit fantasy is introduced.
- Causal post-entry state features include PnL path, MFE/MAE, giveback, velocities, spread, time, and score context.
- Future/path columns are marked label-only and excluded from the feature contract.

Still missing before training:

- Counterfactual flat-slot opportunity cost.
- Explicit switching cost for exit/re-entry churn.
- Calibrated fill, latency, and quote-age uncertainty.
- Distributional risk/uncertainty targets, not just mean or future-best labels.
- Train/live parity for the exact lifecycle policy and candidate set.
- Untouched validation and formal overfit controls.

## Split Summary

| split | rows | trades | hold frac | one-step hold frac | median A_hold | median current PnL |
|---|---:|---:|---:|---:|---:|---:|
| q1_2026 | 43352 | 173 | 0.909 | 0.468 | 890.00 | 190.00 |
| q3_2025 | 45631 | 153 | 0.925 | 0.469 | 590.00 | 100.00 |
| q4_2024_external | 53562 | 187 | 0.912 | 0.460 | 630.00 | 220.00 |
| q4_2025 | 55716 | 196 | 0.898 | 0.453 | 530.00 | 0.00 |

## Protocol101 Exit Decision Summary

| split | exit reason | oracle action | rows | median A_hold | median current PnL |
|---|---|---|---:|---:|---:|
| q1_2026 | hard_stop | hold | 4 | 2885.00 | -1070.00 |
| q1_2026 | protocol054_fallback | exit | 2 | -285.00 | 20.00 |
| q1_2026 | protocol054_fallback | hold | 61 | 1600.00 | 110.00 |
| q1_2026 | sequence_residual_override | exit | 11 | -130.00 | 270.00 |
| q1_2026 | sequence_residual_override | hold | 87 | 1560.00 | 370.00 |
| q1_2026 | target | hold | 8 | 1445.00 | 2660.00 |
| q3_2025 | hard_stop | hold | 1 | 2360.00 | -1610.00 |
| q3_2025 | protocol054_fallback | hold | 19 | 1110.00 | -70.00 |
| q3_2025 | sequence_residual_override | exit | 13 | -100.00 | 190.00 |
| q3_2025 | sequence_residual_override | hold | 119 | 1180.00 | 240.00 |
| q3_2025 | target | hold | 1 | 2510.00 | 3390.00 |
| q4_2024_external | hard_stop | hold | 1 | 910.00 | -1530.00 |
| q4_2024_external | protocol054_fallback | exit | 5 | -70.00 | -590.00 |
| q4_2024_external | protocol054_fallback | hold | 140 | 1180.00 | 120.00 |
| q4_2024_external | sequence_residual_override | exit | 5 | -170.00 | 610.00 |
| q4_2024_external | sequence_residual_override | hold | 29 | 1210.00 | 600.00 |
| q4_2024_external | target | hold | 7 | 730.00 | 2150.00 |
| q4_2025 | protocol054_fallback | exit | 1 | -80.00 | -990.00 |
| q4_2025 | protocol054_fallback | hold | 4 | 2855.00 | 125.00 |
| q4_2025 | sequence_residual_override | exit | 22 | -160.00 | 405.00 |
| q4_2025 | sequence_residual_override | hold | 166 | 1235.00 | 310.00 |
| q4_2025 | target | hold | 3 | 1730.00 | 2950.00 |

## Blockers

- Slot opportunity cost is not included because counterfactual-flat Protocol101 actions are still missing.
- Fill realism is not calibrated.
- Future-best labels are label-only diagnostics, not runtime features.
- No model should be trained from this packet until those blockers are resolved or deliberately scoped.

## Outputs

- Dataset: `v4/audit/autoresearch/protocol101_hold_exit_action_advantage_foundation_v1/protocol101_hold_exit_action_advantage.parquet`
- Summary: `v4/audit/autoresearch/protocol101_hold_exit_action_advantage_foundation_v1/summary.json`
- Split summary: `v4/audit/autoresearch/protocol101_hold_exit_action_advantage_foundation_v1/split_summary.csv`
- Exit-state audit: `v4/audit/autoresearch/protocol101_hold_exit_action_advantage_foundation_v1/protocol101_exit_state_audit.csv`
- Exit decision summary: `v4/audit/autoresearch/protocol101_hold_exit_action_advantage_foundation_v1/protocol101_exit_decision_summary.csv`
- Path skips: `v4/audit/autoresearch/protocol101_hold_exit_action_advantage_foundation_v1/path_skips.csv`
