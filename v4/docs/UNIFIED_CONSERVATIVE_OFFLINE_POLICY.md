# FOUNDATION_UNIFIED_CONSERVATIVE_OFFLINE_POLICY_V1

What is this: foundation spec / unified conservative offline policy direction
Does it change the paper-trading default: no
Paper default baseline: `PAPER_DEFAULT_PROTOCOL101`
Abandoned candidate: `CHALLENGER_INTEGRATED_ENTRY_LIFECYCLE_SERIAL_REPLAY_V1 / Protocol276`
Paid data downloaded: no
Broker endpoint called: no
Model training: no
Decision: `unified_conservative_offline_policy_foundation_frozen_no_model_training`

## Summary

The next model direction is a unified conservative offline policy, not another Protocol276 tweak. The frozen target game is `wait / enter candidate / hold / exit` under the same one-account, one-contract, ask-entry, bid-exit, flat-by-close constraints required for live/paper operation.

## Why Protocol276 Is Abandoned

- Entry policy selected 165 negative-A_enter trades for -$35,480 PnL.
- Lifecycle timing left 159 overhold/late-exit trades with -$33,275 PnL.
- Actual exits left -$424,160 versus each trade's best observed path PnL.
- Replay skipped 16799 rows, dominated by {'unaffordable_current_equity': 13912, 'missing_contract_quotes': 2887}.

## Frozen Contracts

- State/action: `UnifiedDecisionStateV1`
- Execution: `ExecutionModelV1`
- Labels: `ActionAdvantageLabelV1`
- Conservative gate: challenger actions defer to Protocol101 unless advantage clears uncertainty, OOD, and minimum-margin penalties.

## Implementation Status

| area | status | evidence |
|---|---|---|
| state/action contract | `frozen` | UnifiedDecisionStateV1 |
| execution model | `deterministic_ready_fill_calibration_blocked` | ExecutionModelV1 |
| action-advantage labels | `primitive_contract_ready` | ActionAdvantageLabelV1 |
| neural policy training | `blocked_until_foundation_gates_close` | No training run is authorized by this foundation packet. |
| paper default | `unchanged` | PAPER_DEFAULT_PROTOCOL101 |

## Next Allowed Work

1. Build trajectory-dataset extraction against the frozen UnifiedDecisionStateV1 contract.
2. Expand the DP oracle from flat and holding primitives into full wait/enter/hold/exit trajectories.
3. Collect paper/no-order fill observations before enabling stochastic fill replay.
4. Reserve a new untouched evaluation block before any future promotion claim.
5. Build live no-order parity for the exact full candidate surface and feature contract.

## Outputs

- Summary: `v4/audit/autoresearch/unified_conservative_offline_policy_foundation/summary.json`
- Report: `v4/audit/autoresearch/unified_conservative_offline_policy_foundation/report.md`
- State contract: `v4/audit/autoresearch/unified_conservative_offline_policy_foundation/unified_decision_state_v1_contract.json`
- Execution contract: `v4/audit/autoresearch/unified_conservative_offline_policy_foundation/execution_model_v1_contract.json`
- Label contract: `v4/audit/autoresearch/unified_conservative_offline_policy_foundation/action_advantage_label_v1_contract.json`
- Docs copy: `v4/docs/UNIFIED_CONSERVATIVE_OFFLINE_POLICY.md`
