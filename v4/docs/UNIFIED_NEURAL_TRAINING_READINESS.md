# FOUNDATION_UNIFIED_NEURAL_TRAINING_READINESS_V1

What is this: foundation gate / unified conservative policy neural-training readiness
Does it change the paper-trading default: no
Paper default baseline: `PAPER_DEFAULT_PROTOCOL101`
Paid data downloaded: no
Broker endpoint called: no
Model training: no
Training decision: `neural_training_not_ready_foundation_gates_blocked`
Next training decision: `next_training_paused_preregistered_run_complete_protocol101_challenge_blocked`
Protocol101 challenge decision: `protocol101_challenge_not_ready_foundation_gates_blocked`

## Gate Summary

- Training blockers: `['Additional neural training pause']`
- Protocol101 challenge blockers: `['Untouched holdout data availability', 'Live no-order full-action parity']`

| gate | status | blocks training | blocks challenge | evidence | required action |
|---|---|---|---|---|---|
| Unified state/action/execution contract | `pass` | True | True | unified_conservative_offline_policy_foundation_frozen_no_model_training | Freeze UnifiedDecisionStateV1 / ExecutionModelV1 / ActionAdvantageLabelV1 before training. |
| Trajectory dataset foundation | `pass` | True | True | flat_rows=3509721, holding_rows=4467517, feature_status=pass | Materialize causal flat and holding state rows with no future/path columns in model inputs. |
| Full serial wait/enter/hold/exit DP oracle | `pass` | True | True | unified_serial_dp_oracle_ready_for_baseline_aligned_training_scope | Compute wait/enter/hold/exit action values on the exact frozen simulator and same account trajectory before training. |
| Protocol101 baseline action attachment | `pass` | True | True | protocol101_baseline_attachment_ready_for_baseline_aligned_training_scope | Use the baseline-aligned training scope, and add Protocol101 action/candidate/lifecycle fields for any newly admitted splits. |
| Protocol276 failure attribution | `pass` | False | True | protocol276_failure_attribution_complete_foundation_work_required | Keep Protocol276 abandoned as a candidate and use its attribution to shape the new labels. |
| Causal slot opportunity-cost labels | `pass` | True | True | slot_opportunity_cost_labels_ready_for_causal_estimator, rows=1919518 | Materialize candidate-level blocked-Protocol101 opportunity-cost labels without future/runtime-forbidden feature leakage. |
| Causal slot opportunity-cost estimator | `pass` | True | True | slot_opportunity_cost_estimator_ready_for_defer_overlay_replay | Train/calibrate the slot opportunity-cost defer estimator before another neural policy run. |
| Learned slot opportunity-cost defer replay | `pass` | True | True | learned_slot_opportunity_defer_overlay_replay_ready_for_next_preregistered_training | Replay the learned estimator as a strict defer overlay and require Q1/Q3 nonnegative stress deltas before another neural policy run. |
| Preregistered learned-defer neural policy run | `pass` | True | True | conservative_neural_policy_trained_replay_and_challenge_still_blocked | Run exactly one preregistered neural policy after the learned defer overlay is frozen. |
| Preregistered learned-defer neural replay | `pass` | True | True | learned_slot_opportunity_defer_overlay_replay_ready_for_next_preregistered_training | Replay the preregistered neural policy with the frozen learned defer overlay under all deterministic stress levels. |
| Additional neural training pause | `blocked` | True | False | single preregistered run complete | Do not run additional neural experiments until fill, untouched holdout data, live parity, and formal validation gates are addressed. |
| Deterministic execution replay | `pass` | False | True | ExecutionModelV1 uses ask-entry, bid-exit, one account, one contract, affordability, forced flat, and stress-ready slippage. | Keep deterministic ask/bid plus $0.10/$0.25 per-side stress as the approved replay assumption until fills exist. |
| Execution realism fill evidence | `pass` | False | True | fill_status=execution_truth_packet_ready_for_conservative_fill_stress, observations=10/30 | Collect bounded paper fill/cancel/timeout observations; keep conservative stress replay unless a larger calibrated fill model is separately validated. |
| Untouched holdout reservation | `pass` | True | True | untouched_holdout_reserved_pending_new_data_collection | Reserve a new unseen final evaluation block before model selection. |
| Untouched holdout data availability | `blocked` | False | True | data_status=pending_new_data_collection; decision=untouched_holdout_data_pending_collection | Collect/freeze the reserved block before any final better-than-Protocol101 claim. |
| Live no-order full-action parity | `partial` | False | True | live_no_order_full_action_parity_waiting_for_open_market_session; historical_proxy=True | Upgrade historical proxy parity to live no-order parity for full candidate breadth, freshness, Greeks, masks, account state, and latency. |
| Formal validation controls | `pass` | False | True | formal_validation_controls_ready; comparable_strategies=21; cscv_folds=6 | Build the strategy-matrix/PBO-CSCV or equivalent false-discovery control before promotion-grade claims. |

## Next Allowed Work

1. Stop neural training after the completed preregistered run; focus only on Protocol101 challenge blockers.
2. Build a live no-order full-action parity run before paper-default discussion.
3. Do not score final claims until the reserved unseen block is collected/frozen.

## Latest Training Artifacts

- Trained policy decision: `conservative_neural_policy_trained_replay_and_challenge_still_blocked`
- Strict replay decision: `strict_replay_complete_model_deferred_to_protocol101_flat_gate_too_conservative`
- Flat-gate decision: `flat_entry_gate_overconservative_zero_model_overrides`
- Flat-calibrated policy decision: `conservative_neural_policy_trained_replay_and_challenge_still_blocked`
- Flat-calibrated replay decision: `strict_replay_complete_protocol101_challenge_still_blocked`
- Flat-calibrated attribution decision: `override_attribution_mixed_split_research_only`
- Q1/Q3 underperformance decision: `q1_q3_underperformance_explained_by_missed_protocol101_opportunity_cost`
- Slot opportunity overlay decision: `slot_opportunity_defer_overlay_oracle_target_repairs_q1_q3_ready_for_learned_estimator`
- Slot opportunity labels decision: `slot_opportunity_cost_labels_ready_for_causal_estimator` over `1919518` rows
- Slot opportunity estimator decision: `slot_opportunity_cost_estimator_ready_for_defer_overlay_replay` (Q1 AUC `0.810703363520855`)
- Learned slot overlay replay decision: `learned_slot_opportunity_defer_overlay_replay_ready_for_next_preregistered_training`
- Preregistered learned-defer policy decision: `conservative_neural_policy_trained_replay_and_challenge_still_blocked`
- Preregistered learned-defer replay decision: `learned_slot_opportunity_defer_overlay_replay_ready_for_next_preregistered_training`
- Preregistered learned-defer replay totals: `[{'challenger_entries': 60, 'delta_vs_protocol101_same_scope': 66780.0, 'protocol101_defer_entries': 679, 'protocol101_same_scope_pnl': 235740.0, 'total_pnl': 302520.0, 'trades': 739}, {'challenger_entries': 60, 'delta_vs_protocol101_same_scope': 67580.0, 'protocol101_defer_entries': 679, 'protocol101_same_scope_pnl': 220160.0, 'total_pnl': 287740.0, 'trades': 739}, {'challenger_entries': 60, 'delta_vs_protocol101_same_scope': 68780.0, 'protocol101_defer_entries': 679, 'protocol101_same_scope_pnl': 196790.0, 'total_pnl': 265570.0, 'trades': 739}]`
- Flat A_enter positives are rare: 2.9349% of 1,919,518 full-surface candidate rows.
- Gate component pass counts: advantage=0, positive_probability=138, tail=1,752,701, all=0.
- The advantage regression head is the binding bottleneck; target scaling or loss balance must be fixed before retraining/replay.
- At slippage 0.00, split deltas are mixed: winners=['q4_2025', 'recent_2026'], losers=['q1_2026', 'q3_2025'].
- At slippage 0.10, split deltas are mixed: winners=['q4_2025', 'recent_2026'], losers=['q1_2026', 'q3_2025'].
- At slippage 0.25, split deltas are mixed: winners=['q4_2025', 'recent_2026'], losers=['q1_2026', 'q3_2025'].
- Zero-slippage challenger side PnL: {'C': 50720.0, 'P': 62380.0}.
- Zero-slippage challenger exit-reason PnL: {'lifecycle_conservative_exit': 14949.999999999998, 'forced_flat_no_lifecycle_exit_signal': 98150.0}.
- q1_2026: challenger PnL $35,590 minus missed Protocol101 PnL $46,360 explains delta -$10,770.
- q3_2025: challenger PnL -$80 minus missed Protocol101 PnL $660 explains delta -$740.
- q4_2025: challenger PnL $72,280 minus missed Protocol101 PnL $32,460 explains delta $39,820.
- recent_2026: challenger PnL $5,310 minus missed Protocol101 PnL $2,130 explains delta $3,180.
- q1_2026: worst blocker net -$6,220; challenger -$640 blocked 3 Protocol101 entries worth $5,580.
- q3_2025: worst blocker net -$1,330; challenger -$480 blocked 3 Protocol101 entries worth $850.
- Zero-slippage missed Protocol101 entry reasons: {'blocked_by_challenger_open': 174, 'replaced_by_challenger_same_event': 4}.
- q1_2026 @ 0.00: oracle overlay keeps 14/45 overrides and changes delta from -$12,880 to $32,310.
- q3_2025 @ 0.00: oracle overlay keeps 1/5 overrides and changes delta from -$1,760 to $1,390.
- q4_2025 @ 0.00: oracle overlay keeps 9/25 overrides and changes delta from $38,950 to $59,400.
- recent_2026 @ 0.00: oracle overlay keeps 1/3 overrides and changes delta from $2,550 to $4,950.
- q1_2026 @ 0.10: oracle overlay keeps 14/45 overrides and changes delta from -$12,040 to $32,710.
- q3_2025 @ 0.10: oracle overlay keeps 1/5 overrides and changes delta from -$1,740 to $1,410.
- q4_2025 @ 0.10: oracle overlay keeps 9/25 overrides and changes delta from $39,570 to $59,640.
- recent_2026 @ 0.10: oracle overlay keeps 1/3 overrides and changes delta from $2,650 to $5,030.
- q1_2026 @ 0.25: oracle overlay keeps 14/45 overrides and changes delta from -$10,780 to $33,310.
- q3_2025 @ 0.25: oracle overlay keeps 1/5 overrides and changes delta from -$1,710 to $1,440.
- q4_2025 @ 0.25: oracle overlay keeps 9/25 overrides and changes delta from $40,500 to $60,000.
- recent_2026 @ 0.25: oracle overlay keeps 1/3 overrides and changes delta from $2,800 to $5,150.

## Source Artifacts

- foundation: `v4/audit/autoresearch/unified_conservative_offline_policy_foundation/summary.json`
- trajectory: `v4/audit/autoresearch/unified_policy_trajectory_foundation/summary.json`
- protocol276_attribution: `v4/audit/autoresearch/protocol276_integrated_lifecycle_failure_attribution/summary.json`
- fill: `v4/audit/autoresearch/v4_aplus_hypothesis_272_fill_model_readiness/summary.json`
- overfit: `v4/audit/autoresearch/v4_aplus_hypothesis_273_model_selection_overfit_risk/summary.json`
- formal_validation: `v4/audit/autoresearch/formal_validation_governance/summary.json`
- parity: `v4/audit/autoresearch/v4_aplus_hypothesis_269_protocol265_no_order_runtime_parity/summary.json`
- live_parity_readiness: `v4/audit/autoresearch/live_no_order_full_action_parity_readiness/summary.json`
- holdout: `v4/audit/autoresearch/unified_untouched_holdout_reservation/summary.json`
- holdout_availability: `v4/audit/autoresearch/untouched_holdout_availability/summary.json`
- baseline_attachment: `v4/audit/autoresearch/unified_protocol101_baseline_attachment/summary.json`
- serial_dp_oracle: `v4/audit/autoresearch/unified_serial_dp_oracle/summary.json`
- trained_policy: `v4/audit/autoresearch/unified_conservative_neural_policy_v1/summary.json`
- strict_replay: `v4/audit/autoresearch/unified_conservative_neural_policy_strict_replay_v1/summary.json`
- flat_gate_diagnostic: `v4/audit/autoresearch/unified_conservative_flat_gate_diagnostic/summary.json`
- flat_calibrated_policy: `v4/audit/autoresearch/unified_conservative_neural_policy_flat_calibrated_v1/summary.json`
- flat_calibrated_replay: `v4/audit/autoresearch/unified_conservative_neural_policy_flat_calibrated_strict_replay_v1/summary.json`
- flat_calibrated_attribution: `v4/audit/autoresearch/unified_conservative_neural_policy_flat_calibrated_override_attribution_v1/summary.json`
- q1_q3_underperformance: `v4/audit/autoresearch/unified_conservative_q1_q3_underperformance_attribution_v1/summary.json`
- slot_opportunity_overlay: `v4/audit/autoresearch/unified_slot_opportunity_defer_overlay_foundation/summary.json`
- slot_opportunity_labels: `v4/audit/autoresearch/unified_slot_opportunity_cost_label_dataset/summary.json`
- slot_opportunity_estimator: `v4/audit/autoresearch/unified_slot_opportunity_cost_estimator/summary.json`
- slot_opportunity_learned_overlay_replay: `v4/audit/autoresearch/unified_slot_opportunity_learned_defer_overlay_replay_relaxed_m0_w025_e3/summary.json`
- preregistered_learned_defer_policy: `v4/audit/autoresearch/unified_conservative_neural_policy_learned_defer_preregistered_v1/summary.json`
- preregistered_learned_defer_replay: `v4/audit/autoresearch/unified_conservative_neural_policy_learned_defer_preregistered_replay_v1/summary.json`

## Outputs

- Summary: `v4/audit/autoresearch/unified_neural_training_readiness/summary.json`
- Report: `v4/audit/autoresearch/unified_neural_training_readiness/report.md`
- Docs copy: `v4/docs/UNIFIED_NEURAL_TRAINING_READINESS.md`
