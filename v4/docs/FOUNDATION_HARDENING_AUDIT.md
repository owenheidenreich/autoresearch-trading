# FOUNDATION_HARDENING_READINESS_PACKET_V1

What is this: implementation audit / foundation hardening packet
Does it change the paper-trading default: no
Paid data downloaded: no
Broker endpoint called: no
Decision: `foundation_preregistered_neural_run_complete_protocol101_challenge_blocked`

## Bottom Line

The single preregistered learned-defer neural run is complete and passed diagnostic strict replay against same-scope Protocol101 under all deterministic stress levels. Stop neural experiments here: the remaining blockers are calibrated fill evidence, untouched holdout data, live no-order parity, and formal validation controls.

## Protocol276 Versus Protocol101

| split | Protocol276 PnL | Protocol101 PnL | delta | trades | win rate | PF | acct DD | beats? |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| q3_2025 | $2,390 | $59,130 | -$56,740 | 61 | 0.443 | 1.041 | -$14,060 | no |
| q4_2025 | -$9,655 | $92,460 | -$102,115 | 11 | 0.182 | 0.232 | -$9,655 | no |
| q1_2026 | -$9,655 | $95,010 | -$104,665 | 86 | 0.430 | 0.892 | -$31,155 | no |
| march_2026 | -$9,870 | $41,790 | -$51,660 | 24 | 0.375 | 0.600 | -$15,010 | no |
| recent_2026 | $23,810 | $7,450 | $16,360 | 15 | 0.733 | 3.536 | -$7,240 | yes |

## Failure Attribution

- Integrated trade rows: `197`
- Skip rows: `16799`
- Diagnosis:
  - Account-state/affordability skipped many candidate rows; this must be explained before blaming architecture.
  - Quote-path coverage gaps remain in the integrated lifecycle replay.
  - Lifecycle integration appears to overhold many positions versus the entry-only exit path.
  - The entry policy still takes many candidates with negative oracle entry advantage.

### Skip Reasons

| reported_split | skip_reason | rows | median_entry_premium | median_account_equity |
|---|---|---|---|---|
| march_2026 | missing_contract_quotes | 478 |  |  |
| march_2026 | unaffordable_current_equity | 3268 | 3295.00 | 905.00 |
| q1_2026 | missing_contract_quotes | 1667 |  |  |
| q1_2026 | unaffordable_current_equity | 794 | 3300.00 | 1110.00 |
| q3_2025 | missing_contract_quotes | 328 |  |  |
| q3_2025 | unaffordable_current_equity | 1 | 3460.00 | 3440.00 |
| q4_2025 | missing_contract_quotes | 370 |  |  |
| q4_2025 | unaffordable_current_equity | 9849 | 3310.00 | 575.00 |
| recent_2026 | missing_contract_quotes | 44 |  |  |

### Lifecycle Modes

| reported_split | lifecycle_failure_mode | rows | pnl | lifecycle_delta_vs_entry_only_pnl | median_duration_minutes | median_giveback_from_mfe |
|---|---|---|---|---|---|---|
| march_2026 | gave_back_prior_mfe_then_lost | 9 | -19645.00 |  | 202.00 | 3840.00 |
| march_2026 | losing_exit | 6 | -5055.00 |  | 1.50 | 80.00 |
| march_2026 | profitable_or_neutral_exit | 7 | 5860.00 |  | 23.00 | 370.00 |
| march_2026 | overheld_to_forced_flat | 2 | 8970.00 |  | 279.00 | 945.00 |
| q1_2026 | gave_back_prior_mfe_then_lost | 33 | -60415.00 |  | 208.00 | 3410.00 |
| q1_2026 | losing_exit | 7 | -15615.00 |  | 134.00 | 2490.00 |
| q1_2026 | overheld_vs_entry_only_exit | 2 | -2985.00 | -2615.00 | 292.00 | 2502.50 |
| q1_2026 | profitable_or_neutral_exit | 18 | 26990.00 |  | 79.00 | 515.00 |
| q1_2026 | overheld_to_forced_flat | 26 | 42370.00 | 2200.00 | 384.00 | 910.00 |
| q3_2025 | gave_back_prior_mfe_then_lost | 23 | -39855.00 |  | 285.00 | 2880.00 |
| q3_2025 | losing_exit | 3 | -6745.00 |  | 314.00 | 2480.00 |
| q3_2025 | overheld_vs_entry_only_exit | 2 | -620.00 | -1120.00 | 262.00 | 1385.00 |
| q3_2025 | overheld_to_forced_flat | 20 | 18430.00 | 2780.00 | 384.00 | 1090.00 |
| q3_2025 | profitable_or_neutral_exit | 13 | 31180.00 |  | 379.00 | 350.00 |
| q4_2025 | gave_back_prior_mfe_then_lost | 4 | -9005.00 |  | 205.00 | 2982.50 |
| q4_2025 | losing_exit | 3 | -1740.00 |  | 54.00 | 440.00 |
| q4_2025 | overheld_vs_entry_only_exit | 1 | -1200.00 | -1810.00 | 265.00 | 3150.00 |
| q4_2025 | overheld_to_forced_flat | 3 | 2290.00 |  | 384.00 | 560.00 |
| recent_2026 | gave_back_prior_mfe_then_lost | 4 | -9390.00 |  | 317.50 | 3235.00 |
| recent_2026 | overheld_vs_entry_only_exit | 2 | 1030.00 | -1010.00 | 354.50 | 2165.00 |
| ... | 2 more rows |  |  |  |  |  |

### Entry Advantage Buckets

| reported_split | entry_advantage_bucket | rows | pnl | win_rate | median_duration_minutes |
|---|---|---|---|---|---|
| march_2026 | negative | 8 | -5150.00 | 0.25 | 14.00 |
| march_2026 | positive_gte_500 | 2 | 2030.00 | 0.50 | 293.00 |
| march_2026 | positive_lt_500 | 5 | -2220.00 | 0.60 | 25.00 |
| march_2026 | strong_negative | 9 | -4530.00 | 0.33 | 174.00 |
| q1_2026 | negative | 19 | -1030.00 | 0.47 | 137.00 |
| q1_2026 | positive_gte_500 | 2 | 2030.00 | 0.50 | 293.00 |
| q1_2026 | positive_lt_500 | 10 | 12600.00 | 0.60 | 123.00 |
| q1_2026 | strong_negative | 55 | -23255.00 | 0.38 | 315.00 |
| q3_2025 | negative | 17 | 10160.00 | 0.53 | 376.00 |
| q3_2025 | positive_lt_500 | 7 | 8290.00 | 0.57 | 308.00 |
| q3_2025 | strong_negative | 37 | -16060.00 | 0.38 | 379.00 |
| q4_2025 | negative | 3 | -1180.00 | 0.00 | 54.00 |
| q4_2025 | positive_lt_500 | 1 | -1200.00 | 0.00 | 265.00 |
| q4_2025 | strong_negative | 7 | -7275.00 | 0.29 | 177.00 |
| recent_2026 | negative | 4 | 3850.00 | 0.50 | 362.50 |
| recent_2026 | positive_gte_500 | 1 | 420.00 | 1.00 | 370.00 |
| recent_2026 | positive_lt_500 | 4 | 10550.00 | 1.00 | 359.00 |
| recent_2026 | strong_negative | 6 | 8990.00 | 0.67 | 355.00 |

## Entry/Lifecycle Bridge

- Protocol271 entry-only rows: `100`
- Integrated rows with entry-only match: `10`
- Entry-only rows not integrated: `90`
- Entry-only PnL not integrated: `$9,770`
- Common lifecycle delta versus entry-only exits: `$4,065`

## Label Alignment

- Negative `A_enter` trades taken: `165` for `-$35,480` PnL.
- Positive `A_enter` losing trades: `12` for `-$20,040` PnL.
- Median `A_enter`: `-570.00`.

Flat action labels and hold/exit labels exist, but Protocol276 shows the current integrated policy is not deployment-aligned yet.

## Recommendation Audit

| recommendation | status | evidence |
|---|---|---|
| Persist and reproduce Protocol265 artifacts | `implemented` | pass_protocol265_artifact_reproduction_exact |
| Attribute Protocol265 extension regimes | `partially_implemented` | Protocol268 exists, but many regime buckets are missing/constant. |
| Build Protocol265 no-order parity | `partially_implemented` | runtime_protocol265_no_order_parity_passed_historical_proxy |
| Build full-surface action-advantage labels | `implemented` | action_advantage_dataset_ready_for_unified_policy_training |
| Train unified action-advantage policy | `implemented_but_failed_replacement` | research_only_unified_action_advantage_policy_not_yet_paper_default |
| Build hold/exit opportunity-cost labels | `implemented` | position_state_action_advantage_dataset_ready_for_lifecycle_training |
| Integrate entry and lifecycle replay | `implemented_but_failed_replacement` | research_only_integrated_entry_lifecycle_does_not_surpass_protocol101 |
| Attribute Protocol276 integrated replay failure | `implemented` | protocol276_failure_attribution_complete_foundation_work_required |
| Freeze unified conservative offline policy direction | `implemented` | unified_conservative_offline_policy_foundation_frozen_no_model_training |
| Build unified policy trajectory foundation | `implemented_training_blocked` | unified_trajectory_foundation_ready_training_blocked_by_foundation_gates |
| Calibrate stochastic fill model | `blocked` | blocked_insufficient_fill_observations_keep_stress_replay |
| Reserve new untouched holdout | `implemented_pending_data` | untouched_holdout_reserved_pending_new_data_collection |
| Add unified neural training readiness gate | `implemented_training_blocked` | neural_training_not_ready_foundation_gates_blocked |
| Attach Protocol101 baseline actions to unified trajectory | `implemented` | protocol101_baseline_attachment_ready_for_baseline_aligned_training_scope |
| Materialize unified serial DP oracle training scope | `implemented` | unified_serial_dp_oracle_ready_for_baseline_aligned_training_scope |
| Train first conservative neural policy on frozen scope | `implemented_but_abstention_policy` | conservative_neural_policy_trained_replay_and_challenge_still_blocked |
| Run strict replay for trained conservative neural policy | `implemented_but_no_challenger_overrides` | strict_replay_complete_model_deferred_to_protocol101_flat_gate_too_conservative |
| Diagnose conservative flat-entry abstention | `implemented_retraining_formulation_fix_required` | flat_entry_gate_overconservative_zero_model_overrides |
| Run flat-calibrated conservative neural repair | `implemented_mixed_split_research_only` | conservative_neural_policy_trained_replay_and_challenge_still_blocked |
| Replay flat-calibrated conservative neural repair | `implemented_mixed_split_research_only` | strict_replay_complete_protocol101_challenge_still_blocked |
| Attribute flat-calibrated challenger overrides | `implemented_blocks_promotion` | override_attribution_mixed_split_research_only |
| Explain flat-calibrated Q1/Q3 underperformance | `implemented_opportunity_cost_failure_identified` | q1_q3_underperformance_explained_by_missed_protocol101_opportunity_cost |
| Define slot-opportunity-cost defer overlay target | `implemented_oracle_target_only` | slot_opportunity_defer_overlay_oracle_target_repairs_q1_q3_ready_for_learned_estimator |
| Materialize causal blocked-Protocol101 opportunity-cost labels | `implemented_ready_for_estimator` | slot_opportunity_cost_labels_ready_for_causal_estimator |
| Train causal blocked-Protocol101 opportunity-cost estimator | `implemented_ready_for_overlay_replay` | slot_opportunity_cost_estimator_ready_for_defer_overlay_replay |
| Replay learned slot-opportunity-cost defer overlay | `implemented_ready_for_next_preregistered_training` | learned_slot_opportunity_defer_overlay_replay_ready_for_next_preregistered_training |
| Run exactly one preregistered learned-defer neural policy | `implemented_replay_passed_diagnostic_stress` | learned_slot_opportunity_defer_overlay_replay_ready_for_next_preregistered_training |

## Foundation Checklist

| gate | status | evidence | required action |
|---|---|---|---|
| Protocol276 failure attribution | `pass` | Entry policy selected 165 negative-A_enter trades for -$35,480 PnL.; Lifecycle timing left 159 overhold/late-exit trades with -$33,275 PnL.; Actual exits left -$424,160 versus each trade's best observed path PnL.; Replay skipped 16799 rows, dominated by {'unaffordable_current_equity': 13912, 'missing_contract_quotes': 2887}. | Attribution packet complete; fixes are still required before retraining. |
| Fill model calibration | `blocked` | Fill observations: 0 / 30. | No calibrated stochastic fill model may be used until enough observed fills exist. |
| Untouched holdout reservation | `pass` | untouched_holdout_reserved_pending_new_data_collection | Current repeated-research blocks are diagnostic only, not sacred holdouts. |
| Unified neural training readiness | `pass` | next_training_paused_preregistered_run_complete_protocol101_challenge_blocked | Training readiness must either allow the one preregistered run or pause after that run is complete. |
| Challenger runtime parity | `partial` | runtime_protocol265_no_order_parity_passed_historical_proxy | Protocol265 parity exists as historical proxy; live no-order parity remains required. |
| Unified label/policy alignment | `pass` | neural_training_not_ready_foundation_gates_blocked | Protocol276 remains rejected; the new baseline-aligned serial DP oracle is the approved training formulation. |
| Conservative neural policy V1 strict replay | `partial` | strict_replay_complete_model_deferred_to_protocol101_flat_gate_too_conservative | First trained policy was replayed strictly, but it produced no challenger overrides and cannot challenge Protocol101. |
| Flat-entry gate calibration | `pass` | flat_entry_gate_produces_model_overrides_needs_replay_attribution | Flat calibration repair must create overrides without relying on ad hoc threshold loosening. |
| Flat-calibrated strict replay | `partial` | strict_replay_complete_protocol101_challenge_still_blocked | The calibrated repair produced overrides and positive same-scope total PnL, but it remains research-only. |
| Override split stability | `blocked` | At slippage 0.00, split deltas are mixed: winners=['q4_2025', 'recent_2026'], losers=['q1_2026', 'q3_2025'].; At slippage 0.10, split deltas are mixed: winners=['q4_2025', 'recent_2026'], losers=['q1_2026', 'q3_2025'].; At slippage 0.25, split deltas are mixed: winners=['q4_2025', 'recent_2026'], losers=['q1_2026', 'q3_2025'].; Zero-slippage challenger side PnL: {'C': 50720.0, 'P': 62380.0}.; Zero-slippage challenger exit-reason PnL: {'lifecycle_conservative_exit': 14949.999999999998, 'forced_flat_no_lifecycle_exit_signal': 98150.0}. | Q4/recent gains must not mask Q1/Q3 underperformance; add stability/defer constraints before more training. |
| Q1/Q3 underperformance attribution | `pass` | q1_2026: challenger PnL $35,590 minus missed Protocol101 PnL $46,360 explains delta -$10,770.; q3_2025: challenger PnL -$80 minus missed Protocol101 PnL $660 explains delta -$740.; q4_2025: challenger PnL $72,280 minus missed Protocol101 PnL $32,460 explains delta $39,820.; recent_2026: challenger PnL $5,310 minus missed Protocol101 PnL $2,130 explains delta $3,180.; q1_2026: worst blocker net -$6,220; challenger -$640 blocked 3 Protocol101 entries worth $5,580.; q3_2025: worst blocker net -$1,330; challenger -$480 blocked 3 Protocol101 entries worth $850.; Zero-slippage missed Protocol101 entry reasons: {'blocked_by_challenger_open': 174, 'replaced_by_challenger_same_event': 4}. | The mixed-split failure is explained by single-slot opportunity cost versus missed Protocol101 entries. |
| Slot opportunity-cost defer overlay | `pass` | slot_opportunity_defer_overlay_oracle_target_repairs_q1_q3_ready_for_learned_estimator | Oracle target exists; learned replay status is tracked in the estimator and learned-overlay gates. |
| Causal opportunity-cost labels | `pass` | slot_opportunity_cost_labels_ready_for_causal_estimator | Use candidate-level blocked-Protocol101 cost labels only as training targets, never as runtime features. |
| Causal opportunity-cost estimator | `pass` | slot_opportunity_cost_estimator_ready_for_defer_overlay_replay; q1_auc=0.810703363520855 | Train/calibrate the causal estimator; oracle realized blocked PnL is forbidden in live policy inputs. |
| Learned slot-opportunity overlay replay | `pass` | learned_slot_opportunity_defer_overlay_replay_ready_for_next_preregistered_training | Replay the learned estimator as a strict defer overlay before any further neural policy training. |
| Preregistered learned-defer neural run | `pass` | conservative_neural_policy_trained_replay_and_challenge_still_blocked | Exactly one preregistered neural training run is allowed after the learned defer overlay is frozen. |
| Preregistered learned-defer replay | `pass` | learned_slot_opportunity_defer_overlay_replay_ready_for_next_preregistered_training; slippage=0.0: challenger=60, delta=$66,780; slippage=0.1: challenger=60, delta=$67,580; slippage=0.25: challenger=60, delta=$68,780 | The preregistered policy must pass strict one-account replay with Q1/Q3 nonnegative stress deltas. |
| Additional neural experiments | `blocked` | Single preregistered run completed. | Pause model search after the preregistered run; remaining work is simulator, validation, parity, and fill evidence. |
| Paper default status | `pass` | No paper default change is made by this packet. | Protocol101 remains the guarded paper default; challengers remain research-only. |
| Broad paid data expansion | `blocked` | Current evidence supports staged data-value tests only after blockers close. | Do not buy broad historical data until simulator, label, parity, and holdout gates are stable. |

## Prioritized Checklist

1. **Stop neural experiments after the completed preregistered run**: The preregistered learned-defer replay passed diagnostic stress; additional model work now waits on promotion-grade foundation blockers.
2. **Collect fill evidence before stochastic fill replay**: Paper/no-order observations sufficient for calibration; until then use deterministic ask/bid plus `$0.10`/`$0.25` stress only.
3. **Upgrade challenger runtime parity to live no-order**: Full candidate breadth, freshness, Greeks, masks, account state, action schema, and latency logged live without broker orders.
4. **Collect and freeze the untouched evaluation block**: A named block that has not influenced feature, threshold, objective, architecture, sizing, exit, or model choices.
5. **Add formal validation controls**: PBO/CSCV-style false-discovery controls or an equivalent strategy-matrix audit before any better-than-Protocol101 claim.
6. **Freeze a promotion packet before scoring new data**: Fixed artifacts, fixed learned defer overlay, fixed metrics, and Protocol101 as the strict serial baseline before touching the reserved block.
7. **Keep deterministic stress as the approved offline replay assumption**: No stochastic fill replay or paper-default discussion until fill observations support calibration.
8. **Defer broad historical data purchase**: A staged data-value test only after simulator, labels, live parity, and untouched validation are stable.

## Allowed Next Work

- Stop neural/model experiments after the completed preregistered learned-defer run.
- Work the promotion-grade blockers: fill evidence, untouched holdout data, live no-order parity, and formal validation controls.
- Keep Protocol101 as paper default until the challenger passes promotion-grade fill, holdout, parity, and validation gates.
- Keep deterministic ask/bid plus `$0.10`/`$0.25` stress as the approved offline replay assumption until fill observations support calibration.
- Do not run open-ended model experiments, architecture searches, ad hoc threshold loosening, or broad paid-data acquisition while these challenge blockers remain open.

## Outputs

- Summary: `v4/audit/autoresearch/foundation_hardening_review/summary.json`
- Report: `v4/audit/autoresearch/foundation_hardening_review/report.md`
- Docs copy: `v4/docs/FOUNDATION_HARDENING_AUDIT.md`
