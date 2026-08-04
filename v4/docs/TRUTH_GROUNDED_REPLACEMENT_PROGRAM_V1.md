# FOUNDATION_TRUTH_GROUNDED_REPLACEMENT_PROGRAM_V1

What is this: foundation / truth-grounded Protocol101 replacement research program
Does it change the paper-trading default: no
Paper default baseline: `PAPER_DEFAULT_PROTOCOL101`
Paid data downloaded: no
Broker endpoint called: no
Model training: no
Untouched holdout scored: no
Decision: `truth_grounded_replacement_program_created_training_and_replacement_blocked`

## Bottom Line

Do not try another generic model. Protocol101 stays the control strategy while the replacement program converts trader ideas into named, falsifiable playbooks. A replacement model is explicitly blocked until the required diagnostics, data gates, execution evidence, validation controls, and untouched scoring path exist.

## Seeded Strategy Hypotheses

| playbook | track | status | next action |
|---|---|---|---|
| `PROTOCOL101_HARD_STOP_AVOIDANCE_GATE_V1` | A_protocol101_forensics | `diagnostic_only` | Manually classify current hard_stop_autopsy.csv rows into pre-entry avoidable, path-management avoidable, execution artifact, or unavoidable. |
| `PROTOCOL101_CONFIRMED_MFE_RUNNER_OVERLAY_V1` | A_protocol101_forensics | `blocked_missing_post_exit_paths` | Build post-exit path attachment for Protocol101 trades before any runner model. |
| `PROTOCOL101_INTERNAL_SLOT_DEFER_V1` | A_protocol101_forensics | `blocked_missing_counterfactual_flat_actions` | Build counterfactual-flat Protocol101 action table. |
| `PROTOCOL101_MISSED_WINNER_ABSTENTION_EXPANSION_V1` | B_new_alpha_playbooks | `diagnostic_only` | Join full_surface_action_advantage.parquet to Protocol101 baseline actions and build matched controls. |
| `SIDE_SPECIFIC_PUT_REGIME_PLAYBOOK_V1` | B_new_alpha_playbooks | `diagnostic_only` | Build call/put asymmetry report by time, premium, moneyness, exit reason, MFE/MAE, and delay stress. |
| `SIDE_SPECIFIC_CALL_CONTINUATION_PLAYBOOK_V1` | B_new_alpha_playbooks | `diagnostic_only` | Analyze call-only winners/losers by trend and exit archetype. |
| `LATE_AFTERNOON_CONTINUATION_PLAYBOOK_V1` | B_new_alpha_playbooks | `diagnostic_only` | Separate post-open and late-afternoon archetypes with delay/fill stress. |
| `LOWER_PREMIUM_CONVEX_ADDON_PLAYBOOK_V1` | B_new_alpha_playbooks | `high_risk_diagnostic_only` | Use Protocol248/270 evidence to separate valid lower-premium regimes from hindsight bait. |
| `PLAYBOOK_AWARE_REPLACEMENT_POLICY_V1` | C_replacement_ml_stack | `blocked_until_track_a_b_complete` | Do not train until required playbook diagnostics and validation gates pass. |

## Protocol101 Weakness Matrix

| priority | weakness | status | next diagnostic |
|---:|---|---|---|
| 1 | execution_realism | `blocked_missing_fill_and_latency_distribution` | PROTOCOL101_EXECUTION_REALISM_BY_ARCHETYPE_V1 |
| 2 | hard_stop_losses | `ready_for_manual_classification` | PROTOCOL101_HARD_STOP_AUTOPSY_V1 |
| 3 | losing_days | `ready_for_manual_classification` | PROTOCOL101_LOSING_DAY_AUTOPSY_V1 |
| 4 | runner_giveback | `blocked_missing_post_exit_path_attachment` | PROTOCOL101_CONFIRMED_MFE_RUNNER_AUDIT_V1 |
| 5 | score_reliability | `proxy_only_rejected_candidates_missing` | PROTOCOL101_SCORE_RELIABILITY_V1 |
| 6 | internal_slot_cost | `blocked_missing_counterfactual_flat_protocol101_actions` | PROTOCOL101_INTERNAL_SLOT_COST_V1 |
| 7 | missed_winner_abstention | `partially_ready_requires_matched_controls` | PROTOCOL101_MISSED_WINNER_ABSTENTION_AUDIT_V1 |
| 8 | side_asymmetry | `ready_for_side_specific_audit` | PROTOCOL101_SIDE_SPECIFIC_STRATEGY_AUDIT_V1 |
| 9 | validation_overfit | `blocked_missing_strategy_matrix_pbo_cscv` | FORMAL_STRATEGY_MATRIX_PBO_CSCV_V1 |

## Research Data Layer

| data domain | status | next data action |
|---|---|---|
| protocol101_selected_trades | `available` | Keep as current Protocol101 trade-atlas source of truth. |
| full_surface_candidates | `available_for_diagnostics` | Join to Protocol101 baseline actions and build matched rejected-candidate controls. |
| protocol101_baseline_actions | `available_but_deployed_state_only` | Build counterfactual-flat Protocol101 action replay. |
| timing_delay_stress | `available_historical` | Build latency-distribution replay once live/no-order timing logs are sufficient. |
| fill_observations | `blocked_insufficient_fill_observations` | Collect stratified no-order/paper fill evidence by side, premium, spread, quote age, and time bucket. |
| untouched_evaluation_block | `reserved_pending_collection` | Do not score until candidate is frozen and all promotion blockers are closed. |
| formal_strategy_matrix | `blocked_missing_pbo_cscv_matrix` | Build strategy matrix before any replacement packet. |
| trader_research_inputs | `available_unstructured` | Create playbook notes that map each idea to entry, invalidation, exit, risk, and falsification fields. |

## Stage Gates

| gate | status | blocks |
|---|---|---|
| G1_named_playbook | `pass_registry_seeded` | anonymous_model_experiments |
| G2_protocol101_forensics | `partial_packet_exists_more_diagnostics_required` | replacement_training |
| G3_no_future_inputs | `required_for_future_model` | model_training |
| G4_execution_realism | `blocked_zero_fill_observations` | promotion_and_replacement_claims |
| G5_strategy_matrix | `blocked_missing_strategy_matrix` | untouched_scoring |
| G6_untouched_holdout | `blocked_pending_new_data_collection` | paper_default_replacement |
| G7_protocol101_default | `pass_default_preserved` | unapproved_paper_default_change |

## Outputs

- Summary: `v4/audit/autoresearch/truth_grounded_replacement_program_v1/summary.json`
- Strategy hypothesis registry: `v4/audit/autoresearch/truth_grounded_replacement_program_v1/strategy_hypothesis_registry.csv`
- Protocol101 weakness matrix: `v4/audit/autoresearch/truth_grounded_replacement_program_v1/protocol101_weakness_matrix.csv`
- Research data layer: `v4/audit/autoresearch/truth_grounded_replacement_program_v1/research_data_layer.csv`
- Stage-gate rulebook: `v4/audit/autoresearch/truth_grounded_replacement_program_v1/stage_gate_rulebook.csv`
- Replacement candidate protocol spec: `v4/audit/autoresearch/truth_grounded_replacement_program_v1/replacement_candidate_protocol_spec.json`
