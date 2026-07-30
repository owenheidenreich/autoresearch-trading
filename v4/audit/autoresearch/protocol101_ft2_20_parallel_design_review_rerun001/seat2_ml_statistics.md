# Protocol101 ML / Statistics Adversarial Re-Review — Rerun 001

## ISOLATION AND PRODUCER DISCLOSURE

I performed this review as the isolated ML/statistics seat. Isolation was perfect under the stated definition: I did not communicate with another reviewer, did not inspect another reviewer's file, and did not inspect any rerun output. Of the original FT2-20 packet I opened only the original `seat2_ml_statistics.md`. The three repair crosswalks were used only as navigation indexes; no crosswalk assertion is treated as evidence. Every verdict below is based on the repaired contract text or other allowlisted primary evidence.

Producer disclosure, verbatim:

> receipt schema field naming is inconsistent across packets (authority_sha256 vs product_contract_hash); values are verified correct; classify as the seats see fit.

Classification: **MINOR** schema/documentation inconsistency. In the inspected receipts the disclosed field-name variation does not create an observed product-identity ambiguity, but future packets should standardize one field name.

## A. FINDINGS VERIFICATION

### 1. Original **BLOCKING — Future `target_valid` directly controls the composer**

**Verdict: ANSWERED.**

Exact repaired citations:

- `protocol101_ft2_08_data_tensor_label_contract/tensor_schema.json:/flat_state_masks`
- `protocol101_ft2_08_data_tensor_label_contract/label_join_spec.json:/join_contract`
- `protocol101_ft2_10_entry_science_contract/composer_spec.json:/complete_ladder_and_safety`
- `protocol101_ft2_10_entry_science_contract/composer_spec.json:/uncertainty_wait`
- `protocol101_ft2_10_entry_science_contract/composer_spec.json:/target_and_forecast_independence`
- `protocol101_ft2_10_entry_science_contract/objective_spec.json:/p5_under_cap_algorithm/target_and_forecast_independence`

The repaired composer takes decision-time state, action masks, calibrated forecasts, and uncertainty/regret outputs. It explicitly prohibits future target availability or `target_valid` from composer input. Label availability remains a label/join concern. The new wording therefore removes the original direct target-to-action path. This verdict does not bless all action masks: fresh finding B1 identifies a separate future-quote leak in D48/D49.

### 2. Original **BLOCKING — Census consumes the first fold's embargo session**

**Verdict: ANSWERED.**

Exact repaired citations:

- `protocol101_ft2_08_data_tensor_label_contract/fold_roles.json:/census/embargo_intersection_count`
- `protocol101_ft2_08_data_tensor_label_contract/fold_roles.json:/census/excluded_embargo_sessions`
- `protocol101_ft2_08_data_tensor_label_contract/fold_roles.json:/role_semantics/embargo`
- `protocol101_ft2_08_data_tensor_label_contract/fold_roles.json:/outer_test_firewall`
- `protocol101_ft2_04_path_label_freeze/intersection_proof.json`
- `protocol101_ft2_04_path_label_freeze/census_sessions.json`

The repaired manifest excludes all five embargo sessions and records zero census/embargo intersection, rather than treating the first embargo session as census-design data. It separately requires zero outer-test and protected-holdout intersection. The collision is mechanically removed in the supplied role manifests.

### 3. Original **BLOCKING — `DeltaQ` replaces the signed incremental-profit gate**

**Verdict: ANSWERED.**

Exact repaired citations:

- `protocol101_ft2_10_entry_science_contract/objective_spec.json:/entry_component_freeze`
- `protocol101_ft2_10_entry_science_contract/objective_spec.json:/entry_component_freeze/DeltaQ_role`
- `protocol101_ft2_10_entry_science_contract/objective_spec.json:/entry_component_freeze/promotable_dollar_claim_owner`
- `protocol101_ft2_11_evidence_statistics_contract/evidence_standard.json:/two_layer_decision_structure`
- `protocol101_ft2_11_evidence_statistics_contract/evidence_standard.json:/entry_component_freeze_layer`
- `protocol101_ft2_11_evidence_statistics_contract/evidence_standard.json:/combined_trader_promotable_layer`
- `PROTOCOL101_D1_NEGATIVE_CONTROL_AND_INCREMENTAL_EDGE_AMENDMENT_2026_07_28.md`, signed D1 incremental-edge requirement

The repair now makes `DeltaQ` diagnostic only. Component freeze requires all quality, safety, and serial-dollar no-harm channels and explicitly cannot claim incremental edge or promotion. The combined trader's promotable layer separately owns the D1 dollar comparison against exact P5 and matched random. That restores the signed distinction between “component is usable” and “combined policy has incremental dollar edge.”

### 4. Original **BLOCKING — Mandatory action-conditioned calibration gate is absent and unimplementable**

**Verdict: ANSWERED.**

Exact repaired citations:

- `protocol101_ft2_10_entry_science_contract/forecast_heads.json:/selected_contract_regret`
- `protocol101_ft2_10_entry_science_contract/forecast_heads.json:/wait_regret`
- `protocol101_ft2_10_entry_science_contract/calibration_spec.json:/final_outer_calibration_protocol`
- `protocol101_ft2_10_entry_science_contract/calibration_spec.json:/action_conditioned_gate`
- `protocol101_ft2_10_entry_science_contract/calibration_spec.json:/action_conditioned_gate/selected_contract_regret`
- `protocol101_ft2_10_entry_science_contract/calibration_spec.json:/action_conditioned_gate/wait_regret`
- `protocol101_ft2_10_entry_science_contract/calibration_spec.json:/mandatory_checklist`

The repair defines both selected-contract and WAIT regret heads, their targets/losses, a disjoint final calibration slice, same-frozen-model prediction, and mandatory action-conditioned calibration/coverage checks. That is enough to implement the original requested calibration gate. Fresh finding B5 is narrower: passing calibration does not itself make the selected-contract regret magnitude constrain the action.

### 5. Original **BLOCKING — The `$24.45` MDE is unsound in estimand, dependence, and trade count**

**Verdict: PARTIALLY ANSWERED.**

Exact repaired citations:

- `protocol101_ft2_11_evidence_statistics_contract/mde_spec.json:/estimand`
- `protocol101_ft2_11_evidence_statistics_contract/mde_spec.json:/analytic_mde`
- `protocol101_ft2_11_evidence_statistics_contract/mde_spec.json:/effective_sample_size`
- `protocol101_ft2_11_evidence_statistics_contract/mde_spec.json:/trade_count_ceiling`
- `protocol101_ft2_11_evidence_statistics_contract/mde_spec.json:/block_power`
- `protocol101_ft2_11_evidence_statistics_contract/mde_spec.json:/reported_mde`
- `protocol101_ft2_05_opportunity_census/minimum_detectable_improvement.csv`
- `protocol101_ft2_05_opportunity_census/session_variance_components.csv`

The repair replaces the old trade-IID calculation with a session-dollar estimand, a session-count/ESS calculation, a census-derived trade-count ceiling, and a “larger of analytic and block-power” rule. Those changes answer the original estimand and pseudoreplication objections. The answer is incomplete because the required block-power authority inherits the undefined bootstrap replicate standard error in `bootstrap_spec.json:/studentized_maxT/centered_null_resample`; see B4. Until `SE_h_star` is defined, the larger-of MDE is not fully reproducible.

### 6. Original **BLOCKING — Terminal outcomes are mechanically undecidable**

**Verdict: PARTIALLY ANSWERED.**

Exact repaired citations:

- `protocol101_ft2_11_evidence_statistics_contract/terminal_decision_spec.json:/adjusted_bound_authority`
- `protocol101_ft2_11_evidence_statistics_contract/terminal_decision_spec.json:/precedence`
- `protocol101_ft2_11_evidence_statistics_contract/terminal_decision_spec.json:/integrity_and_sufficiency_overrides`
- `protocol101_ft2_11_evidence_statistics_contract/evidence_standard.json:/terminal_decision_table`
- `protocol101_ft2_11_evidence_statistics_contract/evidence_standard.json:/statistic_specific_sufficiency`
- `protocol101_ft2_11_evidence_statistics_contract/worked_terminal_examples.json`

The repair now provides adjusted upper and lower bounds, precedence, D1-specific pass/fail cases, statistic-specific sufficiency rules, integrity overrides, and worked cases including the fifth-fold attribution. This resolves most of the prior ambiguity. It remains only partial because the declared sole adjusted-bound authority depends on an undefined `SE_h_star`; therefore the terminal table cannot yet be executed uniquely from the frozen text. See B4.

### 7. Original **MATERIAL — Empirical-CDF transform is under-specified and not nested**

**Verdict: ANSWERED.**

Exact repaired citations:

- `protocol101_ft2_10_entry_science_contract/objective_spec.json:/nested_empirical_cdf/reference_population`
- `protocol101_ft2_10_entry_science_contract/objective_spec.json:/nested_empirical_cdf/premium_band_assignment`
- `protocol101_ft2_10_entry_science_contract/objective_spec.json:/nested_empirical_cdf/market_phase_assignment`
- `protocol101_ft2_10_entry_science_contract/objective_spec.json:/nested_empirical_cdf/weighting`
- `protocol101_ft2_10_entry_science_contract/objective_spec.json:/nested_empirical_cdf/ties_and_interpolation`
- `protocol101_ft2_10_entry_science_contract/objective_spec.json:/nested_empirical_cdf/minimum_reference`
- `protocol101_ft2_10_entry_science_contract/objective_spec.json:/nested_empirical_cdf/fallback_order`
- `protocol101_ft2_10_entry_science_contract/objective_spec.json:/nested_empirical_cdf/no_cross_band_fallback`
- `protocol101_ft2_10_entry_science_contract/objective_spec.json:/nested_empirical_cdf/nesting`

The repaired contract freezes the causal time-\(t\) premium-band and market-phase population, equal-session/within-session row weights, exact tie and interpolation behavior, endpoints, minimum support, same-band fallback, fail-closed behavior, and separate inner-search/final-outer reference scopes. This is now a nested, reproducible transform rather than a loosely named CDF.

### 8. Original **MATERIAL — OOF conformal residuals are reused for refit**

**Verdict: ANSWERED.**

Exact repaired citations:

- `protocol101_ft2_10_entry_science_contract/calibration_spec.json:/final_outer_calibration_protocol`
- `protocol101_ft2_10_entry_science_contract/calibration_spec.json:/final_outer_calibration_protocol/disjointness`
- `protocol101_ft2_10_entry_science_contract/calibration_spec.json:/final_outer_calibration_protocol/model_freeze`
- `protocol101_ft2_10_entry_science_contract/calibration_spec.json:/final_outer_calibration_protocol/no_post_calibration_refit`
- `protocol101_ft2_08_data_tensor_label_contract/fold_roles.json:/fold_governance/required_embargo_sessions`

The final outer model is frozen before a disjoint, embargo-separated calibration slice is scored, and the contract prohibits any post-calibration refit. The inner OOF machinery is no longer reused as if it were independent calibration for the refit final model.

### 9. Original **MATERIAL — Dependence machinery lacks a validity condition**

**Verdict: PARTIALLY ANSWERED.**

Exact repaired citations:

- `protocol101_ft2_11_evidence_statistics_contract/bootstrap_spec.json:/block_length_selection`
- `protocol101_ft2_11_evidence_statistics_contract/bootstrap_spec.json:/block_length_selection/candidate_block_lengths_sessions`
- `protocol101_ft2_11_evidence_statistics_contract/bootstrap_spec.json:/block_length_selection/geyer_initial_positive_sequence`
- `protocol101_ft2_11_evidence_statistics_contract/bootstrap_spec.json:/synthetic_AR_coverage_demonstration`
- `protocol101_ft2_11_evidence_statistics_contract/bootstrap_spec.json:/phase_F_shadow_intervals`
- `protocol101_ft2_11_evidence_statistics_contract/shadow_sufficiency_spec.json`

The repair makes block selection noncircular with fixed 3/5/8-session candidates and a Geyer initial-positive-sequence rule, supplies AR coverage demonstrations, and keeps session-cluster shadow intervals distinct from hard sequential decisions. That is a substantial answer. It is incomplete for two reasons: the hard interval still uses the undefined replicate standard error in B4, and the demonstrated synthetic fold shapes use 45 sessions while `phase_F_shadow_intervals/minimum_complete_sessions` permits hard reporting at 15 sessions without a corresponding small-\(n\) coverage check. See B6.

### 10. Original **MATERIAL — Multiplicity omits selectable controls**

**Verdict: PARTIALLY ANSWERED.**

Exact repaired citations:

- `protocol101_ft2_10_entry_science_contract/controls_spec.json:/frozen_nonselectable_random_controls`
- `protocol101_ft2_10_entry_science_contract/controls_spec.json:/frozen_nonselectable_random_controls/attempt_cap`
- `protocol101_ft2_10_entry_science_contract/controls_spec.json:/frozen_nonselectable_random_controls/random_key`
- `protocol101_ft2_11_evidence_statistics_contract/multiplicity_spec.json:/family_definition`
- `protocol101_ft2_11_evidence_statistics_contract/multiplicity_spec.json:/all_attempts_and_aggregate`
- `protocol101_ft2_11_evidence_statistics_contract/multiplicity_spec.json:/frozen_nonselectable_matched_random_algorithm`
- `protocol101_ft2_11_evidence_statistics_contract/multiplicity_spec.json:/frozen_nonselectable_matched_random_algorithm/candidate_specific_seed_namespace`
- `protocol101_ft2_11_evidence_statistics_contract/multiplicity_spec.json:/frozen_nonselectable_matched_random_algorithm/attempt_random_key`

The repair caps attempts, freezes seed IDs, prohibits redraw, includes all attempts plus an aggregate, enumerates evidence channels, and places the family into adjusted inference. That answers the original omission in concept. It is only partial because FT2-10 and FT2-11 freeze different random-key algorithms for the supposedly same controls; see B3. A multiplicity family whose comparator schedules are not uniquely identified is not mechanically frozen.

### 11. Original **MATERIAL — No-bid censoring is MNAR and rewards illiquidity**

**Verdict: ANSWERED.**

Exact repaired citations:

- `PROTOCOL101_FULL_TRADER_GRAPH_V2_CONSOLIDATED_AUTHORITY_2026_07_28.md`, section 2.4 “Decision timing, entry timing, and no-bid rule”
- `protocol101_ft2_08_data_tensor_label_contract/label_join_spec.json:/horizon_join`
- `protocol101_ft2_10_entry_science_contract/calibration_spec.json:/no_bid_sensitivity`
- `protocol101_ft2_11_evidence_statistics_contract/evidence_standard.json:/no_bid_primary_and_sensitivity`
- `protocol101_ft2_11_evidence_statistics_contract/evidence_standard.json:/no_bid_primary_and_sensitivity/selected_identity_rule`
- `protocol101_ft2_11_evidence_statistics_contract/evidence_standard.json:/no_bid_primary_and_sensitivity/serious_identity_ordering`
- `protocol101_ft2_05_opportunity_census/mnar_sensitivity.csv`

The full-loss no-bid rule is primary, while a no-bid-excluded result is sensitivity-only. The exact selected identity must keep the same terminal decision, and serious identities must maintain Spearman ordering at the frozen threshold. This prevents favorable omission from becoming the rewarded primary target. The census sensitivity correlation is planning evidence only, not a substitute for the prospective identity-level terminal requirement.

## B. FRESH SCAN

### B1. **BLOCKING — D48/D49 decision masks use the future \(t+1\) fill quote**

Exact citations:

- `protocol101_ft2_08_data_tensor_label_contract/account_state_ledger_spec.json:/d48/prospective_entry_cost_cents`
- `protocol101_ft2_08_data_tensor_label_contract/account_state_ledger_spec.json:/d48/mask_true`
- `protocol101_ft2_08_data_tensor_label_contract/account_state_ledger_spec.json:/d49/mask_true`
- `protocol101_ft2_08_data_tensor_label_contract/account_state_ledger_spec.json:/per_decision_audit_fields`
- `protocol101_ft2_08_data_tensor_label_contract/tensor_schema.json:/flat_state_masks`
- `protocol101_ft2_10_entry_science_contract/composer_spec.json:/complete_ladder_and_safety/per_contract_requirements`
- `protocol101_ft2_10_entry_science_contract/objective_spec.json:/serial_primary_population/entry_commit_and_fill/fill_time_rechecks`
- `protocol101_ft2_10_entry_science_contract/objective_spec.json:/p5_under_cap_algorithm/inputs`
- `protocol101_ft2_10_entry_science_contract/objective_spec.json:/p5_under_cap_algorithm/target_and_forecast_independence`
- `protocol101_ft2_04_path_label_freeze/oracle_rules.json:/census_layer_constraints`
- `PROTOCOL101_FULL_TRADER_GRAPH_V2_CONSOLIDATED_AUTHORITY_2026_07_28.md`, sections 2.3–2.4

The ledger defines “prospective” cost from the actual \(t+1\) ask and uses it to set D48/D49 masks. The tensor and composer require those masks before choosing the action at \(t\). This contradicts the same entry contract's correct sequencing—commit at \(t\), then recheck the actual quote at \(t+1\)—and its prohibition on future availability. Actual \(t+1\) fill price can govern the fill-time D48/D49 recheck, but it cannot causally filter the decision-time action set. The census oracle may use future outcomes for a ceiling, but P5 and candidate live-style policies may not. This contaminates the exact-P5 comparison and candidate/P5 parity. A causal repair needs a time-\(t\) intent/affordability mask plus a distinct \(t+1\) fill recheck and failed-fill transition.

### B2. **BLOCKING — D49 soft close does not implement signed A3 and is inconsistent across fee paths**

Exact citations:

- `PROTOCOL101_FULL_TRADER_GRAPH_V2_CONSOLIDATED_AUTHORITY_2026_07_28.md`, section 1.4, A3 soft-close wording
- `protocol101_ft2_08_data_tensor_label_contract/account_state_ledger_spec.json:/d49/soft_close_trigger`
- `protocol101_ft2_08_data_tensor_label_contract/replay_authority_v5_1_spec.json:/d49`
- `protocol101_ft2_08_data_tensor_label_contract/replay_authority_v5_1_spec.json:/fee_trajectories/four_dollar`
- `protocol101_ft2_08_data_tensor_label_contract/replay_authority_v5_1_spec.json:/boundary_fixtures`
- `protocol101_ft2_10_entry_science_contract/objective_spec.json:/serial_primary_population/account_and_safety_state/soft_close`
- `protocol101_ft2_10_entry_science_contract/objective_spec.json:/serial_primary_population/account_and_safety_state/fee_paths`

Signed A3 says soft close becomes permanent when remaining session budget cannot afford **any otherwise eligible** contract having executable premium at least $1. The repaired contracts replace that ladder-dependent predicate with the constant `remaining_budget_cents < 10300`. That is not equivalent: with $103 or more remaining, all otherwise eligible contracts may still cost more than the balance. The constant also encodes the $3 fee path, while the contract requires a separate $4 stress ledger and explicitly includes a boundary where a $1 contract is eligible under $3 but ineligible under $4. Under the $4 path the minimum $1 contract costs $104, so the same $103 threshold cannot be correct. This changes the signed safety state and potentially the set/timing of later actions.

### B3. **BLOCKING — FT2-10 and FT2-11 freeze different matched-random generators**

Exact citations:

- `protocol101_ft2_10_entry_science_contract/controls_spec.json:/frozen_nonselectable_random_controls/random_key`
- `protocol101_ft2_10_entry_science_contract/controls_spec.json:/frozen_nonselectable_random_controls/seed_ids`
- `protocol101_ft2_11_evidence_statistics_contract/multiplicity_spec.json:/frozen_nonselectable_matched_random_algorithm/source_contract`
- `protocol101_ft2_11_evidence_statistics_contract/multiplicity_spec.json:/frozen_nonselectable_matched_random_algorithm/single_algorithm`
- `protocol101_ft2_11_evidence_statistics_contract/multiplicity_spec.json:/frozen_nonselectable_matched_random_algorithm/candidate_specific_seed_namespace`
- `protocol101_ft2_11_evidence_statistics_contract/multiplicity_spec.json:/frozen_nonselectable_matched_random_algorithm/attempt_random_key`
- `protocol101_ft2_11_evidence_statistics_contract/evidence_standard.json:/combined_trader_promotable_layer`

FT2-10 hashes a tuple beginning with the frozen seed ID and fold/session/minute/identity. FT2-11 first derives a candidate-specific `base_seed` from candidate/grid information and then hashes a different tuple containing that value and attempt ID. FT2-11 simultaneously claims FT2-10 is the source and that there is one algorithm. Both cannot be true: identical attempt labels need not yield identical schedules. Because matched random is a required D1 comparator and part of the multiplicity family, comparator P&L, control validity, and adjusted evidence are not reproducible until one canonical key and namespace is selected.

### B4. **BLOCKING — Studentized max-\(T\) uses an undefined replicate standard error**

Exact citations:

- `protocol101_ft2_11_evidence_statistics_contract/bootstrap_spec.json:/studentized_maxT/standard_error`
- `protocol101_ft2_11_evidence_statistics_contract/bootstrap_spec.json:/studentized_maxT/centered_null_resample`
- `protocol101_ft2_11_evidence_statistics_contract/bootstrap_spec.json:/studentized_maxT/adjusted_bounds`
- `protocol101_ft2_11_evidence_statistics_contract/bootstrap_spec.json:/studentized_maxT/adjusted_p_values`
- `protocol101_ft2_11_evidence_statistics_contract/terminal_decision_spec.json:/adjusted_bound_authority`
- `protocol101_ft2_11_evidence_statistics_contract/mde_spec.json:/block_power`

The observed `SE_h` is defined as the standard deviation of uncentered bootstrap means. The centered-null replicate statistic then divides by `SE_h_star`, but no key defines how `SE_h_star` is computed—inside a replicate, from a nested bootstrap, from an analytic formula, or by reusing `SE_h`. All adjusted bounds and p-values depend on that denominator, and the terminal contract declares those bounds the sole authority. Block-power MDE also uses the same envelope. Different reasonable implementations yield different statistics, so terminal classification and reported MDE are not mechanically reproducible.

### B5. **MATERIAL — The cluster-aware WAIT rule never constrains exact-contract regret magnitude**

Exact citations:

- `protocol101_ft2_10_entry_science_contract/composer_spec.json:/uncertainty_wait/adjacent_strike_cluster`
- `protocol101_ft2_10_entry_science_contract/composer_spec.json:/uncertainty_wait/adjacent_strike_cluster/inside_cluster_substitute`
- `protocol101_ft2_10_entry_science_contract/composer_spec.json:/uncertainty_wait/rules`
- `protocol101_ft2_10_entry_science_contract/composer_spec.json:/wait_conditions`
- `protocol101_ft2_10_entry_science_contract/calibration_spec.json:/action_conditioned_gate/selected_contract_regret`
- `protocol101_ft2_10_entry_science_contract/forecast_heads.json:/selected_contract_regret`

The repair says within-cluster substitution is handled by the mandatory selected-contract regret head, but the actual WAIT rules compare the selected cluster with outside-cluster alternatives and MFE margins; they do not impose a threshold on expected or upper-quantile regret for the exact selected contract. The calibration gate limits calibration error/coverage, not the predicted regret level. A perfectly calibrated model can therefore predict high exact-contract regret, pass the gate, and still buy because its cluster wins. Adjacent-strike clustering resolves the original near-duplicate deadlock but leaves the claimed exact-contract safeguard disconnected from the decision.

### B6. **MATERIAL — Hard Phase-F reporting is allowed below the sample size covered by the AR demonstration**

Exact citations:

- `protocol101_ft2_11_evidence_statistics_contract/bootstrap_spec.json:/synthetic_AR_coverage_demonstration/generator/fold_shapes`
- `protocol101_ft2_11_evidence_statistics_contract/bootstrap_spec.json:/synthetic_AR_coverage_demonstration/acceptance`
- `protocol101_ft2_11_evidence_statistics_contract/bootstrap_spec.json:/phase_F_shadow_intervals/minimum_complete_sessions`
- `protocol101_ft2_11_evidence_statistics_contract/bootstrap_spec.json:/block_length_selection/candidate_block_lengths_sessions`

The synthetic coverage exercise tests 45-session shapes, while Phase F permits reporting once only 15 complete sessions exist using the same 3/5/8-session block candidates. The longest block then spans more than half the available sessions, a materially different finite-sample regime. The supplied coverage condition therefore does not validate the earliest allowed hard-reporting point. Either the minimum must match the validated regime or a preregistered small-\(n\) validity demonstration must cover the actual 15-session boundary.

### Additional focused consistency checks

- **\(t+1\) fill and horizon windows:** apart from B1, the label window is internally consistent. `label_spec.json:/horizon_window_definition` sets \(e=t+1\) and \(W_h=\{u:e<u\le e+h\}\), so path summaries start after entry rather than duplicating the fill minute; the exact 15:55 forced-flat boundary is frozen.
- **No-bid/MNAR:** the full-loss primary plus no-bid-excluded sensitivity and identity/order stability requirements answer the original MNAR defect.
- **Serial state and P5:** `objective_spec.json:/serial_primary_population` freezes the pending/occupied states, fee/account ledgers, serial replay, and fill-time rechecks; `objective_spec.json:/p5_under_cap_algorithm` freezes a deterministic exact-P5 policy. B1 and B2 still prevent that policy from being a causal, signed-safety-exact comparator.
- **Component versus promotable D1:** the two-layer evidence structure is now explicit and internally coherent. Component freeze is no-harm versus P5, while only the combined layer can claim positive incremental dollars versus P5 and matched random. B3 prevents execution of the matched-random half, not the conceptual separation.
- **Premium CDF:** band population, equal-session weighting, ties, interpolation, support minima, nested references, and no-cross-band fallback are now frozen and fail closed. The FT2-05 friction table supplies nonempty planning populations in the eligible D48-reference bands; no new CDF defect was found.
- **Census-v2/P5 ceiling share:** the movement is arithmetically consistent, not mysterious arithmetic. `v1_v2_impact.json` records best-session oracle P&L falling from 377161 to 341796 and P5 P&L rising from 117454 to 156830; `oracle_ceiling_summary.csv` reports the resulting v2 share as 0.45884094606139336. The packet itself says v1/v2 economics are not directly interchangeable because execution, missingness, role membership, and serial-budget semantics all changed. Thus the roughly 31% to 46% movement is the combined numerator increase and denominator decrease, but the supplied evidence does **not** identify which bundled semantic change caused either. It must not be used as causal validation of P5; B1 is additionally a direct reason the live-causal interpretation is presently unsafe.
- **Simulator-v5 activation:** `protocol101_serial_simulator_v5.py` does not itself implement the repaired D48/D49 semantics. `replay_authority_v5_1_spec.json` appropriately labels the v5.1 behavior as a design authority gated on later implementation/validation. I found no basis to treat the design contract as activated code evidence.

## FILES ACTUALLY OPENED

Isolation note: no other review-seat file and no rerun-output file was opened. Directory/file-name listings are not counted as opened evidence. The files whose contents or metadata were actually read were:

- `v4/docs/protocol101/training/contracts/PROTOCOL101_FULL_TRADER_GRAPH_V2_CONSOLIDATED_AUTHORITY_2026_07_28.md`
- `v4/docs/protocol101/training/execution/PROTOCOL101_FULL_TRADER_GRAPH_V2.json`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_TRADER_CHARTER.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_G4_HOLDOUT_REVISION_2026_07_19.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_G8_CALIBRATION_REVISION_2026_07_26.md`
- `v4/docs/protocol101/training/contracts/PROTOCOL101_D1_NEGATIVE_CONTROL_AND_INCREMENTAL_EDGE_AMENDMENT_2026_07_28.md`
- `v4/docs/protocol101/synchronization/contracts/PROTOCOL101_SCOPED_SYNCHRONIZATION_DECISION_2026_07_25.md`
- `v4/model/protocol101_serial_simulator_v5.py`
- `v4/audit/autoresearch/protocol101_ft2_04_path_label_freeze/label_spec.json`
- `v4/audit/autoresearch/protocol101_ft2_04_path_label_freeze/oracle_rules.json`
- `v4/audit/autoresearch/protocol101_ft2_04_path_label_freeze/census_sessions.json`
- `v4/audit/autoresearch/protocol101_ft2_04_path_label_freeze/intersection_proof.json`
- `v4/audit/autoresearch/protocol101_ft2_04_path_label_freeze/report.md`
- `v4/audit/autoresearch/protocol101_ft2_04_path_label_freeze/receipt.json`
- `v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/census_results.json`
- `v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/excluded_winners.csv`
- `v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/family_distributions.csv`
- `v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/friction_by_premium_band.csv`
- `v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/governance_receipt.json`
- `v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/guardrail_curves.csv`
- `v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/guardrail_threshold_curves.json`
- `v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/guardrail_trade_rates.csv`
- `v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/label_build_compute.json`
- `v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/label_session_inventory.csv`
- `v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/minimum_detectable_improvement.csv`
- `v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/mnar_sensitivity.csv`
- `v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/oracle_ceiling_summary.csv`
- `v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/oracle_replay_audit.json`
- `v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/oracle_session_results.csv`
- `v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/oracle_trade_results.csv`
- `v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/pareto_frontier.csv`
- `v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/progress.json`
- `v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/receipt.json`
- `v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/regime_headlines.csv`
- `v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/report.md`
- `v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/session_variance_components.csv`
- `v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/smoke_labels.parquet`
- `v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/smoke_summary.json`
- `v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/v1_v2_impact.json`
- `v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/contract.md`
- `v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/field_semantics_manifest.json`
- `v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/fold_roles.json`
- `v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/label_join_spec.json`
- `v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/tensor_schema.json`
- `v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/account_state_ledger_spec.json`
- `v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/replay_authority_v5_1_spec.json`
- `v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/receipt.json`
- `v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/validation.json`
- `v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/synthetic_golden_vectors.json`
- `v4/audit/autoresearch/protocol101_ft2_08_repair_attempt001/findings_crosswalk.json`
- `v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/contract.md`
- `v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/objective_spec.json`
- `v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/composer_spec.json`
- `v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/calibration_spec.json`
- `v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/forecast_heads.json`
- `v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/controls_spec.json`
- `v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/receipt.json`
- `v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/findings_crosswalk.json`
- `v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/contract.md`
- `v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/evidence_standard.json`
- `v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/bootstrap_spec.json`
- `v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/mde_spec.json`
- `v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/multiplicity_spec.json`
- `v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/shadow_sufficiency_spec.json`
- `v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/tripwire_spec.json`
- `v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/terminal_decision_spec.json`
- `v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/worked_terminal_examples.json`
- `v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/receipt.json`
- `v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/findings_crosswalk.json`
- `v4/audit/autoresearch/protocol101_ft2_20_parallel_design_review/seat2_ml_statistics.md`
