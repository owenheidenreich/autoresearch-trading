# Protocol101 FT2-20 FINAL RE-REVIEW rerun002 — Seat 1 Trading Realism

Review mode: fresh isolated, evidence-only, no repairs.

Authority verification:

- Consolidated authority SHA-256: `2363d3f986daba20bd5087ed751dc5b2d839e76cd6413aeca0bcd255eb98857a` — matches the required hash.
- Final aggregate receipt SHA-256: `19b74e513f0c6c68b50167b37e4bcfd40cb4ccd0f3c2240201ff3cbbd178f4f3` — matches the required hash.
- Current v3 receipt SHA-256 values reproduce: FT2-08 `731cc6fb4c44bd0650d658e71fffbdac7c3e0f5d68701b2e05b8576b014b078c`; FT2-10 `c04a591fd3034a2caba756e8f9dc1d7f5c24c397170894b0b6b0eca7a083fa6f`; FT2-11 `bdb52c4f9178da1cab75ec4a80bd13c1db8c0074c1c398f380729d1df80c5ef9`; census v3 `a252feb2007b2aece77f377461bc6fa5a882e929bd2f7d3f23a47c07401cfdce`.
- Every current top-level deliverable named by the FT2-08, FT2-10, FT2-11, and FT2-05 v3 receipts reproduced its receipt hash: 14/14, 8/8, 10/10, and 23/23 respectively. Nested census receipt entries were not traversed.
- Shared intent law SHA-256 `5c117d716cea3c986605faf7b58d510eedce3264a0c04f9368f6dc509dea6bd0`, canonical matched-random generator SHA-256 `8bdbe8beb4734852526cd2981be76098f792d5c4b9fee30f89f8a2d952abf754`, Graph V2 SHA-256 `b06a26be59307c130da84f2dc5b6f3224c272e6c4093e83abd5bc0b280ca6d09`, and simulator-v5 SHA-256 `7296a437577ed006326d2ad35ad1f3499c4925334556d64d8c5fb75e4985f548` reproduce their receipts.

Files/categories checked:

- Authority and signed sources: the consolidated authority, Graph V2 JSON, Trader Charter, G4 holdout revision, G8 calibration revision, scoped synchronization decision, and D1 amendment.
- FT2-04 top level: `label_spec.json`, `oracle_rules.json`, `receipt.json`, and `intersection_proof.json`.
- FT2-05 top level: `report.md`, `census_results.json`, `v2_v3_impact.json`, `label_build_compute.json`, `fill_recheck_rejections.csv`, `oracle_replay_audit.json`, `oracle_ceiling_summary.csv`, `mnar_sensitivity.csv`, `governance_receipt.json`, `progress.json`, and `receipt.json`; all receipt-declared top-level deliverables were hash-checked.
- FT2-08 top level: `contract.md`, `intent_fill_recheck_law.json`, `account_state_ledger_spec.json`, `tensor_schema.json`, `field_semantics_manifest.json`, `label_join_spec.json`, `replay_authority_v5_1_spec.json`, `validation.json`, `findings_crosswalk.json`, and `receipt.json`; all receipt-declared top-level deliverables were hash-checked.
- FT2-10 top level: `contract.md`, `objective_spec.json`, `composer_spec.json`, `calibration_spec.json`, `matched_random_generator_spec.json`, `controls_spec.json`, `forecast_heads.json`, and `receipt.json`.
- FT2-11 top level: `contract.md`, `bootstrap_spec.json`, `evidence_standard.json`, `multiplicity_spec.json`, `shadow_sufficiency_spec.json`, `terminal_decision_spec.json`, `mde_spec.json`, `tripwire_spec.json`, and `receipt.json`.
- Review/repair evidence: original Seat-1 review and joined review; rerun001 Seat-1 review, joined review, and receipt; final-repair unified `findings_crosswalk.json`, `aggregate_receipt.json`, and `consistency_checker_output.json`.
- Simulator: `v4/model/protocol101_serial_simulator_v5.py`.

All packet directories were enumerated at top level only. I did not enter, recursively list, search, or read any `superseded/` directory. I read no repository file outside the allowlist and inspected no scratch path except the assigned output directory.

## A. FINDINGS VERIFICATION

### S1-01 — no-bid periods erased instead of losses

- Prior severity/disposition: `BLOCKING`; rerun001 concluded `ANSWERED`.
- Repair inspected: `protocol101_ft2_04_path_label_freeze/label_spec.json:/missing_minute_rule` assigns every non-executable bid the full-loss bound and forbids earlier-bid substitution; `protocol101_ft2_08_data_tensor_label_contract/label_join_spec.json:/no_bid` preserves underwater/run-breaking behavior; `protocol101_ft2_11_evidence_statistics_contract/evidence_standard.json:/mnar_no_bid_sensitivity` makes full-loss primary and the excluded view sensitivity-only.
- Independent conclusion: **ANSWERED**. Exact 15:55 no-bid realization is zero option value, with no forward/prior fill.

### S1-02 — entry and exit use prices known before execution

- Prior severity/disposition: `BLOCKING`; rerun001 concluded `ANSWERED`.
- Repair inspected: `protocol101_ft2_08_data_tensor_label_contract/intent_fill_recheck_law.json:/decision_time_intent` commits from `A_t`; `/fill_time_recheck` fills only at the selected identity's actual `A_(t+1)`; `field_semantics_manifest.json:/completed_minute_law` makes EXIT at `v` fill at `v+1`; `replay_authority_v5_1_spec.json:/required_v5_1_behavior` preserves that ordering and gates activation.
- Independent conclusion: **ANSWERED** for the fill-timing defect. The separate decision-mask authority conflict under FRESH-S1-B1 does not change the fact that the fill itself is post-decision.

### S1-03 — primary entry test is not one-account serial and P5 is unfair

- Prior severity/disposition: `BLOCKING`; rerun001 concluded `ANSWERED`.
- Repair inspected: `protocol101_ft2_10_entry_science_contract/objective_spec.json:/serial_primary_population` freezes the universal chronological grid, pending/occupied states, separate causal ledgers, failed-fill accounting, neutral lifecycle, and fee paths; `/p5_under_cap_algorithm` freezes P5 timing, side, exact eligible set, deterministic tie break, WAIT states, selected-only recheck, and its own ledger; `protocol101_ft2_11_evidence_statistics_contract/evidence_standard.json:/authoritative_population` adopts rather than redefines it.
- Independent conclusion: **ANSWERED**.

### S1-04 — composer can structurally delete the Charter big-win column

- Prior severity/disposition: `BLOCKING`; rerun001 concluded `ANSWERED`.
- Repair inspected: `protocol101_ft2_10_entry_science_contract/composer_spec.json:/stage_1_quality_screen/no_screen_control_arm` requires the alpha-zero arm; `objective_spec.json:/tail_guard_relative_capture` freezes like-for-like tail comparisons; `protocol101_ft2_11_evidence_statistics_contract/tripwire_spec.json:/relative_capture_tripwire` routes an incidence ratio below 0.50 to owner decision and treats a zero control denominator as insufficient evidence.
- Independent conclusion: **ANSWERED**.

### S1-05 — dual-unit `min()` biases middle-premium contracts

- Prior severity/disposition: `MATERIAL`; rerun001 concluded `ANSWERED`.
- Repair inspected: `protocol101_ft2_10_entry_science_contract/objective_spec.json:/nested_empirical_cdf` freezes decision-time premium-band/phase-specific, session-balanced CDFs, midranks, support floors, and no cross-band fallback; `composer_spec.json:/stage_2_conservative_upside_rank/dollar_percent_arbitration` takes the minimum only after both units are transformed within the same causal band.
- Independent conclusion: **ANSWERED**. MATERIAL classification: `acceptable-with-documentation`; the deterministic cross-premium bias is removed, and the retained dual-unit conservatism is explicit.

### S1-06 — D49 size-down can force lottery exposure and simulator v5 does not enforce it

- Prior severity/disposition: `MATERIAL`; rerun001 concluded `PARTIALLY ANSWERED` because the `$4` path still inherited `$103`.
- Repair inspected: `protocol101_ft2_08_data_tensor_label_contract/intent_fill_recheck_law.json:/soft_close` uses the current exact ladder and a computed active-fee threshold; `/fee_paths` computes 10,300/10,400 cents and forbids a shared literal; `account_state_ledger_spec.json:/d49` repeats the ladder predicate; `replay_authority_v5_1_spec.json:/activation_gate` accurately leaves current simulator v5 non-authoritative until FT2-30/31.
- Independent conclusion: **ANSWERED** at the design-contract level. MATERIAL classification: `acceptable-with-documentation`; v5 remains unimplemented, but activation is explicitly fail-closed rather than represented as evidence.

### S1-07 — exact-contract uncertainty deadlocks directional sessions

- Prior severity/disposition: `MATERIAL`; rerun001 concluded `ANSWERED`, then cross-seat fresh scan identified insufficient cluster breadth.
- Repair inspected: `protocol101_ft2_10_entry_science_contract/composer_spec.json:/uncertainty_wait/directional_substitute_cluster` now covers same-right/same-expiry strikes within ±10 points (two ladder steps each side); `/mandatory_action_conditioned_gate/selected_contract_regret_action_constraint` separately requires finite q90 normalized regret `<=0.10`.
- Independent conclusion: **ANSWERED** for the original exact-neighbor deadlock. MATERIAL classification: `acceptable-with-documentation`; the final fixed breadth and exact-contract regret ceiling are explicit design choices.

### FRESH-S1-B1 — D48/D49 action masks leak actual `t+1` ask

- Prior severity/disposition: rerun001 fresh `BLOCKING`, routed to FT2-08 with FT2-10/11 synchronization; final producer crosswalk claims `ANSWERED`.
- Repair inspected: `protocol101_ft2_08_data_tensor_label_contract/intent_fill_recheck_law.json:/decision_time_intent` correctly forbids `A_(t+1)` and `/fill_time_recheck` correctly rechecks only the committed identity; FT2-08/10/11 consumers pin its exact hash.
- Contradictory active evidence: `protocol101_ft2_04_path_label_freeze/oracle_rules.json:/census_layer_constraints/D48_per_trade_premium_cap/rule` still makes a BUY at `t` eligible from actual `A_(t+1)` and `/effect` masks the action before oracle choice. `/D49_budget_aware_entry/rule` and `/effect` likewise use `A_(t+1)`, while `/oracle_sequencing/serial_ceiling_replay` requires the future ask to exist and pass before commitment. Its `/same_game` text requires the same mask in the model and live guard. This file is not superseded: FT2-04 receipt v3 hashes it as `b435d5fd8e37aca0919196e9f8e4b2d79132e3dd8dca04cc592887e2c6b79ddb`, and FT2-05 receipt v3 lists that same hash as an active input alongside the new intent-law hash. No precedence rule reconciles them. `consistency_checker_output.json:/checks/all_FT2_10_FT2_11_intent_refs_exact` checks only FT2-10/11 consumers, while `/checks/census_v3_chain_and_scope` verifies hashes rather than the conflicting semantics.
- Independent conclusion: **NOT ACTUALLY ANSWERED** end to end. The new law is sound in isolation, but the active FT2-04/05 authority chain still freezes the exact prohibited future-informed mask and cannot be live-equivalent.

### FRESH-S1-B2 — `$1 + fee` soft close fixed at `$103` on the `$4` path

- Prior severity/disposition: rerun001 fresh `MATERIAL`, retained; final producer crosswalk claims `ANSWERED`.
- Repair inspected: `protocol101_ft2_08_data_tensor_label_contract/intent_fill_recheck_law.json:/fee_paths` derives 10,300 and 10,400 cents from the active fee and forbids a shared threshold; `/soft_close` uses those bounds only inside the current-ladder existence predicate; `account_state_ledger_spec.json:/d49`, `replay_authority_v5_1_spec.json:/fee_trajectories`, and `protocol101_ft2_10_entry_science_contract/objective_spec.json:/serial_primary_population/account_and_safety_state/soft_close` agree.
- Independent conclusion: **ANSWERED** in the final v3 law. MATERIAL classification: `acceptable-with-documentation`. The stale FT2-04 oracle authority remains part of the separate blocking end-to-end conflict above.

### FRESH-S1-B3 — finite near-close censored labels enter empirical-CDF references

- Prior severity/disposition: rerun001 fresh `MATERIAL`, retained; final producer crosswalk claims `ANSWERED`.
- Repair inspected: `protocol101_ft2_10_entry_science_contract/objective_spec.json:/nested_empirical_cdf/full_window_filter` requires `censored=false`, exact deadline, nominal=available=N, and all N marks; `finite_but_censored_near_close` explicitly excludes shortened values; `forecast_heads.json:/full_window_training_filter` matches it and preserves only the survival-likelihood censoring exception.
- Independent conclusion: **ANSWERED**. MATERIAL classification: `acceptable-with-documentation`; the runtime and fitted reference populations now use the same nominal horizon meaning.

## B. FRESH SCAN

### 1. Intent-mask/fill-recheck law end to end

The final shared law is internally correct: causal `A_t` intent arithmetic; one exact commitment; selected-only `A_(t+1)` recheck; failed fills open no position, charge no premium/fee, change PnL by zero, permit no substitution, and may retry no earlier than `t+2`; `historical_live_guard_equivalence` requires identical integer arithmetic and fail-closed behavior. FT2-10 candidate, P5, and matched random consume that law.

However, the active, receipt-hashed FT2-04 `oracle_rules.json` still freezes actual-`t+1` D48/D49 decision masks and claims same-game model/live use. The current evidence therefore contains two incompatible end-to-end laws. Conclusion: **BLOCKING**; this is F-TR-001 and makes FRESH-S1-B1 NOT ACTUALLY ANSWERED.

### 2. D48-reference population change: 159,312 to 121,553

`v2_v3_impact.json:/scalar_changes` and `/v3_intent_and_recheck_counts`, together with `label_build_compute.json:/per_session`, reconcile exactly:

```text
old v2 reference rows                         159,312
minus new causal A_t intent rows              128,758
decision-population change                     30,554
new causal intents                             128,758
minus explicit t+1 rejected fills                7,205
new realized-entry/reference rows             121,553
total decline                                  37,759 = 30,554 + 7,205
decline / old population                       23.7013%
```

All 45 per-session rows satisfy `intent - rejected = reference` with zero residual, and `label_rows_all_governed_candidates` stays exactly 460,937 in v2 and v3. Thus the new definition explains the entire arithmetic shift: 30,554 rows leave the causal intent population rather than being selected from a future price, and 7,205 legal intents remain explicit rejected-fill events without realized-entry labels. No master governed row was silently deleted. Economically attractive future paths may exist among the 30,554, but they are not causally actionable at `t`; that exclusion is economically necessary, not silent selection loss. This arithmetic consistency does not resolve the contradictory FT2-04 authority noted above.

### 3. Parametric soft-close floor on both paths

`intent_fill_recheck_law.json:/fee_paths/threshold_formula` computes `100*100+active_fee`, yielding 10,300 cents for fee_3 and 10,400 for fee_4; `/soft_close` uses a current-ladder existence predicate and expressly says the threshold is not a shared literal. FT2-08 ledger/replay, FT2-10 population, and FT2-11 signed-safety check agree. Conclusion: the final v3 repair itself is **ANSWERED**; the old oracle-rule conflict prevents an end-to-end pass but does not leave the v3 formula under-specified.

### 4. One canonical matched-random comparator hash

`protocol101_ft2_10_entry_science_contract/matched_random_generator_spec.json` is the sole source at SHA-256 `8bdbe8beb4734852526cd2981be76098f792d5c4b9fee30f89f8a2d952abf754`. FT2-10 `controls_spec.json:/frozen_nonselectable_random_controls` and FT2-11 `multiplicity_spec.json:/frozen_nonselectable_matched_random_algorithm` pin exactly that file/hash, the same eight attempts, no candidate seed namespace, and no local override. Conclusion: **ANSWERED**.

### 5. `SE_h_star` independent numeric recomputation

Using only the frozen worked fixture `[1,2,3,4,5]`, `L=2`, `K=4`:

```text
observed delete-block means = [4, 10/3, 8/3, 2]
delete-mean average         = 3
sum squared deviations      = 20/9 = 2.22222222222222
factor (n-L)/(L*K)          = 3/8 = 0.375
V                           = (3/8)*(20/9) = 5/6
SE_h                        = sqrt(5/6) = 0.912870929175277
observed mean               = 3
t_h                         = 3/SE_h = 3.286335345031
```

For centered replicate `[-2,-2,0,0,2]`:

```text
replicate delete means      = [2/3, 0, -2/3, -4/3]
replicate delete average    = -1/3
sum squared deviations      = 20/9
V_star                      = 5/6
SE_h_star                   = sqrt(5/6) = 0.912870929175277
replicate mean              = -0.4
t_h_star                    = -0.4/SE_h_star = -0.438178046004133
```

These independently reproduce `bootstrap_spec.json:/worked_studentization_example`. `/studentized_maxT/standard_error_authority/replicate_SE_h_star` also uniquely requires this direct per-replicate delete-L calculation and forbids observed-SE reuse or nested bootstrap. Conclusion: **ANSWERED**.

### 6. Source-transfer ordering versus Graph V2

Graph V2 routes `FT2-91 pass -> FT2-92 -> FT2-93`. FT2-10 `calibration_spec.json:/source_transfer_gate` and FT2-11 `evidence_standard.json:/source_transfer_graph_topology` plus `shadow_sufficiency_spec.json:/graph_topology` now permit `transfer_not_yet_run` before FT2-92, make FT2-92 produce the transfer outcome, and require its pass only to leave for FT2-93. Conclusion: **ANSWERED**; no circular precondition remains.

### 7. 15:55 terminal action law

FT2-08 `tensor_schema.json:/open_state/actions_at_1555` is empty; `/terminal_minute_1555` is `FORCED_FLAT_TERMINAL`, `model_invoked=false`, with no learned action or `t+1` execution. `field_semantics_manifest.json:/completed_minute_law` fixes the last learned EXIT at 15:54/15:55 and suppresses duplicate forced flat. Conclusion: **ANSWERED**.

### 8. Deterministic session-start equity primitive

`account_state_ledger_spec.json:/session_start/live_paper_value` selects the greatest authenticated source timestamp in `[09:29:00,09:30:00]` on the session date, ties by greatest authenticated update sequence, requires receipt by 09:30:05 and source age at most 60 seconds, parses decimal USD once with half-even cent rounding, and fails closed. `/account_scope` requires a dedicated Protocol101 paper account and zero unrelated positions, orders, executions, commissions, or unexplained cash flows. Offline equity is one immutable causal-cash source. Conclusion: **ANSWERED**.

### 9. Near-close full-window CDF filter

FT2-10 `objective_spec.json:/nested_empirical_cdf/full_window_filter` and `forecast_heads.json:/full_window_training_filter` exclude finite shortened/censored hN values and require the complete N-mark window. Remaining-session rows require the complete exact-15:55 grid. Only right-censored first-profit rows enter their registered survival likelihood. Conclusion: **ANSWERED**.

### 10. Substitute-cluster breadth

`composer_spec.json:/uncertainty_wait/directional_substitute_cluster` widens same-right/same-expiry membership to ±10 strike points, covering the rerun001 ±10 counterexample, while the selected exact contract separately faces a hard conformal q90 regret bound of 0.10. The outside-cluster comparison and equality-to-WAIT law are explicit. Conclusion: **ANSWERED** for the recorded breadth defect.

### 11. Phase-F hard reporting floor

`bootstrap_spec.json:/phase_F_shadow_intervals/minimum_complete_sessions_for_hard_bounds` is 45, matching the 45-session synthetic AR shapes. Sessions 1–44 permit only point estimates/raw counts and route insufficient shadow evidence. `shadow_sufficiency_spec.json:/minimum_no_order_live_shadow_evidence/complete_sessions_for_hard_gate` also equals 45. Conclusion: **ANSWERED**.

### 12. MNAR alternate refit

`evidence_standard.json:/mnar_no_bid_sensitivity/alternate_refit_protocol` requires independent refit of every label-dependent head, nested CDF, calibrator, threshold, WAIT/regret head, inner selection, and ensemble; holds features, architecture/grid, folds, seeds, comparators, fee paths, multiplicity, and resolver fixed; forbids cross-view reuse. `terminal_decision_spec.json:/statistic_specific_sufficiency/MNAR_ranking_stability` makes that full refit part of completeness. Conclusion: **ANSWERED**.

### 13. Census-v3 internal consistency

The oracle/P5 behavior is reconcilable. `v2_v3_impact.json:/oracle_changes` shows `best_session` oracle PnL changing only `$341,796 -> $340,736` with 108 trades unchanged, while P5 changes `$156,830 -> $154,008` and `580 -> 589` trades. `oracle_replay_audit.json` reports zero rejected fills for all hindsight-oracle variants but six for `p5:best_session` (five D48, one D49), with additional rejections across other P5 variants. A hindsight ceiling can choose among successful post-recheck labels, while causal P5 commits from `A_t` and can fail at `t+1`; approximately flat oracle ceilings alongside lower P5 are therefore not numerically inconsistent.

The stated `1.6%` rejection figure is reproducible only as:

```text
7,205 / 460,937 all governed candidate rows = 1.56312034%
```

It is not the fill-recheck rejection rate conditional on a legal intent:

```text
7,205 / 128,758 causal intents = 5.59576881%
```

Nor is it the share of the old v2 reference population (`4.52257206%`). Therefore `1.6%` must be labeled “rejected intents as a share of all governed candidate rows,” not an intent-conditional fill rejection rate. The top-level census artifacts reproduce all counts and the per-session identity `intent - rejected = realized reference`; no unexplained count residual remains. The authority conflict in item 1 remains blocking provenance despite this internal arithmetic.

Additional trading-realism scan: no further defect was found within the allowlist beyond F-TR-001.

FINDINGS

- **F-TR-001 — BLOCKING — incompatible active intent-mask authorities.** Affected contracts/artifacts: FT2-04 path-label/oracle freeze, FT2-05 opportunity census, FT2-08 data/tensor/label contract, and downstream FT2-10/11 consumers. Evidence: `protocol101_ft2_04_path_label_freeze/oracle_rules.json:/census_layer_constraints/D48_per_trade_premium_cap`, `/D49_budget_aware_entry`, and `/oracle_sequencing` freeze future-`A_(t+1)` action legality, while `protocol101_ft2_08_data_tensor_label_contract/intent_fill_recheck_law.json:/decision_time_intent` forbids it. Both hashes are active inputs of the current v3 receipts, and the final consistency check does not semantically compare FT2-04/05 to the new law. This is the exact rerun001 blocker, so FRESH-S1-B1 is NOT ACTUALLY ANSWERED and the historical/census/live same game is not uniquely implementable.

SEAT ROUTING RECOMMENDATION: STOP-REDESIGN-REQUIRED
