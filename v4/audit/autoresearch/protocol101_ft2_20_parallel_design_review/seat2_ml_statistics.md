# Protocol101 ML / Statistics Adversarial Review

**Verdict:** The frozen design is not ready for implementation or evidence generation. Direct future-label leakage, an embargo collision, a conflict with signed D1, invalid power arithmetic under dependence, and non-decidable terminal rules are blocking.

## [BLOCKING] Future `target_valid` directly controls the composer

**Anchor:** FT2-10 `composer_spec.json:/stage_2_conservative_upside_rank/available_horizon_rule`; FT2-08 `label_join_spec.json:/encoding/target_valid`, `/censoring/no_usable_future_mark`, and `/feature_firewall`.

The composer uses only horizons whose `target_valid` and `forecast_valid` masks are true. FT2-08 defines `target_valid` from future executable marks, including whether any usable future bid exists. It is stored in the physically separate label partition and explicitly unavailable to the runtime loader.

This lets historical actions avoid horizons based on future quote availability, while live actions cannot. Missing future bids can therefore change contract ranking, positive-upside checks, and WAIT decisions. The resulting outer evidence is not evidence for the runtime policy.

**Required acceptance test:** Strip all label partitions and label-derived masks before composing; historical actions must remain bit-identical. Runtime horizon availability must derive only from causal clock/model state.

## [BLOCKING] The census consumes the first fold’s embargo session

**Anchor:** FT2-08 `fold_roles.json:/folds[0]/embargo_session`, `/role_semantics/embargo`, and `/census`; FT2-10 `calibration_spec.json:/joint_conservatism/census_trade_rate_provenance`; FT2-10 `composer_spec.json:/stage_1_quality_screen/guardrail_anchor_grid`.

Fold 1 embargoes `2025-03-11`, while the frozen census contains that date. FT2-08 says embargo sessions may not fit, calibrate, select, evaluate, or provide diagnostics, yet FT2-10 uses census outcomes to freeze the anchor grid, trade-rate band, and permitted composer settings.

This is a role collision immediately adjacent to fold-1 validation. The broad statement that census sessions may inform design does not resolve the explicit embargo prohibition or the fail-closed incompatible-role rule.

## [BLOCKING] DeltaQ replaces the signed incremental-profit gate and can pass without incremental money

**Anchor:** FT2-11 §4.1, §4.5, and §5; `evidence_standard.json:/primary_metric`, `/entry_acceptance_all_required/pooled_improvement`, and `/retired_gate_role_mapping/G1_profitability`; FT2-10 `objective_spec.json:/decision_local_entry_metric`.

The signed D1 contract requires a statistically credible **fee-adjusted profit increment** over a candidate-specific matched-random control. FT2-11 instead tests transformed whole-path `DeltaQ`; G4 requires only positive absolute economics, not incremental economics against either comparator.

Moreover:

- Acceptance requires only `DeltaQ > 0`; `0.05` is an MDE/sufficiency boundary, not a minimum accepted effect.
- `DeltaU` needs only a positive point estimate.
- Q is scored at every governed decision without an occupancy or exit assumption, including actions that may be unreachable after an earlier entry.
- WAIT receives zero, so changing abstention can improve Q without establishing superior dollars.

A smooth, low-upside policy can therefore pass while making less money than P5 or matched random. This violates signed D1 and breaks the claimed interpretation of `no_genuine_entry_signal`.

## [BLOCKING] The mandatory action-conditioned calibration gate is unimplementable and absent from entry acceptance

**Anchor:** FT2-10 §6; `calibration_spec.json:/action_conditioned_gate`; `forecast_heads.json:/families`; FT2-11 `evidence_standard.json:/retired_gate_role_mapping/G8_calibration` and `/entry_acceptance_all_required`.

Confidence always controls WAIT and contract choice through the conformal margin. Signed G8 therefore requires an accepted action-conditioned gate first. But the forecast-head contract defines neither:

- a probability head for “the realized-label composer says WAIT”; nor
- an expected-regret head or 90% regret bound.

No deterministic derivation, training loss, joint-risk-set dependence model, or fallback is specified. FT2-11’s pass checklist also does not require this gate to pass. Thus the required prerequisite cannot be reproduced, yet `entry_evidence_pass` remains reachable on paper.

## [BLOCKING] The $24.45 MDE margin is arithmetically correct but statistically unsound

**Anchor:** FT2-11 §3, §8, and §9.2; `mde_spec.json:/mde_formula`, `/census_provenance`, `/pre_tranche_measurement`, and `/first_gpu_tranche_gate`; `bootstrap_spec.json:/effective_sample_size`.

The subtraction is correct: `$1,411.4510869565 - $1,386.9974872666 = $24.4535996899`. The inference is not.

- MDE uses `SD/sqrt(raw n)` even though FT2-11 assumes inter-session dependence and permits pooled ESS as low as 100 for 225 sessions.
- The pilot replacement requests marginal DeltaQ session SD, not long-run variance, autocovariances, or bootstrap power.
- The economic variance comes from a 46-session January–March hindsight `best_session` oracle-minus-P5 estimand, not candidate-minus-comparator neutral-lifecycle PnL.
- The 25% “plausible-edge cap” is an unsupported fraction of the oracle mean.
- Projected counts reach 1,193 “trades,” while the required hold-to-15:55 neutral lifecycle permits at most one executed trade per session.

Dependence adjustment or the correct economic estimand can erase the entire margin. The current machinery can wrongly authorize spend or classify an underpowered result as precise.

## [BLOCKING] Terminal outcomes are not mechanically decidable

**Anchor:** FT2-11 §9; `evidence_standard.json:/terminal_boundaries`; `bootstrap_spec.json:/confidence_intervals` and `/studentized_maxT`.

The classifications require an “adjusted upper 95% bound,” but the bootstrap specification defines only an adjusted one-sided lower bound and adjusted p-value. No adjusted upper-bound construction exists.

Other conflicts remain:

- Failed D1 matching is listed as `invalid_evidence`, while inability to obtain a matched control after valid attempts is `insufficient_evidence`.
- “Sufficient evidence” for fold, seed, metric, and regime failures has no statistic-specific MDE rule.
- A negative fifth fold may be excused using broad codes such as `sampling_uncertainty` or `mask_or_coverage_mix`, without a frozen quantitative attribution test.
- Calibration insufficiency and the inactive mandatory action gate are not mapped into the terminal precedence.

Different producers can therefore assign different terminal states to the same packet.

## [MATERIAL] The empirical-CDF metric is under-specified and not fully nested

**Anchor:** FT2-10 §5 and §7; `objective_spec.json:/decision_local_entry_metric/training_reference_distributions` and `/component_transform`.

`empirical_CDF_train_phase` does not define the reference population, session versus row weighting, valid-target filtering, ties, interpolation, metric/horizon-specific pooling, or fallback behavior. It also permits a CDF built from the complete outer-training role while scoring inner OOF blocks, including the scored block and chronologically later blocks.

Outer-validation use is explicitly prohibited, which is good, but the inner metric remains transductive and non-reproducible. Candidate ordering and whether `DeltaQ MDE <= 0.05` can change with implementation choices.

**Acceptable with documentation: no.** The primary outcome itself changes with unresolved choices; this requires a frozen algorithm and nested fit scope, not narrative clarification.

## [MATERIAL] OOF conformal residuals are reused for a different refitted model

**Anchor:** FT2-10 `calibration_spec.json:/chronological_inner_oof/final_outer_model`, `/distribution_calibration`, and `/uncertainty`.

Calibration residuals come from several earlier-block models trained on different sample sizes and regimes. The final outer model is then refit on all training sessions while reusing those residual calibrators. Ordinary split-conformal coverage does not automatically transfer to a different fitted predictor, especially under chronology and distribution drift.

Undercoverage makes the WAIT margin too small and increases false entries; overcoverage can manufacture rare-trading insufficiency.

**Acceptable with documentation: no.** A valid cross-conformal construction or a disjoint calibration set for the final fitted model is required.

## [MATERIAL] The dependence machinery lacks a validity condition

**Anchor:** FT2-11 `bootstrap_spec.json:/moving_block_bootstrap`, `/effective_sample_size`, and `shadow_sufficiency_spec.json:/agreement_gates`.

The fixed five-session circular block bootstrap assumes approximate stationarity but provides no stationarity check, block-length selection, or sensitivity rule. Circular wrapping also creates artificial adjacency between each fold’s last and first sessions. The ESS rule stops at the first nonpositive autocorrelation, so later positive dependence is ignored.

Phase-F shadow bounds then use ordinary decision-level Wilson intervals despite FT2-11 declaring the session to be the dependence cluster. Thousands of correlated minute decisions can create spurious precision from only 15 sessions.

**Acceptable with documentation: no.** Coverage must be demonstrated under frozen block-length alternatives or a valid dependence model, and shadow uncertainty must cluster by session.

## [MATERIAL] Multiplicity omits selectable random-control attempts and invalidating controls

**Anchor:** FT2-11 §6–§7; `multiplicity_spec.json:/registered_families`, `/not_hypotheses`, and `/d1_v2_full_ladder`; FT2-10 `controls_spec.json:/mandatory_controls`.

The 96 entry statistics correctly include three shortlisted lineages, all 16 alpha-by-k variants, and both comparators. Treating seeds as replications is defensible only because every seed and the ensemble must pass. Lifecycle variants are at least required to enter their later ledger.

The family nevertheless omits:

- candidate-specific random-control draws or redraw attempts;
- shuffled-model contrasts that can invalidate selection;
- any selectable combination of multiple admitted extension channels; and
- controls whose undefined “validity” can alter plateau resets or eligibility.

No random-control seed set, attempt cap, aggregation rule, or pre-outcome matching algorithm is frozen. Repeatedly drawing until post-replay exposure matches creates an uncounted selectable comparator and can condition on outcome-affected execution state.

**Acceptable with documentation: no.** Every attempted comparator/control identity must be preregistered and included, or the control must be generated by one frozen nonselectable algorithm.

## [MATERIAL] No-bid censoring is missing-not-at-random and rewards illiquidity

**Anchor:** FT2-08 §9; `label_join_spec.json:/censoring/no_bid_or_stale_future_minute` and `/censoring/no_usable_future_mark`.

Partially missing future no-bid minutes are excluded from downside, underwater duration, stability, and run continuity. A disappearing bid is plausibly associated with adverse option value and poor executability, so this is not neutral censoring. Contracts with the worst future liquidity can receive artificially favorable Q whenever some other marks remain.

**Acceptable with documentation: no.** The design needs a preregistered adverse/missingness sensitivity and must show that policy ranking does not depend on selectively omitted no-bid periods.

# FILES ACTUALLY OPENED

Brace groups below expand to every named top-level file; no checkpoint or superseded subdirectory contents were opened.

- `/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/contracts/{PROTOCOL101_FULL_TRADER_GRAPH_V2_CONSOLIDATED_AUTHORITY_2026_07_28.md,PROTOCOL101_TRADER_CHARTER.md,PROTOCOL101_G4_HOLDOUT_REVISION_2026_07_19.md,PROTOCOL101_G8_CALIBRATION_REVISION_2026_07_26.md,PROTOCOL101_D1_NEGATIVE_CONTROL_AND_INCREMENTAL_EDGE_AMENDMENT_2026_07_28.md}`
- `/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/execution/PROTOCOL101_FULL_TRADER_GRAPH_V2.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/synchronization/contracts/PROTOCOL101_SCOPED_SYNCHRONIZATION_DECISION_2026_07_25.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/{contract.md,fold_roles.json,label_join_spec.json,receipt.json,storage_spec.json,tensor_schema.json}`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/{calibration_spec.json,composer_spec.json,contract.md,controls_spec.json,forecast_heads.json,objective_spec.json,receipt.json}`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/{bootstrap_spec.json,contract.md,evidence_standard.json,mde_spec.json,multiplicity_spec.json,receipt.json,shadow_sufficiency_spec.json,tripwire_spec.json}`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_04_path_label_freeze/{census_sessions.json,compute_census_outer_test_intersection.py,intersection_proof.json,label_spec.json,oracle_rules.json,receipt.json,report.md,test_compute_census_intersection.py}`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/{census_results.json,excluded_winners.csv,family_distributions.csv,friction_by_premium_band.csv,governance_receipt.json,guardrail_curves.csv,guardrail_threshold_curves.json,guardrail_trade_rates.csv,label_build_compute.json,label_session_inventory.csv,minimum_detectable_improvement.csv,oracle_ceiling_summary.csv,oracle_replay_audit.json,oracle_session_results.csv,oracle_trade_results.csv,pareto_frontier.csv,progress.json,receipt.json,regime_headlines.csv,report.md,session_variance_components.csv}`
- `/Users/gduby/Documents/autoresearch-trading/v4/model/protocol101_serial_simulator_v5.py`
