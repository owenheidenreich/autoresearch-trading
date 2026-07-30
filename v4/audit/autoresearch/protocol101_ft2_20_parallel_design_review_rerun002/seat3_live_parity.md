# Protocol101 FT2-20 Final Re-review Rerun002 — Seat 3 Live Parity

Verdict: **NOT DESIGN-ACCEPTABLE.** All 19 rerun001 Seat 3 findings are actually answered by the final repaired contracts, but the mandatory fresh census scan found one new **MATERIAL, not-acceptable** evidence defect. The 37,759-row D48-reference reduction is arithmetically visible but lacks a row-level v2-to-v3 transition and economic-impact audit. Under the binding routing law, the recommendation is `STOP-REDESIGN-REQUIRED`.

Pinned authority verification:

- `v4/docs/protocol101/training/contracts/PROTOCOL101_FULL_TRADER_GRAPH_V2_CONSOLIDATED_AUTHORITY_2026_07_28.md`
- required SHA-256: `2363d3f986daba20bd5087ed751dc5b2d839e76cd6413aeca0bcd255eb98857a`
- observed SHA-256: `2363d3f986daba20bd5087ed751dc5b2d839e76cd6413aeca0bcd255eb98857a`
- result: match

Aggregate receipt verification:

- `v4/audit/autoresearch/protocol101_ft2_final_repair_attempt002/aggregate_receipt.json`
- required and observed SHA-256: `19b74e513f0c6c68b50167b37e4bcfd40cb4ccd0f3c2240201ff3cbbd178f4f3`
- child receipt hashes observed and matched aggregate receipt: FT2-08 `731cc6fb4c44bd0650d658e71fffbdac7c3e0f5d68701b2e05b8576b014b078c`; FT2-10 `c04a591fd3034a2caba756e8f9dc1d7f5c24c397170894b0b6b0eca7a083fa6f`; FT2-11 `bdb52c4f9178da1cab75ec4a80bd13c1db8c0074c1c398f380729d1df80c5ef9`
- FT2-04 v3 receipt observed: `c763dee85293d73a4f367627e8f483a209565d3f486f82fb5b1c17f3ff611775`
- FT2-05 v3 receipt observed and matched downstream pins: `a252feb2007b2aece77f377461bc6fa5a882e929bd2f7d3f23a47c07401cfdce`
- every top-level deliverable listed by the FT2-04 and FT2-05 v3 receipts matched its listed hash; every deliverable listed by the FT2-08, FT2-10, and FT2-11 v3 receipts matched its listed hash
- unified crosswalk, checker, and checker-output hashes matched the aggregate receipt; the shared intent-law hash is `5c117d716cea3c986605faf7b58d510eedce3264a0c04f9368f6dc509dea6bd0`, and the canonical matched-random generator hash is `8bdbe8beb4734852526cd2981be76098f792d5c4b9fee30f89f8a2d952abf754`

Isolation and files/categories checked:

- Authority content: the pinned consolidated authority, Graph V2 JSON, Trader Charter, G4 revision, G8 revision, scoped synchronization decision, and D1 amendment.
- Review content: rerun001 `seat3_live_parity.md`, rerun001 `joined_review.md`, rerun001 receipt, the unified final-repair `findings_crosswalk.json`, aggregate receipt, and consistency-checker output. The top-level regular-file inventories of both allowlisted prior-review directories were inspected.
- FT2-04 top-level content relevant to timing and labels: `label_spec.json`, `oracle_rules.json`, `census_sessions.json`, `intersection_proof.json`, `report.md`, and receipt; all top-level receipt deliverables were hash-checked.
- FT2-05 top-level content relevant to the census: `census_results.json`, `report.md`, `v2_v3_impact.json`, `label_build_compute.json`, `oracle_ceiling_summary.csv`, `oracle_replay_audit.json`, `guardrail_trade_rates.csv`, `fill_recheck_rejections.csv`, `mnar_sensitivity.csv`, `session_variance_components.csv`, `minimum_detectable_improvement.csv`, governance receipt, and receipt; all top-level receipt deliverables were hash-checked.
- FT2-08 top-level contract content: `contract.md`, `intent_fill_recheck_law.json`, `account_state_ledger_spec.json`, `tensor_schema.json`, `field_semantics_manifest.json`, `identity_mapping_spec.json`, `storage_spec.json`, `label_join_spec.json`, `replay_authority_v5_1_spec.json`, `synthetic_golden_vectors.json`, `fold_roles.json`, validation, crosswalk, and receipt.
- FT2-10 top-level contract content: `contract.md`, `objective_spec.json`, `composer_spec.json`, `calibration_spec.json`, `matched_random_generator_spec.json`, `forecast_heads.json`, `controls_spec.json`, crosswalk, and receipt.
- FT2-11 top-level contract content: `contract.md`, `bootstrap_spec.json`, `evidence_standard.json`, `multiplicity_spec.json`, `mde_spec.json`, `shadow_sufficiency_spec.json`, `terminal_decision_spec.json`, `tripwire_spec.json`, `worked_terminal_examples.json`, crosswalk, and receipt.
- Simulator content: `v4/model/protocol101_serial_simulator_v5.py`.
- Final-repair top-level content: aggregate receipt, unified crosswalk, checker output, and the two checker/finalization scripts by hash.

Only top-level regular files were enumerated in allowlisted directories. No recursive directory command was used. No `superseded/` directory was entered, listed recursively, searched, or read. No unallowlisted repository file, future rerun002 destination, other scratch path, broker/recorder/protected data, paid resource, web source, or git history was accessed. No model was trained or fitted, no new session statistic was calculated, no reviewed artifact was edited, and FT2-21 was not started.

## A. FINDINGS VERIFICATION

### S3-01 — Simulator v5 lacks the assigned safety game

Prior locator/severity/disposition: rerun001 joined review locator `joined_review.md:805`; original **BLOCKING**; rerun001 Seat 3 concluded `ANSWERED at the design-contract level`.

Repair inspected: `v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/replay_authority_v5_1_spec.json:/base_simulator`, `/required_v5_1_behavior`, `/fee_trajectories`, `/activation_gate`, and `/independent_replay_assertions`; `v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/evidence_standard.json:/two_layer_acceptance/entry_component_acceptance/all_required/signed_safety`.

Independent verification: current `protocol101_serial_simulator_v5.py` still has `STRESS_APPLICATION="metrics_only"`, `quote_age_gate=false`, and no D48/D49/soft-close implementation. The repaired contract accurately marks it unmodified and prevents v5.1 authority until later producer implementation and independent acceptance. FT2-11 classifies a wrong or unreconstructable replay as invalid evidence.

Conclusion: **ANSWERED** as a design/activation-gate finding; current v5 is not falsely treated as the repaired authority.

### S3-02 — Live D48/D49 state was not frozen or reconstructable

Prior locator/severity/disposition: `joined_review.md:830`; original **BLOCKING**; rerun001 disposition `PARTIALLY ANSWERED`, residual **MATERIAL** because session-start equity was nondeterministic and could include stale or unrelated state.

Repair inspected: `v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/account_state_ledger_spec.json:/session_start`, `/owned_trade_ledger`, `/d48`, `/d49`, `/restart_recovery`, `/per_decision_audit_fields`, and `/fail_closed_conditions`.

Independent verification: live paper equity now selects the greatest authenticated broker-source timestamp in `[09:29:00,09:30:00]` ET for the session date, breaks timestamp ties by authenticated update sequence, requires receipt by 09:30:05, normalizes once to integer cents, and has no fallback. The configured paper account must be dedicated and free of non-Protocol101 positions, orders, executions, commissions, cash flows, and adjustments; failure blocks all entries. Restart sources and the exact reconstructed fields are explicit.

Conclusion: **ANSWERED**. The prior residual MATERIAL no longer exists.

### S3-03 — Historical-to-IBKR transfer was a Boolean rather than a gate

Prior locator/severity/disposition: `joined_review.md:855`; original **BLOCKING**; rerun001 disposition `ANSWERED as a measurement contract`, subject to the separate circular-order finding.

Repair inspected: `v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/calibration_spec.json:/source_transfer_gate`; `v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/shadow_sufficiency_spec.json:/historical_to_IBKR_source_transfer` and `/graph_topology`; `v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/evidence_standard.json:/source_transfer_graph_topology`.

Independent verification: the population, pair key, minimums, natural strata, exact/FP32 tolerances, action-agreement threshold, p95 score-gap artifact, and pass/insufficient/fail/invalid outcomes remain machine-specified. The final repair also makes FT2-92 the producer of this measurement rather than requiring it before entry to FT2-92.

Conclusion: **ANSWERED**.

### S3-04 — IBKR identity and ATM recentering were underdefined

Prior locator/severity/disposition: `joined_review.md:876`; original **BLOCKING**; rerun001 disposition `ANSWERED`.

Repair inspected: `v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/identity_mapping_spec.json:/source_neutral_contract_key`, `/canonical_contract_id`, `/historical_alias_record`, `/ibkr_alias_record`, `/ibkr_definition_fail_closed`, `/atm_and_ladder`, `/recentring`, and `/required_acceptance_fixtures`; `storage_spec.json:/contract_path_key`, `/label_join_key`, and `/key_validation`.

Independent verification: source-neutral identity, IBKR aliases, class/expiry/right/multiplier/currency fail-closed checks, half-even ATM rounding, exact 42-slot order, identity-keyed history, and no slot inheritance are explicit.

Conclusion: **ANSWERED**.

### S3-05 — Tensor field semantics and the E-family contract axis were undefined

Prior locator/severity/disposition: `joined_review.md:897`; original **BLOCKING**; rerun001 disposition `ANSWERED at the frozen-design level`.

Repair inspected: `v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/tensor_schema.json:/market_history`, `/contract_path_core`, and `/exact_contract_path_arrays`; `field_semantics_manifest.json:/completed_minute_law`, `/market_context_axis`, `/contract_axis`, and `/ft2_20_legacy_field_crosswalk`; `synthetic_golden_vectors.json:/vectors`.

Independent verification: contract-invariant fields and contract-dependent alignment/D/E fields occupy separate axes; delta/gamma are exact-contract internally computed values; all 19+4 legacy reviewed fields map with no unmapped fields; raw vendor Greeks are forbidden; acceptance vectors cover call/put, timing, masks, recentering, and fee-path boundaries.

Conclusion: **ANSWERED**.

### S3-06 — Complete-ladder safety was absent from action-mask assembly

Prior locator/severity/disposition: `joined_review.md:919`; original **BLOCKING**; rerun001 disposition `ANSWERED`.

Repair inspected: `tensor_schema.json:/flat_state_masks`; `composer_spec.json:/composer_order` and `/complete_ladder_and_safety`; `shadow_sufficiency_spec.json:/runtime_integrity`.

Independent verification: one missing/stale flat-entry ladder slot makes all 42 BUY intents false while WAIT stays true. The candidate, P5, controls, and future live guard consume the same time-t physical/safety mask. Open-position safety remains identity-specific and is not disabled by an unrelated incomplete ladder.

Conclusion: **ANSWERED**.

### S3-07 — Recentered open-state rows lost the open contract

Prior locator/severity/disposition: `joined_review.md:937`; original **BLOCKING**; rerun001 disposition `ANSWERED`.

Repair inspected: `tensor_schema.json:/open_state/dedicated_open_contract_record`; `storage_spec.json:/partitions` entries `open_state_scaffolds` and `open_contract_paths`; `replay_authority_v5_1_spec.json:/required_v5_1_behavior/open_contract` and `/required_boundary_fixtures`; `synthetic_golden_vectors.json` vector `open_contract_outside_recentered_ladder`.

Independent verification: the open contract has a dedicated source-neutral identity, clocks, exact quotes, and 90-minute path independent of the current flat ladder, with explicit outside-ladder replay and acceptance fixtures.

Conclusion: **ANSWERED**.

### S3-08 — Admitted-price activation lacked the required signed amendment

Prior locator/severity/disposition: `joined_review.md:955`; original **BLOCKING**; rerun001 disposition `ANSWERED`.

Repair inspected: `tensor_schema.json:/admitted_price_extension`; the owner-signed scope in `PROTOCOL101_SCOPED_SYNCHRONIZATION_DECISION_2026_07_25.md`.

Independent verification: extensions begin inactive and require FT2-25 producer pass, FT2-26 independent acceptance, an actually owner-signed synchronization-amendment hash, and candidate-bundle pinning. Missing signature fails closed, consistent with the signed 17-feature quarantine.

Conclusion: **ANSWERED**.

### S3-09 — Completed-minute, freshness, and opening-history laws were non-executable

Prior locator/severity/disposition: `joined_review.md:971`; original **MATERIAL**; rerun001 disposition `ANSWERED`, subject to its separate 15:55 finding.

Repair inspected: `field_semantics_manifest.json:/completed_minute_law`, `/quote_completeness_and_freshness`, and `/opening_and_reconnect_law`; `tensor_schema.json:/decision_convention` and `/flat_state_masks`.

Independent verification: interval endpoints, option/context cutoffs, decision/fill clocks, 90-second age, same-state bid/ask, exact identity, 15 consecutive complete minutes, reconnect reset, no prior-session backfill, and open-position behavior during reset are explicit.

Conclusion: **ANSWERED**. The earlier MATERIAL is fully repaired, including the separately verified 15:55 law below.

### S3-10 — Primary entry population was incompatible with eventual live occupancy/D49

Prior locator/severity/disposition: `joined_review.md:989`; original **MATERIAL**; rerun001 disposition `PARTIALLY ANSWERED`, residual **MATERIAL** because the neutral hold-to-15:55 game was not the later learned-lifecycle game.

Repair inspected: `objective_spec.json:/serial_primary_population`, `/entry_component_freeze/source_transfer_topology`, and `/entry_component_freeze/combined_lifecycle_boundary`; `evidence_standard.json:/two_layer_acceptance`; `mde_spec.json:/serial_trade_count_projection`.

Independent verification: the one-trade neutral lifecycle is now explicitly only an entry-component quality/safety game. It makes no live-parity or incremental-edge claim. The assembled learned entry/lifecycle trader, including earlier exits, re-entry, and its own D49/occupancy path, must be evaluated on complete combined-system ledgers at FT2-80 before a promotable claim.

Conclusion: **ANSWERED** by a clear claim boundary and required downstream combined replay. The prior residual MATERIAL no longer attaches.

### FRESH-S3-B1 — Decision-time masks used unknowable t+1 asks

Prior locator/severity/disposition: `joined_review.md:1015`; **BLOCKING**, unresolved fresh finding.

Repair inspected: `intent_fill_recheck_law.json:/decision_time_intent`, `/fill_time_recheck`, `/label_oracle_census_semantics`, and `/historical_live_guard_equivalence`; `account_state_ledger_spec.json:/d48`, `/d49`, and `/rejected_fill_accounting`; `composer_spec.json:/complete_ladder_and_safety`; `objective_spec.json:/serial_primary_population/entry_commit_and_fill`.

Independent verification: action composition uses only `A_t`; `A_(t+1)` existence, price, freshness, and outcome are expressly forbidden. Only the committed identity is rechecked at t+1. Rejection opens no position, charges no premium/fee, changes no PnL, allows no substitute, and resumes no earlier than t+2. Historical, source-plane, and live-guard arithmetic/effect must be identical.

Conclusion: **ANSWERED**.

### FRESH-S3-B2 — The $103 shortcut did not implement signed A3

Prior locator/severity/disposition: `joined_review.md:1041`; **BLOCKING**, unresolved fresh finding.

Repair inspected: `intent_fill_recheck_law.json:/fee_paths` and `/soft_close`; `account_state_ledger_spec.json:/d49`; `objective_spec.json:/serial_primary_population/account_and_safety_state`.

Independent verification: soft close is the absence of an otherwise-eligible current-ladder contract with `A_t>=100` cents and passing D48/D49 intent masks. The arithmetic bound is computed per fee path: 10,300 cents for fee_3 and 10,400 cents for fee_4. A shared literal is forbidden, and the ladder-existence predicate—not the threshold alone—owns the state change.

Conclusion: **ANSWERED**.

### FRESH-S3-B3 — Source transfer was required before its graph node produced it

Prior locator/severity/disposition: `joined_review.md:1063`; **BLOCKING**, unresolved fresh finding.

Repair inspected: `evidence_standard.json:/source_transfer_graph_topology`; `shadow_sufficiency_spec.json:/graph_topology`, `/historical_to_IBKR_source_transfer/activation_boundary`, and `/candidate_state_before_FT2_93_live_shadow`; Graph V2 nodes/edges for FT2-91, FT2-92, and FT2-93.

Independent verification: Graph V2 and both repaired contracts agree: FT2-91 pass enters FT2-92 with `transfer_not_yet_run`; FT2-92 performs the measurement and emits its outcome; only FT2-92 pass reaches FT2-93. There is no circular precondition.

Conclusion: **ANSWERED**.

### FRESH-S3-B4 — Schema exposed an impossible learned EXIT at 15:55

Prior locator/severity/disposition: `joined_review.md:1080`; **MATERIAL**, unresolved fresh finding.

Repair inspected: `tensor_schema.json:/open_state`; `field_semantics_manifest.json:/completed_minute_law`; `storage_spec.json:/partitions/open_state_scaffolds`; `synthetic_golden_vectors.json` vector `terminal_minute_has_no_learned_exit`.

Independent verification: 15:54 is the last learned EXIT decision and may fill once at 15:55. `actions_at_1555` is empty; 15:55 is `FORCED_FLAT_TERMINAL`, model invocation is false, and a 15:54 fill suppresses duplicate forced flat.

Conclusion: **ANSWERED**. No residual MATERIAL.

### FRESH-S3-B5 — Neutral serial game conflicted with census trade-rate selection

Prior locator/severity/disposition: `joined_review.md:1094`; **MATERIAL**, unresolved fresh finding.

Repair inspected: `calibration_spec.json:/joint_conservatism/selection_order`, `/joint_conservatism/census_v3_trade_rate_provenance`, `/joint_conservatism/design_trade_rate_band_mean_trades_per_session`, and `/joint_conservatism/no_setting_in_band`; `mde_spec.json:/serial_trade_count_projection`.

Independent verification: census activity is report-only. It cannot discard, retain, rank, select, calibrate, fail, or tie-break a component setting. A diagnostic band mismatch does not alter the frozen inner objective. Neutral-lifecycle projections are explicitly clipped to one successful trade per session and cannot multiply census multi-trade rates.

Conclusion: **ANSWERED**. No residual MATERIAL.

### FRESH-S3-B6 — One-strike substitute cluster could deadlock a smooth surface

Prior locator/severity/disposition: `joined_review.md:1118`; **MATERIAL**, unresolved fresh finding.

Repair inspected: `composer_spec.json:/uncertainty_wait/directional_substitute_cluster` and `/composer_order`; `calibration_spec.json:/uncertainty`.

Independent verification: membership now spans the same expiry/right within ±10 strike points, two standard ladder steps on either side (up to five centered strikes). Inside-cluster substitutes do not become the WAIT comparator. Exact-contract regret remains a separate hard q90 action constraint.

Conclusion: **ANSWERED** for the recorded smooth-three-or-more-strike failure mode. No residual MATERIAL.

### FRESH-S3-B7 — MNAR stability omitted the alternate refit law

Prior locator/severity/disposition: `joined_review.md:1134`; **MATERIAL**, unresolved fresh finding.

Repair inspected: `evidence_standard.json:/mnar_no_bid_sensitivity/alternate_refit_protocol` and `/hard_ranking_stability`; `terminal_decision_spec.json:/statistic_specific_sufficiency/MNAR_ranking_stability`.

Independent verification: every label-dependent forecast head, CDF, calibrator, threshold, WAIT/regret action head, inner selection, and ensemble must be refit independently under the alternate label view. Cross-view reuse is forbidden; features, architecture/grid, roles, seeds, loss, comparators, serial game, tie-breaks, multiplicity, and resolver remain fixed. Separate hashes and traces are required.

Conclusion: **ANSWERED**. No residual MATERIAL.

### FRESH-S3-B8 — Session-start live equity could be stale or unrelated

Prior locator/severity/disposition: `joined_review.md:1158`; **MATERIAL**, unresolved fresh finding and residual of S3-02.

Repair inspected: `account_state_ledger_spec.json:/session_start/live_paper_value`, `/session_start/account_scope`, `/session_start/snapshot_fields_logged`, and `/fail_closed_conditions`.

Independent verification: exact session-date freshness, deterministic selection, receipt deadline, account dedication, external-state exclusions, authenticated verification sources, cent normalization, and fail-closed handling are all frozen.

Conclusion: **ANSWERED**. No residual MATERIAL.

### FRESH-S3-MINOR-RECEIPT — Receipt authority-field naming differed

Prior locator/severity/disposition: `joined_review.md:75`; **MINOR**, disclosure-only and unresolved.

Repair inspected: `receipt.json:/product_contract_hash` in FT2-08, FT2-10, and FT2-11, plus FT2-04/05 receipts and the aggregate receipt.

Independent verification: all current v3 node receipts use `product_contract_hash` with the same pinned authority value, and all observed receipt/deliverable hashes match.

Conclusion: **ANSWERED**.

## B. FRESH SCAN

### B1. Intent mask and fill-recheck law end to end

Evidence: `intent_fill_recheck_law.json:/decision_time_intent` defines `intent_cost=A_t_quote_cents*100+fee`; `/fill_time_recheck` defines `fill_cost=A_(t+1)_quote_cents*100+fee`; `/fill_time_recheck/failure` and `account_state_ledger_spec.json:/rejected_fill_accounting` freeze zero position, premium, fee, and PnL with no substitution; `/historical_live_guard_equivalence` requires identical historical, source-plane, and live-guard integer arithmetic and chooses invalid/WAIT on mismatch. FT2-10 candidate/P5/random consumers pin the same law hash. The golden vector `intent_pass_fill_recheck_rejects_without_charge` checks an exact one-cent D48 crossing.

Conclusion: coherent causal two-stage law; no defect.

### B2. D48 reference population 159,312 → 121,553

Evidence and arithmetic:

- `v2_v3_impact.json:/scalar_changes/label_rows_all_governed_candidates`: 460,937 in both v2 and v3.
- `v2_v3_impact.json:/scalar_changes/label_rows_d48_reference`: 159,312 to 121,553, delta −37,759, or `37,759/159,312 = 23.7013%`.
- `v2_v3_impact.json:/v3_intent_and_recheck_counts`: 128,758 time-t intent-eligible rows and 7,205 t+1 rejected rows.
- `label_build_compute.json:/per_session`: for every one of the 45 sessions, `reference_d48_rows = intent_eligible_fee3_rows - fill_recheck_rejected_fee3_rows`; aggregate `128,758 - 7,205 = 121,553`.
- The net change can be written `159,312 - 128,758 = 30,554`, followed by 7,205 rejections, and `30,554 + 7,205 = 37,759`.

The new definition therefore explains the v3 endpoint and preserves every governed label row, so the change is a legality/reference-population reclassification rather than physical deletion. But it does **not** evidence the entire v2-to-v3 transition. No inspected top-level artifact lists the v2 and v3 row identities side by side, separates gross removals from gross additions, or reports the outcome/economic distribution of the 30,554 net time-t-intent change. The top-level `fill_recheck_rejections.csv` contains 247 selected P5 policy/variant rejection events (248 lines including its header), not the 7,205 label-reference rejections and not the 37,759 transition rows. Aggregate oracle stability cannot prove that economically meaningful individual rows were not removed from the reference population.

Conclusion: **MATERIAL, not-acceptable** fresh finding `LP-RERUN002-01`. The current v3 population is internally defined, but the approximately 24% transition is not independently auditable for row identity or economics from the allowed final packet.

### B3. Parametric soft-close floor on both fee paths

Evidence: `intent_fill_recheck_law.json:/fee_paths` computes `100*100+active_fee`, yielding 10,300 and 10,400 cents; `/soft_close` owns the current-ladder existence predicate and prohibits a shared literal; `account_state_ledger_spec.json:/d49`, `objective_spec.json:/serial_primary_population/account_and_safety_state`, `replay_authority_v5_1_spec.json:/fee_trajectories`, and FT2-11 signed-safety text agree. The synthetic boundary vector has 10,300 remaining and correctly leaves fee_3 open while soft-closing fee_4.

Conclusion: repaired correctly; no defect.

### B4. One canonical matched-random comparator hash

Evidence: FT2-10 `matched_random_generator_spec.json` observed SHA-256 is `8bdbe8beb4734852526cd2981be76098f792d5c4b9fee30f89f8a2d952abf754`. FT2-10 receipt and controls pin it. FT2-11 `multiplicity_spec.json:/frozen_nonselectable_matched_random_algorithm` pins that exact path/hash, declares it the sole source, forbids local override and candidate-specific seed namespaces, and imports the exact canonical key. FT2-11 receipt and contract carry the same hash.

Conclusion: one canonical generator; no defect.

### B5. Independent SE_h_star arithmetic

For the invented observed series `[1,2,3,4,5]`, `n=5`, `L=2`, and `K=4`:

- delete-block means: `(3+4+5)/3=4`, `(1+4+5)/3=10/3`, `(1+2+5)/3=8/3`, `(1+2+3)/3=2`
- their average: `(4+10/3+8/3+2)/4=3`
- squared-deviation sum: `(1)^2+(1/3)^2+(-1/3)^2+(-1)^2=20/9`
- factor: `(n-L)/(L*K)=3/(2*4)=3/8`
- variance: `(3/8)*(20/9)=5/6=0.8333333333333334`
- `SE_h=sqrt(5/6)=0.9128709291752769`
- observed mean `=3`, so `t_h=3/SE_h=3.2863353450309964`

For centered replicate `[-2,-2,0,0,2]`:

- delete-block means: `2/3`, `0`, `-2/3`, `-4/3`
- their average: `-1/3`
- deviations are `1,1/3,-1/3,-1`, so the squared-deviation sum is again `20/9`
- replicate variance is again `5/6`; `SE_h_star_b=0.9128709291752769`
- replicate mean `=(-2-2+0+0+2)/5=-0.4`
- `t_h_star_b=-0.4/SE_h_star_b=-0.4381780460041329`

These independently recomputed values match `bootstrap_spec.json:/worked_studentization_example` and the consistency-checker fixture. The replicate SE is mechanically recomputed rather than reusing the observed SE.

Conclusion: repaired and reproducible; no defect.

### B6. Source-transfer ordering versus Graph V2

Evidence: Graph V2 edges are FT2-91 pass → FT2-92 and FT2-92 pass → FT2-93. `evidence_standard.json:/source_transfer_graph_topology`, `shadow_sufficiency_spec.json:/graph_topology`, and `calibration_spec.json:/source_transfer_gate/graph_order` match exactly. `transfer_not_yet_run` is legal on entry to FT2-92; a pass is required only to leave FT2-92.

Conclusion: no circularity; no defect.

### B7. 15:55 terminal action law

Evidence: `tensor_schema.json:/open_state/actions_at_1555=[]`; `/terminal_minute_1555` says model not invoked and no learned action; `field_semantics_manifest.json:/completed_minute_law` makes 15:54 the last learned EXIT and 15:55 its fill/forced-flat boundary; storage and golden-vector fixtures agree and suppress duplication.

Conclusion: no defect.

### B8. Deterministic session-start equity primitive

Evidence: `account_state_ledger_spec.json:/session_start` freezes timezone, exact boundary, selection interval, greatest-source-timestamp rule, tie-break, receipt deadline, maximum age, cent normalization, no fallback, account dedication, external-state exclusions, logged primitives, and block behavior.

Conclusion: deterministic and fail-closed; no defect.

### B9. Near-close full-window CDF filter

Evidence: `objective_spec.json:/nested_empirical_cdf/full_window_filter` requires uncensored exact N-mark finite horizons and a complete remaining-session grid through exact 15:55; finite but censored/shortened near-close values are excluded from every continuous empirical-CDF reference. `forecast_heads.json:/full_window_training_filter` independently requires the complete uncensored window and restricts right-censored first-profit rows to the discrete-survival likelihood.

Conclusion: no defect.

### B10. Substitute-cluster breadth

Evidence: `composer_spec.json:/uncertainty_wait/directional_substitute_cluster` is same expiry/right and ±10 points, two ladder steps on each side. The selected cluster uses its maximum score; inside-cluster substitutes cannot force WAIT as the outside comparator. Exact-contract regret is separately bounded by the hard q90 `<=0.10` action rule.

Conclusion: the rerun001 smooth three-or-more-strike defect is repaired; no new breadth defect established.

### B11. Phase-F hard reporting floor

Evidence: `bootstrap_spec.json:/phase_F_shadow_intervals` requires at least 45 complete sessions for hard clustered bounds; sessions 1–44 allow only point estimates and raw counts and route insufficient evidence. `shadow_sufficiency_spec.json:/minimum_no_order_live_shadow_evidence` repeats the 45-session floor. The synthetic AR coverage design explicitly includes 45-session fold shapes.

Conclusion: hard reporting is no longer allowed below the demonstrated floor; no defect.

### B12. MNAR alternate refit

Evidence: `evidence_standard.json:/mnar_no_bid_sensitivity/alternate_refit_protocol` says `refit_required=true`, enumerates every label-dependent artifact to refit, freezes all cross-view non-label inputs, forbids cross-view reuse, and requires separate hashes/traces. `terminal_decision_spec.json:/statistic_specific_sufficiency/MNAR_ranking_stability` requires that complete refit before interpreting identity, terminal, pairwise order, or Spearman stability.

Conclusion: unambiguous full alternate refit; no defect.

### B13. Census v3 internal consistency, ceilings, P5, and the 1.6% figure

Internal count law:

- all governed rows: 460,937
- time-t intent-eligible fee_3 rows: 128,758
- t+1 rejected fee_3 rows: 7,205
- successful D48-reference labels: 121,553
- `128,758-7,205=121,553`, and the same equality holds in each of 45 session summaries

The stated approximately 1.6% rejection figure is reproducible only with **all governed candidate rows** as denominator: `7,205/460,937=1.5631%`. It is **not** the conditional fill-recheck rejection rate among time-t eligible intents, which is `7,205/128,758=5.5958%`. Any use of “1.6% fill-recheck rejection” must name the all-row denominator.

Ceiling/P5 reconciliation from `v2_v3_impact.json:/oracle_changes` and `oracle_ceiling_summary.csv`:

- Oracle deltas are small relative to their levels because the hindsight oracle ranks successful legal realized-entry paths and records zero selected rejections in each variant. For example, `best_session` moves only −$1,060 with unchanged 108 trades; hold-to-forced-flat moves −$2,130 with unchanged 45 trades.
- P5 is causal and cannot screen on t+1. Its variant ledgers therefore record selected fill rejections and altered later occupancy. Seven of nine P5 variant PnLs decline (for example best_3 −$17,912, best_5 −$12,662, best_10 −$12,179, and best_session −$2,822); first-real-profit and hold-to-forced-flat increase, so “P5 down” is a majority-pattern summary rather than a universal statement.
- `fill_recheck_rejections.csv` has 247 selected P5 policy/variant events, which reconciles the variant `rejected_fill_count` fields in `oracle_ceiling_summary.csv`; it is a different population from the 7,205 label-reference rejections.

Conclusion: the v3 endpoint, oracle/P5 behavior, and both rejection denominators are internally coherent. The unresolved row-transition/economic audit is the same MATERIAL finding recorded in B2, not a second finding.

Additional live-parity scan: exact IBKR identity, recentering, complete-ladder WAIT, dedicated outside-ladder open state, 90-second freshness, reconnect reset, label firewall, separate policy/fee ledgers, t+1 rejection handling, and v5.1 activation gates were checked. No additional live-parity defect was found.

## FINDINGS

1. `LP-RERUN002-01` — **MATERIAL — not-acceptable**.
   - Affected contract(s): primary `v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/` v3 census evidence; downstream census-pinning/consuming portions of FT2-08 data/tensor/label, FT2-10 entry science, and FT2-11 evidence/statistics.
   - Evidence: `v2_v3_impact.json:/scalar_changes` reports 159,312→121,553 and an unchanged 460,937-row governed corpus; `/v3_intent_and_recheck_counts` reports 128,758 intents and 7,205 rejections; `label_build_compute.json:/per_session` proves `intent-rejection=reference` in every session; the top-level `fill_recheck_rejections.csv` contains only 247 selected P5 variant events. No inspected top-level artifact supplies a row-identity v2/v3 transition crosswalk, gross removal/addition counts, or label/economic distribution for the 30,554 net time-t-intent change or the full 37,759-row reference change.
   - Why material/not-acceptable: a 23.7% reference-population change cannot be independently checked for silent economic selection from aggregate counts and near-flat oracle summaries alone. The current v3 law is coherent, but the final packet does not answer the mandated question of which rows changed and what economic content left or entered the reference population.

SEAT ROUTING RECOMMENDATION: STOP-REDESIGN-REQUIRED
