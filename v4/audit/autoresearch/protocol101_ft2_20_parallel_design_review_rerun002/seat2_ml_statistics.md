# Protocol101 FT2-20 Final Re-Review Rerun002 — Seat 2 ML/Statistics

Review mode: fresh isolated, evidence-only, no repair authority.

Pinned authority SHA-256 independently verified:
`2363d3f986daba20bd5087ed751dc5b2d839e76cd6413aeca0bcd255eb98857a`.
Final aggregate receipt SHA-256 independently verified:
`19b74e513f0c6c68b50167b37e4bcfd40cb4ccd0f3c2240201ff3cbbd178f4f3`.
The FT2-08/10/11 v3 receipt hashes are respectively
`731cc6fb4c44bd0650d658e71fffbdac7c3e0f5d68701b2e05b8576b014b078c`,
`c04a591fd3034a2caba756e8f9dc1d7f5c24c397170894b0b6b0eca7a083fa6f`,
and `bdb52c4f9178da1cab75ec4a80bd13c1db8c0074c1c398f380729d1df80c5ef9`.
Every top-level deliverable named by those three receipts independently matched
its recorded hash. The FT2-05 v3 receipt and v2-v3 impact hashes also matched
the aggregate receipt.

Files/categories checked were limited to the allowlist:

- The consolidated authority, Graph V2 JSON, all five signed contracts, and
  `v4/model/protocol101_serial_simulator_v5.py`.
- FT2-04 top-level `label_spec.json`, `oracle_rules.json`,
  `census_sessions.json`, `intersection_proof.json`, `report.md`, and
  `receipt.json`.
- FT2-05 top-level census summary/report/receipt and impact artifacts, plus the
  top-level ceiling, fill-recheck, guardrail-rate, MNAR, variance, and MDE
  tables needed for the requested reconciliation.
- All top-level FT2-08, FT2-10, and FT2-11 contract/JSON receipt deliverables
  relevant to causality, population, targets, calibration, composer,
  multiplicity, bootstrap, terminal, and Phase-F laws.
- Original FT2-20 seat-2 report and receipt; rerun001 seat-2 report, joined
  review, and receipt; final-repair unified crosswalk, aggregate receipt,
  consistency-checker output, and checker source.

All allowlisted packet directories were enumerated at top level only. I did not
enter, recursively list, search, or read any `superseded/` directory. I did not
inspect any unassigned scratch path or the future rerun002 destination.

## A. FINDINGS VERIFICATION

The identifiers and locators below are from the rerun001 joined review and the
unified crosswalk. A producer crosswalk claim was used only to locate the
purported repair.

1. **S2-01, `joined_review.md:431` — future `target_valid` controls the
   composer.** Prior severity/disposition: **BLOCKING, ANSWERED** in rerun001;
   final crosswalk claims ANSWERED. Repair inspected:
   `FT2-10/composer_spec.json:/causal_horizon_availability` forbids
   `target_valid`, label presence, future bid availability, and future
   censoring; `FT2-08/label_join_spec.json:/runtime_label_firewall` requires
   bit-identical actions with label partitions removed. Independent conclusion:
   **ANSWERED**.

2. **S2-02, `joined_review.md:446` — census consumes the first fold embargo.**
   Prior severity/disposition: **BLOCKING, ANSWERED**; final crosswalk claims
   ANSWERED. Repair inspected: `FT2-08/fold_roles.json:/census` records 45
   sessions, all five embargo dates excluded, and zero embargo/outer-test/
   holdout intersections; `FT2-04/receipt.json:/intersection_counts` records
   the same zero intersections and complete role equation. Independent
   conclusion: **ANSWERED**.

3. **S2-03, `joined_review.md:461` — `DeltaQ` replaces signed incremental
   profit.** Prior severity/disposition: **BLOCKING, ANSWERED**; final crosswalk
   claims ANSWERED. Repair inspected:
   `FT2-10/objective_spec.json:/entry_component_freeze` makes `DeltaQ`
   diagnostic and requires serial-dollar no-harm versus P5;
   `FT2-11/evidence_standard.json:/two_layer_acceptance` reserves strict
   incremental-dollar claims versus P5 and matched random for the assembled
   FT2-80 trader, matching the signed D1 amendment. Independent conclusion:
   **ANSWERED**.

4. **S2-04, `joined_review.md:477` — mandatory action-conditioned calibration
   gate is absent and unimplementable.** Prior severity/disposition:
   **BLOCKING, ANSWERED** in rerun001; final crosswalk claims ANSWERED. Repair
   inspected:
   `FT2-10/forecast_heads.json:/action_conditioned_outputs`,
   `FT2-10/calibration_spec.json:/action_conditioned_gate`,
   `FT2-10/composer_spec.json:/mandatory_action_conditioned_gate`, and
   `FT2-11/evidence_standard.json:/two_layer_acceptance/entry_component_acceptance/all_required/action_conditioned_calibration`.
   Heads, losses, calibration thresholds, and an action constraint now exist,
   but target construction is still mechanically circular and underdefined:
   the regret heads are scoped to rows where “the frozen composer” selects
   ENTER, while the only frozen composer requires the already-trained and
   calibrated q90 regret head to decide ENTER. The training sequence says to
   run an undefined “frozen realized-label audit composer” before fitting the
   action heads; no provisional pre-regret composer, staged fit, or fixed-point
   algorithm is specified anywhere in FT2-10/11. The WAIT target depends on
   that same undefined audit composer. Independent conclusion:
   **NOT ACTUALLY ANSWERED**. Affected contracts: FT2-10 and its FT2-11
   acceptance consumer.

5. **S2-05, `joined_review.md:493` — MDE unsound in estimand, dependence, and
   trade count.** Prior severity/disposition: **BLOCKING, PARTIALLY ANSWERED**;
   final crosswalk claims ANSWERED. Repair inspected:
   `FT2-11/mde_spec.json:/estimands`, `/dependence_adjusted_MDE`,
   `/serial_trade_count_projection`, and `/pre_tranche_1_pilot`, plus
   `bootstrap_spec.json:/studentized_maxT/standard_error_authority`.
   Candidate-minus-comparator session-dollar deltas, statistic-specific
   SD/LRV/ESS, a one-trade/session cap, and block-power with mechanically
   defined replicate SE are now required. Independent conclusion:
   **ANSWERED**.

6. **S2-06, `joined_review.md:510` — terminal outcomes mechanically
   undecidable.** Prior severity/disposition: **BLOCKING, PARTIALLY ANSWERED**;
   final crosswalk claims ANSWERED. Repair inspected:
   `FT2-11/terminal_decision_spec.json:/adjusted_bound_authority`, `/resolver`,
   and `/statistic_specific_sufficiency`, plus
   `worked_terminal_examples.json:/examples`. Lower/upper bounds, precedence,
   sufficiency mappings, and deterministic fixtures are frozen. Independent
   conclusion: **ANSWERED** for the entry-evidence resolver. The separate
   Phase-F graph routing defect is recorded in Section B.

7. **S2-07, `joined_review.md:525` — empirical CDF under-specified and not
   nested.** Prior severity/disposition: **MATERIAL, ANSWERED**; final crosswalk
   claims ANSWERED. Repair inspected:
   `FT2-10/objective_spec.json:/nested_empirical_cdf` freezes fit-only
   reference populations, equal-session weighting, midrank/interpolation,
   support minima, same-band fallback, fail-closed behavior, and separate
   inner/final scopes. Independent conclusion: **ANSWERED; no residual MATERIAL
   finding**.

8. **S2-08, `joined_review.md:543` — OOF conformal residuals reused for a
   refitted model.** Prior severity/disposition: **MATERIAL, ANSWERED**; final
   crosswalk claims ANSWERED. Repair inspected:
   `FT2-10/calibration_spec.json:/final_outer_fit_and_disjoint_calibration`
   freezes the final model before a disjoint chronological calibration slice
   and forbids post-calibration refit. Independent conclusion: **ANSWERED; no
   residual MATERIAL finding**.

9. **S2-09, `joined_review.md:557` — dependence machinery lacks a validity
   condition.** Prior severity/disposition: **MATERIAL, PARTIALLY ANSWERED**;
   final crosswalk claims ANSWERED. Repair inspected:
   `FT2-11/bootstrap_spec.json:/moving_block_bootstrap`,
   `/synthetic_AR_coverage_demonstration`, and `/phase_F_shadow_intervals`.
   Noncircular 3/5/8-session envelopes, a preregistered coverage demonstration,
   and session-cluster Phase-F inference with a 45-session hard floor are now
   explicit. Independent conclusion: **ANSWERED; no residual MATERIAL
   statistical-method finding**. Graph routing of the below-floor state remains
   defective separately.

10. **S2-10, `joined_review.md:572` — multiplicity omits selectable controls.**
    Prior severity/disposition: **MATERIAL, PARTIALLY ANSWERED**; final crosswalk
    claims ANSWERED. Repair inspected:
    `FT2-10/matched_random_generator_spec.json`,
    `FT2-10/controls_spec.json:/frozen_nonselectable_random_controls`, and
    `FT2-11/multiplicity_spec.json:/frozen_nonselectable_matched_random_algorithm`
    plus `/family_counting`. All eight fixed attempts, failures, aggregate,
    shuffle contrasts, selectable channel sets, and action-gate statistics are
    registered under one canonical source hash. Independent conclusion:
    **ANSWERED; no residual MATERIAL finding**.

11. **S2-11, `joined_review.md:589` — no-bid censoring is MNAR and rewards
    illiquidity.** Prior severity/disposition: **MATERIAL, ANSWERED**; final
    crosswalk claims ANSWERED. Repair inspected:
    `FT2-04/label_spec.json:/missing_minute_rule` and
    `FT2-11/evidence_standard.json:/mnar_no_bid_sensitivity`, including
    `/alternate_refit_protocol`. Full-loss is primary and every
    label-dependent artifact is independently refit under the alternate label
    view, with selected identity, terminal, pairwise ordering, and Spearman
    stability gates. Independent conclusion: **ANSWERED; no residual MATERIAL
    finding**.

12. **FRESH-S2-B1, `joined_review.md:607` — D48/D49 decision masks use the
    future t+1 quote.** Prior severity/disposition: **BLOCKING fresh finding**;
    final crosswalk claims ANSWERED. Repair inspected:
    `FT2-08/intent_fill_recheck_law.json:/decision_time_intent` and
    `/fill_time_recheck`, `FT2-10/composer_spec.json:/complete_ladder_and_safety`,
    and `FT2-10/objective_spec.json:/serial_primary_population/entry_commit_and_fill`.
    Candidate, P5, and live-style controls are now causal at t and recheck only
    the selected identity at t+1. Independent conclusion: **ANSWERED for the
    candidate/P5/live-style finding**. A stale FT2-04 oracle contract conflict
    is a separate fresh finding in Section B.

13. **FRESH-S2-B2, `joined_review.md:625` — D49 soft close violates A3 and fee
    paths.** Prior severity/disposition: **BLOCKING fresh finding**; final
    crosswalk claims ANSWERED. Repair inspected:
    `FT2-08/intent_fill_recheck_law.json:/soft_close` and `/fee_paths`,
    `account_state_ledger_spec.json:/d49`, and
    `FT2-10/objective_spec.json:/serial_primary_population/account_and_safety_state`.
    The current-ladder existence predicate is restored and $103/$104 are
    separately computed. Independent conclusion: **ANSWERED**.

14. **FRESH-S2-B3, `joined_review.md:639` — FT2-10/11 freeze different random
    generators.** Prior severity/disposition: **BLOCKING fresh finding**; final
    crosswalk claims ANSWERED. Repair inspected:
    `FT2-10/matched_random_generator_spec.json` (SHA-256
    `8bdbe8beb4734852526cd2981be76098f792d5c4b9fee30f89f8a2d952abf754`)
    and `FT2-11/multiplicity_spec.json:/frozen_nonselectable_matched_random_algorithm`.
    FT2-11 references that exact source/hash and forbids any local namespace.
    Independent conclusion: **ANSWERED**.

15. **FRESH-S2-B4, `joined_review.md:653` — `SE_h_star` undefined.** Prior
    severity/disposition: **BLOCKING fresh finding**; final crosswalk claims
    ANSWERED. Repair inspected:
    `FT2-11/bootstrap_spec.json:/studentized_maxT/standard_error_authority` and
    `/worked_studentization_example`. The same direct contiguous delete-L
    jackknife is explicitly applied inside every outer replicate with fixed raw
    fold weights; observed-SE reuse and nested bootstrap are forbidden.
    Independent conclusion: **ANSWERED**, including the independent arithmetic
    reproduction in Section B.

16. **FRESH-S2-B5, `joined_review.md:666` — WAIT never constrains exact-contract
    regret magnitude.** Prior severity/disposition: **MATERIAL fresh finding**;
    final crosswalk claims ANSWERED. Repair inspected:
    `FT2-10/composer_spec.json:/mandatory_action_conditioned_gate/selected_contract_regret_action_constraint`
    and `calibration_spec.json:/action_conditioned_gate/selected_contract_regret`.
    A finite same-final-model conformalized q90 bound `<=0.10` is now a hard
    ENTER condition. Independent conclusion: **ANSWERED as an action law; no
    residual MATERIAL magnitude-rule finding**. Its training target remains
    unimplementable under S2-04.

17. **FRESH-S2-B6, `joined_review.md:679` — Phase-F hard reporting below the AR
    demonstration sample size.** Prior severity/disposition: **MATERIAL fresh
    finding**; final crosswalk claims ANSWERED. Repair inspected:
    `FT2-11/bootstrap_spec.json:/phase_F_shadow_intervals` and
    `shadow_sufficiency_spec.json:/minimum_no_order_live_shadow_evidence`.
    Both require 45 complete sessions; sessions 1-44 are point-estimate-only.
    Independent conclusion: **ANSWERED as a statistical floor; no residual
    MATERIAL floor finding**. Its graph outcome is unroutable, separately
    BLOCKING in Section B.

18. **FRESH-S2-MINOR-RECEIPT, `joined_review.md:75` — receipt authority field
    names differ.** Prior severity/disposition: **MINOR fresh finding**; final
    crosswalk claims ANSWERED. Repair inspected:
    all three current v3 receipts use `product_contract_hash`, omit the former
    competing `authority_sha256` field, and reproduce every deliverable hash.
    Independent conclusion: **ANSWERED**.

## B. FRESH SCAN

1. **Intent mask/fill-recheck law end to end.** The repaired core law is exact:
   `intent_cost_cents=A_t_quote_cents*100+active_fee`; D48/D49 use only t and
   the causal policy ledger; one selected identity is rechecked at t+1; failure
   opens no position, charges no premium/fee, changes PnL by zero, forbids
   substitution, and next permits a decision at t+2.
   `intent_fill_recheck_law.json:/historical_live_guard_equivalence` requires
   identical historical, source-plane, and live-guard recomputation.

   The law is nevertheless not end-to-end unique. The still-current, receipt-
   pinned `FT2-04/oracle_rules.json:/census_layer_constraints` says D48/D49
   decision eligibility uses the actual `A_(t+1)` and
   `/oracle_sequencing` says an oracle may commit only when that future ask
   passes. `FT2-04/report.md` repeats actual-t+1 eligibility. FT2-05 v3 pins
   both this oracle-rule hash and the newer causal intent-law hash. The newer
   law permits hindsight outcome ranking but expressly forbids renaming a t+1
   outcome as the decision-time legality mask. Two pinned inputs therefore
   specify different oracle/census eligibility machines. This is a fresh
   **BLOCKING** reproducibility defect affecting FT2-04, FT2-05, and the
   FT2-08 intent-law boundary.

2. **D48-reference population 159,312 to 121,553.** Top-level totals fully
   reconcile the aggregate shift:

   ```text
   v2 reference rows                         159,312
   v3 causal t-intent eligible rows          128,758
   change from moving legality to t          -30,554
   explicit selected t+1 rejections           -7,205
   v3 successful realized-entry labels       121,553
   total change                              -37,759 = -23.701291%
   ```

   Thus the new two-population definition explains the entire numerical shift:
   `128,758 - 7,205 = 121,553` and
   `30,554 + 7,205 = 37,759`. The 30,554 rows are not legal time-t intents under
   the new rule; the 7,205 are retained explicitly as failed-fill events with
   no position, fee, or PnL. No economically realizable policy fill is silently
   dropped in the top-level accounting. The stale FT2-04 oracle definition in
   item 1 still prevents one unique contract-level reproduction.

3. **Parametric soft-close floor.** Independently:
   `$1.00 = 100 quote cents`; `100*100 + 300 = 10,300 cents ($103)` and
   `100*100 + 400 = 10,400 cents ($104)`.
   `intent_fill_recheck_law.json:/fee_paths`,
   `account_state_ledger_spec.json:/d49`, and both FT2-10/11 constant blocks
   compute these per active fee path and forbid a shared literal. Soft close
   uses absence of a qualifying current-ladder contract, not merely remaining
   cash below a constant. Conclusion: repaired.

4. **One canonical matched-random comparator.** The sole file hash is
   `8bdbe8beb4734852526cd2981be76098f792d5c4b9fee30f89f8a2d952abf754`.
   FT2-10 controls, FT2-11 multiplicity, both v3 receipts, and the aggregate
   receipt all reference it exactly; no candidate-specific namespace survives.
   Conclusion: repaired.

5. **Independent `SE_h_star` arithmetic.** For `[1,2,3,4,5]`, `L=2`, deleting
   contiguous pairs gives means `[4,10/3,8/3,2]`; their mean is `3`, squared
   deviations sum to `20/9`, and the factor is
   `(5-2)/(2*(5-2+1))=3/8`. Hence
   `V=(3/8)*(20/9)=5/6`,
   `SE=sqrt(5/6)=0.912870929175277`, and
   `t=3/SE=3.286335345031`.

   For `[-2,-2,0,0,2]`, delete-pair means are
   `[2/3,0,-2/3,-4/3]`, their mean is `-1/3`, and their squared deviations
   again sum to `20/9`. Thus `V*=5/6`,
   `SE*=0.912870929175277`, replicate mean `=-2/5=-0.4`, and
   `t*=-0.4/SE*=-0.438178046004133`. These independently reproduce the
   contract fixture.

6. **Source-transfer ordering versus Graph V2.** Ordering itself now matches:
   Graph V2 routes FT2-91 pass to FT2-92 and FT2-92 pass to FT2-93, while
   FT2-11 permits `transfer_not_yet_run` on entry to FT2-92.

   The emitted outcome vocabulary does not match the graph. Graph V2 gives
   FT2-92 only `pass` and `fail` edges, but
   `evidence_standard.json:/source_transfer_graph_topology/FT2_92_output` and
   `shadow_sufficiency_spec.json:/historical_to_IBKR_source_transfer` emit
   `pass`, `source_transfer_insufficient`, `source_transfer_fail`, or
   `invalid_source_transfer_evidence`, with no frozen mapping to graph `fail`.
   Similarly Graph V2 gives FT2-93 only `producer_complete` and `fail`, while
   the repaired Phase-F law emits `insufficient_shadow_evidence` for sessions
   1-44 and also names `shadow_transfer_fail` and `invalid_shadow_evidence`.
   None is a declared Graph V2 terminal. This is a fresh **BLOCKING** routing
   defect affecting Graph V2 and FT2-11; insufficient evidence cannot silently
   be collapsed into candidate failure.

7. **15:55 terminal law.** `FT2-08/tensor_schema.json:/open_state` exposes no
   action at 15:55, does not invoke the model, types the row
   `FORCED_FLAT_TERMINAL`, and suppresses duplicate forced flat after a 15:54
   EXIT fills at 15:55. `field_semantics_manifest.json:/completed_minute_law`
   agrees. Conclusion: repaired.

8. **Deterministic session-start equity.**
   `account_state_ledger_spec.json:/session_start` freezes offline carried cash
   and, live, the latest authenticated NetLiquidation source update in
   `[09:29:00,09:30:00]`, received by 09:30:05, with deterministic tie-break,
   cent normalization, maximum age, dedicated Protocol101 paper account, and
   fail-closed external-state checks. Conclusion: repaired.

9. **Near-close full-window CDF filter.**
   `objective_spec.json:/nested_empirical_cdf/full_window_filter` admits finite
   hN values only when uncensored with exactly N marks and excludes
   finite-but-censored near-close values; remaining-session rows require the
   complete grid to 15:55. The survival exception remains separate.
   Conclusion: repaired.

10. **Substitute-cluster breadth.**
    `composer_spec.json:/uncertainty_wait/directional_substitute_cluster`
    expands the same-right/same-expiry cluster to plus/minus 10 points (two
    standard ladder steps) and excludes in-cluster substitutes from the outside
    comparator. This directly covers the prior smooth three-or-more-strike
    example; the separate q90 exact-contract-regret threshold is intended to
    control within-cluster choice. No additional breadth defect is established
    from the allowlisted evidence. The action-head target circularity remains
    S2-04, not a cluster-definition defect.

11. **Phase-F hard reporting floor.** The AR design covers 45-session fold
    shapes; both `bootstrap_spec.json:/phase_F_shadow_intervals` and
    `shadow_sufficiency_spec.json:/minimum_no_order_live_shadow_evidence`
    require 45 complete sessions for hard bounds and gates. Sessions 1-44
    permit only point estimates/raw counts. The statistical repair is coherent.
    The mandatory below-floor outcome is unroutable in Graph V2 as stated in
    item 6.

12. **MNAR alternate refit.**
    `evidence_standard.json:/mnar_no_bid_sensitivity/alternate_refit_protocol`
    explicitly refits every path head, nested CDF, calibrator, threshold,
    WAIT/regret action head, inner selection, and ensemble; only the label view
    changes, and cross-view artifact reuse is forbidden.
    `terminal_decision_spec.json:/statistic_specific_sufficiency/MNAR_ranking_stability`
    consumes that exact definition. Conclusion: repaired.

13. **Census v3 internal consistency and the claimed 1.6%.** For
    `best_session`, oracle PnL moves from `$341,796` to `$340,736`
    (`-0.310127%`) with 108 trades unchanged, while P5 moves from `$156,830`
    to `$154,008` (`-1.799401%`) with trades rising 580 to 589. Across variants,
    hindsight-oracle ceilings are generally close while causal P5 outcomes move
    more. That is internally consistent with the repair: a hindsight oracle can
    re-rank remaining opportunities, while P5 commits one causal t identity,
    takes t+1 rejection/no-redraw, and follows a changed policy-specific ledger.
    It is not evidence that the new definition is economically immaterial.

    No inspected top-level census artifact actually states “1.6% fill-recheck
    rejection rate.” The number is reproducible only as
    `7,205 / 460,937 = 1.563120%`, using **all governed candidate label rows**.
    The economically natural conditional rejection rate among legal causal
    intents is `7,205 / 128,758 = 5.595769%`. Therefore an unqualified 1.6%
    description understates conditional rejection by using a much broader
    denominator. This is a **MATERIAL, acceptable-with-documentation** reporting
    issue: the exact counts are present and the contract economics do not
    depend on the shorthand, but any downstream citation must name the
    denominator and preferably report both rates.

Additional ML/statistical scan: the action-conditioned target circularity in
S2-04 is the only additional target/calibration defect found. Receipt integrity,
matched-random identity, studentization arithmetic, nested CDFs, MDE population,
MNAR refit, and the 45-session statistical floor otherwise reproduce from the
allowlisted final artifacts.

## FINDINGS

1. **BLOCKING — rerun001 S2-04 is NOT ACTUALLY ANSWERED.** Affected contracts:
   FT2-10 and FT2-11. Evidence:
   `FT2-10/forecast_heads.json:/action_conditioned_outputs/wait_probability_head/training_target`,
   `/selected_contract_regret_heads/scope`, and `/training_data_path`;
   `FT2-10/composer_spec.json:/composer_order` and
   `/mandatory_action_conditioned_gate`. The only composer requires the q90
   head to emit ENTER, while that head is trained only on composer-ENTER rows
   and the purported pre-fit “realized-label audit composer” is undefined.

2. **BLOCKING — current FT2-04 oracle/census eligibility contradicts the
   canonical causal intent law.** Affected contracts: FT2-04, FT2-05, and
   FT2-08. Evidence:
   `FT2-04/oracle_rules.json:/census_layer_constraints` and
   `/oracle_sequencing` use actual t+1 ask as decision eligibility;
   `FT2-08/intent_fill_recheck_law.json:/decision_time_intent` and
   `/label_oracle_census_semantics` prohibit that classification; FT2-05 v3
   pins both hashes.

3. **BLOCKING — repaired Phase-F outcomes are not legal Graph V2 routes.**
   Affected contracts: Graph V2 and FT2-11. Evidence: Graph V2 FT2-92 edges
   (`pass`, `fail`) and FT2-93 edges (`producer_complete`, `fail`) versus
   `FT2-11/evidence_standard.json:/source_transfer_graph_topology/FT2_92_output`
   and `shadow_sufficiency_spec.json:/terminal_rules`, which emit distinct
   insufficient/fail/invalid outcomes without a mapping.

4. **MATERIAL — fill-recheck rejection shorthand is denominator-ambiguous;
   acceptable-with-documentation.** Affected contract: FT2-05 reporting.
   Evidence: `FT2-05/census_results.json` and `v2_v3_impact.json` record 7,205
   rejections, 128,758 causal intents, and 460,937 governed rows. The conditional
   rejection rate is 5.595769%; 1.563120% uses all governed rows.

SEAT ROUTING RECOMMENDATION: STOP-REDESIGN-REQUIRED
