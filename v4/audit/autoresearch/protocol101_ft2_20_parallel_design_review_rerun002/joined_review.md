# Protocol101 FT2-20 Parallel Design Review — Final Re-review Rerun 002

Node: `FT2-20-PARALLEL-DESIGN-REVIEW`

Review kind: final re-review after coordinated repair attempt 2 of 2.

Pinned consolidated authority:
`2363d3f986daba20bd5087ed751dc5b2d839e76cd6413aeca0bcd255eb98857a`

Pinned final-repair aggregate receipt:
`19b74e513f0c6c68b50167b37e4bcfd40cb4ccd0f3c2240201ff3cbbd178f4f3`

## Outcome and Graph Routing

**Outcome: `STOP-REDESIGN-REQUIRED`.**

The legal route is:

```text
irreducible_design_defect -> STOP-REDESIGN-REQUIRED
```

`pass -> FT2-21-OWNER-DESIGN-APPROVAL` is unavailable. All repair budgets
are exhausted, and this final re-review found both rerun001 findings that were
not actually answered and new stop-class evidence. This join does not create,
imply, or authorize another repair route.

No reviewed artifact was edited or resolved. FT2-21 was not started.

## Review-Target Verification

Before dispatch, the controller independently verified:

| Target | SHA-256 / result |
|---|---|
| Consolidated authority | `2363d3f986daba20bd5087ed751dc5b2d839e76cd6413aeca0bcd255eb98857a` |
| Graph V2 JSON | `b06a26be59307c130da84f2dc5b6f3224c272e6c4093e83abd5bc0b280ca6d09` |
| Simulator v5 | `7296a437577ed006326d2ad35ad1f3499c4925334556d64d8c5fb75e4985f548` |
| FT2-04 current receipt | `c763dee85293d73a4f367627e8f483a209565d3f486f82fb5b1c17f3ff611775`; 7/7 receipt-listed deliverables exact |
| FT2-05 census v3 receipt | `a252feb2007b2aece77f377461bc6fa5a882e929bd2f7d3f23a47c07401cfdce`; 114/114 receipt-listed deliverables exact |
| FT2-05 v2-v3 impact | `5341e2ead4ff6e0d4bbdbb59ba6312180c457e41a89ab6e8dfc7b28796d9f9a1` |
| FT2-08 v3 receipt | `731cc6fb4c44bd0650d658e71fffbdac7c3e0f5d68701b2e05b8576b014b078c`; 14/14 deliverables exact |
| FT2-10 v3 receipt | `c04a591fd3034a2caba756e8f9dc1d7f5c24c397170894b0b6b0eca7a083fa6f`; 8/8 deliverables exact |
| FT2-11 v3 receipt | `bdb52c4f9178da1cab75ec4a80bd13c1db8c0074c1c398f380729d1df80c5ef9`; 10/10 deliverables exact |
| Unified 47-finding crosswalk | `78f69b89e0857c30af6f354cd8a26db923f075c9700f01d3971012462d7fa16d` |
| Final consistency checker | `7b327ba8956ef03c581a4e4e390440a22d92c962cbf4e564ac8a876978a38d04` |
| Final consistency output | `64a051d401381deddbdea34ec6b439d6bfe15d399e1b3461c79a5243e5b86664`; 26/26 producer checks passed |
| Final-repair aggregate receipt | `19b74e513f0c6c68b50167b37e4bcfd40cb4ccd0f3c2240201ff3cbbd178f4f3` |

All three seats independently reverified the authority and final aggregate
hashes. Their reports record their own receipt/deliverable checks.

## A. Joined Rerun001-Finding Verification

Every rerun001 finding was independently inspected in the responsible seat's
domain. The crosswalk was used only as a locator; no seat treated a crosswalk
claim as repair evidence.

| Seat | Domain findings | ANSWERED | NOT ACTUALLY ANSWERED |
|---|---:|---:|---:|
| Seat 1 — trading realism | 10 | 9 | 1 |
| Seat 2 — ML/statistics | 18 | 17 | 1 |
| Seat 3 — live parity | 19 | 19 | 0 |
| **Joined** | **47** | **45** | **2** |

The two failed verifications are binding:

1. **FRESH-S1-B1 — NOT ACTUALLY ANSWERED; BLOCKING.** Seat 1 found that
   FT2-08's causal time-`t` intent law conflicts with the still-current,
   receipt-pinned FT2-04 `oracle_rules.json` future-`A_(t+1)` decision mask,
   and FT2-05 v3 pins both sources. Affected: FT2-04, FT2-05, FT2-08, and
   downstream FT2-10/11.
2. **S2-04 — NOT ACTUALLY ANSWERED; BLOCKING.** Seat 2 found the mandatory
   action-conditioned calibration target mechanically circular: the regret
   heads are trained only on rows selected ENTER by the composer, while the
   only frozen composer requires those already-trained/calibrated heads to
   select ENTER. The referenced pre-fit audit composer is undefined. Affected:
   FT2-10 and the FT2-11 acceptance consumer.

Either failed verification independently forces `STOP-REDESIGN-REQUIRED`
under the exhausted-budget routing law.

## B. Joined Fresh Scan and Findings

Seat conclusions are preserved without severity downgrade or deletion:

| Seat | BLOCKING | MATERIAL acceptable-with-documentation | MATERIAL not-acceptable | MINOR | Seat route |
|---|---:|---:|---:|---:|---|
| Seat 1 | 1 | 0 | 0 | 0 | STOP |
| Seat 2 | 3 | 1 | 0 | 0 | STOP |
| Seat 3 | 0 | 0 | 1 | 0 | STOP |

Current findings, in the seats' own classifications:

1. **Seat 1 F-TR-001 — BLOCKING.** Incompatible active intent-mask
   authorities across FT2-04/05 and FT2-08, with FT2-10/11 downstream. This is
   also the failed verification of FRESH-S1-B1.
2. **Seat 2 — BLOCKING.** Rerun001 S2-04 is not actually answered because the
   FT2-10 action-conditioned target/composer dependency is circular and
   mechanically underdefined; FT2-11 consumes the defective gate.
3. **Seat 2 — BLOCKING.** Current FT2-04 oracle/census eligibility contradicts
   the canonical causal intent law. This independently converges with Seat 1
   F-TR-001; both reports remain present separately.
4. **Seat 2 — BLOCKING.** Repaired Phase-F/source-transfer outcome vocabulary
   is not mapped to legal Graph V2 FT2-92/93 edges. Affected: Graph V2 and
   FT2-11.
5. **Seat 2 — MATERIAL, acceptable-with-documentation.** The shorthand
   “1.6% fill-recheck rejection” is denominator-ambiguous: 1.563120% is
   `7,205 / 460,937` governed rows, whereas the conditional rejection rate is
   `7,205 / 128,758 = 5.595769%`. Affected: FT2-05 reporting.
6. **Seat 3 LP-RERUN002-01 — MATERIAL, not-acceptable.** The 37,759-row
   D48-reference reduction reconciles arithmetically, but the allowed final
   packet lacks a row-identity v2/v3 transition crosswalk, gross
   removal/addition counts, and economic/label distributions for the changed
   population. Affected: FT2-05 and downstream census consumers.

The join does not soften Seat 3's `not-acceptable` classification. That finding
alone would force `STOP-REDESIGN-REQUIRED`, even absent the blocking set.

## Convergence, Disagreement, and Recomputed Hot Spots

Convergence:

- Seats 1 and 2 independently found the same active FT2-04/05 versus FT2-08
  intent-law contradiction. Seat 3 judged the repaired two-stage law coherent
  and did not record that conflict; the disagreement is preserved and cannot
  erase either blocking report.
- All three independently reproduced the canonical comparator SHA-256
  `8bdbe8beb4734852526cd2981be76098f792d5c4b9fee30f89f8a2d952abf754`.
- All three independently recomputed the delete-`L` example:
  `V=5/6`, `SE=0.9128709291752769`,
  `t=3.2863353450309964`, replicate `SE*=0.9128709291752769`, and
  `t*=-0.4381780460041329`.
- All three found the fee-path soft-close arithmetic, 15:55 terminal action
  law, session-start equity primitive, near-close CDF filter, comparator
  identity, cluster repair, and MNAR refit mechanically specified.
- Seats 1 and 2 found the aggregate D48 count identity fully explains
  `159,312 - 121,553 = 30,554 + 7,205`. Seat 3 agreed on the arithmetic but
  classified the absent row-level/economic transition audit MATERIAL and
  not-acceptable. That classification is preserved.
- Seats 2 and 3 independently reproduced both census rejection denominators.
  Seat 2 classified the shorthand MATERIAL acceptable-with-documentation;
  Seat 3 treated the denominator reconciliation as coherent and attached its
  stop classification to the missing transition audit instead.
- Seats 1 and 3 found the source-transfer ordering repaired. Seat 2 agreed the
  order is no longer circular but separately found the emitted FT2-92/93
  outcome names lack legal Graph V2 edge mappings. The Seat 2 BLOCKING finding
  is preserved.

Census v3's approximately flat oracle ceilings and generally lower causal-P5
outcomes were reconciled by each seat as compatible with hindsight re-ranking
versus selected-identity causal commitment and no-redraw fill rejection. No seat
used that aggregate reconciliation to override a stop-class finding.

## Exhausted-Budget Routing

No repair route remains. The owner-design approval path is not reached.

Affected contracts/artifacts named by the stop record:

- FT2-04 path-label/oracle freeze;
- FT2-05 opportunity census v3 and its transition evidence;
- FT2-08 intent/fill law;
- FT2-10 action-conditioned target/composer contract;
- FT2-11 acceptance and Phase-F routing contracts;
- Graph V2 FT2-92/93 outcome edges;
- downstream FT2-10/11 consumers of the conflicting intent/census authorities.

Required terminal state:

```text
STOP-REDESIGN-REQUIRED
```

## Isolation Record

The upgraded isolation protocol completed without a disclosed imperfection:

1. Each seat was created with `fork_turns=none`, so no seat inherited the
   controller conversation or another seat's report.
2. Seats ran strictly sequentially. Only one reviewer agent was active at a
   time.
3. Every role prompt contained the identical evidence allowlist and explicitly
   prohibited recursive traversal, entry into `superseded/`, other scratch
   paths, the future rerun002 destination, and all writes except the assigned
   private report. Only the review lens and assigned scratch path differed.
4. Seat 1 wrote only
   `/tmp/protocol101-ft2-rerun002-seat1.ciPDP5/seat1_trading_realism.md`
   (22,299 bytes; SHA-256
   `cfd0883039f788dcae3f421d5d23aa18c81e427d54cecef4b22bec07838d8437`).
   The controller captured it in controller-only memory and deleted the file
   and directory before Seat 2 was created.
5. Seat 2 wrote only
   `/tmp/protocol101-ft2-rerun002-seat2.I4uPWb/seat2_ml_statistics.md`
   (24,595 bytes; SHA-256
   `bf4b21ad70e0ce4f8b914a206cccc000c24280323c7a20ce1223d99c08c95270`).
   The controller captured it and deleted the file and directory before Seat 3
   was created.
6. Seat 3 wrote only
   `/tmp/protocol101-ft2-rerun002-seat3.HwWHeT/seat3_live_parity.md`
   (32,985 bytes; SHA-256
   `671894e61886501597a1f4b71f0539820ccf745e73419c2d30c07c05e584423c`).
   The controller captured it and deleted the file and directory before the
   shared rerun002 packet was created.
7. No completed report was physically present in a shared review location or
   any scratch path while another seat ran. No seat was told another seat's
   path, result, finding, or routing recommendation.
8. Every seat affirmed top-level-only enumeration and no entry, recursive
   listing, search, or read of any `superseded/` directory.
9. The three files below were materialized only after all seats completed and
   all private scratch copies were absent. Their hashes exactly match the
   private-report hashes.

## Scope and Prohibitions

The re-review performed no repair, target edit, training, fitting, new session
statistic, protected-data access, recorder access, broker access, paid compute,
promotion/runtime change, or FT2-21 work. Only the three verbatim seat files,
this join, and the rerun002 receipt belong to this goal.

Highest allowed claim:

> The final-repaired contract set has been independently re-reviewed; the graph routes to redesign stop.

## Verbatim Seat Outputs

The following blocks preserve the isolated completed seat reports verbatim.
Convergent findings are not deduplicated and no severity or acceptability
classification is altered.

<!-- VERBATIM OUTPUTS FOLLOW -->

<!-- BEGIN VERBATIM SEAT 1 -->
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

<!-- END VERBATIM SEAT 1 -->

<!-- BEGIN VERBATIM SEAT 2 -->
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

<!-- END VERBATIM SEAT 2 -->

<!-- BEGIN VERBATIM SEAT 3 -->
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

<!-- END VERBATIM SEAT 3 -->

<!-- VERBATIM OUTPUTS END -->
