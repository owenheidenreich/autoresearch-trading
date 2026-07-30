# Protocol101 FT2-20 Parallel Design Review Rerun 001

Node: `FT2-20-PARALLEL-DESIGN-REVIEW`

Pinned consolidated authority:
`2363d3f986daba20bd5087ed751dc5b2d839e76cd6413aeca0bcd255eb98857a`

## Outcome and Graph Routing

**Outcome: NOT DESIGN-ACCEPTABLE.**

Primary legal outcome:

```text
data_tensor_label_defect -> FT2-08-DATA-TENSOR-LABEL-CONTRACT
```

Additional defect outcomes, retained in dependency order:

```text
entry_science_defect -> FT2-10-ENTRY-SCIENCE-CONTRACT
evidence_statistics_defect -> FT2-11-EVIDENCE-STATISTICS-CONTRACT
```

The graph repairs one contract at a time. FT2-08 is first because the
decision-time D48/D49 masks and permanent soft-close state are foundational
inputs to both the entry-science game and the evidence-statistics game. Routing
FT2-08 first does not clear, merge, resolve, or downgrade any FT2-10 or FT2-11
finding recorded below. `pass -> FT2-21-OWNER-DESIGN-APPROVAL` is not available.

No reviewed defect is judged irreducible at this review node. Each owner packet
has a bounded, contract-specification repair set that can plausibly be handled
in one coordinated final attempt. The legal repair budgets are therefore:

| Contract owner | Graph budget | Attempts used | Attempts remaining | Routed attempt |
|---|---:|---:|---:|---|
| FT2-08 data/tensor/label | 2 | 1 | 1 | 2 of 2, primary |
| FT2-10 entry science | 2 | 1 | 1 | 2 of 2, retained |
| FT2-11 evidence/statistics | 2 | 1 | 1 | 2 of 2, retained |

If any owner cannot repair its recorded set in that one remaining attempt, or a
blocking defect for that owner persists after the final attempt, the remaining
budget is exhausted and the graph must route:

```text
irreducible_design_defect -> STOP-REDESIGN-REQUIRED
```

This join records routing only. It does not start a repair, FT2-21, training,
fitting, or new statistical analysis.

## Original-Finding Verification

Every original domain finding was individually rechecked against repaired
primary evidence:

| Seat | Original findings | ANSWERED | PARTIALLY ANSWERED | UNANSWERED |
|---|---:|---:|---:|---:|
| Trading realism | 7 | 6 | 1 | 0 |
| ML/statistics | 11 | 7 | 4 | 0 |
| Live parity | 10 | 8 | 2 | 0 |
| **Joined** | **28** | **21** | **7** | **0** |

The partial verdicts remain live findings at no lower than MATERIAL severity.
No original finding was silently treated as answered.

## Fresh-Scan Findings

No exact duplicate was deleted. Independently convergent findings remain
separately present in the verbatim seat outputs.

| Seat | BLOCKING | MATERIAL | MINOR |
|---|---:|---:|---:|
| Trading realism | 1 | 2 | 0 |
| ML/statistics | 4 | 2 | 1 |
| Live parity | 3 | 5 | 1 |
| **Joined** | **8** | **9** | **2** |

The two MINOR entries are the independently recorded receipt-field naming
disclosure. Seat 1 disclosed the same inconsistency without assigning it a
fresh-finding severity. No seat severity was downgraded.

## Convergence and Contract Ownership

1. **Future `t+1` ask in decision masks — BLOCKING, three-seat convergence.**
   All three reviewers independently found that D48/D49 consume the actual
   `t+1` fill ask before the time-`t` action is selected. Primary owner:
   FT2-08. FT2-10/11 consumers must remain synchronized after the causal
   time-`t` intent mask and separate time-`t+1` fill recheck are frozen.
2. **D49 soft close does not implement signed A3 — BLOCKING at joined
   severity, three-seat convergence.** Seat 1 retained MATERIAL severity;
   Seats 2 and 3 independently classified the ladder- and fee-path mismatch as
   BLOCKING. All three severities are preserved. Primary owner: FT2-08, with
   FT2-10 consumer synchronization.
3. **Matched-random comparator identity — BLOCKING.** FT2-10 and FT2-11 freeze
   different random-key namespaces for the required D1 comparator. Owner:
   FT2-11 must align its evidence machinery to the canonical FT2-10 source
   contract or freeze one unambiguous shared algorithm.
4. **Undefined studentized replicate standard error — BLOCKING.** `SE_h_star`
   is not mechanically defined, so adjusted bounds, terminal decisions, and
   block-power MDE are not reproducible. Owner: FT2-11.
5. **Source-transfer topology — BLOCKING.** FT2-11 requires the
   candidate-specific source-transfer pass before the graph node that produces
   it. Owner: FT2-11 must remove the circular activation boundary while
   preserving the graph.
6. **FT2-10 retained MATERIAL set.** This includes the missing full-window
   filter for finite near-close CDF references, the selected-contract regret
   head not constraining the action, the neutral one-trade game conflicting
   with census-derived trade-rate selection, and an insufficiently broad
   substitute cluster.
7. **FT2-08 retained MATERIAL set.** This includes the impossible learned EXIT
   action exposed at 15:55 and the nondeterministic/stale or unrelated
   session-start paper-equity primitive.
8. **FT2-11 retained MATERIAL set.** This includes Phase-F hard reporting at a
   sample size below the demonstrated coverage regime and an MNAR alternate
   view that does not specify whether label-dependent machinery is refit.

These ownership assignments determine repair routing; they do not resolve the
findings or authorize contract edits in this node.

## Isolation and Disclosure Record

- Three fresh reviewer sessions were created with no parent conversation
  history (`fork_turns=none`) and the same evidence allowlist and disclosure.
- All three were dispatched before any completed and were observed running
  concurrently before the first completion.
- Each reviewer states that it did not communicate with another reviewer or
  inspect another rerun seat/output. The controller did not inspect any seat
  file until all three sessions had completed.
- Because each reviewer persisted its assigned report directly into the shared
  workspace when it completed, earlier completed seat files were physically
  present while later seats were still running. The role prompts prohibited
  inspecting rerun outputs, and Seats 2 and 3 explicitly report that they did
  not inspect them, but the shared-filesystem harness did not make those files
  physically invisible. This is a second procedural imperfection and prevents
  a perfect no-cross-seat-visibility claim.
- Seats 2 and 3 report perfect substantive isolation under the prompt.
- Seat 1 reports a procedural isolation breach: one recursive search was
  mistakenly scoped to repaired packet directories rather than top-level
  regular files. It could traverse the conservatively listed `superseded/`
  files. The seat states that no nested match was returned, no nested text was
  intentionally inspected or cited, and none informed its findings. Therefore
  this run is not represented as perfectly isolated, though the seat reports
  substantive independence.
- Each seat includes `FILES ACTUALLY OPENED` and an honest isolation statement.
- The producer disclosure supplied identically to all seats was:
  `"receipt schema field naming is inconsistent across packets (authority_sha256 vs product_contract_hash); values are verified correct; classify as the seats see fit."`

## Scope and Highest Allowed Claim

No contract, frozen artifact, graph, runtime, promotion state, or finding was
edited or resolved. No training, fitting, new session statistic, paid-data
access, broker access, repair work, or FT2-21 work occurred.

> The repaired three-contract set has been independently re-reviewed; the graph routes to named contract repair.

## Verbatim Seat Outputs

The following three blocks preserve the completed seat files verbatim.

<!-- VERBATIM OUTPUTS FOLLOW -->

<!-- BEGIN VERBATIM SEAT 1 -->
# Protocol101 FT2-20 Rerun Seat 1 — Trading Realism

## ISOLATION AND DISCLOSURE

Isolation was **not perfect**. I did not inspect either other original review seat, the original joined review, any rerun output, market data outside the allowlist, or any external source. However, one `rg` command was mistakenly scoped recursively to the three repaired packet directories instead of to their top-level regular files. It could therefore open the nested `superseded/` files listed in `FILES ACTUALLY OPENED`; no nested match was returned, no nested text was intentionally inspected or cited, and none informed the findings below, but the maxdepth-1 isolation rule was breached. This review is substantively independent but procedurally tainted and should not be represented as perfectly isolated.

Producer disclosure: "receipt schema field naming is inconsistent across packets (authority_sha256 vs product_contract_hash); values are verified correct; classify as the seats see fit."

The review remained read-only except for this required Seat 1 report. No model was trained, no new session statistic was computed, no finding was repaired, and FT2-21 was not started.

## A. FINDINGS VERIFICATION

### S1-01 — original `[BLOCKING] No-bid periods are erased instead of treated as losses`

**Verdict: ANSWERED.**

Repaired evidence:

- `protocol101_ft2_04_path_label_freeze/label_spec.json`, keys `economic_primitives.no_bid_full_loss_bound`, `missing_minute_rule.primary_no_bid_convention`, and `missing_minute_rule.forced_flat`.
- `protocol101_ft2_08_data_tensor_label_contract/label_join_spec.json`, keys `no_bid.primary`, `no_bid.underwater`, `no_bid.breaks_profitable_run`, `no_bid.forward_fill`, and `timing.forced_flat`.
- `protocol101_ft2_11_evidence_statistics_contract/evidence_standard.json`, key `mnar_no_bid_sensitivity.hard_ranking_stability`.
- `protocol101_ft2_11_evidence_statistics_contract/contract.md`, §11 “MNAR No-Bid Sensitivity.”

The primary label now values every missing, stale, nonpositive, or otherwise non-executable future bid at zero option value, charges the full premium plus fee loss, marks the minute underwater, breaks profitable runs, and prohibits earlier-bid substitution at 15:55. The separately named no-bid-excluded view is sensitivity-only, and acceptance requires identical selected lineage/composer, identical terminal, unchanged selected-lineage pairwise orderings, and rank correlation at least 0.95. The economic erasure identified originally is therefore removed rather than merely documented.

### S1-02 — original `[BLOCKING] Entry and exit fills use prices known before the action can execute`

**Verdict: ANSWERED.**

Repaired evidence:

- `protocol101_ft2_04_path_label_freeze/label_spec.json`, keys `economic_primitives.entry_fill`, `economic_primitives.realized_exit_fill`, `horizon_window_definition`, and `session_boundaries`.
- `protocol101_ft2_08_data_tensor_label_contract/label_join_spec.json`, key `timing`.
- `protocol101_ft2_08_data_tensor_label_contract/replay_authority_v5_1_spec.json`, keys `required_v5_1_behavior.entry_fill`, `entry_failure`, `exit_fill`, `occupancy`, and `forced_flat`.
- `protocol101_ft2_11_evidence_statistics_contract/contract.md`, §12 “Recorder `+5s` Gap Study.”

A BUY committed from completed minute `t` now fills only from the executable `t+1` ask; an EXIT decided at `v` fills only from the executable `v+1` bid. A missing `t+1` ask is a failed execution, not a backfilled fill. Descriptive marks begin at `t+2`; the last BUY decision is 15:29/15:30 fill, and a 15:54 EXIT realizes once at 15:55. This supplies the post-decision executable quote required by the original acceptance test. The unavailable subminute `+5s` comparison is accurately classified as a report-only future study, not treated as existing evidence.

### S1-03 — original `[BLOCKING] The primary entry test is not the required one-account serial game, making P5-under-cap unfair`

**Verdict: ANSWERED.**

Repaired evidence:

- `protocol101_ft2_10_entry_science_contract/objective_spec.json`, keys `serial_primary_population.universal_grid`, `policy_states`, `occupied_minute_treatment`, `entry_commit_and_fill`, `neutral_lifecycle`, `paired_comparison`, and the complete `p5_under_cap_algorithm` object.
- `protocol101_ft2_10_entry_science_contract/contract.md`, §§7–9.
- `protocol101_ft2_11_evidence_statistics_contract/contract.md`, §§4 and 5.1.D.

The hard population is now one universal chronological grid with explicit WAIT, BUY-commit, pending-fill, occupied, soft-closed, and daily-stopped states. Pending and occupied rows remain in denominators but cannot receive a counterfactual BUY. Candidate, P5, controls, alpha arms, and fee paths have separate causal cash and safety ledgers. P5 timing, direction, eligible set, deterministic tie break, WAIT cases, fill law, lifecycle, and prohibited inputs are frozen exactly. The entry component's economic condition is session-paired serial-dollar no-harm versus that P5 ledger; `DeltaQ` is diagnostic only.

The new future-informed action-mask defect in §B does invalidate live causality of both ledgers, but it is a newly exposed mask-definition defect, not the original absence of serial occupancy or an exact P5 algorithm.

### S1-04 — original `[BLOCKING] The composer can structurally delete the Charter’s big-win column`

**Verdict: ANSWERED.**

Repaired evidence:

- `protocol101_ft2_10_entry_science_contract/composer_spec.json`, key `stage_1_quality_screen.no_screen_control_arm`.
- `protocol101_ft2_10_entry_science_contract/objective_spec.json`, key `tail_guard_relative_capture`.
- `protocol101_ft2_11_evidence_statistics_contract/tripwire_spec.json`, keys `relative_capture_tripwire` and `four_bucket_profile_tripwire`.
- `protocol101_ft2_11_evidence_statistics_contract/contract.md`, §10.

Every campaign must retain an otherwise-identical `alpha=0.0` no-quality-screen arm. The selected arm is paired to it on the same model, uncertainty multiplier, calibrators, masks, serial law, sessions, seed/ensemble, no-bid convention, and fee trajectory. Big-win incidence, big-win opportunity dollars, and positive-PnL concentration are reported relative to that arm; incidence capture below 50% forces `owner_decision_required`. A zero control denominator is insufficient evidence, not a pass. The independent four-bucket tripwire remains active. This prevents the old quiet optimization among only nonzero screens and makes material tail deletion an explicit owner event.

### S1-05 — original `[MATERIAL] Dual-unit min() arbitration biases selection toward middle-premium contracts`

**Verdict: ANSWERED.**

Repaired evidence:

- `protocol101_ft2_10_entry_science_contract/objective_spec.json`, complete key `nested_empirical_cdf`.
- `protocol101_ft2_10_entry_science_contract/composer_spec.json`, keys `stage_2_conservative_upside_rank.premium_band_at_decision`, `favorable_percentile_rule`, and `dollar_percent_arbitration`.
- `protocol101_ft2_10_entry_science_contract/contract.md`, §5.4.

Dollar and return values are no longer percentile-ranked across the whole phase population. Each unit is transformed within the same causal decision-time premium band and market phase, with session-balanced weights, explicit midrank ties, interpolation, minimum reference sizes, same-band fallback only, and no cross-band fallback. The `min()` now arbitrates two conditional percentiles rather than mechanically premium-scaled raw populations. The CDF censoring defect in §B is separate from the original cross-premium bias.

### S1-06 — original `[MATERIAL] D49 “size-down” can force post-loss lottery exposure and is not independently enforced by simulator v5`

**Verdict: PARTIALLY ANSWERED.**

Repaired evidence:

- Consolidated authority, §1.4 D49 and post-signature amendment A3.
- `protocol101_ft2_08_data_tensor_label_contract/account_state_ledger_spec.json`, key `d49`.
- `protocol101_ft2_08_data_tensor_label_contract/replay_authority_v5_1_spec.json`, keys `required_v5_1_behavior.d49`, `independent_replay_assertions`, `required_boundary_fixtures`, and `activation_gate`.
- `protocol101_ft2_10_entry_science_contract/objective_spec.json`, keys `serial_primary_population.account_and_safety_state` and `serial_primary_population.paired_comparison`.
- `protocol101_serial_simulator_v5.py`, `SerialSimulatorV5Config` and the replay skip/enforcement loop.

The substantive lottery concern is addressed for the primary `$3` path: remaining budget below the cost of a `$1` contract plus its fee permanently soft-closes new entries, and v5.1 is required to recompute D48/D49 independently from integer-cent ledger primitives. Current simulator v5 still enforces affordability and the realized daily stop but not D48, D49, or soft close; the repair honestly leaves it unmodified and blocks v5.1 activation until FT2-30 implementation and FT2-31 acceptance.

The answer is only partial because the machine contract hard-codes `$103` even though both `$3` and `$4` causal fee paths are mandatory. As detailed in §B, the `$4` path needs a `$104` floor to implement `$1 + fee`; the present rule can still admit a sub-$1 wing in the exact post-loss interval the amendment was intended to close.

### S1-07 — original `[MATERIAL] Exact-contract uncertainty can deadlock valid directional sessions`

**Verdict: ANSWERED.**

Repaired evidence:

- `protocol101_ft2_10_entry_science_contract/composer_spec.json`, key `uncertainty_wait.adjacent_strike_cluster`.
- `protocol101_ft2_10_entry_science_contract/contract.md`, §§4.2 and 5.5.
- `protocol101_ft2_10_entry_science_contract/composer_spec.json`, keys `uncertainty_wait.model_gap_error`, `source_transfer_error`, and `rules`.

The WAIT comparison is now selected-cluster versus best surviving contract outside the same-right, same-expiry, ±5-point cluster. An inside-cluster adjacent substitute is explicitly prohibited from creating the WAIT gap; exact-contract uncertainty instead goes to the mandatory selected-regret head. Model and source-transfer margins use the same cluster-level quantity. Thus a nearly tied adjacent strike no longer deadlocks an otherwise strong directional cluster.

## B. FRESH SCAN

### `[BLOCKING]` D48/D49 action masks leak the future `t+1` ask into the decision at `t`

Exact citations:

- `protocol101_ft2_08_data_tensor_label_contract/account_state_ledger_spec.json`, keys `d48.prospective_entry_cost_cents` and `d49.mask_true`: prospective cost is defined as the **`t+1` executable ask** times 100 plus fee.
- `protocol101_ft2_08_data_tensor_label_contract/tensor_schema.json`, key `flat_state_masks.action_mask.contract_rule`: a BUY at decision time requires the D48 and D49 masks.
- `protocol101_ft2_10_entry_science_contract/composer_spec.json`, key `complete_ladder_and_safety.per_contract_requirements`: the composer consumes both masks before selecting.
- `protocol101_ft2_10_entry_science_contract/objective_spec.json`, key `p5_under_cap_algorithm.eligible_set`: P5 also consumes those masks before selecting.
- `protocol101_ft2_08_data_tensor_label_contract/label_join_spec.json`, keys `runtime_label_firewall.future_bid_or_path_in_feature_tensor` and `historical_action_without_labels_partition`; `objective_spec.json`, key `p5_under_cap_algorithm.target_and_forecast_independence`.

At completed decision minute `t`, neither the existence nor price of the `t+1` ask is causal. Yet the mask that determines which exact contract the candidate and P5 may choose is defined from that future ask. Fill-time rechecks do not repair this: a causal policy must first choose from information at `t`, then either fill or fail at `t+1`. The current contract can suppress a contract whose ask will rise, admit one whose ask will fall, avoid a no-ask failed intent, or choose another contract specifically because that other contract will have a future fill. It therefore leaks fill outcome and price into the action set, conflicts with the label firewall, and makes the serial/P5 and census opportunity game non-live-reproducible.

The correct causal split is not frozen: decision-time D48/D49 masks need the executable `t` quote plus fee reserve, while the already-specified `t+1` checks independently accept, reject, or fail the committed order. Until that distinction is explicit throughout FT2-08/10/11 and the census provenance is addressed, the three-contract set is not design-ready.

### `[MATERIAL]` The `$1 + fee` soft-close law is incorrectly fixed at `$103` on the `$4` path

Exact citations:

- Consolidated authority, §1.4 D49: a new entry includes premium plus fee, and amendment A3 soft-closes when no otherwise eligible contract with executable premium at least `$1.00` fits.
- `protocol101_ft2_08_data_tensor_label_contract/account_state_ledger_spec.json`, keys `d49.soft_close_floor_cost_cents=10300`, `soft_close_trigger`, and `reason`.
- `protocol101_ft2_08_data_tensor_label_contract/replay_authority_v5_1_spec.json`, keys `required_v5_1_behavior.d49`, `fee_trajectories`, and `required_boundary_fixtures`.
- `protocol101_ft2_10_entry_science_contract/objective_spec.json`, keys `serial_primary_population.account_and_safety_state.soft_close` and `fee_paths`.
- `protocol101_ft2_11_evidence_statistics_contract/contract.md`, §§4 and 5.1.C, which describe `$1 + fee` and require separate `$3`/`$4` trajectories.

`$103` is correct only for the `$3` trajectory. On the `$4` stress path the lane must close below `$104`. With remaining budget between `$103` and `$104`, no `$1` contract plus `$4` fits, but the fixed trigger stays open; a sub-$1 contract can still fit and keep the lane active. This recreates the prohibited cheap-wing selector in a narrow but intentional hard-stress boundary. The floor and its fixtures must be fee-trajectory-specific.

### `[MATERIAL]` Finite near-close censored labels are not explicitly excluded from empirical-CDF references

Exact citations:

- `protocol101_ft2_04_path_label_freeze/label_spec.json`, key `censoring_rules.near_close`: a finite horizon crossing 15:55 is shortened, retained, and marked `censored=true`.
- `protocol101_ft2_10_entry_science_contract/objective_spec.json`, key `decision_local_quality_diagnostic.component_transform.censored_or_invalid`: censored components are missing for supervised scoring.
- The same file, key `nested_empirical_cdf.reference_population`: CDFs consume finite FT2-04 label values but state no `censored=false` or full-window filter.
- `protocol101_ft2_10_entry_science_contract/composer_spec.json`, key `causal_horizon_availability.near_close`: runtime composition makes a crossing horizon unavailable and never shortens it.

The machine-readable CDF population can therefore be read to include finite shortened h20/h45/h90 outcomes even though the runtime score represents only fully available horizons. This would change percentile knots and guardrail/rank values using a different economic window. The prose that a censored component is absent from supervised scoring does not mechanically define the separate CDF build filter. Each finite-horizon CDF must expressly require a complete `h`-mark window (while preserving right-censor-aware treatment for the time-to-event head).

### Requested realism checks that otherwise hold

- **`t+1` endpoints:** `label_spec.json` defines `W(t,h)={u:e<u<=e+h}` with `e=t+1`, so the first mark is `t+2`, a complete h-minute window has exactly h marks, the last BUY decision/fill is 15:29/15:30, and the last learned EXIT decision/fill is 15:54/15:55 without duplicate forced flat. Apart from the future mask and CDF-population defects above, the endpoints are internally coherent.
- **No-bid and MNAR:** primary full-loss valuation, exact-boundary zero valuation, no forward/prior fill, and hard candidate-ranking stability are all explicit. FT2-05 v2 reports 2,690,689 no-bid marks out of 31,579,138 remaining-session marks and a minimum tabulated rank correlation of 0.9793244835544471; FT2-11 correctly treats that as label-surface context, not candidate evidence.
- **Serial state/P5:** the one-position pending/occupied state machine, failed-fill retry no earlier than `t+2`, separate causal ledgers, hold-to-15:55 neutral lifecycle, and exact P5 selection/tie/WAIT laws are explicit. The sole blocking realism exception found is that their upstream safety mask uses future `t+1` information.
- **Component versus promotable D1 structure:** `contract.md` §5 clearly limits entry success to `entry_component_freeze_pass`, with no selection eligibility or dollar-edge claim. The FT2-80 combined learned-entry-plus-lifecycle gate separately requires strict positive adjusted dollar evidence versus both exact P5 and exposure-matched random. This does not let component `DeltaQ` substitute for signed D1 monetary evidence.
- **Rebuilt MDE/census bound:** `mde_spec.json` uses actual pilot candidate-minus-P5 and candidate-minus-matched-random session deltas, statistic-specific SD/LRV/ESS, and the larger analytic/block-power MDE. It caps tranche feasibility at the existing hold-to-flat oracle-minus-P5 planning ceiling of `$2,138.6666666666665/session`, explicitly forbids census variance substitution, and respects the one-successful-trade-per-session neutral lifecycle. This is internally coherent and deliberately does not authorize spend.
- **Premium-band CDF:** current-`t` ask bands, session-balanced weights, exact midrank/tie/interpolation rules, same-band phase fallback, and no cross-band fallback correctly answer the original middle-premium bias. The finite-censor filter is the separate defect above.
- **Cluster-aware WAIT:** adjacent same-right/same-expiry ±5-point substitutes cannot themselves trigger WAIT; only the best outside-cluster alternative or WAIT=0 is compared, and exact-contract regret is separately gated.
- **Census-v2 internal consistency and P5 ceiling share:** the report, `oracle_ceiling_summary.csv`, and `v1_v2_impact.json` agree on 45 sessions and the `best_session` values. V1 tabulates oracle/P5 pooled PnL of `$377,161/$117,454` (roughly 31%); v2 tabulates `$341,796/$156,830` and directly reports `p5_share_of_oracle_ceiling=0.45884094606139336` (roughly 46%). The share moved because its numerator rose by `$39,376` while its denominator fell by `$35,365`; v2 simultaneously changed next-minute fills, no-bid valuation, exact forced flat, embargo membership, and D49 soft close. The packet correctly says the movement cannot be assigned to one semantic change. The tabulated movement is not itself an inconsistency, although the future-informed eligibility mask above blocks treating the repaired replay as live-causal evidence.

## FILES ACTUALLY OPENED

Directly opened or content-searched:

- `/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/contracts/PROTOCOL101_FULL_TRADER_GRAPH_V2_CONSOLIDATED_AUTHORITY_2026_07_28.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/execution/PROTOCOL101_FULL_TRADER_GRAPH_V2.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/contracts/PROTOCOL101_TRADER_CHARTER.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/contracts/PROTOCOL101_G4_HOLDOUT_REVISION_2026_07_19.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/contracts/PROTOCOL101_G8_CALIBRATION_REVISION_2026_07_26.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/contracts/PROTOCOL101_D1_NEGATIVE_CONTROL_AND_INCREMENTAL_EDGE_AMENDMENT_2026_07_28.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/synchronization/contracts/PROTOCOL101_SCOPED_SYNCHRONIZATION_DECISION_2026_07_25.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_04_path_label_freeze/label_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_04_path_label_freeze/oracle_rules.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_04_path_label_freeze/report.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/census_results.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/report.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/v1_v2_impact.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/oracle_replay_audit.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/label_build_compute.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/guardrail_threshold_curves.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/oracle_ceiling_summary.csv`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/mnar_sensitivity.csv`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/friction_by_premium_band.csv`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/guardrail_trade_rates.csv`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/minimum_detectable_improvement.csv`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_20_parallel_design_review/seat1_trading_realism.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/model/protocol101_serial_simulator_v5.py`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/account_state_ledger_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/contract.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/field_semantics_manifest.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/fold_roles.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/identity_mapping_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/label_join_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/receipt.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/replay_authority_v5_1_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/storage_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/synthetic_golden_vectors.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/tensor_schema.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/validate_contract_v2.py`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/validation.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/calibration_spec 2.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/calibration_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/composer_spec 2.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/composer_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/contract.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/controls_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/findings_crosswalk.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/forecast_heads 2.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/forecast_heads.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/objective_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/receipt.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/bootstrap_spec 2.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/bootstrap_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/contract 2.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/contract.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/evidence_standard 2.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/evidence_standard.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/findings_crosswalk.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/mde_spec 2.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/mde_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/multiplicity_spec 2.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/multiplicity_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/receipt 2.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/receipt.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/shadow_sufficiency_spec 2.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/shadow_sufficiency_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/terminal_decision_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/tripwire_spec 2.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/tripwire_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/worked_terminal_examples.json`

Potentially opened by the accidental recursive search, and listed here conservatively with no omission:

- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/__pycache__/validate_contract_v2.cpython-312.pyc`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/superseded/v1_pre_tplus1_20260729/contract.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/superseded/v1_pre_tplus1_20260729/fold_roles.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/superseded/v1_pre_tplus1_20260729/label_join_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/superseded/v1_pre_tplus1_20260729/receipt.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/superseded/v1_pre_tplus1_20260729/storage_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/superseded/v1_pre_tplus1_20260729/tensor_schema.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/superseded/v1_pre_ft2_20_repair_20260729/calibration_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/superseded/v1_pre_ft2_20_repair_20260729/composer_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/superseded/v1_pre_ft2_20_repair_20260729/contract.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/superseded/v1_pre_ft2_20_repair_20260729/controls_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/superseded/v1_pre_ft2_20_repair_20260729/forecast_heads.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/superseded/v1_pre_ft2_20_repair_20260729/objective_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/superseded/v1_pre_ft2_20_repair_20260729/receipt.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/superseded/v1_pre_ft2_20_repair_20260729/bootstrap_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/superseded/v1_pre_ft2_20_repair_20260729/contract.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/superseded/v1_pre_ft2_20_repair_20260729/evidence_standard.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/superseded/v1_pre_ft2_20_repair_20260729/mde_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/superseded/v1_pre_ft2_20_repair_20260729/multiplicity_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/superseded/v1_pre_ft2_20_repair_20260729/receipt.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/superseded/v1_pre_ft2_20_repair_20260729/shadow_sufficiency_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/superseded/v1_pre_ft2_20_repair_20260729/tripwire_spec.json`
<!-- END VERBATIM SEAT 1 -->

<!-- BEGIN VERBATIM SEAT 2 -->
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
<!-- END VERBATIM SEAT 2 -->

<!-- BEGIN VERBATIM SEAT 3 -->
# Protocol101 FT2-20 Rerun — Seat 3 Live-Parity Review

## ISOLATION / DISCLOSURE

**Verdict: NOT DESIGN-ACCEPTABLE.** The repairs answer most of the original Seat
3 findings at the design-contract level, but the fresh scan found three blocking
cross-contract defects and several material residual defects.

Isolation was perfect for substantive evidence. I opened only allowlisted
files, did not open either original Seat 1 or Seat 2 output, did not open the
original joined review, did not inspect any other rerun seat or rerun output,
did not communicate with another reviewer, and did not open any `superseded/`
or other nested packet file. I listed the permitted packet directories only to
identify their top-level regular files. Crosswalks were used only as navigation;
no finding below treats a crosswalk assertion as evidence.

Producer disclosure, reproduced verbatim: "receipt schema field naming is inconsistent across packets (authority_sha256 vs product_contract_hash); values are verified correct; classify as the seats see fit."

I classify that naming inconsistency as **MINOR, disclosure-only** because the
values are stipulated verified correct; it still should be normalized before a
machine consumer relies on one field name.

No model was trained or fitted, no new session statistic was computed, no
finding was resolved, no FT2-21 work was started, and no market data was opened
beyond the allowlisted tabulated packet files.

## A. FINDINGS VERIFICATION

### 1. Original [BLOCKING] — Simulator v5 does not enforce the safety game FT2-11 assigns to it

**Verdict: ANSWERED at the design-contract level.**

The repair no longer pretends that current v5 is the required authority.
`protocol101_ft2_08_data_tensor_label_contract/replay_authority_v5_1_spec.json`
at `/base_simulator/modified_by_ft2_08_repair` explicitly says `false`, while
`/required_v5_1_behavior` requires t+1 fills, D48, D49, the soft close,
complete-ladder WAIT, dedicated open-contract state, and exact forced flat.
`/fee_trajectories` requires separate causal `$3` and `$4` replays, and
`/activation_gate` bars authority status until FT2-30 implementation and FT2-31
independent acceptance pass.

That acknowledgement is accurate: current
`v4/model/protocol101_serial_simulator_v5.py` still declares
`STRESS_APPLICATION = "metrics_only"` at line 39, reports
`quote_age_gate: False` at lines 626-633, and enforces only the old daily stop
and affordability path at lines 771-818. FT2-11 now consumes the repaired
design correctly:
`protocol101_ft2_11_evidence_statistics_contract/evidence_standard.json`
`/two_layer_acceptance/entry_component_acceptance/all_required/signed_safety`
requires independent D48/D49 enforcement and separate causal fee trajectories,
with wrong replay classified `invalid_evidence`. The old source is therefore
not silently blessed; an unimplemented v5.1 cannot activate.

### 2. Original [BLOCKING] — The live D48/D49 account state is neither frozen nor reconstructable

**Verdict: PARTIALLY ANSWERED. Residual severity: MATERIAL.**

Most of the missing ledger contract is now present.
`protocol101_ft2_08_data_tensor_label_contract/account_state_ledger_spec.json`
freezes integer cents and rounding at `/numeric_unit` and `/rounding`; paper
account scope and NetLiquidation logging at `/session_start`; exact
fill/commission ordering, partial-fill reservations, cancel/reject handling,
late-commission reserves, and realized-PnL definitions at
`/owned_trade_ledger`; D48/D49 arithmetic at `/d48` and `/d49`; restart
reconciliation at `/restart_recovery`; and reconstructable per-decision
primitives at `/per_decision_audit_fields`. `/policy_comparison` also gives
candidate and comparator independent causal ledgers.

The residual is the session-start equity primitive. At
`/session_start/live_paper_value`, the repair says “first authenticated
pre-open ... snapshot at or before 09:30” but gives no lower freshness bound,
session-date restriction, rule for choosing among multiple snapshots, or
dedicated-account/external-position exclusion. `/session_start/account_scope`
only says configured paper account. A stale earlier snapshot or unrelated
position in that account can therefore change D48/D49 while satisfying the
text. The ledger is substantially reconstructable after a snapshot is chosen,
but the live/offline starting primitive is not yet deterministic or same-game.

### 3. Original [BLOCKING] — Historical-to-IBKR transfer is assumed as a Boolean, not specified as a gate

**Verdict: ANSWERED as a measurement contract.**

`protocol101_ft2_10_entry_science_contract/calibration_spec.json`
`/source_transfer_gate` now defines a candidate-specific historical-versus-IBKR
population rather than two implementations consuming one live observation. It
freezes permitted governance classes, forbidden roles, an exact pair key,
minimums of three sessions and 500 paired flat decisions, natural strata,
exact and FP32 tolerances, `0.99` candidate action agreement, and a nearest-rank
p95 cluster-gap artifact. Insufficient evidence blocks live activation.

`protocol101_ft2_11_evidence_statistics_contract/shadow_sufficiency_spec.json`
`/historical_to_IBKR_source_transfer` independently repeats the population,
pairing, minimums, tolerances, pass/insufficient/fail/invalid outcomes, and
candidate-specific artifact. It expressly says at
`/historical_to_IBKR_source_transfer/same_live_observation_two_implementation_planes_is_not_source_transfer`
that the old two-plane shortcut is not source transfer. A separate graph-order
defect in this gate is reported in Fresh Scan finding B3; it does not erase the
fact that the original missing measurement law is now specified.

### 4. Original [BLOCKING] — Exact IBKR identity and ATM recentering are underdefined

**Verdict: ANSWERED.**

`protocol101_ft2_08_data_tensor_label_contract/identity_mapping_spec.json`
defines the source-neutral key at `/source_neutral_contract_key`, its canonical
serialization at `/canonical_contract_id`, historical and IBKR alias records at
`/historical_alias_record` and `/ibkr_alias_record`, and fail-closed IBKR
definition checks at `/ibkr_definition_fail_closed`. `/atm_and_ladder` freezes
the causally lagged SPX source, `int(round(spx/5)*5)`, Python half-to-even ties,
the 21 strikes, right order, and slot linearization. `/recentring` requires
identity joins and forbids slot-history inheritance. The exact-half-step,
wrong-class, wrong-expiry, alias, recenter, and outside-ladder cases are named
under `/required_acceptance_fixtures`.

`protocol101_ft2_08_data_tensor_label_contract/storage_spec.json`
`/contract_path_key`, `/label_join_key`, and `/key_validation` persist redundant
identity fields and fail closed when decomposition disagrees. This is enough to
make historical symbols and IBKR `conId` aliases auditable without treating
either alias as identity.

### 5. Original [BLOCKING] — The frozen tensor does not define identical field semantics, and its E-family Greeks have no contract axis

**Verdict: ANSWERED at the frozen-design level.**

`protocol101_ft2_08_data_tensor_label_contract/tensor_schema.json`
`/market_history/values` now keeps only contract-invariant fields on `[90,11]`,
while `/contract_path_core` places right/strike-dependent alignment fields,
D-family fields, and `E.bs.delta`/`E.bs.gamma` on `[90,42,10]` with exact
identity alignment. Raw vendor Greeks are forbidden.

`protocol101_ft2_08_data_tensor_label_contract/field_semantics_manifest.json`
freezes completed-minute endpoints and provenance under
`/completed_minute_law` and `/source_provenance_required_per_value`; formulas,
units, quantization, internal-Greek constants, and expiry clock under
`/market_context_axis` and `/contract_axis`; and explicitly resolves the old
contract-free E values under
`/ft2_20_legacy_field_crosswalk/global_e_family_resolution`.
`synthetic_golden_vectors.json:/vectors` contains separate call and put expected
Greeks plus timing, recenter, complete-ladder, outside-ladder, and D49 boundary
vectors. The repair therefore supplies both the contract axis and the requested
hashed semantics/golden-vector mechanism.

### 6. Original [BLOCKING] — The complete-ladder safety rule is missing from action-mask assembly

**Verdict: ANSWERED.**

`protocol101_ft2_08_data_tensor_label_contract/tensor_schema.json`
`/flat_state_masks/global` adds `complete_ladder_mask`, warmup, context,
soft-close, and daily-stop state. `/flat_state_masks/action_mask` requires every
per-contract and global mask and says an incomplete ladder makes all 42 BUY
actions false while WAIT stays true.

`protocol101_ft2_10_entry_science_contract/composer_spec.json`
`/complete_ladder_and_safety` consumes that mask first, forbids mask use as
alpha, and fails closed. For open positions,
`protocol101_ft2_11_evidence_statistics_contract/shadow_sufficiency_spec.json`
`/runtime_integrity/global_complete_ladder` preserves HOLD/floor/forced-flat
precedence rather than incorrectly applying the flat-entry abstention rule to
the existing position.

### 7. Original [BLOCKING] — Open-state rows lose the open contract when recentering moves it outside the 42-slot ladder

**Verdict: ANSWERED.**

`protocol101_ft2_08_data_tensor_label_contract/tensor_schema.json`
`/open_state/dedicated_open_contract_record` freezes exact identity, entry
clocks/prices, current quote timestamps, a dedicated 90-minute identity-keyed
history, and independence from the current ladder. It expressly requires the
record outside the current ladder.

`protocol101_ft2_08_data_tensor_label_contract/storage_spec.json`
`/partitions/open_contract_paths` persists the exact open-position history and
quotes independently of recentering.
`replay_authority_v5_1_spec.json:/required_v5_1_behavior/open_contract` and
`/required_boundary_fixtures` require execution of an open contract outside the
recentered ladder. The old loss of lifecycle state is no longer permitted by
the repaired schema.

### 8. Original [BLOCKING] — The admitted-price activation path does not require the owner-signed amendment needed to override the 17-feature quarantine

**Verdict: ANSWERED.**

`protocol101_ft2_08_data_tensor_label_contract/tensor_schema.json`
`/admitted_price_extension/activation_requires_all` requires FT2-25 producer
pass, FT2-26 independent acceptance, an **owner-signed synchronization
amendment hash**, and candidate-bundle pinning. `/missing_signed_amendment` is
fail closed, and the initial state keeps all channels inactive.

This matches the signed boundary in
`PROTOCOL101_SCOPED_SYNCHRONIZATION_DECISION_2026_07_25.md`, “Explicitly Not
Authorized As Initial Alpha” and “Scope Boundary,” and is stricter than the
consolidated authority’s Appendix A R2 wording about an owner-*signable* draft.
The repaired machine law requires the actually signed amendment.

### 9. Original [MATERIAL] — Completed-minute, freshness, and opening-history laws are not executable definitions

**Verdict: ANSWERED, subject to the new 15:55 conflict in Fresh Scan B4.**

`protocol101_ft2_08_data_tensor_label_contract/field_semantics_manifest.json`
`/completed_minute_law` freezes `(t-1 minute,t]`, option and index/context
cutoffs, t+1 BUY/EXIT fills, first descriptive mark, last entry and learned-exit
decisions, and forced-flat boundary. `/quote_completeness_and_freshness` freezes
both quote legs, same completed state, exact identity, a 90,000 ms maximum age,
and no-substitution behavior. `/opening_and_reconnect_law` requires 15
consecutive complete minutes, resets on reconnect/delay, forbids prior-session
backfill, and leaves open-position safety active during a reset.

`tensor_schema.json:/decision_convention` and
`/flat_state_masks/global` make those laws part of the runtime action mask.
The old 09:32-versus-15-minute warmup divergence and unspecified stale-quote
threshold are therefore answered.

### 10. Original [MATERIAL] — The primary entry comparison has no serial flat-state population compatible with live D49 and occupancy

**Verdict: PARTIALLY ANSWERED. Residual severity: MATERIAL.**

`protocol101_ft2_10_entry_science_contract/objective_spec.json`
`/serial_primary_population` now freezes the universal grid, the six structural
states, occupied/pending treatment, WAIT and failed-fill accounting, t+1
commit/fill behavior, separate policy ledgers, D48/D49/soft-close/daily-stop
state, separate `$3`/`$4` trajectories, neutral lifecycle, re-entry law, and
required ledger hashes. `/p5_under_cap_algorithm` defines an exact causal P5
comparator with its own ledger. FT2-11 adopts the population without changing
it at `evidence_standard.json:/authoritative_population`.

The residual is live compatibility: the chosen neutral lifecycle holds every
successful entry to 15:55
(`objective_spec.json:/serial_primary_population/neutral_lifecycle`), so one
successful entry structurally eliminates all later same-session flat decisions.
`mde_spec.json:/serial_trade_count_projection` confirms a one-trade/session
cap. The eventual learned lifecycle can exit and re-enter earlier, so this
component ledger is now fully specified but is not a proof that the selected
entry behavior, D49 path, or WAIT population matches the eventual live combined
trader. Fresh Scan B5 records the additional census-rate contradiction created
by this choice.

## B. FRESH SCAN

### B1. [BLOCKING] Decision-time D48/D49 action masks require the unknowable actual t+1 ask

The repair correctly separates decision `t` from fill `t+1`, but it does not
freeze a causal decision-time safety price. Instead,
`protocol101_ft2_08_data_tensor_label_contract/account_state_ledger_spec.json`
`/d48/prospective_entry_cost_cents` defines the D48 mask from the **t+1
executable ask**, and `/d49/mask_true` uses that same prospective cost.
`/per_decision_audit_fields` even requires
`prospective_entry_ask_cents_by_slot`.

Those booleans are consumed before action selection:
`tensor_schema.json:/flat_state_masks/per_contract` includes D48/D49 in the
action mask, and
`protocol101_ft2_10_entry_science_contract/composer_spec.json`
`/composer_order` and `/complete_ladder_and_safety/per_contract_requirements`
apply them before ranking. At live decision `t`, the actual completed-minute
`t+1` ask does not yet exist.

`objective_spec.json:/serial_primary_population/entry_commit_and_fill/fill_time_rechecks`
does separately and correctly recheck actual fill-time affordability, D48, and
D49. What is missing is a distinct causal `t`-quote commitment mask followed by
that `t+1` recheck. As written, historical composition can use the future ask
to remove actions that live composition must expose, or live can invent a
different preliminary rule. This is direct historical/live action-set
divergence and future leakage.

### B2. [BLOCKING] The repaired `$103` soft close is not the owner-authorized ladder-dependent predicate

The consolidated authority §1.4, Owner amendment A3, says soft close occurs
when the remaining budget cannot afford **any otherwise eligible contract**
whose executable premium is at least `$1.00`. That predicate depends on the
current eligible ladder and the active fee trajectory.

The repair replaces it with a constant:
`account_state_ledger_spec.json:/d49/soft_close_trigger` is merely
`remaining < 10300`, and
`objective_spec.json:/serial_primary_population/account_and_safety_state/soft_close`
repeats that fixed `$103` rule. These are not equivalent. For example, with
`$300` remaining and no otherwise eligible contract cheaper than `$4.00`, the
authority soft-closes; the repaired predicate does not. The error also ignores
that the separate `$4` stress trajectory has a `$104`, not `$103`, minimum
`$1 + fee` cost. That conflicts with
`replay_authority_v5_1_spec.json:/fee_trajectories/reason`, which says the fee
change can alter soft-close timing.

This changes permanent session state, later WAIT behavior, and offline/live
safety. It is not a reporting detail.

### B3. [BLOCKING] The source-transfer prerequisite is required before the graph node that performs it

Graph V2 defines `FT2-92-IBKR-DECISION-SHADOW` as the candidate-specific
historical/IBKR tensor-and-action transfer node
(`PROTOCOL101_FULL_TRADER_GRAPH_V2.json`, node
`FT2-92-IBKR-DECISION-SHADOW`) and routes a protected-holdout pass into FT2-92.

But
`protocol101_ft2_11_evidence_statistics_contract/shadow_sufficiency_spec.json`
`/historical_to_IBKR_source_transfer/activation_boundary` says the transfer
pass is mandatory **before FT2-92** or any Phase-F live observation. The same
object says the pass is the output of the candidate-specific transfer battery.
Thus the graph reaches FT2-92 to produce evidence that the contract requires
already to exist before FT2-92. The detailed measurement law is good, but the
activation topology is circular and cannot be executed without violating one
of the two authorities.

### B4. [MATERIAL] The open-state schema creates an impossible learned EXIT at 15:55

`field_semantics_manifest.json:/completed_minute_law` correctly says the last
learned EXIT decision is 15:54 and forced-flat fill is 15:55.
`label_join_spec.json:/timing` agrees that an EXIT decision at `v` fills at
`v+1`. Yet `tensor_schema.json:/open_state/actions_at_1555` exposes `["EXIT"]`.

Unless this is explicitly typed as a non-model forced-flat event, it is an EXIT
decision at 15:55 whose lawful t+1 fill would be 15:56, after the forced-flat
boundary. The replay spec’s no-duplication rule does not define how the runtime
distinguishes that schema action from forced flat. The row should carry a
forced-flat terminal event/state, not a learned EXIT action with impossible
timing.

### B5. [MATERIAL] The neutral serial game and census trade-rate selection rule contradict each other

`objective_spec.json:/serial_primary_population/neutral_lifecycle` and
`mde_spec.json:/serial_trade_count_projection` make the component game at most
one successful trade per session. `mde_spec.json:/serial_trade_count_projection/census_trade_rate_role`
correctly says the multi-trade census rate is incompatible with neutral-lifecycle
trade-count projection.

Nevertheless,
`protocol101_ft2_10_entry_science_contract/calibration_spec.json`
`/joint_conservatism/selection_order` discards settings outside a census-derived
trade-rate band, and
`/joint_conservatism/design_trade_rate_band_mean_trades_per_session` sets that
band to `[0.65,5.2]` from a 2.6-trade/session oracle diagnostic. Under a
one-trade ceiling, the lower bound is effectively a hard requirement to enter
on at least about 65% of sessions, despite the authority allowing zero-trade
days and forbidding a hard daily frequency policy. The upper half of the band
is unreachable. Marking it `hard_component_freeze_gate:false` does not cure the
problem because the preceding selection rule says to discard out-of-band
settings before the frozen objective selects a component.

This can prefer an overactive early-entry component and makes the repaired
census-rate use internally inconsistent with the repaired MDE/population law.

### B6. [MATERIAL] The one-strike “cluster-aware” WAIT fix can still deadlock a smooth directional surface

`composer_spec.json:/uncertainty_wait/adjacent_strike_cluster/membership`
defines a substitute cluster as same right and expiry within only `5` strike
points. `/rules` then forces WAIT unless this cluster beats the best contract
outside the cluster by the full model-plus-source margin.

A same-right contract two ladder steps away (`10` points) is therefore an
outside competitor even when it is an economically interchangeable member of a
broad directional surface. If scores are smooth across three or more strikes,
the ±10 substitute recreates the original near-tie WAIT deadlock. Reusing the
same cluster definition for residual calibration makes the calculation
consistent but does not validate that ±5 is the full substitutability cluster.
The repair protects only the nearest adjacent strike, not the stated broader
failure mode.

### B7. [MATERIAL] MNAR candidate stability does not say whether the label-dependent machinery is refit

The primary full-loss convention itself is coherent:
`protocol101_ft2_04_path_label_freeze/label_spec.json`
`/missing_minute_rule` values every non-executable bid at full loss, forbids
forward fill, and uses exact 15:55 zero value. The allowlisted census table
`protocol101_ft2_05_opportunity_census/mnar_sensitivity.csv` also supplies the
reported label-surface sensitivity.

The candidate-level repair is underdefined.
`protocol101_ft2_11_evidence_statistics_contract/evidence_standard.json`
`/mnar_no_bid_sensitivity/required_evaluation_output` says to “replay and
report” the serial population under both label views and demands identical
selected identity, terminal, selected-pair order, and Spearman rank. It does
not say whether path-head models, label-fitted CDFs, calibrators, and composer
thresholds are refit under the no-bid-excluded label convention or whether one
primary-trained frozen model is merely rescored.

Those procedures answer different MNAR questions because
`objective_spec.json:/nested_empirical_cdf/reference_population` and the
forecast/calibration machinery are fitted from labels. Without one frozen
choice and matched identity rules, two compliant implementations can return
different rankings and terminals.

### B8. [MATERIAL] Live session-start equity can include stale or unrelated account state

This is the residual from original finding 2.
`account_state_ledger_spec.json:/session_start/live_paper_value` has no maximum
age or deterministic selection rule for the pre-open NetLiquidation snapshot,
and `/session_start/account_scope` does not require a dedicated Protocol101
paper account or block unrelated positions/cash flows.

Offline D48/D49 starts from carried simulator cash. Live can start from a stale
or externally changed whole-account NetLiquidation value. Logging the raw text
and timestamp makes the divergence reconstructable after the fact, but it does
not make the historical and live safety game identical.

### Checks completed without an additional defect

- **t+1 label windows:** `label_spec.json:/horizon_window_definition` is
  internally exact: `e=t+1`, marks satisfy `e<u<=e+h`, the first mark is `t+2`,
  and a full h-minute horizon has h marks. The 15:29 BUY/15:30 fill and
  15:54 EXIT/15:55 fill endpoints also agree across FT2-04 and FT2-10. The
  separate 15:55 schema-action conflict is B4.

- **No-bid full loss:** primary economics, path contribution, profitable-run
  breaking, first-profit exclusion, and exact-boundary zero valuation agree
  across `label_spec.json:/missing_minute_rule`,
  `label_join_spec.json:/no_bid`, and
  `replay_authority_v5_1_spec.json:/required_v5_1_behavior/forced_flat`.
  The remaining problem is only the underdefined candidate-level alternate
  fitting procedure in B7.

- **Premium-band CDF:** the core CDF law is causal and nested.
  `objective_spec.json:/nested_empirical_cdf` assigns bands from the decision-t
  ask, balances sessions, freezes midrank ties and interpolation, excludes the
  scored/future blocks, and fails closed without same-band evidence.
  `composer_spec.json:/stage_2_conservative_upside_rank/premium_band_at_decision`
  uses the same boundaries. I found no additional live-parity defect in that
  mapping.

- **Component versus promotable D1 claims:** the text keeps the layers separate.
  `evidence_standard.json:/two_layer_acceptance/entry_component_acceptance`
  denies selection eligibility and incremental-edge claims, while
  `/two_layer_acceptance/promotable_combined_system` reserves strict
  candidate-minus-random and candidate-minus-P5 dollar evidence for FT2-80.
  No component-only promotable claim is made in the repaired text.

- **Rebuilt MDE and census bound:** the allowlisted table arithmetic is
  internally consistent. `oracle_ceiling_summary.csv` reports hold-to-flat
  oracle PnL `$89,035`, P5 PnL `-$7,205`, and 45 sessions; their difference
  divided by 45 is the `$2,138.6666666666665/session` row in
  `session_variance_components.csv` and the planning ceiling in
  `mde_spec.json:/plausible_detectable_improvement_bound`.
  `mde_spec.json:/pre_tranche_1_pilot` correctly forbids substituting that
  census variance for measured candidate-minus-comparator pilot variance.

- **Why the P5 best-session ceiling share moved from roughly 31% to 46%:**
  `v1_v2_impact.json:/oracle_changes` gives v1 oracle/P5 PnL
  `$377,161/$117,454` and v2 `$341,796/$156,830`; those pairs reproduce
  approximately `31.1%` and the `45.884%` recorded in
  `oracle_ceiling_summary.csv`. The share rose because the numerator of the
  ratio (P5 PnL) increased by `$39,376` while its oracle denominator decreased
  by `$35,365`. The packet also correctly warns at
  `v1_v2_impact.json:/interpretation` that fills, missingness, role membership,
  and budget semantics all changed and the movement cannot be assigned to one
  change. The arithmetic is consistent; it is not a stable model-edge statistic
  and does not explain causal contribution by amendment.

## FILES ACTUALLY OPENED

- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_20_parallel_design_review/seat3_live_parity.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_repair_attempt001/findings_crosswalk.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/findings_crosswalk.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/findings_crosswalk.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/contracts/PROTOCOL101_FULL_TRADER_GRAPH_V2_CONSOLIDATED_AUTHORITY_2026_07_28.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/execution/PROTOCOL101_FULL_TRADER_GRAPH_V2.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/contracts/PROTOCOL101_TRADER_CHARTER.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/contracts/PROTOCOL101_G4_HOLDOUT_REVISION_2026_07_19.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/contracts/PROTOCOL101_G8_CALIBRATION_REVISION_2026_07_26.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/contracts/PROTOCOL101_D1_NEGATIVE_CONTROL_AND_INCREMENTAL_EDGE_AMENDMENT_2026_07_28.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/synchronization/contracts/PROTOCOL101_SCOPED_SYNCHRONIZATION_DECISION_2026_07_25.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/contract.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/tensor_schema.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/field_semantics_manifest.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/identity_mapping_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/account_state_ledger_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/replay_authority_v5_1_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/storage_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/synthetic_golden_vectors.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/label_join_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/receipt.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/validation.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_04_path_label_freeze/label_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_04_path_label_freeze/oracle_rules.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_04_path_label_freeze/report.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_04_path_label_freeze/intersection_proof.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_04_path_label_freeze/receipt.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/contract.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/objective_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/composer_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/calibration_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/forecast_heads.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/controls_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/receipt.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/contract.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/evidence_standard.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/mde_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/shadow_sufficiency_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/receipt.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/report.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/census_results.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/v1_v2_impact.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/oracle_replay_audit.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/oracle_ceiling_summary.csv`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/minimum_detectable_improvement.csv`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/session_variance_components.csv`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/mnar_sensitivity.csv`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/friction_by_premium_band.csv`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/receipt.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/governance_receipt.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/model/protocol101_serial_simulator_v5.py`
<!-- END VERBATIM SEAT 3 -->
