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
