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
