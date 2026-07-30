# Protocol101 Live-Parity Adversarial Review

**Verdict: NOT DESIGN-ACCEPTABLE.** The frozen set has eight blocking and two material live-parity defects. No files were changed and no market-session statistics were computed.

[BLOCKING] **Simulator v5 does not enforce the safety game FT2-11 assigns to it**

**Anchor:** [FT2-11 contract.md](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/contract.md:177), §4.5; [evidence_standard.json](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/evidence_standard.json:100), `/entry_acceptance_all_required/fees_and_stress` and `/signed_safety`.

FT2-11 requires candidate and comparator replays through simulator v5 with identical D48/D49 safety at both `$3` and `$4`. Simulator v5 has no D48 cap or prospective D49 remaining-budget check. It rejects only after the daily stop is already reached and for ordinary affordability. Its `$4` stress is `metrics_only`: realized cash, session PnL, and subsequent eligibility continue on the `$3` path. The FT2-05 `oracle_replay_audit.json` confirms D48/D49 were applied before the simulator, so v5 cannot independently catch an incorrect pre-mask.

A stream can therefore include a premium above 5% or an entry whose premium exhausts the remaining daily budget and still pass v5. Under `$4`, the extra fee can alter later D49 eligibility and equity minima, but v5 never replays that causal path.

**Acceptance test needed:** independently enforce D48 and D49 inside the named replay authority, and run `$3` and `$4` as separate causal account trajectories with boundary, accumulated-loss, recovery, and prospective-budget fixtures.

[BLOCKING] **The live D48/D49 account state is neither frozen nor reconstructable**

**Anchor:** [FT2-08 tensor_schema.json](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/tensor_schema.json:357), mask objects `d48_premium_cap_mask` and `d49_budget_mask`; [FT2-08 contract.md](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/contract.md:239), §6.6.

Only Boolean results are stored. The contract does not freeze the live source or timestamp for session-start equity, whether equity means simulator cash or an IBKR account value, account scope, cent rounding, realized-loss event ordering, commission arrival, partial fills, restart recovery, or unavailable-account-state behavior. D49 says state is injected, but no underlying equity/PnL fields or event-ledger identity are required.

The historical source uses cash carried between sessions and realizes label PnL at occupancy exit. A live guard could instead snapshot NetLiquidation at first connection or consume lagging broker realized PnL. Both could emit plausible Boolean masks while playing different games.

**Acceptance test needed:** freeze and log exact integer-cent account primitives, session-reset timestamp, broker field mapping, fill/commission sequencing, restart rules, and per-decision D48/D49 arithmetic; reproduce masks from that ledger independently.

[BLOCKING] **Historical-to-IBKR transfer is assumed as a Boolean, not specified as a gate**

**Anchor:** [FT2-10 composer_spec.json](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/composer_spec.json:139), `/uncertainty_wait/source_transfer_error`; [FT2-11 shadow_sufficiency_spec.json](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/shadow_sufficiency_spec.json:7), `/candidate_state/historical_IBKR_tensor_and_action_transfer_passed` and `/paired_planes`.

FT2-11’s reference and runtime planes operate on the same live observations. That can test two implementations, but not historical-vendor versus IBKR source equivalence. The prerequisite historical/IBKR pass is merely asserted, while FT2-10 does not define the pairing population, tensor-field tolerances, mask/identity requirements, minimum evidence, or p95 score-gap calculation. The graph’s FT2-92 node name supplies no acceptance law.

This permits perfect shadow agreement even when both planes share an IBKR-specific tensor that differs materially from training. The signed synchronization decision explicitly says the feeds were not certified identical and covered only the original 17-feature scope.

**Acceptance test needed:** a frozen paired-source battery covering every active field, mask, slot map, score, and action, stratified across opening history, recentering, missing quotes, reconnects, extension channels, and open-state rows.

[BLOCKING] **Exact IBKR identity and ATM recentering are underdefined**

**Anchor:** [FT2-08 contract.md](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/contract.md:79), §§4.1–4.3; [storage_spec.json](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/storage_spec.json:7), `/contract_path_key` and `/key_validation`.

The contract requires `contract_id_utf8` to decompose consistently, but never defines its canonical construction or mapping between historical symbols and IBKR `conId`/contract details. The identity omits auditable SPXW trading class, symbol, multiplier, currency, and exchange. Canonical ATM calculation, spot timestamp, five-point rounding, and midpoint tie-breaking are also absent.

Simulator v5 weakens this further: `SerialCandidateV5` carries `contract_id`, right, and a mutable canonical slot, but not exact expiry or strike, and preflight does not decompose or validate SPXW 0DTE identity. A recentered slot can therefore resolve to a different live contract without the replay detecting it.

**Acceptance test needed:** freeze a source-neutral contract key, historical and IBKR alias maps, complete IBKR definition checks, ATM rounding/tie law, and recenter/re-entry fixtures proving no slot inheritance.

[BLOCKING] **The frozen tensor does not define identical field semantics, and its E-family Greeks have no contract axis**

**Anchor:** [FT2-08 contract.md](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/contract.md:169), §§6.2–6.3; [tensor_schema.json](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/tensor_schema.json:52), `/market_history/values/feature_order` and `/contract_path_core`.

The schema freezes names but not formulas, units, minute aggregation, quantization, VWAP/session reset, denominator-zero behavior, OMAR construction, D-family ladder membership, Black-Scholes constants, spot/option timestamp pairing, or normalization. It also puts `E.bs.delta` and `E.bs.gamma` into a single `[90,19]` market history while separately supplying 42 contract-specific delta/gamma paths. The global E values have no exact contract identity in a 42-action game.

Historical and live builders can consequently produce different yet schema-valid tensors, or choose different contracts for the global Greeks. This also obscures whether the per-contract Greek expansion remains within the signed exact-17-feature scope.

**Acceptance test needed:** a hashed field-by-field formula manifest with source timestamps and synthetic golden vectors, including an explicit identity for each Greek value and its relationship to the signed 17 channels.

[BLOCKING] **The complete-ladder safety rule is missing from action-mask assembly**

**Anchor:** [FT2-08 contract.md](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/contract.md:239), §6.6; [FT2-10 composer_spec.json](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/composer_spec.json:13), `/composer_order/physical_and_safety_action_mask`; [FT2-11 shadow_sufficiency_spec.json](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/shadow_sufficiency_spec.json:73), `/runtime_integrity`.

FT2-08 defines quote and eligibility masks per contract. It provides no global `complete_ladder` bit and says each contract is actionable when that contract’s checks pass. FT2-11, however, requires every incomplete-ladder boundary to take the frozen safe action, matching the signed hard-safety rule.

Historical composition can therefore rank the available subset while the live guard must abstain globally. The contracts also do not define the safe open-position action when the ladder is incomplete.

**Acceptance test needed:** freeze the global completeness predicate, subscription/reset timing, all-action masking behavior, and flat/open safe actions, then test one missing slot, stale slot, reconnect, and partial rebuild identically offline and live.

[BLOCKING] **Open-state rows lose the open contract when recentering moves it outside the 42-slot ladder**

**Anchor:** [FT2-08 contract.md](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/contract.md:266), §7; [tensor_schema.json](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/tensor_schema.json:383), `/open_state`.

The row records open-contract identity and entry state, but all quote/path tensors remain tied to the newly recentered current 42-slot ladder. There is no dedicated current quote, 90-minute path, availability mask, or slot-independent tensor for an open contract outside ±50 points.

A contract bought at 15:30 can leave the recentered ladder before 15:55. Historical lifecycle scoring and live HOLD/EXIT then lack the same open-contract state, and forced-flat execution cannot be reconstructed from the frozen row schema.

**Acceptance test needed:** preserve a dedicated identity-keyed open-position path and quote record independently of the current flat-entry ladder, including outside-ladder and reconnect fixtures through forced flat.

[BLOCKING] **The admitted-price activation path does not require the owner-signed amendment needed to override the 17-feature quarantine**

**Anchor:** [FT2-08 contract.md](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/contract.md:203), §6.4; [tensor_schema.json](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/tensor_schema.json:170), `/admitted_price_extension/activation_authority`.

The prose requires a “signed admission manifest,” but the machine authority lists FT2-25, FT2-26, and an “owner-signable” amendment when required. The graph routes directly from FT2-26 acceptance onward; FT2-26 produces an owner-signable amendment but is not an owner-signature gate. That is insufficient to amend the already owner-signed contract authorizing exactly 17 features and quarantining direct per-slot price paths.

A channel could be activated and enter historical training before the governing synchronization contract is actually amended, leaving its live availability legally and semantically unresolved.

**Acceptance test needed:** require the hash of an actually owner-signed synchronization amendment in every activation manifest and frozen candidate bundle; fail closed when absent.

[MATERIAL] **Completed-minute, freshness, and opening-history laws are not executable definitions**

**Anchor:** [FT2-08 contract.md](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/contract.md:156), §6.1; [tensor_schema.json](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/tensor_schema.json:6), `/decision_convention` and `current_quote_available_mask`; [FT2-11 shadow_sufficiency_spec.json](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/shadow_sufficiency_spec.json:73), `/runtime_integrity`.

No exact interval endpoints, event/source/receipt timestamp law, bid/ask leg completeness, maximum quote age, market-context timestamps, minimum valid-history requirement, or reconnect clean-window reset is frozen. The supporting label contract refers to `max_quote_age` without giving a value, while simulator v5 declares `quote_age_gate: false`. The signed synchronization evidence records a 15-complete-minute runtime freshness guard, but FT2-08 creates flat rows from 09:32 and allows masked short history without applying that warmup to `action_mask`.

Historical opening actions may therefore be legal while live actions are forced to WAIT. A different interpretation of the “15:55 completed minute” can also make the forced-flat decision occur after the stated deadline.

`Acceptable with documentation: no.` Documentation cannot make different decision boundaries, warmup populations, or stale-quote eligibility play the same game.

**Acceptance test needed:** freeze nanosecond interval conventions, freshness thresholds, leg timestamps, warmup/reset policy, partial-history action law, and the exact timer behavior at 09:32, 15:30, and 15:55.

[MATERIAL] **The primary entry comparison has no serial flat-state population compatible with live D49 and occupancy**

**Anchor:** [FT2-10 objective_spec.json](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/objective_spec.json:6), `/decision_local_entry_metric/occupancy_or_exit_assumption` and `/p5_comparison/identical_safety`; [FT2-11 contract.md](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/contract.md:109), §4.

FT2-10 says the primary metric has no occupancy or exit assumption and scores every governed flat-state minute, yet identical safety includes policy-dependent D49 state. Live D49 and flatness depend on prior entries, realized exits, and losses. FT2-11 introduces a neutral lifecycle for safety economics but does not say that it generates the primary DeltaQ decision population.

Historical evaluation can thus score an entry at a minute when the live policy would still be open, or construct different D49 masks for candidate and comparator without a frozen pairing rule.

`Acceptable with documentation: no.` This changes which actions exist and which minutes enter the primary denominator, not merely how results are described.

**Acceptance test needed:** freeze one serial state machine for primary entry evaluation, including occupied-minute treatment, comparator-specific state, WAIT accounting, and D49 updates, and prove its flat-decision ledger matches live behavior.

## FILES ACTUALLY OPENED

- `/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/contracts/PROTOCOL101_FULL_TRADER_GRAPH_V2_CONSOLIDATED_AUTHORITY_2026_07_28.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/execution/PROTOCOL101_FULL_TRADER_GRAPH_V2.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/contracts/PROTOCOL101_TRADER_CHARTER.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/contracts/PROTOCOL101_G4_HOLDOUT_REVISION_2026_07_19.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/contracts/PROTOCOL101_G8_CALIBRATION_REVISION_2026_07_26.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/synchronization/contracts/PROTOCOL101_SCOPED_SYNCHRONIZATION_DECISION_2026_07_25.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/contracts/PROTOCOL101_D1_NEGATIVE_CONTROL_AND_INCREMENTAL_EDGE_AMENDMENT_2026_07_28.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/model/protocol101_serial_simulator_v5.py`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/contract.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/fold_roles.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/label_join_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/receipt.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/storage_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/tensor_schema.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/calibration_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/composer_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/contract.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/controls_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/forecast_heads.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/objective_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/receipt.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/bootstrap_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/contract.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/evidence_standard.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/mde_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/multiplicity_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/receipt.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/shadow_sufficiency_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/tripwire_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_04_path_label_freeze/census_sessions.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_04_path_label_freeze/compute_census_outer_test_intersection.py`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_04_path_label_freeze/intersection_proof.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_04_path_label_freeze/label_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_04_path_label_freeze/oracle_rules.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_04_path_label_freeze/receipt.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_04_path_label_freeze/report.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_04_path_label_freeze/test_compute_census_intersection.py`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/census_results.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/excluded_winners.csv`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/family_distributions.csv`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/friction_by_premium_band.csv`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/governance_receipt.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/guardrail_curves.csv`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/guardrail_threshold_curves.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/guardrail_trade_rates.csv`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/label_build_compute.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/label_session_inventory.csv`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/minimum_detectable_improvement.csv`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/oracle_ceiling_summary.csv`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/oracle_replay_audit.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/oracle_session_results.csv`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/oracle_trade_results.csv`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/pareto_frontier.csv`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/progress.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/receipt.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/regime_headlines.csv`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/report.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/session_variance_components.csv`
