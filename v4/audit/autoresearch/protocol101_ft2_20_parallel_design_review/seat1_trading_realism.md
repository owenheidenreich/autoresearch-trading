# Protocol101 Trading-Realism Adversarial Review

**Verdict: not design-ready.** The frozen contracts can overstate execution quality, evaluate impossible overlapping entries, suppress a material share of big winners, and drift toward either a scratch-mill or post-loss lottery exposure.

### [BLOCKING] No-bid periods are erased instead of treated as losses

**Anchor:** [FT2-08 `label_join_spec.json`](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/label_join_spec.json), key `censoring.no_bid_or_stale_future_minute`; [FT2-08 `contract.md`](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/contract.md:327), §9.

No-bid minutes are excluded from aggregates and break continuity. The inherited oracle also permits hold-to-flat to use the last available bid before 15:55 when the boundary bid is masked. For a cheap SPXW wing, loss of the bid is often the economic event itself. Removing it makes adverse excursion, underwater burden, profitable-window stability, and forced-flat PnL look better precisely for lottery-like contracts.

**Acceptance test:** prove conservative liquidation accounting for no-bid states and prohibit forced-flat valuation at an earlier bid that was not acted upon.

### [BLOCKING] Entry and exit fills use prices known before the action can execute

**Anchor:** [FT2-08 `tensor_schema.json`](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/tensor_schema.json), key `decision_convention.option_information_cutoff`; [FT2-08 `label_join_spec.json`](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/label_join_spec.json), key `economic_primitives`; [FT2-11 `evidence_standard.json`](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/evidence_standard.json), key `entry_acceptance_all_required.fees_and_stress`.

The model may consume option information from completed minute `t` and then buy at that same minute’s ask. Exit labels similarly observe a completed-minute bid and assume sale at that bid. No maximum entry-quote age, post-inference timestamp ordering, adverse price movement, or delayed-fill stress is frozen. FT2-11’s only hard execution stress changes the fee from $3 to $4. In a fast SPX move, the displayed ask or bid that caused the action is commonly gone before the order reaches the market.

**Acceptance test:** require post-decision executable quotes or a preregistered latency/slippage replay stratified by spread, premium, phase, and fast-market state.

### [BLOCKING] The primary entry test is not the required one-account serial game, making P5-under-cap unfair

**Anchor:** [FT2-10 `objective_spec.json`](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/objective_spec.json), keys `decision_local_entry_metric.occupancy_or_exit_assumption` and `p5_comparison.paired_session_metrics`; [FT2-11 `evidence_standard.json`](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/evidence_standard.json), keys `primary_metric.definition` and `entry_acceptance_all_required.signed_safety`.

`DeltaQ` scores every governed flat-state minute with no occupancy assumption. Serial economics are checked separately by holding accepted entries to 15:55. A policy can therefore earn its hard entry claim from many hypothetical BUY decisions that could never execute while the account was already occupied. This conflicts with Graph V2’s FT2-50 requirement for one-account serial entry evaluation.

P5 fairness is additionally unresolved: the target names “P5 VWAP-side nearest-ATM under its frozen historical rule” but freezes no exact timing, tie, WAIT, or policy-specific D49 state algorithm. “Same masks” cannot mean the same D49 risk set after two policies have generated different prior trades and realized losses.

**Acceptance test:** recompute the hard candidate-versus-P5 claim on synchronized, policy-causal serial ledgers with an exact hashed P5-under-cap algorithm.

### [BLOCKING] The composer can structurally delete the Charter’s big-win column

**Anchor:** [FT2-10 `composer_spec.json`](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/composer_spec.json), keys `stage_1_quality_screen.guardrail_anchor_grid`, `census_provenance.excluded_winners`, and `stage_2_conservative_upside_rank`; [FT2-11 `tripwire_spec.json`](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/tripwire_spec.json), key `four_bucket_profile.big_win_collapse_tripwire`.

At the 0.25 anchor, the realized-label screen already excludes 19,366 of 81,899 big-win paths, or **23.646%**, before forecast error and uncertainty abstention. The 0.30 setting excludes 30.671%; even 0.20 excludes 16.768%. The grid contains no zero-screen control, while the objective rewards `Q` and conservative q10 upside, not tail capture. The profile tripwire activates only below 2% big wins, so a roughly 2% big-win policy can avoid even owner review despite the Charter’s 18% north star.

The 0.25 anchor is not defensible as “loose.” The available range gives the optimizer multiple ways to prefer smooth, quickly profitable paths while discarding red-before-green home runs. This quietly rebuilds the forbidden scratch-mill.

**Acceptance test:** require the selected operating point to preserve economically material big-win incidence and positive-PnL concentration relative to a no-screen composer.

### [MATERIAL] Dual-unit `min()` arbitration biases selection toward middle-premium contracts

**Anchor:** [FT2-10 `composer_spec.json`](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/composer_spec.json), key `stage_2_conservative_upside_rank.dollar_percent_arbitration`; [FT2-10 `objective_spec.json`](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/objective_spec.json), key `decision_local_entry_metric.component_transform.dual_unit_arbitration`.

Dollar PnL and return on premium are mechanically linked through entry premium. Taking the worse percentile suppresses cheap convex contracts on dollars and expensive high-delta contracts on percentage, favoring the premium region where the two ranks happen to cross. Because percentiles are phase-conditioned rather than premium- or moneyness-conditioned, this is a systematic strike-selection prior, not neutral protection against one-axis gaming.

`Acceptable with documentation: no` because documentation cannot remove a deterministic premium and strike bias from the hard composer.

### [MATERIAL] D49 “size-down” can force post-loss lottery exposure and is not independently enforced by simulator v5

**Anchor:** [FT2-08 `tensor_schema.json`](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/tensor_schema.json), key `masks[d49_budget_mask]`; [FT2-08 `contract.md`](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/contract.md:409), §11; [FT2-11 `evidence_standard.json`](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/evidence_standard.json), key `entry_acceptance_all_required.signed_safety`.

With one-contract sizing, lower premium is not ordinary size reduction. After a loss, D49 progressively removes higher-premium, higher-delta contracts and leaves cheap wings, especially late in the day. The frozen census reports 11.4% median friction for sub-$1 contracts. That can turn “soft landing” into repeated low-delta, high-friction lottery attempts. The supplied simulator independently enforces affordability and the realized daily stop, but has no D48/D49 configuration or replay rejection.

`Acceptable with documentation: no` because the defect changes the eligible action set after losses and weakens independent safety verification.

### [MATERIAL] Exact-contract uncertainty can deadlock valid directional sessions

**Anchor:** [FT2-10 `composer_spec.json`](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/composer_spec.json), key `uncertainty_wait`; [FT2-10 `calibration_spec.json`](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/calibration_spec.json), keys `joint_conservatism.design_trade_rate_band_mean_trades_per_session` and `action_conditioned_gate.wait_reliability`; [FT2-11 `shadow_sufficiency_spec.json`](/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract/shadow_sufficiency_spec.json), key `agreement_gates.flat_state_WAIT_or_ENTER`.

Adjacent SPXW strikes often represent the same directional thesis and have nearly tied scores. Requiring the top exact contract to beat its nearest substitute by a 90% model-gap error plus p95 source drift can force WAIT even when the whole cluster has strong positive edge. The trade-rate band is not a hard gate, and WAIT calibration is self-referential to the same realized composer rather than missed big wins. Shadow agreement proves reproducibility, not that a deadlocked session was economically correct.

`Acceptable with documentation: no` because systematic abstention from valid clusters is a behavioral failure, not a caveat.

## FILES ACTUALLY OPENED

- `/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/contracts/PROTOCOL101_FULL_TRADER_GRAPH_V2_CONSOLIDATED_AUTHORITY_2026_07_28.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/execution/PROTOCOL101_FULL_TRADER_GRAPH_V2.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/contracts/PROTOCOL101_TRADER_CHARTER.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/contracts/PROTOCOL101_G4_HOLDOUT_REVISION_2026_07_19.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/contracts/PROTOCOL101_G8_CALIBRATION_REVISION_2026_07_26.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/synchronization/contracts/PROTOCOL101_SCOPED_SYNCHRONIZATION_DECISION_2026_07_25.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/docs/protocol101/training/contracts/PROTOCOL101_D1_NEGATIVE_CONTROL_AND_INCREMENTAL_EDGE_AMENDMENT_2026_07_28.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_04_path_label_freeze/label_spec.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_04_path_label_freeze/oracle_rules.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/report.md`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/census_results.json`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/excluded_winners.csv`
- `/Users/gduby/Documents/autoresearch-trading/v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/friction_by_premium_band.csv`
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
- `/Users/gduby/Documents/autoresearch-trading/v4/model/protocol101_serial_simulator_v5.py`
