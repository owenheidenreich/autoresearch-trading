# Protocol101 FT2-20 Parallel Design Review

Node: `FT2-20-PARALLEL-DESIGN-REVIEW`

Product contract:
`893aa0664944680e053ffd12a4d44c8a798397cbeed6ad56cd682fd864d9f832`

## Routing Decision

Primary legal outcome:

```text
data_tensor_label_defect -> FT2-08-DATA-TENSOR-LABEL-CONTRACT
```

Additional defect outcomes, retained in dependency order:

```text
entry_science_defect -> FT2-10-ENTRY-SCIENCE-CONTRACT
evidence_statistics_defect -> FT2-11-EVIDENCE-STATISTICS-CONTRACT
```

The graph repairs one contract at a time. FT2-08 is first because all later
science and evidence depend on causal, live-constructible tensor, identity,
mask, and label semantics. Repairing FT2-08 does not clear or downgrade the
recorded FT2-10 or FT2-11 defects. FT2-20 must be rerun after the bounded
repairs before owner approval can be requested.

The join found:

- 18 BLOCKING findings;
- 10 MATERIAL findings;
- 0 MINOR findings;
- no exact duplicate findings eligible for deletion.

No seat severity was downgraded. No finding was resolved in this node.

## Convergent Findings

The following were reached independently and remain separately preserved:

1. **No-bid censoring:** Seat 1 classified erased no-bid periods as BLOCKING;
   Seat 2 independently classified missing-not-at-random no-bid censoring as
   MATERIAL.
2. **Non-serial entry evidence:** Seat 1 classified the decision-local
   primary test as BLOCKING; Seat 2 independently classified DeltaQ replacing
   incremental profit as BLOCKING; Seat 3 independently classified the
   missing serial flat-state population as MATERIAL.
3. **D48/D49 state and enforcement:** Seat 1 classified post-loss
   low-premium distortion as MATERIAL; Seat 3 independently produced
   BLOCKING findings for simulator enforcement and live account-state
   reconstruction.
4. **Historical/live executability:** Seat 1 classified same-minute fills as
   BLOCKING; Seat 3 independently classified timestamp, freshness, and
   opening-history definitions as MATERIAL.
5. **Contract and feature transfer:** Seat 3 found multiple BLOCKING
   identity, field-semantics, complete-ladder, open-position, and activation
   defects. These are foundational FT2-08 defects even where FT2-10 or FT2-11
   later consumes the affected fields.

## Isolation Record

The three reviews were produced by three fresh `multi_agent_v1` sessions:

| Seat | Agent ID | Forked parent context | Initial evidence |
|---|---|---:|---|
| Trading realism | `019faf28-5783-7150-ad69-f512d2318fe6` | false | Role prompt plus frozen allowlist only |
| ML/statistics | `019faf28-ab19-7473-9c13-f509f9e3ee6c` | false | Role prompt plus frozen allowlist only |
| Live parity | `019faf29-0369-73d3-a37f-123c1db15d61` | false | Role prompt plus frozen allowlist only |

All three were dispatched before any had completed. Each completed its review
before any seat output was persisted in the workspace. Therefore no reviewer
could read another seat's output. The sessions received the harness's normal
base agent instructions and the role-specific initial prompt, but no parent
conversation history, planning chat, producer summary, revision narrative, or
other seat output.

The identical evidence allowlist contained only:

- consolidated authority and Graph V2 JSON;
- the five signed source contracts named by the consolidated authority;
- top-level FT2-08, FT2-10, and FT2-11 packet files;
- top-level FT2-04 and FT2-05 packet files, excluding checkpoint
  subdirectories;
- simulator v5 source.

Each verbatim seat output ends with its own `FILES ACTUALLY OPENED` section.
After all reviews completed, each same seat received one persistence-only
instruction: write its already-completed final response verbatim to its
assigned file, read nothing else, and report only the resulting SHA-256.

## Verbatim Seat Outputs

No exact duplicate finding was removed.

<!-- BEGIN VERBATIM SEAT 1 -->
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
<!-- END VERBATIM SEAT 1 -->

<!-- BEGIN VERBATIM SEAT 2 -->
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
<!-- END VERBATIM SEAT 2 -->

<!-- BEGIN VERBATIM SEAT 3 -->
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
<!-- END VERBATIM SEAT 3 -->

## Join Conclusion

FT2-20 does not pass. Owner approval is not available.

Primary route:

```text
data_tensor_label_defect -> FT2-08-DATA-TENSOR-LABEL-CONTRACT
```

Retained downstream routes:

```text
entry_science_defect -> FT2-10-ENTRY-SCIENCE-CONTRACT
evidence_statistics_defect -> FT2-11-EVIDENCE-STATISTICS-CONTRACT
```

No repair, FT2-21 work, training, or new session analysis occurred in this
node.

> Three isolated design reviews are complete and joined; the graph routes to named contract repair.
