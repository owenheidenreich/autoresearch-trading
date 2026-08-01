# Path-D Composite Calibration Terminal Rule — Read-Only Evidence Analysis

Date: 2026-08-01

Status: analysis for owner/Claude decision; no rule adopted.

Highest allowed claim: **read-only evidence analysis to inform an owner decision; no rule adopted, nothing sealed or fit.**

## Scope and authorities

This analysis is intentionally outside the frozen foundation. It performs no model fit, corpus decode, fold-evidence open, machinery seal, foundation-stability seal, protected-holdout access, live operation, or broker operation. It does not modify the corrected preregistration or session assignments.

Required holdout caveat: “The final 30-session firewall is protected from model/economic inspection but is not pristine from the already published full-corpus aggregate label statistics.”

Primary authorities:

| Authority | Identity |
|---|---|
| Claude findings | v4/docs/protocol101/training/execution/PROTOCOL101_PATHD_COMPOSITE_CALIBRATION_TERMINAL_RULE_FINDINGS_2026_08_01.md; file SHA-256 35b58995a1627adf93cfb031c201111b78c374fbd3e3537dc4b0e7dbd5ac510f |
| Corrected preregistration | v4/audit/autoresearch/protocol101_pathd_entry_exit_model_research_corrected_2026_08_01/preregistration.json; file SHA-256 4a01f145abfb2a5da664af4f3de1c122069678dc7c3aa1b953745672f2822ec8 |
| Corrected session assignments | v4/audit/autoresearch/protocol101_pathd_entry_exit_model_research_corrected_2026_08_01/session_assignments.json; file SHA-256 431cd14879ad6a14b278cb2683e4f3b860eef960e6b74a4f0edb3e620cf82475; prereg semantic /sessions_hash 597afa94b1b3983163bd6280b2c1138b151ccb9ace6990b69596a0f930dd8627 |
| Published source-quality report | v4/audit/autoresearch/protocol101_pathd_data_acquisition/data_quality_report.json; file SHA-256 b029e1a55e63014d6ed870707453f847c458cbf3f340cf38bc0da3dcc81a223f |
| Published full-corpus exit-smoke label summary | v4/audit/autoresearch/protocol101_pathd_data_acquisition/pathd_exit_label_validation_12m.json; file SHA-256 8971cbecef2a7ed5e691d697be02845adf66762e390fe2f0fb9af33c01d936d2 |

JSON citations below use zero-based array indices because they are exact JSON Pointers.

## Executive findings

1. The apparent 54/75/97/119/142 versus “all <60” discrepancy is resolved. The first vector is the outer **entry** model-fit history. The exit denominator is the aggregate of calibration-valid nested-entry-OOF **validation sessions assigned to exit weights**, and is exactly 0/0/19/48/56. All five learned-exit outer folds therefore must abstain before exit weights. Some nested entry blocks are valid; no outer-fold exit fit is sufficiently powered.
2. The literal frozen calibration graph contains 40 entry marginal calibration outputs, two entry action-composite corrections, one exit mean correction, and 15 exit quantile outputs. The entry action-composite statuses are implemented; the marginal-head and exit status-to-terminal routing is not.
3. Fold 1 has 14 calibration dates and a ten-session minimum, so a marginal head needs at least five whole sessions with no valid row for that same head to fall below ten. All 14 dates have published session-level required-file presence, including nonempty CBBO-1s files, and none is OPRA-degraded or shortened; within-session exact-contract and target coverage remain unknown. The actual composite-failure probability is **UNKNOWN**. Per-head label-validity counts require the prohibited corpus decode/dataset build, while ENTER/WAIT incidence and selected-composite coverage additionally require model predictions/fit.
4. Claude’s “four days of slack” is arithmetically correct but does not establish a high empirical failure probability. Source-loss risk looks low/theoretical; action-incidence and target-coverage risk is genuinely possible but not probabilistically identified.
5. Recommended terminal policy: **B-R, a status-preserving whole-run hard stop.** Every required outer/final calibrator must be valid in all five folds before the existing pooled and at-least-four-of-five economic gates are evaluated. Lawful frozen nested skip blocks remain skips rather than whole-run terminals. An evidence-count miss terminates the run as insufficient_evidence; B-R proposes that invalid target coverage terminate outside the feasibility verdict as invalid_result. Required targets are never dropped. Predeclared diagnostic-only families may be reported diagnostic_incomplete_only.
6. A separate frozen contradiction must be resolved before B-R or any other terminal rule is adopted: A_ref_q50/q90 are declared isolated diagnostics, but joint monotone calibration needs q50 and the binding EXIT action gate consumes calibrated A_ref_q90. A terminal rule cannot determine which clause wins.

## Findings versus Claude

| Claude finding | Independent result |
|---|---|
| Finding 1: exit abstains this run | **Agree.** Every outer-fold exit-weight count is below 60. Clarification: the denominator is not an outer model-fit count or a nested model-fit count; it is the aggregate of valid nested-entry-OOF validation sessions assigned to exit weights. |
| Finding 2: trade/trajectory power is distinct | **Agree.** The composite-calibration decision must not lower or substitute the frozen trade, trajectory, session, or action-power thresholds. |
| Finding 3: composite spans mean composites and ENTER/WAIT | **Agree but incomplete.** The exact graph is enumerated below. It also exposes an A_ref q50/q90 role conflict not identified in Claude’s note. |
| Finding 4: fold 1 makes the issue live | **Partly agree.** Four marginal-session losses of slack is exact. The inference that label completeness makes failure likely is unsupported: all 14 source sessions are present and no new-head validity or action-incidence counts have been published. |
| Finding 5: “<60” is probably nested | **Agree in direction, refine the noun.** It is the valid nested-entry-OOF validation-session aggregate used for exit weights, not the nested inner model-fit arrays themselves. |
| Candidate A is most aligned with abstention-first | **Disagree for this frozen run.** A would change all-five per-fold power, exact five-row reconstruction, and pooled-population contracts and would introduce survivor-selection risk. |
| Candidate C may continue after target removal | **Reject.** Dynamic target deletion changes the estimand and composite weights and is reward-hacking-adjacent. |

## 1. Reconciliation of the exit “all <60” denominator

### 1.1 Exact preregistration definition

The following paths are dispositive:

- /combined_evaluation/known_run_consequence states that all frozen exit model-fit session counts are below 60.
- /exit/power_count_semantics/model_fit_sessions defines those sessions as distinct nested-entry-OOF trajectory sessions assigned to exit weights only; exit-calibration sessions and raw entry weight-fit history do not count.
- /exit/power_count_semantics/known_exit_weight_session_counts_by_fold is [0, 0, 19, 48, 56].
- /exit/power_count_semantics/known_exit_calibration_session_counts_by_fold is [0, 15, 19, 23, 28].
- /exit/power_count_semantics/minimum_model_fit_sessions is 60.
- /exit/power_count_semantics/consequence requires insufficient_evidence before exit weights because every count is below 60.
- /metrics_and_gates/minimum_power/model_fit_sessions independently freezes 60.
- /exit/power_count_semantics/minimum_eligible_trajectories_per_fit_fold freezes a separate 500-trajectory conjunct; it cannot rescue the session-count failure.

The apparently contradictory 54/75/97/119/142 values are the lengths of session_assignments /folds/0/model_fit through /folds/4/model_fit. They are outer entry-weight histories and are expressly excluded from the exit denominator.

### 1.2 Exact nested mapping

For each outer fold, session_assignments /folds/i/exit_training_oof_mapping/rule says that scored inner validation blocks 2–4 supply exit weights and block 5 supplies exit calibration. A validation block contributes only when its nested entry calibration has at least ten sessions and /calibration_valid is true.

| Outer fold | Outer entry model_fit | Inner 1: fit/cal/valid/validation | Inner 2 | Inner 3 | Inner 4 | Exit-weight aggregate | Exit calibration | Learned-exit disposition |
|---:|---:|---|---|---|---|---:|---:|---|
| 1 | 54 | 7/2/false/11 | 15/5/false/11 | 24/7/false/11 | 33/9/false/10 | 0 | 0 | Abstain |
| 2 | 75 | 10/3/false/15 | 22/6/false/15 | 34/9/false/15 | 46/12/true/15 | 0 | 15 | Abstain |
| 3 | 97 | 14/4/false/20 | 30/8/false/19 | 45/12/true/19 | 60/16/true/19 | 19 | 19 | Abstain |
| 4 | 119 | 17/5/false/24 | 36/10/true/24 | 55/15/true/24 | 75/19/true/23 | 48 | 23 | Abstain |
| 5 | 142 | 21/6/false/29 | 44/12/true/28 | 67/17/true/28 | 89/23/true/28 | 56 | 28 | Abstain |

Exact evidence paths for every table cell are:

- /folds/i/inner_forward_folds/scored_forward_folds/j/model_fit
- /folds/i/inner_forward_folds/scored_forward_folds/j/calibration
- /folds/i/inner_forward_folds/scored_forward_folds/j/calibration_minimum_sessions
- /folds/i/inner_forward_folds/scored_forward_folds/j/calibration_valid
- /folds/i/inner_forward_folds/scored_forward_folds/j/validation
- /folds/i/exit_training_oof_mapping/exit_model_weight_sessions
- /folds/i/exit_training_oof_mapping/exit_calibration_sessions
- /folds/i/exit_training_oof_mapping/weight_session_count
- /folds/i/exit_training_oof_mapping/calibration_session_count
- /folds/i/exit_training_oof_mapping/invalid_inner_blocks

Conclusion: folds 3–5 have some lawful contributing blocks, but their final aggregates remain below 60. The learned exit must abstain in **every** outer fold, not only fold 1. Therefore this run cannot instantiate learned-exit Boxes C/D or make a combined entry+exit claim.

## 2. Complete calibration dependency graph

### 2.1 Entry marginal heads

Preregistration /entry/head_targets, /entry/mean_heads, and /entry/q10_heads contain the same exact 20 target names:

1. h10_mfe_dollars
2. h10_mfe_return
3. h10_profit_area_dollars
4. h10_profit_area_return
5. h20_mfe_dollars
6. h20_mfe_return
7. h20_profit_area_dollars
8. h20_profit_area_return
9. h45_mfe_dollars
10. h45_mfe_return
11. h45_profit_area_dollars
12. h45_profit_area_return
13. h90_mfe_dollars
14. h90_mfe_return
15. h90_profit_area_dollars
16. h90_profit_area_return
17. session_mfe_dollars
18. session_mfe_return
19. session_profit_area_dollars
20. session_profit_area_return

Each target has:

- one conditional-mean correction under /calibration_and_statistics/mean_lcb; and
- one marginal q10 correction under /calibration_and_statistics/q10_conformal.

Thus the entry marginal layer has 40 calibration outputs. “session” is the registered target prefix for the composer’s remaining_session horizon.

### 2.2 Mean composites

Under /entry/composer/mean_composites:

- LCB_mean_upside_$ consumes the MFE-dollar and profit-area-dollar mean LCBs for each causally available member of h10, h20, h45, h90, and remaining_session.
- LCB_mean_upside_return consumes the corresponding ten return mean LCBs.

/entry/composer/causal_horizon_availability fixes the decision-clock-only availability set. /entry/composer/gate requires both composites finite and strictly positive; a component required by that row’s available set fails the contract closed if missing or nonfinite. A causally unavailable finite horizon is ignored by design and is not post-hoc target dropping.

### 2.3 Marginal q10 ranking and second-stage ENTER/WAIT corrections

/entry/q10_heads registers 20 calibrated q10 outputs. At each decision, /entry/composer/q10_ranking consumes only the dollar and return components in that row’s causal available-horizon subset: dollars rank first and returns break the q10 composite tie after both mean gates pass.

ENTER then adds a second-stage composite correction:

- /metrics_and_gates/action_calibration/actions/ENTER/raw_composed_lower uses the available ten dollar q10 components for the selected contract.
- /metrics_and_gates/action_calibration/actions/ENTER/realized uses the matching realized dollar composite.
- /metrics_and_gates/action_calibration/actions/ENTER/composite_conformal requires at least ten distinct sessions.

The ENTER population indirectly depends on all 40 marginal entry outputs because mean dollars/returns gate contracts and q10 dollars/returns select the contract. Its conformal residual directly uses the ten available dollar q10 components.

WAIT is a second-stage episode correction:

- /entry/wait_action_calibration_audit/raw_lower_statistic uses the maximum available dollar q10 composite over legal actions.
- /entry/wait_action_calibration_audit/realized_target uses the matching maximum missed-upside composite.
- /entry/wait_action_calibration_audit/calibration_power requires at least ten distinct sessions and 30 retained WAIT episodes.

The exact action status implementation is in v4/research/pathd_entry_models.py, approximately lines 3172–3419. The current runner rejects either non-VALID status before composer creation in v4/scripts/run_pathd_entry_exit_research.py, approximately lines 344–380.

The head names repeat across HGB and neural and across exact nested, outer, and full-fit scopes under /calibration_and_statistics/seed_derivation/binding_key_rules. Those are separate calibration instances, not additional semantic target names. The five outer calibration partitions at session_assignments /folds/0..4/calibration_last_20_percent contain 14, 20, 25, 31, and 36 sessions. Nested blocks have their own frozen calibration-validity/skip topology and must not be silently treated as outer/final terminal failures.

### 2.4 Exit mean and quantile heads

The exit action mean is:

- A_ref_mean, called conditional_mean_A_ref at /exit/composer/mean_bound/target.

The literal /calibration_and_statistics/monotone_q10_q50_q90/scope covers every exit target family’s q10/q50/q90 outputs. Reconstructing /exit/distributional_targets gives 15 quantile outputs:

1. A_ref_q10
2. A_ref_q50
3. A_ref_q90
4. downside_300_q10
5. downside_300_q50
6. downside_300_q90
7. recovery_300_q10
8. recovery_300_q50
9. recovery_300_q90
10. giveback_300_q10
11. giveback_300_q50
12. giveback_300_q90
13. remaining_tail_300_q10
14. remaining_tail_300_q50
15. remaining_tail_300_q90

A_ref_mean and A_ref_q10 drive /exit/composer/utility_formula. /metrics_and_gates/action_calibration/actions/HOLD uses calibrated A_ref_q10 as predicted lower. /metrics_and_gates/action_calibration/actions/EXIT uses negative calibrated A_ref_q90 as predicted lower. The four local-path triplets are explicitly diagnostic-only under /exit/diagnostic_target_isolation.

### 2.5 Exact failure taxonomy

| Calibration node | INVALID_TARGET_COVERAGE | INSUFFICIENT_EVIDENCE |
|---|---|---|
| Each of 20 entry mean_lcb heads | Any nonfinite prediction or residual on a preregistered target-valid row. The row may not be dropped after prediction. | Fewer than ten distinct sessions retain at least one valid row for that exact head. |
| Each of 20 entry q10_conformal heads | Any nonfinite prediction or residual on a target-valid row. | Fewer than ten distinct retained sessions. With finite residuals and adequate sessions, the weighted empirical q10 exists. |
| ENTER composite correction | Any selected ENTER intent lacks any dollar target required by its causal available-horizon set. One such retained missing outcome is enough. | Fewer than ten distinct retained ENTER-intent sessions; the implementation also checks at least one row. |
| WAIT composite correction | Any legal action at a retained WAIT anchor lacks any required dollar target in its causal available-horizon set. | Fewer than ten distinct WAIT sessions or fewer than 30 retained WAIT episodes. |
| Exit A_ref_mean mean_lcb | Any nonfinite prediction/residual on a target-valid row. | Fewer than ten distinct retained sessions. |
| Each literal exit monotone triplet | Any nonfinite prediction or width on a target-valid row; or lower/upper 0.90 coverage is unattainable because of permanently uncovered zero-width rows or no finite candidate scale. | Fewer than ten distinct retained sessions for that target family. |
| Four local-path diagnostic triplets | The same head-local conditions. | The same head-local session condition, but the frozen run-level consequence is diagnostic_incomplete_only. |

Common authorities are /calibration_and_statistics/input_validity, /calibration_and_statistics/mean_lcb/minimum_sessions, /calibration_and_statistics/q10_conformal/minimum_sessions, and /calibration_and_statistics/monotone_q10_q50_q90/{failure,minimum_sessions,lower_width,upper_width}.

A preregistered label-invalid row is legally masked. Its existence is not INVALID_TARGET_COVERAGE. If masking leaves fewer than ten distinct sessions, the result is INSUFFICIENT_EVIDENCE. Conversely, a nonfinite prediction, residual, or width on a row frozen target-valid is invalid and cannot be relabeled as a power miss or rescued by deletion.

Downstream action-gate power is distinct from formation of the composite correction. /metrics_and_gates/action_calibration also requires 30 distinct trajectories per action, at least three trajectories and three distinct sessions per decile, and 5,000 valid adjacent-decile bootstrap replicates before the 50,000-draw cap. Those misses are insufficient_evidence; they must not be confused with marginal-head coverage failure.

### 2.6 Frozen A_ref topology contradiction

The exit graph has two mutually incompatible literal readings:

1. /calibration_and_statistics/monotone_q10_q50_q90/scope requires a joint A_ref q10/q50/q90 calibration. Its formula uses q50 as the calibrated location for q10 and q90.
2. /exit/diagnostic_target_isolation places A_ref_q50 and A_ref_q90 in separate diagnostic models/calibration bundles, forbids diagnostics from changing A_ref calibration, and says diagnostic failure never changes the verdict.
3. /metrics_and_gates/action_calibration/actions/EXIT/predicted_lower nevertheless requires calibrated A_ref_q90, while /combined_evaluation/acceptance/action_gate requires EXIT to pass.

Therefore A_ref_q90 is simultaneously verdict-required and verdict-isolated, and q50 is simultaneously needed by the literal joint calibrator and forbidden from influencing A_ref calibration. No frozen clause establishes precedence. The current source contains no exit monotone calibrator that resolves it.

This is not merely a terminal-policy choice. Before any owner adoption, a superseding target-role clarification must either:

- make the A_ref q10/q50/q90 calibration triplet decision-critical calibration support, while keeping only the four local-path triplets diagnostic; or
- specify a standalone A_ref q10 calibration and remove A_ref_q90 from the binding EXIT action gate.

The first option preserves more of the current action-calibration contract and is the smaller scientific change, but it still must explicitly supersede the conflicting diagnostic-isolation clauses. This analysis adopts neither option.

## 3. Quantification of fold-1 risk from published completeness only

### 3.1 Deterministic frozen evidence

session_assignments /folds/0/calibration_last_20_percent fixes these 14 dates:

2025-10-20, 2025-10-21, 2025-10-23, 2025-10-24, 2025-10-27, 2025-10-28, 2025-10-29, 2025-10-30, 2025-10-31, 2025-11-03, 2025-11-04, 2025-11-05, 2025-11-06, and 2025-11-07.

All 14:

- are members of /folds/0/train_primary_excluding_prior_embargo_and_opra_degraded;
- have zero overlap with session_assignments /opra_degraded_diagnostic_only;
- have zero overlap with /scheduled_short_sessions_diagnostic_only; and
- occur after /folds/0/calibration_embargo and before the outer embargo/test.

Published source evidence:

- preregistration /corpus/integrity_contract/partitions/CORE_AUTHORITATIVE/path_grammars records 251 files for minute-entry, CBBO-1s, and official-SPX substrates;
- data_quality_report /pairing records 251/251 minute-entry, SPX, and option-session pairing;
- data_quality_report /option_schemas/cbbo-1s records 251 nonempty sessions;
- all source-quality checks passed;
- pathd_exit_label_validation_12m /per_day_rows records exactly 147,600 legacy exit-smoke rows (a_hold and oracle_adv) on each of the 14 dates, or 2,066,400 rows total.

This proves complete session-level substrate presence and old exit-smoke label materialization. It does **not** prove per-session validity for any of the 20 new entry targets or action-selected composite outcomes. Preregistration /entry/label_builder/artifact_status_at_preregistration is NOT_BUILT.

### 3.2 What can be determined exactly

For a marginal head with a ten-session minimum and 14 available sessions:

- zero through four entirely unusable sessions leaves at least ten sessions;
- at least five sessions with no valid row for the same head produces fewer than ten and therefore insufficient evidence.

For second-stage composites:

- ENTER must occur with a retained valid composite in at least 10 of 14 sessions, i.e. at least 71.4% of fold-1 calibration days;
- WAIT must have retained episodes in at least 10 sessions and at least 30 episodes total, i.e. 2.143 episodes per day averaged over all 14 days.

Neither action count is published. Both depend on predictions, gating, ranking, fills, and causal account state and cannot be inferred from source files.

### 3.3 Probability is not identified

The frozen data and any future fixed-seed fit are deterministic. The preregistration defines no probability measure over hypothetical missing sessions, future model actions, or calibration geometry. Published artifacts provide no per-head valid-session counts and no ENTER/WAIT calibration population.

Accordingly:

- known whole-session source-loss count in the fixed fold-1 partition: **0**;
- actual probability of marginal-head insufficiency from published evidence: **UNKNOWN**;
- actual probability of ENTER/WAIT composite insufficiency: **UNKNOWN**;
- actual probability of INVALID_TARGET_COVERAGE: **UNKNOWN**;
- nonparametric identification interval under probability models consistent with the published summaries: **0%–100%**.

Reporting a single “actual probability” would manufacture information that is not present.

### 3.4 Explicitly nonbinding sensitivity proxy

For intuition only, suppose independent exchangeable session failures with probability p. Marginal insufficiency from five or more failed days has probability:

P(X >= 5) = sum from k=5 to 14 of C(14,k) p^k (1-p)^(14-k).

Using the published OPRA-degradation rate 2/251 as a crude vendor-notice proxy:

| Proxy | p | P(at least 5 of 14) |
|---|---:|---:|
| Plug-in 2/251 | 0.7968% | 6.0565e-8, or 0.00000606% |
| One-sided 95% exact upper p after 2/251 | 2.4869% | 1.5780e-5, or 0.001578% |

These are **not composite-failure probabilities**. The binomial calculation models one hypothetical head or one shared whole-session failure mechanism, not the union across 20 heads, ENTER, WAIT, and INVALID_TARGET_COVERAGE. A degradation notice is not a whole target-session loss, sessions need not be independent, and the new target-validity/action population is different. Treating the two notices as losses is conservative for those two dates but can be anti-conservative for unnotified within-session exact-contract gaps, correlated failures, and the multi-head union. The exact fold-1 dates contain no published OPRA degradation.

For a separate future-replication sensitivity, zero observed failures in 14 gives a one-sided 95% Clopper–Pearson upper per-session p of 19.2636%; applying that p to a **new independent** 14-session binomial gives P(at least 5) = 11.4525%. This is not an upper probability for the already-fixed fold. It illustrates that the result is assumption-driven.

### 3.5 Risk characterization

- **Whole-session source absence:** not observed; low/theoretical for this fixed fold.
- **New marginal target-validity loss:** structurally possible but unmeasured.
- **ENTER/WAIT action-incidence power:** genuinely live because action counts are model-dependent and no counts exist pre-fit.
- **Missing selected/action composite outcomes:** genuinely live and can invalidate coverage even when all 14 session-level source files are present.
- **Nonfinite predictions or unattainable exit quantile coverage:** model-dependent and unmeasured.

Thus Claude is right that a terminal rule is needed before fitting, but the evidence does not support calling fold-1 failure probable.

## 4. Candidate evaluation

| Candidate | Frozen-contract compatibility | Abstention quality | Reward-hacking/survivor risk | Decision |
|---|---|---|---|---|
| A — fold-scoped abstain, continue with at least four valid folds | Conflicts with the current all-five power formulas, exact five non-null fold rows, and pooled sum over five rows. It would require a superseding scientific-topology amendment. | Good locally; preserves a clear abstention record. | Material: calibration failure may correlate with a hard regime, so deleting that fold can improve pooled economics. Exactly four valid folds would need all four positive but still omit the hard era. | Do not adopt for this frozen run. Consider only in a new data-era preregistration with an explicit missing-fold estimand. |
| B — whole-run hard stop | Matches all-five per-fold power, exact five-row reconstruction, and the separation between power and signal gates. | Strongest: it refuses a feasibility claim when any required calibration is unavailable. | None from fold/target deletion. | Preferred base policy. |
| C — drop target and continue | Contradicts /calibration_and_statistics/input_validity, /entry/composer/gate, fixed equal-weight composites, and no-post-hoc-rescue language. | Poor: hides the reason for abstention. | High: changes the objective toward easier-to-calibrate targets and makes folds incomparable. | Reject. |
| B-R — status-preserving whole-run hard stop | Same compatibility as B while preserving invalid-versus-insufficient semantics. | Strong; emits a precise terminal cause rather than laundering every failure into one label. | None from dynamic deletion. | **Single recommendation.** |

## 5. Recommended rule: B-R

Recommended owner-decision text:

> For each outer/final model bundle lawfully eligible under the frozen nested skip topology, every required marginal head and ENTER/WAIT disjoint-calibration correction must be VALID before that fold’s outer economic evidence is opened. A lawful nested block whose frozen session-count calibration_valid is false remains a durable skip and contributes only through the existing mapped exit-power counts; it does not itself trigger this whole-run rule. A required outer/final node with fewer than its frozen minimum distinct sessions or fewer than its frozen calibrator-specific observations is INSUFFICIENT_EVIDENCE and terminates the complete run with feasibility verdict insufficient_evidence. B-R proposes that a required node with a nonfinite prediction/residual/width on a target-valid row, a retained composite outcome missing, or unattainable nominal monotone coverage be classified INVALID_TARGET_COVERAGE and terminate the complete run as invalid_result outside the feasibility-verdict vocabulary. No failed fold, row, horizon, action, or required target may be deleted, substituted, zero-imputed, reweighted, or used to change the composite. Only families preregistered diagnostic-only before results may end diagnostic_incomplete_only without changing the primary verdict. Later pooled action-trajectory, decile, and bootstrap power remains at its existing post-evidence verdict stage; those are not pre-open calibrator checks. The existing pooled and at-least-four-of-five economic gates are evaluated only after all five required outer/final calibrators are valid and every frozen minimum-power criterion is met.

### Why status preservation matters

Mapping INVALID_TARGET_COVERAGE to insufficient_evidence would conflate a malformed/unattainable model output with a lack of sample size. The corrected foundation separates invalidity from power, but it does **not** freeze INVALID_TARGET_COVERAGE-to-invalid_result as the durable mapping:

- /calibration_and_statistics/input_validity: invalid target-valid predictions invalidate the head/bundle;
- /entry/pooled_acceptance/reconstruction/invalid_structure_or_reconstruction: invalid_result is already out of band for structural/reconstruction/hash invalidity and writes no scientific result or terminal success receipt;
- /verdict_vocabulary/entry_failure_mapping/minimum_power_misses: genuine power misses map to insufficient_evidence; and
- /verdict_vocabulary/entry_failure_mapping/adequate_power_but_signal_or_action_gate_fails: an adequately powered scientific failure maps to no_genuine_signal.

B-R proposes extending that out-of-band treatment to INVALID_TARGET_COVERAGE while preserving all three concepts. That extension is an owner decision, not an inference from the current freeze. Its adoption contract must define a failure-only audit receipt without writing a scientific result or terminal success receipt.

### Required contract bindings

| Frozen contract | B-R binding |
|---|---|
| /metrics_and_gates/minimum_power | Thresholds remain unchanged and conjunctive. Calibration validity is a prerequisite, never a substitute for 40 trades/fold, 200 pooled, 50 OOF trajectories/fold, 300 pooled, 60 exit-weight sessions, 500 exit trajectories/fold, or the action minima. |
| /entry/pooled_acceptance/reconstruction/gate_inputs/pass_criteria_exact_keys_and_formulas | “all five” power formulas remain literal. No invalid fold receives a fabricated zero or null economic cell. |
| /entry/pooled_acceptance/reconstruction/gate_inputs/closed_shapes/per_fold and /pooled | Exact five-row structure and pooled sum over five rows remain unchanged because economics are never constructed unless all five are valid. |
| /metrics_and_gates/fold_consistency and /combined_evaluation/acceptance/folds | Positive economic delta in at least four of five remains a signal-consistency gate after validity/power. It is not reinterpreted as four survivors out of an incomplete set. |
| /combined_evaluation/acceptance/primary and /metrics_and_gates/pooled | Pooled positivity/bootstrap requirements remain unchanged and use the complete lawful five-fold evidence population. |
| /verdict_vocabulary/entry_failure_mapping/minimum_power_misses | INSUFFICIENT_EVIDENCE routes to insufficient_entry_evidence for entry. A corresponding explicit exit stop reason must be added by owner decision rather than borrowed silently. |
| /verdict_vocabulary/entry_failure_mapping/adequate_power_but_signal_or_action_gate_fails | Reserved for no_genuine_signal after calibration and power are adequate; calibration failure cannot be called no signal. |
| /calibration_and_statistics/input_validity | INVALID_TARGET_COVERAGE remains invalid; no row/target deletion or post-hoc rescue. |
| /exit/diagnostic_target_isolation/diagnostic_failure_effect | The four frozen local-path diagnostic families may be incomplete without changing the primary verdict. This exception is role-based and preregistered, not Candidate C. |

The B-R terminal policy itself does not contradict these clauses. Candidate A would. However, no terminal policy can resolve the pre-existing A_ref_q50/q90 contradiction described in section 2.6. That target-role conflict requires an explicit superseding owner clarification before the phrase “decision-critical calibration node” has one unambiguous exit definition.

## 6. Required owner/implementation follow-up before any seal or fit

This analysis recommends but does not adopt B-R. A lawful adoption packet should, before machinery or stability sealing:

1. Resolve the A_ref q10/q50/q90 action-versus-diagnostic conflict explicitly.
2. Enumerate the exact decision-critical calibration-node IDs and exact diagnostic-only node IDs.
3. Bind node-level failure-only audit receipts and one failure terminal receipt, while forbidding a scientific result or terminal success receipt, with precedence:
   invalid_result, then insufficient_evidence, then owner_decision_required, then no_genuine_signal, then PASS.
4. Bind an explicit exit minimum-power stop reason rather than overloading the entry-only name.
5. Require all five calibration-valid folds before the existing pooled and at-least-four-of-five gates.
6. Add negative tests proving that target deletion, fold deletion, zero imputation, status relabeling, and survivor-only pooling fail closed.
7. Re-freeze only through an owner-approved superseding correction; do not mutate the current corrected preregistration.

## Final conclusion

The “all <60” statement is correct for the exact exit-weight denominator, and every learned-exit fold abstains. All fold-1 required source files are present/nonempty at session level, while within-session target coverage and the true new-head/action-composite failure probability are not identified. The durable rule most compatible with the frozen scientific topology and the owner’s abstention-first, no-reward-hacking stance is B-R: preserve the failure taxonomy and stop the whole run at the earliest lawful stage when any required outer/final calibration node is not valid. Dynamic target or fold deletion is not honest evidence.

No rule is adopted by this document.

STOP_FOR_CLAUDE_VERIFICATION
