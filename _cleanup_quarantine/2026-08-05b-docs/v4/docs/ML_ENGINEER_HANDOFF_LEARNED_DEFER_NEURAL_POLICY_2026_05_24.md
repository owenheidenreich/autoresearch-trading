# ML Engineer Handoff: Learned-Defer Neural Policy Foundation

Date: 2026-05-24

## Executive Summary

The project goal is to build a model that takes better trades than `PAPER_DEFAULT_PROTOCOL101`: higher expected PnL, lower downside, cleaner execution assumptions, and promotion-grade evidence.

In this session, we completed the first foundation-hardening loop after Protocol276 failed. We did **not** change the paper-trading default. Protocol101 remains the default. We built a causal slot-opportunity-cost label set, trained a small learned estimator for the cost of blocking Protocol101 trades, froze a learned defer overlay, and ran exactly one preregistered neural policy experiment through that overlay.

The preregistered learned-defer policy passed diagnostic strict replay against same-scope Protocol101 on the current repeated diagnostic blocks:

| Slippage | Challenger Entries | Total Delta vs Same-Scope Protocol101 | Q1 2026 Delta | Q3 2025 Delta | Q4 2025 Delta | Recent 2026 Delta |
|---:|---:|---:|---:|---:|---:|---:|
| `0.00` | 60 | `$66,780` | `$2,540` | `$3,475` | `$52,600` | `$8,165` |
| `0.10` | 60 | `$67,580` | `$3,080` | `$3,535` | `$52,640` | `$8,325` |
| `0.25` | 60 | `$68,780` | `$3,890` | `$3,625` | `$52,700` | `$8,565` |

This is promising, but it is **not** a promotion claim. These splits are heavily research-exposed diagnostics, not clean final holdouts. The remaining Protocol101 challenge blockers are:

1. No calibrated stochastic fill model; current fill observations are `0`.
2. Untouched future holdout data is reserved but not collected/frozen/scored.
3. Live no-order full-action parity is still incomplete; current parity is historical/proxy only.
4. Formal validation controls, such as PBO/CSCV or an equivalent strategy-matrix false-discovery audit, are not complete.

The current recommended posture is: **pause further neural experiments and ask an ML specialist to review the formulation, labels, estimator, replay, and validation plan before any model-search resumes.**

## Why This Work Was Needed

Protocol276 integrated:

- Protocol271 full-action entry policy
- Protocol275 learned hold/exit lifecycle policy
- strict one-account serial replay
- `$10,000` starting cash
- one contract
- one open position max
- ask-entry
- bid-exit
- affordability check
- flat by close

It failed Protocol101 on the protected older diagnostic blocks and only won on recent 2026. The strongest diagnosis was not merely “bad model,” but **single-slot opportunity cost**: challenger entries consumed the only open-position slot and blocked stronger Protocol101 trades. A later attribution showed:

- Q1 2026: challenger PnL `$35,590` minus missed Protocol101 PnL `$46,360` explained `-$10,770`.
- Q3 2025: challenger PnL `-$80` minus missed Protocol101 PnL `$660` explained `-$740`.
- Q4 2025 and recent 2026 were positive, so the failure was split-unstable, not universally bad.

The foundation plan therefore became: make the model pay a causal estimate of the Protocol101 slot cost before it is allowed to override/defer away from Protocol101.

## What Was Implemented In This Session

### 1. Candidate-Level Blocked Protocol101 Cost Labels

Artifacts:

- Script: `v4/scripts/run_unified_slot_opportunity_cost_label_dataset.py`
- Helper: `v4/model/unified_slot_opportunity_defer.py`
- Test: `v4/tests/test_unified_slot_opportunity_cost_label_dataset.py`
- Test: `v4/tests/test_unified_slot_opportunity_defer.py`
- Report: `v4/audit/autoresearch/unified_slot_opportunity_cost_label_dataset/report.md`
- Labels: `v4/audit/autoresearch/unified_slot_opportunity_cost_label_dataset/slot_opportunity_cost_labels.parquet`
- Docs copy: `v4/docs/UNIFIED_SLOT_OPPORTUNITY_COST_LABEL_DATASET.md`

The label dataset computes, for each flat candidate row, the Protocol101 entries in the same session that would be blocked by entering that candidate:

`current_decision_dt <= baseline_entry_dt < candidate_exit_dt`

Emitted label columns:

- `blocked_protocol101_entries`
- `blocked_protocol101_pnl_0_00`
- `blocked_protocol101_pnl_0_10`
- `blocked_protocol101_pnl_0_25`
- `has_blocked_protocol101_entry`

These are **training labels only**, not runtime features.

Forbidden as model inputs:

- blocked Protocol101 label columns
- `candidate_exit_dt`
- future path PnL
- oracle action/value columns

Label dataset summary:

| Split | Rows | Sessions | Blocked Entry Rows | Blocked Row Fraction | Mean Blocked PnL `0.00` | P95 Blocked PnL `0.00` |
|---|---:|---:|---:|---:|---:|---:|
| q1_2026 | 551,413 | 53 | 128,190 | 0.2325 | 96.36 | 870 |
| q3_2025 | 529,645 | 55 | 122,343 | 0.2310 | 72.32 | 560 |
| q4_2025 | 534,830 | 54 | 130,530 | 0.2441 | 117.48 | 870 |
| recent_2026 | 303,630 | 29 | 48,656 | 0.1602 | 11.09 | 320 |

Total rows: `1,919,518`.

Decision: `slot_opportunity_cost_labels_ready_for_causal_estimator`.

### 2. Causal Slot Opportunity-Cost Estimator

Artifacts:

- Model primitives: `v4/model/unified_slot_opportunity_cost_estimator.py`
- Training script: `v4/scripts/run_unified_slot_opportunity_cost_estimator.py`
- Tests: `v4/tests/test_unified_slot_opportunity_cost_estimator.py`
- Report: `v4/audit/autoresearch/unified_slot_opportunity_cost_estimator/report.md`
- Model: `v4/audit/autoresearch/unified_slot_opportunity_cost_estimator/model_artifacts/slot_opportunity_cost_estimator.joblib`
- Docs copy: `v4/docs/UNIFIED_SLOT_OPPORTUNITY_COST_ESTIMATOR.md`

Estimator purpose:

- Predict nonnegative Protocol101 slot opportunity cost.
- Predict blocked-entry count.
- Predict probability that blocked Protocol101 cost is positive.
- Provide a conservative uncertainty charge.

This estimator is not a trading policy. It is a guardrail used by the defer overlay.

Training setup:

- Train splits: `q3_2025`, `q4_2025`
- Validation split: `q1_2026`
- Diagnostic split: `recent_2026`
- Train rows: `350,000`
- Feature count: `78`
- Model family: small scikit-learn histogram gradient boosting regressors/classifier
- Inputs: current-state flat candidate features only
- Explicitly forbidden: labels, future exits, oracle values

Key metrics:

| Split | Rows | Positive Cost Rate | Cost MAE | P90 Abs Error | Positive AUC | Brier | Count MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| train_sample | 350,000 | 0.4000 | 115.35 | 350.46 | 0.9911 | 0.0477 | 0.2421 |
| q1_2026 | 120,000 | 0.1631 | 148.91 | 443.34 | 0.8107 | 0.1645 | 0.4383 |
| q3_2025 | 120,000 | 0.1880 | 53.39 | 143.47 | 0.9908 | 0.0640 | 0.2286 |
| q4_2025 | 120,000 | 0.1981 | 71.52 | 217.10 | 0.9904 | 0.0603 | 0.2355 |
| recent_2026 | 120,000 | 0.0813 | 80.85 | 161.90 | 0.7418 | 0.1246 | 0.4620 |

Decision: `slot_opportunity_cost_estimator_ready_for_defer_overlay_replay`.

Specialist note: Q1 and recent AUC/calibration are weaker than Q3/Q4. The estimator is useful enough for a guardrail, but calibration quality should be reviewed before promotion-grade use.

### 3. Learned Defer Overlay Replay Calibration

Artifacts:

- Replay script: `v4/scripts/run_unified_slot_opportunity_learned_defer_overlay_replay.py`
- Tests: `v4/tests/test_unified_slot_opportunity_learned_defer_overlay_replay.py`
- Selected overlay report: `v4/audit/autoresearch/unified_slot_opportunity_learned_defer_overlay_replay_relaxed_m0_w025_e3/report.md`
- Docs copy: `v4/docs/UNIFIED_SLOT_OPPORTUNITY_LEARNED_DEFER_OVERLAY_REPLAY_RELAXED_M0_W025_E3.md`

The overlay decision charges a challenger override:

`predicted_challenger_advantage - estimated_blocked_protocol101_cost - uncertainty_weight * blocked_cost_uncertainty`

The candidate is allowed only if adjusted advantage clears the configured margin and the estimated number of blocked Protocol101 entries is within budget.

Calibration runs:

1. Default strict overlay:
   - margin `250`
   - uncertainty weight `1`
   - max blocked entries `1`
   - result: safe but all-defer; zero challenger entries.

2. Relaxed no-uncertainty overlay:
   - margin `0`
   - uncertainty weight `0`
   - max blocked entries `3`
   - result: allowed 66 challenger entries and improved total PnL, but failed Q1/Q3 stress.

3. Relaxed count cap only:
   - margin `0`
   - uncertainty weight `0`
   - max blocked entries `1`
   - result: still failed Q1/Q3 and recent; count alone was not the right guardrail.

4. Selected overlay:
   - margin `0`
   - uncertainty weight `0.25`
   - max blocked entries `3`
   - result: allowed 62 challenger entries and preserved nonnegative Q1/Q3 stress deltas.

Selected overlay replay on the prior flat-calibrated policy:

| Slippage | Challenger Entries | Total Delta | Q1 Delta | Q3 Delta | Q4 Delta | Recent Delta |
|---:|---:|---:|---:|---:|---:|---:|
| `0.00` | 62 | `$44,200` | `$60` | `$90` | `$45,820` | `-$1,770` |
| `0.10` | 62 | `$45,320` | `$1,020` | `$150` | `$45,860` | `-$1,710` |
| `0.25` | 62 | `$47,000` | `$2,460` | `$240` | `$45,920` | `-$1,620` |

Decision: `learned_slot_opportunity_defer_overlay_replay_ready_for_next_preregistered_training`.

Specialist note: this guardrail repaired Q1/Q3 but still lost recent 2026 on the prior flat-calibrated policy. It was accepted only as a fixed guardrail for the next preregistered run, not as a champion.

### 4. Exactly One Preregistered Learned-Defer Neural Policy Run

Preregistration:

- Spec: `v4/docs/UNIFIED_CONSERVATIVE_NEURAL_POLICY_LEARNED_DEFER_PREREGISTERED_SPEC_V1.md`

Training artifacts:

- Training report: `v4/audit/autoresearch/unified_conservative_neural_policy_learned_defer_preregistered_v1/report.md`
- Model artifacts:
  - `v4/audit/autoresearch/unified_conservative_neural_policy_learned_defer_preregistered_v1/model_artifacts/flat_entry_model.pt`
  - `v4/audit/autoresearch/unified_conservative_neural_policy_learned_defer_preregistered_v1/model_artifacts/holding_lifecycle_model.pt`
  - `v4/audit/autoresearch/unified_conservative_neural_policy_learned_defer_preregistered_v1/model_artifacts/flat_scaler.json`
  - `v4/audit/autoresearch/unified_conservative_neural_policy_learned_defer_preregistered_v1/model_artifacts/holding_scaler.json`

Training recipe:

- seed `1`
- epochs `5`
- batch size `4096`
- hidden dim `128`
- flat train rows `450,000`
- holding train rows `450,000`
- flat train positive fraction `0.35`
- flat positive class weight `16`
- flat positive regression weight `8`
- flat tail class weight `1`
- train splits `q3_2025`, `q4_2025`
- validation split `q1_2026`
- diagnostic split `recent_2026`

Training report highlights:

| Head | Split | Rows | MAE | Corr | Allowed Rows | Allowed True Advantage Mean | Allowed True Positive Rate |
|---|---|---:|---:|---:|---:|---:|---:|
| flat_entry | train | 450,000 | 478.44 | 0.363 | 1,152 | -409.57 | 0.536 |
| flat_entry | validation | 120,000 | 647.08 | 0.205 | 798 | -1080.31 | 0.209 |
| flat_entry | recent | 120,000 | 516.11 | 0.135 | 184 | -615.19 | 0.272 |
| holding_lifecycle | train | 450,000 | 624.46 | 0.589 | 336,247 | 1122.57 | 0.939 |
| holding_lifecycle | validation | 150,000 | 983.63 | 0.422 | 109,508 | 1537.55 | 0.931 |
| holding_lifecycle | recent | 150,000 | 767.82 | 0.246 | 87,058 | 1141.23 | 0.936 |

Specialist note: the flat-entry sampled metrics remain concerning. Allowed rows still have negative mean oracle advantage in train/validation/recent samples. The strict serial replay is positive, but this discrepancy should be reviewed carefully. It may indicate label noise, calibration issues, serial-selection effects, a mismatch between sampled evaluation and replay, or an overly crude entry advantage target.

Replay artifacts:

- Replay report: `v4/audit/autoresearch/unified_conservative_neural_policy_learned_defer_preregistered_replay_v1/report.md`
- Trades: `v4/audit/autoresearch/unified_conservative_neural_policy_learned_defer_preregistered_replay_v1/trades_slippage_0_00.csv`
- Decisions: `v4/audit/autoresearch/unified_conservative_neural_policy_learned_defer_preregistered_replay_v1/decisions_slippage_0_00.csv`
- Docs copy: `v4/docs/UNIFIED_CONSERVATIVE_NEURAL_POLICY_LEARNED_DEFER_PREREGISTERED_REPLAY_V1.md`

Replay result:

| Slippage | Challenger Entries | Total PnL | Same-Scope Protocol101 | Delta | Trades |
|---:|---:|---:|---:|---:|---:|
| `0.00` | 60 | `$302,520` | `$235,740` | `$66,780` | 739 |
| `0.10` | 60 | `$287,740` | `$220,160` | `$67,580` | 739 |
| `0.25` | 60 | `$265,570` | `$196,790` | `$68,780` | 739 |

Per-split deltas:

| Slippage | Q1 2026 | Q3 2025 | Q4 2025 | Recent 2026 |
|---:|---:|---:|---:|---:|
| `0.00` | `$2,540` | `$3,475` | `$52,600` | `$8,165` |
| `0.10` | `$3,080` | `$3,535` | `$52,640` | `$8,325` |
| `0.25` | `$3,890` | `$3,625` | `$52,700` | `$8,565` |

Decision: `learned_slot_opportunity_defer_overlay_replay_ready_for_next_preregistered_training`.

Readiness after replay:

- `v4/docs/UNIFIED_NEURAL_TRAINING_READINESS.md`
- `v4/docs/FOUNDATION_HARDENING_AUDIT.md`

Current readiness decision:

- `foundation_preregistered_neural_run_complete_protocol101_challenge_blocked`

Additional neural training is now intentionally paused. Remaining work should focus on challenge/promotion blockers.

## Current System State

Protocol101 remains the paper default.

The learned-defer challenger is a research artifact with promising diagnostic replay, not a deployable or promoted model.

What is currently good:

- The unified serial replay is live-like in the important basics: one account, one contract, ask-entry, bid-exit, one open position, affordability, forced flat.
- The model now prices the opportunity cost of blocking Protocol101 before overriding.
- Diagnostic replay now beats same-scope Protocol101 across Q1, Q3, Q4, and recent under deterministic stress.
- The current pipeline explicitly tracks that model training should stop after the single preregistered run.

What is still weak:

- Same-scope diagnostic splits are research-exposed and cannot support final claims.
- Fill model calibration is absent.
- Live no-order parity for the challenger is not complete.
- Formal overfit controls are not complete.
- The flat-entry head’s sampled advantage metrics are suspicious: allowed rows have negative average oracle advantage.
- The slot-cost estimator is weaker on Q1/recent than on train-like splits.
- The holding/lifecycle head remains very hold-positive and may overhold in some distributions.

## Reproduction Commands

Build slot-cost labels:

```bash
python3 v4/scripts/run_unified_slot_opportunity_cost_label_dataset.py
```

Train slot-cost estimator:

```bash
python3 v4/scripts/run_unified_slot_opportunity_cost_estimator.py
```

Replay selected learned defer overlay on prior flat-calibrated policy:

```bash
python3 v4/scripts/run_unified_slot_opportunity_learned_defer_overlay_replay.py \
  --out-dir v4/audit/autoresearch/unified_slot_opportunity_learned_defer_overlay_replay_relaxed_m0_w025_e3 \
  --doc-path v4/docs/UNIFIED_SLOT_OPPORTUNITY_LEARNED_DEFER_OVERLAY_REPLAY_RELAXED_M0_W025_E3.md \
  --min-net-advantage-margin 0 \
  --blocked-cost-uncertainty-weight 0.25 \
  --max-blocked-protocol101-entries 3
```

Train exactly one preregistered learned-defer policy:

```bash
python3 v4/scripts/run_unified_conservative_neural_policy.py \
  --out-dir v4/audit/autoresearch/unified_conservative_neural_policy_learned_defer_preregistered_v1 \
  --doc-path v4/docs/UNIFIED_CONSERVATIVE_NEURAL_POLICY_LEARNED_DEFER_PREREGISTERED_V1.md \
  --epochs 5 \
  --batch-size 4096 \
  --hidden-dim 128 \
  --max-flat-train-rows 450000 \
  --max-flat-eval-rows 120000 \
  --max-holding-train-rows 450000 \
  --max-holding-eval-rows 150000 \
  --flat-train-positive-fraction 0.35 \
  --flat-positive-class-weight 16 \
  --flat-positive-regression-weight 8 \
  --flat-tail-class-weight 1
```

Replay that policy with the frozen learned defer overlay:

```bash
python3 v4/scripts/run_unified_slot_opportunity_learned_defer_overlay_replay.py \
  --model-artifacts v4/audit/autoresearch/unified_conservative_neural_policy_learned_defer_preregistered_v1/model_artifacts \
  --training-summary v4/audit/autoresearch/unified_conservative_neural_policy_learned_defer_preregistered_v1/summary.json \
  --out-dir v4/audit/autoresearch/unified_conservative_neural_policy_learned_defer_preregistered_replay_v1 \
  --doc-path v4/docs/UNIFIED_CONSERVATIVE_NEURAL_POLICY_LEARNED_DEFER_PREREGISTERED_REPLAY_V1.md \
  --min-net-advantage-margin 0 \
  --blocked-cost-uncertainty-weight 0.25 \
  --max-blocked-protocol101-entries 3
```

Refresh readiness:

```bash
python3 v4/scripts/run_unified_neural_training_readiness.py
python3 v4/scripts/run_foundation_hardening_audit.py
```

Run focused tests:

```bash
python3 -m pytest \
  v4/tests/test_foundation_hardening_audit.py \
  v4/tests/test_neural_training_readiness.py \
  v4/tests/test_unified_slot_opportunity_learned_defer_overlay_replay.py \
  v4/tests/test_unified_conservative_neural_policy.py \
  v4/tests/test_unified_conservative_neural_replay.py \
  v4/tests/test_unified_slot_opportunity_cost_estimator.py \
  v4/tests/test_unified_slot_opportunity_cost_label_dataset.py \
  v4/tests/test_unified_slot_opportunity_defer.py
```

Last focused verification result: `29 passed`.

## Questions For The ML Specialist

### Label Quality

1. Is `A_enter = Q(enter candidate) - Q(wait)` the right flat-entry target under a one-slot serial simulator?
2. Is `blocked_protocol101_pnl` the right opportunity-cost target, or should it be risk-adjusted utility, quantile loss, expected shortfall, or drawdown-adjusted cost?
3. Does the current label create an oracle/path optimism problem by using realized candidate exit horizons?
4. Should slot opportunity cost be predicted as a distribution rather than mean cost plus global p90 uncertainty?

### Model Objective

1. Why do sampled flat-entry “allowed” rows have negative mean oracle advantage, even though replay improves?
2. Is the multi-head MLP objective too weak for candidate-set ranking?
3. Should the entry model be trained pairwise/listwise over full candidate sets instead of rowwise?
4. Should the model predict baseline-relative incremental utility directly instead of raw `A_enter`?
5. Should the policy optimize a conservative utility target that includes drawdown and tail loss, not just expected advantage?

### Calibration And Risk

1. Is the learned defer overlay’s `0.25` uncertainty weight principled enough, or should it be replaced with calibrated conformal/quantile uncertainty?
2. Should the overlay require per-split or regime-conditioned uncertainty?
3. How should catastrophic loss probability and giveback risk be modeled?
4. Should position lifecycle training be reweighted toward actual entry-policy state distribution instead of oracle-entry or broad holding states?

### Validation

1. What is the right formal false-discovery control given many previous protocol attempts?
2. How should PBO/CSCV or a strategy-matrix audit be structured for these artifacts?
3. How much of the current diagnostic replay improvement could be data-mined from repeated split exposure?
4. What untouched block size is needed before any better-than-Protocol101 claim is credible?

### Execution Realism

1. How many paper/no-order fill observations are needed to calibrate a first empirical fill model?
2. Should the current deterministic ask/bid plus slippage replay remain the main offline gate?
3. What live no-order parity logs are necessary before paper deployment discussion?
4. How should quote freshness, latency, candidate availability, and missing quotes be modeled in training?

## Recommended Next Work

Do not continue neural model search immediately.

Recommended order:

1. Freeze the current learned-defer challenger artifacts and produce a promotion-readiness packet, clearly marked research-only.
2. Build live no-order parity for this exact challenger contract: full candidate surface, features, masks, slot-cost estimates, chosen action, latency, quote freshness, and account state.
3. Collect paper/no-order fill evidence; do not introduce stochastic fill replay until enough observations exist.
4. Build formal validation controls: strategy matrix, PBO/CSCV, block bootstrap, and daily/trade risk decomposition.
5. Collect/freeze the untouched evaluation block and score only once after the packet is frozen.
6. Ask an ML specialist to review the label/objective mismatch before any further architecture work.

The strongest ML direction is probably not “make the MLP bigger.” It is to improve the training formulation:

- candidate-set/listwise entry ranking
- baseline-relative incremental utility
- distributional slot-cost and downside-risk heads
- lifecycle training on actual entry-policy distributions
- conservative deferral when uncertainty is high

## Bottom Line

This session produced the first diagnostic unified challenger that beats same-scope Protocol101 across the current diagnostic splits under deterministic stress while respecting a learned Protocol101 slot-cost guardrail.

However, it is still a research challenger. The next step is not more model tweaking. The next step is expert review plus promotion-grade validation and execution realism. Only after that should we decide whether this challenger deserves a clean untouched evaluation against Protocol101.
