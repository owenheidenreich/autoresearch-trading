# Protocol101 Fair-Contract Synchronization Handoff

Generated: 2026-07-05

This document is a self-contained handoff for an external AI/research assistant that will not have full repository access. It summarizes the project objective, the evidence gathered, the current blocker, the failed repair attempts, and the specific help needed next.

## 1. Executive Summary

We are building a paper-trade-ready SPXW 0DTE options model for IBKR paper trading.

The core project requirement is:

```text
The model must be trained, validated, replayed, and eventually paper-traded under the same causal game.
```

That means historical validation cannot depend on information that the live system cannot see. The model must use a live-reproducible feature contract, and captured IBKR data must be replayable offline to reproduce the same model decisions.

The project originally had a strong historical Protocol101 result, but later investigation showed that the old result was not a trustworthy live benchmark because it came from a different stack:

| Comparison | Approx Result | Meaning |
|---|---:|---|
| Older official result | 359 trades, about $136.8k PnL | Old event-policy / precomputed-exit style simulation |
| Same-runner legacy contract | 221 trades, about $69.2k PnL | Current strict serial runner with historical-style features |
| Live-reproducible contract | 151 trades, about $17.5k PnL | Current strict serial runner with live-style/fair features |

The key discovery was that the old backtest and live IBKR were not playing the same game. Historical replays from Databento/ThetaData produced trades on days where the live IBKR paper runtime produced no trades. That triggered a full synchronization investigation.

We then moved to a fair/live-reproducible contract called:

```text
protocol101-live-v1
```

This contract is intended to define exactly what both historical replay and live IBKR replay are allowed to see.

Current status:

```text
No candidate is paper-trade ready yet.
```

The current blocker is not data recording, not IBKR startup, and not lack of code infrastructure. The blocker is model/design:

```text
We have not yet found a candidate that is both:
1. profitable enough under strict fair historical replay, and
2. stable/synchronized enough across captured IBKR replay and matching Databento/ThetaData replay.
```

Attempt107 was the best early fair-contract candidate. It passed strict historical replay, but failed paired IBKR-vs-Databento replay because tiny option-side differences changed model scores, selected contracts, or entry actions.

Subsequent aggregate/stable-selection attempts improved vendor-jitter robustness, but lost too much diagnostic edge.

The next research question is:

```text
Can we design a fair-contract model/label/lifecycle objective that preserves real edge while remaining robust to IBKR-vs-Databento option microstructure differences?
```

## 2. What We Need Help With

We need help diagnosing and solving the remaining blocker:

```text
Find a model design, label design, lifecycle/exit objective, or feature-contract repair that can pass both historical quality and live/historical synchronization under the fair contract.
```

More specifically, please help answer:

1. Is the remaining degradation mostly caused by the entry model, the labels, the lifecycle/exit policy, contract selection, or account/risk simulation?
2. Is `protocol101-live-v1` too restrictive, or is it the correct fair contract and the model should be retrained around it?
3. Are we using the wrong learning target? For example, should the model predict minute-level tradability, path-conditioned expected value, lifecycle-adjusted expected value, probability of achieving a target before stop, or something else?
4. Should entry and exit/lifecycle be trained as separate problems?
5. How can we keep option microstructure information without making the model brittle to tiny vendor differences in IV, spread, and quote sizes?
6. Should the model choose the exact contract directly, or first decide that a timestamp/setup is tradable and then select a contract through a deterministic stable policy?
7. What experiments should be run next, in what order, with what passing criteria?

Please do not suggest simply restoring the old backtest behavior unless you can prove it is causal and live-reproducible. The goal is not to recover fake edge. The goal is to recover real edge without sacrificing the synchronization guarantees.

## 3. Project Goal

The end goal is:

```text
Deploy a paper-trade-ready Protocol101 successor or repaired variant to IBKR.
```

Paper-trade-ready means:

- historical validation uses the same causal feature game that live trading uses;
- captured IBKR input can be replayed exactly after hours;
- matching Databento/ThetaData replay produces equivalent entry/exit decisions at the same timestamps;
- every decision, candidate, score, block, entry, exit, and account/risk state is reconstructable;
- the model has positive, robust historical expectancy under strict one-account serial replay;
- the model passes stress, drawdown, concentration, and stability checks;
- paper-submit remains disabled until the final readiness packet recommends owner review.

## 4. Non-Negotiable Guardrails

These are not optional:

1. No lookahead bias.
2. No precomputed future exits/PnL as runtime inputs.
3. No path-derived values in model features.
4. No hidden use of labels in runtime features.
5. Historical and live feature names/order/calculation must match.
6. Strict one-account serial replay.
7. Ask-side entry accounting.
8. Bid-side exit accounting.
9. No overlapping headline trades.
10. No unaffordable headline trades.
11. All positions flat by close.
12. No tuning thresholds directly on confirmation/recorder days.
13. June/July IBKR recorder days are confirmation/synchronization evidence, not training/model-selection data.
14. Real-money trading is out of scope.
15. IBKR paper-submit is out of scope until separate owner approval.

## 5. Important Definitions

### Older Official Result

The old headline Protocol101 result. It looked excellent, but it used older event-policy or precomputed-exit semantics and is no longer treated as a valid live benchmark.

### Same-Runner Legacy Contract

The current strict replay machinery applied to legacy historical-style features. This is more trustworthy than the old official result but still not necessarily live-reproducible.

### Live-Reproducible Contract

The fair contract:

```text
protocol101-live-v1
```

This restricts the model to fields and timestamp semantics that can be reproduced live. This is the current law of the project unless proven too restrictive in a specific, causal, repairable way.

### IBKR Recorder

An independent recorder-first data capture process. It captures IBKR SPX/VIX/SPXW market events without running live inference or submitting paper orders. It writes immutable raw capture logs. These logs are later replayed offline.

### Same-Input Replay

Run the model twice on the exact same captured IBKR input. This must be exact: same candidates, feature hashes, score hashes, actions, and lifecycle decisions.

### Paired Replay

Run the model on captured IBKR data and on matching Databento/ThetaData historical data at the same timestamps. Raw vendor values do not need to be identical, but non-threshold-adjacent actions should match, and mismatches must be explained.

## 6. Evidence Gathered So Far

### Recorder-First Captures

The project successfully moved away from fragile live paper trading and toward recorder-first capture.

Important completed capture days:

- 2026-06-30
- 2026-07-01
- 2026-07-02

The recorder captured full days with 390 regular-session minute checkpoints and no broker order calls.

Same-input replay for attempt107 over IBKR captures:

| Session | Decisions | Candidates | Entry Intents | Same-Input Exact |
|---|---:|---:|---:|---|
| 2026-06-30 | 360 | 10,772 | 3 | true |
| 2026-07-01 | 360 | 10,188 | 3 | true |
| 2026-07-02 | 360 | 11,598 | 3 | true |

Interpretation:

```text
The captured IBKR input is replayable. Same-input determinism is not the current blocker.
```

### Paired IBKR vs Databento/ThetaData Replay For Attempt107

| Session | Action Matches | Action Mismatches | Selected Contract Mismatches | Interpretation |
|---|---:|---:|---:|---|
| 2026-06-30 | 360 | 0 | 0 | Good action/contract match despite feature/score drift |
| 2026-07-01 | 358 | 2 | 2 | Failed: score-ceiling drift changed sequence |
| 2026-07-02 | 360 | 0 | 1 | Partial: entry times matched, but selected put differed |

Key paired mismatch examples:

1. July 1 at 10:36 ET:
   - IBKR entered `SPXW-20260701-07510.000-P` with score about `25.3371`.
   - Historical waited because the same contract scored about `55.5838`, triggering `max_score_ceiling=50`.
   - Main drift source: option-side features, especially tiny IV differences.

2. July 2 at 09:57 ET:
   - Both feeds entered.
   - IBKR selected `SPXW-20260702-07545.000-P`.
   - Historical selected `SPXW-20260702-07520.000-P`.
   - Main drift source: option spread/size/ranking sensitivity among close candidates.

Interpretation:

```text
Attempt107 is deterministic on the same input, but too sensitive to cross-vendor option microstructure differences.
```

## 7. Candidate Attempt History

### Attempt107: First Fair-Contract Lead

Candidate:

```text
attempt_107_policy0_hgb_blend35_relative_put_near_after0940_vwap_m2_10_cap3_scoreceil50_dailyloss500_plateau_s42
```

Historical strict replay after stress:

| Split | Trades | PnL | Profit Factor | Max Drawdown % Start | Status |
|---|---:|---:|---:|---:|---|
| validation | 24 | +$3,495 | 1.978 | -21.15% | pass |
| diagnostic_test | 26 | +$1,355 | 1.389 | -9.70% | pass |

Why it failed:

- paired IBKR-vs-historical replay failed;
- option IV/spread/size differences changed scores and selected contracts;
- score ceiling was brittle;
- concentration and training-window stability were not cleared.

### Feature-Jitter Gate

A development gate perturbed only the option-side fields known to cause IBKR-vs-Databento drift:

- IV +0.002
- IV -0.002
- spread widen 0.05
- spread tighten 0.05
- bid size half / ask size double
- bid size double / ask size half

Attempt107 result:

```text
fail
```

The biggest issue was not IV alone. The larger problem was spread/size-sensitive contract ranking. Small perturbations often changed which contract was selected and materially changed diagnostic PnL.

### Repair Families Tested

The following repair families were tested and did not produce a paper-ready candidate:

1. Mask vendor-sensitive option microstructure.
2. Bucket/quantize option microstructure.
3. Add top-vs-runner-up score margin filters.
4. Add deterministic fit-time feature noise augmentation.
5. Use jitter-stressed threshold selection.
6. Broaden entry filters.
7. Add compound context gates.
8. Use fixed absolute strike-offset selection.
9. Use aggregate decision-level objectives.
10. Block positive-momentum put entries.
11. Switch side based on momentum.

Important nuance:

```text
The aggregate decision-objective attempts were the best-shaped failures.
```

They improved option-feature jitter stability, but failed to recover enough diagnostic edge.

## 8. Best-Shaped Failure: Aggregate/Stable Selection

The most promising direction so far changed the structure from:

```text
score every contract -> pick top-scored contract
```

to:

```text
decide whether the timestamp/setup is tradable -> select contract using a stable deterministic policy
```

This was intended to avoid vendor-sensitive per-contract ranking.

### Aggregate Attempts 127-130

| Attempt | Target | Selection Mode | Validation PnL | Diagnostic PnL | Validation PF | Diagnostic PF | Jitter Gate |
|---|---|---|---:|---:|---:|---:|---|
| 127 | decision_best_profit_regression | stable_abs_offset_15 | +$3,100 | -$2,140 | 1.695 | 0.620 | pass |
| 128 | decision_best_profit_regression | stable_abs_offset_20 | +$3,690 | -$2,000 | 1.893 | 0.610 | pass |
| 129 | decision_profit_presence_classifier | stable_abs_offset_15 | +$4,280 | +$120 | 3.365 | 1.043 | pass |
| 130 | decision_profit_presence_classifier | stable_abs_offset_20 | +$4,760 | +$50 | 3.850 | 1.020 | pass |

Interpretation:

- These attempts passed the option-feature jitter gate.
- They did not pass historical diagnostic quality.
- Attempt130 is an important failed branch because it points toward the right robustness shape but too little edge.

### Broader Aggregate Attempts 131-132

| Attempt | Entry Filter | Validation PnL | Diagnostic PnL | Validation PF | Diagnostic PF | Jitter Gate |
|---|---|---:|---:|---:|---:|---|
| 131 | broader put filter | +$1,970 | -$330 | 1.540 | 0.925 | pass |
| 132 | premium>=7.5 variant | +$1,970 | -$330 | 1.540 | 0.925 | pass |

Interpretation:

- Still jitter-stable.
- Diagnostic edge missing.

### Momentum-Filtered Aggregate Attempts 133-135

| Attempt | Filter | Validation PnL | Diagnostic PnL | Diagnostic PF | Jitter Gate |
|---|---|---:|---:|---:|---|
| 133 | non-positive momentum puts | +$3,430 | -$460 | 0.859 | pass |
| 134 | positive OMAR + non-positive momentum puts | +$1,630 | -$150 | 0.947 | pass |
| 135 | premium>=7.5 + non-positive momentum puts | +$2,730 | -$180 | 0.945 | pass |

Interpretation:

- Blocking positive-momentum put entries did not recover edge.

### Side-Aware Aggregate Attempts 136-137

| Attempt | Filter | Validation PnL | Diagnostic PnL | Diagnostic PF | Jitter Gate |
|---|---|---:|---:|---:|---|
| 136 | calls if momentum positive, puts otherwise | +$3,050 | -$2,700 | 0.636 | pass |
| 137 | same plus premium>=7.5 | +$3,050 | -$2,700 | 0.636 | pass |

Interpretation:

- Side-switching on momentum made diagnostic performance worse.
- The next fix is probably not another simple side/context filter.

## 9. Current Blocker In Plain English

The old model made better-looking money partly because it was playing a historical-only game that live IBKR cannot reliably reproduce.

When we force the model to use only information that can exist live, performance drops.

When we make the model robust to IBKR-vs-Databento option differences, performance drops further.

Therefore the blocker is:

```text
Find a fair/live-reproducible model objective that still has real edge after removing unfair or vendor-brittle advantages.
```

The current best path is not to keep patching IBKR or Databento to agree. The current best path is to redesign the model objective and/or lifecycle labels so that the model learns robust causal setups instead of fragile per-contract microstructure differences.

## 10. Suspected Root Causes

The remaining degradation may come from a mixture of:

1. Entry labels are too tied to a fixed lifecycle policy.
2. Lifecycle/exit assumptions may be mispricing continuation vs reversal under the fair contract.
3. The model may be learning whether a specific contract had favorable path PnL rather than whether the timestamp/setup was truly tradable.
4. Exact per-contract ranking is vendor-sensitive.
5. Option microstructure fields contain useful information but cannot be used raw because tiny differences across vendors flip rankings/actions.
6. The current put/VWAP pocket may be too narrow or too regime-specific.
7. The validation/diagnostic split reveals that some candidate families overfit one window and fail the next.
8. Current labels may reward paths that are not realistic once strict serial account, ask/bid accounting, cooldowns, trade caps, and forced-flat rules are imposed.

## 11. Recommended Next Work

Please focus on this next:

```text
Diagnose entry-label vs lifecycle/exit edge loss under the fair contract.
```

Suggested experiment sequence:

### Step 1: Label/Lifecycle Attribution

For the aggregate/stable candidates, especially attempts 130 and 131:

- inspect selected trades in validation vs diagnostic;
- compute outcome under multiple lifecycle policies if available;
- compare entry signal quality before lifecycle;
- compare MFE/MAE after entry;
- measure whether losers were bad entries or bad exits;
- measure whether winners were exited too early or selected with the wrong contract;
- measure whether fixed stop/target/hold labels are mismatched to the live-reproducible features.

Goal:

```text
Determine whether diagnostic edge loss is caused by bad entry timing, bad contract selection, or bad lifecycle/exit labeling.
```

### Step 2: Test A Decision-Level Expected-Value Target

Instead of per-contract label targets, test a minute/setup-level target such as:

- best achievable fair-contract lifecycle value among eligible contracts;
- probability at least one eligible contract exceeds a profit threshold;
- expected value after strict lifecycle and account rules;
- robust lower-quantile expected value across nearby strikes;
- risk-adjusted expected value after spread/slippage stress.

The exact contract can then be selected by a stable deterministic rule.

Goal:

```text
Avoid fragile per-contract ranking while preserving timestamp-level tradability.
```

### Step 3: Test Robust Contract Selection

Instead of selecting the top-scored contract, test stable contract-selection policies:

- fixed offset bucket selected by context;
- nearest strike with acceptable spread/premium;
- select among a stable strike band by median score;
- select contract only if nearby strikes agree on direction/quality;
- choose lower-variance contract within the eligible cluster, not the highest raw score.

Goal:

```text
Make selected contract stable under small IBKR-vs-Databento microstructure drift.
```

### Step 4: Separate Entry From Exit

Train/validate entry as:

```text
Should this timestamp/setup be traded?
```

Train/validate lifecycle as:

```text
Given an open trade, should it hold, exit, or force-flat?
```

Goal:

```text
Stop forcing one entry score to encode both entry edge and exit/lifecycle path quality.
```

### Step 5: Add Robustness To The Objective, Not Only The Gate

Previous robustness was mostly post-hoc gating. Consider training objectives that penalize unstable decisions:

- require neighboring strikes to agree;
- train with grouped candidate/timestamp losses;
- penalize prediction variance under simulated vendor perturbations;
- optimize worst-case or lower-quantile outcome across microstructure perturbations;
- use ranking only at the timestamp/cluster level, not exact contract level.

Goal:

```text
Keep edge while making robustness a learned property rather than an after-the-fact filter.
```

## 12. Passing Criteria For A Proposed Fix

A candidate is not useful unless it passes all of these:

### Historical Quality

Minimum:

- validation profitable after stress;
- diagnostic profitable after stress;
- profit factor >= 1.25, preferably >= 1.50;
- at least 20 trades in both validation and diagnostic unless a strong reason justifies fewer;
- max drawdown <= 35%;
- no excessive top-day or top-trade concentration;
- no unaffordable trades;
- no overlapping headline trades;
- flat by close.

### Robustness

- passes option-feature jitter gate;
- selected contract and action are stable under small IV/spread/size perturbations;
- no brittle score ceiling that turns tiny vendor differences into action flips;
- no dependence on unavailable volume/open-interest semantics unless captured live and proven causal.

### Synchronization

- exact same-input replay on captured IBKR input;
- paired IBKR-vs-Databento replay matches non-threshold-adjacent actions;
- selected contract mismatches must be economically equivalent or explained;
- threshold-adjacent mismatches must be bounded and non-systematic;
- at least one threshold-crossing day must be included.

### Governance

- no paper-submit;
- no default change;
- no real-money path;
- no tuning on confirmation recorder days;
- experiment registry updated.

## 13. Minimal File Upload Package

If you can upload only a small number of files to the external AI, upload this document plus these files.

### Tier 1: Minimal Context

1. `v4/docs/PROTOCOL101_FAIR_CONTRACT_EXTERNAL_AI_HANDOFF_2026_07_05.md`
2. `v4/audit/autoresearch/protocol101_fair_contract_attempt107_readiness_packet/report.md`
3. `v4/audit/autoresearch/protocol101_fair_contract_attempt107_readiness_packet/summary.json`

This is enough for high-level reasoning.

### Tier 2: Add Key Code For Understanding Current Mechanics

4. `v4/live/protocol101_feature_contract.py`
5. `v4/model/supervised_pilot.py`
6. `v4/scripts/run_protocol101_fair_contract_training_runner.py`
7. `v4/scripts/run_protocol101_fair_contract_selected_candidate_export.py`
8. `v4/scripts/run_protocol101_fair_contract_feature_jitter_gate.py`
9. `v4/scripts/run_protocol101_fair_contract_failure_diagnostic.py`

This is enough to understand the fair contract, training runner, selection/export logic, jitter gate, and diagnostics.

### Tier 3: Add Search/Replay Evidence

10. `v4/scripts/run_protocol101_fair_contract_model_search.py`
11. `v4/scripts/run_protocol101_paired_live_historical_diff.py`
12. `v4/audit/autoresearch/protocol101_fair_contract_model_search_expanded_jul_dec2025_128_q1_aggregate_decision_objective/report.md`
13. `v4/audit/autoresearch/protocol101_fair_contract_model_search_expanded_jul_dec2025_128_q1_broader_aggregate_decision_objective/report.md`
14. `v4/audit/autoresearch/protocol101_fair_contract_model_search_expanded_jul_dec2025_128_q1_momentum_filtered_aggregate_objective/report.md`
15. `v4/audit/autoresearch/protocol101_fair_contract_model_search_expanded_jul_dec2025_128_q1_side_aware_aggregate_objective/report.md`
16. `v4/audit/autoresearch/protocol101_fair_contract_attempt107_pair_attribution/report.md`

This is enough for the external AI to audit what failed and recommend a next experiment.

### Tier 4: Only If The External AI Will Run Code

If the external AI is expected to run experiments, it will need more than this handoff. It will need the relevant processed datasets, model artifacts, manifests, and probably most of the `v4` package. That is much larger and should be handled separately.

## 14. Recommended Prompt To Give The External AI

```text
You are helping diagnose a trading ML synchronization problem.

We are building a paper-trade-ready SPXW 0DTE options model for IBKR. The model must use the same causal live-reproducible feature game in historical validation, captured IBKR replay, and future live paper trading.

The old Protocol101 historical result was strong but is no longer trusted as a live benchmark because it used older event-policy/precomputed-exit semantics and historical-only feature behavior. We moved to a fair contract called protocol101-live-v1.

The current blocker is that no candidate passes both:
1. strict fair historical replay profitability, and
2. IBKR-vs-Databento/ThetaData synchronization robustness.

Attempt107 passed strict historical replay but failed paired replay because small option IV/spread/size differences changed scores, actions, or selected contracts. Masking, bucketing, score margins, noise augmentation, jitter thresholds, broader filters, compound filters, fixed-offset selection, momentum filters, and side-aware filters failed to produce a paper-ready candidate.

Aggregate decision-level objectives plus stable contract selection improved option-feature jitter robustness, but diagnostic profitability collapsed. This suggests the remaining problem may be entry-label design, lifecycle/exit objective design, candidate target construction, or how continuation/reversal is priced under the fair contract.

Please review the attached handoff/report/code and propose the next scientifically valid experiments. Focus on diagnosing whether the edge loss comes from entry timing, contract selection, labels, lifecycle/exit policy, or account simulation. Do not propose restoring old historical-only edge unless it can be proven causal and live-reproducible. Do not suggest threshold tuning on confirmation days.

Deliver:
1. your diagnosis of the most likely remaining root cause;
2. the next 3-5 experiments in priority order;
3. exact passing/failing criteria;
4. any changes to feature contract, labels, model objective, or lifecycle design;
5. whether the project should preserve frozen Protocol101 behavior, retrain on the fair contract, or redesign entry/exit as separate models.
```

## 15. Current Best Working Hypothesis

The most likely remaining blocker is:

```text
The model is still trying to learn trade value through labels/objectives that are too entangled with exact contract ranking and lifecycle path outcomes.
```

The fair data contains opportunity, as shown by oracle-style diagnostics, but the learned model has not yet converted that opportunity into robust diagnostic expectancy.

Therefore the next best direction is:

```text
Move from fragile per-contract entry scoring toward a robust timestamp/setup tradability model plus stable contract selection plus separately validated lifecycle/exit logic.
```

This should be tested, not assumed.

## 16. What Not To Do

Do not:

- tune thresholds until June/July paired days match;
- patch historical data to look like IBKR unless the transformation is part of a general causal contract;
- patch IBKR data to mimic Databento quirks;
- use the old 359-trade result as the target to recover;
- use future exits or precomputed PnL in runtime features;
- promote any candidate that only wins on validation and fails diagnostic;
- accept no-trade paired days as sufficient synchronization proof;
- use IBKR paper fills as proof of real execution quality;
- start hill climbing without an experiment registry and protected confirmation process.

## 17. One-Sentence Problem Statement

We need a Protocol101 successor that can make money under a strict live-reproducible feature contract and make the same decisions when replayed from captured IBKR data and matching Databento/ThetaData historical data; current candidates either make money but are too vendor-fragile, or become vendor-stable but lose too much edge.

