# Protocol101 External AI Round 2 Prompt

Generated: 2026-07-05

Use this as the next message to the external AI.

## Round 2 Message

Thank you. Your response was useful, especially the distinction between:

1. synchronization architecture, which may be mostly solved by aggregate decision objectives plus deterministic/stable contract selection; and
2. edge/generalization, which is still unsolved because the stable candidates lose diagnostic expectancy.

I want to turn your recommendations into exact next experiments that can be implemented in the codebase without accidentally reintroducing lookahead bias, overfitting, or vendor-specific patches.

Please be as concrete as possible. I am not asking for general advice; I am asking you to specify the exact experiment designs, target definitions, null baselines, scoring rules, and failure interpretations.

## Current Corrections / Clarifications

One wording distinction matters:

```text
I agree that the synchronization architecture appears to be identified.
I do not yet want to call vendor synchronization solved in the final project-gate sense.
```

The project should still require paired IBKR-vs-Databento/ThetaData confirmation for any final candidate. But for now, I accept your point that the next blocker is not “make IBKR match Databento.” The next blocker is proving that the fair game has real edge and that the learned model beats simple stable gates without mining noise.

The current best failed shape is:

```text
aggregate decision-level objective
  + deterministic/stable contract selection
  + option microstructure mostly moved out of exact score ranking
```

The immediate question is:

```text
Is there real tradable edge in the fair contract, or have we mostly been selecting noise from small samples and max-over-contract labels?
```

## What I Need From You Now

Please answer the following in detail.

### 1. Define Experiment 1 Exactly: Gate-Only And Null Baselines

You proposed:

```text
gate-only strategy:
base gate + deterministic offset-20 + first-come cap-3 serial + fixed lifecycle
```

Please define this exactly.

Questions:

1. Which gate should be tested first?
   - `put_near_after_0940_vwap_m2_10`
   - `put_near_after_0940_vwap_m2_10_near_vwap`
   - `put_near_after_0940_vwap_m2_10_premium_gte_7_5`
   - some other gate
2. Should the gate-only strategy enter the first eligible minute of the day, the highest label minute in hindsight is forbidden, the first eligible minute after cooldown, or a deterministic schedule?
3. How should the fixed contract selector work?
   - nearest absolute offset 20?
   - nearest offset 15 or 20 depending on VIX?
   - premium band first, then offset?
   - spread/freshness eligibility first, then offset?
4. How should we handle multiple eligible contracts at the same offset distance?
5. Should the gate-only baseline be tested in both:
   - unconstrained parallel signal mode, and
   - strict serial cap-3 account replay?
6. Which one is allowed to count as paper-readiness evidence?
7. What exactly is the random-entry-in-gate null?
   - sample random eligible minutes within each day?
   - match trade count per day?
   - preserve time-of-day distribution?
   - preserve side/offset/premium distribution?
   - apply the same cooldown/trade cap/daily loss rules?
8. How many bootstrap/random runs are enough?
9. What p-value or percentile threshold should a learned model beat?
10. What would each possible result mean?

Please give a table like:

```text
Experiment 1A: gate-only strict serial
Inputs:
Selection:
Lifecycle:
Metrics:
Pass:
Fail means:

Experiment 1B: random-in-gate null
Inputs:
Sampling:
Metrics:
Pass:
Fail means:
```

### 2. Define The Apples-To-Apples Label Uplift Table

You wrote:

```text
average realizable label of the offset-20 candidate across all gated minutes vs across model-selected minutes, per split
```

Please define this precisely.

Questions:

1. Is the unit of analysis decision-minute, candidate-contract, or serial-trade?
2. Should we compute this before or after account constraints?
3. Should label uplift compare:
   - all gated decision minutes,
   - gate-only selected minutes,
   - model-selected minutes,
   - random-selected minutes?
4. How should we handle multiple contracts in a minute?
5. Should we use mean, median, lower quartile, trimmed mean, or winsorized mean?
6. How should this be stratified?
   - split
   - month
   - time of day
   - VIX regime
   - below/above VWAP
   - momentum bucket
   - premium bucket
7. What would prove the learned model adds value over the gate?
8. What would prove the learned model is just selecting noise?

Please provide a recommended schema for the output CSV and summary JSON.

### 3. Define The Correct Label Surgery

You recommended replacing max-over-contract labels with:

```text
realizable-policy value
```

Please specify the exact label.

Current suspected issue:

- `decision_best_profit_regression` may be biased by max-over-many-contract paths.
- `decision_profit_presence_classifier` may be too degenerate because too many minutes have at least one profitable contract.
- Old per-contract labels may reward noisy path extremes.

Questions:

1. Should the label be based on the deterministic selected contract only?
2. Should deterministic selection happen before labeling?
3. Should the label be:
   - raw PnL dollars,
   - return on premium,
   - risk-adjusted PnL,
   - lower-quantile band PnL,
   - winsorized PnL,
   - classification threshold based on fees/spread buffer?
4. If using a band-median label across offsets 10/15/20/25, how exactly is that computed?
5. Should label be per decision minute rather than per contract?
6. Should labels be unconstrained by account state, then account constraints applied only in replay?
7. How do we avoid lifecycle leakage while still using future outcome labels for supervised training?
8. What is the exact positive-class threshold for classification?
9. What winsorization bounds do you recommend?
10. Should rejected/stale/wide-spread contracts be absent from the label universe or labeled as non-tradable?

Please define 2-3 candidate label formulas and rank them.

### 4. Define The Feature Set For Experiment 2

You recommended:

```text
Train HGB on index-level features only.
Option data belongs in gates and labels, not the score.
```

Please define what should and should not be in the score model.

Questions:

1. Which exact feature categories should be allowed?
   - SPX return/momentum
   - SPX VWAP distance
   - OMAR/opening range
   - realized range
   - VIX level/change
   - time of day
   - day of week
   - distance from open/close
   - prior day context
   - option premium band?
   - option spread band?
   - option IV/Greek fields?
2. Should option microstructure be completely excluded from the model score, or included only as coarse eligibility gates?
3. If any option field is allowed in the model, how should it be made vendor-stable?
4. Should in-house Greeks/IV be allowed if computed from raw quotes using the same pricer on both feeds?
5. Should volume/open interest be excluded entirely unless live-equivalent semantics are proven?
6. Should the model score a decision minute only, with no per-contract feature rows?
7. How should deterministic contract selection use option data without creating score brittleness?

Please give an allowed-feature matrix:

```text
Feature category | Allowed in score? | Allowed in gate? | Allowed in label? | Reason
```

### 5. Define Walk-Forward Evaluation

You proposed:

```text
2+ years, purged walk-forward folds, >=8 folds, >=150 pooled out-of-fold trades
```

Please make this implementable.

Questions:

1. What train/test fold size do you recommend for 0DTE SPXW?
   - train N sessions
   - embargo/skipped sessions
   - test M sessions
2. Should folds be anchored expanding windows or rolling windows?
3. Should hyperparameters be frozen globally or selected within each training fold?
4. How should thresholds be selected without leaking?
5. How should model families be compared with multiple-testing control?
6. What if we do not yet have two years of fully processed `protocol101-live-v1` rows?
7. What is the minimum acceptable fallback evaluation using the current available data?
8. Should June/July 2026 IBKR recorder days remain completely excluded from training/model selection?
9. How should confirmation days be used once a candidate is frozen?

Please provide a concrete fold template.

### 6. Define Lifecycle/Exit Attribution

You said lifecycle is genuinely unmeasured.

Please specify the next diagnostic.

Questions:

1. How should we decompose PnL gap into:
   - entry-minute quality,
   - selected-contract delta,
   - lifecycle/exit delta,
   - account-simulation/cap/daily-loss delta?
2. What MFE/MAE/horizon metrics should be reported?
3. Should lifecycle grid search happen now or only after entry edge is proven?
4. Which lifecycle policies should be pre-registered?
5. How do we prevent lifecycle search from becoming another overfit surface?

Please provide an exact output table for this diagnostic.

### 7. Challenge Or Confirm This Revised Plan

Based on your response, my current revised plan is:

```text
1. Stop training new model candidates until model-free/null baselines are complete.
2. Run gate-only strict serial replay and random-in-gate null baseline.
3. Run apples-to-apples label uplift diagnostics.
4. If gate-only is positive and beats random, treat it as a possible simple candidate shape.
5. If gate-only fails but labels still look positive, diagnose oracle/time-sampling bias.
6. Rebuild labels as deterministic realizable-policy value.
7. Train an index/context-only aggregate decision model.
8. Keep exact contract selection deterministic and coarse.
9. Keep lifecycle fixed until entry edge is proven.
10. Only after historical/fold evidence passes, return to paired IBKR-vs-Databento confirmation.
```

Please challenge this plan. What is wrong, missing, or dangerously underspecified?

## Files I Can Upload

I can upload up to 20 files, each under 30 MB. I am attaching the most important reports and code files available.

Please tell me if you need a different file or if one of these is unnecessary.

