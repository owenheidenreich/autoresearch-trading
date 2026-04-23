# Layer-2 Route-Aware Fallback — 2026-04-21

## Summary

This cycle implemented the full route-aware fallback plan and falsified it.

The intended hypothesis was:

- keep the winning entry-selection trunk from the detach-side branch
- replace the scalar side head with detached fallback value heads
- let non-teacher bars choose among `call`, `put`, and implicit `flat`
- solve the chop failure mode without disturbing the teacher-directed bars

What actually happened:

- the code path worked end-to-end
- the route-aware branch trained, replayed, and produced artifacts for all four fallback quantiles
- but it did **not** improve the no-teacher fallback mechanism
- instead it collapsed into a teacher-only selector and regressed badly on PF and drawdown

Verdict:

- **negative result**
- do **not** launch GPU from this branch
- do **not** broaden the neural search from here

## What Was Implemented

Code surface:

- [v3/layer2/neural.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/neural.py)
  - added `Layer2RouteAwareSharedEncoder`
  - detached fallback call / put heads
- [v3/layer2/train_neural.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/train_neural.py)
  - added `--policy-mode route_aware_fallback`
  - trains entry + fallback-call + fallback-put heads
- [v3/layer2/replay.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/replay.py)
  - route-aware replay path
  - implicit `flat` action at policy time
- [v3/layer2/common.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/common.py)
  - route-aware thresholding, ranking, and direction helpers
- [v3/analysis/layer2_fallback_route_diagnostic.py](/Users/gduby/Documents/autoresearch-trading/v3/analysis/layer2_fallback_route_diagnostic.py)
  - exact chosen-bar controls for `teacher+put/call/flat/random/oracle_best`
- [v3/analysis/layer2_random_direction_ablation.py](/Users/gduby/Documents/autoresearch-trading/v3/analysis/layer2_random_direction_ablation.py)
  - now supports `fallback_only` randomization mode

No new features were added and no dataset rebuild was required.

## Baseline Mechanism Check

Before training the new branch, the current max-PF winner was decomposed:

- artifact: [v3/artifacts/layer2_shared_enc_fixedq_detach](/Users/gduby/Documents/autoresearch-trading/v3/artifacts/layer2_shared_enc_fixedq_detach)
- chosen bars: `275`
- teacher bars: `99`
- fallback bars: `176`

Fallback-route controls on that winner:

| control | PF | DD | comment |
|---|---:|---:|---|
| teacher + put | **1.455** | **36.9%** | actual winning branch |
| teacher + call | 0.868 | 140.3% | clearly worse |
| teacher + flat | 1.142 | 58.7% | better chop protection, but gives up too much edge |
| teacher + random mean | 1.209 | 76.3% | shows routing head has room to add value |
| teacher + oracle-best | 5.321 | 7.2% | huge theoretical upper bound |

This confirmed the target problem:

- the fallback path matters
- `put` was the best blunt fallback
- but the gap to oracle-best was large enough to justify a route-aware test

Control note:

- effective trade count varies by control because `teacher+flat` skips fallback bars and `teacher+call` / `teacher+oracle_best` can only use fallback bars where the relevant side passes the contract filters

## Route-Aware Sweep

Runs:

- [v3/artifacts/layer2_route_fallback_q50](/Users/gduby/Documents/autoresearch-trading/v3/artifacts/layer2_route_fallback_q50)
- [v3/artifacts/layer2_route_fallback_q60](/Users/gduby/Documents/autoresearch-trading/v3/artifacts/layer2_route_fallback_q60)
- [v3/artifacts/layer2_route_fallback_q70](/Users/gduby/Documents/autoresearch-trading/v3/artifacts/layer2_route_fallback_q70)
- [v3/artifacts/layer2_route_fallback_q80](/Users/gduby/Documents/autoresearch-trading/v3/artifacts/layer2_route_fallback_q80)

All four runs produced the **same** replay result:

| run | PF | DD | trades | trades/day |
|---|---:|---:|---:|---:|
| q50 | 1.131 | 93.1% | 270 | 0.900 |
| q60 | 1.131 | 93.1% | 270 | 0.900 |
| q70 | 1.131 | 93.1% | 270 | 0.900 |
| q80 | 1.131 | 93.1% | 270 | 0.900 |

Per-fold calibration showed why:

- `entry_threshold` changed by fold as expected
- `side_threshold = 0.0` in **every fold** for **every quantile**

So the fallback quantile sweep never became an operative lever.

## Failure Mechanism

The representative run is `q50`.

Artifact:
- [v3/artifacts/layer2_route_fallback_q50](/Users/gduby/Documents/autoresearch-trading/v3/artifacts/layer2_route_fallback_q50)

Fallback-route diagnostic:

- chosen bars: `270`
- teacher bars: `270`
- fallback bars: `0`

That is the whole story.

The route-aware branch did **not** learn a better fallback policy. It avoided fallback trades altogether.

Consequences:

- the `q50/q60/q70/q80` sweep tied because the fallback gate never bound
- fallback-route controls on `q50` collapsed to the same replay result because there were no fallback bars to reroute
- fallback-only randomization is degenerate on this run because there are no fallback directions to randomize

This branch therefore failed on mechanism, not just on headline PF.

## Why It Lost

Compared with the detach-side max-PF baseline:

| artifact | PF | DD | trades/day |
|---|---:|---:|---:|
| detach-side baseline | **1.455** | **36.9%** | 0.917 |
| route-aware q50 | 1.131 | 93.1% | 0.900 |

What got worse:

- entry learnability degraded relative to the detach-side branch
- fallback confidence collapsed toward zero
- the model selected only teacher bars
- fold 0 remained badly negative, and fold 4 also weakened

The branch solved nothing about chop fallback because it never actually expressed a fallback action.

## Acceptance Check

Planned bar for promotion to GPU:

- aggregate PF `> 1.455`
- aggregate DD `<= 40%`
- trades/day in `[0.75, 1.00]`
- fold 0 PF `>= 1.00` or major loss reduction
- positive delta over fallback-randomized and fallback-flat controls
- slippage PF `> 1.20` at +`$25` cost

Actual result:

- PF failed
- DD failed
- fold 0 failed
- mechanism failed because no fallback trades were selected

So the branch is not close. This is a clean stop, not a “tune one more thing” situation.

## What This Means

The detach-side result is still real, but its edge is not coming from a healthy learned fallback route.

The next grounded question, if Layer-2 is revisited, is narrower:

- on **non-teacher bars only**, should the model trade at all?
- if so, should it choose `call`, `put`, or `flat`?

That is a different training problem from the route-aware branch implemented here, because this cycle showed teacher bars can dominate the policy and hide the fallback task entirely.

## What Not To Do Next

- Do not launch GPU on the route-aware branch.
- Do not broaden the fallback quantile sweep.
- Do not add more late-session teachers.
- Do not interpret the tied `q50/q60/q70/q80` runs as robustness. They tied because the fallback gate collapsed.
