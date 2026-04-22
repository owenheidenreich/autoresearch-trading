# v3 Handoff — Post-A1, Layer-2 Mainline

**Date:** 2026-04-21  
**Project framing:** SPX 0DTE long premium, $25k account, 1 contract, Moderate-tier Layer 0 rails

## Purpose

This is the current handoff for `v3` after:
- Stage 1 research completed
- A1 ORC sigma gate shipped
- late-session teacher search falsified
- W2a soft features shipped into `v2.2`
- first `v3` Layer-2 supervised branch implemented and screened

The mainline is no longer “write more teachers” or “keep tuning `v2/train.py`.”
The mainline is now the `v3` Layer-2 bar-state branch in [v3/layer2](/Users/gduby/Documents/autoresearch-trading/v3/layer2).

## TL;DR

What is actually true now:

- **A1 is real and shipped.** ORC with the hard `sigma_pos` direction veto reduced `side_error 402 → 176` with exact dry-run/runtime agreement. See [orc_direction_fix_2026_04_20.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/orc_direction_fix_2026_04_20.md).
- **Late-session teacher families did not survive.** The tournament went `0/8`; that regime is Layer-2-only. See [late_session_tournament_2026_04_20.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/late_session_tournament_2026_04_20.md).
- **W2a features shipped, but `v2` still failed.** `exp_179` confirmed the old training loop is the bottleneck, not the feature surface. See [w2a_handoff_2026_04_21.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/w2a_handoff_2026_04_21.md).
- **The new mainline is `v3` Layer-2.** Export/train/replay tooling now exists under [v3/layer2](/Users/gduby/Documents/autoresearch-trading/v3/layer2).
- **There are now two Layer-2 baselines with different roles.**
  - Stable CPU reference:
    - fixed-quantile hybrid tree policy
    - `287` trades, `PF 1.122`, `DD 56.2%`, `0.957` trades/day
  - Max-PF neural research baseline:
    - shared encoder + `--detach-side`
    - `275` trades, `PF 1.455`, `DD 36.9%`, `0.917` trades/day
    - not yet cleared for GPU or paper trading because the later diagnostics showed the edge is still dominated by entry selection plus teacher-conditioned fallback
- **The route-aware fallback follow-up failed.**
  - `q50/q60/q70/q80` all tied at `270` trades, `PF 1.131`, `DD 93.1%`, `0.900` trades/day
  - the representative run chose `270` teacher bars and `0` fallback bars
  - no GPU launch is justified from that branch
- **The fallback-only probe also failed to beat blunt put fallback.**
  - `teacher+put_or_flat_model`: `PF 1.340`, `DD 58.8%`, `0.613` trades/day
  - `teacher+fallback_model`: `PF 1.159`, `DD 96.6%`, `0.697` trades/day
  - baseline `teacher+put` remains best at `PF 1.455`, `DD 36.9%`, `0.917` trades/day
  - this points away from routing and toward fallback regime gating

References:
- [layer2_entry_side_2026_04_21.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/layer2_entry_side_2026_04_21.md)
- [layer2_shared_encoder_diagnostics_2026_04_21.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/layer2_shared_encoder_diagnostics_2026_04_21.md)
- [layer2_diagnostic_full_2026_04_21.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/layer2_diagnostic_full_2026_04_21.md)
- [layer2_route_aware_fallback_2026_04_21.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/layer2_route_aware_fallback_2026_04_21.md)
- [layer2_fallback_only_probe_2026_04_21.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/layer2_fallback_only_probe_2026_04_21.md)

## What Changed

### Stage 1 / teacher layer

- ORC now carries the shipped `sigma_pos` hard veto.
- FailedBreak remains the only other baseline teacher.
- No new late-session teacher should be added on the current evidence.

### Feature / denominator layer

- `v2.core.market_structure` is the shared source of truth for:
  - `sigma_pos`
  - OMAR fields
  - last-10 structure
- The 8 W2a regime features are stable in `v2.2` and are now joined into `v3` Layer-2 export.

### Layer-2 tooling

New package:
- [v3/layer2/export_dataset.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/export_dataset.py)
- [v3/layer2/train_entry_side.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/train_entry_side.py)
- [v3/layer2/replay.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/replay.py)
- [v3/layer2/README.md](/Users/gduby/Documents/autoresearch-trading/v3/layer2/README.md)

What they do:
- export one row per eligible bar after both oracle passes
- join the W2a regime block from `v2.2`
- train simple tabular entry + side models
- calibrate thresholds on val folds only, now with explicit fixed-quantile support
- replay a one-trade-per-day policy against the post-A1 teacher baseline

New neural path:
- [v3/layer2/neural.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/neural.py)
- [v3/layer2/train_neural.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/train_neural.py)

What it does:
- keeps the same export bundle, replay interface, and artifact layout
- swaps the tree regressors for small torch MLP regressors
- supports `--latest-only` smoke tests and `--device cuda` for GPU training
- now also supports:
  - shared-encoder `--detach-side`
  - `route_aware_fallback` policy mode with detached call/put fallback heads
  - fallback-route diagnostics via [v3/analysis/layer2_fallback_route_diagnostic.py](/Users/gduby/Documents/autoresearch-trading/v3/analysis/layer2_fallback_route_diagnostic.py)
  - fallback-only randomization controls via [v3/analysis/layer2_random_direction_ablation.py](/Users/gduby/Documents/autoresearch-trading/v3/analysis/layer2_random_direction_ablation.py)

## Current Best Hypothesis

The surviving signal is still **bar-quality first, structural direction second**.

More concretely:

- The entry model is learning something real. Top-decile and top-1/day bar selection still beat random eligible bars and teacher-triggered bars on oracle-quality metrics.
- The detach-side neural branch improved aggregate PF and DD, but the full diagnostic showed the directional edge is still not clean enough to promote as a standalone learned routing policy.
- The route-aware fallback follow-up did not fix the real problem. Instead of improving fallback action quality, it collapsed into teacher-only selections.
- The fallback-only probe also failed. Training only on non-teacher rows did not beat the blunt put fallback; even the narrower put-vs-flat control regressed versus always-put.
- The main unresolved problem is now very specific:
  - what should happen on **no-teacher bars**
  - but now more specifically whether fallback puts should be **suppressed in low-payoff regimes**
  - rather than whether no-teacher bars should be rerouted to calls

So the next useful hypothesis is not “run GPU.” It is:

- model fallback **regime gating** explicitly and separately
- likely as a `put-vs-flat` payoff-sufficiency problem on no-teacher bars
- with emphasis on IV / payoff environment, not free-form direction routing
- keep the proven entry-selection trunk and fixed-quantile gate as the stable scaffold

## Commands

Dataset export:

```bash
.venv/bin/python -m v3.layer2.export_dataset
```

Current best train run:

```bash
.venv/bin/python -m v3.layer2.train_entry_side \
  --run-dir v3/artifacts/layer2_entry_side_fixedq_60_10 \
  --entry-target entry_value_rank \
  --side-target time_stop_margin_raw \
  --direction-mode teacher_if_triggered_else_put \
  --calibration-mode fixed_quantiles \
  --entry-quantile 0.60 \
  --side-quantile 0.10 \
  --score-mode product
```

Replay against post-A1 teacher baseline:

```bash
.venv/bin/python -m v3.layer2.replay \
  --run-dir v3/artifacts/layer2_entry_side_fixedq_60_10
```

Neural GPU-prep command:

```bash
.venv/bin/python -m v3.layer2.train_neural \
  --run-dir v3/artifacts/layer2_neural_gpu_fixedq \
  --entry-target entry_value_rank \
  --side-target time_stop_margin_raw \
  --direction-mode teacher_if_triggered_else_put \
  --calibration-mode fixed_quantiles \
  --entry-quantile 0.60 \
  --side-quantile 0.10 \
  --score-mode product \
  --device cuda
```

Route-aware fallback reproduction:

```bash
.venv/bin/python -m v3.layer2.train_neural \
  --run-dir v3/artifacts/layer2_route_fallback_q50 \
  --policy-mode route_aware_fallback \
  --entry-target entry_value_rank \
  --side-target time_stop_margin_raw \
  --direction-mode teacher_if_triggered_else_model \
  --calibration-mode fixed_quantiles \
  --entry-quantile 0.60 \
  --side-quantile 0.50 \
  --score-mode product \
  --device cpu

.venv/bin/python -m v3.layer2.replay \
  --run-dir v3/artifacts/layer2_route_fallback_q50

.venv/bin/python -m v3.analysis.layer2_fallback_route_diagnostic \
  --run-dir v3/artifacts/layer2_route_fallback_q50
```

## Immediate Next Work

### 1. Keep the right baselines separate

- Keep the fixed-quantile hybrid tree branch as the stable CPU reference.
- Keep the shared-encoder `--detach-side` run as the max-PF neural research baseline.
- Do not confuse “best aggregate replay artifact” with “cleared for promotion.”

### 2. Do not launch GPU from the route-aware branch

- The route-aware fallback experiment was fully implemented and failed:
  - all four fallback quantile runs tied
  - fallback threshold collapsed to `0.0`
  - the representative run selected `0` fallback trades
  - PF and DD both regressed versus the detach-side baseline
- So GPU is blocked by research, not by code readiness.

### 3. Next research target: fallback-only modeling

- If Layer-2 is revisited before exit modeling, focus the next cycle on the no-teacher subset only.
- Good candidate questions:
  - when should a no-teacher **put** be suppressed because the payoff regime is too weak?
  - do `atm_iv`, `iv_percentile`, first-15 range, or similar context explain the fold-0-vs-fold-3 payoff gap better than direction models?
  - can a fallback regime gate beat `teacher_if_triggered_else_put` without damaging the strong sell-off folds?

### 4. Keep exit modeling deferred

- Current replay still uses time-stop.
- Exit modeling remains a later Layer-2/3 addition after entry+direction are more stable.
- Do not let fixed `-35/+60` stop-target logic back into Layer 0.

## Files To Read First

1. [v3/reference/orc_direction_fix_2026_04_20.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/orc_direction_fix_2026_04_20.md)
2. [v3/reference/late_session_tournament_2026_04_20.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/late_session_tournament_2026_04_20.md)
3. [v3/reference/w2a_handoff_2026_04_21.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/w2a_handoff_2026_04_21.md)
4. [v3/reference/layer2_entry_side_2026_04_21.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/layer2_entry_side_2026_04_21.md)
5. [v3/reference/layer2_diagnostic_full_2026_04_21.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/layer2_diagnostic_full_2026_04_21.md)
6. [v3/reference/layer2_route_aware_fallback_2026_04_21.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/layer2_route_aware_fallback_2026_04_21.md)
7. [v3/reference/layer2_fallback_only_probe_2026_04_21.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/layer2_fallback_only_probe_2026_04_21.md)
8. [v3/layer2/README.md](/Users/gduby/Documents/autoresearch-trading/v3/layer2/README.md)

## Known Gotchas

- `v3/artifacts/layer2_dataset.pkl` is a derived artifact, not source of truth. Rebuild it if the logger/oracle/export logic changes.
- The current positive Layer-2 result is still **hybrid**. The learned side model alone has not beaten the teacher baseline.
- The detach-side neural artifact is the strongest replay result, but the route-aware follow-up showed the fallback problem is still unresolved.
- The fallback-only probe showed the unresolved piece is likely fallback regime gating, not fallback direction choice.
- The key improvement after the initial hybrid branch was **stable calibration**, not a new feature set or a more complex model.
- `replay.py --latest-only` is now available for one-fold smoke validation of neural runs.
- `v2` remains useful for sidecars and shared market-structure features, but it is no longer the mainline training loop.
