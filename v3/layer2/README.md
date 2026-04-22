# v3 Layer-2

This package is the current `v3` mainline.

It does four things:

1. export one supervised row per eligible `v3` bar
2. train tabular or neural entry/side models on walk-forward folds
3. replay a one-trade-per-day policy against the post-A1 teacher baseline
4. support diagnostic follow-ups on direction, fallback routing, and calibration

## Current State

There are now three distinct Layer-2 reference points:

1. Stable CPU reference:
   - [v3/artifacts/layer2_entry_side_fixedq_60_10](/Users/gduby/Documents/autoresearch-trading/v3/artifacts/layer2_entry_side_fixedq_60_10)
   - `287` trades, `PF 1.122`, `DD 56.2%`, `0.957` trades/day
   - this is the last branch that clearly beat the post-A1 teacher baseline without extra qualification caveats
2. Max-PF neural research baseline:
   - [v3/artifacts/layer2_shared_enc_fixedq_detach](/Users/gduby/Documents/autoresearch-trading/v3/artifacts/layer2_shared_enc_fixedq_detach)
   - `275` trades, `PF 1.455`, `DD 36.9%`, `0.917` trades/day
   - this is the strongest aggregate artifact, but it is not cleared for GPU or paper-trading promotion because the later reality checks showed the edge is dominated by entry selection plus teacher-conditioned fallback rather than a trustworthy learned direction head
3. Route-aware fallback follow-up:
   - [v3/artifacts/layer2_route_fallback_q50](/Users/gduby/Documents/autoresearch-trading/v3/artifacts/layer2_route_fallback_q50)
   - representative result for `q50/q60/q70/q80`: `270` trades, `PF 1.131`, `DD 93.1%`, `0.900` trades/day
   - branch failed; all four fallback-quantile runs tied because the effective fallback threshold collapsed to `0.0`, and the chosen set contained `270` teacher trades and `0` fallback trades
4. Fallback-only probe on top of the detach-side winner:
   - [v3/artifacts/layer2_fallback_only_probe/fallback_only_probe.json](/Users/gduby/Documents/autoresearch-trading/v3/artifacts/layer2_fallback_only_probe/fallback_only_probe.json)
   - `teacher+put_or_flat_model`: `PF 1.340`, `DD 58.8%`, `0.613` trades/day
   - `teacher+fallback_model`: `PF 1.159`, `DD 96.6%`, `0.697` trades/day
   - both were worse than blunt `teacher_if_triggered_else_put`, so the next bottleneck is more likely fallback regime gating than fallback action choice

The current honest position is:

- the Layer-2 edge is real
- the best max-PF artifact is still `detach-side`
- the route-aware fallback hypothesis was implemented and falsified
- the fallback-only probe also failed to beat blunt put fallback
- GPU remains blocked until a fallback-specific hypothesis clears CPU gates

## Commands

Export:

```bash
.venv/bin/python -m v3.layer2.export_dataset
```

Train the stable CPU reference:

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

Replay:

```bash
.venv/bin/python -m v3.layer2.replay \
  --run-dir v3/artifacts/layer2_entry_side_fixedq_60_10
```

Train the current max-PF neural baseline:

```bash
.venv/bin/python -m v3.layer2.train_neural \
  --run-dir v3/artifacts/layer2_shared_enc_fixedq_detach \
  --entry-target entry_value_rank \
  --side-target time_stop_margin_raw \
  --direction-mode teacher_if_triggered_else_put \
  --calibration-mode fixed_quantiles \
  --entry-quantile 0.60 \
  --side-quantile 0.10 \
  --score-mode product \
  --detach-side \
  --device cpu

.venv/bin/python -m v3.layer2.replay \
  --run-dir v3/artifacts/layer2_shared_enc_fixedq_detach
```

Train the route-aware fallback follow-up:

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
```

## Stable CPU Reference Settings

- `entry_target = entry_value_rank`
- `side_target = time_stop_margin_raw`
- `direction_mode = teacher_if_triggered_else_put`
- `calibration_mode = fixed_quantiles`
- `entry_quantile = 0.60`
- `side_quantile = 0.10`
- `score_mode = product`

Interpretation:

- Entry is learned from oracle-quality ranking.
- Side magnitude is learned as a confidence / separation signal.
- Direction is structural:
  - teacher direction if a teacher is active on the chosen bar
  - otherwise default to `put`
- Fold-local threshold chasing was too noisy on 40-day validation windows.
- A fixed quantile gate is the better current bias:
  - moderately selective on entry
  - permissive on side confidence
- Within the gated set, ranking by `entry_score * side_conf` is still better than entry-only ranking.

This is the best current approximation of the evidence:

- A1 proved teacher-side structure matters.
- Pure learned-side replay is still weak.
- Late-session abstention bars are where the fallback direction matters.
- The main remaining problem was calibration variance, not missing features.

## Stable CPU Reference Result

Artifact directory:
- [v3/artifacts/layer2_entry_side_fixedq_60_10](/Users/gduby/Documents/autoresearch-trading/v3/artifacts/layer2_entry_side_fixedq_60_10)

Headline:

- Layer-2 hybrid fixed-q: `287` trades, `PF 1.122`, `DD 56.2%`, `0.957` trades/day
- Post-A1 teacher baseline: `298` trades, `PF 0.961`, `DD 65.4%`, `0.993` trades/day

See:
- [audit.json](/Users/gduby/Documents/autoresearch-trading/v3/artifacts/layer2_entry_side_fixedq_60_10/audit.json)
- [replay_report.json](/Users/gduby/Documents/autoresearch-trading/v3/artifacts/layer2_entry_side_fixedq_60_10/replay_report.json)
- [layer2_entry_side_2026_04_21.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/layer2_entry_side_2026_04_21.md)

## Max-PF Neural Baseline

Artifact directory:
- [v3/artifacts/layer2_shared_enc_fixedq_detach](/Users/gduby/Documents/autoresearch-trading/v3/artifacts/layer2_shared_enc_fixedq_detach)

Headline:

- Shared encoder + `--detach-side`: `275` trades, `PF 1.455`, `DD 36.9%`, `0.917` trades/day

Important qualifier:

- This branch is the strongest aggregate replay artifact.
- It is **not** promoted to GPU or paper-trading work because the follow-up diagnostics showed the edge is mostly:
  - entry selection
  - fixed-quantile gating
  - top-1/day ranking
  - teacher direction when present
  - blunt put fallback otherwise
- The learned side head is not yet the deployable core mechanism.

See:
- [layer2_shared_encoder_diagnostics_2026_04_21.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/layer2_shared_encoder_diagnostics_2026_04_21.md)
- [layer2_diagnostic_full_2026_04_21.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/layer2_diagnostic_full_2026_04_21.md)
- [layer2_fallback_only_probe_2026_04_21.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/layer2_fallback_only_probe_2026_04_21.md)

## Route-Aware Fallback Follow-Up

Representative artifact:
- [v3/artifacts/layer2_route_fallback_q50](/Users/gduby/Documents/autoresearch-trading/v3/artifacts/layer2_route_fallback_q50)

Headline:

- `q50/q60/q70/q80` all tied: `270` trades, `PF 1.131`, `DD 93.1%`, `0.900` trades/day

What failed:

- the fallback quantile sweep had no effect because every fold calibrated to `side_threshold = 0.0`
- the chosen set contained `270` teacher trades and `0` fallback trades
- fallback-route controls all collapsed to the same result on the representative run because there were no fallback selections to reroute

Verdict:

- route-aware fallback is a negative result
- do not launch GPU from this branch
- do not widen the neural search from here

See:
- [layer2_route_aware_fallback_2026_04_21.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/layer2_route_aware_fallback_2026_04_21.md)

## Fallback-Only Probe

Artifact:
- [v3/artifacts/layer2_fallback_only_probe/fallback_only_probe.json](/Users/gduby/Documents/autoresearch-trading/v3/artifacts/layer2_fallback_only_probe/fallback_only_probe.json)

Headline:

- `teacher+put_or_flat_model`: `PF 1.340`, `DD 58.8%`, `0.613` trades/day
- `teacher+fallback_model`: `PF 1.159`, `DD 96.6%`, `0.697` trades/day
- baseline `teacher+put`: `PF 1.455`, `DD 36.9%`, `0.917` trades/day

What it means:

- training on non-teacher rows only did not beat the blunt put fallback
- even the narrower `put vs flat` control was still worse than always-put
- this shifts the next hypothesis away from routing and toward fallback regime gating / payoff sufficiency

See:
- [layer2_fallback_only_probe_2026_04_21.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/layer2_fallback_only_probe_2026_04_21.md)

## GPU Prep

GPU is **not** the next step right now.

The correct blocker is not missing CUDA support. It is mechanism clarity.

What is currently true:

- the route-aware fallback branch did not solve the chop/fallback problem
- the fallback-only probe also failed to beat always-put
- the detach-side neural baseline still has a real random-direction hard stop
- a GPU run would currently scale an unresolved policy shape rather than validate a cleared one

So the current GPU status is:

- code path exists
- replay parity path exists
- launch is blocked by research, not by engineering

The next acceptable GPU trigger is:

- a CPU branch that explicitly improves the no-teacher fallback mechanism and clears the existing route/direction diagnostics
- more specifically, the next acceptable branch should target fallback **regime gating**, not broader fallback routing
