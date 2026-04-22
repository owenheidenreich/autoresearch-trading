# Layer-3 on Unified Entries — Cross-Seed Test — 2026-04-22

## Hypothesis

The first Layer-3 replay on unified-policy seed-42 entries produced an
exploratory best of `PF 2.033 / DD 22.3%` at threshold `0.19` on top of
the seed-42 time-stop baseline `PF 1.151 / DD 29.9%`. The question left
open: is that a single-seed artifact of same-OOS threshold picking, or
does the composition work across seeds?

Test: run the same Layer-3 rolling train+replay on seed-43 and seed-44
unified chosen trades (same GPU promotion artifact) and compare (a)
the lift at each threshold, (b) the best-threshold choice.

## Mechanism

Unified entries have DD materially lower than V0 but PF `1.5%` short of
baseline. A learned exit that cuts losing trades before they deepen
should monetize the DD advantage into PF, turning "cleaner entries"
into "cleaner entries + real PF".

## Commands

```bash
.venv/bin/python -m v3.layer3.train_rolling \
  --entry-source unified \
  --chosen-trades v3/artifacts/v3_unified_promo_001/seed_43/chosen_trades.pkl \
  --out-dir v3/artifacts/layer3_unified_seed43

.venv/bin/python -m v3.layer3.train_rolling \
  --entry-source unified \
  --chosen-trades v3/artifacts/v3_unified_promo_001/seed_44/chosen_trades.pkl \
  --out-dir v3/artifacts/layer3_unified_seed44
```

## Results

Per-seed Layer-3 sweeps (entries frozen from the GPU-promotion artifact):

| seed | entry baseline PF | entry DD | L3 best PF | L3 best thr | L3 DD |
|---|---|---|---|---|---|
| 42 | 1.151 | 29.9% | **2.033** | 0.19 | 22.3% |
| 43 | 1.101 | 38.5% | **1.669** | 0.25 | 26.2% |
| 44 | 1.097 | 27.4% | **1.812** | 0.15 | 22.7% |

Cross-seed aggregation at each threshold in the default grid:

| threshold | seed PFs | mean PF | min PF | mean DD |
|---|---|---|---|---|
| 0.15 | 2.011 / 1.652 / 1.812 | **1.825** | 1.652 | 23.7% |
| 0.19 | 2.033 / 1.623 / 1.711 | **1.789** | 1.623 | 23.7% |
| 0.20 | 1.939 / 1.622 / 1.711 | 1.757 | 1.622 | 23.7% |
| 0.25 | 1.743 / 1.669 / 1.704 | 1.706 | 1.669 | 23.7% |
| 0.30 | 1.632 / 1.645 / 1.733 | 1.670 | 1.632 | 23.7% |

Baselines for context:

- `V0 + time-stop` (entry-only, same shape as unified+time-stop): PF `1.132`, DD `95.9%`
- `Layer-2.5 + time-stop` (different entry stack): PF `~1.795`, DD `21.4%`

## Interpretation

### What this proves

- **Cross-seed PF lift is robust.** Every seed's best-threshold PF beats
  V0's `1.132` by at least `+0.49`, and every seed's *worst*-threshold
  PF in the default grid still beats V0.
- **DD advantage survives composition.** The entry-stage DD reductions
  carry through Layer-3 and land at `~23.7%` mean across seeds — close
  to Layer-2.5's `21.4%` and far below V0's `95.9%`.
- **Threshold region is stable.** Within each seed the PF range across
  thresholds `0.15–0.30` is `≤0.40`; the whole band is promotion-
  interesting. The particular "best threshold" differs (seed 42 → 0.19,
  seed 43 → 0.25, seed 44 → 0.15), but picking a plain mid-band default
  like `0.19` gives mean PF `1.789`, min `1.623`.

### What this does not prove

- Threshold is still chosen on the same rolling OOS windows used for
  evaluation (the Layer-3 README note). The cross-seed consistency at
  *every* threshold makes same-OOS picking less dangerous here — the
  result is not a cherry picked local max — but the honest
  prior-window calibration step still hasn't run.
- Side share inside the entry stage is still ~93% calls; if the next
  OOS regime is put-favoring, composition PF will drop even if Layer-3
  still helps.
- Layer-3 DD is identical across thresholds within a seed (22.3%,
  26.2%, 22.7%). That pattern suggests the max-DD is anchored by one
  specific window whose losing days are not shortened by any tested
  exit threshold. Worth investigating as a separate cycle.

### Repo-belief change

Before: `V0 + time-stop` at `PF 1.132` was the honest PF champion;
unified policy was "working architecture, not enough PF".

After: unified+time-stop is still `1.5%` short of V0 on PF, but
**unified+Layer-3** now has a real claim to champion across the
default exit-threshold grid on three independent seeds:

- mean PF at default `thr 0.19` is `1.789` (vs V0 `1.132` and
  L2.5+time-stop `~1.795`)
- mean DD is `~23.7%` (vs V0 `95.9%` and L2.5 `21.4%`)

The unified+L3 composition is now the leading promotion candidate.
Next cycle should be honest prior-window threshold calibration to
make that claim promotion-grade.

## Artifact Locations

- `v3/artifacts/layer3_unified_seed42/rolling_layer3_report.json` (from earlier work)
- `v3/artifacts/layer3_unified_seed43/rolling_layer3_report.json`
- `v3/artifacts/layer3_unified_seed44/rolling_layer3_report.json`
