# Handoff: Current v2 State

Read this file, then [program.md](program.md), then [current_state.md](docs/current_state.md).

## What Changed

The 2026-04-08 / 2026-04-09 harness repair is no longer a staging plan. It is the active system.

- Contract drift was repaired.
- POC / value-area look-ahead was removed.
- Tier 3 labels now include `hold=390`.
- Replay, dataset metadata, and artifact lineage now key off the repaired dataset.
- The canonical dataset is now `v2/data.pt`.

## Canonical State

```text
Dataset path:        v2/data.pt
Backup copy:         v2/data_harness_repair.pt
Dataset version:     v2_harness_repair
Dataset fingerprint: 03566aeb8adf1040
Features:            47
Signal bars:         236,641
Trade bars:          157,232
Trade window:        bar 30-270
Label scheme:        dual_direction_pnl_tier3
Label grid:          stops=[0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
                     targets=[0.2, 0.3, 0.5, 0.8, 1.2]
                     holds=[30, 60, 120, 240, 390]
ATM source:          dynamic_nearest_per_bar
POC/VA source:       incremental_bars_seen_so_far
```

`v2/data_harness_repair.pt` is currently just a backup copy of the same repaired dataset, not a different staging artifact.

## What To Trust

- Trust `v2/data.pt`, not the old audit-era assumptions.
- Trust promoted artifacts only.
- Trust post-repair replay scores only.
- Trust `run_experiment_wf.py` as the canonical training runner.

## What Not To Trust

- Any pre-repair score as a current baseline.
- Any legacy artifact that was trained on the old dataset fingerprint.
- Any doc that still says:
  - 39 features
  - per-day z-score normalization
  - session-open ATM arrays
  - full-day POC / VA
  - warm-start training
  - live execution is already operating

## Current Research Position

- The harness is ready for honest experimentation.
- The current local model is not a meaningful repaired-era baseline.
- The next important event is the first honest walk-forward GPU run on the repaired harness.
- That run establishes the new baseline for all future keep/revert decisions.

## Practical Operator Notes

- `./v2/ops/deploy.sh start` uploads the canonical dataset and clears the remote model path.
- `./v2/ops/deploy.sh run_one exp_NNN` downloads:
  - `v2/model_candidate.pt`
  - `v2/artifacts/exp_NNN/`
- Use:
  - `python v2/ops/model_manage.py keep`
  - `python v2/ops/model_manage.py revert`

## Live Status

Live and paper trading code paths are still stubs. This repo is currently a data, training, replay, and experiment-management system.
