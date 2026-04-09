# Fresh Session Handoff

Use this document to bootstrap a brand-new session quickly and safely.

## Read Order

1. Read this file.
2. Read `v2/program.md`.
3. Read `v2/COMMANDS.md`.
4. Read `v2/docs/current_state.md`.

## Current Branch And Commits

```text
Branch: autoresearch/v2-gate-fix
HEAD:   2b5b485  Refresh v2 docs for repaired harness
Prev:   d0b7924  Finalize harness repair baseline
```

`d0b7924` is the code baseline for the repaired harness.

`2b5b485` is the docs re-freeze that updates the operator story to match the repaired codebase.

## Canonical Dataset

```text
Active path:        v2/data.pt
Backup copy:        v2/data_harness_repair.pt
Version:            v2_harness_repair
Fingerprint:        03566aeb8adf1040
Features:           47
Signal bars:        236,641
Trade bars:         157,232
Label scheme:       dual_direction_pnl_tier3
Trade window:       bar 30-270
ATM source:         dynamic_nearest_per_bar
POC/VA source:      incremental_bars_seen_so_far
```

Important: `v2/data_harness_repair.pt` is currently just a backup copy of the same repaired dataset, not a separate staging artifact.

## What Was Recently Repaired

The harness repair invalidated the old pre-repair model lineage.

Key fixes that are now active:

- dynamic nearest-ATM replay tensors instead of session-open ATM drift
- incremental POC / value-area features instead of full-day look-ahead
- Tier 3 labels include `hold=390`
- truthful dataset metadata and content-based fingerprinting
- replay enforces artifact/data fingerprint compatibility
- only promoted artifacts are eligible for default replay
- `deploy.sh start` no longer pretends to warm-start training

## Current Truth About Readiness

The project is ready for:

- honest local replay on `v2/data.pt`
- the first honest GPU walk-forward training run on the repaired harness
- normal keep/revert workflow after that first repaired-era run

The project is not ready for:

- live trading
- paper trading
- treating old scores as current baselines

## Current Model Status

The currently available local model is not a valid repaired-era baseline.

What matters:

- pre-repair artifact scores are historical only
- default replay now rejects old dataset fingerprints
- if no compatible promoted artifact exists yet, replay falls back to the raw checkpoint on disk
- the first repaired-era `run_one` result becomes the new honest baseline

## Important Nuances

These are easy for a fresh session to miss:

1. Training is from scratch every experiment.
2. The canonical experiment runner is `v2/ops/run_experiment_wf.py` via `./v2/ops/deploy.sh run_one exp_NNN`.
3. The mutable research surface is still only:
   - `v2/train.py`
   - `v2/core/policy.py`
4. The dataset label grid includes `hold=390`, but the current policy decode range is capped at `250`, so the model cannot emit a full 390-bar hold directly right now.
5. Replay can score coarse ATM/OTM buckets, but the current model is still effectively ATM-biased because strike selection is not strongly supervised.

## Source-Of-Truth Docs

These docs now reflect the repaired system:

- `v2/program.md`
- `v2/COMMANDS.md`
- `v2/HANDOFF.md`
- `v2/docs/current_state.md`
- `v2/docs/data_contract.md`
- `v2/docs/feature_schema.md`
- `v2/docs/labeling.md`
- `v2/docs/evaluator.md`
- `v2/docs/how_training_works.md`

Treat `v2/docs/audit/` as historical analysis, not as the current operator contract.

## Worktree State

There are still unrelated local changes in the worktree that were intentionally left alone:

```text
M  v2/.baseline_cache.json
M  v2/ops/inner_loop.py
?? AGENTS.md
?? archive/runs/run-2026-03-22-215002/artifacts 2/
?? archive/v1/training/live/entitlements 2.py
?? v2/analysis/pre_repair_snapshot.md
```

Do not revert those blindly unless the user asks.

## Recommended Next Move

If the goal is to begin honest experimentation, the next safe sequence is:

1. Form the first repaired-era hypothesis.
2. Edit only `v2/train.py` and/or `v2/core/policy.py`.
3. Run `python -m py_compile` on changed files.
4. Commit.
5. Boot and start the GPU:
   - `./v2/ops/deploy.sh boot`
   - `./v2/ops/deploy.sh start`
6. Run the first repaired-era experiment:
   - `./v2/ops/deploy.sh run_one exp_068`
7. Read the aggregate walk-forward score and per-fold scores.
8. Keep or revert:
   - `python v2/ops/model_manage.py keep`
   - or `python v2/ops/model_manage.py revert`

If `exp_068` is already taken by the time a fresh session runs, use the next unused `exp_NNN`.

## Local Validation Commands

These are the most useful local checks before GPU spend:

```bash
python -m v2.analysis.contract_drift_audit --data v2/data.pt
python -m v2.replay --data v2/data.pt --mask promote
python -m v2.replay --model v2/model.pt --data v2/data.pt --mask promote
```

## Short Prompt For A Fresh Session

If you want to hand this to a new agent, this prompt is enough:

```text
Read v2/FRESH_SESSION_HANDOFF.md, then v2/program.md, then v2/COMMANDS.md, then v2/docs/current_state.md. Use those as source of truth. We are on branch autoresearch/v2-gate-fix at commit 2b5b485 with repaired canonical dataset v2/data.pt fingerprint 03566aeb8adf1040. Old model scores are invalidated. Prepare and run the first honest repaired-era experiment unless I redirect you.
```
