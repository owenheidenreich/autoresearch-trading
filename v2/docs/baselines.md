# v2 Baselines

This file documents the baselines that replay currently computes and compares against.

## Important Clarification

The repo does not currently enforce a heavyweight pre-GPU local gate suite.
There is no root `tests/` gate that blocks `run_one`.

What is real today:

- local replay and audit smoke checks are good practice
- baseline comparison happens during replay and walk-forward evaluation
- `run_one` itself does not run an external preflight test battery beyond the GPU-side preflight script

## Current Baselines

Replay compares the model against four baselines.

### 1. Random

Implemented in `compute_baseline_random()`:

- 2% chance to enter on each eligible bar
- random option price array from the stored replay universe
- random direction implied by the selected array
- fixed stop `30%`
- fixed target `50%`
- fixed hold `120`
- averaged over 5 seeds

### 2. ATM-Always

Implemented in `compute_baseline_atm_always()`:

- one ATM call
- entered at bar `30`
- one trade per eligible day
- fixed stop `30%`
- fixed target `50%`
- fixed hold `120`

### 3. Simple-Rules

Implemented in `compute_baseline_simple_rules()`:

- 5-bar momentum signal from `ret_6`
- call if momentum `> 0.005`
- put if momentum `< -0.005`
- no trade otherwise
- entry window bar `30` through `299`
- 10-bar cooldown
- fixed stop `25%`
- fixed target `40%`
- fixed hold `60`

### 4. ATM-Trailing

Implemented in `compute_baseline_atm_trailing()`:

- one ATM call
- entered at bar `30`
- `TRAILING` exit policy
- stop, target, and hold are set to the midpoint of the active policy ranges

This is the "honest exit" baseline because it uses the model-era risk regime without using the model itself.

## Baseline Cache

Replay caches baseline outputs in:

- `v2/.baseline_cache.json`

The cache key includes:

- dataset fingerprint
- mask key
- policy fingerprint
- optional `max_days`

If the dataset fingerprint changes, the cache invalidates automatically.

## What Matters Operationally

For keep/revert, the experiment must beat all four baselines under the current repaired harness.

The most useful local smoke checks before GPU spend are:

1. `python -m py_compile` on changed Python files
2. `python -m v2.analysis.contract_drift_audit --data v2/data.pt`
3. `python -m v2.replay --data v2/data.pt --mask promote`

Those are current reality. Treat any older doc that claims a larger enforced local gate suite as historical planning, not current behavior.
