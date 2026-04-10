# Current v2 State

Last refreshed: 2026-04-10

## Snapshot

- Project mode: exact-chain harness rebuild
- Live trading: not implemented
- Active manifest: `v2/data.pt`
- Dataset version: `v4_exact_chain`
- Raw option source: full same-day SPXW 0DTE chain cache
- Per-day sidecars: `v2/data_sidecars/*.pt`
- First new baseline experiment: `exp_074`

## End-To-End Flow

```text
raw SPX/SPY/VIX pickles + full-chain SPXW pickles
    -> build_v2_dataset.py
    -> v2/data.pt + v2/data_sidecars/
    -> train.py + core/policy.py
    -> ops/run_experiment_wf.py
    -> v2/model_candidate.pt + v2/artifacts/exp_NNN/
    -> model_manage.py keep/revert
    -> promoted artifact + model_best.pt
    -> replay.py / analysis / plots
```

## What The Repo Currently Does

### Data

- The manifest stores 47 normalized market-context features in `X`, raw replay features in `X_sim`, plus bar metadata and split masks.
- Exact contracts do not live inside the manifest tensor payload.
- Each day sidecar stores:
  - exact contract identities
  - per-contract intraday mid / bid / ask proxy arrays
  - per-bar executable contract snapshots
  - per-contract forward P&L labels under the fixed policy
  - `bar_labelable` so unlabeled bars are not mistaken for `NO_TRADE`
- Hard or incomplete market days are retained. Missing or unusable future paths are flagged at the contract/bar level instead of dropping whole days.

### Training

- Training is from scratch every experiment.
- The mutable research surface remains:
  - `v2/train.py`
  - `v2/core/policy.py`
- The current model scores:
  - `NO_TRADE`
  - each executable contract on the current bar
- Risk is policy-driven, not learned, in the first frozen v4 harness.

### Replay

- Replay uses exact contract identity from sidecars.
- No ATM/OTM ladder fallback remains in the main scoring path.
- Entry fills are next-bar.
- Trade simulation uses the same stop / target / trailing / hold rules as policy and still charges spread plus commission.

### Harness Eval Suite

- `core_regression` protects non-negotiable invariants.
- `optimization` cases are used during harness repair.
- `holdout` cases verify that harness fixes generalize.

## Operational Workflow

### Before A Training Run

1. `python3 -m v2.analysis.harness_eval --data v2/data.pt`
2. `./v2/ops/deploy.sh boot`
3. `./v2/ops/deploy.sh start`

### For Each Experiment

1. Form one hypothesis.
2. Edit `v2/train.py` and/or `v2/core/policy.py`.
3. `python3 -m py_compile` the changed Python files.
4. Commit.
5. `./v2/ops/deploy.sh run_one exp_NNN`
6. KEEP or REVERT.
7. Update logs and plots.
