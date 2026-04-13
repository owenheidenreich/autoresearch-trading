# Current v2 State

Last refreshed: 2026-04-13 (Wave 1 overhaul — scoring reset)

## Mission And Phase

- Mission: build a trustworthy exact-chain research system for SPX 0DTE long-options training and replay
- Current phase: **Wave 1 overhaul complete; fresh slate for model improvements**
- Live trading: not implemented and not part of the live `v2/` surface
- The current mission is not live trading. It is a trustworthy exact-chain research system.

## Snapshot

- Active manifest: `v2/data.pt` (rebuilt 2026-04-13 with enriched features)
- Dataset version: `v4_exact_chain`
- Context features: 49 (was 47) — added aggregate_charm, vwap_slope
- Contract features: 19 (was 15) — added vega, charm, mid_chg_5, mid_chg_10
- Per-day sidecars: `v2/data_sidecars/*.pt` (rebuilt)
- Unique days: 986
- Scoring: v3.0 composite PF/sortino (DD gate 25%, penalty-free ≤12%)
- **results.tsv: RESET** — old scores (exp_074–exp_146) not comparable under new scoring
- Next experiment: `exp_147` — first experiment under new feature set + scoring
- Archived non-live surfaces now live under `archive/v2_historical/`

## End-To-End Flow

```text
raw SPX/SPY/VIX pickles + full-chain SPXW pickles
    -> build_v2_dataset.py
    -> v2/data.pt + v2/data_sidecars/
    -> train.py + core/policy.py
    -> ops/pre_run_gate.py
    -> ops/run_experiment_wf.py
    -> v2/models/model_candidate.pt + v2/artifacts/exp_NNN/
    -> model_manage.py keep/revert
    -> promoted artifact + v2/models/model_best.pt
    -> replay.py / analysis / plots
```

## What The Repo Currently Does

### Data

- The manifest stores 47 normalized market-context features in `X`, raw replay features in `X_sim`, plus bar metadata and split masks.
- Exact contracts do not live inside the manifest tensor payload.
- Each day sidecar stores exact contract identities, executable snapshots, and per-contract forward P&L labels under the fixed policy.
- Hard or incomplete market days are retained. Missing forward paths are flagged at the contract/bar level instead of dropping whole days.

### Training

- Training is from scratch every experiment.
- The default mutable research surface remains:
  - `v2/train.py`
  - `v2/core/policy.py`
- The approved expanded mutable surface for the current Kronos-inspired block is:
  - `v2/train.py`
  - `v2/core/metrics.py`
  - `v2/replay.py`
  - `v2/core/data_integrity.py`
  - `v2/ops/pre_run_gate.py`
  - live docs that must stay in sync with the active protocol
- The current live baseline scores:
  - `NO_TRADE`
  - each executable contract on the current bar
- The live loss stack is balanced gate BCE plus soft KL selection only.
- The working config uses `SOFT_TEMP=0.10` and `NOISE_MARGIN=0.01`.
- The current official policy window is morning-only (`bar 60` through `105`).
- Direction mix is now diagnostic output, not a hard score gate.
- Risk is policy-driven, not learned, in the frozen v4 harness.

### Replay

- Replay uses exact contract identity from sidecars.
- No ATM/OTM ladder fallback remains in the main scoring path.
- Entry fills are next-bar.
- Trade simulation uses the same stop / target / trailing / hold rules as policy and still charges spread plus commission.

### Scope Boundary

- The live `v2/` path is now only data, training, replay, experiment management, and the docs needed to operate that system.
- Paper/live trading stubs, old dataset builders, and one-off repair diagnostics were archived out of `v2/` to reduce context drift.

### Harness Eval Suite

- `core_regression` protects non-negotiable invariants.
- `optimization` cases are used during harness repair.
- `holdout` cases verify that harness fixes generalize.

## Operational Workflow

### Before A Training Run

1. `python3 -m v2.ops.pre_run_gate --data v2/data.pt`
2. `./v2/ops/deploy.sh boot`
3. `./v2/ops/deploy.sh start`

### For Each Experiment

1. Form one hypothesis from trace or audit evidence.
2. Edit the approved mutable surface for that hypothesis.
3. Commit.
4. Screen first: `./v2/ops/deploy.sh run_screen exp_NNN`
5. If screening passes or justifies a same-family follow-up: `./v2/ops/deploy.sh run_one exp_NNN`
6. Run the side-bias audit alongside the decision trace for meaningful candidates.
7. Official promotion still requires score, baselines, and hard gates.
8. KEEP or REVERT, then update `v2/lab_notebook.md` and required plots.

See `v2/program.md` for the definitive workflow and decision rules.
