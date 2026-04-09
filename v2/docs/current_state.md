# Current v2 State

This is the detailed source-of-truth overview for how the project works right now.

Last refreshed: 2026-04-09

## Snapshot

- Project mode: research harness only
- Live trading: not implemented
- Active dataset: `v2/data.pt`
- Dataset backup copy: `v2/data_harness_repair.pt`
- Dataset version: `v2_harness_repair`
- Dataset fingerprint: `03566aeb8adf1040`
- Current model status: pre-repair models are invalidated; the first repaired-era walk-forward run establishes the new honest baseline

## End-To-End Flow

```text
raw SPX/SPY/VIX pickles + wide-grid option pickles
    -> build_v2_dataset.py
    -> relabel_tier3.py
    -> v2/data.pt
    -> train.py + core/policy.py
    -> ops/run_experiment_wf.py
    -> v2/model_candidate.pt + v2/artifacts/exp_NNN/
    -> model_manage.py keep/revert
    -> promoted artifact + model_best.pt
    -> replay.py / analysis / plots
```

## What The Repo Currently Does

### Data

- The canonical dataset is a repaired, content-fingerprinted PyTorch artifact.
- Feature count is 47.
- Features are computed from raw SPX, SPY, VIX, and wide-grid SPXW option caches.
- Price tensors for replay are keyed to the dynamic nearest ATM per bar, not the session-open ATM.
- POC and value area are incremental and only use bars seen so far within the day.
- Normalization is 60-day rolling z-score with bounded and categorical features excluded.

### Labels

- The active dataset uses `dual_direction_pnl_tier3`.
- Labels store both call and put forward P&L for each supervised bar.
- Risk labels are variable per bar:
  - stop
  - target
  - max hold
- Tier 3 search grid:
  - stops: `[0.15, 0.2, 0.25, 0.3, 0.4, 0.5]`
  - targets: `[0.2, 0.3, 0.5, 0.8, 1.2]`
  - holds: `[30, 60, 120, 240, 390]`
- `label_trade=True` when best directional P&L exceeds `DEFAULT_POLICY.label_gate_min_pnl`, currently `0.04`.
- The label grid includes `hold=390`, but the current policy decode range is `10-250`, so the current model cannot emit a full 390-bar hold directly.

### Training

- Training is from scratch every experiment.
- The mutable research surface is:
  - `v2/train.py`
  - `v2/core/policy.py`
- The current model is a small transformer that predicts:
  - `call_pnl`
  - `put_pnl`
  - `risk` = stop, target, max-hold proxy
- Gate and direction are derived from the predicted P&Ls at inference time.
- A strike head still exists for replay compatibility, but current training does not directly supervise strike selection.
- In practice the current model remains strongly ATM-biased.

### Walk-Forward Evaluation

- Experiments run through `v2/ops/run_experiment_wf.py`.
- There are 5 folds.
- Each fold uses:
  - expanding train window
  - last 40 training days as validation
  - 60 future days as test
- Aggregate score is the mean of the 5 fold scores.
- The last fold's checkpoint is copied to `v2/model.pt`.

### Replay

- Replay uses `v2/replay.py`.
- Entry fills are at the next bar, not the decision bar.
- Trade simulation uses adaptive spread plus commission.
- New entries stop for the day after the policy daily loss cap is hit.
- Replay also stops taking new trades if equity is wiped out.
- Default replay loads the best promoted artifact compatible with the current dataset fingerprint.
- If there is no compatible promoted artifact yet, replay falls back to the raw checkpoint on disk.

### Artifact Lineage

- Every completed run saves an artifact bundle.
- `run_one` downloads the bundle and the checkpoint to local disk.
- `model_manage.py keep` marks the matching artifact `promoted=true`.
- `model_manage.py revert` marks the matching artifact `promoted=false`.
- Legacy artifacts without promotion metadata are treated as ineligible by default replay.

## Canonical Dataset Facts

```text
Path:              v2/data.pt
Version:           v2_harness_repair
Fingerprint:       03566aeb8adf1040
Features:          47
Signal bars:       236,641
Trade bars:        157,232
Trade rate:        66.44%
Trade window:      bar 30-270
ATM source:        dynamic_nearest_per_bar
POC/VA source:     incremental_bars_seen_so_far
```

The current `v2/data_harness_repair.pt` is a backup copy of the same artifact, not a different staging dataset.

## Operational Workflow

### Before A Training Run

1. Verify the canonical dataset:
   - `python -m v2.analysis.contract_drift_audit --data v2/data.pt`
2. Boot the GPU:
   - `./v2/ops/deploy.sh boot`
3. Upload code and data:
   - `./v2/ops/deploy.sh start`

### For Each Experiment

1. Form one hypothesis.
2. Edit `v2/train.py` and/or `v2/core/policy.py`.
3. `python -m py_compile` the changed Python files.
4. Commit.
5. Run:
   - `./v2/ops/deploy.sh run_one exp_NNN`
6. Decide:
   - KEEP: `python v2/ops/model_manage.py keep`
   - REVERT: `git checkout HEAD~1 -- v2/train.py v2/core/policy.py` then `python v2/ops/model_manage.py revert`
7. Update logs and plots.

### After Every Experiment

- `python -m v2.plot_trades --model v2/model.pt`
- `python v2/plot_progress.py`
- `python -m v2.analysis.analyze_losses`

## What Is Not True Anymore

These older descriptions are wrong for the active harness:

- 39-feature input contract
- per-day z-score normalization
- session-open ATM replay tensors
- full-day POC / value-area features
- warm-start training
- unrestricted legacy-artifact replay loading
- "live system already runs in paper"

## Which Docs To Use

### Current Truth

- [program.md](../program.md)
- [data_contract.md](data_contract.md)
- [feature_schema.md](feature_schema.md)
- [labeling.md](labeling.md)
- [evaluator.md](evaluator.md)
- [how_training_works.md](how_training_works.md)

### Historical Or Roadmap Context

- `v2/docs/audit/` -- historical audit findings that motivated the repair
- `v2/docs/domain/` -- domain knowledge and research notes
- `v2/docs/execution.md` -- future live-execution roadmap, not current runtime
- `v2/docs/goal.md` -- roadmap and mission, not the operator contract
