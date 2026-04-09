# How Training Works

This is the current training and experiment workflow for the repaired v2 harness.

## Two Loops

There are still two loops, but the outer loop is now fully walk-forward.

```text
OUTER LOOP
    hypothesis
    edit train.py and/or core/policy.py
    commit
    deploy.sh run_one exp_NNN
    read walk-forward score
    keep or revert

INNER LOOP
    for each fold:
        train from scratch
        pick best epoch by fold validation loss
        replay on that fold's held-out test days
```

## What `run_one` Actually Does

`./v2/ops/deploy.sh run_one exp_NNN`:

1. uploads `v2/train.py` and `v2/core/policy.py`
2. ensures the remote `v2/data.pt` matches the selected local dataset
3. runs `python3 -m v2.ops.run_experiment_wf --id exp_NNN` on the GPU
4. downloads the resulting checkpoint to `v2/model_candidate.pt`
5. downloads the matching artifact bundle to `v2/artifacts/exp_NNN/`

The remote runner is `v2/ops/run_experiment_wf.py`, not the older single-split runner.

## Walk-Forward Geometry

The current experiment runner uses:

- 5 folds
- 60-day test windows
- 40-day validation windows taken from the tail of each training range
- 20 final shadow days held out from walk-forward

The aggregate experiment score is the mean of the 5 fold scores.

## From-Scratch Training

Every fold trains from random initialization.

- `deploy.sh start` clears the remote `v2/model.pt`
- `train.py` always constructs a fresh `TradingModel()`
- there is no warm-start path in the current honest loop

## Current Model

The current model in `v2/train.py` is a small transformer.

Input:

- shape `(batch, 30, 47)`

Outputs:

- `call_pnl`
- `put_pnl`
- `risk` head with 3 values
- replay-compatibility heads derived from those outputs:
  - gate
  - direction
  - strike logits
  - confidence

Important nuance:

- the strike head exists for replay compatibility
- current training does not directly supervise strike choice
- current models are therefore still heavily ATM-biased
- labels include `hold=390`, but the current decoded policy range is capped at `250`, so model-emitted holds are currently limited to that range

## Current Loss

The active loss is P&L regression, not a direct TradeIntent loss.

Components:

- call P&L regression
- put P&L regression
- risk regression

Current details:

- Huber loss on both directional P&Ls
- optimistic errors are penalized more heavily than pessimistic errors
- put optimism is penalized more than call optimism
- larger-magnitude P&L bars are up-weighted
- risk loss is weighted by `RISK_W`

The current risk target also includes a direction-dependent shortening of hold targets for put labels.

## Epoch Selection

Within each fold:

- training and validation loaders are built from fold-specific masks
- the checkpoint with the lowest validation loss is saved
- that fold checkpoint is replayed on the fold test days

After all folds:

- the last fold model is copied to `v2/model.pt`

## What Gets Saved

Each checkpoint stores:

- `model_state_dict`
- best epoch
- validation loss
- gate accuracy
- direction accuracy
- selected hyperparameters
- score-config fingerprint
- dataset fingerprint

Each experiment artifact bundle stores:

- checkpoint copy
- policy snapshot
- train.py snapshot
- policy.py snapshot
- manifest with score, fingerprints, and walk-forward summary

## Keep / Revert

After a run finishes:

- `python v2/ops/model_manage.py keep`
  - promotes `model_candidate.pt`
  - updates `model.pt` and `model_best.pt`
  - marks the artifact `promoted=true`

- `python v2/ops/model_manage.py revert`
  - deletes `model_candidate.pt`
  - leaves the promoted model unchanged
  - marks the artifact `promoted=false`

## Session Limits

The active operator protocol uses:

- 20 experiments max
- 10 hours max
- 6 consecutive no-improve reverts
- 4 hour plateau
- 3 consecutive crashes

## What Training Is Not Doing Yet

- It is not learning a rich strike-selection objective.
- It is not producing a live-trading service artifact.
- It is not using paper-trading feedback.
- It is not using warm-start checkpoints.
