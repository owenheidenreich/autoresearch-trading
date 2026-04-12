# How Training Works

## Two Loops

```text
OUTER LOOP
    hypothesis (informed by previous trace analysis)
    edit train.py and/or core/policy.py
    commit
    pre-run gate: v2.ops.pre_run_gate (includes data integrity)
    screen: deploy.sh run_screen exp_NNN (1 fold)
    if screening passes:
        official: deploy.sh run_one exp_NNN (5 folds)
        DECISION TRACE: replay --traces (mandatory before keep/revert)
        analyze trace: gate accuracy, selection accuracy, P&L gap
        keep or revert (informed by trace, not score alone)
        form next hypothesis from trace failure modes

INNER LOOP
    for each fold:
        train from scratch (re-read TRAIN_SEED per fold)
        pick best epoch by fold validation loss
        replay on that fold's held-out test days
```

## Current Model

The current model is an exact-chain contract scorer.

Inputs:

- `(batch, 30, 47)` context window
- `(batch, max_contracts_per_bar, contract_features)` current executable snapshot

Outputs:

- `no_trade_score`
- `contract_scores`
- `valid_mask`

The model does not emit a synthetic strike class, direction head, or learned risk head. It scores the actual contracts visible on the current bar.

## Current Baseline Loss

The live reset baseline is intentionally minimal and replay-aligned:

- Gate BCE on supervised rows
- Soft KL selection with `SOFT_TEMP=0.20`
- No direct PnL regression
- No auxiliary side head

Details:

- `gate_loss` compares `max(contract_scores) - no_trade_score` against `label_trade`
- `sel_loss` builds a soft target from sidecar `row_labels` with `softmax(pnl / SOFT_TEMP)`
- invalid or unlabeled contracts receive zero target mass
- selection loss is weighted by label quality (top margin between best and second-best contract)
  - bars with margin >= 0.05 get full weight; margin 0 gets zero weight
  - this downweights the 30.6% of bars where the oracle answer is ambiguous noise
- total loss is `GATE_W * gate_loss + SEL_W * sel_loss`

`row_labels` are still used, but only to build the KL target distribution. The live baseline does not regress score magnitudes directly to realized P&L.

## Risk Policy

The frozen v4 harness keeps risk policy-driven:

- fixed stop
- fixed target
- fixed max hold
- configured exit policy

Those live in `v2/core/policy.py` and are part of the mutable surface.

## What `run_one` Does

`./v2/ops/deploy.sh run_one exp_NNN`:

1. runs the local pre-GPU integrity gate
2. uploads `v2/train.py` and `v2/core/policy.py`
3. uploads `v2/data.pt`
4. uploads `v2/data_sidecars/` when referenced by the manifest
5. runs `python3 -m v2.ops.run_experiment_wf --id exp_NNN`
6. downloads `v2/models/model_candidate.pt`
7. downloads the matching artifact bundle
8. appends official results to `v2/results.tsv`

## What `run_screen` Does

`./v2/ops/deploy.sh run_screen exp_NNN`:

1. runs the local pre-GPU integrity gate
2. uploads `v2/train.py` and `v2/core/policy.py`
3. syncs data if needed
4. runs `python3 -m v2.ops.run_experiment_wf --id exp_NNN_screen --n-folds 1 --no-artifacts`
5. does not download a model or artifact
6. prints results to stdout only; screening notes belong in `v2/lab_notebook.md`
