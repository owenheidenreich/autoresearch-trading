# How Training Works

## Two Loops

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

## Current Model

The current model is an exact-chain contract scorer.

Input:

- `(batch, 30, 47)` context window
- `(batch, max_contracts_per_bar, contract_features)` current executable snapshot

Outputs:

- `no_trade_score`
- `contract_scores`

The model does not emit a synthetic strike class. It scores the actual contracts visible on the current bar.

## Current Loss

- regression on realized contract P&L for valid rows
- selection cross-entropy across `NO_TRADE` plus the executable contracts
- gate/selection loss is applied only on `label_trade_valid` bars so missing forward labels are not trained as false no-trades

## Risk Policy

The first frozen v4 harness keeps risk policy-driven:

- fixed stop
- fixed target
- fixed max hold
- configured exit policy

Those live in `v2/core/policy.py` and are part of the nightly mutable surface.

## What `run_one` Does

`./v2/ops/deploy.sh run_one exp_NNN`:

1. uploads `v2/train.py` and `v2/core/policy.py`
2. uploads `v2/data.pt`
3. uploads `v2/data_sidecars/` when referenced by the manifest
4. runs `python3 -m v2.ops.run_experiment_wf --id exp_NNN`
5. downloads `v2/model_candidate.pt`
6. downloads the matching artifact bundle
