# How Training Works

## Two Loops

```text
OUTER LOOP
    hypothesis (informed by previous trace analysis)
    edit the approved mutable surface for the current hypothesis
    commit
    pre-run gate: v2.ops.pre_run_gate (includes data integrity)
    screen: deploy.sh run_screen exp_NNN (1 fold)
    if screening passes or justifies a trace-informed follow-up:
        official: deploy.sh run_one exp_NNN (5 folds)
        DECISION TRACE: replay --traces (mandatory before keep/revert)
        SIDE-BIAS AUDIT: data_integrity --side-bias-audit (standard after important runs)
        analyze trace: gate accuracy, selection accuracy, P&L gap
        keep or revert (promotion by score, continuation may use traces)
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

The live working baseline is the restored `exp_119` family:

- Balanced gate BCE on supervised rows
- Soft KL selection loss
- `SOFT_TEMP=0.10`
- `NOISE_MARGIN=0.01`
- No direct PnL regression
- No auxiliary side head

Details:

- `gate_loss` compares `max(contract_scores) - no_trade_score` against `label_trade`
- when both gate classes are present, the BCE is computed on a class-balanced subset so the trade-majority class does not dominate the gate gradient
- `sel_loss` builds a soft target from sidecar `row_labels` with `softmax(pnl / SOFT_TEMP)`
- invalid or unlabeled contracts receive zero target mass
- noise bars where the top PnL margin is below `NOISE_MARGIN` are excluded from the selection loss
- total loss is `GATE_W * gate_loss + SEL_W * sel_loss`

The rejected `exp_121` ranking-loss screen is an important finding, but not the live baseline. It showed that side collapse survives a loss-family change; it did not prove that the executable KL targets are inherently put-biased.

## Risk Policy

The frozen v4 harness keeps risk policy-driven:

- fixed stop
- fixed target
- fixed max hold
- configured exit policy

Those live in `v2/core/policy.py` and are part of the mutable surface.
The current official baseline policy is morning-only: entries are allowed from bar `60` through `120`.

## What `run_one` Does

`./v2/ops/deploy.sh run_one exp_NNN`:

1. runs the local pre-GPU integrity gate
2. uploads the currently approved live code needed by the hypothesis (`v2/train.py` plus any explicitly approved replay/audit companions)
3. uploads `v2/data.pt`
4. uploads `v2/data_sidecars/` when referenced by the manifest
5. runs `python3 -m v2.ops.run_experiment_wf --id exp_NNN`
6. downloads `v2/models/model_candidate.pt`
7. downloads the matching artifact bundle
8. appends official results to `v2/results.tsv`

## What `run_screen` Does

`./v2/ops/deploy.sh run_screen exp_NNN`:

1. runs the local pre-GPU integrity gate
2. uploads the currently approved live code needed by the hypothesis
3. syncs data if needed
4. runs `python3 -m v2.ops.run_experiment_wf --id exp_NNN_screen --n-folds 1 --no-artifacts`
5. does not download a model or artifact
6. prints results to stdout only; screening notes belong in `v2/lab_notebook.md`

## Standard Diagnostics

After meaningful screening or official runs, the standard local diagnostics are:

- `python3 -m v2.replay --model v2/models/model_candidate.pt --data v2/data.pt --mask promote --traces`
- `.venv/bin/python3 -m v2.core.data_integrity --data v2/data.pt --side-bias-audit --model v2/models/model_candidate.pt`

The side-bias audit is the canonical way to separate:

- label-side structure
- valid contract availability by side
- soft-target side mass
- model traded-side share
- side-conditioned accuracy and selected-label P&L
