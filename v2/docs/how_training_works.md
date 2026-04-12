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

Optional context wiring for future Kronos-inspired experiments:

- learned temporal embeddings may be added on top of the 30-bar context window
- if temporal IDs are needed at inference, replay must pass the same IDs the trainer used
- train-only masking of the flow feature slice may be used to reduce brittleness to noisy auxiliary volume inputs
- exact-chain contract scoring semantics must remain unchanged

Optional training-side calibration for future trace-targeted experiments:

- a small cross-side ranking term may be added on traded rows to force the oracle side to outrank the opposite side without adding a separate direction head or inference-time side mask

Outputs:

- `no_trade_score`
- `contract_scores`
- `valid_mask`

The model does not emit a synthetic strike class, direction head, or learned risk head. It scores the actual contracts visible on the current bar.

## Current Baseline Loss

The live reset baseline is intentionally minimal and replay-aligned:

- Gate BCE on supervised rows, with balanced subsampling when both classes are present
- Pairwise ranking selection loss (oracle best contract must outscore all other valid contracts)
- No direct PnL regression
- No auxiliary side head

Details:

- `gate_loss` compares `max(contract_scores) - no_trade_score` against `label_trade`
- when both gate classes are present, the BCE is computed on a class-balanced subset so the trade-majority class does not dominate the gate gradient
- `sel_loss` uses a pairwise BPR formulation: `-log(sigmoid(best_score - other_score))` averaged over all valid (oracle_best, other) pairs
- noise bars where the top PnL margin is below `NOISE_MARGIN` are excluded from the selection loss
- total loss is `GATE_W * gate_loss + SEL_W * sel_loss`

The pairwise ranking loss replaced the previous soft KL selection loss (`softmax(pnl / SOFT_TEMP)`) because the KL target had a structural put bias: the softmax exponentially amplified asymmetric put PnL margins, causing the model to systematically favor puts regardless of architecture or hyperparameters.

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
