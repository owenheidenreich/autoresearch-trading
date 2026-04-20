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

- `(batch, 30, 79)` context window on the rebuilt slice-era dataset
- `(batch, max_contracts_per_bar, 22)` current executable snapshot (19 base + 3 economic)

Outputs:

- `contract_scores`
- `valid_mask`
- `gate_logit` / `opportunity_logit` — the live trade/no-trade gate
- `side_logit` — P(call is better) from context alone
- `aggression_logits` — moneyness bucket prediction (ATM / near-OTM / far-OTM)

The model scores the actual contracts visible on the current bar with a two-stage decision process: first gate the bar, then rank contracts with `contract_scores` plus any replay-time side adjustment from policy. On the current Direction B path, the gate can be structurally decoupled from the ranker with `GATE_ARCH=decoupled_mlp`, and replay can calibrate that gate by quantile instead of a fixed zero threshold.

## Current Baseline Loss

The active research path is now the slice-era supervised model with Direction B available:

- Dynamic near-ATM slice (ATM ±10 strikes) is the canonical competition set
- Sparse gate supervision is available through `OPP_LABEL=sparse_high_conviction`
- Gate path can be shared-context or decoupled (`GATE_ARCH=shared_context|decoupled_mlp`)
- Soft KL selection loss remains the canonical contract-ranking objective over in-slice contracts
- `SEL_TARGET_MODE=soft_pnl` and `SOFT_TEMP=0.40` are the recommended next-screen settings
- No direct PnL regression

Details:

- `gate_loss` supervises `gate_logit` directly
  - `GATE_TARGET_MODE=binary`: balanced BCE on `label_trade`
  - `GATE_TARGET_MODE=max_pnl`: MSE to `bar_max_pnl - GATE_PNL_THRESHOLD`
- `OPP_LABEL=sparse_high_conviction`: positive only when the slice has a strong winner and only a small number of profitable contracts
- `comp_loss` optionally reuses `opportunity_logit` as a competence head when `COMP_W > 0`
- `side_loss` supervises `side_logit` with BCE on oracle side (trade rows only)
- `agg_loss` supervises `aggression_logits` with cross-entropy on oracle moneyness bucket
- when both gate classes are present, the BCE is computed on a class-balanced subset
- `sel_loss` builds a soft target from sidecar `row_labels` with `softmax(pnl / SOFT_TEMP)`
- ambiguous bars (top PnL margin < `NOISE_MARGIN`) use uniform target at reduced weight instead of being dropped
- total loss is `GATE_W * gate + SEL_W * sel + SIDE_SEL_W * side_sel + EXACT_W * exact + COMP_W * comp + SIDE_W * side + AGG_W * agg`
- replay gating can now use:
  - fixed threshold: `POLICY_GATE_THRESHOLD`
  - quantile threshold: `POLICY_GATE_THRESHOLD_MODE=quantile` and `POLICY_GATE_TARGET_PASS_RATE=<rate>`

## Risk Policy

The frozen v4 harness keeps risk policy-driven:

- fixed stop
- fixed target
- fixed max hold
- configured exit policy

Those live in `v2/core/policy.py` and are part of the mutable surface.
The current official policy is full supervised day: entries are allowed from bar `30` through `270`.

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
