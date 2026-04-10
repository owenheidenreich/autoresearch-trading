# How Training Works

## Two Loops

```text
OUTER LOOP
    hypothesis
    edit train.py and/or core/policy.py
    commit
    screen: deploy.sh run_screen exp_NNN (1 fold)
    if screening passes:
        official: deploy.sh run_one exp_NNN (5 folds)
        read walk-forward score
        keep or revert

INNER LOOP
    for each fold:
        train from scratch (re-read TRAIN_SEED per fold)
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

## Current Loss (exp_079 recovery)

The exact-chain recovery plan replaces hard contract selection with staged supervision:

### Stage 1: Side supervision through contract_scores (exp_079)
- **Side CE**: 2-class call/put cross-entropy derived from `contract_scores`
  - `best_call_score = max(scores where right_is_put < 0.5)`
  - `best_put_score = max(scores where right_is_put >= 0.5)`
  - BCE on `(best_put_score - best_call_score)` vs oracle side
  - Applied on trade rows with both call and put contracts present
- **PnL regression**: Huber loss on per-contract P&L for valid rows (unchanged)
- **Gate BCE**: binary CE on supervised rows (unchanged)

### Stage 2: Soft within-side ranking (exp_080, planned)
- Restrict to oracle side on trade rows
- Build target distribution as `softmax(oracle_side_pnl / 0.05)`
- KL divergence between target probs and model side-contract logits
- Keeps side CE from stage 1

### Stage 3: Gate calibration (exp_081, conditional)
- Only if overtrading remains after stage 2
- `no_trade_weight = 3.0` via manual sample weighting on `gate_target == 0`

Explicitly forbidden: "auxiliary head not used by replay" as the primary recovery path. Every loss term must flow through `contract_scores` and `no_trade_score` which replay actually reads.

## Risk Policy

The first frozen v4 harness keeps risk policy-driven:

- fixed stop
- fixed target
- fixed max hold
- configured exit policy

Those live in `v2/core/policy.py` and are part of the mutable surface.

## What `run_one` Does

`./v2/ops/deploy.sh run_one exp_NNN`:

1. uploads `v2/train.py` and `v2/core/policy.py`
2. uploads `v2/data.pt`
3. uploads `v2/data_sidecars/` when referenced by the manifest
4. runs `python3 -m v2.ops.run_experiment_wf --id exp_NNN`
5. downloads `v2/model_candidate.pt`
6. downloads the matching artifact bundle

## What `run_screen` Does

`./v2/ops/deploy.sh run_screen exp_NNN`:

1. uploads `v2/train.py` and `v2/core/policy.py`
2. syncs data if needed
3. runs `python3 -m v2.ops.run_experiment_wf --id exp_NNN_screen --n-folds 1 --no-artifacts`
4. no model or artifact download
5. results printed to stdout only — does not modify `results.tsv`
