# v2 Labeling Contract

This document describes the labels that are actually in the active dataset.

## Active Label Regime

The canonical dataset currently uses:

- `label_scheme = dual_direction_pnl_tier3`
- variable stop, target, and hold labels
- current fingerprint `03566aeb8adf1040`

## What Is Stored Per Supervised Bar

For each valid supervised bar, the dataset stores:

- `label_call_pnl`
- `label_put_pnl`
- `label_pnl` = max of the two
- `label_direction` = whichever side had the higher P&L
- `label_trade` = whether `label_pnl > DEFAULT_POLICY.label_gate_min_pnl`
- `label_stop_pct`
- `label_target_pct`
- `label_max_hold`
- `label_confidence`

This is not a serialized oracle `TradeIntent` label. Training is component-supervised through these tensors.

## Current Two-Step Label Path

### Step 1: Base Build

`v2/pipeline/build_v2_dataset.py` creates the initial dataset with:

- fixed stop `0.30`
- fixed target `0.50`
- fixed hold `30`
- dual-direction P&L labels
- current nearest ATM entry prices

It labels bars only inside the supervised trade window.

### Step 2: Tier 3 Relabel

`v2/pipeline/relabel_tier3.py` upgrades those labels using a grid search over:

- stops: `[0.15, 0.2, 0.25, 0.3, 0.4, 0.5]`
- targets: `[0.2, 0.3, 0.5, 0.8, 1.2]`
- holds: `[30, 60, 120, 240, 390]`

The relabeler:

- uses `nearest_call_close` and `nearest_put_close`
- prices spread from the actual exit bar timing
- recomputes label counts and fingerprint after relabeling

## Supervised Window

Current supervised bars are:

- `30 <= bar_of_day < 270`

That matches the current replay-time no-trade window in the default policy.

## Current Label Facts

From the active dataset metadata:

```text
total_signal_bars = 236,641
total_trades      = 157,232
gate_true_rate    = 0.6644
```

Current label grid in metadata:

```text
stops=[0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
targets=[0.2, 0.3, 0.5, 0.8, 1.2]
holds=[30, 60, 120, 240, 390]
```

## Gate Rule

`label_trade` is not hardcoded in multiple places anymore.

The single source of truth is:

- `DEFAULT_POLICY.label_gate_min_pnl`

Current value:

- `0.04`

## What Labels Are Teaching Today

The model is being asked to learn:

- whether there is enough directional edge to trade
- which direction has better expected P&L
- what stop / target / hold regime tends to work

The model is not yet strongly supervised on rich strike choice.

## What Is No Longer True

These older statements are no longer correct for the active dataset:

- Tier 3 holds stop at 240
- session-open ATM prices drive relabeling
- relabeling leaves metadata stale
- labels are oracle `TradeIntent` objects
