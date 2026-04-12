# v2 Labeling Contract

This document describes the labels that are actually in the active dataset.

## Active Label Regime

The canonical dataset uses exact-chain sidecar labels:

- dataset version: `v4_exact_chain`
- dataset fingerprint: `46f2d184e186496f`

## What Is Stored Per Day Sidecar

Each day sidecar (`v2/data_sidecars/*.pt`) stores:

- `row_labels`: per-contract forward P&L under the fixed policy (stop/target/hold from `core/policy.py`)
- `bar_best_contract_idx`: index of the best contract on each bar
- `bar_best_pnl`: realized P&L of the best contract
- `bar_label_trade`: whether the best P&L exceeds `DEFAULT_POLICY.label_gate_min_pnl`
- `bar_labelable`: whether the bar has valid forward contract labels

## What Is Stored In The Manifest

The manifest (`v2/data.pt`) stores:

- `X`: `(bars, 30, 47)` normalized context features
- `X_sim`: raw replay features
- bar metadata and split masks
- no contract-level data (that lives in sidecars)

## Supervised Window

Current supervised bars are:

- `30 <= bar_of_day < 270`

That matches the current replay-time no-trade window in the default policy.

## Gate Rule

The single source of truth for the gate threshold is:

- `DEFAULT_POLICY.label_gate_min_pnl`
- Current value: `0.04`

## Key Dataset Properties

- 986 unique trading days
- ~38 executable contracts per bar (median)
- top-vs-second contract margin: ~0.024 (median, very tight)
- best-call vs best-put margin: ~0.703 (median, strongly learnable)

## What Labels Are Teaching Today

The model is being asked to learn:

- whether to trade at all (gate)
- which specific contract to select from the executable snapshot
- a soft ranking target derived from realized forward P&L

The live reset baseline does not perform direct score-to-P&L regression. It uses `row_labels` only to build the KL target distribution over executable contracts.

## What Is No Longer True

These older statements are no longer correct for the active dataset:

- separate per-direction P&L tensors in the manifest
- a binary call/put direction label in the manifest
- the pre-v4 label scheme with fixed stop/target grids
- session-open ATM prices drive relabeling
- labels are oracle `TradeIntent` objects
