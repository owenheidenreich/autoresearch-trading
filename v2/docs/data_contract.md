# v2 Data Contract

This document describes the active dataset artifact used by training and replay.

## Canonical Artifact

- Active path: `v2/data.pt`
- Backup copy: `v2/data_harness_repair.pt`
- Current version: `v2_harness_repair`
- Current fingerprint: `03566aeb8adf1040`

The active dataset is a PyTorch artifact. Its fingerprint is content-based and is recomputed whenever labels or metadata materially change.

## Raw Inputs

`build_v2_dataset.py` reads from the local pickle caches:

- `~/.cache/autoresearch-trading/data/spx_1min.pkl`
- `~/.cache/autoresearch-trading/data/spy_1min.pkl`
- `~/.cache/autoresearch-trading/data/vix_1min.pkl`
- `~/.cache/autoresearch-trading/data/spxw_wide/*.pkl`

Those inputs provide:

- SPX OHLC
- SPY volume and close
- VIX close
- wide-grid SPXW option bars, volumes, and transaction counts

## Build Pipeline

The full canonical rebuild is a two-step process:

1. `python -m v2.pipeline.build_v2_dataset --output v2/data.pt`
2. `python -m v2.pipeline.relabel_tier3 --data v2/data.pt --tier 3`

Step 1 builds:

- all 47 features from raw caches
- dynamic ATM and OTM replay price tensors
- fixed-risk dual-direction P&L labels
- train / val / promote / shadow masks
- initial metadata and fingerprint

Step 2 upgrades the labels to Tier 3 variable-risk labels and recomputes the fingerprint.

## Dataset Structure

```python
{
    "X": torch.Tensor,                  # (N_bars, 47) float32
    "feature_names": list[str],         # 47 names, canonical order

    # Labels
    "label_trade": torch.Tensor,        # (N_bars,) bool
    "label_direction": torch.Tensor,    # (N_bars,) int32, 0=call 1=put -1=invalid
    "label_outcome": torch.Tensor,      # (N_bars,) int32
    "label_pnl": torch.Tensor,          # (N_bars,) best directional P&L
    "label_stop_pct": torch.Tensor,     # (N_bars,) float32
    "label_target_pct": torch.Tensor,   # (N_bars,) float32
    "label_max_hold": torch.Tensor,     # (N_bars,) int32
    "label_confidence": torch.Tensor,   # (N_bars,) float32
    "label_call_pnl": torch.Tensor,     # (N_bars,) float32
    "label_put_pnl": torch.Tensor,      # (N_bars,) float32

    # Price references
    "spot_prices": torch.Tensor,        # (N_bars,) float32
    "spx_estimated": torch.Tensor,      # (N_bars,) float32
    "nearest_call_close": torch.Tensor, # (N_bars,) float32
    "nearest_put_close": torch.Tensor,  # (N_bars,) float32
    "atm_call_prices": torch.Tensor,    # (N_bars,) float32
    "atm_put_prices": torch.Tensor,     # (N_bars,) float32
    "otm5_call_prices": torch.Tensor,   # (N_bars,) float32
    "otm5_put_prices": torch.Tensor,    # (N_bars,) float32
    "otm10_call_prices": torch.Tensor,  # (N_bars,) float32
    "otm10_put_prices": torch.Tensor,   # (N_bars,) float32
    "otm15_call_prices": torch.Tensor,  # (N_bars,) float32
    "otm15_put_prices": torch.Tensor,   # (N_bars,) float32
    "otm20_call_prices": torch.Tensor,  # (N_bars,) float32
    "otm20_put_prices": torch.Tensor,   # (N_bars,) float32
    "otm25_call_prices": torch.Tensor,  # (N_bars,) float32
    "otm25_put_prices": torch.Tensor,   # (N_bars,) float32
    "otm30_call_prices": torch.Tensor,  # (N_bars,) float32
    "otm30_put_prices": torch.Tensor,   # (N_bars,) float32

    # Bar context
    "dates": list[str],                 # length N_bars
    "bar_of_day": torch.Tensor,         # (N_bars,) int32, 0..389

    # Masks
    "train_mask": torch.Tensor,         # (N_bars,) bool
    "val_mask": torch.Tensor,           # (N_bars,) bool
    "promote_mask": torch.Tensor,       # (N_bars,) bool
    "shadow_mask": torch.Tensor,        # (N_bars,) bool

    "metadata": { ... },
}
```

## Metadata Contract

Important metadata fields in the active dataset:

- `version`
- `build_timestamp`
- `fingerprint`
- `n_features`
- `normalization`
- `total_signal_bars`
- `total_trades`
- `gate_true_rate`
- `mean_pnl_trade`
- `fixed_stop`
- `fixed_target`
- `fixed_hold`
- `label_gate_min_pnl`
- `split`
- `cost_model`
- `direction_signal`
- `label_scheme`
- `label_tier`
- `label_grid`
- `atm_source`
- `poc_va_source`
- `trade_window`

The metadata must describe the actual tensor contents. `relabel_tier3.py` now updates the counts and recomputes the fingerprint after relabeling.

## Current Active Values

```text
n_features:        47
normalization:     rolling_zscore_60day
label_scheme:      dual_direction_pnl_tier3
trade_window:      bar 30-270
atm_source:        dynamic_nearest_per_bar
poc_va_source:     incremental_bars_seen_so_far
total_signal_bars: 236,641
total_trades:      157,232
```

## Split Contract

The dataset stores four masks:

- `train_mask`
- `val_mask`
- `promote_mask`
- `shadow_mask`

The canonical fixed split at the dataset level is used for local replay and analysis.
Walk-forward experiments do not rely on those masks for training. `v2/core/walkforward.py` builds fold-specific masks from the date list.

## Replay Price Contract

Replay uses coarse option price arrays:

- current nearest ATM
- OTM 5
- OTM 10
- OTM 15
- OTM 20
- OTM 25
- OTM 30

These arrays are keyed to the dynamic nearest ATM per bar, not to the session-open ATM.

## Fingerprint Rules

Dataset compatibility is enforced by `v2/core/dataset_fingerprint.py`.

The fingerprint changes when any of these materially change:

- tensors
- dates
- feature names
- core metadata fields

Artifact loading and replay use the fingerprint to reject models trained on a different dataset.
