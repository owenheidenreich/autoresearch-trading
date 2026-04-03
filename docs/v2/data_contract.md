# v2 Data Contract

## Purpose

Defines the raw data inputs, derived dataset format, and migration path
from v1 data artifacts.

---

## Raw Data Sources

### SPX 1-Minute OHLCV
- **Source:** IBKR historical data API
- **Format:** Parquet files, one per date
- **Fields:** timestamp, open, high, low, close (no volume for index)
- **Cache:** `data/spx_1min/YYYY-MM-DD.parquet`
- **Coverage:** 4+ years rolling (March 2022 to present)

### SPY 1-Minute Volume
- **Source:** Polygon.io API
- **Format:** Parquet files, one per date
- **Fields:** timestamp, volume, vwap
- **Cache:** `data/spy_1min/YYYY-MM-DD.parquet`
- **Purpose:** SPX has no volume. SPY volume proxies SPX activity.

### VIX 1-Minute
- **Source:** IBKR historical data API
- **Format:** Parquet files, one per date
- **Fields:** timestamp, open, high, low, close
- **Cache:** `data/vix_1min/YYYY-MM-DD.parquet`

### Option Chain Snapshots
- **Source:** IBKR historical data API (or computed from Greeks models)
- **Format:** Parquet files, keyed by (date, bar_index)
- **Fields:** strike, right, bid, ask, last, IV, delta, gamma, theta, vega
- **Cache:** `data/options/YYYY-MM-DD.parquet`
- **Coverage:** ATM +/- 50 points, both calls and puts, 0DTE only

---

## Derived Dataset Format

### v2 data.pt Structure

```python
{
    # Features
    "X": torch.Tensor,              # (N_bars, 39) float32
    "feature_names": list[str],      # 39 names, matches FEATURE_NAMES

    # Oracle labels (from labeling.md)
    "oracle_intents": list[dict],    # TradeIntent.asdict() per bar, or None for no-trade
    "oracle_pnl": torch.Tensor,      # (N_bars,) best achievable P&L per bar
    "oracle_trade": torch.Tensor,    # (N_bars,) bool: should trade here?
    "oracle_direction": torch.Tensor, # (N_bars,) 0=call, 1=put (NaN if no trade)
    "oracle_strike_offset": torch.Tensor, # (N_bars,) offset from ATM in points
    "oracle_stop_pct": torch.Tensor,  # (N_bars,) optimal stop as % of premium
    "oracle_target_pct": torch.Tensor, # (N_bars,) optimal target as % of premium

    # Auxiliary prediction labels (for market head, if used)
    "y_return_15": torch.Tensor,     # (N_bars,) 15-bar forward SPX return
    "y_return_30": torch.Tensor,     # (N_bars,) 30-bar forward SPX return
    "y_return_60": torch.Tensor,     # (N_bars,) 60-bar forward SPX return

    # Metadata
    "dates": list[str],              # date string per bar
    "bar_indices": torch.Tensor,     # (N_bars,) bar-of-day index (0-389)
    "timestamps": list[str],         # ISO 8601 per bar

    # Split masks
    "train_mask": torch.Tensor,      # (N_bars,) bool
    "val_mask": torch.Tensor,        # (N_bars,) bool

    # Provenance
    "metadata": {
        "feature_version": str,       # e.g. "v2.0"
        "label_version": str,         # e.g. "oracle_tier3"
        "evaluator_version": str,     # SHA-256 of evaluator rules
        "data_dates": [str, str],     # [first_date, last_date]
        "num_bars": int,
        "num_train_bars": int,
        "num_val_bars": int,
        "build_timestamp": str,       # when this dataset was created
        "raw_data_fingerprint": str,  # SHA-256 of raw data files used
    }
}
```

### Train/Val Split

- Split by date: last 60 trading days = validation
- No bar-level mixing (entire days are train or val)
- Split date stored in metadata for reproducibility

---

## Fingerprinting

**Raw data fingerprint:** SHA-256 of sorted list of raw data file paths + sizes.
If any raw file changes, the fingerprint changes, and the dataset is stale.

**Feature version:** String incremented when feature computation logic changes.
If feature version in data.pt doesn't match current code, the dataset must be rebuilt.

**Label version:** String identifying the oracle labeler tier and parameters.
Different label versions produce incompatible datasets.

**Evaluator version:** SHA-256 of the evaluator rules (spread, stops, fills).
Oracle labels are only valid under the evaluator version they were computed with.

---

## Backward Compatibility with v1

### What v2 Can Read from v1

- **Raw caches** (data/spx_1min/, data/spy_1min/, data/vix_1min/): Same format.
  v2 pipeline reads them directly. No migration needed.

- **Feature computation** (prepare.py compute_features()): Same 39 features.
  v2 core/features.py produces identical output given identical input.

### What v2 Cannot Read from v1

- **v1 data.pt**: Different label scheme. v1 labels are proxy-based (MFE/MAE).
  v2 labels are oracle-based (TradeIntent). The feature tensor (X) is compatible,
  but the label tensors are not.

- **v1 model weights**: v2 may use a different architecture. Even if the
  architecture matches, the loss function targets are different, so v1 weights
  are not meaningful for v2 training.

### Separation

- v1 data lives at `training/data.pt` (unchanged, never overwritten)
- v2 data lives at `v2/data.pt` (new path)
- v1 raw caches are shared (read-only for both systems)
- v2 never modifies v1 artifacts

---

## Data Pipeline (v2/pipeline/build_dataset.py)

### Steps

1. **Download** raw data for missing dates (IBKR + Polygon APIs)
2. **Compute features** using v2/core/features.py (same 39 features)
3. **Compute oracle labels** using v2/core/labels.py (new)
4. **Normalize** features using v2/core/features.py normalization
5. **Split** into train/val by date
6. **Fingerprint** all components
7. **Save** to v2/data.pt

### Incremental Updates

When new trading days are available:
1. Download only new dates
2. Recompute features and labels for new dates only
3. Append to existing dataset
4. Re-split (val window slides forward)
5. Update fingerprint

Full rebuild is only required when feature or label logic changes.
