# v2 Data Contract

**Authority**: `v2/pipeline/build_v2_dataset.py` output and `v2/core/config.py:RuntimeConfig`
are the single sources of truth. Any discrepancy between this doc and the code means this doc is wrong.

## Canonical Artifact

- Manifest path: `v2/data.pt`
- Sidecar directory: `v2/data_sidecars/`
- Current schema version: `v4_exact_chain_v2_paths`

The manifest fingerprint is content-based and includes the sidecar digest.

## Raw Inputs

`build_v2_dataset.py` reads from:

- `~/.cache/autoresearch-trading/data/spx_1min.pkl`
- `~/.cache/autoresearch-trading/data/spy_1min.pkl`
- `~/.cache/autoresearch-trading/data/vix_1min.pkl`
- `~/.cache/autoresearch-trading/data/spxw_full_chain/*.pkl`

## Manifest Structure

```python
{
    "X": torch.Tensor,                # (N_bars, 52) normalized model features
    "X_sim": torch.Tensor,            # (N_bars, 52) raw replay/simulator features
    "feature_names": list[str],       # 52 canonical feature names
    "spot_prices": torch.Tensor,      # (N_bars,)
    "label_trade": torch.Tensor,      # (N_bars,) bool
    "label_trade_valid": torch.Tensor,# (N_bars,) bool — supervision-valid bars only
    "best_contract_pnl": torch.Tensor,
    "best_contract_strike": torch.Tensor,
    "best_contract_right": torch.Tensor,
    "dates": list[str],
    "bar_of_day": torch.Tensor,
    "train_mask": torch.Tensor,
    "val_mask": torch.Tensor,
    "promote_mask": torch.Tensor,
    "shadow_mask": torch.Tensor,
    "metadata": { ... },
}
```

## Sidecar Structure

Each `v2/data_sidecars/YYYY-MM-DD.pt` stores one session.

22 contract features per row (see `v2/core/chain_data.py:CONTRACT_FEATURE_FIELDS`).

```python
{
    "schema_version": str,              # "v4_exact_chain_v2_paths"
    "date": str,
    "expiry": str,
    "n_bars": int,
    "bar_timestamps": np.ndarray,       # (n_bars,) int64
    "contract_strike": np.ndarray,      # (M,) float32
    "contract_right": np.ndarray,       # (M,) int8 — 0=call 1=put
    "contract_mid": np.ndarray,         # (M, n_bars) float32
    "contract_bid": np.ndarray,         # (M, n_bars) float32 — proxy from OHLC
    "contract_ask": np.ndarray,         # (M, n_bars) float32 — proxy from OHLC
    "contract_quality": np.ndarray,     # (M, n_bars) int8
    "row_features": np.ndarray,         # (R, 22) float32 — flattened per-bar contract features
    "row_labels": np.ndarray,           # (R,) float32 — realized net PnL, NaN if unlabeled
    "row_labels_short": np.ndarray,     # (R,) float32 — SHORT_POLICY overlay
    "row_labels_eod": np.ndarray,       # (R,) float32 — EOD_POLICY overlay
    "row_contract_idx": np.ndarray,     # (R,) int32 — maps row -> contract index
    "row_raw_returns": np.ndarray,      # (R, 5) float32 — returns at 5/10/15/30/60 bar horizons
    "row_mfe": np.ndarray,              # (R, 5) float32 — max favorable excursion
    "row_mae": np.ndarray,              # (R, 5) float32 — max adverse excursion
    "row_bars_to_breakeven": np.ndarray,# (R,) float32
    "row_impulse_fraction": np.ndarray, # (R,) float32
    "bar_ptrs": np.ndarray,             # (n_bars+1,) int32 — row slicing per bar
    "bar_best_contract_idx": np.ndarray,# (n_bars,) int32
    "bar_best_pnl": np.ndarray,         # (n_bars,) float32
    "bar_label_trade": np.ndarray,      # (n_bars,) bool
    "bar_labelable": np.ndarray,        # (n_bars,) bool — bar has >=1 finite forward label
    "bar_quality": np.ndarray,          # (n_bars,) int8
}
```

## Metadata Contract

Important metadata fields:

- `version` — "v4_exact_chain"
- `fingerprint` — SHA256-based content fingerprint
- `chain_schema_version` — "v4_exact_chain_v2_paths"
- `chain_sidecar_dir` — "v2/data_sidecars"
- `chain_sidecar_digest` — SHA256 of all sidecar file hashes
- `contract_feature_fields` — 22-element list from chain_data.py
- `max_contracts_per_bar` — observed max (currently 285)
- `n_features` — 52
- `label_scheme` — "exact_contract_fixed_risk"
- `label_gate_min_pnl` — 0.04 (4%)
- `risk_policy` — stop_pct, target_pct, max_hold_bars, exit_policy
- `execution_filters` — min_contract_mid, max_spread_fraction, require_volume_or_transactions
- `split` — train_days, val_days, promote_days, shadow_days
- `build_git_sha` — commit SHA at build time
- `build_timestamp` — ISO timestamp of build
- `config_fingerprint` — RuntimeConfig fingerprint

## Replay Contract

- Replay loads exact contracts from sidecars.
- Training scores the normalized manifest `X`; replay uses raw `X_sim` so labels and simulator costs match exactly.
- Missing contract history is represented as `NaN` row labels and `bar_labelable=False`, not as silent remapping to another strike.
- Current executable contracts remain visible in the snapshot even when a forward label is unavailable.
