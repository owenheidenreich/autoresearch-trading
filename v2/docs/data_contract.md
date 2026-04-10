# v2 Data Contract

## Canonical Artifact

- Manifest path: `v2/data.pt`
- Sidecar directory: `v2/data_sidecars/`
- Current version: `v4_exact_chain`

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
    "X": torch.Tensor,                # (N_bars, 47) normalized model features
    "X_sim": torch.Tensor,            # (N_bars, 47) raw replay/simulator features
    "feature_names": list[str],
    "spot_prices": torch.Tensor,      # (N_bars,)
    "label_trade": torch.Tensor,      # (N_bars,)
    "label_trade_valid": torch.Tensor,# (N_bars,) supervision-valid bars only
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

Each `v2/data_sidecars/YYYY-MM-DD.pt` stores one session:

```python
{
    "schema_version": str,
    "date": str,
    "expiry": str,
    "n_bars": int,
    "bar_timestamps": np.ndarray,
    "contract_strike": np.ndarray,      # (M,)
    "contract_right": np.ndarray,       # (M,) 0=call 1=put
    "contract_mid": np.ndarray,         # (M, n_bars)
    "contract_bid": np.ndarray,         # proxy
    "contract_ask": np.ndarray,         # proxy
    "contract_quality": np.ndarray,     # (M, n_bars)
    "row_features": np.ndarray,         # flattened current executable rows
    "row_labels": np.ndarray,           # flattened realized net pnl labels, NaN if unlabeled
    "row_contract_idx": np.ndarray,     # maps row -> contract index
    "bar_ptrs": np.ndarray,             # row slicing per bar
    "bar_best_contract_idx": np.ndarray,
    "bar_best_pnl": np.ndarray,
    "bar_label_trade": np.ndarray,
    "bar_labelable": np.ndarray,        # bar has at least one honest forward label
    "bar_quality": np.ndarray,
}
```

## Metadata Contract

Important metadata fields:

- `version`
- `fingerprint`
- `chain_schema_version`
- `chain_sidecar_dir`
- `chain_sidecar_digest`
- `contract_feature_fields`
- `max_contracts_per_bar`
- `label_scheme`
- `label_gate_min_pnl`
- `risk_policy`
- `execution_filters`
- `split`

## Replay Contract

- Replay loads exact contracts from sidecars.
- Training scores the normalized manifest `X`; replay uses raw `X_sim` so labels and simulator costs match exactly.
- Missing contract history is represented as `NaN` row labels and `bar_labelable=False`, not as silent remapping to another strike.
- Current executable contracts remain visible in the snapshot even when a forward label is unavailable.
