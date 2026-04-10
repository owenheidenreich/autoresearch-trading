"""Deterministic fingerprinting for versioned dataset artifacts."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import torch


FINGERPRINT_SCHEMA_VERSION = "v4_dataset_fp_v1"

_METADATA_KEYS = (
    "version",
    "n_features",
    "normalization",
    "split",
    "label_scheme",
    "label_gate_min_pnl",
    "trade_window",
    "chain_schema_version",
    "chain_sidecar_digest",
    "contract_feature_fields",
    "max_contracts_per_bar",
    "risk_policy",
    "execution_filters",
)


def _update_with_json(hasher: "hashlib._Hash", value: Any) -> None:
    hasher.update(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    )


def _update_with_tensor(hasher: "hashlib._Hash", tensor: torch.Tensor) -> None:
    arr = tensor.detach().cpu().contiguous()
    hasher.update(str(arr.dtype).encode("utf-8"))
    hasher.update(str(tuple(arr.shape)).encode("utf-8"))
    hasher.update(arr.numpy().tobytes())


def compute_dataset_fingerprint(dataset: dict[str, Any]) -> str:
    hasher = hashlib.sha256()
    hasher.update(FINGERPRINT_SCHEMA_VERSION.encode("utf-8"))

    for key in sorted(dataset.keys()):
        if key == "metadata":
            continue
        hasher.update(key.encode("utf-8"))
        value = dataset[key]
        if isinstance(value, torch.Tensor):
            _update_with_tensor(hasher, value)
        else:
            _update_with_json(hasher, value)

    meta = dataset.get("metadata", {})
    semantic_meta = {key: meta.get(key) for key in _METADATA_KEYS if key in meta}
    _update_with_json(hasher, semantic_meta)
    return hasher.hexdigest()[:16]
