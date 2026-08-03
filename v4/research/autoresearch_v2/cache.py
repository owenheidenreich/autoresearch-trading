"""OOF prediction cache keyed independently of action thresholds."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def prediction_cache_key(
    *,
    features: Sequence[str],
    target: Mapping[str, Any],
    model: Mapping[str, Any],
    fold: Mapping[str, Any],
    seed: int,
    foundation_hash: str,
) -> str:
    # Deliberately no threshold field: threshold screens reuse one OOF fit.
    return stable_hash(
        {
            "features": list(features),
            "target": dict(target),
            "model": dict(model),
            "fold": dict(fold),
            "seed": int(seed),
            "foundation_hash": foundation_hash,
        }
    )


class PredictionCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def load(self, key: str, *, row_hash: str) -> tuple[np.ndarray, Any] | None:
        manifest_path = self.root / f"{key}.json"
        values_path = self.root / f"{key}.npz"
        model_path = self.root / f"{key}.model.pkl"
        if not manifest_path.exists() or not values_path.exists() or not model_path.exists():
            return None
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("schema_version") != "autoresearch_v2.oof_cache.v2"
            or manifest.get("row_hash") != row_hash
            or manifest.get("mutate_future_invariance_status") != "pass"
        ):
            return None
        with np.load(values_path, allow_pickle=False) as values:
            predictions = np.asarray(values["predictions"], dtype=float)
        if len(predictions) != int(manifest["row_count"]):
            return None
        with model_path.open("rb") as handle:
            model = pickle.load(handle)
        return predictions, model

    def store(
        self,
        key: str,
        values: np.ndarray,
        *,
        row_hash: str,
        model: Any,
        mutate_future_invariance_status: str,
    ) -> None:
        if mutate_future_invariance_status != "pass":
            raise ValueError("only future-invariant OOF predictions may enter cache")
        self.root.mkdir(parents=True, exist_ok=True)
        predictions = np.asarray(values, dtype=float)
        np.savez_compressed(self.root / f"{key}.npz", predictions=predictions)
        with (self.root / f"{key}.model.pkl").open("wb") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        (self.root / f"{key}.json").write_text(
            json.dumps(
                {
                    "schema_version": "autoresearch_v2.oof_cache.v2",
                    "key": key,
                    "row_hash": row_hash,
                    "row_count": len(predictions),
                    "mutate_future_invariance_status": mutate_future_invariance_status,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
