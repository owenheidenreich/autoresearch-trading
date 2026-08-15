from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from v5.ops.build_causal_day_dataset import canonical_json, file_sha256
from v5.research.causal_day_fit_cache import (
    CachedSession,
    FeatureScaler,
    collate_minutes,
    verify_cache_index,
)


def test_collation_preserves_prefix_chain_actions_and_contract_keys() -> None:
    cached = CachedSession(
        session="2025-01-02",
        minutes=np.asarray(["09:35", "12:46"]),
        candle_lengths=np.asarray([2, 3]),
        candles=np.arange(3 * 18, dtype=np.float32).reshape(3, 18),
        ladder_offsets=np.asarray([0, 2, 5]),
        ladder=np.arange(5 * 23, dtype=np.float32).reshape(5, 23),
        action_mask=np.asarray([True, False, False, True, True]),
        targets=np.arange(5 * 3, dtype=np.float32).reshape(5, 3),
        contract_ids=np.asarray(["a", "b", "c", "d", "e"]),
    )
    scaler = FeatureScaler(
        np.zeros(18), np.ones(18), np.zeros(23), np.ones(23)
    )
    batch, targets, keys = collate_minutes(cached, np.asarray([0, 1]), scaler)
    assert batch.candles.shape == (2, 3, 18)
    assert batch.candle_mask.sum(dim=1).tolist() == [2, 3]
    assert batch.ladder_mask.sum(dim=1).tolist() == [2, 3]
    assert batch.entry_action_mask.sum(dim=1).tolist() == [1, 2]
    assert batch.roles == ("morning_entry", "afternoon_entry")
    assert keys == [
        ("2025-01-02", "09:35", "a"),
        ("2025-01-02", "12:46", "d"),
        ("2025-01-02", "12:46", "e"),
    ]
    assert targets.shape == (2, 3, 3)


def test_cache_verifier_binds_receipt_manifest_and_session_bytes(tmp_path) -> None:
    root = tmp_path / "cache"
    session_path = root / "sessions" / "2025-01-02.npz"
    session_path.parent.mkdir(parents=True)
    np.savez_compressed(session_path, value=np.asarray([1, 2, 3]))
    manifest = pd.DataFrame(
        [
            {
                "session": "2025-01-02",
                "path": str(session_path),
                "sha256": file_sha256(session_path),
                "minutes": 1,
                "whole_chain_nodes": 2,
                "eligible_actions": 1,
            }
        ]
    )
    manifest_path = root / "cache_manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    receipt = {
        "schema_version": "v5.causal-day-fit-cache.v1",
        "sessions": 1,
        "minutes": 1,
        "whole_chain_nodes": 2,
        "eligible_actions": 1,
        "manifest": {
            "path": str(manifest_path),
            "sha256": file_sha256(manifest_path),
        },
    }
    receipt["receipt_sha256"] = hashlib.sha256(canonical_json(receipt)).hexdigest()
    (root / "receipt.json").write_text(json.dumps(receipt))
    got, got_manifest = verify_cache_index(root)
    assert got["sessions"] == 1
    assert got_manifest["session"].tolist() == ["2025-01-02"]
