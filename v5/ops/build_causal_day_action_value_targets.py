"""Align action-value labels to the immutable whole-chain feature cache."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v5.ops.build_causal_day_dataset import canonical_json, file_sha256
from v5.research.causal_day_action_value_targets import target_path
from v5.research.causal_day_fit_cache import load_cached_session, verify_cache_index


SCHEMA = "v5.causal-day-action-value-target-cache.v1"


def _verify_receipt(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    expected = payload.get("receipt_sha256")
    unsigned = dict(payload)
    unsigned.pop("receipt_sha256", None)
    if hashlib.sha256(canonical_json(unsigned)).hexdigest() != expected:
        raise RuntimeError(f"receipt self-hash mismatch: {path}")
    return payload


def run(
    *,
    feature_cache_root: Path,
    action_value_receipt_path: Path,
    out_root: Path,
    evidence_dir: Path,
) -> dict[str, Any]:
    if out_root.exists() or evidence_dir.exists():
        raise RuntimeError("refusing to overwrite target cache or evidence")
    feature_receipt, feature_manifest = verify_cache_index(feature_cache_root)
    action_receipt = _verify_receipt(action_value_receipt_path)
    if action_receipt.get("schema_version") != "v5.causal-day-action-advantage.v1":
        raise RuntimeError("action-value receipt schema drift")
    artifacts = action_receipt["artifacts"]
    candidate_path = Path(artifacts["candidate_action_values"]["path"])
    minute_path = Path(artifacts["minute_wait_values"]["path"])
    for path, key in ((candidate_path, "candidate_action_values"), (minute_path, "minute_wait_values")):
        if not path.is_file() or file_sha256(path) != artifacts[key]["sha256"]:
            raise RuntimeError(f"action-value artifact hash mismatch: {key}")
    candidates = pd.read_parquet(
        candidate_path,
        columns=["session", "entry_minute", "contract_id", "q_enter_bid_120m_usd"],
    )
    minutes = pd.read_parquet(
        minute_path,
        columns=["session", "entry_minute", "q_wait_bid_120m_usd"],
    )
    if candidates.duplicated(["session", "entry_minute", "contract_id"]).any():
        raise RuntimeError("candidate action-value key is duplicated")
    if minutes.duplicated(["session", "entry_minute"]).any():
        raise RuntimeError("minute WAIT key is duplicated")

    out_root.mkdir(parents=True, exist_ok=False)
    rows = []
    for row in feature_manifest.sort_values("session").itertuples(index=False):
        cached = load_cached_session(Path(str(row.path)))
        session_candidates = candidates[candidates["session"].astype(str).eq(cached.session)]
        candidate_key = session_candidates.set_index(["entry_minute", "contract_id"])[
            "q_enter_bid_120m_usd"
        ]
        q_enter = np.full(len(cached.ladder), np.nan, dtype=np.float32)
        cursor = 0
        for minute_index, minute in enumerate(cached.minutes.astype(str)):
            start = int(cached.ladder_offsets[minute_index])
            stop = int(cached.ladder_offsets[minute_index + 1])
            count = stop - start
            ids = cached.contract_ids[start:stop].astype(str)
            keys = pd.MultiIndex.from_arrays(
                [np.repeat(str(minute), count), ids], names=["entry_minute", "contract_id"]
            )
            values = candidate_key.reindex(keys).to_numpy(float)
            action = cached.action_mask[start:stop]
            if not np.isfinite(values[action]).all() or np.isfinite(values[~action]).any():
                raise RuntimeError(f"{cached.session}: candidate target/cache alignment failed")
            q_enter[start:stop] = values.astype(np.float32)
            cursor += int(action.sum())
        session_minutes = minutes[minutes["session"].astype(str).eq(cached.session)].set_index(
            "entry_minute"
        )["q_wait_bid_120m_usd"]
        q_wait = session_minutes.reindex(cached.minutes.astype(str)).to_numpy(np.float32)
        if not np.isfinite(q_wait).all() or len(q_wait) != cached.minute_count:
            raise RuntimeError(f"{cached.session}: WAIT target/cache alignment failed")
        path = target_path(out_root, cached.session)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            session=np.asarray(cached.session),
            q_enter_bid_usd=q_enter,
            q_wait_bid_usd=q_wait,
        )
        rows.append(
            {
                "session": cached.session,
                "path": str(path),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
                "decision_minutes": len(q_wait),
                "eligible_actions": cursor,
                "whole_chain_nodes": len(q_enter),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest_path = out_root / "target_manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "created_on": "2026-08-14",
        "purpose": "deterministic alignment of frozen Q(enter)/Q(wait) labels to whole-chain feature rows",
        "model_fit": False,
        "feature_cache": {
            "path": str(feature_cache_root),
            "receipt_sha256": feature_receipt["receipt_sha256"],
        },
        "action_values": {
            "receipt_path": str(action_value_receipt_path),
            "receipt_file_sha256": file_sha256(action_value_receipt_path),
            "receipt_sha256": action_receipt["receipt_sha256"],
        },
        "population": {
            "sessions": len(manifest),
            "decision_minutes": int(manifest["decision_minutes"].sum()),
            "eligible_actions": int(manifest["eligible_actions"].sum()),
            "whole_chain_nodes": int(manifest["whole_chain_nodes"].sum()),
        },
        "integrity": {
            "one_target_per_eligible_action": True,
            "no_target_on_ineligible_ladder_node": True,
            "one_wait_value_per_decision_minute": True,
            "feature_rows_changed": False,
        },
        "manifest": {
            "path": str(manifest_path),
            "rows": len(manifest),
            "sha256": file_sha256(manifest_path),
        },
        "implementation_sha256": file_sha256(Path(__file__)),
    }
    payload["receipt_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    receipt_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    (out_root / "receipt.json").write_text(receipt_text)
    evidence_dir.mkdir(parents=True, exist_ok=False)
    (evidence_dir / "receipt.json").write_text(receipt_text)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache-root", type=Path, required=True)
    parser.add_argument("--action-value-receipt", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = run(
        feature_cache_root=args.feature_cache_root,
        action_value_receipt_path=args.action_value_receipt,
        out_root=args.out_root,
        evidence_dir=args.evidence_dir,
    )
    print(json.dumps({"receipt_sha256": payload["receipt_sha256"], "population": payload["population"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
