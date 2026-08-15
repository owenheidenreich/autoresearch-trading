"""Build the frozen one-trade 120-minute action-value label dataset.

No model is fitted.  The builder verifies the pre-outcome declaration, the
settlement-complete candidate artifact, and its own implementation hash before
reading the declared outcome columns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v5.ops.build_causal_day_dataset import canonical_json, file_sha256
from v5.research.causal_day_action_advantage import (
    LABEL_NAME,
    REQUIRED_COLUMNS,
    label_one_session,
)


SCHEMA = "v5.causal-day-action-advantage.v1"
DECLARATION_SCHEMA = "v5.causal-day-action-advantage-declaration.v2"
QUANTILES = (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0)


def _verify_self_hash(payload: dict[str, Any], *, field: str = "receipt_sha256") -> None:
    expected = payload.get(field)
    unsigned = dict(payload)
    unsigned.pop(field, None)
    actual = hashlib.sha256(canonical_json(unsigned)).hexdigest()
    if expected != actual:
        raise RuntimeError(f"self-hash mismatch: {actual} != {expected}")


def _quantiles(values: pd.Series) -> dict[str, float]:
    got = values.quantile(QUANTILES)
    return {f"q{int(round(level * 100)):02d}": float(got.loc[level]) for level in QUANTILES}


def run(
    *,
    declaration_path: Path,
    out_root: Path,
    evidence_dir: Path,
) -> dict[str, Any]:
    if out_root.exists() or evidence_dir.exists():
        raise RuntimeError("refusing to overwrite action-value output or evidence")
    declaration = json.loads(declaration_path.read_text())
    _verify_self_hash(declaration)
    if declaration.get("schema_version") != DECLARATION_SCHEMA:
        raise RuntimeError("action-value declaration schema drift")
    if declaration.get("label", {}).get("name") != LABEL_NAME:
        raise RuntimeError("action-value label name drift")
    implementation = declaration.get("implementation_hashes", {})
    for relative, expected in implementation.items():
        path = Path(relative)
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"implementation hash mismatch: {relative}")

    candidate_info = declaration["immutable_input"]["candidates"]
    candidate_path = Path(candidate_info["path"])
    if not candidate_path.is_file() or file_sha256(candidate_path) != candidate_info["sha256"]:
        raise RuntimeError("candidate artifact is missing or hash-mismatched")
    receipt_info = declaration["immutable_input"]["dataset_receipt"]
    receipt_path = Path(receipt_info["path"])
    if not receipt_path.is_file() or file_sha256(receipt_path) != receipt_info["sha256"]:
        raise RuntimeError("dataset receipt is missing or hash-mismatched")
    dataset_receipt = json.loads(receipt_path.read_text())
    _verify_self_hash(dataset_receipt)
    if dataset_receipt.get("schema_version") != "v5.causal-day-dataset.v2":
        raise RuntimeError("settlement-complete dataset v2 is required")
    if dataset_receipt["outputs"]["candidates"]["sha256"] != candidate_info["sha256"]:
        raise RuntimeError("declaration candidate hash differs from dataset receipt")

    candidates = pd.read_parquet(candidate_path, columns=list(REQUIRED_COLUMNS))
    expected_rows = int(declaration["population"]["candidate_rows"])
    expected_sessions = int(declaration["population"]["sessions"])
    if len(candidates) != expected_rows or candidates["session"].astype(str).nunique() != expected_sessions:
        raise RuntimeError("declared action-value population drift")

    candidate_parts: list[pd.DataFrame] = []
    minute_parts: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for session, group in candidates.groupby("session", sort=True):
        labelled = label_one_session(group)
        candidate_parts.append(labelled.candidates)
        minute_parts.append(labelled.minutes)
        summaries.append(labelled.summary)
    labelled_candidates = pd.concat(candidate_parts, ignore_index=True)
    labelled_minutes = pd.concat(minute_parts, ignore_index=True)
    labelled_sessions = pd.DataFrame(summaries).sort_values("session").reset_index(drop=True)
    if len(labelled_candidates) != expected_rows:
        raise RuntimeError("label builder changed the candidate population")
    if len(labelled_minutes) != expected_sessions * 326:
        raise RuntimeError("label builder changed the decision-minute population")

    out_root.mkdir(parents=True, exist_ok=False)
    candidate_out = out_root / "candidate_action_values.parquet"
    minute_out = out_root / "minute_wait_values.parquet"
    session_out = out_root / "session_oracle_summary.parquet"
    labelled_candidates.to_parquet(candidate_out, index=False)
    labelled_minutes.to_parquet(minute_out, index=False)
    labelled_sessions.to_parquet(session_out, index=False)

    time_counts = (
        labelled_sessions["oracle_time_band"].fillna("no_trade").value_counts().sort_index()
    )
    right_counts = labelled_sessions["oracle_right"].fillna("no_trade").value_counts().sort_index()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "created_on": "2026-08-14",
        "purpose": "pre-fit one-trade Q(enter) versus Q(wait) labels; no model or operating point",
        "declaration": {"path": str(declaration_path), "sha256": file_sha256(declaration_path)},
        "immutable_input": declaration["immutable_input"],
        "population": {
            "sessions": int(labelled_sessions["session"].nunique()),
            "decision_minutes": int(len(labelled_minutes)),
            "candidate_rows": int(len(labelled_candidates)),
            "first_session": str(labelled_sessions["session"].min()),
            "last_session": str(labelled_sessions["session"].max()),
        },
        "integrity": {
            "model_fit": False,
            "selector_fit": False,
            "reserved_sessions_used": False,
            "candidate_population_preserved": True,
            "complete_326_minute_clock_each_session": True,
            "minutes_without_eligible_action": int(
                labelled_sessions["minutes_without_eligible_action"].sum()
            ),
            "wait_only_minutes_are_explicit": True,
            "q_wait_excludes_current_minute": True,
            "q_wait_nonincreasing_each_session": True,
            "primary_oracle_actions": int(labelled_candidates["is_primary_oracle_action"].sum()),
            "sessions_with_positive_oracle": int(
                labelled_sessions["global_oracle_bid_120m_usd"].gt(0.0).sum()
            ),
        },
        "declared_diagnostics": {
            "q_enter_bid_quantiles": _quantiles(labelled_candidates["q_enter_bid_120m_usd"]),
            "q_wait_bid_quantiles": _quantiles(labelled_minutes["q_wait_bid_120m_usd"]),
            "a_enter_bid_quantiles": _quantiles(labelled_candidates["a_enter_bid_120m_usd"]),
            "q_enter_mid_quantiles": _quantiles(labelled_candidates["q_enter_mid_120m_usd"]),
            "session_oracle_bid_quantiles": _quantiles(
                labelled_sessions["global_oracle_bid_120m_usd"]
            ),
            "session_oracle_mid_quantiles": _quantiles(
                labelled_sessions["global_oracle_mid_120m_usd"]
            ),
            "positive_q_enter_bid_rows": int(labelled_candidates["q_enter_bid_120m_usd"].gt(0.0).sum()),
            "positive_a_enter_bid_rows": int(labelled_candidates["a_enter_bid_120m_usd"].gt(0.0).sum()),
            "positive_best_advantage_minutes": int(
                labelled_minutes["best_a_enter_bid_120m_usd"].gt(0.0).sum()
            ),
            "oracle_time_band_counts": {str(key): int(value) for key, value in time_counts.items()},
            "oracle_right_counts": {str(key): int(value) for key, value in right_counts.items()},
        },
        "artifacts": {
            "candidate_action_values": {
                "path": str(candidate_out),
                "rows": len(labelled_candidates),
                "sha256": file_sha256(candidate_out),
            },
            "minute_wait_values": {
                "path": str(minute_out),
                "rows": len(labelled_minutes),
                "sha256": file_sha256(minute_out),
            },
            "session_oracle_summary": {
                "path": str(session_out),
                "rows": len(labelled_sessions),
                "sha256": file_sha256(session_out),
            },
        },
        "implementation_hashes": implementation,
        "fit_gate_status": "NOT_CALLED_FOR_FIT; LABEL_BUILD_ONLY",
    }
    payload["receipt_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    evidence_dir.mkdir(parents=True, exist_ok=False)
    receipt_path_out = evidence_dir / "receipt.json"
    receipt_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    receipt_path_out.write_text(receipt_text)
    (out_root / "receipt.json").write_text(receipt_text)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = run(
        declaration_path=args.declaration,
        out_root=args.out_root,
        evidence_dir=args.evidence_dir,
    )
    print(json.dumps({"receipt_sha256": payload["receipt_sha256"], "population": payload["population"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
