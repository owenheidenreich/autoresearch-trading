"""Fail-closed multi-session arrival receipts for Phase-0b Track A.

This module reads already captured files only.  It has no Databento client,
broker, order, model, or holdout imports and cannot initiate a connection.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from v4.research.pathd_feature_admission_ledger import REPO_ROOT, sha256_file
from v4.research.pathd_phase0b_tracka_capture_plan import (
    DECLARATION_PATH,
    DECLARED_SESSIONS,
    EXPECTED_SYMBOL_COUNT,
    SCHEMAS,
    verify_declaration,
)
from v4.scripts.capture_databento_live_opra_training_twin import (
    LOCAL_RECEIPT_SCHEMA,
    _stable_hash,
)


FAMILY_RTYPES = {
    "entry.opra_cbbo1m_native.v1": 193,
    "entry.opra_cbbo1s_rolling.v1": 192,
    "entry.opra_ohlcv1m_sparse.v1": 33,
}
ARRIVAL_CLOCK_KIND = "measured_upstream_arrival_p99"
RECEIPT_SCHEMA = "pathd.phase0b.tracka-multisession-arrival-receipt.v1"


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _resolve_artifact(path_value: str, *, summary_path: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    candidate = summary_path.parent / path
    return candidate if candidate.exists() else REPO_ROOT / path


def _p99_ms(values_ns: Iterable[int]) -> float:
    values = np.asarray(tuple(values_ns), dtype=np.int64)
    if not len(values):
        raise RuntimeError("arrival-latency distribution is empty")
    return round(float(np.quantile(values, 0.99, method="nearest") / 1_000_000.0), 6)


def verify_session_capture(summary_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("summary_sha256") != _stable_hash(summary):
        raise RuntimeError(f"capture summary signature mismatch:{summary_path}")
    if summary.get("status") != "CAPTURED_NO_ORDER_LIVE_SAMPLE":
        raise RuntimeError(f"capture did not complete successfully:{summary_path}")
    plan = summary.get("plan", {})
    session = str(plan.get("session_date"))
    if session not in {item.isoformat() for item in DECLARED_SESSIONS}:
        raise RuntimeError(f"capture session was not declared:{session}")
    if int(plan.get("symbol_count", -1)) != EXPECTED_SYMBOL_COUNT:
        raise RuntimeError(f"capture universe was not exactly {EXPECTED_SYMBOL_COUNT}:{session}")
    if tuple(plan.get("schemas", ())) != SCHEMAS:
        raise RuntimeError(f"capture schema scope drift:{session}")
    if plan.get("network") is not True or plan.get("broker_or_order_path") is not False:
        raise RuntimeError(f"capture provenance flags invalid:{session}")
    hard_stops = summary.get("hard_stops", {})
    expected_false = (
        "broker_accessed",
        "order_path_accessed",
        "paper_runtime_accessed",
        "model_loaded_or_fit",
        "promotion_or_default_changed",
    )
    if any(hard_stops.get(name) is not False for name in expected_false):
        raise RuntimeError(f"capture hard stop violated:{session}")
    if hard_stops.get("holdout_open_count") != 0:
        raise RuntimeError(f"capture holdout firewall violated:{session}")

    raw = summary.get("raw_dbn", {})
    raw_path = _resolve_artifact(str(raw.get("path", "")), summary_path=summary_path)
    if not raw_path.is_file() or sha256_file(raw_path) != raw.get("sha256"):
        raise RuntimeError(f"capture raw DBN hash mismatch:{session}")
    receipt_meta = summary.get("local_receipts", {})
    if receipt_meta.get("schema_version") != LOCAL_RECEIPT_SCHEMA:
        raise RuntimeError(f"per-message local receipt metadata missing:{session}")
    receipt_path = _resolve_artifact(str(receipt_meta.get("path", "")), summary_path=summary_path)
    if not receipt_path.is_file() or sha256_file(receipt_path) != receipt_meta.get("sha256"):
        raise RuntimeError(f"local receipt sidecar hash mismatch:{session}")

    lags_by_rtype: dict[int, list[int]] = defaultdict(list)
    row_count = 0
    with receipt_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if row.get("schema_version") != LOCAL_RECEIPT_SCHEMA:
                raise RuntimeError(f"local receipt row schema drift:{session}:{row_count}")
            if row.get("sequence_index") != row_count:
                raise RuntimeError(f"local receipt sequence drift:{session}:{row_count}")
            local_ns = row.get("local_receipt_unix_ns")
            interval_end_ns = row.get("interval_end_unix_ns")
            rtype = row.get("rtype")
            if (
                isinstance(rtype, int)
                and isinstance(local_ns, int)
                and isinstance(interval_end_ns, int)
                and interval_end_ns > 0
                and local_ns >= interval_end_ns
            ):
                lags_by_rtype[rtype].append(local_ns - interval_end_ns)
            row_count += 1
    if row_count != int(receipt_meta.get("rows", -1)) or row_count != int(summary.get("records_total", -2)):
        raise RuntimeError(f"per-message local receipt row count drift:{session}")
    return {
        "session": session,
        "summary_path": _portable(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "summary_semantic_sha256": summary["summary_sha256"],
        "raw_path": _portable(raw_path),
        "raw_sha256": raw["sha256"],
        "local_receipt_path": _portable(receipt_path),
        "local_receipt_sha256": receipt_meta["sha256"],
        "local_receipt_rows": row_count,
        "lags_by_rtype": dict(lags_by_rtype),
    }


def build_multisession_arrival_receipt(
    summary_paths: Sequence[Path],
    *,
    declaration_path: Path = DECLARATION_PATH,
) -> dict[str, Any]:
    declaration = verify_declaration(declaration_path)
    sessions = [verify_session_capture(path) for path in summary_paths]
    observed = tuple(item["session"] for item in sessions)
    expected = tuple(item.isoformat() for item in DECLARED_SESSIONS)
    if observed != expected:
        raise RuntimeError(f"capture sessions must match declared order:{observed!r}")

    families: dict[str, Any] = {}
    for contract_id, rtype in FAMILY_RTYPES.items():
        per_session = []
        pooled: list[int] = []
        for evidence in sessions:
            values = evidence["lags_by_rtype"].get(rtype, [])
            if not values:
                raise RuntimeError(f"missing family records:{contract_id}:{evidence['session']}")
            pooled.extend(values)
            per_session.append(
                {
                    "session": evidence["session"],
                    "message_count": len(values),
                    "arrival_p99_ms": _p99_ms(values),
                }
            )
        families[contract_id] = {
            "status": "PASS_MULTISESSION_ARRIVAL_DISTRIBUTION",
            "source_rtype": rtype,
            "session_count": len(per_session),
            "message_count": len(pooled),
            "availability_clock_ms": _p99_ms(pooled),
            "availability_clock_kind": ARRIVAL_CLOCK_KIND,
            "per_session": per_session,
        }

    evidence_rows = [
        {key: value for key, value in item.items() if key != "lags_by_rtype"}
        for item in sessions
    ]
    payload: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "PASS",
        "declaration_path": _portable(declaration_path),
        "declaration_sha256": declaration["declaration_sha256"],
        "sessions": list(expected),
        "session_count": len(expected),
        "per_message_local_receipts_verified": True,
        "family_distributions": families,
        "capture_evidence": evidence_rows,
        "clock_inheritance_used": False,
        "theta_2336ms_used": False,
        "definition_30603_667ms_used": False,
        "prior_319_5ms_used_as_measurement": False,
        "hard_stops": declaration["hard_stops"],
    }
    payload["receipt_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return payload


def write_multisession_arrival_receipt(
    summary_paths: Sequence[Path], output_path: Path, *, declaration_path: Path = DECLARATION_PATH
) -> dict[str, Any]:
    payload = build_multisession_arrival_receipt(summary_paths, declaration_path=declaration_path)
    if output_path.exists():
        raise FileExistsError(f"arrival receipt already exists:{output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
