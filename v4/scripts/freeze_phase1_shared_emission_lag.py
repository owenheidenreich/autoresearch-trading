"""Freeze one conservative completed-minute lag from OPRA and ThetaData receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "pathd.shared-emission-lag.v1"


def _hash(payload: dict[str, Any]) -> str:
    material = dict(payload)
    material.pop("receipt_sha256", None)
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--databento-summary", type=Path, required=True)
    parser.add_argument("--theta-receipt", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite shared lag receipt: {args.output}")
    databento = json.loads(args.databento_summary.read_text())
    timing = databento.get("local_receipt_minus_ts_recv_ns", {})
    cbbo = [value for key, value in timing.items() if "CBBOMsg" in key and value]
    if not cbbo:
        raise SystemExit("Databento receipt has no CBBO timing distribution")
    databento_p99_ns = max(int(value["p99"]) for value in cbbo)
    theta_samples: list[dict[str, Any]] = []
    theta_sources = []
    for path in args.theta_receipt:
        receipt = json.loads(path.read_text())
        if (
            receipt.get("schema_version")
            != "autoresearch.thetadata-completed-minute-timing.v1"
            or receipt.get("errors")
            or receipt.get("secrets_recorded") is not False
        ):
            raise SystemExit(f"invalid ThetaData timing receipt: {path}")
        theta_samples.extend(receipt.get("samples", []))
        theta_sources.append({"path": str(path.resolve()), "sha256": _sha256(path)})
    if len(theta_samples) < 5:
        raise SystemExit(
            f"at least five completed ThetaData minute samples are required, got {len(theta_samples)}"
        )
    theta_lags = [int(row["receipt_lag_ns"]) for row in theta_samples]
    if any(value < 0 or value > 10_000_000_000 for value in theta_lags):
        raise SystemExit("ThetaData timing sample is outside the supported 0..10s range")
    theta_p99_ns = sorted(theta_lags)[math.ceil(0.99 * len(theta_lags)) - 1]
    shared_ns = max(databento_p99_ns, theta_p99_ns)
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "FROZEN",
        "clock_law": {
            "opra_cbbo_1m_boundary": "completed interval end t",
            "thetadata_bar": "event_time t-60s becomes complete when bar t first appears",
            "decision_emission": "t plus frozen_emission_lag_ms",
        },
        "databento_summary": {
            "path": str(args.databento_summary.resolve()),
            "sha256": _sha256(args.databento_summary),
            "cbbo_p99_ns": databento_p99_ns,
        },
        "theta_receipts": theta_sources,
        "theta_sample_count": len(theta_samples),
        "theta_p99_ns": theta_p99_ns,
        "frozen_emission_lag_ms": int(math.ceil(shared_ns / 1_000_000)),
        "broker_or_order_path": False,
        "protected_holdout_opened": False,
    }
    result["receipt_sha256"] = _hash(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
