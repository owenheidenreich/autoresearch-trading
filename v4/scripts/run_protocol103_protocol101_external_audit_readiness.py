"""Protocol 103: Protocol 101 external-audit readiness and data request.

This protocol does not train a model, score trades, download paid data, or use
broker/live endpoints. It answers a narrower question:

Can the already-collected Q4 2024 block be scored as a frozen Protocol 101
external audit right now?

The answer is intentionally explicit. Q4 2024 has clean official-context raw
and processed data, but the current Protocol 101 scoring stack needs a
Protocol 092-style candidate table whose candidate outcomes come from frozen
Protocol 081 exits. That table does not exist for Q4 2024 yet.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_103_protocol101_external_audit_readiness")
PROTOCOL101_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy")
PROTOCOL092_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_092_serial_opportunity_policy")
PROTOCOL081_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_081_q4start_residual_sequence_deterministic_artifacts")
PROTOCOL077_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_077_q4start_lifecycle_sequence_dataset")
PROTOCOL076_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_076_q4_2024_prehistory_protocol054")
Q4_2024_INTEGRITY = Path("v4/audit/q4_2024_data_integrity_summary.json")
Q4_2024_BUILD = Path("v4/audit/q4_2024_neural_build_summary.json")
LEDGER_HINT = Path("v4/ledger/RESEARCH_LEDGER.md")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    checks = _readiness_checks()
    request = _data_request()
    payload = {
        "protocol": "103_protocol101_external_audit_readiness",
        "paid_data_downloaded": False,
        "live_orders": False,
        "decision": _decision(checks),
        "checks": checks,
        "next_paid_data_request": request,
        "implementation_next_steps": _implementation_next_steps(),
    }
    (OUT_DIR / "summary.json").write_text(_json_dumps(payload))
    _write_report(OUT_DIR / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "report": str(OUT_DIR / "report.md")}, indent=2))
    return 0


def _readiness_checks() -> dict[str, Any]:
    q4_integrity = _load_json(Q4_2024_INTEGRITY)
    q4_build = _load_json(Q4_2024_BUILD)
    protocol092_dataset = PROTOCOL092_DIR / "serial_opportunity_dataset.parquet"
    protocol081_selected = PROTOCOL081_DIR / "selected_trades_sequence_exits.json"
    protocol077_trades = PROTOCOL077_DIR / "protocol054_lifecycle_trades.parquet"
    protocol076_selected = PROTOCOL076_DIR / "selected_trades_with_lifecycle_exits.json"

    protocol092_splits = _parquet_splits(protocol092_dataset)
    protocol077_splits = _parquet_splits(protocol077_trades)
    protocol081_splits = _json_splits(protocol081_selected)
    protocol076_splits = _json_splits(protocol076_selected)

    q4_integrity_status = q4_integrity.get("status")
    q4_sessions = int(q4_integrity.get("sessions_built", 0) or 0)
    q4_processed_dir = q4_build.get("processed_dir")
    q4_processed_files = _processed_file_count(Path(q4_processed_dir)) if q4_processed_dir else 0

    protocol101_artifacts = sorted(PROTOCOL101_DIR.glob("model_artifacts/*/seed_*/manifest.json"))
    return {
        "q4_2024_official_context_available": q4_integrity_status == "pass" and q4_sessions > 0,
        "q4_2024_integrity_status": q4_integrity_status,
        "q4_2024_sessions_built": q4_sessions,
        "q4_2024_processed_dir": q4_processed_dir,
        "q4_2024_processed_files": q4_processed_files,
        "protocol101_manifest_count": len(protocol101_artifacts),
        "protocol092_candidate_dataset_exists": protocol092_dataset.exists(),
        "protocol092_splits": protocol092_splits,
        "protocol092_has_q4_2024": "q4_2024" in protocol092_splits,
        "protocol081_selected_exits_exists": protocol081_selected.exists(),
        "protocol081_splits": protocol081_splits,
        "protocol081_has_q4_2024": "q4_2024" in protocol081_splits,
        "protocol077_lifecycle_trades_exists": protocol077_trades.exists(),
        "protocol077_splits": protocol077_splits,
        "protocol077_has_q4_2024": "q4_2024" in protocol077_splits,
        "protocol076_prehistory_selected_exists": protocol076_selected.exists(),
        "protocol076_splits": protocol076_splits,
        "protocol076_has_q4_2024_test_rows": "q4_2024" in protocol076_splits,
        "can_score_protocol101_on_q4_2024_now": False,
        "blocking_reason": (
            "Q4 2024 has clean official-context data, but no Protocol 092-style "
            "candidate dataset exists for q4_2024 and no frozen Protocol 081 "
            "candidate exits exist for q4_2024. Protocol 076 used Q4 2024 as "
            "prehistory and does not provide Q4 2024 test rows."
        ),
    }


def _data_request() -> dict[str, Any]:
    costs = _historical_databento_costs()
    return {
        "status": "approval_required_before_download",
        "purpose": (
            "After the Q4 2024 external-audit candidate builder exists, download one "
            "additional adjacent locked quarter to test whether Protocol 101's thin "
            "Q4 2025 edge is sample-specific before buying years of data."
        ),
        "source": "Databento + ThetaData",
        "date_range": "2024-07-01 through 2024-09-30",
        "databento_dataset": "OPRA.PILLAR",
        "databento_schemas": ["definition", "cbbo-1m", "ohlcv-1m", "statistics"],
        "databento_symbols": "SPXW.OPT parent definitions, then filtered SPXW 0DTE raw symbols",
        "thetadata_products": ["SPX 1-minute index bars", "VIX 1-minute index bars"],
        "estimated_databento_cost_usd": {
            "basis": "recent collected quarter estimates from local audit logs",
            "q4_2024_actual_estimate": costs.get("databento_q4_2024_downloads.jsonl"),
            "q3_2025_actual_estimate": costs.get("databento_q3_2025_downloads.jsonl"),
            "q4_2025_actual_estimate": costs.get("databento_q4_2025_downloads.jsonl"),
            "expected_range": [25.0, 36.0],
        },
        "hard_spend_cap_usd": 45.0,
        "why_this_batch": (
            "Q3 2024 is adjacent to the already-collected Q4 2024 block, extends "
            "the continuous history backward without jumping around, and gives a "
            "new locked market regime for Protocol 101 before committing to the "
            "full 2022-present purchase."
        ),
        "approval_required_text": (
            "Before running any Databento get_range or ThetaData historical download "
            "for this batch, ask the user to approve this exact source/date/schema/"
            "symbol/cap request."
        ),
    }


def _implementation_next_steps() -> list[str]:
    return [
        "Build a no-training Q4 2024 candidate builder that applies the frozen Protocol 051 entry stack to Q4 2024 processed sessions.",
        "Build Q4 2024 same-contract lifecycle paths from v4/normalized_official_context using executable bid/ask data.",
        "Apply frozen Protocol 081 lifecycle artifacts to those candidate paths to produce Protocol 081-compatible candidate_exit_time, candidate_exit_reason, and candidate_pnl fields.",
        "Create a Protocol 092-compatible candidate table with split=q4_2024_external and preserve entry-only feature columns.",
        "Score the Q4 2024 candidate table with frozen Protocol 101 artifacts and label the result as a temporal-regime stress audit, not a chronological promotion gate.",
        "Only after that local audit path works, ask for explicit approval to download Q3 2024 as the next locked paid batch.",
    ]


def _decision(checks: dict[str, Any]) -> str:
    if checks["can_score_protocol101_on_q4_2024_now"]:
        return "ready_to_score_q4_2024_external_audit"
    return "blocked_missing_protocol101_q4_2024_candidate_outcomes"


def _historical_databento_costs() -> dict[str, float]:
    out: dict[str, float] = {}
    for name in [
        "databento_q4_2024_downloads.jsonl",
        "databento_q3_2025_downloads.jsonl",
        "databento_q4_2025_downloads.jsonl",
        "databento_q2_2025_downloads.jsonl",
    ]:
        path = Path("v4/audit") / name
        if not path.exists():
            continue
        total = 0.0
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            total += float(json.loads(line).get("cost_estimate_usd", 0.0))
        out[name] = round(total, 4)
    return out


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _parquet_splits(path: Path) -> list[str]:
    if not path.exists():
        return []
    frame = pd.read_parquet(path, columns=["split"])
    return sorted(str(item) for item in frame["split"].dropna().unique().tolist())


def _json_splits(path: Path) -> list[str]:
    if not path.exists():
        return []
    rows = json.loads(path.read_text())
    return sorted({str(row.get("split")) for row in rows if row.get("split") is not None})


def _processed_file_count(path: Path) -> int:
    if not path.exists():
        return 0
    return len(list(path.glob("*.pkl")))


def _json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    checks = payload["checks"]
    request = payload["next_paid_data_request"]
    lines = [
        "# Protocol 103: Protocol 101 External-Audit Readiness",
        "",
        "No paid market data was downloaded. No live broker data or order endpoint was used.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Q4 2024 official-context integrity: `{checks['q4_2024_integrity_status']}`",
        f"- Q4 2024 sessions built: `{checks['q4_2024_sessions_built']}`",
        f"- Protocol 101 artifact manifests: `{checks['protocol101_manifest_count']}`",
        "",
        "## Readiness Checks",
        "",
        f"- Protocol 092 candidate dataset has q4_2024: `{checks['protocol092_has_q4_2024']}`",
        f"- Protocol 081 selected exits have q4_2024: `{checks['protocol081_has_q4_2024']}`",
        f"- Protocol 077 lifecycle trades have q4_2024: `{checks['protocol077_has_q4_2024']}`",
        f"- Protocol 076 prehistory selected file has Q4 2024 test rows: `{checks['protocol076_has_q4_2024_test_rows']}`",
        "",
        "## Blocking Reason",
        "",
        checks["blocking_reason"],
        "",
        "## Build Instructions",
        "",
    ]
    lines.extend(f"{idx}. {item}" for idx, item in enumerate(payload["implementation_next_steps"], start=1))
    lines += [
        "",
        "## Next Paid Data Request",
        "",
        f"- Source: `{request['source']}`",
        f"- Date range: `{request['date_range']}`",
        f"- Databento dataset: `{request['databento_dataset']}`",
        f"- Databento schemas: `{', '.join(request['databento_schemas'])}`",
        f"- Databento symbols: `{request['databento_symbols']}`",
        f"- ThetaData products: `{', '.join(request['thetadata_products'])}`",
        f"- Estimated Databento cost: `${request['estimated_databento_cost_usd']['expected_range'][0]:.2f}` to `${request['estimated_databento_cost_usd']['expected_range'][1]:.2f}`",
        f"- Hard spend cap: `${request['hard_spend_cap_usd']:.2f}`",
        "",
        request["why_this_batch"],
        "",
        "**Approval required before any paid download.**",
    ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
