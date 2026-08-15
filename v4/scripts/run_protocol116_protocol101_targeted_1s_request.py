"""Protocol 116: approval-ready targeted high-resolution request for Protocol 101.

This is a no-download preflight. It uses Protocol 115's replay rows to identify
the exact Protocol 101 selected contracts that still lack one-second coverage,
optionally asks Databento's non-billable metadata endpoint for cost estimates,
and writes an approval manifest. It never calls ``timeseries.get_range``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_116_protocol101_targeted_1s_request")
REPLAY_JSON = Path("v4/audit/autoresearch/v4_aplus_hypothesis_115_protocol101_existing_1s_path_audit/report.json")
MANIFEST_PATH = Path("v4/promotion/PROTOCOL_101_TARGETED_CBBO_1S_DOWNLOAD_MANIFEST.json")
LEDGER = Path("v4/ledger/RESEARCH_LEDGER.md")
DATASET = "OPRA.PILLAR"
CBBO_1S_AVAILABLE_FROM = "2025-02-20"


@dataclass(frozen=True)
class SessionRequest:
    session: str
    schema: str
    splits: list[str]
    symbols: int
    selected_trades: int
    missing_1s_session_rows: int
    missing_symbol_rows: int
    existing_1s_file: bool
    estimated_cost_usd: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-json", type=Path, default=REPLAY_JSON)
    parser.add_argument("--cbbo-1s-dir", type=Path, default=Path("data/raw/audit/opra_spxw_cbbo_1s"))
    parser.add_argument("--env-file", type=Path, default=Path("v4/.env"))
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--manifest-path", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--no-ledger", action="store_true")
    parser.add_argument(
        "--scope",
        choices=("all_missing", "critical_missing"),
        default="all_missing",
        help="all_missing covers every Protocol 101 unaudited selected symbol; critical_missing covers q4_2024/q3/q4_2025 only.",
    )
    parser.add_argument("--skip-metadata-estimate", action="store_true")
    parser.add_argument("--hard-cap", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = pd.DataFrame(json.loads(args.replay_json.read_text())["rows"])
    missing = missing_rows(rows, scope=args.scope)
    fallback_rates = {
        "cbbo-1s": historical_selected_1s_cost_per_symbol_session(),
        "cmbp-1": 0.0110,
    }
    metadata_status = "skipped"
    estimates: dict[str, float] = {}
    if not args.skip_metadata_estimate:
        _load_env_file(args.env_file)
        try:
            client = _client()
            estimates = estimate_sessions(client, missing)
            metadata_status = "ok"
        except Exception as exc:  # metadata is allowed, but the report should survive local auth/package issues.
            metadata_status = f"failed: {type(exc).__name__}: {exc}"

    requests = build_requests(
        missing,
        cbbo_1s_dir=args.cbbo_1s_dir,
        estimates=estimates,
        fallback_rates=fallback_rates,
    )
    estimated_total = float(sum(req.estimated_cost_usd or 0.0 for req in requests))
    hard_cap = float(args.hard_cap) if args.hard_cap is not None else default_hard_cap(estimated_total)
    decision = "ready_for_explicit_user_approval" if requests else "no_targeted_1s_request_needed"
    approval_text = approval_text_for(hard_cap=hard_cap)
    manifest = build_manifest(
        decision=decision,
        approval_text=approval_text,
        requests=requests,
        estimated_total=estimated_total,
        hard_cap=hard_cap,
        metadata_status=metadata_status,
        fallback_rates=fallback_rates,
        replay_json=args.replay_json,
        scope=args.scope,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json_dumps(manifest))
    (args.out_dir / "session_symbol_plan.csv").write_text(session_plan_csv(requests))
    write_request(args.out_dir / "request.md", manifest)
    args.manifest_path.write_text(json_dumps(manifest))
    if not args.no_ledger:
        append_ledger(args.ledger, manifest, args.out_dir / "request.md")
    print(json.dumps({"decision": decision, "estimated_total_usd": estimated_total, "hard_cap_usd": hard_cap, "request": str(args.out_dir / "request.md")}, indent=2, sort_keys=True))
    return 0


def missing_rows(rows: pd.DataFrame, *, scope: str) -> pd.DataFrame:
    rows = rows.copy()
    rows["split"] = rows["split"].astype(str)
    rows["session"] = rows["session"].astype(str)
    rows["raw_symbol"] = rows["raw_symbol"].astype(str)
    missing = rows[~rows["audit_status"].astype(str).eq("audited")].copy()
    if scope == "critical_missing":
        missing = missing[missing["split"].isin(["q4_2024_external", "q3_2025", "q4_2025"])].copy()
    missing = missing[missing["raw_symbol"].notna() & missing["raw_symbol"].ne("")].copy()
    return missing


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _client() -> Any:
    try:
        import databento as db
    except ImportError:
        print("databento is not installed; using fallback cost estimate", file=sys.stderr)
        raise
    return db.Historical()


def _bounds(session: str) -> tuple[str, str]:
    day = pd.Timestamp(session).date()
    start = datetime.combine(day, time(0, 0), tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return start.isoformat().replace("+00:00", "Z"), end.isoformat().replace("+00:00", "Z")


def estimate_sessions(client: Any, missing: pd.DataFrame) -> dict[str, float]:
    estimates: dict[str, float] = {}
    for session, group in missing.groupby("session", sort=True):
        start, end = _bounds(str(session))
        schema = schema_for_session(str(session))
        symbols = sorted(set(group["raw_symbol"].astype(str)))
        estimates[str(session)] = float(
            client.metadata.get_cost(
                dataset=DATASET,
                schema=schema,
                symbols=symbols,
                stype_in="raw_symbol",
                start=start,
                end=end,
            )
        )
    return estimates


def schema_for_session(session: str) -> str:
    if session < CBBO_1S_AVAILABLE_FROM:
        return "cmbp-1"
    return "cbbo-1s"


def historical_selected_1s_cost_per_symbol_session() -> float:
    path = Path("v4/audit/databento_cbbo_1s_selected_downloads.jsonl")
    if not path.exists():
        return 0.004
    total_cost = 0.0
    total_symbols = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        total_cost += float(row.get("cost_estimate_usd", 0.0) or 0.0)
        total_symbols += int(row.get("symbols", 0) or 0)
    if total_symbols <= 0:
        return 0.004
    return total_cost / total_symbols


def build_requests(
    missing: pd.DataFrame,
    *,
    cbbo_1s_dir: Path,
    estimates: dict[str, float],
    fallback_rates: dict[str, float],
) -> list[SessionRequest]:
    requests = []
    for session, group in missing.groupby("session", sort=True):
        symbols = sorted(set(group["raw_symbol"].astype(str)))
        schema = schema_for_session(str(session))
        existing = (cbbo_1s_dir / f"{session}.cbbo-1s.parquet").exists()
        estimate = estimates.get(str(session))
        if estimate is None:
            estimate = len(symbols) * fallback_rates.get(schema, 0.01)
        requests.append(
            SessionRequest(
                session=str(session),
                schema=schema,
                splits=sorted(set(group["split"].astype(str))),
                symbols=len(symbols),
                selected_trades=int(len(group)),
                missing_1s_session_rows=int(group["audit_status"].astype(str).eq("missing_1s_session").sum()),
                missing_symbol_rows=int(group["audit_status"].astype(str).eq("missing_symbol").sum()),
                existing_1s_file=bool(existing),
                estimated_cost_usd=float(estimate),
            )
        )
    return requests


def default_hard_cap(estimated_total: float) -> float:
    if estimated_total <= 0:
        return 10.0
    return float(math.ceil(max(10.0, estimated_total * 1.5)))


def approval_text_for(*, hard_cap: float) -> str:
    return (
        "I approve the Protocol 101 targeted high-resolution validation batch exactly as specified in "
        "v4/audit/autoresearch/v4_aplus_hypothesis_116_protocol101_targeted_1s_request/request.md: "
        "Databento OPRA.PILLAR cmbp-1/cbbo-1s for Protocol 101 selected SPXW raw symbols only, using the listed sessions, "
        f"with hard cap ${hard_cap:.2f}."
    )


def build_manifest(
    *,
    decision: str,
    approval_text: str,
    requests: list[SessionRequest],
    estimated_total: float,
    hard_cap: float,
    metadata_status: str,
    fallback_rates: dict[str, float],
    replay_json: Path,
    scope: str,
) -> dict[str, Any]:
    return {
        "protocol": "116_protocol101_targeted_1s_request",
        "paid_data_downloaded": False,
        "live_orders": False,
        "market_data_download_endpoint_called": False,
        "model_training": False,
        "decision": decision,
        "scope": scope,
        "why": (
            "Protocol 114 showed timing sensitivity, and Protocol 115 found supportive but incomplete one-second evidence. "
            "This request fills only the Protocol 101 selected-contract 1s gaps instead of buying broad history."
        ),
        "source_replay_json": str(replay_json),
        "requested_data": {
            "source": "Databento",
            "dataset": DATASET,
            "schemas": sorted({req.schema for req in requests}),
            "stype_in": "raw_symbol",
            "symbols": "Only raw symbols selected by frozen Protocol 101 that are missing high-resolution coverage.",
            "date_range": f"{requests[0].session} through {requests[-1].session}" if requests else None,
            "sessions": [req.session for req in requests],
            "session_count": len(requests),
            "session_count_by_schema": {
                schema: sum(1 for req in requests if req.schema == schema)
                for schema in sorted({req.schema for req in requests})
            },
            "selected_trade_rows": int(sum(req.selected_trades for req in requests)),
            "symbol_session_count": int(sum(req.symbols for req in requests)),
        },
        "cost_estimate": {
            "metadata_status": metadata_status,
            "estimated_total_usd": estimated_total,
            "hard_cap_usd": hard_cap,
            "fallback_cost_per_symbol_session": fallback_rates,
        },
        "approval_required": {
            "required": True,
            "exact_approval_text": approval_text,
            "reason": "Any Databento timeseries.get_range request for CMBP-1 or CBBO-1s is a paid market-data download.",
        },
        "implementation_note": (
            "Use CMBP-1 before 2025-02-20 because CBBO-1s is not available for older OPRA.PILLAR history. "
            "Download should write a Protocol101-specific overlay or merge safely; do not overwrite existing broad/selected "
            "CBBO-1s files for sessions that already have partial one-second coverage."
        ),
        "requests": [asdict(req) for req in requests],
    }


def session_plan_csv(requests: list[SessionRequest]) -> str:
    frame = pd.DataFrame([asdict(req) for req in requests])
    if frame.empty:
        return ""
    frame["splits"] = frame["splits"].map(lambda values: ",".join(values))
    return frame.to_csv(index=False)


def write_request(path: Path, manifest: dict[str, Any]) -> None:
    rows = manifest["requests"]
    cost = manifest["cost_estimate"]
    req = manifest["requested_data"]
    lines = [
        "# Protocol 116: Targeted High-Resolution Request For Protocol 101",
        "",
        "No paid market data was downloaded. No live broker data or order endpoint was used. No model was trained.",
        "",
        f"- Decision: `{manifest['decision']}`",
        f"- Source: `{req['source']}`",
        f"- Dataset/schemas: `{req['dataset']} / {', '.join(req['schemas'])}`",
        f"- Symbols: `{req['symbols']}`",
        f"- Date range: `{req['date_range']}`",
        f"- Sessions: `{req['session_count']}`",
        f"- Sessions by schema: `{req['session_count_by_schema']}`",
        f"- Selected trade rows to cover: `{req['selected_trade_rows']}`",
        f"- Symbol-session count: `{req['symbol_session_count']}`",
        f"- Estimated cost: `${cost['estimated_total_usd']:.4f}`",
        f"- Hard cap: `${cost['hard_cap_usd']:.2f}`",
        f"- Metadata status: `{cost['metadata_status']}`",
        "",
        "## Why This Batch",
        "",
        manifest["why"],
        "",
        "This is not broad history. It requests only the exact selected SPXW contracts needed to challenge Protocol 101's timing assumptions.",
        "",
        "For sessions before `2025-02-20`, the request uses `cmbp-1` because Databento metadata reports `cbbo-1s` is not available there. CMBP-1 is consolidated top-of-book update data and can support a stricter path audit than trade-sampled TCBBO.",
        "",
        "Reference: Databento schema docs describe CMBP-1 as consolidated top-of-book update data, CBBO as time-sampled consolidated BBO, and TCBBO as trade-sampled consolidated BBO: https://databento.com/docs/schemas-and-data-formats/whats-a-schema",
        "",
        "## Approval Text",
        "",
        "```text",
        manifest["approval_required"]["exact_approval_text"],
        "```",
        "",
        "## Implementation Note",
        "",
        manifest["implementation_note"],
        "",
        "## Session Plan",
        "",
        "| session | schema | splits | symbols | selected_trades | existing_1s_file | estimated_cost |",
        "| --- | --- | --- | ---: | ---: | --- | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['session']} | {row['schema']} | {', '.join(row['splits'])} | {row['symbols']} | {row['selected_trades']} | "
            f"{row['existing_1s_file']} | ${float(row['estimated_cost_usd']):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=json_default) + "\n"


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return str(value)


def append_ledger(ledger: Path, manifest: dict[str, Any], request_path: Path) -> None:
    heading = "## 2026-05-13 Protocol 116 Protocol101 Targeted High-Resolution Request"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    existing = ledger.read_text() if ledger.exists() else ""
    if heading in existing:
        return
    entry = f"""

{heading}

```text
Date: 2026-05-13
Decision / Experiment: Prepared a no-download targeted CMBP-1/CBBO-1s validation request for frozen Protocol 101.
Reason: Protocol 114 showed timing sensitivity and Protocol 115 found supportive but incomplete one-second evidence. The next paid step, if approved, should be selected-contract one-second validation rather than broad historical backfill.
Data Used: Existing Protocol 115 replay rows and local Databento CBBO-1s coverage metadata only. No paid data was downloaded, no live broker data was used, and no orders were placed.
Cost: $0 incremental paid data. Estimated requested Databento cost is ${manifest['cost_estimate']['estimated_total_usd']:.4f} with proposed hard cap ${manifest['cost_estimate']['hard_cap_usd']:.2f}.
Result: Decision {manifest['decision']}. Request and exact approval text are in {request_path}.
Next Gate: Only run a paid high-resolution Databento download if the user provides the exact approval text from the Protocol 116 request.
Owner: Codex
```
"""
    with ledger.open("a") as f:
        f.write(entry)


if __name__ == "__main__":
    raise SystemExit(main())
