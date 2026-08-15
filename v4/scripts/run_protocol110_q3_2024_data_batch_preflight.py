"""Protocol 110: no-paid Q3 2024 data batch preflight.

This script does not call Databento, ThetaData, Cboe, IBKR, or any other market
data endpoint. It only inspects local files and writes an approval-ready manifest
for the next chronological validation block.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_110_q3_2024_data_batch_preflight")
DEFAULT_REQUEST_PATH = Path("v4/promotion/PROTOCOL_101_Q3_2024_DATA_REQUEST.md")
DEFAULT_MANIFEST_PATH = Path("v4/promotion/PROTOCOL_101_Q3_2024_DOWNLOAD_MANIFEST.json")

DATE_RE = re.compile(r"(20\d\d-\d\d-\d\d)")
US_EQUITY_MARKET_HOLIDAYS = {
    "2024-01-01",
    "2024-01-15",
    "2024-02-19",
    "2024-03-29",
    "2024-05-27",
    "2024-06-19",
    "2024-07-04",
    "2024-09-02",
    "2024-11-28",
    "2024-12-25",
    "2025-01-01",
    "2025-01-09",
    "2025-01-20",
    "2025-02-17",
    "2025-04-18",
    "2025-05-26",
    "2025-06-19",
    "2025-07-04",
    "2025-09-01",
    "2025-11-27",
    "2025-12-25",
    "2026-01-01",
    "2026-01-19",
    "2026-02-16",
    "2026-04-03",
    "2026-05-25",
    "2026-06-19",
    "2026-07-03",
    "2026-09-07",
    "2026-11-26",
    "2026-12-25",
}

LOCAL_INPUTS = {
    "databento_definition": Path("data/raw/databento/opra_spxw_definition"),
    "databento_cbbo_1m": Path("data/raw/databento/opra_spxw_cbbo_1m"),
    "databento_ohlcv_1m": Path("data/raw/databento/opra_spxw_ohlcv_1m"),
    "databento_statistics": Path("data/raw/databento/opra_spxw_statistics"),
    "thetadata_spx_1m": Path("data/vendor/thetadata/index/spx_1m"),
    "thetadata_vix_1m": Path("data/vendor/thetadata/index/vix_1m"),
    "normalized_official_context": Path("v4/normalized_official_context"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-start", default="2024-07-01")
    parser.add_argument("--target-end", default="2024-09-30")
    parser.add_argument("--existing-start", default="2024-10-01")
    parser.add_argument("--existing-end", default="2026-03-31")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--request-path", type=Path, default=DEFAULT_REQUEST_PATH)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    existing_sessions = expected_sessions(args.existing_start, args.existing_end)
    target_sessions = expected_sessions(args.target_start, args.target_end)
    coverage = {
        name: coverage_summary(path, existing_sessions=existing_sessions, target_sessions=target_sessions)
        for name, path in LOCAL_INPUTS.items()
    }
    existing_pass = all(not item["existing_missing_sessions"] for item in coverage.values())
    target_already_present = any(item["target_present_sessions"] for item in coverage.values())
    manifest = build_manifest(
        args=args,
        existing_sessions=existing_sessions,
        target_sessions=target_sessions,
        coverage=coverage,
        existing_pass=existing_pass,
        target_already_present=target_already_present,
    )
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "summary.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_report(args.out_dir / "report.md", manifest)
    print(json.dumps({"decision": manifest["decision"], "report": str(args.out_dir / "report.md"), "manifest": str(args.manifest_path)}, indent=2, sort_keys=True))
    return 0


def expected_sessions(start: str, end: str) -> list[str]:
    sessions = []
    for day in pd.date_range(start=start, end=end, freq="D"):
        iso = day.date().isoformat()
        if day.weekday() < 5 and iso not in US_EQUITY_MARKET_HOLIDAYS:
            sessions.append(iso)
    return sessions


def dates_in_path(path: Path) -> set[str]:
    dates: set[str] = set()
    if not path.exists():
        return dates
    for file in path.glob("*"):
        match = DATE_RE.search(file.name)
        if match:
            dates.add(match.group(1))
    return dates


def coverage_summary(path: Path, *, existing_sessions: list[str], target_sessions: list[str]) -> dict[str, Any]:
    dates = dates_in_path(path)
    existing = set(existing_sessions)
    target = set(target_sessions)
    return {
        "path": str(path),
        "exists": path.exists(),
        "local_session_count": len(dates),
        "first_local_session": min(dates) if dates else None,
        "last_local_session": max(dates) if dates else None,
        "existing_expected_sessions": len(existing_sessions),
        "existing_present_sessions": len(existing & dates),
        "existing_missing_sessions": sorted(existing - dates),
        "existing_extra_sessions": sorted(date for date in dates - existing if min(existing_sessions) <= date <= max(existing_sessions)),
        "target_expected_sessions": len(target_sessions),
        "target_present_sessions": sorted(target & dates),
        "target_missing_sessions": sorted(target - dates),
    }


def build_manifest(
    *,
    args: argparse.Namespace,
    existing_sessions: list[str],
    target_sessions: list[str],
    coverage: dict[str, dict[str, Any]],
    existing_pass: bool,
    target_already_present: bool,
) -> dict[str, Any]:
    approval_text = (
        "I approve the Protocol 101 Q3 2024 data batch exactly as specified in "
        "v4/promotion/PROTOCOL_101_Q3_2024_DATA_REQUEST.md: Databento OPRA.PILLAR "
        "definition/cbbo-1m/ohlcv-1m/statistics for filtered SPXW 0DTE raw symbols "
        "and ThetaData SPX/VIX 1-minute bars for 2024-07-01 through 2024-09-30, "
        "with Databento hard cap $45."
    )
    return {
        "protocol": "110_q3_2024_data_batch_preflight",
        "paid_data_downloaded": False,
        "live_orders": False,
        "market_data_endpoints_called": False,
        "decision": "ready_for_explicit_user_approval" if existing_pass else "blocked_existing_coverage_gap",
        "existing_block": {
            "start": args.existing_start,
            "end": args.existing_end,
            "expected_sessions": len(existing_sessions),
            "continuity_pass": bool(existing_pass),
        },
        "target_batch": {
            "start": args.target_start,
            "end": args.target_end,
            "expected_sessions": len(target_sessions),
            "sessions": target_sessions,
            "target_already_present_anywhere": bool(target_already_present),
        },
        "coverage": coverage,
        "approval_required": {
            "required": True,
            "reason": "Any Databento or ThetaData market-data request for Q3 2024 is a paid/licensed data operation.",
            "exact_approval_text": approval_text,
        },
        "requested_data": {
            "source": ["Databento", "ThetaData"],
            "date_range": f"{args.target_start} through {args.target_end}",
            "databento": {
                "dataset": "OPRA.PILLAR",
                "schemas": ["definition", "cbbo-1m", "ohlcv-1m", "statistics"],
                "symbols": "SPXW.OPT parent definitions, then filtered SPXW 0DTE raw symbols",
                "estimated_cost_usd": "25-36",
                "hard_spend_cap_usd": 45,
            },
            "thetadata": {
                "products": ["SPX 1-minute index bars", "VIX 1-minute index bars"],
                "billing": "existing subscription only; no new paid add-on",
            },
        },
        "blocked_commands_after_approval_only": [
            f"export V4_PAID_DATA_APPROVAL_TEXT='{approval_text}'",
            "python3 -m v4.scripts.download_thetadata_index_bars --start-date 2024-07-01 --end-date 2024-09-30 --symbols SPX VIX --interval 1m --approval-manifest v4/promotion/PROTOCOL_101_Q3_2024_DOWNLOAD_MANIFEST.json",
            "python3 -m v4.scripts.download_databento_pilot --start-date 2024-07-01 --days 64 --include-ohlcv --include-statistics --max-cost 45 --audit-out v4/audit/databento_q3_2024_downloads.jsonl --approval-manifest v4/promotion/PROTOCOL_101_Q3_2024_DOWNLOAD_MANIFEST.json",
        ],
    }


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Protocol 110: Q3 2024 Data Batch Preflight",
        "",
        "No paid market data was downloaded. No market-data endpoint was called. No live broker data or order endpoint was used.",
        "",
        f"- Decision: `{manifest['decision']}`",
        f"- Existing block continuity: `{manifest['existing_block']['continuity_pass']}`",
        f"- Existing expected sessions: `{manifest['existing_block']['expected_sessions']}`",
        f"- Target Q3 2024 expected sessions: `{manifest['target_batch']['expected_sessions']}`",
        "",
        "## Existing Coverage",
        "",
        _table(
            [
                {
                    "input": name,
                    "sessions": item["local_session_count"],
                    "first": item["first_local_session"],
                    "last": item["last_local_session"],
                    "missing_existing": len(item["existing_missing_sessions"]),
                    "extra_existing": len(item["existing_extra_sessions"]),
                    "target_present": len(item["target_present_sessions"]),
                }
                for name, item in manifest["coverage"].items()
            ],
            ["input", "sessions", "first", "last", "missing_existing", "extra_existing", "target_present"],
        ),
        "",
        "## Approval Gate",
        "",
        "The next batch remains blocked until the user explicitly approves this exact request:",
        "",
        "```text",
        manifest["approval_required"]["exact_approval_text"],
        "```",
        "",
        "## Requested Data",
        "",
        "```json",
        json.dumps(manifest["requested_data"], indent=2, sort_keys=True),
        "```",
        "",
        "## Commands Blocked Until Approval",
        "",
        "```bash",
        "\n".join(manifest["blocked_commands_after_approval_only"]),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n")


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
