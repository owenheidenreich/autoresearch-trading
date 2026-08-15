"""Protocol206: freeze Protocol202 as a research challenger.

Protocol202 is not the paper-trading default and is not promoted. It is,
however, the strongest lifecycle challenger after Protocol200-205. This runner
creates a reproducible evidence packet and repeat-gap checklist so future live
logs or new historical blocks can test the same hypothesis without drifting.

No paid data is downloaded. No broker endpoint is called.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


LOOP_ID = "v4_aplus_hypothesis_206_protocol202_challenger_freeze"
DEFAULT_OUT_DIR = Path(f"v4/audit/autoresearch/{LOOP_ID}")
PROTOCOL202_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_202_slot_aware_lifecycle_policy")
PROTOCOL203_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_203_protocol202_mixed_result_attribution")
PROTOCOL205_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_205_protocol202_recent_gap_attribution")
PROMOTION_PACKET = Path("v4/promotion/PROTOCOL_202_RESEARCH_CHALLENGER_PACKET.md")
FREEZE_MANIFEST = Path("v4/promotion/PROTOCOL_202_RESEARCH_CHALLENGER_FREEZE.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    PROMOTION_PACKET.parent.mkdir(parents=True, exist_ok=True)
    protocol202 = load_json(PROTOCOL202_DIR / "summary.json")
    protocol203 = load_json(PROTOCOL203_DIR / "summary.json")
    protocol205 = load_json(PROTOCOL205_DIR / "summary.json")
    files = source_files()
    manifest = {
        "protocol": "202_research_challenger_freeze",
        "status": "research_challenger_not_paper_default",
        "paper_default": "Protocol101",
        "paid_data_downloaded_by_runner": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "model_training": False,
        "source_protocols": {
            "protocol202": str(PROTOCOL202_DIR),
            "protocol203": str(PROTOCOL203_DIR),
            "protocol205": str(PROTOCOL205_DIR),
        },
        "source_file_hashes": [file_record(path) for path in files],
        "evidence": {
            "protocol202_decision": protocol202.get("decision"),
            "protocol202_comparison": protocol202.get("comparison", []),
            "protocol203_decision": protocol203.get("decision"),
            "protocol205_decision": protocol205.get("decision"),
            "protocol205_contribution_summary": protocol205.get("contribution_summary", {}),
            "protocol205_side_summary": protocol205.get("side_summary", []),
            "protocol205_time_bucket_summary": protocol205.get("time_bucket_summary", []),
        },
        "operational_rule": {
            "protocol101_remains_paper_default": True,
            "protocol202_live_orders_enabled": False,
            "protocol202_allowed_use": "offline research challenger and future no-order shadow comparison only",
        },
        "repeat_gap_watchlist": {
            "gap_to_watch": "Protocol202 recent gap: calls and late_afternoon/midday exits gave back a small amount versus baseline.",
            "repeat_condition": (
                "On a new locked block or live-paper replay, flag if Protocol202 underperforms the matching "
                "baseline by more than $5,000 median and the gap is again call-heavy or late-afternoon/midday-heavy."
            ),
            "do_not_change_model_unless": (
                "The same gap pattern repeats on additional protected data or live-paper logs; otherwise treat "
                "the recent_2026 miss as too small to justify a new architecture knob."
            ),
        },
        "decision": decision(protocol202, protocol205),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    FREEZE_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", manifest)
    write_packet(PROMOTION_PACKET, manifest)
    print(json.dumps({"decision": manifest["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def source_files() -> list[Path]:
    paths = [
        PROTOCOL202_DIR / "summary.json",
        PROTOCOL202_DIR / "report.md",
        PROTOCOL202_DIR / "protocol202_model_serial_trades.csv",
        PROTOCOL202_DIR / "protocol194_baseline_serial_trades.csv",
        PROTOCOL202_DIR / "threshold_sweep.csv",
        PROTOCOL203_DIR / "summary.json",
        PROTOCOL203_DIR / "report.md",
        PROTOCOL205_DIR / "summary.json",
        PROTOCOL205_DIR / "report.md",
        PROTOCOL205_DIR / "combo_seed_summary.csv",
        PROTOCOL205_DIR / "side_summary.csv",
        PROTOCOL205_DIR / "time_bucket_summary.csv",
    ]
    return [path for path in paths if path.exists()]


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256(path),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decision(protocol202: dict[str, Any], protocol205: dict[str, Any]) -> str:
    comparisons = {row.get("reported_split"): row for row in protocol202.get("comparison", [])}
    q1 = bool(comparisons.get("q1_2026", {}).get("beats_baseline"))
    march = bool(comparisons.get("march_2026", {}).get("beats_baseline"))
    recent = bool(comparisons.get("recent_2026", {}).get("beats_baseline"))
    if q1 and march and not recent and protocol205.get("decision") == "recent_gap_broad_across_combo_seeds":
        return "freeze_protocol202_as_research_challenger_not_paper_default"
    return "do_not_freeze_protocol202_until_evidence_reviewed"


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    comparison = manifest["evidence"].get("protocol202_comparison", [])
    lines = [
        "# Protocol206 Protocol202 Research Challenger Freeze",
        "",
        "No paid data was downloaded. No broker endpoint was called. No model was trained.",
        "",
        f"- Decision: `{manifest['decision']}`",
        f"- Paper default remains: `{manifest['paper_default']}`",
        "- Protocol202 status: `research_challenger_not_paper_default`",
        "",
        "## Why This Exists",
        "",
        "Protocol202 is useful because it trains lifecycle decisions with single-slot opportunity cost. "
        "It is not promoted because it still missed the protected recent_2026 block by a small amount.",
        "",
        "## Evidence",
        "",
        "| split | model | baseline | delta | beats baseline |",
        "|---|---:|---:|---:|---|",
    ]
    for row in comparison:
        lines.append(
            f"| {row['reported_split']} | {money(row['model_median_total_pnl'])} | "
            f"{money(row['baseline_median_total_pnl'])} | {money(row['delta_vs_baseline'])} | "
            f"{row['beats_baseline']} |"
        )
    c = manifest["evidence"].get("protocol205_contribution_summary", {})
    lines.extend(
        [
            "",
            "## Recent Gap Watchlist",
            "",
            f"- Recent same-entry exit changes: {money(c.get('matched_exit_change_contribution'))}",
            f"- Recent baseline-only missed/blocked trades: {money(c.get('baseline_only_contribution'))}",
            f"- Recent Protocol202-only extra trades: {money(c.get('protocol202_only_contribution'))}",
            f"- Repeat condition: {manifest['repeat_gap_watchlist']['repeat_condition']}",
            "",
            "## Files",
            "",
            f"- Freeze manifest: `{FREEZE_MANIFEST}`",
            f"- Research packet: `{PROMOTION_PACKET}`",
            f"- Summary: `{path.parent / 'summary.json'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def write_packet(path: Path, manifest: dict[str, Any]) -> None:
    comparison = manifest["evidence"].get("protocol202_comparison", [])
    lines = [
        "# Protocol 202 Research Challenger Packet",
        "",
        "## Status",
        "",
        "- Protocol202 is frozen as a research challenger only.",
        "- Protocol101 remains the paper-trading default.",
        "- Protocol202 must not submit paper or live orders.",
        "- Allowed use: offline comparison and future no-order shadow analysis.",
        "",
        "## Evidence Summary",
        "",
        "| split | Protocol202 | baseline | delta | decision |",
        "|---|---:|---:|---:|---|",
    ]
    for row in comparison:
        lines.append(
            f"| {row['reported_split']} | {money(row['model_median_total_pnl'])} | "
            f"{money(row['baseline_median_total_pnl'])} | {money(row['delta_vs_baseline'])} | "
            f"{'beat' if row['beats_baseline'] else 'miss'} |"
        )
    lines.extend(
        [
            "",
            "## Recent Gap",
            "",
            manifest["repeat_gap_watchlist"]["gap_to_watch"],
            "",
            "Do not add another architecture knob unless this same pattern repeats on additional protected data or live-paper logs.",
            "",
            "## Source Hashes",
            "",
        ]
    )
    for record in manifest["source_file_hashes"]:
        lines.append(f"- `{record['path']}` sha256 `{record['sha256']}`")
    path.write_text("\n".join(lines) + "\n")


def money(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not number == number:
        return "n/a"
    sign = "-" if number < 0 else ""
    return f"{sign}${abs(number):,.0f}"


if __name__ == "__main__":
    raise SystemExit(main())
