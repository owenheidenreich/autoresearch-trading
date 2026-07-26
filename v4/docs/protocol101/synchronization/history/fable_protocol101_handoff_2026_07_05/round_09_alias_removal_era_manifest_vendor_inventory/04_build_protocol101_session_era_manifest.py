"""Build a fail-closed Protocol101 session era manifest.

The fold scaffold should consume an explicit session-era manifest instead of
recomputing split/era logic internally. This script scans existing canonical
processed-session manifests and recorder capture quality files, assigns each
session through explicit date-range rules, and marks anything unmatched as
``unassigned_requires_decision``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "Protocol101SessionEraManifestV1"
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/protocol101_session_era_manifest")
DEFAULT_AUDIT_ROOT = Path("v4/audit/autoresearch")
DEFAULT_CAPTURE_ROOT = Path("~/.autoresearch-trading/live_runtime/ibkr_capture").expanduser()
UNASSIGNED_ERA = "unassigned_requires_decision"
SESSION_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


@dataclass(frozen=True)
class EraRule:
    era: str
    start: date
    end: date

    @classmethod
    def parse(cls, value: str) -> "EraRule":
        parts = value.split(":")
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                "era rules must have form era_name:YYYY-MM-DD:YYYY-MM-DD"
            )
        era, start_text, end_text = parts
        return cls(era=era, start=date.fromisoformat(start_text), end=date.fromisoformat(end_text))

    def contains(self, session: str) -> bool:
        session_date = date.fromisoformat(session)
        return self.start <= session_date <= self.end

    def as_dict(self) -> dict[str, str]:
        return {"era": self.era, "start": self.start.isoformat(), "end": self.end.isoformat()}


DEFAULT_ERA_RULES = (
    EraRule.parse("owned_jul_dec2025:2025-07-01:2025-12-31"),
    EraRule.parse("q1_2026_development:2026-01-01:2026-03-31"),
    EraRule.parse("confirmation_jun_jul2026:2026-06-01:2026-07-31"),
)


@dataclass
class SessionEvidence:
    session: str
    source_types: set[str] = field(default_factory=set)
    source_paths: set[str] = field(default_factory=set)
    source_statuses: set[str] = field(default_factory=set)
    evidence: dict[str, Any] = field(default_factory=dict)

    def add(
        self,
        *,
        source_type: str,
        source_path: Path,
        source_status: str = "",
        evidence: dict[str, Any] | None = None,
    ) -> None:
        self.source_types.add(source_type)
        self.source_paths.add(str(source_path))
        if source_status:
            self.source_statuses.add(source_status)
        if evidence:
            for key, value in evidence.items():
                self.evidence.setdefault(key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--era-rule",
        type=EraRule.parse,
        action="append",
        default=None,
        help="Explicit era rule as era_name:YYYY-MM-DD:YYYY-MM-DD. May repeat.",
    )
    parser.add_argument(
        "--allow-unassigned",
        action="store_true",
        help="Write manifest with unassigned sessions and status warn instead of fail.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def session_from_text(value: Any) -> str | None:
    match = SESSION_RE.search(str(value or ""))
    if not match:
        return None
    session = match.group(0)
    try:
        date.fromisoformat(session)
    except ValueError:
        return None
    return session


def collect_processed_manifests(audit_root: Path) -> dict[str, SessionEvidence]:
    sessions: dict[str, SessionEvidence] = {}
    for manifest_path in sorted(audit_root.glob("**/canonical_processed_session_manifest.json")):
        payload = read_json(manifest_path)
        if not payload:
            continue
        for item in payload.get("included_sessions") or []:
            session = session_from_text(item.get("session") if isinstance(item, dict) else item)
            if not session:
                continue
            sessions.setdefault(session, SessionEvidence(session=session)).add(
                source_type="canonical_processed_session",
                source_path=manifest_path,
                source_status=str(payload.get("status") or ""),
                evidence={
                    "processed_file": item.get("processed_file") if isinstance(item, dict) else "",
                    "normalized_base_file": item.get("normalized_base_file") if isinstance(item, dict) else "",
                },
            )
        for item in payload.get("excluded_recorder_or_parity_sessions") or []:
            session = session_from_text(item)
            if not session:
                continue
            sessions.setdefault(session, SessionEvidence(session=session)).add(
                source_type="excluded_recorder_or_parity_session",
                source_path=manifest_path,
                source_status=str(payload.get("status") or ""),
            )
    return sessions


def collect_recorder_manifests(capture_root: Path) -> dict[str, SessionEvidence]:
    sessions: dict[str, SessionEvidence] = {}
    if not capture_root.exists():
        return sessions
    for quality_path in sorted(capture_root.glob("*/protocol101-recorder-*/ibkr_capture_quality.json")):
        session = session_from_text(quality_path)
        payload = read_json(quality_path) or {}
        if not session:
            session = session_from_text(payload.get("capture_id"))
        if not session:
            continue
        checks = payload.get("checks") if isinstance(payload.get("checks"), dict) else {}
        complete = checks.get("complete_regular_session")
        status = "complete" if complete is True else "partial_or_failed"
        evidence = payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {}
        sessions.setdefault(session, SessionEvidence(session=session)).add(
            source_type="ibkr_recorder_capture",
            source_path=quality_path,
            source_status=status,
            evidence={
                "capture_id": payload.get("capture_id", ""),
                "checkpoint_count": evidence.get("checkpoint_count"),
                "expected_checkpoint_count": evidence.get("expected_checkpoint_count"),
                "broker_order_endpoint_called": evidence.get("broker_order_endpoint_called"),
            },
        )
    return sessions


def merge_session_maps(*maps: dict[str, SessionEvidence]) -> dict[str, SessionEvidence]:
    merged: dict[str, SessionEvidence] = {}
    for session_map in maps:
        for session, item in session_map.items():
            target = merged.setdefault(session, SessionEvidence(session=session))
            target.source_types.update(item.source_types)
            target.source_paths.update(item.source_paths)
            target.source_statuses.update(item.source_statuses)
            target.evidence.update(item.evidence)
    return merged


def assign_era(session: str, rules: Iterable[EraRule]) -> str:
    matches = [rule.era for rule in rules if rule.contains(session)]
    if len(matches) == 1:
        return matches[0]
    return UNASSIGNED_ERA


def manifest_hash(records: list[dict[str, Any]], rules: list[EraRule]) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "era_rules": [rule.as_dict() for rule in rules],
        "sessions": records,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_manifest(
    *,
    audit_root: Path,
    capture_root: Path,
    era_rules: list[EraRule],
    allow_unassigned: bool,
) -> dict[str, Any]:
    processed = collect_processed_manifests(audit_root)
    recorder = collect_recorder_manifests(capture_root)
    merged = merge_session_maps(processed, recorder)
    records: list[dict[str, Any]] = []
    for session, item in sorted(merged.items()):
        era = assign_era(session, era_rules)
        records.append(
            {
                "session": session,
                "era": era,
                "source_types": sorted(item.source_types),
                "source_statuses": sorted(status for status in item.source_statuses if status),
                "source_paths": sorted(item.source_paths),
                "evidence": item.evidence,
            }
        )
    unassigned = [record["session"] for record in records if record["era"] == UNASSIGNED_ERA]
    digest = manifest_hash(records, era_rules)
    status = "pass"
    if unassigned:
        status = "warn" if allow_unassigned else "fail"
    counts_by_era: dict[str, int] = {}
    for record in records:
        counts_by_era[record["era"]] = counts_by_era.get(record["era"], 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "manifest_hash": digest,
        "fail_closed_default_era": UNASSIGNED_ERA,
        "allow_unassigned": bool(allow_unassigned),
        "audit_root": str(audit_root),
        "capture_root": str(capture_root),
        "era_rules": [rule.as_dict() for rule in era_rules],
        "session_count": len(records),
        "counts_by_era": dict(sorted(counts_by_era.items())),
        "unassigned_sessions": unassigned,
        "sessions": records,
    }


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = ("session", "era", "source_types", "source_statuses", "source_paths")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "session": record["session"],
                    "era": record["era"],
                    "source_types": ";".join(record["source_types"]),
                    "source_statuses": ";".join(record["source_statuses"]),
                    "source_paths": ";".join(record["source_paths"]),
                }
            )


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Protocol101 Session Era Manifest",
        "",
        f"- Status: `{payload['status']}`",
        f"- Manifest hash: `{payload['manifest_hash']}`",
        f"- Session count: `{payload['session_count']}`",
        f"- Fail-closed default: `{payload['fail_closed_default_era']}`",
        f"- Allow unassigned: `{str(payload['allow_unassigned']).lower()}`",
        "",
        "## Counts By Era",
        "",
    ]
    for era, count in payload["counts_by_era"].items():
        lines.append(f"- `{era}`: `{count}`")
    lines.extend(["", "## Era Rules", ""])
    for rule in payload["era_rules"]:
        lines.append(f"- `{rule['era']}`: `{rule['start']}` to `{rule['end']}`")
    if payload["unassigned_sessions"]:
        lines.extend(["", "## Unassigned Sessions", ""])
        lines.extend(f"- `{session}`" for session in payload["unassigned_sessions"])
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The fold scaffold should consume this manifest and refuse to place `unassigned_requires_decision` sessions.",
            "- Recorder sessions are included as confirmation-era evidence and retain their complete/partial source status.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    era_rules = list(args.era_rule or DEFAULT_ERA_RULES)
    payload = build_manifest(
        audit_root=args.audit_root,
        capture_root=args.capture_root,
        era_rules=era_rules,
        allow_unassigned=bool(args.allow_unassigned),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sessions_path = args.out_dir / "sessions_manifest.json"
    summary_path = args.out_dir / "summary.json"
    csv_path = args.out_dir / "sessions_manifest.csv"
    report_path = args.out_dir / "report.md"
    sessions_path.write_text(json.dumps(payload["sessions"], indent=2, sort_keys=True) + "\n")
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_csv(csv_path, payload["sessions"])
    report_path.write_text(render_report(payload))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "session_count": payload["session_count"],
                "manifest_hash": payload["manifest_hash"],
                "report": str(report_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["status"] in {"pass", "warn"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
