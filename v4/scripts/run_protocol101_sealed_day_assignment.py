"""Protocol101 sealed-day assignment: seal-on-arrival for parity recorder sessions.

Rule v2 (revised 2026-07-10, owner-directed, while the sealed set was still
EMPTY — no sealed session existed at revision time, so no data is tainted):
  - 2026-06-30, 2026-07-01, 2026-07-02: BURNED (design/repair set; used to build
    canonical v1..v1.4; certified frozen contract sha256
    602fd8eff564a059ad114dd051b6793cb50bcc50269c83b81bdfe25aa119ef57).
  - 2026-07-10: VALIDATION day. Openly inspected to verify capture quality.
    Never sealed, never confirmation evidence.
  - 2026-07-13, 2026-07-14, 2026-07-20: DEVELOPMENT days. Openly inspectable
    for rehearsal, diagnosis, and repair. 07-14 is the June-CPI release
    session, giving the development set one event-regime day. Never sealed,
    never confirmation evidence.
  - Every other parity-recorder session dated >= 2026-07-13: SEALED ON
    ARRIVAL (expected: 07-15..17, 07-21..24, 07-27..30 = 11 sessions,
    including the 07-28/29 FOMC days, which deliberately stay sealed).
    Moved to the sealed root; no analysis/audit tooling may read its market
    data until the one-shot preregistered confirmation battery is executed.
    Health checks read this script's manifests only.
  Rationale for v2: a development slice lets battery failures be diagnosed
  and repaired on fresh same-regime days, and lets a rehearsal run of the
  identical battery gate the sealed exam, instead of a failure spending the
  entire sealed set. v1 (all >= 07-13 sealed) never governed any actual data.

Sealed evidence integrity depends on this rule pre-dating the data it governs.
Changing SEAL_FROM_SESSION, VALIDATION_SESSIONS, or the sealed root after
2026-07-13 requires an owner-signed revision and taints affected days.

Modes:
  assign  (default) move eligible sessions from the capture root to the sealed
          root, write per-session seal manifests (names/sizes/sha256 only).
  check   read-only: verify every expected sealed session is sealed and intact
          per its manifest; exit nonzero on any gap. Safe for health automation.
  status  print the rule and current assignment table.

This script never parses, loads, or prints market-data content. Hashing reads
bytes for integrity only; values are never displayed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

RULE_VERSION = "protocol101_sealed_day_assignment_v2"
RULE_FIXED_UTC = "2026-07-10T21:15:00+00:00"
RULE_V1_FIXED_UTC = "2026-07-09T23:30:00+00:00"
SEAL_FROM_SESSION = "2026-07-13"
VALIDATION_SESSIONS = ("2026-07-10",)
DEVELOPMENT_SESSIONS = ("2026-07-13", "2026-07-14", "2026-07-20")
BURNED_SESSIONS = ("2026-06-30", "2026-07-01", "2026-07-02")
FROZEN_CONTRACT_SHA256 = "602fd8eff564a059ad114dd051b6793cb50bcc50269c83b81bdfe25aa119ef57"

DEFAULT_CAPTURE_ROOT = Path.home() / ".autoresearch-trading/live_runtime/ibkr_capture"
DEFAULT_SEALED_ROOT = Path.home() / ".autoresearch-trading/live_runtime/ibkr_capture_sealed"
REPO_ROOT = Path(__file__).resolve().parents[2]
GOVERNANCE_DIR = REPO_ROOT / "v4/audit/autoresearch/protocol101_sealed_day_assignment"
SENTINEL_NAME = "DO_NOT_OPEN_SEALED_EVIDENCE.md"
SENTINEL_TEXT = (
    "# SEALED CONFIRMATION EVIDENCE\n\n"
    "Sessions under this directory are sealed-on-arrival per "
    f"{RULE_VERSION} (rule fixed {RULE_FIXED_UTC}).\n"
    "No analysis, parity, repair, or audit tooling may read market data here.\n"
    "They are spent exactly once, by the preregistered one-shot confirmation\n"
    "battery against frozen canonical contract "
    f"{FROZEN_CONTRACT_SHA256}.\n"
    "Opening a session's market data before that run burns the day.\n"
)


def classify(session: str) -> str:
    if session in BURNED_SESSIONS:
        return "burned"
    if session in VALIDATION_SESSIONS:
        return "validation"
    if session in DEVELOPMENT_SESSIONS:
        return "development"
    if session >= SEAL_FROM_SESSION:
        return "sealed"
    return "unassigned_pre_rule"


def rule_payload() -> dict[str, Any]:
    return {
        "schema_version": "Protocol101SealedDayAssignmentRuleV1",
        "rule_version": RULE_VERSION,
        "rule_fixed_utc": RULE_FIXED_UTC,
        "rule_v1_fixed_utc": RULE_V1_FIXED_UTC,
        "rule_v2_revision_note": (
            "v2 adds a development slice (07-13, 07-14 CPI, 07-20) for "
            "rehearsal/diagnosis/repair; revised 2026-07-10 while the sealed "
            "set was empty, so no sealed data existed under v1; FOMC "
            "07-28/29 deliberately remain sealed"
        ),
        "seal_from_session": SEAL_FROM_SESSION,
        "validation_sessions": list(VALIDATION_SESSIONS),
        "development_sessions": list(DEVELOPMENT_SESSIONS),
        "burned_sessions": list(BURNED_SESSIONS),
        "frozen_canonical_contract_sha256": FROZEN_CONTRACT_SHA256,
        "sealed_root_policy": (
            "sealed sessions move to the sealed root on the same volume; "
            "no analysis/audit tooling reads market data there; health "
            "automation uses this script's check mode and manifests only"
        ),
        "spend_policy": (
            "sealed days are opened exactly once, by the preregistered "
            "one-shot confirmation battery; after that run they are burned"
        ),
        "revision_policy": (
            "changing this rule after 2026-07-13 requires an owner-signed "
            "revision and taints affected days"
        ),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_payload(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def session_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    out = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and len(child.name) == 10 and child.name[4] == "-":
            try:
                date.fromisoformat(child.name)
            except ValueError:
                continue
            out.append(child)
    return out


def file_inventory(session_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(session_dir.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "relpath": str(path.relative_to(session_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return rows


def ensure_governance(sealed_root: Path) -> None:
    write_json(GOVERNANCE_DIR / "sealing_rule.json", rule_payload())
    (GOVERNANCE_DIR / "sealing_rule.sha256").write_text(sha256_payload(rule_payload()) + "\n")
    sealed_root.mkdir(parents=True, exist_ok=True)
    sentinel = sealed_root / SENTINEL_NAME
    if not sentinel.exists():
        sentinel.write_text(SENTINEL_TEXT)


def assign(capture_root: Path, sealed_root: Path, dry_run: bool) -> int:
    ensure_governance(sealed_root)
    actions = []
    for session_dir in session_dirs(capture_root):
        session = session_dir.name
        cls = classify(session)
        if cls != "sealed":
            actions.append({"session": session, "class": cls, "action": "left_in_place"})
            continue
        target = sealed_root / session
        if target.exists():
            actions.append({"session": session, "class": cls, "action": "already_sealed"})
            continue
        manifest = {
            "schema_version": "Protocol101SealedSessionManifestV1",
            "session": session,
            "rule_version": RULE_VERSION,
            "sealed_at_utc": datetime.now(UTC).isoformat(),
            "source": str(session_dir),
            "files": file_inventory(session_dir),
        }
        if dry_run:
            actions.append({"session": session, "class": cls, "action": "would_seal",
                            "files": len(manifest["files"])})
            continue
        shutil.move(str(session_dir), str(target))
        write_json(GOVERNANCE_DIR / f"seal_manifest_{session.replace('-', '_')}.json", manifest)
        os.chmod(target, 0o500)
        actions.append({"session": session, "class": cls, "action": "sealed",
                        "files": len(manifest["files"])})
    report = {
        "schema_version": "Protocol101SealedDayAssignmentRunV1",
        "mode": "assign",
        "dry_run": dry_run,
        "run_at_utc": datetime.now(UTC).isoformat(),
        "capture_root": str(capture_root),
        "sealed_root": str(sealed_root),
        "rule_sha256": sha256_payload(rule_payload()),
        "actions": actions,
    }
    write_json(GOVERNANCE_DIR / "last_assignment_run.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def check(capture_root: Path, sealed_root: Path) -> int:
    problems = []
    today = datetime.now(UTC).date().isoformat()
    # any session that should be sealed but still sits in the capture root
    for session_dir in session_dirs(capture_root):
        if classify(session_dir.name) == "sealed" and session_dir.name < today:
            problems.append(f"unsealed_past_session_in_capture_root:{session_dir.name}")
    # sealed sessions must match their manifests (names/sizes only; no content reads)
    for target in session_dirs(sealed_root):
        manifest_path = GOVERNANCE_DIR / f"seal_manifest_{target.name.replace('-', '_')}.json"
        if not manifest_path.is_file():
            problems.append(f"sealed_session_missing_manifest:{target.name}")
            continue
        manifest = json.loads(manifest_path.read_text())
        for row in manifest.get("files", []):
            path = target / row["relpath"]
            if not path.is_file():
                problems.append(f"sealed_file_missing:{target.name}/{row['relpath']}")
            elif path.stat().st_size != row["bytes"]:
                problems.append(f"sealed_file_size_changed:{target.name}/{row['relpath']}")
    status = {
        "schema_version": "Protocol101SealedDayAssignmentCheckV1",
        "mode": "check",
        "run_at_utc": datetime.now(UTC).isoformat(),
        "sealed_sessions": [d.name for d in session_dirs(sealed_root)],
        "problems": problems,
        "ok": not problems,
    }
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0 if not problems else 1


def status(capture_root: Path, sealed_root: Path) -> int:
    rows = []
    for session_dir in session_dirs(capture_root):
        rows.append({"session": session_dir.name, "class": classify(session_dir.name),
                     "location": "capture_root"})
    for session_dir in session_dirs(sealed_root):
        rows.append({"session": session_dir.name, "class": "sealed",
                     "location": "sealed_root"})
    print(json.dumps({"rule": rule_payload(), "sessions": sorted(rows, key=lambda r: r["session"])},
                     indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", default="assign", choices=["assign", "check", "status"])
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--sealed-root", type=Path, default=DEFAULT_SEALED_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.mode == "assign":
        return assign(args.capture_root, args.sealed_root, args.dry_run)
    if args.mode == "check":
        return check(args.capture_root, args.sealed_root)
    return status(args.capture_root, args.sealed_root)


if __name__ == "__main__":
    sys.exit(main())
