"""Build the governed Protocol101 protected-holdout artifact.

This artifact is intentionally small: it declares sessions that are reserved
from model selection and binds that declaration with a stable hash. The
governed loader refuses to train/evaluate through the fair-contract runner
unless this artifact is present and passing, or an explicit owner override is
provided for one-shot holdout evaluation.
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from v4.scripts.run_protocol101_owned_raw_acceptance_verifier import stable_hash


SCHEMA_VERSION = "Protocol101ProtectedHoldoutArtifactV1"
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/protocol101_protected_holdout")
HASH_FIELDS = (
    "schema_version",
    "scope",
    "role",
    "status",
    "session_count",
    "sessions",
    "owner_note",
)


def normalize_sessions(sessions: list[str] | tuple[str, ...]) -> list[str]:
    return sorted({str(session).strip() for session in sessions if str(session).strip()})


def protected_holdout_hash_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {field: payload.get(field) for field in HASH_FIELDS}


def compute_protected_holdout_hash(payload: dict[str, Any]) -> str:
    return stable_hash(protected_holdout_hash_payload(payload))


def build_artifact(
    *,
    sessions: list[str] | tuple[str, ...],
    owner_note: str = "",
    scope: str = "protocol101_fair_contract_model_selection_lockbox",
    role: str = "protected_holdout",
) -> dict[str, Any]:
    normalized = normalize_sessions(sessions)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "scope": str(scope),
        "role": str(role),
        "status": "pass" if normalized else "pending_owner_declaration",
        "session_count": len(normalized),
        "sessions": normalized,
        "owner_note": str(owner_note),
        "paper_submit_allowed": False,
        "model_training_executed": False,
        "threshold_tuning_executed": False,
    }
    payload["protected_holdout_hash"] = compute_protected_holdout_hash(payload)
    return payload


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Protocol101 Protected Holdout Artifact",
        "",
        f"- status: `{payload.get('status')}`",
        f"- session_count: `{payload.get('session_count')}`",
        f"- protected_holdout_hash: `{payload.get('protected_holdout_hash')}`",
        f"- paper_submit_allowed: `{payload.get('paper_submit_allowed')}`",
        f"- model_training_executed: `{payload.get('model_training_executed')}`",
        f"- threshold_tuning_executed: `{payload.get('threshold_tuning_executed')}`",
        "",
        "## Sessions",
        "",
    ]
    sessions = payload.get("sessions") or []
    if sessions:
        lines.extend(f"- `{session}`" for session in sessions)
    else:
        lines.append("- No protected holdout sessions have been owner-declared yet.")
    lines.extend(
        [
            "",
            "## Loader Meaning",
            "",
            "- `pass` means the governed loader can train/evaluate non-holdout splits.",
            "- `pending_owner_declaration` blocks the governed loader by default.",
            "- Protected sessions are rejected unless an explicit owner one-shot evaluation override is supplied, and even then only for the `test` role — never train/tune.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--session", action="append", default=[], help="Protected holdout session. May repeat.")
    parser.add_argument("--owner-note", default="")
    parser.add_argument("--scope", default="protocol101_fair_contract_model_selection_lockbox")
    parser.add_argument("--role", default="protected_holdout")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_artifact(
        sessions=list(args.session or []),
        owner_note=str(args.owner_note),
        scope=str(args.scope),
        role=str(args.role),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "summary.json"
    artifact_path = args.out_dir / "protected_holdout.json"
    report_path = args.out_dir / "report.md"
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    summary_path.write_text(text)
    artifact_path.write_text(text)
    report_path.write_text(render_report(payload))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "session_count": payload["session_count"],
                "protected_holdout_hash": payload["protected_holdout_hash"],
                "summary": str(summary_path),
                "report": str(report_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
