"""Build Protocol101 era role policy artifact.

The session era manifest is a facts layer. This artifact is the policy layer:
it maps era names to permitted fold/evidence roles without changing the facts
manifest hash whenever governance rules are revised.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "Protocol101EraRolePolicyV1"
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/protocol101_era_role_policy")
DEFAULT_SESSION_MANIFEST = Path("v4/audit/autoresearch/protocol101_session_era_manifest/summary.json")
KNOWN_ROLES = {
    "train",
    "test",
    "diagnostics_only",
    "report_only",
    "confirmation_one_shot",
}
ROLE_TAXONOMY = {
    "train": "May appear in model-tier training windows when all acceptance and fold predicates pass.",
    "test": "May appear in model-tier test windows when all acceptance and fold predicates pass.",
    "diagnostics_only": "May appear in diagnostics-tier fold test windows for gates, samplers, nulls, and uplift; never model-tier train/test.",
    "report_only": "May be summarized for context but not used for model selection, diagnostics gates, or promotion claims.",
    "confirmation_one_shot": "May be used only for one-shot live/parity confirmation, not training or tuning.",
}
PROMOTION_GUARDS = {
    "pre_program_systematic_negative_guard": {
        "if": "pooled criteria pass but pre_program_oct2024_jun2025 test folds are systematically negative",
        "status": "regime_bound_requires_owner_review",
        "reason": "Pooled fold success must not hide failure in the cleanest owned pre-program era.",
    }
}


DEFAULT_POLICY: dict[str, dict[str, Any]] = {
    "pre_program_oct2024_jun2025": {
        "permitted_roles": ["train", "test", "diagnostics_only"],
        "evidence_tier": "owned_pre_program_raw_pending_acceptance",
        "notes": "Cleanest owned pre-program era after raw acceptance gates pass.",
    },
    "owned_jul_dec2025": {
        "permitted_roles": ["train", "test", "diagnostics_only"],
        "evidence_tier": "owned_mechanically_clean_for_new_fold_models_with_family_selection_caveat",
        "notes": "Mechanically valid for new fold-trained candidates; interpret family-level claims with historical program-ancestry caveat.",
    },
    "q1_2026_development": {
        "permitted_roles": ["diagnostics_only", "report_only"],
        "evidence_tier": "development_report_only_until_split_ancestry_audit",
        "notes": "Default report-only. Jan-Feb may be upgraded only after explicit design/threshold ancestry audit.",
    },
    "post_q1_gap_apr_may2026": {
        "permitted_roles": ["report_only"],
        "evidence_tier": "post_q1_nonfold_report_only",
        "notes": "Post-Q1 gap stays out of fold selection unless separately governed.",
    },
    "confirmation_jun_jul2026": {
        "permitted_roles": ["confirmation_one_shot", "report_only"],
        "evidence_tier": "recorder_confirmation_only",
        "notes": "Recorder/live-parity era. Never train/tune on these sessions.",
    },
    "unassigned_requires_decision": {
        "permitted_roles": [],
        "evidence_tier": "blocked",
        "notes": "Fail-closed. No fold scaffold may place this era.",
    },
    "extension_2024h1": {
        "permitted_roles": ["train", "test", "diagnostics_only"],
        "evidence_tier": "placeholder_pending_owner_approval_and_acceptance",
        "notes": "Placeholder for a possible 2024-01 through 2024-09 purchase tier. No sessions should use this until approved and accepted.",
    },
    "extension_2023": {
        "permitted_roles": ["train", "test", "diagnostics_only"],
        "evidence_tier": "placeholder_pending_owner_approval_and_acceptance",
        "notes": "Placeholder for a possible 2023 extension tier. No sessions should use this until approved and accepted.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-manifest", type=Path, default=DEFAULT_SESSION_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_session_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def validate_policy(policy: dict[str, dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    role_errors: list[str] = []
    for era, item in sorted(policy.items()):
        roles = item.get("permitted_roles") or []
        unknown = sorted(set(roles) - KNOWN_ROLES)
        if unknown:
            role_errors.append(f"{era}:unknown_roles:{','.join(unknown)}")
    manifest_eras = set((manifest.get("counts_by_era") or {}).keys())
    missing_policy_eras = sorted(era for era in manifest_eras if era not in policy)
    return {
        "known_roles": sorted(KNOWN_ROLES),
        "role_errors": role_errors,
        "manifest_eras": sorted(manifest_eras),
        "missing_policy_eras": missing_policy_eras,
        "pass": not role_errors and not missing_policy_eras,
    }


def build_policy_artifact(session_manifest: Path) -> dict[str, Any]:
    manifest = load_session_manifest(session_manifest)
    validation = validate_policy(DEFAULT_POLICY, manifest)
    policy_hash = stable_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "policy": DEFAULT_POLICY,
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if validation["pass"] else "fail",
        "policy_hash": policy_hash,
        "session_manifest": str(session_manifest),
        "session_manifest_hash": manifest.get("manifest_hash", ""),
        "known_roles": validation["known_roles"],
        "role_taxonomy": ROLE_TAXONOMY,
        "promotion_guards": PROMOTION_GUARDS,
        "policy": DEFAULT_POLICY,
        "validation": validation,
    }


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Protocol101 Era Role Policy",
        "",
        f"- Status: `{payload['status']}`",
        f"- Policy hash: `{payload['policy_hash']}`",
        f"- Session manifest hash: `{payload.get('session_manifest_hash', '')}`",
        "",
        "## Policy",
        "",
    ]
    for era, item in payload["policy"].items():
        lines.append(
            f"- `{era}`: roles=`{item['permitted_roles']}`, tier=`{item['evidence_tier']}`"
        )
    lines.extend(["", "## Role Taxonomy", ""])
    for role, meaning in payload["role_taxonomy"].items():
        lines.append(f"- `{role}`: {meaning}")
    lines.extend(["", "## Promotion Guards", ""])
    for name, guard in payload["promotion_guards"].items():
        lines.append(f"- `{name}`: if `{guard['if']}` then `{guard['status']}`.")
    if payload["validation"]["missing_policy_eras"]:
        lines.extend(["", "## Missing Policy Eras", ""])
        lines.extend(f"- `{era}`" for era in payload["validation"]["missing_policy_eras"])
    if payload["validation"]["role_errors"]:
        lines.extend(["", "## Role Errors", ""])
        lines.extend(f"- `{item}`" for item in payload["validation"]["role_errors"])
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The fold scaffold should consume this policy alongside the session era manifest.",
            "- Policy changes should update this artifact without mutating the session facts manifest.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    payload = build_policy_artifact(args.session_manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    policy_path = args.out_dir / "era_role_policy.json"
    summary_path = args.out_dir / "summary.json"
    report_path = args.out_dir / "report.md"
    policy_path.write_text(json.dumps(payload["policy"], indent=2, sort_keys=True) + "\n")
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    report_path.write_text(render_report(payload))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "policy_hash": payload["policy_hash"],
                "report": str(report_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
