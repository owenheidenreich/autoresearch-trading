"""Derive a pass-only Protocol101 training-scope acceptance registry.

The owned-raw verifier should keep reporting every session in a date span,
including report-only and hard-fail days. Training, however, must consume only
placeable pass sessions through a hash-stable registry. This helper narrows a
full verifier registry into an explicit training-scope registry without
weakening any session-level checks.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from v4.scripts.run_protocol101_owned_raw_acceptance_verifier import (
    compute_registry_hash,
    fold_placement_predicate,
    stable_hash,
)


DEFAULT_SOURCE_REGISTRY = Path(
    "v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_15mo_acceptance/summary.json"
)
DEFAULT_ERA_MANIFEST = Path("v4/audit/autoresearch/protocol101_session_era_manifest/summary.json")
DEFAULT_ROLE_POLICY = Path(
    "v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_training_role_policy/summary.json"
)
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_15mo_training_scope_acceptance"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-registry", type=Path, default=DEFAULT_SOURCE_REGISTRY)
    parser.add_argument("--era-manifest", type=Path, default=DEFAULT_ERA_MANIFEST)
    parser.add_argument("--role-policy", type=Path, default=DEFAULT_ROLE_POLICY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--placement-role",
        default="diagnostics_only",
        help="Role used only for the embedded placement-predicate audit.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def excluded_session_record(record: dict[str, Any]) -> dict[str, Any]:
    failed_checks = [
        name for name, passed in (record.get("checks") or {}).items() if passed is False
    ]
    return {
        "session": str(record.get("session") or ""),
        "source_status": str(record.get("status") or "UNKNOWN"),
        "report_only_reason": str(record.get("report_only_reason") or ""),
        "failed_checks": failed_checks,
        "reason": "not_training_placeable_acceptance_status",
    }


def recompute_label_outcomes(records: list[dict[str, Any]]) -> dict[str, int]:
    outcomes: dict[str, int] = {}
    for record in records:
        for reason, count in (record.get("label_spot_check", {}).get("outcome_reasons") or {}).items():
            outcomes[str(reason)] = outcomes.get(str(reason), 0) + int(count)
    return dict(sorted(outcomes.items()))


def build_training_scope_registry(
    *,
    source_registry: dict[str, Any],
    source_registry_path: Path,
    era_manifest: dict[str, Any],
    era_manifest_path: Path,
    role_policy: dict[str, Any],
    role_policy_path: Path,
    placement_role: str = "diagnostics_only",
) -> dict[str, Any]:
    source_sessions = [
        record for record in source_registry.get("sessions") or [] if isinstance(record, dict)
    ]
    included = [record for record in source_sessions if record.get("status") == "pass"]
    excluded = [excluded_session_record(record) for record in source_sessions if record.get("status") != "pass"]
    excluded_hash = stable_hash(excluded)
    source_hash = str(source_registry.get("registry_hash") or "")
    governance_checks = dict(source_registry.get("governance_checks") or {})
    governance_checks.update(
        {
            "source_registry_path": str(source_registry_path),
            "source_registry_status": str(source_registry.get("status") or "missing"),
            "source_registry_hash": source_hash,
            "training_scope_filter": "include_acceptance_status_pass_only",
            "training_scope_excluded_session_count": len(excluded),
            "training_scope_excluded_fail_count": sum(
                1 for item in excluded if item["source_status"] == "fail"
            ),
            "training_scope_excluded_report_only_count": sum(
                1 for item in excluded if item["source_status"] == "report_only"
            ),
            "training_scope_excluded_sessions_hash": excluded_hash,
        }
    )
    placement = [
        fold_placement_predicate(
            session=str(record.get("session")),
            role=str(placement_role),
            era_manifest=era_manifest,
            role_policy=role_policy,
            acceptance_registry={"sessions": included},
        )
        for record in included
    ]
    batch_checks = dict(source_registry.get("batch_checks") or {})
    batch_checks["training_scope_all_included_sessions_pass"] = all(
        record.get("status") == "pass" for record in included
    )
    batch_checks["training_scope_has_included_sessions"] = bool(included)
    payload = {
        **{
            key: value
            for key, value in source_registry.items()
            if key
            not in {
                "status",
                "batch_id",
                "session_count",
                "pass_count",
                "report_only_count",
                "fail_count",
                "governance_checks",
                "batch_checks",
                "batch_label_outcome_reasons",
                "sessions",
                "placement_predicates",
                "registry_hash",
                "era_manifest_hash",
                "role_policy_hash",
            }
        },
        "schema_version": "Protocol101OwnedRawAcceptanceRegistryV3_5TrainingScopeV1",
        "status": "pass" if included and all(batch_checks.values()) else "fail",
        "batch_id": f"{source_registry.get('batch_id', 'owned_raw_acceptance')}_training_scope_pass_only",
        "session_count": len(included),
        "pass_count": len(included),
        "report_only_count": 0,
        "fail_count": 0 if included and all(record.get("status") == "pass" for record in included) else 1,
        "governance_checks": governance_checks,
        "batch_checks": batch_checks,
        "batch_label_outcome_reasons": recompute_label_outcomes(included),
        "sessions": included,
        "placement_predicates": placement,
        "era_manifest_hash": stable_hash(era_manifest),
        "role_policy_hash": stable_hash(role_policy),
        "training_scope_filter": {
            "source_registry_path": str(source_registry_path),
            "source_registry_status": str(source_registry.get("status") or "missing"),
            "source_registry_hash": source_hash,
            "include_statuses": ["pass"],
            "excluded_session_count": len(excluded),
            "excluded_sessions_hash": excluded_hash,
            "excluded_sessions": excluded,
            "era_manifest_path": str(era_manifest_path),
            "role_policy_path": str(role_policy_path),
            "placement_role": str(placement_role),
        },
    }
    payload["registry_hash"] = compute_registry_hash(payload)
    return payload


def render_report(payload: dict[str, Any]) -> str:
    scope = payload.get("training_scope_filter") or {}
    lines = [
        "# Protocol101 Training-Scope Acceptance Registry",
        "",
        "## Decision",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Included pass sessions: `{payload.get('pass_count')}`",
        f"- Excluded sessions: `{scope.get('excluded_session_count')}`",
        f"- Source registry status: `{scope.get('source_registry_status')}`",
        f"- Source registry hash: `{scope.get('source_registry_hash')}`",
        f"- Training-scope registry hash: `{payload.get('registry_hash')}`",
        "",
        "## Exclusions",
        "",
    ]
    exclusions = scope.get("excluded_sessions") or []
    if exclusions:
        for item in exclusions:
            failed = ",".join(item.get("failed_checks") or [])
            lines.append(
                f"- `{item.get('session')}`: source_status=`{item.get('source_status')}`, "
                f"report_only_reason=`{item.get('report_only_reason')}`, failed_checks=`{failed}`"
            )
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- This artifact narrows training inputs; it does not alter verifier thresholds.",
            "- Excluded sessions are not placeable for train, validation, diagnostic gates, or uplift claims.",
            "- Model training, threshold selection, broker calls, and paper-submit remain false.",
        ]
    )
    return "\n".join(lines) + "\n"


def flat_registry_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in payload.get("sessions") or []:
        rows.append(
            {
                "session": record.get("session"),
                "status": record.get("status"),
                "processed_path": (record.get("processed") or {}).get("processed_path"),
                "processed_sha256": (record.get("processed") or {}).get("processed_sha256"),
                "neural_rows": (record.get("processed") or {}).get("neural_rows"),
                "feature_contract_version": (record.get("processed") or {}).get(
                    "feature_contract_version"
                ),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    source_registry = load_json(args.source_registry)
    era_manifest = load_json(args.era_manifest)
    role_policy = load_json(args.role_policy)
    payload = build_training_scope_registry(
        source_registry=source_registry,
        source_registry_path=args.source_registry,
        era_manifest=era_manifest,
        era_manifest_path=args.era_manifest,
        role_policy=role_policy,
        role_policy_path=args.role_policy,
        placement_role=args.placement_role,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "acceptance_registry.json").write_text(
        json.dumps(payload.get("sessions") or [], indent=2, sort_keys=True) + "\n"
    )
    write_csv(args.out_dir / "acceptance_registry.csv", flat_registry_rows(payload))
    (args.out_dir / "fold_placement_predicates.json").write_text(
        json.dumps(payload.get("placement_predicates") or [], indent=2, sort_keys=True) + "\n"
    )
    (args.out_dir / "report.md").write_text(render_report(payload))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "session_count": payload["session_count"],
                "pass_count": payload["pass_count"],
                "excluded_session_count": payload["training_scope_filter"]["excluded_session_count"],
                "registry_hash": payload["registry_hash"],
                "report": str(args.out_dir / "report.md"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
