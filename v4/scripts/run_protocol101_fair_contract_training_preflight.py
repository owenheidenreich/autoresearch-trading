"""Preflight Protocol101 retraining readiness on the fair live contract.

This is an offline inspection tool. It does not train a model, tune thresholds,
contact vendors, call broker APIs, change runtime defaults, or enable paper
orders. Its purpose is to decide whether the certified
protocol101-live-v2-microstructure-masked Q1 data is clean enough to be used as
the first canonical retraining substrate.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from v4.live.protocol101_feature_contract import FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED
from v4.live.protocol101_synchronization import Protocol101FairContractTrainingPreflightV1
from v4.scripts.run_protocol101_owned_raw_acceptance_verifier import compute_registry_hash


DEFAULT_PROCESSED_DIR = Path(
    "data/processed/spxw_0dte_neural_q1_2026_protocol101_live_v2_microstructure_masked_greekgate"
)
DEFAULT_NORMALIZED_DIR = Path(
    "v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_greekgate_q1_2026_build/normalized"
)
DEFAULT_SYNC_ROOT = Path("v4/audit/autoresearch/protocol101_synchronization_resolution")
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_training_preflight"
)

ISO_DATE = r"\d{4}-\d{2}-\d{2}"
EXACT_PROCESSED_RE = re.compile(rf"^({ISO_DATE})\.pkl$")
LOOSE_PROCESSED_RE = re.compile(rf"^({ISO_DATE})(?: .+)?\.pkl$")
NORMALIZED_RE = re.compile(rf"^databento_spxw_0dte_({ISO_DATE})(?:_official_context)?\.parquet$")
FORBIDDEN_RUNTIME_FEATURE_FRAGMENTS = (
    "candidate_pnl",
    "candidate_exit",
    "best_exit",
    "future",
    "label",
    "pnl_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    parser.add_argument("--sync-root", type=Path, default=DEFAULT_SYNC_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--feature-contract", default=FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED)
    parser.add_argument(
        "--acceptance-registry",
        type=Path,
        default=None,
        help=(
            "Optional pass-only governed acceptance registry used to narrow the canonical "
            "training manifest to sessions accepted for placement."
        ),
    )
    return parser.parse_args()


def load_json_optional(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_ts(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def processed_file_inventory(processed_dir: Path) -> dict[str, Any]:
    files = sorted(processed_dir.glob("*.pkl")) if processed_dir.exists() else []
    exact: dict[str, Path] = {}
    loose_by_date: dict[str, list[Path]] = defaultdict(list)
    unmatched: list[str] = []
    for path in files:
        exact_match = EXACT_PROCESSED_RE.match(path.name)
        loose_match = LOOSE_PROCESSED_RE.match(path.name)
        if exact_match:
            exact[exact_match.group(1)] = path
            loose_by_date[exact_match.group(1)].append(path)
        elif loose_match:
            loose_by_date[loose_match.group(1)].append(path)
        else:
            unmatched.append(str(path))
    duplicate_or_ambiguous = {
        date: [str(path) for path in paths]
        for date, paths in sorted(loose_by_date.items())
        if len(paths) > 1
    }
    return {
        "exists": processed_dir.exists(),
        "file_count": len(files),
        "exact_session_count": len(exact),
        "exact_dates": sorted(exact),
        "exact_files": {date: str(path) for date, path in sorted(exact.items())},
        "duplicate_or_ambiguous": duplicate_or_ambiguous,
        "unmatched_files": unmatched,
    }


def normalized_file_inventory(normalized_dir: Path) -> dict[str, Any]:
    files = sorted(normalized_dir.glob("*.parquet")) if normalized_dir.exists() else []
    by_date: dict[str, set[str]] = defaultdict(set)
    unmatched: list[str] = []
    for path in files:
        match = NORMALIZED_RE.match(path.name)
        if not match:
            unmatched.append(str(path))
            continue
        date = match.group(1)
        kind = "official_context" if path.name.endswith("_official_context.parquet") else "base"
        by_date[date].add(kind)
    missing_base = [date for date, kinds in sorted(by_date.items()) if "base" not in kinds]
    missing_official_context = [
        date for date, kinds in sorted(by_date.items()) if "official_context" not in kinds
    ]
    return {
        "exists": normalized_dir.exists(),
        "file_count": len(files),
        "session_count": len(by_date),
        "dates": sorted(by_date),
        "missing_base": missing_base,
        "missing_official_context": missing_official_context,
        "unmatched_files": unmatched,
    }


def feature_name_violations(row: dict[str, Any]) -> list[str]:
    names = row.get("feature_names") or []
    if isinstance(names, (list, tuple)):
        iterable = list(names)
    elif hasattr(names, "tolist"):
        iterable = list(names.tolist())
    else:
        return ["feature_names_not_list"]
    bad = []
    for name in iterable:
        text = str(name).lower()
        if any(fragment in text for fragment in FORBIDDEN_RUNTIME_FEATURE_FRAGMENTS):
            bad.append(str(name))
    return bad


def inspect_processed_rows(
    exact_files: dict[str, str],
    *,
    feature_contract: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    session_rows: list[dict[str, Any]] = []
    version_counter: Counter[str] = Counter()
    row_count_values: list[int] = []
    future_quote_rows = 0
    future_context_rows = 0
    missing_timestamp_rows = 0
    forbidden_feature_rows = 0
    context_lags: list[float] = []
    quote_lags: list[float] = []
    load_errors: dict[str, str] = {}
    first_decisions: dict[str, str] = {}
    last_decisions: dict[str, str] = {}

    for session, file_text in sorted(exact_files.items()):
        path = Path(file_text)
        try:
            with path.open("rb") as handle:
                rows = pickle.load(handle)
        except Exception as exc:
            load_errors[session] = f"{type(exc).__name__}: {exc}"
            continue
        if not isinstance(rows, list):
            load_errors[session] = f"expected list, got {type(rows).__name__}"
            continue
        row_count_values.append(len(rows))
        session_versions: Counter[str] = Counter()
        session_forbidden = 0
        session_future_quote = 0
        session_future_context = 0
        session_missing_ts = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            version = str(row.get("feature_contract_version") or "UNKNOWN")
            version_counter[version] += 1
            session_versions[version] += 1
            decision = parse_ts(row.get("decision_time"))
            quote = parse_ts(row.get("source_quote_time"))
            context = parse_ts(row.get("source_context_time"))
            if decision is None or quote is None or context is None:
                missing_timestamp_rows += 1
                session_missing_ts += 1
                continue
            first_decisions.setdefault(session, decision.isoformat())
            last_decisions[session] = decision.isoformat()
            if quote > decision:
                future_quote_rows += 1
                session_future_quote += 1
            if context > decision:
                future_context_rows += 1
                session_future_context += 1
            quote_lags.append((decision - quote).total_seconds() / 60.0)
            context_lags.append((decision - context).total_seconds() / 60.0)
            if feature_name_violations(row):
                forbidden_feature_rows += 1
                session_forbidden += 1
        session_rows.append(
            {
                "session": session,
                "file": file_text,
                "rows": len(rows),
                "versions": json.dumps(dict(sorted(session_versions.items())), sort_keys=True),
                "first_decision_utc": first_decisions.get(session),
                "last_decision_utc": last_decisions.get(session),
                "future_quote_rows": session_future_quote,
                "future_context_rows": session_future_context,
                "missing_timestamp_rows": session_missing_ts,
                "forbidden_feature_rows": session_forbidden,
            }
        )

    expected_version_rows = int(version_counter.get(feature_contract, 0))
    total_version_rows = sum(version_counter.values())
    checks = {
        "load_errors": load_errors,
        "processed_sessions_inspected": len(session_rows),
        "row_count_min": min(row_count_values) if row_count_values else 0,
        "row_count_max": max(row_count_values) if row_count_values else 0,
        "row_count_total": sum(row_count_values),
        "version_counts": dict(sorted(version_counter.items())),
        "all_rows_expected_feature_contract": total_version_rows > 0
        and expected_version_rows == total_version_rows,
        "future_quote_rows": future_quote_rows,
        "future_context_rows": future_context_rows,
        "missing_timestamp_rows": missing_timestamp_rows,
        "forbidden_runtime_feature_rows": forbidden_feature_rows,
        "quote_lag_minutes_median": float(pd.Series(quote_lags).median()) if quote_lags else None,
        "context_lag_minutes_median": float(pd.Series(context_lags).median()) if context_lags else None,
    }
    return session_rows, checks


def processed_file_brief(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            rows = pickle.load(handle)
    except Exception as exc:
        return {"file": str(path), "load_error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(rows, list):
        return {"file": str(path), "load_error": f"expected list, got {type(rows).__name__}"}
    first = rows[0] if rows and isinstance(rows[0], dict) else {}
    label_names = first.get("label_names") or []
    labels_net = first.get("labels_net_pnl")
    labels_mid = first.get("labels_mid_pnl")
    return {
        "file": str(path),
        "rows": len(rows),
        "feature_contract_version": first.get("feature_contract_version"),
        "first_decision_time": str(first.get("decision_time")),
        "last_decision_time": str((rows[-1] if rows and isinstance(rows[-1], dict) else {}).get("decision_time")),
        "label_names_count": len(label_names) if hasattr(label_names, "__len__") else None,
        "label_names": list(label_names) if isinstance(label_names, (list, tuple)) else [],
        "labels_net_pnl_shape": list(getattr(labels_net, "shape", [])),
        "labels_mid_pnl_shape": list(getattr(labels_mid, "shape", [])),
    }


def duplicate_processed_file_details(processed: dict[str, Any]) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    exact_files = processed.get("exact_files") or {}
    for date, paths in (processed.get("duplicate_or_ambiguous") or {}).items():
        canonical = Path(exact_files.get(date, ""))
        for path_text in paths:
            path = Path(path_text)
            if path == canonical:
                continue
            details.append(
                {
                    "session": date,
                    "canonical": processed_file_brief(canonical),
                    "duplicate": processed_file_brief(path),
                }
            )
    return details


def acceptance_session_filter(
    registry: dict[str, Any] | None,
) -> tuple[set[str] | None, dict[str, Any], list[str]]:
    if not registry:
        return None, {"enabled": False}, []
    blockers: list[str] = []
    embedded_hash = str(registry.get("registry_hash") or "")
    computed_hash = compute_registry_hash(registry) if embedded_hash else ""
    if registry.get("status") != "pass":
        blockers.append("acceptance_registry_status_not_pass")
    if not embedded_hash:
        blockers.append("acceptance_registry_hash_missing")
    elif computed_hash != embedded_hash:
        blockers.append("acceptance_registry_hash_mismatch")
    records = [
        record for record in registry.get("sessions") or [] if isinstance(record, dict)
    ]
    accepted = {str(record.get("session")) for record in records if record.get("status") == "pass"}
    accepted.discard("")
    if not accepted:
        blockers.append("acceptance_registry_has_no_pass_sessions")
    by_status: Counter[str] = Counter(str(record.get("status") or "UNKNOWN") for record in records)
    return accepted, {
        "enabled": True,
        "status": registry.get("status"),
        "registry_hash": embedded_hash,
        "computed_registry_hash": computed_hash,
        "session_count": len(records),
        "pass_session_count": len(accepted),
        "status_counts": dict(sorted(by_status.items())),
    }, blockers


def build_packet(args: argparse.Namespace) -> tuple[Protocol101FairContractTrainingPreflightV1, list[dict[str, Any]]]:
    processed = processed_file_inventory(args.processed_dir)
    normalized = normalized_file_inventory(args.normalized_dir)
    session_rows, row_checks = inspect_processed_rows(
        processed["exact_files"],
        feature_contract=args.feature_contract,
    )
    live_gate = load_json_optional(args.sync_root / "historical_non_inferiority_gate.json") or {}
    fair_timing_gate = load_json_optional(
        args.sync_root / "historical_non_inferiority_gate_fair_timing_context_lag0_decision_plus1.json"
    ) or {}
    single_day = load_json_optional(args.sync_root / "single_day_synchronization_gate.json") or {}
    paper_gate = load_json_optional(args.sync_root / "paper_readiness_gate.json") or {}
    fair_policy = load_json_optional(args.sync_root / "fair_game_policy.json") or {}

    processed_dates = set(processed["exact_dates"])
    normalized_dates = set(normalized["dates"])
    acceptance_registry_path = getattr(args, "acceptance_registry", None)
    acceptance_registry = (
        load_json_optional(acceptance_registry_path)
        if acceptance_registry_path
        else None
    )
    accepted_sessions, acceptance_filter, acceptance_blockers = acceptance_session_filter(
        acceptance_registry
    )
    missing_normalized_for_processed = sorted(processed_dates - normalized_dates)
    extra_normalized_without_processed = sorted(normalized_dates - processed_dates)
    duplicate_paths = {
        path
        for paths in processed["duplicate_or_ambiguous"].values()
        for path in paths
        if not EXACT_PROCESSED_RE.match(Path(path).name)
    }
    canonical_dates = [
        date
        for date in processed["exact_dates"]
        if accepted_sessions is None or date in accepted_sessions
    ]
    excluded_acceptance_sessions = [
        {
            "session": date,
            "processed_file": processed["exact_files"][date],
            "reason": "not_present_as_pass_in_acceptance_registry",
        }
        for date in processed["exact_dates"]
        if accepted_sessions is not None and date not in accepted_sessions
    ]
    canonical_processed_files = [
        {
            "session": date,
            "processed_file": processed["exact_files"][date],
            "normalized_base_file": str(
                args.normalized_dir / f"databento_spxw_0dte_{date}.parquet"
            ),
            "normalized_official_context_file": str(
                args.normalized_dir / f"databento_spxw_0dte_{date}_official_context.parquet"
            ),
        }
        for date in canonical_dates
    ]
    excluded_processed_files = [
        {
            "file": path,
            "reason": "duplicate_or_ambiguous_noncanonical_session_file",
        }
        for path in sorted(duplicate_paths)
    ] + [
        {
            "file": path,
            "reason": "unmatched_processed_filename",
        }
        for path in processed["unmatched_files"]
    ]
    duplicate_details = duplicate_processed_file_details(processed)

    blockers: list[str] = []
    if not processed["exists"]:
        blockers.append("processed_dir_missing")
    if not normalized["exists"]:
        blockers.append("normalized_dir_missing")
    if processed["duplicate_or_ambiguous"]:
        blockers.append("duplicate_or_ambiguous_processed_session_files")
    if processed["unmatched_files"]:
        blockers.append("unmatched_processed_files")
    if normalized["missing_base"]:
        blockers.append("normalized_base_files_missing")
    if normalized["missing_official_context"]:
        blockers.append("normalized_official_context_files_missing")
    if missing_normalized_for_processed:
        blockers.append("processed_dates_missing_normalized_rows")
    blockers.extend(acceptance_blockers)
    if accepted_sessions is not None and not canonical_processed_files:
        blockers.append("no_canonical_processed_files_after_acceptance_filter")
    if row_checks["load_errors"]:
        blockers.append("processed_load_errors")
    if not row_checks["all_rows_expected_feature_contract"]:
        blockers.append("unexpected_feature_contract_versions")
    if row_checks["future_quote_rows"]:
        blockers.append("future_quote_timestamp_rows")
    if row_checks["future_context_rows"]:
        blockers.append("future_context_timestamp_rows")
    if row_checks["forbidden_runtime_feature_rows"]:
        blockers.append("forbidden_runtime_feature_names")

    live_contract_failed = live_gate.get("status") == "fail"
    fair_timing_failed = fair_timing_gate.get("status") == "fail"
    single_day_passed = single_day.get("status") == "pass"
    paper_blocked = paper_gate.get("status") == "blocked"
    same_minute_rejected = fair_policy.get("status") == "reject_same_minute_timing_repair_continue_other_causal_routes"

    if blockers:
        status = "blocked_dataset_cleanup_required"
        decision = "clean_canonical_fair_contract_dataset_before_any_training_design_or_paper_submit"
    elif not (live_contract_failed and fair_timing_failed and single_day_passed and same_minute_rejected):
        status = "blocked_synchronization_evidence_incomplete"
        decision = "do_not_train_or_paper_submit_until_synchronization_evidence_is_consistent"
    else:
        status = "ready_for_owner_authorized_fair_contract_training_design"
        decision = "prepare_retraining_on_protocol101_live_v2_microstructure_masked"

    dataset_checks = {
        "processed_dir": str(args.processed_dir),
        "normalized_dir": str(args.normalized_dir),
        "processed": {
            key: value
            for key, value in processed.items()
            if key not in {"exact_files", "exact_dates"}
        },
        "processed_exact_session_count": processed["exact_session_count"],
        "processed_exact_dates_first": processed["exact_dates"][:5],
        "processed_exact_dates_last": processed["exact_dates"][-5:],
        "acceptance_registry": acceptance_filter,
        "normalized": normalized,
        "missing_normalized_for_processed": missing_normalized_for_processed,
        "extra_normalized_without_processed": extra_normalized_without_processed,
        "canonical_processed_files": canonical_processed_files,
        "excluded_processed_files": excluded_processed_files,
        "excluded_acceptance_sessions": excluded_acceptance_sessions,
        "duplicate_processed_file_details": duplicate_details,
        "row_checks": row_checks,
    }
    synchronization_checks = {
        "historical_non_inferiority_status": live_gate.get("status", "missing"),
        "fair_timing_non_inferiority_status": fair_timing_gate.get("status", "missing"),
        "single_day_synchronization_status": single_day.get("status", "missing"),
        "paper_readiness_status": paper_gate.get("status", "missing"),
        "fair_game_policy_status": fair_policy.get("status", "missing"),
        "frozen_protocol101_paper_ready": False,
        "paper_blocked": paper_blocked,
    }
    next_actions = []
    if "duplicate_or_ambiguous_processed_session_files" in blockers:
        next_actions.append(
            "Archive or remove duplicate/ambiguous processed session files before any training runner is allowed to glob this dataset."
        )
    else:
        next_actions.append(
            "Require future fair-contract training runners to load canonical_processed_session_manifest.json instead of globbing the processed directory."
        )
    next_actions.extend([
        "Keep paper-submit disabled; this preflight only validates offline training inputs.",
        (
            "Preregister a new candidate trained on "
            f"{args.feature_contract} with chronological purging and embargoing."
        ),
        "Use July 6+ recorder sessions only as confirmation/out-of-sample evidence, not as a reason to revive same-minute completed-bar semantics.",
    ])
    packet = Protocol101FairContractTrainingPreflightV1(
        status=status,
        selected_feature_contract=args.feature_contract,
        model_training_authorized=False,
        paper_submit_allowed=False,
        decision=decision,
        dataset_checks=dataset_checks,
        synchronization_checks=synchronization_checks,
        blockers=blockers,
        next_actions=next_actions,
    )
    return packet, session_rows


def render_report(packet: Protocol101FairContractTrainingPreflightV1) -> str:
    data = packet.dataset_checks
    row_checks = data["row_checks"]
    sync = packet.synchronization_checks
    processed = data["processed"]
    normalized = data["normalized"]
    acceptance = data.get("acceptance_registry") or {"enabled": False}
    lines = [
        "# Protocol101 Fair Contract Training Preflight",
        "",
        "## Decision",
        "",
        f"- Status: `{packet.status}`",
        f"- Decision: `{packet.decision}`",
        f"- Selected feature contract: `{packet.selected_feature_contract}`",
        f"- Model training authorized: `{str(packet.model_training_authorized).lower()}`",
        f"- Paper-submit allowed: `{str(packet.paper_submit_allowed).lower()}`",
        "- Broker endpoint called: `false`",
        "- Threshold tuning: `false`",
        "",
        "## Dataset Checks",
        "",
        f"- Processed sessions: `{data['processed_exact_session_count']}` exact session files from `{data['processed_dir']}`.",
        f"- Processed file count: `{processed['file_count']}`.",
        f"- Duplicate/ambiguous processed files: `{len(processed['duplicate_or_ambiguous'])}` date(s).",
        f"- Normalized sessions: `{normalized['session_count']}` from `{data['normalized_dir']}`.",
        f"- Normalized file count: `{normalized['file_count']}`.",
        f"- Acceptance-registry filter enabled: `{str(acceptance.get('enabled')).lower()}`.",
        f"- Acceptance-registry pass sessions: `{acceptance.get('pass_session_count', 'n/a')}`.",
        f"- Processed rows inspected: `{row_checks['row_count_total']}`.",
        f"- Row count min/max by session: `{row_checks['row_count_min']}` / `{row_checks['row_count_max']}`.",
        f"- Feature contract versions: `{row_checks['version_counts']}`.",
        f"- Future quote/context rows: `{row_checks['future_quote_rows']}` / `{row_checks['future_context_rows']}`.",
        f"- Forbidden runtime feature-name rows: `{row_checks['forbidden_runtime_feature_rows']}`.",
        f"- Median quote/context lag minutes: `{row_checks['quote_lag_minutes_median']}` / `{row_checks['context_lag_minutes_median']}`.",
        f"- Canonical manifest included/excluded files: `{len(data['canonical_processed_files'])}` / `{len(data['excluded_processed_files'])}`.",
        f"- Canonical manifest acceptance exclusions: `{len(data.get('excluded_acceptance_sessions') or [])}`.",
        "- Canonical manifest status: `usable_only_if_training_runner_requires_this_manifest`.",
        "",
        "## Synchronization Checks",
        "",
        f"- Historical non-inferiority: `{sync['historical_non_inferiority_status']}`.",
        f"- Fair-timing non-inferiority: `{sync['fair_timing_non_inferiority_status']}`.",
        f"- Single-day synchronization: `{sync['single_day_synchronization_status']}`.",
        f"- Paper readiness: `{sync['paper_readiness_status']}`.",
        f"- Fair-game policy: `{sync['fair_game_policy_status']}`.",
        "",
        "## Blockers",
        "",
    ]
    lines.extend(f"- `{item}`" for item in packet.blockers) if packet.blockers else lines.append("- None.")
    duplicate_details = data.get("duplicate_processed_file_details") or []
    if duplicate_details:
        lines.extend(["", "## Duplicate Details", ""])
        for item in duplicate_details:
            canonical = item.get("canonical") or {}
            duplicate = item.get("duplicate") or {}
            lines.append(
                f"- `{item.get('session')}` duplicate `{duplicate.get('file')}` has "
                f"`{duplicate.get('label_names_count')}` labels versus canonical "
                f"`{canonical.get('label_names_count')}` labels."
            )
    acceptance_exclusions = data.get("excluded_acceptance_sessions") or []
    if acceptance_exclusions:
        lines.extend(["", "## Acceptance Exclusions", ""])
        for item in acceptance_exclusions[:25]:
            lines.append(
                f"- `{item.get('session')}`: `{item.get('reason')}`"
            )
        if len(acceptance_exclusions) > 25:
            lines.append(f"- ... `{len(acceptance_exclusions) - 25}` more.")
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {item}" for item in packet.next_actions)
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    packet, session_rows = build_packet(args)
    (args.out_dir / "summary.json").write_text(
        json.dumps(packet.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    (args.out_dir / "report.md").write_text(render_report(packet))
    write_csv(args.out_dir / "processed_session_audit.csv", session_rows)
    manifest = {
        "schema_version": "Protocol101FairContractCanonicalSessionManifestV1",
        "selected_feature_contract": packet.selected_feature_contract,
        "status": "usable_only_if_training_runner_requires_this_manifest",
        "included_sessions": packet.dataset_checks["canonical_processed_files"],
        "excluded_processed_files": packet.dataset_checks["excluded_processed_files"],
        "excluded_acceptance_sessions": packet.dataset_checks.get("excluded_acceptance_sessions") or [],
        "acceptance_registry": packet.dataset_checks.get("acceptance_registry") or {"enabled": False},
        "model_training_authorized": False,
        "paper_submit_allowed": False,
    }
    (args.out_dir / "canonical_processed_session_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "status": packet.status,
        "decision": packet.decision,
        "blockers": packet.blockers,
        "report": str(args.out_dir / "report.md"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
