"""Build curated Stage 1 snapshot manifests.

This script does not mutate trading runtime state. It classifies local files,
builds Git include and local evidence-bundle path lists, writes a local-only
full manifest, and writes a Git-safe redacted summary.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Iterable


AUDIT_ROOT = Path("v4/audit/autoresearch")
CURRENT_ARTIFACT_DIRS = {
    AUDIT_ROOT / "v4_aplus_hypothesis_101_event_history_policy",
    AUDIT_ROOT / "v4_aplus_hypothesis_075_protocol054_frozen_stack_artifacts",
    AUDIT_ROOT / "v4_aplus_hypothesis_081_q4start_residual_sequence_deterministic_artifacts",
}
SELECTED_AUDIT_DIRS = {
    "formal_validation_governance",
    "foundation_hardening_review",
    "live_no_order_full_action_parity_readiness",
    "project_section_readiness",
    "truth_grounded_replacement_program_v1",
    "untouched_holdout_availability",
    "unified_untouched_holdout_reservation",
    "unified_neural_training_readiness",
    "unified_protocol101_baseline_attachment",
    "tuesday_no_order_evidence_packet",
    "v4_aplus_hypothesis_075_protocol054_frozen_stack_artifacts",
    "v4_aplus_hypothesis_081_q4start_residual_sequence_deterministic_artifacts",
    "v4_aplus_hypothesis_084_protocol081_promotion_readiness",
    "v4_aplus_hypothesis_090_protocol081_strict_shadow_lifecycle",
    "v4_aplus_hypothesis_101_event_history_policy",
    "v4_aplus_hypothesis_102_protocol101_readiness",
    "v4_aplus_hypothesis_103_protocol101_external_audit_readiness",
    "v4_aplus_hypothesis_107_protocol101_q4_2024_external_stress",
    "v4_aplus_hypothesis_109_frozen_protocol101_seed_ensemble",
    "v4_aplus_hypothesis_112_protocol101_money_breakdown",
    "v4_aplus_hypothesis_113_protocol101_trade_charts",
    "v4_aplus_hypothesis_114_protocol101_skeptical_falsification",
    "v4_aplus_hypothesis_115_protocol101_existing_1s_path_audit",
    "v4_aplus_hypothesis_116_protocol101_targeted_1s_request",
    "v4_aplus_hypothesis_117_protocol101_targeted_highres_path_audit",
    "v4_aplus_hypothesis_117_protocol101_targeted_highres_validation",
    "v4_aplus_hypothesis_118_protocol101_shadow_rehearsal",
    "v4_aplus_hypothesis_119_protocol101_live_readiness",
    "v4_aplus_hypothesis_121_protocol101_entry_router_smoke",
    "v4_aplus_hypothesis_122_protocol101_capital_realism",
    "v4_aplus_hypothesis_123_protocol101_order_state_rehearsal",
    "v4_aplus_hypothesis_124_protocol101_live_data_parity_checkpoint",
    "v4_aplus_hypothesis_125_protocol101_pre_tuesday_readiness",
    "v4_aplus_hypothesis_126_protocol101_timing_fragility_hardening",
    "v4_aplus_hypothesis_127_protocol101_live_shadow_schema_hardening",
    "v4_aplus_hypothesis_128_protocol101_paper_risk_gate",
    "v4_aplus_hypothesis_140_ibkr_autostart_prep",
    "v4_aplus_hypothesis_146_ibc_credential_readiness",
    "v4_aplus_hypothesis_147_protocol101_morning_session",
    "v4_aplus_hypothesis_150_protocol101_paper_order_enablement_gate",
    "v4_aplus_hypothesis_155_protocol101_live_timing_evidence",
    "v4_aplus_hypothesis_156_ibkr_autostart_observability",
    "v4_aplus_hypothesis_157_protocol101_daily_ops_monitor",
    "v4_aplus_hypothesis_158_protocol101_live_entry_paper_bridge",
    "v4_aplus_hypothesis_160_protocol101_persistent_paper_trader",
    "v4_aplus_hypothesis_161_may2026_historical_replay",
    "v4_aplus_hypothesis_162_may2026_serial_lifecycle_replay",
    "v4_aplus_hypothesis_163_recent_protocol101_historical_replay",
    "v4_aplus_hypothesis_163_recent_protocol101_serial_lifecycle_replay",
    "v4_aplus_hypothesis_168_protocol163_threshold_replay",
    "v4_aplus_hypothesis_272_fill_model_readiness",
    "v4_aplus_hypothesis_273_model_selection_overfit_risk",
}
AUDIT_META_NAMES = {"report.md", "summary.json", "report.json"}
ARTIFACT_META_NAMES = {
    "manifest.json",
    "summary.json",
    "report.md",
    "report.json",
    "scaler.json",
    "entry_standardizer.json",
    "protocol054_risk_scaler.json",
    "threshold_sweep.json",
}
BUNDLE_EXACT = {
    Path("v4/runtime/protocol101_paper_order_enablement.json"),
    Path("v4/runtime/protocol101_live_index_context.jsonl"),
    Path("v4/logs/paper_trading/2026-05-21/protocol101_persistent-paper_2026-05-21.jsonl"),
    Path("v4/logs/paper_trading/2026-05-21/protocol101_persistent-paper_2026-05-21.csv"),
    AUDIT_ROOT / "v4_aplus_hypothesis_075_protocol054_frozen_stack_artifacts/model_artifacts/train_through_q4_2025_test_q1_2026/seed_11/entry_model.pt",
    AUDIT_ROOT / "v4_aplus_hypothesis_075_protocol054_frozen_stack_artifacts/model_artifacts/train_through_q4_2025_test_q1_2026/seed_11/protocol054_risk_model.pt",
    AUDIT_ROOT / "v4_aplus_hypothesis_081_q4start_residual_sequence_deterministic_artifacts/model_artifacts/train_q1_2025_q2_2025_q3_2025_q4_2025_test_q1_2026/seed_1/model.pt",
    AUDIT_ROOT / "v4_aplus_hypothesis_101_event_history_policy/serial_policy_trades.json",
    AUDIT_ROOT / "protocol101_strategy_forensics_packet_v1/score_calibration.csv",
    AUDIT_ROOT / "protocol101_internal_slot_cost_counterfactual_v1/blocked_protocol101_internal_slot_events.csv",
    AUDIT_ROOT / "protocol101_internal_slot_cost_counterfactual_v1/hypothetical_flat_protocol101_entries.csv",
    AUDIT_ROOT / "protocol101_internal_slot_cost_counterfactual_v1/open_trade_slot_cost_summary.csv",
    AUDIT_ROOT / "protocol101_loss_reversal_full_serial_replay_v1/full_serial_replay_trades.csv",
    AUDIT_ROOT / "v4_aplus_hypothesis_272_fill_model_readiness/live_and_paper_event_inventory.csv",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--git-include-out", type=Path)
    parser.add_argument("--bundle-paths-out", type=Path)
    parser.add_argument("--full-manifest-out", type=Path)
    parser.add_argument("--full-manifest-in", type=Path)
    parser.add_argument("--redacted-manifest-out", type=Path, required=True)
    parser.add_argument("--bundle-archive", type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    if args.full_manifest_in:
        rows = read_manifest(args.full_manifest_in)
        write_redacted_manifest(rows, args.redacted_manifest_out, args.bundle_archive)
        return 0

    git_include = sorted(build_git_include(source))
    dirty_or_untracked = dirty_paths(source)
    explicit = explicit_manifest_paths(source)
    excluded = sorted((dirty_or_untracked | explicit) - set(git_include))

    rows = [classify(source, rel) for rel in excluded if (source / rel).is_file()]
    bundle_paths = sorted(row["path"] for row in rows if row["bundle_action"] == "bundle_only")

    if args.git_include_out:
        write_lines(args.git_include_out, [str(path) for path in git_include])
    if args.bundle_paths_out:
        write_lines(args.bundle_paths_out, bundle_paths)
    if args.full_manifest_out:
        args.full_manifest_out.parent.mkdir(parents=True, exist_ok=True)
        with args.full_manifest_out.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    write_redacted_manifest(rows, args.redacted_manifest_out, args.bundle_archive)
    return 0


def build_git_include(source: Path) -> set[Path]:
    paths: set[Path] = set()
    for rel in [Path(".gitignore"), Path("README.md"), Path("pyproject.toml")]:
        add_if_file(source, paths, rel)

    for root in [
        Path("docs"),
        Path("research_ops"),
        Path("v4/checks"),
        Path("v4/dataset"),
        Path("v4/foundation"),
        Path("v4/ingest"),
        Path("v4/live"),
        Path("v4/model"),
        Path("v4/ops"),
        Path("v4/schema"),
        Path("v4/scripts"),
        Path("v4/sim"),
        Path("v4/tests"),
        Path("v4/docs"),
        Path("v4/promotion"),
    ]:
        add_tree(source, paths, root)
    add_if_file(source, paths, Path("v4/README.md"))
    add_if_file(source, paths, Path("v4/ledger/RESEARCH_LEDGER.md"))
    add_if_file(source, paths, Path("v4/audit/ibkr_live_data_entitlements/report.md"))
    add_if_file(source, paths, Path("v4/audit/ibkr_live_data_entitlements/summary.json"))
    add_selected_audit_metadata(source, paths)
    return paths


def add_tree(source: Path, paths: set[Path], root: Path) -> None:
    base = source / root
    if not base.exists():
        return
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(source)
        if safe_for_git(rel):
            paths.add(rel)


def add_if_file(source: Path, paths: set[Path], rel: Path) -> None:
    if (source / rel).is_file() and safe_for_git(rel):
        paths.add(rel)


def add_selected_audit_metadata(source: Path, paths: set[Path]) -> None:
    root = source / AUDIT_ROOT
    if not root.exists():
        return
    for child in root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        selected = name in SELECTED_AUDIT_DIRS or (name.startswith("protocol101_") and name.endswith("_v1"))
        if not selected:
            continue
        for file_path in child.rglob("*"):
            if not file_path.is_file():
                continue
            rel = file_path.relative_to(source)
            if file_path.name in AUDIT_META_NAMES:
                paths.add(rel)
            elif any(rel == current or current in rel.parents for current in CURRENT_ARTIFACT_DIRS):
                if file_path.name in ARTIFACT_META_NAMES:
                    paths.add(rel)


def safe_for_git(rel: Path) -> bool:
    parts = set(rel.parts)
    if ".git" in parts or "__pycache__" in parts:
        return False
    if rel.name in {".env", ".DS_Store"} or rel.suffix in {".pyc", ".pyo"}:
        return False
    if str(rel).startswith(("v4/runtime/", "v4/logs/", "data/")):
        return False
    if rel.suffix in {".pt", ".parquet", ".pkl", ".jsonl", ".html"}:
        return False
    if rel.suffix in {".md", ".py", ".sh", ".plist", ".json", ".csv", ".txt", ""}:
        return True
    return rel.name == ".gitkeep"


def dirty_paths(source: Path) -> set[Path]:
    paths: set[Path] = set()
    modified = subprocess.check_output(["git", "-C", str(source), "diff", "--name-only", "-z"], text=False)
    untracked = subprocess.check_output(
        ["git", "-C", str(source), "ls-files", "--others", "--exclude-standard", "-z"],
        text=False,
    )
    for payload in (modified, untracked):
        for item in payload.split(b"\0"):
            if not item:
                continue
            paths.add(Path(item.decode("utf-8", errors="surrogateescape")))
    return paths


def _status_paths(source: Path) -> set[Path]:
    """Fallback parser kept for rare status-only paths."""
    out = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain=v1", "-z"], text=False)
    items = out.split(b"\0")
    paths: set[Path] = set()
    i = 0
    while i < len(items):
        item = items[i]
        i += 1
        if not item:
            continue
        text = item.decode("utf-8", errors="surrogateescape")
        status = text[:2]
        path_text = text[3:]
        if status.startswith("R") or status.startswith("C"):
            if i < len(items) and items[i]:
                path_text = items[i].decode("utf-8", errors="surrogateescape")
                i += 1
        paths.add(Path(path_text))
    return paths


def explicit_manifest_paths(source: Path) -> set[Path]:
    roots = [
        Path("data"),
        Path("v4/raw"),
        Path("v4/normalized"),
        Path("v4/normalized_official_context"),
        Path("v4/normalized_official_context_smoke"),
        Path("v4/normalized_official_context_fix_smoke"),
        Path("v4/feature"),
        Path("v4/label"),
        Path("v4/runtime"),
        Path("v4/logs/paper_trading"),
    ]
    paths: set[Path] = {Path(".env"), Path("v4/.env")}
    for root in roots:
        abs_root = source / root
        if not abs_root.exists():
            continue
        for path in abs_root.rglob("*"):
            if path.is_file():
                paths.add(path.relative_to(source))
    for rel in BUNDLE_EXACT:
        if (source / rel).is_file():
            paths.add(rel)
    return paths


def classify(source: Path, rel: Path) -> dict[str, object]:
    path = source / rel
    size = path.stat().st_size
    category = category_for(rel)
    sensitivity = sensitivity_for(rel, category)
    bundle_action = "excluded_entirely"
    reason = "not_selected_for_curated_git_snapshot"

    if rel in {Path(".env"), Path("v4/.env")}:
        bundle_action = "sensitive_excluded"
        reason = "local_env_file_never_copied_or_committed"
    elif rel in BUNDLE_EXACT or is_protocol101_model_binary(rel):
        bundle_action = "bundle_only"
        reason = "needed_for_local_control_reconstruction_but_not_git"
    elif category in {"paid_data", "raw_data"}:
        reason = "paid_or_rebuildable_market_data_excluded_from_default_bundle"
    elif category == "paper_log":
        reason = "paper_log_excluded_from_git"
    elif category == "runtime_state":
        reason = "runtime_state_excluded_from_git"
    elif category == "model_binary":
        reason = "model_binary_excluded_from_git"
    elif category == "generated_audit":
        reason = "generated_audit_artifact_excluded_from_git"

    return {
        "path": str(rel),
        "size_bytes": size,
        "sha256": sha256_file(path),
        "reason_excluded": reason,
        "category": category,
        "bundle_action": bundle_action,
        "sensitivity": sensitivity,
    }


def is_protocol101_model_binary(rel: Path) -> bool:
    return (
        rel.suffix == ".pt"
        and str(rel).startswith("v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy/model_artifacts/")
        and rel.name == "model.pt"
    )


def category_for(rel: Path) -> str:
    text = str(rel)
    if text.startswith(("data/", "v4/normalized_official_context", "v4/normalized/")):
        return "paid_data"
    if text.startswith(("v4/raw/", "v4/feature/", "v4/label/")):
        return "raw_data"
    if text.startswith("v4/logs/paper_trading/"):
        return "paper_log"
    if text.startswith("v4/runtime/") or rel.name.endswith(".lock") or rel.name.startswith(".deploy-state"):
        return "runtime_state"
    if rel.suffix == ".pt":
        return "model_binary"
    if text.startswith("v4/audit/") or text.startswith("v3/artifacts/"):
        return "generated_audit"
    return "unknown"


def sensitivity_for(rel: Path, category: str) -> str:
    text = str(rel)
    if rel in {Path(".env"), Path("v4/.env")} or "credential" in text.lower():
        return "sensitive_path"
    if category == "paper_log":
        return "paper_log"
    if category == "runtime_state":
        return "runtime_or_account"
    if category in {"paid_data", "raw_data"}:
        return "paid_or_licensed_data"
    return "public_path"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_redacted_manifest(rows: list[dict[str, object]], out: Path, bundle_archive: Path | None) -> None:
    categories = Counter(str(row["category"]) for row in rows)
    bytes_by_category: dict[str, int] = defaultdict(int)
    actions = Counter(str(row["bundle_action"]) for row in rows)
    for row in rows:
        bytes_by_category[str(row["category"])] += int(row["size_bytes"])
    bundle_sha = "PENDING"
    bundle_path = str(bundle_archive) if bundle_archive else "PENDING"
    if bundle_archive and bundle_archive.exists():
        bundle_sha = sha256_file(bundle_archive)

    lines = [
        "# Stage 1 Redacted Excluded Artifact Manifest",
        "",
        "This Git-tracked manifest is intentionally redacted. The local-only full manifest contains exact paths and hashes for sensitive/runtime/paper/paid-data artifacts.",
        "",
        f"- Evidence bundle path: `{bundle_path}`",
        f"- Evidence bundle sha256: `{bundle_sha}`",
        f"- Bundle-only file count: `{actions.get('bundle_only', 0)}`",
        f"- Excluded-entirely count: `{actions.get('excluded_entirely', 0)}`",
        f"- Sensitive-excluded count: `{actions.get('sensitive_excluded', 0)}`",
        "",
        "## Counts By Category",
        "",
        "| category | file_count | total_bytes |",
        "|---|---:|---:|",
    ]
    for category in sorted(categories):
        lines.append(f"| `{category}` | {categories[category]} | {bytes_by_category[category]} |")
    lines.extend(
        [
            "",
            "## Redaction Rules",
            "",
            "- No `.env` contents or hashes are included here.",
            "- No credential material is included here.",
            "- No paper-log contents are included here.",
            "- No account identifiers are included here.",
            "- No raw paid-data contents are included here.",
        ]
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
