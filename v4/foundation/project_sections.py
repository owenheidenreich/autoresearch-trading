"""Section contracts for safe model-improvement work.

This module is intentionally lightweight and read-only. It turns the project
shape into explicit checks so Section 1/2 drift is visible before model work
starts again.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


PASS = "pass"
WARN = "warn"
BLOCKED = "blocked"


@dataclass(frozen=True)
class ProjectSection:
    id: int
    name: str
    owns: tuple[str, ...]
    may_mutate: tuple[str, ...]
    forbidden_without_approval: tuple[str, ...]
    hill_climb_gate: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "owns": list(self.owns),
            "may_mutate": list(self.may_mutate),
            "forbidden_without_approval": list(self.forbidden_without_approval),
            "hill_climb_gate": self.hill_climb_gate,
        }


@dataclass(frozen=True)
class SectionCheck:
    section_id: int
    name: str
    status: str
    evidence: str
    action: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "name": self.name,
            "status": self.status,
            "evidence": self.evidence,
            "action": self.action,
        }


SECTIONS: tuple[ProjectSection, ...] = (
    ProjectSection(
        id=1,
        name="architecture_foundation",
        owns=(
            "project sections and stage gates",
            "single source of truth",
            "default/challenger naming",
            "safe-to-run command boundaries",
            "model-work blockers",
        ),
        may_mutate=("docs", "read-only audit/check scripts", "tests"),
        forbidden_without_approval=(
            "runtime paper-order flags",
            "launchd operational defaults",
            "model artifacts",
            "broker-connected paths",
        ),
        hill_climb_gate="section_1_architecture_foundation_pass",
    ),
    ProjectSection(
        id=2,
        name="data_acquisition_preparation",
        owns=(
            "raw/normalized/feature/label/audit layer contracts",
            "paid data approval and download safety",
            "data provenance",
            "schema and causal timestamp invariants",
            "data inventory for approved research windows",
        ),
        may_mutate=("data contracts", "safe data-audit scripts", "tests"),
        forbidden_without_approval=(
            "paid market-data endpoint calls",
            "protected holdout scoring",
            "raw data rewrites",
            "unstamped feature/label datasets",
        ),
        hill_climb_gate="section_2_data_foundation_pass",
    ),
    ProjectSection(
        id=3,
        name="model_experiments_training_testing_validating",
        owns=(
            "hypothesis preregistration",
            "training runs",
            "threshold/model selection",
            "strict serial replay",
            "validation and promotion claims",
        ),
        may_mutate=("experiment outputs", "research-only model artifacts"),
        forbidden_without_approval=(
            "new training before Section 1/2 gates pass",
            "threshold tuning on exposed diagnostics",
            "protected holdout-driven model selection",
            "paper-default promotion",
        ),
        hill_climb_gate="section_3_training_readiness_pass",
    ),
    ProjectSection(
        id=4,
        name="live_paper_trade",
        owns=(
            "IBKR paper runtime",
            "no-order shadow/live parity",
            "paper order guards",
            "quote freshness",
            "fill/cancel observations",
            "forced-flat safety",
        ),
        may_mutate=("paper logs", "runtime diagnostics"),
        forbidden_without_approval=(
            "broker endpoint calls",
            "paper-submit sessions",
            "runtime enablement flag changes",
            "real-money trading",
        ),
        hill_climb_gate="section_4_execution_truth_pass_before_promotion",
    ),
    ProjectSection(
        id=5,
        name="promotion_governance",
        owns=(
            "research freezes",
            "paper-default decisions",
            "rollback/demotion criteria",
            "stale documentation resolution",
        ),
        may_mutate=("promotion packets", "governance docs"),
        forbidden_without_approval=(
            "changing the operational default",
            "promoting challengers",
            "demoting Protocol101 paper default",
        ),
        hill_climb_gate="section_5_governance_packet_required_for_default_change",
    ),
)


SECTION_BY_ID = {section.id: section for section in SECTIONS}


# The root docs/ drawer was retired on 2026-08-05 and its contents quarantined
# under _cleanup_quarantine/2026-08-05b-docs/. The single-source-of-truth
# requirement is unchanged; it is now satisfied by the v5 status page.
REQUIRED_SECTION_1_FILES = (
    "v5/STATUS.md",
    "v5/AGENTS.md",
    "v4/docs/PROJECT_SECTIONS_AND_HILL_CLIMB_GATES.md",
    "v4/docs/NAMING_GUIDE.md",
    "v4/docs/MODEL_IMPROVEMENT_GUIDELINES.md",
    "v4/docs/HYPOTHESIS_TO_PROMOTION_PROCESS.md",
    "v4/docs/FOUNDATION_HARDENING_AUDIT.md",
)

REQUIRED_SECTION_2_FILES = (
    "v4/docs/DATA_CONTRACT.md",
    "v4/checks/paid_data_guard.py",
    "v4/checks/integrity.py",
    "v4/checks/sanity.py",
    "v4/ingest/databento_opra.py",
    "v4/scripts/audit_data_sufficiency.py",
    "v4/scripts/audit_context_provenance.py",
)

DATA_CONTRACT_TOKENS = (
    "decision_time",
    "event_time",
    "receive_time",
    "quote_age_ms",
    "is_live_reproducible",
    "vendor_source",
    "ingest_run_id",
)

SECTION3_PREFLIGHT_SUMMARY = Path("v4/audit/autoresearch/section3_model_experiment_preflight/summary.json")
NEURAL_TRAINING_READINESS_SUMMARY = Path("v4/audit/autoresearch/unified_neural_training_readiness/summary.json")


def project_sections() -> tuple[ProjectSection, ...]:
    return SECTIONS


def build_readiness_payload(repo_root: Path = Path(".")) -> dict[str, Any]:
    checks = collect_checks(repo_root)
    by_section = {
        section.id: {
            "section": section.to_dict(),
            "checks": [check.to_dict() for check in checks if check.section_id == section.id],
        }
        for section in SECTIONS
    }
    for section_id, payload in by_section.items():
        payload["status"] = section_status(payload["checks"])

    section_1_2_ready = (
        by_section[1]["status"] == PASS
        and by_section[2]["status"] == PASS
    )
    model_hill_climb_blockers = [
        check.name
        for check in checks
        if check.section_id in {3, 4, 5} and check.status == BLOCKED
    ]
    return {
        "schema_version": "project_section_readiness_v1",
        "sections": by_section,
        "section_1_2_decision": "section_1_2_ready" if section_1_2_ready else "section_1_2_blocked",
        "model_hill_climb_decision": (
            "model_hill_climb_ready"
            if section_1_2_ready and not model_hill_climb_blockers
            else "model_hill_climb_blocked_until_truth_gates_pass"
        ),
        "model_hill_climb_blockers": model_hill_climb_blockers,
        "broker_endpoint_called": False,
        "paid_data_downloaded": False,
        "model_training": False,
    }


def collect_checks(repo_root: Path = Path(".")) -> list[SectionCheck]:
    root = repo_root.resolve()
    checks: list[SectionCheck] = []
    checks.extend(section_1_checks(root))
    checks.extend(section_2_checks(root))
    checks.extend(section_3_checks(root))
    checks.extend(section_4_checks(root))
    checks.extend(section_5_checks(root))
    return checks


def section_1_checks(root: Path) -> list[SectionCheck]:
    checks = [required_files_check(1, root, REQUIRED_SECTION_1_FILES, "section_1_required_files")]
    naming = read_text(root / "v4/docs/NAMING_GUIDE.md")
    checks.append(
        SectionCheck(
            1,
            "paper_default_named",
            PASS if "PAPER_DEFAULT_PROTOCOL101" in naming else BLOCKED,
            "PAPER_DEFAULT_PROTOCOL101 present in v4/docs/NAMING_GUIDE.md",
            "Restore explicit current-default naming before model work.",
        )
    )
    guidelines = read_text(root / "v4/docs/MODEL_IMPROVEMENT_GUIDELINES.md")
    checks.append(
        SectionCheck(
            1,
            "model_work_gate_documented",
            PASS if "Before any new challenger training" in guidelines else BLOCKED,
            "MODEL_IMPROVEMENT_GUIDELINES documents pre-training foundation gates",
            "Add a hard gate before training, threshold sweeps, or architecture search.",
        )
    )
    return checks


def section_2_checks(root: Path) -> list[SectionCheck]:
    checks = [required_files_check(2, root, REQUIRED_SECTION_2_FILES, "section_2_required_files")]
    data_contract = read_text(root / "v4/docs/DATA_CONTRACT.md")
    missing_tokens = [token for token in DATA_CONTRACT_TOKENS if token not in data_contract]
    checks.append(
        SectionCheck(
            2,
            "data_contract_core_fields",
            PASS if not missing_tokens else BLOCKED,
            "DATA_CONTRACT includes causal time, freshness, provenance, and live-reproducibility fields",
            "Add missing fields: " + ", ".join(missing_tokens) if missing_tokens else "None.",
        )
    )
    unguarded = unguarded_paid_download_scripts(root)
    checks.append(
        SectionCheck(
            2,
            "paid_download_scripts_guarded",
            PASS if not unguarded else BLOCKED,
            "All v4/scripts/download_*.py files that can call paid endpoints import require_paid_data_approval",
            "Add paid-data approval guard to: " + ", ".join(unguarded) if unguarded else "None.",
        )
    )
    parquet_count = len(list((root / "v4/normalized_official_context").glob("*.parquet")))
    processed_dirs = len([path for path in (root / "data/processed").glob("spxw_0dte_neural*") if path.is_dir()])
    checks.append(
        SectionCheck(
            2,
            "local_prepared_data_inventory_present",
            PASS if parquet_count > 0 and processed_dirs > 0 else WARN,
            f"normalized_official_context_parquet={parquet_count}; processed_dataset_dirs={processed_dirs}",
            "Run read-only data sufficiency inventory before any training window is selected.",
        )
    )
    return checks


def section_3_checks(root: Path) -> list[SectionCheck]:
    checks = [
        SectionCheck(
            3,
            "section3_preflight_runner_present",
            PASS
            if (root / "v4/foundation/model_experiment_preflight.py").exists()
            and (root / "v4/scripts/run_section3_model_experiment_preflight.py").exists()
            else BLOCKED,
            "Section 3 preflight code exists",
            "Restore v4/foundation/model_experiment_preflight.py and v4/scripts/run_section3_model_experiment_preflight.py.",
        )
    ]
    preflight = read_json(root / SECTION3_PREFLIGHT_SUMMARY)
    if preflight:
        decision = str(preflight.get("section3_model_experiment_decision", "missing"))
        blockers = preflight.get("training_blockers", [])
        checks.append(
            SectionCheck(
                3,
                "section3_model_experiment_preflight",
                PASS if decision == "section3_model_experiment_preflight_ready" else BLOCKED,
                f"decision={decision}; training_blockers={blockers}",
                "Close Section 3 preflight training blockers before training, threshold search, or architecture search.",
            )
        )
        return checks

    neural = read_json(root / NEURAL_TRAINING_READINESS_SUMMARY)
    if not neural:
        checks.append(
            SectionCheck(
                3,
                "section3_model_experiment_preflight",
                BLOCKED,
                f"missing {SECTION3_PREFLIGHT_SUMMARY} and {NEURAL_TRAINING_READINESS_SUMMARY}",
                "Run python3 -m v4.scripts.run_section3_model_experiment_preflight.",
            )
        )
        return checks

    decision = str(neural.get("training_decision", "missing"))
    blockers = neural.get("training_blockers", [])
    checks.append(
        SectionCheck(
            3,
            "section3_model_experiment_preflight",
            PASS if decision == "neural_training_ready_for_preregistered_conservative_policy_run" else BLOCKED,
            f"fallback_neural_training_decision={decision}; blockers={blockers}",
            "Run python3 -m v4.scripts.run_section3_model_experiment_preflight and close any training blockers.",
        )
    )
    return checks


def section_4_checks(root: Path) -> list[SectionCheck]:
    bridge = read_text(root / "v4/scripts/run_protocol158_protocol101_live_entry_paper_bridge.py")
    hardcoded_quote_age = '"quote_age_ms": 0' in bridge
    return [
        SectionCheck(
            4,
            "live_quote_age_truth",
            BLOCKED if hardcoded_quote_age else PASS,
            "Live bridge contains hard-coded quote_age_ms=0" if hardcoded_quote_age else "No hard-coded quote_age_ms=0 found in bridge",
            "Record raw quote timestamp, receipt timestamp, decision timestamp, and derived guard age.",
        )
    ]


def section_5_checks(root: Path) -> list[SectionCheck]:
    current_truth = read_text(root / "v5/STATUS.md")
    stale_conflict_noted = "conflict" in current_truth.lower() and (
        "supersed" in current_truth.lower() or "stale" in current_truth.lower()
    )
    return [
        SectionCheck(
            5,
            "stale_doc_conflicts_registered",
            WARN if stale_conflict_noted else BLOCKED,
            "Current source-of-truth registers stale/conflicting docs" if stale_conflict_noted else "No stale-doc conflict register found",
            "Resolve or mark stale operational docs before changing defaults.",
        )
    ]


def required_files_check(section_id: int, root: Path, files: tuple[str, ...], name: str) -> SectionCheck:
    missing = [path for path in files if not (root / path).exists()]
    return SectionCheck(
        section_id,
        name,
        PASS if not missing else BLOCKED,
        f"required={len(files)} missing={len(missing)}",
        "Create or restore: " + ", ".join(missing) if missing else "None.",
    )


def unguarded_paid_download_scripts(root: Path) -> list[str]:
    scripts_dir = root / "v4/scripts"
    unguarded: list[str] = []
    for path in sorted(scripts_dir.glob("download_*.py")):
        text = read_text(path)
        calls_paid_endpoint = "timeseries.get_range" in text or "metadata.get_cost" in text
        if calls_paid_endpoint and "require_paid_data_approval" not in text:
            unguarded.append(str(path.relative_to(root)))
    return unguarded


def section_status(checks: list[dict[str, Any]]) -> str:
    statuses = {str(check.get("status")) for check in checks}
    if BLOCKED in statuses:
        return BLOCKED
    if WARN in statuses:
        return WARN
    return PASS


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Project Section Readiness",
        "",
        f"Section 1/2 decision: `{payload['section_1_2_decision']}`",
        f"Model hill-climb decision: `{payload['model_hill_climb_decision']}`",
        "",
        "| Section | Status | Check | Evidence | Action |",
        "|---|---|---|---|---|",
    ]
    for section_id in sorted(payload["sections"], key=int):
        section_payload = payload["sections"][section_id]
        section_name = section_payload["section"]["name"]
        for check in section_payload["checks"]:
            lines.append(
                f"| {section_id}. {section_name} | `{check['status']}` | "
                f"{check['name']} | {check['evidence']} | {check['action']} |"
            )
    if payload["model_hill_climb_blockers"]:
        lines += ["", "## Model Hill-Climb Blockers", ""]
        for blocker in payload["model_hill_climb_blockers"]:
            lines.append(f"- `{blocker}`")
    return "\n".join(lines) + "\n"


def write_outputs(payload: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    report_path = out_dir / "report.md"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    report_path.write_text(render_markdown(payload))
    return summary_path, report_path


def read_text(path: Path) -> str:
    try:
        return path.read_text()
    except FileNotFoundError:
        return ""


def read_json(path: Path) -> dict[str, Any]:
    text = read_text(path)
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}
