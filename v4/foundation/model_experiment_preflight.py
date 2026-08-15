"""Section 3 preflight for model experiments.

This module does not train models, tune thresholds, score holdouts, download
data, or touch broker endpoints. It only turns the current model-readiness
artifacts into an explicit go/no-go packet for future hill-climb work.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from v4.model.neural_training_readiness import (
    CHALLENGE_READY_DECISION,
    TRAINING_READY_DECISION,
)


PASS = "pass"
WARN = "warn"
BLOCKED = "blocked"

ROLE_LABEL = "SECTION3_MODEL_EXPERIMENT_PREFLIGHT_V1"
READY_DECISION = "section3_model_experiment_preflight_ready"
BLOCKED_DECISION = "section3_model_experiment_preflight_blocked"

DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/section3_model_experiment_preflight")
DEFAULT_NEURAL_READINESS_SUMMARY = Path("v4/audit/autoresearch/unified_neural_training_readiness/summary.json")

# The root docs/ drawer was retired on 2026-08-05; the source of truth is v5.
REQUIRED_SECTION_1_2_ARTIFACTS = (
    "v5/STATUS.md",
    "v5/AGENTS.md",
    "v4/docs/PROJECT_SECTIONS_AND_HILL_CLIMB_GATES.md",
    "v4/docs/DATA_CONTRACT.md",
    "v4/checks/paid_data_guard.py",
)

REQUIRED_HYPOTHESIS_TOKENS = (
    "Hypothesis:",
    "Candidate role label:",
    "Baseline:",
    "Data allowed:",
    "Training splits:",
    "Validation splits:",
    "Protected test splits:",
    "Primary metric:",
    "Required stress checks:",
    "Expected failure mode if wrong:",
    "Does this change paper default: no",
)


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    status: str
    evidence: str
    required_action: str
    blocks_training: bool = True
    blocks_protocol101_challenge: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_model_experiment_preflight(
    repo_root: Path = Path("."),
    *,
    hypothesis_packet: Path | None = None,
    require_hypothesis: bool = False,
) -> dict[str, Any]:
    root = repo_root.resolve()
    checks = [
        section_1_2_artifact_check(root),
        neural_training_readiness_check(root),
        protocol101_challenge_readiness_check(root),
        hypothesis_packet_check(root, hypothesis_packet=hypothesis_packet, require_hypothesis=require_hypothesis),
        preflight_safety_check(),
    ]
    training_blockers = [check.name for check in checks if check.blocks_training and check.status == BLOCKED]
    challenge_blockers = [check.name for check in checks if check.blocks_protocol101_challenge and check.status == BLOCKED]
    decision = READY_DECISION if not training_blockers else BLOCKED_DECISION
    return {
        "role_label": ROLE_LABEL,
        "what_is_this": "Section 3 model-experiment preflight; no training or scoring is performed.",
        "changes_paper_default": False,
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "section3_model_experiment_decision": decision,
        "training_blockers": training_blockers,
        "protocol101_challenge_blockers": challenge_blockers,
        "checks": [check.to_dict() for check in checks],
        "source_artifacts": {
            "neural_training_readiness": str(DEFAULT_NEURAL_READINESS_SUMMARY),
            "hypothesis_packet": None if hypothesis_packet is None else str(hypothesis_packet),
        },
        "next_allowed_work": next_allowed_work(training_blockers, challenge_blockers),
    }


def section_1_2_artifact_check(root: Path) -> PreflightCheck:
    missing = [path for path in REQUIRED_SECTION_1_2_ARTIFACTS if not (root / path).exists()]
    return PreflightCheck(
        name="section_1_2_artifacts_present",
        status=PASS if not missing else BLOCKED,
        evidence=f"missing={missing}" if missing else f"required={len(REQUIRED_SECTION_1_2_ARTIFACTS)} missing=0",
        required_action="Restore Section 1/2 source-of-truth, gates, data contract, and paid-data guard before model work.",
    )


def neural_training_readiness_check(root: Path) -> PreflightCheck:
    summary = read_json(root / DEFAULT_NEURAL_READINESS_SUMMARY)
    if not summary:
        return PreflightCheck(
            name="neural_training_readiness_summary_present",
            status=BLOCKED,
            evidence=f"missing {DEFAULT_NEURAL_READINESS_SUMMARY}",
            required_action="Run the safe readiness audit: python3 -m v4.scripts.run_unified_neural_training_readiness --skip-ledger.",
        )
    decision = str(summary.get("training_decision", "missing"))
    blockers = summary.get("training_blockers", [])
    return PreflightCheck(
        name="neural_training_readiness",
        status=PASS if decision == TRAINING_READY_DECISION else BLOCKED,
        evidence=f"training_decision={decision}; blockers={blockers}",
        required_action="Do not run new model experiments until neural readiness training blockers are closed.",
    )


def protocol101_challenge_readiness_check(root: Path) -> PreflightCheck:
    summary = read_json(root / DEFAULT_NEURAL_READINESS_SUMMARY)
    if not summary:
        return PreflightCheck(
            name="protocol101_challenge_readiness",
            status=BLOCKED,
            evidence=f"missing {DEFAULT_NEURAL_READINESS_SUMMARY}",
            required_action="Run the safe readiness audit before making Protocol101 challenge claims.",
            blocks_training=False,
        )
    decision = str(summary.get("protocol101_challenge_decision", "missing"))
    blockers = summary.get("challenge_blockers", [])
    return PreflightCheck(
        name="protocol101_challenge_readiness",
        status=PASS if decision == CHALLENGE_READY_DECISION else BLOCKED,
        evidence=f"protocol101_challenge_decision={decision}; blockers={blockers}",
        required_action="Treat model outputs as research-only until fill, holdout, live parity, and validation controls pass.",
        blocks_training=False,
        blocks_protocol101_challenge=True,
    )


def hypothesis_packet_check(
    root: Path,
    *,
    hypothesis_packet: Path | None,
    require_hypothesis: bool,
) -> PreflightCheck:
    if hypothesis_packet is None:
        return PreflightCheck(
            name="hypothesis_packet_preregistered",
            status=BLOCKED if require_hypothesis else WARN,
            evidence="no hypothesis packet supplied",
            required_action="Before an actual model run, provide a preregistered hypothesis packet with data, baseline, metrics, and protected-test rules.",
            blocks_training=require_hypothesis,
            blocks_protocol101_challenge=True,
        )
    path = hypothesis_packet if hypothesis_packet.is_absolute() else root / hypothesis_packet
    text = read_text(path)
    missing = [token for token in REQUIRED_HYPOTHESIS_TOKENS if token not in text]
    return PreflightCheck(
        name="hypothesis_packet_preregistered",
        status=PASS if not missing else BLOCKED,
        evidence=f"path={path}; missing={missing}" if missing else f"path={path}; required={len(REQUIRED_HYPOTHESIS_TOKENS)} missing=0",
        required_action="Complete the preregistration template before any training, threshold search, or architecture search.",
        blocks_training=True,
        blocks_protocol101_challenge=True,
    )


def preflight_safety_check() -> PreflightCheck:
    return PreflightCheck(
        name="preflight_runner_safety",
        status=PASS,
        evidence="read-only summary generation; model_training=false; paid_data_downloaded=false; broker_endpoint_called=false",
        required_action="None.",
        blocks_training=False,
        blocks_protocol101_challenge=False,
    )


def next_allowed_work(training_blockers: list[str], challenge_blockers: list[str]) -> list[str]:
    if training_blockers:
        return [
            "Close the listed training blockers before any model hill climb.",
            "Continue attribution, validation-provenance, execution-realism, and parity work only.",
            "Keep Protocol101 as the paper default and challengers research-only.",
        ]
    if challenge_blockers:
        return [
            "A preregistered research run may be considered, but it cannot challenge Protocol101 or change defaults.",
            "Do not score protected holdouts or promote a challenger until challenge blockers pass.",
        ]
    return [
        "A preregistered model experiment may be considered under the frozen Section 3 rules.",
        "Protocol101 challenge claims still require an explicit promotion/governance packet.",
    ]


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Section 3 Model Experiment Preflight",
        "",
        f"Decision: `{payload['section3_model_experiment_decision']}`",
        "",
        "This report is generated without training, threshold tuning, protected-holdout scoring, paid-data downloads, broker endpoint calls, or paper/live orders.",
        "",
        "## Checks",
        "",
        "| Check | Status | Evidence | Required action | Blocks training? | Blocks Protocol101 challenge? |",
        "|---|---|---|---|---:|---:|",
    ]
    for check in payload["checks"]:
        lines.append(
            "| {name} | `{status}` | {evidence} | {required_action} | {blocks_training} | {blocks_protocol101_challenge} |".format(
                **{key: markdown_cell(value) for key, value in check.items()}
            )
        )
    lines.extend(["", "## Training Blockers", ""])
    if payload["training_blockers"]:
        lines.extend(f"- `{item}`" for item in payload["training_blockers"])
    else:
        lines.append("- none")
    lines.extend(["", "## Protocol101 Challenge Blockers", ""])
    if payload["protocol101_challenge_blockers"]:
        lines.extend(f"- `{item}`" for item in payload["protocol101_challenge_blockers"])
    else:
        lines.append("- none")
    lines.extend(["", "## Next Allowed Work", ""])
    lines.extend(f"- {item}" for item in payload["next_allowed_work"])
    lines.append("")
    return "\n".join(lines)


def write_outputs(payload: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = out_dir / "summary.json"
    report = out_dir / "report.md"
    summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    report.write_text(render_report(payload))
    return summary, report


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text()


def markdown_cell(value: Any) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")
