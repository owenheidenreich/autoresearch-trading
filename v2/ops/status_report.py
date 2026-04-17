"""Print the current health and operating status of the live v2 project."""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import torch

from v2.ops.artifact import iter_artifacts_by_score, load_artifact


PROJECT_ROOT = Path(__file__).resolve().parents[2]
V2_ROOT = PROJECT_ROOT / "v2"
DATA_PATH = V2_ROOT / "data.pt"
RESULTS_PATH = V2_ROOT / "results.tsv"
CURRENT_STATE_PATH = V2_ROOT / "docs" / "current_state.md"
FOUNDER_INTENT_PATH = V2_ROOT / "docs" / "founder_intent.md"
DECISION_LOG_PATH = V2_ROOT / "docs" / "decision_log.md"
OPEN_QUESTIONS_PATH = V2_ROOT / "docs" / "open_questions.md"
STATUS_REQUIRED_PATHS = (
    FOUNDER_INTENT_PATH,
    DECISION_LOG_PATH,
    OPEN_QUESTIONS_PATH,
    V2_ROOT / "ops" / "status_report.py",
)


def _read_text(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def _extract_md_bullet(text: str, label: str) -> str | None:
    match = re.search(rf"^- {re.escape(label)}: (.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def _extract_backtick_value(text: str, label: str) -> str | None:
    match = re.search(rf"^- {re.escape(label)}: `([^`]+)`$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def load_dataset_summary() -> dict:
    data = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
    meta = data.get("metadata", {})
    return {
        "path": str(DATA_PATH.relative_to(PROJECT_ROOT)),
        "version": meta.get("version", "unknown"),
        "fingerprint": meta.get("fingerprint", "unknown"),
        "unique_days": len(sorted(set(data.get("dates", [])))),
        "features": len(data.get("feature_names", [])),
    }


def load_state_summary() -> dict:
    text = _read_text(CURRENT_STATE_PATH)
    mission = _extract_md_bullet(text, "Mission")
    phase = _extract_md_bullet(text, "Current phase")
    next_experiment = _extract_backtick_value(text, "Next experiment")
    return {
        "mission": mission or "unknown",
        "phase": phase or "unknown",
        "next_experiment": next_experiment or "unknown",
        "not_live_trading_statement": "not live trading" in text.lower()
        and "trustworthy exact-chain research system" in _read_text(FOUNDER_INTENT_PATH).lower(),
    }


def load_results_summary() -> dict:
    """Parse the CVReport TSV schema by column name.

    Accepts rows with screening_mode in {"full", "legacy"} as official-ish; full
    is the canonical post-repair ledger entry, legacy is the historical entries
    migrated from the pre-repair TSV. Non-"full" modes are non-official.
    Everything is parsed via DictReader so schema drift fails loudly rather than
    silently degrading numeric columns to zero.
    """
    from v2.core.cv_report import RESULTS_TSV_HEADER

    if not RESULTS_PATH.exists():
        return {
            "official_rows": [],
            "non_official_rows": [],
            "latest_official": None,
            "best_official": None,
            "schema_error": None,
        }

    with RESULTS_PATH.open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, [])
        if header != RESULTS_TSV_HEADER:
            return {
                "official_rows": [],
                "non_official_rows": [],
                "latest_official": None,
                "best_official": None,
                "schema_error": (
                    f"results.tsv header does not match CVReport schema. "
                    f"expected={RESULTS_TSV_HEADER}, got={header}. "
                    f"Run the harness-integrity repair migration."
                ),
            }
        raw_rows = [row for row in reader if row]

    official_rows: list[dict] = []
    non_official_rows: list[dict] = []
    for row in raw_rows:
        if len(row) < len(header):
            continue
        record = dict(zip(header, row))
        try:
            stability = float(record.get("stability_score", "-999"))
        except (TypeError, ValueError):
            stability = -999.0
        try:
            pooled_pf = float(record.get("pooled_pf", "0"))
        except (TypeError, ValueError):
            pooled_pf = 0.0
        entry = {
            "experiment": record.get("experiment", ""),
            "screening_mode": record.get("screening_mode", ""),
            "stability_score": stability,
            "pooled_pf": pooled_pf,
            "status": record.get("status", ""),
            "any_gate_failure": record.get("any_gate_failure", "") == "true",
            "description": record.get("description", ""),
            # back-compat alias so callers that still print `score` keep working
            "score": stability,
        }
        mode = entry["screening_mode"]
        if mode == "full":
            official_rows.append(entry)
        else:
            non_official_rows.append(entry)

    latest_official = official_rows[-1] if official_rows else None
    best_official = (
        max(official_rows, key=lambda item: item["stability_score"])
        if official_rows else None
    )
    return {
        "official_rows": official_rows,
        "non_official_rows": non_official_rows,
        "latest_official": latest_official,
        "best_official": best_official,
        "schema_error": None,
    }


def load_artifact_summary(dataset_fingerprint: str) -> dict:
    promoted_dirs = iter_artifacts_by_score(promoted_only=True)
    compatible: list[dict] = []
    incompatible: list[dict] = []

    for artifact_dir in promoted_dirs:
        try:
            loaded = load_artifact(artifact_dir, current_dataset_fingerprint=dataset_fingerprint)
            compatible.append(
                {
                    "artifact_dir": artifact_dir,
                    "experiment_id": loaded["manifest"].get("experiment_id", Path(artifact_dir).name),
                    "score": float(loaded["manifest"].get("score", float("nan"))),
                }
            )
        except Exception as exc:  # noqa: BLE001 - want operator-facing reason
            incompatible.append(
                {
                    "artifact_dir": artifact_dir,
                    "reason": str(exc).splitlines()[0],
                }
            )

    best_compatible = max(compatible, key=lambda item: item["score"]) if compatible else None
    return {
        "promoted_total": len(promoted_dirs),
        "compatible_total": len(compatible),
        "best_compatible": best_compatible,
        "incompatible": incompatible,
    }


def load_open_questions() -> list[dict]:
    text = _read_text(OPEN_QUESTIONS_PATH)
    questions: list[dict] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("- `OQ-") or " | " not in line:
            continue
        body = line[3:-1] if line.endswith("`") else line[2:]
        parts = [part.strip() for part in body.split(" | ")]
        if len(parts) < 5:
            continue
        questions.append(
            {
                "id": parts[0],
                "priority": parts[1],
                "area": parts[2],
                "question": parts[3],
                "next_step": parts[4].removeprefix("next: ").strip(),
            }
        )
    return questions


def run_gate(skip_gate: bool) -> dict:
    if skip_gate:
        return {"status": "skipped", "summary": "not run"}

    proc = subprocess.run(
        ["python3", "-m", "v2.ops.pre_run_gate", "--data", "v2/data.pt"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    output = proc.stdout.strip()
    if proc.returncode == 0:
        return {"status": "pass", "summary": output.splitlines()[-1] if output else "passed"}
    first_failure = next((line for line in output.splitlines() if line.startswith("- ")), output.splitlines()[-1] if output else "failed")
    return {"status": "fail", "summary": first_failure}


def build_blockers(required_paths_ok: bool, results_summary: dict, artifact_summary: dict, gate_summary: dict, open_questions: list[dict]) -> list[str]:
    blockers: list[str] = []
    if not required_paths_ok:
        blockers.append("Required operating-system docs or commands are missing.")
    if results_summary.get("schema_error"):
        blockers.append(f"results.tsv schema error: {results_summary['schema_error']}")
    if results_summary["non_official_rows"]:
        blockers.append("results.tsv contains non-official rows (screening_mode != 'full').")
    if artifact_summary["compatible_total"] == 0:
        blockers.append("No compatible promoted artifact bundle exists for the current dataset and restored architecture.")
    if gate_summary["status"] == "fail":
        blockers.append(f"Pre-run gate is failing: {gate_summary['summary']}")
    for item in open_questions:
        if item["priority"] == "high":
            blockers.append(f"{item['id']}: {item['question']}")
    return blockers


def determine_health(blockers: list[str], gate_summary: dict) -> str:
    if gate_summary["status"] == "fail":
        return "unhealthy"
    if blockers:
        return "attention"
    return "healthy"


def print_human_report(report: dict) -> None:
    print("ART2 v2 Status")
    print(f"Health: {report['health'].upper()}")
    print(f"Phase: {report['state']['phase']}")
    print(f"Mission: {report['state']['mission']}")
    print(f"Next experiment: {report['state']['next_experiment']}")
    print()
    print("Current mission boundary:")
    print("  Not live trading; trustworthy exact-chain research system.")
    print()
    print("Dataset:")
    print(f"  Path: {report['dataset']['path']}")
    print(f"  Version: {report['dataset']['version']}")
    print(f"  Fingerprint: {report['dataset']['fingerprint']}")
    print(f"  Unique days: {report['dataset']['unique_days']}")
    print(f"  Features: {report['dataset']['features']}")
    print()
    print("Repo health:")
    print(f"  Pre-run gate: {report['gate']['status']} ({report['gate']['summary']})")
    print(f"  Required operating-system files present: {'yes' if report['required_paths_ok'] else 'no'}")
    print(f"  Official CV rows (full mode): {len(report['results']['official_rows'])}")
    if report["results"]["latest_official"]:
        latest = report["results"]["latest_official"]
        print(
            f"  Latest CV run: {latest['experiment']} "
            f"(stability={latest['stability_score']:.3f}, "
            f"pooled_pf={latest['pooled_pf']:.2f}, status={latest['status']})"
        )
    if report["results"]["best_official"]:
        best = report["results"]["best_official"]
        print(
            f"  Best CV stability: {best['experiment']} "
            f"(stability={best['stability_score']:.3f}, pooled_pf={best['pooled_pf']:.2f})"
        )
    print(f"  Non-official rows present: {len(report['results']['non_official_rows'])}")
    if report["results"].get("schema_error"):
        print(f"  SCHEMA ERROR: {report['results']['schema_error']}")
    print()
    print("Artifacts:")
    print(f"  Promoted artifacts: {report['artifacts']['promoted_total']}")
    print(f"  Compatible promoted artifacts: {report['artifacts']['compatible_total']}")
    if report["artifacts"]["best_compatible"]:
        best = report["artifacts"]["best_compatible"]
        print(f"  Best compatible promoted artifact: {best['experiment_id']} ({best['score']:.3f})")
    else:
        print("  Best compatible promoted artifact: none")
    print()
    print("Open questions:")
    for item in report["open_questions"][:5]:
        print(f"  {item['id']} [{item['priority']}] {item['area']} — {item['question']}")
    if not report["open_questions"]:
        print("  none")
    print()
    print("Current blockers:")
    if report["blockers"]:
        for blocker in report["blockers"][:8]:
            print(f"  - {blocker}")
    else:
        print("  - none")


def main() -> int:
    parser = argparse.ArgumentParser(description="Report live v2 project health and current truth")
    parser.add_argument("--skip-gate", action="store_true", help="Skip the expensive pre-run gate check")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when health is not healthy")
    args = parser.parse_args()

    dataset = load_dataset_summary()
    state = load_state_summary()
    results_summary = load_results_summary()
    artifact_summary = load_artifact_summary(dataset["fingerprint"])
    open_questions = load_open_questions()
    gate_summary = run_gate(skip_gate=args.skip_gate)
    required_paths_ok = all(path.exists() for path in STATUS_REQUIRED_PATHS)
    blockers = build_blockers(required_paths_ok, results_summary, artifact_summary, gate_summary, open_questions)
    health = determine_health(blockers, gate_summary)

    report = {
        "health": health,
        "dataset": dataset,
        "state": state,
        "gate": gate_summary,
        "results": {
            "official_rows": results_summary["official_rows"],
            "latest_official": results_summary["latest_official"],
            "best_official": results_summary["best_official"],
            "non_official_row_count": len(results_summary["non_official_rows"]),
        },
        "artifacts": artifact_summary,
        "required_paths_ok": required_paths_ok,
        "open_questions": open_questions,
        "blockers": blockers,
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_human_report(
            {
                **report,
                "results": results_summary,
            }
        )

    if args.strict and health != "healthy":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
