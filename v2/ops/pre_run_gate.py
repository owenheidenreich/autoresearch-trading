"""Local integrity gate that must pass before GPU spend."""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = PROJECT_ROOT / "v2" / "results.tsv"
TRAIN_PATH = PROJECT_ROOT / "v2" / "train.py"
HOW_TRAINING_WORKS_PATH = PROJECT_ROOT / "v2" / "docs" / "how_training_works.md"
REQUIRED_LIVE_PATHS = (
    PROJECT_ROOT / "v2" / "docs" / "founder_intent.md",
    PROJECT_ROOT / "v2" / "docs" / "decision_log.md",
    PROJECT_ROOT / "v2" / "docs" / "open_questions.md",
    PROJECT_ROOT / "v2" / "ops" / "status_report.py",
)


def _parts(*items: str) -> str:
    return "".join(items)


GLOBAL_LEGACY_PATTERNS = (
    re.compile(rf"\b{re.escape(_parts('run_', 'experiment.py'))}\b"),
    re.compile(rf"\b{re.escape(_parts('v2.ops.', 'run_experiment'))}\b(?!_wf)"),
)
CODE_LEGACY_PATTERNS = (
    re.compile(rf"\b{re.escape(_parts('STRIKE', '_OFFSETS'))}\b"),
    re.compile(rf"outputs\[['\"]{re.escape(_parts('direction'))}['\"]\]"),
    re.compile(rf"outputs\[['\"]{re.escape(_parts('risk'))}['\"]\]"),
    re.compile(rf"outputs\[['\"]{re.escape(_parts('confidence'))}['\"]\]"),
    re.compile(rf"\b{re.escape(_parts('WEIGHT', '_PNL'))}\b"),
    re.compile(rf"\b{re.escape(_parts('WEIGHT', '_SIDE'))}\b"),
    re.compile(rf"\b{re.escape(_parts('NO', '_TRADE_W'))}\b"),
    re.compile(rf"\b{re.escape(_parts('SCORE', '_REG_W'))}\b"),
    re.compile(rf"\b{re.escape(_parts('side', '_head'))}\b"),
    re.compile(rf"\b{re.escape(_parts('side', '_logit'))}\b"),
    re.compile(rf"\b{re.escape(_parts('targets_', 'contract_', 'field'))}\b"),
)
DOC_REQUIRED_PHRASES = (
    "Balanced gate BCE on supervised rows",
    "Soft KL selection loss",
    "No direct PnL regression",
    "No auxiliary side head",
)
DOC_FORBIDDEN_PHRASES = (
    "Stage 1:",
    "Stage 2:",
    "Stage 3:",
    " ".join(("side", "supervision", "first")),
    " ".join(("soft", "within-side", "ranking")),
    " ".join(("gate", "calibration")),
    " ".join(("exp_079", "recovery")),
)
REMOVED_LIVE_PATHS = (
    PROJECT_ROOT / "v2" / ".baseline_cache.json",
    PROJECT_ROOT / "v2" / ".best_score",
    PROJECT_ROOT / "v2" / ".inner_loop_state.json",
    PROJECT_ROOT / "v2" / "FRESH_SESSION_HANDOFF.md",
    PROJECT_ROOT / "v2" / "live",
    PROJECT_ROOT / "v2" / "data",
    PROJECT_ROOT / "v2" / "model.pt",
    PROJECT_ROOT / "v2" / "model_best.pt",
    PROJECT_ROOT / "v2" / "model_candidate.pt",
    PROJECT_ROOT / "v2" / "data.pt.bak",
    PROJECT_ROOT / "v2" / "data_harness_repair.pt",
    PROJECT_ROOT / "v2" / "data_pre_harness_repair.pt",
    PROJECT_ROOT / "v2" / "analysis" / "analyze_whipsaw.py",
    PROJECT_ROOT / "v2" / "analysis" / "contract_drift_audit.py",
    PROJECT_ROOT / "v2" / "analysis" / "diagnostic_direction_signal.py",
    PROJECT_ROOT / "v2" / "analysis" / "diagnostic_oracle_replay.py",
    PROJECT_ROOT / "v2" / "pipeline" / "build_dataset.py",
    PROJECT_ROOT / "v2" / "pipeline" / "download_wide_grid.py",
    PROJECT_ROOT / "v2" / "pipeline" / "extract_raw.py",
    PROJECT_ROOT / "v2" / "pipeline" / "relabel_tier3.py",
    PROJECT_ROOT / "v2" / "docs" / "goal.md",
    PROJECT_ROOT / "v2" / "docs" / "baselines.md",
    PROJECT_ROOT / "v2" / "ops" / "fast_sweep.py",
    PROJECT_ROOT / "v2" / "ops" / "run_loop.sh",
    PROJECT_ROOT / "v2" / "docs" / "PLAN-codex-docs-train-restart.md",
)
TEXT_FILE_SUFFIXES = {".md", ".py", ".sh"}
TEXT_FILE_EXCLUDES = {
    ".DS_Store",
}
SKIP_DIRS = {
    "archive",
    "__pycache__",
    "artifacts",
    "data_sidecars",
    "harness_eval",
    "output",
    "sweep_results",
}


def run(cmd: list[str], cwd: Path | None = None) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd or PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout


def iter_live_text_files() -> list[Path]:
    files = [PROJECT_ROOT / "AGENTS.md", PROJECT_ROOT / "CLAUDE.md"]
    for path in (PROJECT_ROOT / "v2").rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name in TEXT_FILE_EXCLUDES:
            continue
        if path.suffix not in TEXT_FILE_SUFFIXES:
            continue
        files.append(path)
    return sorted(set(files))


def check_removed_paths(errors: list[str]) -> None:
    for path in REMOVED_LIVE_PATHS:
        if path.exists():
            errors.append(f"stale live file still present: {path.relative_to(PROJECT_ROOT)}")


def check_required_paths(errors: list[str]) -> None:
    for path in REQUIRED_LIVE_PATHS:
        if not path.exists():
            errors.append(f"required live operating-system path is missing: {path.relative_to(PROJECT_ROOT)}")


def check_results_tsv(errors: list[str]) -> None:
    if not RESULTS_PATH.exists():
        errors.append("results.tsv is missing")
        return

    with RESULTS_PATH.open(newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    if not rows:
        errors.append("results.tsv is empty")
        return
    if rows[0] != ["experiment", "score", "status", "description"]:
        errors.append("results.tsv header is not the canonical four-column header")
        return

    for row in rows[1:]:
        if len(row) < 4:
            errors.append(f"results.tsv malformed row: {row}")
            continue
        exp_id, _, status, description = row[:4]
        if status == "unknown" or description.startswith("screening:"):
            errors.append(f"results.tsv contains non-official row: {exp_id}")
        if exp_id.startswith("exp_"):
            exp_num = exp_id[4:]
            if exp_num.isdigit() and int(exp_num) < 74:
                errors.append(f"results.tsv contains pre-exact-chain row: {exp_id}")


def check_live_text_patterns(errors: list[str]) -> None:
    for path in iter_live_text_files():
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text()
        for pattern in GLOBAL_LEGACY_PATTERNS:
            if pattern.search(text):
                errors.append(f"legacy reference '{pattern.pattern}' found in {path.relative_to(PROJECT_ROOT)}")
        if path.suffix in {".py", ".sh"}:
            for pattern in CODE_LEGACY_PATTERNS:
                if pattern.search(text):
                    errors.append(f"legacy code token '{pattern.pattern}' found in {path.relative_to(PROJECT_ROOT)}")


def check_doc_sync(errors: list[str]) -> None:
    if not HOW_TRAINING_WORKS_PATH.exists():
        errors.append("docs/how_training_works.md is missing")
        return

    text = HOW_TRAINING_WORKS_PATH.read_text()
    for phrase in DOC_REQUIRED_PHRASES:
        if phrase not in text:
            errors.append(f"docs/how_training_works.md is missing required phrase: {phrase}")
    for phrase in DOC_FORBIDDEN_PHRASES:
        if phrase in text:
            errors.append(f"docs/how_training_works.md still contains stale phrasing: {phrase}")

    train_text = TRAIN_PATH.read_text()
    expected_train_tokens = (
        "sel_loss",
        "SOFT_TEMP",
        "NOISE_MARGIN",
    )
    for token in expected_train_tokens:
        if token not in train_text:
            errors.append(f"train.py is missing expected baseline token: {token.splitlines()[0]}")


def check_train_smoke(data_path: str, errors: list[str]) -> None:
    code, output = run(["python3", "-m", "py_compile", "v2/train.py", "v2/core/policy.py"])
    if code != 0:
        errors.append(f"py_compile failed:\n{output}")
        return

    data = torch.load(data_path, map_location="cpu", weights_only=False)
    max_contracts = int(data["metadata"]["max_contracts_per_bar"])

    from v2.core.chain_data import NUM_CONTRACT_FEATURES
    from v2.train import LOOKBACK, NUM_FEATURES, TradingModel

    model = TradingModel()
    model.eval()
    window = torch.zeros(1, LOOKBACK, NUM_FEATURES, dtype=torch.float32)
    contracts = torch.zeros(1, max_contracts, NUM_CONTRACT_FEATURES, dtype=torch.float32)
    outputs = model(window, contracts)
    required_keys = {"contract_scores", "valid_mask"}
    if not required_keys.issubset(set(outputs.keys())):
        errors.append(f"train.py smoke output missing required keys: got {sorted(outputs.keys())}, need {sorted(required_keys)}")


def check_harness_eval(data_path: str, errors: list[str]) -> None:
    code, output = run(["python3", "-m", "v2.analysis.harness_eval", "--data", data_path])
    if code != 0:
        errors.append(f"harness_eval failed:\n{output}")


def check_data_integrity(data_path: str, errors: list[str]) -> None:
    """Run data integrity validation on manifest and sampled sidecars."""
    from v2.core.data_integrity import run_full_validation, print_validation_summary

    reports = run_full_validation(data_path=data_path, sidecar_sample=10)
    print_validation_summary(reports)

    for r in reports:
        if not r.passed:
            errors.append(f"data integrity FAIL: {r.source} — {'; '.join(r.errors)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local exact-chain integrity gate")
    parser.add_argument("--data", default="v2/data.pt", help="Dataset path used by the live exact-chain harness")
    args = parser.parse_args()

    data_path = str((PROJECT_ROOT / args.data).resolve() if not os.path.isabs(args.data) else Path(args.data))
    errors: list[str] = []

    check_required_paths(errors)
    check_removed_paths(errors)
    check_results_tsv(errors)
    check_live_text_patterns(errors)
    check_doc_sync(errors)
    check_train_smoke(data_path, errors)
    check_data_integrity(data_path, errors)
    check_harness_eval(data_path, errors)

    if errors:
        print("Pre-run gate failed:", file=sys.stderr)
        for err in errors:
            print(f"- {err}", file=sys.stderr)
        return 1

    print("Pre-run gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
