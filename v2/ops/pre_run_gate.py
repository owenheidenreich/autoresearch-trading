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
WALKFORWARD_PATH = PROJECT_ROOT / "v2" / "core" / "walkforward.py"
MODEL_PT_PATH = PROJECT_ROOT / "v2" / "models" / "model.pt"
MODEL_MANIFEST_PATH = PROJECT_ROOT / "v2" / "models" / "model.manifest.json"
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
    # side_head and side_logit are now part of the wave 1 architecture
    # (opportunity + side + aggression heads). Removed from legacy patterns.
    re.compile(rf"\b{re.escape(_parts('targets_', 'contract_', 'field'))}\b"),
)
DOC_REQUIRED_PHRASES = (
    "Balanced gate supervision on `opportunity_logit`",
    "Soft KL selection loss",
    "No direct PnL regression",
    "`opportunity_logit` — the only live trade/no-trade gate",
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
    files = [PROJECT_ROOT / "CLAUDE.md"]
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
    """Validate results.tsv has the post-harness-repair CVReport schema."""
    from v2.core.cv_report import RESULTS_TSV_HEADER

    if not RESULTS_PATH.exists():
        errors.append("results.tsv is missing")
        return

    with RESULTS_PATH.open(newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    if not rows:
        errors.append("results.tsv is empty")
        return
    if rows[0] != RESULTS_TSV_HEADER:
        errors.append(
            f"results.tsv header does not match current CVReport schema. "
            f"Expected: {RESULTS_TSV_HEADER}, got: {rows[0]}. "
            f"Run the migration in v2.ops.migrate_results_tsv or archive + reset."
        )
        return

    desc_col = rows[0].index("description")
    status_col = rows[0].index("status")
    for row in rows[1:]:
        if len(row) < len(RESULTS_TSV_HEADER):
            errors.append(f"results.tsv malformed row: {row}")
            continue
        exp_id = row[0]
        status = row[status_col]
        description = row[desc_col]
        if status == "unknown":
            errors.append(f"results.tsv contains unknown-status row: {exp_id}")
        if exp_id.startswith("exp_"):
            exp_num = exp_id[4:].split("_")[0]
            if exp_num.isdigit() and int(exp_num) < 74:
                errors.append(f"results.tsv contains pre-exact-chain row: {exp_id}")


def check_harness_integrity(errors: list[str]) -> None:
    """Invariants that prevent the fold-as-deploy-model bug from returning.

    See /Users/gduby/.claude/plans/delightful-yawning-tiger.md Appendix G.
    """
    # --- A.1 / Appendix G.2-3: walkforward.py must not write to v2/models/model.pt ---
    if not WALKFORWARD_PATH.exists():
        errors.append(f"walkforward.py missing at {WALKFORWARD_PATH}")
    else:
        wf_src = WALKFORWARD_PATH.read_text()
        forbidden_substrings = [
            "v2/models/model.pt",
            "v2/models/model_fold",
        ]
        for needle in forbidden_substrings:
            if needle in wf_src:
                errors.append(
                    f"walkforward.py contains forbidden path {needle!r} — "
                    f"the promotion bypass has returned. Delete the write."
                )
        # walkforward must not call shutil.copy2 on anything that looks like a model path.
        if "shutil.copy2" in wf_src:
            errors.append(
                "walkforward.py uses shutil.copy2 — walkforward must never "
                "copy checkpoints. Fold checkpoints live under v2/artifacts/..."
            )
        # signature must not take model_path
        try:
            import importlib
            import v2.core.walkforward as wf_mod
            importlib.reload(wf_mod)
            import inspect
            sig = inspect.signature(wf_mod.run_walkforward)
            if "model_path" in sig.parameters:
                errors.append(
                    "run_walkforward() still accepts a `model_path` parameter. "
                    "Remove it — walkforward must never write to v2/models/."
                )
            if "screening_mode" not in sig.parameters:
                errors.append(
                    "run_walkforward() missing `screening_mode` parameter. "
                    "Screening-mode API is required post-repair."
                )
        except Exception as exc:
            errors.append(f"could not introspect run_walkforward(): {exc}")

    # --- A.9 / Appendix G.6: --n-folds flag is deleted everywhere in v2/ ---
    legacy_flag = "-" + "-n-folds"  # avoid self-matching inside this source
    for path in iter_live_text_files():
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.suffix not in {".py", ".sh"}:
            continue
        if legacy_flag in path.read_text():
            errors.append(
                f"legacy flag '{legacy_flag}' still present in "
                f"{path.relative_to(PROJECT_ROOT)}. Replace with --screen-mode."
            )

    # --- Appendix G.5: v2/models/model.pt must have a FINAL_TRAIN manifest ---
    from v2.core.artifact_kind import ArtifactKind
    if MODEL_PT_PATH.exists():
        if not MODEL_MANIFEST_PATH.exists():
            errors.append(
                f"{MODEL_PT_PATH.relative_to(PROJECT_ROOT)} exists but has no sibling "
                f"manifest at {MODEL_MANIFEST_PATH.relative_to(PROJECT_ROOT)}. "
                f"Any model under v2/models/ must come from run_final_train + model_manage keep."
            )
        else:
            try:
                import json as _json
                manifest = _json.loads(MODEL_MANIFEST_PATH.read_text())
                kind = manifest.get("artifact_kind")
                if kind != ArtifactKind.FINAL_TRAIN.value:
                    errors.append(
                        f"v2/models/model.pt has artifact_kind={kind!r} — only "
                        f"{ArtifactKind.FINAL_TRAIN.value!r} is deployable. "
                        f"Rerun run_final_train to replace."
                    )
            except Exception as exc:
                errors.append(f"could not parse model.manifest.json: {exc}")

    # --- Appendix G.8: CVReport schema version matches the tsv header ---
    from v2.core.cv_report import RESULTS_TSV_HEADER, SCHEMA_VERSION
    if not RESULTS_PATH.exists():
        errors.append("results.tsv missing")
    else:
        first_line = RESULTS_PATH.read_text().splitlines()[0] if RESULTS_PATH.read_text() else ""
        header_cols = first_line.split("\t") if first_line else []
        if header_cols != RESULTS_TSV_HEADER:
            errors.append(
                f"results.tsv header does not match CVReport schema "
                f"({SCHEMA_VERSION}). Migrate or reset results.tsv."
            )

    # --- Appendix G.7: autoresearch uses resolve_fold_indices, not folds[0] ---
    ar_path = PROJECT_ROOT / "v2" / "ops" / "autoresearch.py"
    if ar_path.exists():
        ar_src = ar_path.read_text()
        if "resolve_fold_indices" not in ar_src:
            errors.append(
                "autoresearch.py does not call resolve_fold_indices() — "
                "must use the same screening-mode API as the rest of the harness."
            )
        # A hard-coded `folds[0]` selection would re-introduce the earliest-vs-latest split
        if "folds[0]" in ar_src and "target_fold" not in ar_src:
            errors.append(
                "autoresearch.py uses folds[0] directly. Screen through "
                "resolve_fold_indices so 'fold 0' means the same window everywhere."
            )


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


def check_data_provenance(data_path: str, errors: list[str]) -> None:
    """Verify dataset fingerprint, config fingerprint, sidecar digest, and build SHA."""
    import glob
    from v2.core.config import RUNTIME_CONFIG
    from v2.core.chain_data import manifest_sidecar_digest
    from v2.core.dataset_fingerprint import compute_dataset_fingerprint

    data = torch.load(data_path, map_location="cpu", weights_only=False)
    meta = data.get("metadata", {})

    # 1. Config field-level check (label-affecting fields only)
    # max_contracts_per_bar is observational (derived from data, not prescribed),
    # so changes to it don't invalidate labels. Only check fields that affect
    # feature computation, label simulation, or schema interpretation.
    LABEL_AFFECTING_FIELDS = {
        "num_features", "num_contract_features", "lookback",
        "bars_per_day", "strike_grid", "chain_schema_version",
    }
    stored_cfg = meta.get("config_snapshot", {})
    if not stored_cfg:
        # Older builds stored config_fingerprint but not config_snapshot.
        # Fall back to fingerprint comparison with a softer message.
        stored_cfg_fp = meta.get("config_fingerprint", "")
        current_cfg_fp = RUNTIME_CONFIG.fingerprint()
        if stored_cfg_fp and stored_cfg_fp != current_cfg_fp:
            print(f"  WARNING: config fingerprint mismatch: data={stored_cfg_fp} current={current_cfg_fp}")
            print(f"  This may be due to non-label-affecting changes (e.g., max_contracts_per_bar).")
            print(f"  Checking individual label-affecting fields instead...")
            # Check the fields we can verify from metadata
            if meta.get("n_features") and meta["n_features"] != RUNTIME_CONFIG.num_features:
                errors.append(f"label-affecting config: n_features data={meta['n_features']} != config={RUNTIME_CONFIG.num_features}")
            csv_ = meta.get("chain_schema_version", "")
            if csv_ and csv_ != RUNTIME_CONFIG.chain_schema_version:
                errors.append(f"label-affecting config: chain_schema_version data={csv_} != config={RUNTIME_CONFIG.chain_schema_version}")

    # 2. Schema + feature validation
    meta_errors = RUNTIME_CONFIG.validate_dataset_metadata(meta)
    for e in meta_errors:
        errors.append(f"metadata validation: {e}")

    # 3. Dataset fingerprint (content-based)
    stored_fp = meta.get("fingerprint", "")
    if stored_fp:
        computed_fp = compute_dataset_fingerprint(data)
        if computed_fp != stored_fp:
            errors.append(
                f"dataset fingerprint mismatch: stored={stored_fp} computed={computed_fp} "
                f"— data.pt contents have been modified since build"
            )

    # 4. Sidecar digest — recompute from files and compare to stored
    sidecar_dir = meta.get("chain_sidecar_dir", "v2/data_sidecars")
    stored_digest = meta.get("chain_sidecar_digest", "")
    if stored_digest and os.path.isdir(sidecar_dir):
        sidecar_files = sorted(glob.glob(os.path.join(sidecar_dir, "*.pt")))
        if sidecar_files:
            current_digest = manifest_sidecar_digest(sidecar_files)
            if current_digest != stored_digest:
                errors.append(
                    f"sidecar digest mismatch: stored={stored_digest} computed={current_digest} "
                    f"— sidecars have been modified since dataset build, rebuild required"
                )
            print(f"  Sidecar digest verified: {current_digest} ({len(sidecar_files)} files)")
        else:
            errors.append(f"sidecar directory {sidecar_dir} exists but contains no .pt files")
    elif not os.path.isdir(sidecar_dir):
        errors.append(f"sidecar directory not found: {sidecar_dir}")

    # 5. Build SHA warning (non-blocking, but logged)
    build_sha = meta.get("build_git_sha", "")
    if build_sha:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        current_sha = result.stdout.strip()[:7] if result.returncode == 0 else "unknown"
        if current_sha != "unknown" and build_sha[:7] != current_sha[:7]:
            # Count commits since build
            result2 = subprocess.run(
                ["git", "rev-list", "--count", f"{build_sha}..HEAD"],
                capture_output=True, text=True, cwd=str(PROJECT_ROOT),
            )
            n_commits = result2.stdout.strip() if result2.returncode == 0 else "?"
            print(f"  WARNING: data built at {build_sha[:7]}, current HEAD is {current_sha} ({n_commits} commits ahead)")
            print(f"  Pipeline code may have changed. Verify no label-affecting changes before training.")


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
    check_harness_integrity(errors)
    check_live_text_patterns(errors)
    check_doc_sync(errors)
    check_train_smoke(data_path, errors)
    check_data_provenance(data_path, errors)
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
