"""Model promotion gate. Enforces the CV → final-train → deploy boundary.

Only artifacts tagged ArtifactKind.FINAL_TRAIN may be promoted to
v2/models/model.pt. CV_EVAL and FOLD_CHECKPOINT artifacts are hard-rejected.

Flow:
    v2.ops.run_final_train         → produces candidate FINAL_TRAIN artifact,
                                     copies its model to model_candidate.pt
    v2.ops.model_manage.keep       → promotes candidate; writes sibling
                                     v2/models/model.manifest.json
    v2.ops.model_manage.revert     → discards candidate; model.pt unchanged

Usage:
    python v2/ops/model_manage.py keep     # candidate -> model.pt + model_best.pt
    python v2/ops/model_manage.py revert   # discard candidate, model.pt unchanged
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from v2.core.artifact_kind import ArtifactKind, is_promotable
from v2.core.cv_report import RESULTS_TSV_HEADER
from v2.ops.artifact import (
    ARTIFACTS_DIR,
    _file_fingerprint,
    mark_promoted,
    mark_reverted,
    write_model_pt_manifest,
)

MODEL_DIR = Path("v2/models")
MODEL_PT = MODEL_DIR / "model.pt"
MODEL_BEST = MODEL_DIR / "model_best.pt"
MODEL_CANDIDATE = MODEL_DIR / "model_candidate.pt"
RESULTS_TSV = Path("v2/results.tsv")


def _set_results_tsv_status(experiment_id: str, new_status: str) -> bool:
    """Flip the `status` column in results.tsv for the row matching experiment_id.

    `experiment_id` here is the CV experiment (the row that was written by
    `cmd_run_cv` via `_append_results_tsv`). The FINAL_TRAIN artifact stores
    this value under `extra.source_experiment`; model_manage passes that
    through so keep/revert flips the evidence ledger, not the final-train row
    (which doesn't exist in results.tsv).

    Returns True if a row was updated.
    """
    if not RESULTS_TSV.exists():
        print(f"  WARNING: {RESULTS_TSV} missing — cannot reflect status update")
        return False
    raw = RESULTS_TSV.read_text().splitlines()
    if not raw:
        print(f"  WARNING: {RESULTS_TSV} is empty")
        return False
    header = raw[0].split("\t")
    if header != RESULTS_TSV_HEADER:
        print(
            f"  WARNING: {RESULTS_TSV} header does not match CVReport schema "
            f"({RESULTS_TSV_HEADER}); refusing to patch status blindly."
        )
        return False
    exp_col = header.index("experiment")
    status_col = header.index("status")
    updated = False
    out_lines = [raw[0]]
    for line in raw[1:]:
        if not line.strip():
            out_lines.append(line)
            continue
        cols = line.split("\t")
        if len(cols) < len(header):
            out_lines.append(line)
            continue
        if cols[exp_col] == experiment_id and cols[status_col] != new_status:
            cols[status_col] = new_status
            updated = True
        out_lines.append("\t".join(cols))
    if updated:
        RESULTS_TSV.write_text("\n".join(out_lines) + "\n")
        print(f"  results.tsv: {experiment_id} status -> {new_status}")
    else:
        print(f"  results.tsv: no row matched experiment_id={experiment_id}")
    return updated


def _find_candidate_artifact_dir() -> Path | None:
    """Locate the local artifact bundle that matches model_candidate.pt."""
    if not MODEL_CANDIDATE.exists():
        return None
    artifacts_dir = Path(ARTIFACTS_DIR)
    if not artifacts_dir.exists():
        return None

    candidate_fp = _file_fingerprint(str(MODEL_CANDIDATE))
    matches: list[tuple[str, Path]] = []
    for manifest_path in artifacts_dir.glob("*/manifest.json"):
        with open(manifest_path) as f:
            manifest = json.load(f)
        if manifest.get("model_fingerprint") == candidate_fp:
            matches.append((manifest.get("timestamp", ""), manifest_path.parent))
    if not matches:
        return None
    matches.sort(key=lambda item: item[0], reverse=True)
    return matches[0][1]


def _load_manifest(artifact_dir: Path) -> dict:
    with open(artifact_dir / "manifest.json") as f:
        return json.load(f)


def _artifact_is_research_tier(artifact_dir: Path, manifest: dict) -> bool:
    """Return True if the artifact is tagged research_tier. Inspects manifest
    (extra.research_tier, provenance.research_tier) and any sibling
    provenance.json. Absent flags are treated as False — enforcement is
    additive, not retroactive."""
    extra = manifest.get("extra") or {}
    if bool(extra.get("research_tier")):
        return True
    manifest_prov = manifest.get("provenance") or {}
    if bool(manifest_prov.get("research_tier")):
        return True
    sidecar = artifact_dir / "provenance.json"
    if sidecar.exists():
        try:
            with open(sidecar) as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False
        block = payload.get("provenance", payload)
        if bool(block.get("research_tier")):
            return True
    return False


def keep():
    """Promote candidate to best. Only FINAL_TRAIN artifacts accepted."""
    if not MODEL_CANDIDATE.exists():
        print(f"ERROR: {MODEL_CANDIDATE} not found (did run_final_train finish?)")
        sys.exit(1)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact_dir = _find_candidate_artifact_dir()

    if artifact_dir is None:
        print("ERROR: Cannot promote — no local artifact matches model_candidate.pt")
        print("       Promotion requires a FINAL_TRAIN artifact. Did you run")
        print("       v2.ops.run_final_train? CV_EVAL artifacts are not promotable.")
        sys.exit(1)

    manifest = _load_manifest(artifact_dir)
    kind = manifest.get("artifact_kind")
    if not is_promotable(kind):
        print("ERROR: Cannot promote — artifact is not a FINAL_TRAIN artifact.")
        print(f"  artifact_dir:   {artifact_dir}")
        print(f"  artifact_kind:  {kind!r}")
        print(f"  required:       {ArtifactKind.FINAL_TRAIN.value!r}")
        print("")
        print("  Walk-forward CV produces CV_EVAL artifacts (NOT promotable).")
        print("  Run v2.ops.run_final_train --config-from <exp_id> to produce a")
        print("  FINAL_TRAIN artifact from the chosen CV config.")
        sys.exit(1)

    if _artifact_is_research_tier(artifact_dir, manifest):
        print("ERROR: Cannot promote — artifact is flagged research_tier=True.")
        print(f"  artifact_dir:   {artifact_dir}")
        print("")
        print("  Research-tier labels are diagnostic bridge targets (learnable != ")
        print("  tradable). To promote, re-run training with the OPP_LABEL renamed")
        print("  out of the research_* namespace — that code change makes the")
        print("  promotion visible in git, not implicit in 'the screen looked fine'.")
        sys.exit(1)

    # --- Artifact gate: validate required observability artifacts ---
    from v2.core.observability import validate_artifact_presence
    errors = validate_artifact_presence(artifact_dir)
    if errors:
        print("ERROR: Cannot promote — missing or malformed artifacts:")
        for e in errors:
            print(f"  - {e}")
        print("Fix artifacts or use 'revert' to discard candidate.")
        sys.exit(1)

    shutil.copy2(MODEL_CANDIDATE, MODEL_BEST)
    shutil.copy2(MODEL_CANDIDATE, MODEL_PT)

    # Sibling manifest at v2/models/model.manifest.json so loaders can banner.
    # Pull training_config_fingerprint from extra metadata that run_final_train
    # wrote (the artifact's top-level kept that distinct from evaluator_fingerprint).
    extra = manifest.get("extra", {}) or {}
    write_model_pt_manifest(str(MODEL_PT), {
        "artifact_kind": ArtifactKind.FINAL_TRAIN.value,
        "source_experiment": extra.get("source_experiment", manifest.get("experiment_id", "")),
        "source_artifact_dir": str(artifact_dir),
        "training_config_fingerprint": extra.get("training_config_fingerprint", ""),
        "applied_env_overrides": extra.get("applied_env_overrides", {}),
        "dataset_fingerprint": manifest.get("dataset_fingerprint", ""),
        "evaluator_fingerprint": manifest.get("evaluator_fingerprint", ""),
        "policy_fingerprint": manifest.get("policy_fingerprint", ""),
        "trained_on_span": extra.get("trained_on_span", ""),
        "internal_val_slice": extra.get("internal_val_slice", ""),
        "model_fingerprint": manifest.get("model_fingerprint", ""),
        "git_sha": manifest.get("git_sha", ""),
    })

    mark_promoted(str(artifact_dir))
    print(f"  marked artifact promoted: {artifact_dir}")

    # Reflect promotion into the evidence ledger. The results.tsv row is keyed
    # by the *CV* experiment, not the FINAL_TRAIN experiment — the CV is what
    # `_append_results_tsv` wrote. The source_experiment pointer is the one we
    # want to flip.
    source_exp = (manifest.get("extra") or {}).get("source_experiment") \
        or manifest.get("experiment_id")
    if source_exp:
        _set_results_tsv_status(source_exp, "keep")

    MODEL_CANDIDATE.unlink()
    size_kb = MODEL_BEST.stat().st_size / 1024
    print(f"  model_best.pt + model.pt updated ({size_kb:.0f}K)")


def revert():
    """Discard candidate. model.pt and model_best.pt unchanged."""
    artifact_dir = _find_candidate_artifact_dir()
    source_exp = None
    if artifact_dir is not None:
        try:
            with open(artifact_dir / "manifest.json") as f:
                m = json.load(f)
            source_exp = (m.get("extra") or {}).get("source_experiment") \
                or m.get("experiment_id")
        except Exception:
            pass
        mark_reverted(str(artifact_dir))
        print(f"  marked artifact reverted: {artifact_dir}")
    else:
        print("  WARNING: could not find matching local artifact to revert")
    if MODEL_CANDIDATE.exists():
        MODEL_CANDIDATE.unlink()
        print(f"  model_candidate.pt discarded")
    else:
        print(f"  no candidate to discard")

    # Reflect revert in results.tsv (idempotent: no-op if already "revert").
    if source_exp:
        _set_results_tsv_status(source_exp, "revert")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("keep", "revert"):
        print("Usage: python v2/ops/model_manage.py [keep|revert]")
        sys.exit(1)

    {"keep": keep, "revert": revert}[sys.argv[1]]()


if __name__ == "__main__":
    main()
