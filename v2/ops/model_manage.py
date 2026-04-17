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
    MODEL_CANDIDATE.unlink()
    size_kb = MODEL_BEST.stat().st_size / 1024
    print(f"  model_best.pt + model.pt updated ({size_kb:.0f}K)")


def revert():
    """Discard candidate. model.pt and model_best.pt unchanged."""
    artifact_dir = _find_candidate_artifact_dir()
    if artifact_dir is not None:
        mark_reverted(str(artifact_dir))
        print(f"  marked artifact reverted: {artifact_dir}")
    else:
        print("  WARNING: could not find matching local artifact to revert")
    if MODEL_CANDIDATE.exists():
        MODEL_CANDIDATE.unlink()
        print(f"  model_candidate.pt discarded")
    else:
        print(f"  no candidate to discard")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("keep", "revert"):
        print("Usage: python v2/ops/model_manage.py [keep|revert]")
        sys.exit(1)

    {"keep": keep, "revert": revert}[sys.argv[1]]()


if __name__ == "__main__":
    main()
