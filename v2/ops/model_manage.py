"""Model management for ART² experiment loop.

deploy.sh downloads to v2/models/model_candidate.pt (never overwrites the
promoted checkpoint directly). This script promotes or discards the candidate
based on keep/revert decision.

Usage:
    python v2/ops/model_manage.py keep     # candidate -> model.pt + model_best.pt
    python v2/ops/model_manage.py revert   # discard candidate, model.pt unchanged
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from v2.ops.artifact import ARTIFACTS_DIR, _file_fingerprint, mark_promoted, mark_reverted

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


def keep():
    """Promote candidate to best."""
    if not MODEL_CANDIDATE.exists():
        print(f"ERROR: {MODEL_CANDIDATE} not found (did run_one finish?)")
        sys.exit(1)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact_dir = _find_candidate_artifact_dir()
    shutil.copy2(MODEL_CANDIDATE, MODEL_BEST)
    shutil.copy2(MODEL_CANDIDATE, MODEL_PT)
    if artifact_dir is not None:
        mark_promoted(str(artifact_dir))
        print(f"  marked artifact promoted: {artifact_dir}")
    else:
        print("  WARNING: could not find matching local artifact to promote")
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
