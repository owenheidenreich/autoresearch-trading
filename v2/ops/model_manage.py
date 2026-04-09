"""Model management for ART² experiment loop.

deploy.sh downloads to model_candidate.pt (never overwrites model.pt directly).
This script promotes or discards the candidate based on keep/revert decision.

Usage:
    python v2/ops/model_manage.py keep     # candidate -> model.pt + model_best.pt
    python v2/ops/model_manage.py revert   # discard candidate, model.pt unchanged
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

MODEL_PT = Path("v2/model.pt")
MODEL_BEST = Path("v2/model_best.pt")
MODEL_CANDIDATE = Path("v2/model_candidate.pt")


def keep():
    """Promote candidate to best."""
    if not MODEL_CANDIDATE.exists():
        print(f"ERROR: {MODEL_CANDIDATE} not found (did run_one finish?)")
        sys.exit(1)
    shutil.copy2(MODEL_CANDIDATE, MODEL_BEST)
    shutil.copy2(MODEL_CANDIDATE, MODEL_PT)
    MODEL_CANDIDATE.unlink()
    size_kb = MODEL_BEST.stat().st_size / 1024
    print(f"  model_best.pt + model.pt updated ({size_kb:.0f}K)")


def revert():
    """Discard candidate. model.pt and model_best.pt unchanged."""
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
