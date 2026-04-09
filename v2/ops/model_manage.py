"""Model management for ART² experiment loop.

Maintains v2/model_best.pt as the canonical best model.

deploy.sh run_one always downloads model.pt after training, overwriting
whatever was there -- even if the experiment was worse. This script
ensures model.pt always reflects the best kept model.

Usage:
    python v2/ops/model_manage.py keep     # model.pt -> model_best.pt, then both match
    python v2/ops/model_manage.py revert   # model_best.pt -> model.pt, undoes bad overwrite

Call IMMEDIATELY after reading the score, BEFORE any analysis.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

MODEL_PT = Path("v2/model.pt")
MODEL_BEST = Path("v2/model_best.pt")


def keep():
    """After KEEP: current model.pt (just downloaded) becomes the new best."""
    if not MODEL_PT.exists():
        print(f"ERROR: {MODEL_PT} not found")
        sys.exit(1)
    shutil.copy2(MODEL_PT, MODEL_BEST)
    size_kb = MODEL_BEST.stat().st_size / 1024
    print(f"  model_best.pt updated from model.pt ({size_kb:.0f}K)")


def revert():
    """After REVERT: restore model.pt from best (undo run_one's overwrite)."""
    if not MODEL_BEST.exists():
        print(f"WARNING: {MODEL_BEST} not found, model.pt unchanged")
        return
    shutil.copy2(MODEL_BEST, MODEL_PT)
    print(f"  model.pt restored from model_best.pt (reverted bad overwrite)")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("keep", "revert"):
        print("Usage: python v2/ops/model_manage.py [keep|revert]")
        sys.exit(1)

    action = sys.argv[1]
    if action == "keep":
        keep()
    elif action == "revert":
        revert()


if __name__ == "__main__":
    main()
