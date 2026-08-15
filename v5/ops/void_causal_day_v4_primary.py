"""Issue an immutable correction for the economically void V4 selector."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from v5.ops.build_causal_day_dataset import canonical_json, file_sha256
from v5.research.causal_day_magnitude import TARGET_CLIP_POINTS
from v5.research.causal_day_selection import (
    SelectorSpecificationError,
    assert_absolute_threshold_attainable,
)


def _verified(path: Path) -> dict:
    value = json.loads(path.read_text())
    expected = value.get("receipt_sha256")
    unsigned = dict(value)
    unsigned.pop("receipt_sha256", None)
    if hashlib.sha256(canonical_json(unsigned)).hexdigest() != expected:
        raise RuntimeError(f"self-hash mismatch: {path}")
    return value


def run(*, declaration_path: Path, primary_receipt_path: Path, out_dir: Path) -> dict:
    if out_dir.exists():
        raise RuntimeError(f"refusing to overwrite correction: {out_dir}")
    declaration = _verified(declaration_path)
    primary = _verified(primary_receipt_path)
    threshold = float(
        declaration["primary_kill_cell"]["predicted_depth_threshold_points"]
    )
    ceiling = float(TARGET_CLIP_POINTS[1])
    try:
        assert_absolute_threshold_attainable(
            threshold, label_clip_ceiling_points=ceiling
        )
    except SelectorSpecificationError as error:
        defect = str(error)
    else:
        raise RuntimeError("V4 does not reproduce the declared selector defect")
    if primary.get("primary_kill_condition", {}).get("trades") != 0:
        raise RuntimeError("the voided V4 receipt unexpectedly contains selected trades")

    out_dir.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema_version": "v5.causal-day-primary-interpretation-correction.v1",
        "created_on": "2026-08-14",
        "status": "INCONCLUSIVE_SPECIFICATION_DEFECT",
        "economic_conclusion": "VOID_DO_NOT_RECORD_AS_NEGATIVE",
        "defect": defect,
        "derivable_before_outcomes": True,
        "label_clip_ceiling_points": ceiling,
        "voided_absolute_threshold_points": threshold,
        "fit_and_oof_predictions_remain_valid": True,
        "reopening_spent": False,
        "superseded_interpretation": {
            "declaration": {
                "path": str(declaration_path),
                "sha256": file_sha256(declaration_path),
            },
            "primary_receipt": {
                "path": str(primary_receipt_path),
                "sha256": file_sha256(primary_receipt_path),
            },
        },
        "required_correction": (
            "Freeze a causal rank selector from strictly prior fold-training sessions; "
            "then measure kill condition 1 without changing the fit, folds, label, "
            "horizon, architecture, width, trade cap or risk mode"
        ),
        "owner_controlled_files_modified": False,
        "implementation_sha256": file_sha256(Path(__file__)),
    }
    receipt["receipt_sha256"] = hashlib.sha256(canonical_json(receipt)).hexdigest()
    path = out_dir / "receipt.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(path)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--primary-receipt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    run(
        declaration_path=args.declaration,
        primary_receipt_path=args.primary_receipt,
        out_dir=args.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
