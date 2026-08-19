"""Run the known-answer preflight for the drawdown-ordered lifecycle experiment.

Authorized by the signed 2026-08-15 STOP override and scoped ledger reopening.
The runner refuses to execute unless the declaration's self-hash verifies, the
implementation hashes match the code about to run, and the calibration draws
match their declared digest. It refuses to overwrite an existing receipt.

Synthetic worlds only: no real targets, no real economics, no vendor, no
reserved sessions. The verdict gates the REAL fit: only `PREFLIGHT_PASSED`
satisfies kill condition 1 of the reopening.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from v5.ops.build_causal_day_dataset import canonical_json, file_sha256
from v5.research.capacity_campaign import stable_seed
from v5.research.drawdown_preflight import (
    CALIBRATION_PATH,
    CALIBRATION_SHA256,
    Calibration,
    LAW,
    run_trial,
    wilson_upper,
)

DECLARATION_SCHEMA = "v5.drawdown-preflight-declaration.v1"


def _verified(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    expected = value.get("declaration_sha256")
    unsigned = dict(value)
    unsigned.pop("declaration_sha256", None)
    if hashlib.sha256(canonical_json(unsigned)).hexdigest() != expected:
        raise RuntimeError(f"declaration self-hash mismatch: {path}")
    if value.get("schema_version") != DECLARATION_SCHEMA:
        raise RuntimeError(f"declaration schema mismatch: {path}")
    if value["law_sha256"] != LAW.sha256():
        raise RuntimeError("declared law hash does not match the code about to run")
    for relative, digest in value["implementation_hashes"].items():
        actual = file_sha256(Path(relative))
        if actual != digest:
            raise RuntimeError(f"implementation hash mismatch: {relative}")
    if value["calibration"]["sha256"] != CALIBRATION_SHA256:
        raise RuntimeError("declared calibration hash does not match the module pin")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", required=True, type=Path)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    args = parser.parse_args()

    declaration = _verified(args.declaration)
    receipt_path = args.evidence_dir / "receipt.json"
    if receipt_path.exists():
        raise RuntimeError(f"refusing to overwrite existing receipt: {receipt_path}")
    args.evidence_dir.mkdir(parents=True, exist_ok=True)

    calibration = Calibration.load(CALIBRATION_PATH)

    trials: list[dict[str, Any]] = []
    planted_passes = 0
    null_passes = 0
    for arm, slope, count in (
        ("planted_minimum", LAW.effect_slope_minimum, LAW.planted_trials),
        ("null", LAW.effect_slope_null, LAW.null_trials),
    ):
        for index in range(count):
            seed = stable_seed("drawdown-preflight-v1", arm, index)
            result = run_trial(
                effect_slope=slope,
                trial_seed=seed,
                calibration=calibration,
                law=LAW,
            )
            record = {
                "arm": arm,
                "index": index,
                "seed": seed,
                "effect_slope": slope,
                "gate": asdict(result.gate),
                "epochs_run": result.epochs_run,
                "stop_epoch_of_best": result.stop_epoch_of_best,
                "score_cutoff": result.score_cutoff,
            }
            trials.append(record)
            if result.gate.passed:
                if arm == "null":
                    null_passes += 1
                else:
                    planted_passes += 1
            done = len(trials)
            print(
                f"[{done:3d}/{LAW.planted_trials + LAW.null_trials}] {arm}"
                f" seed={seed} passed={result.gate.passed}"
                f" reason={result.gate.reason} trades={result.gate.trades}"
                f" precision={result.gate.selected_precision:.3f}"
                f" lcb={result.gate.corrected_lcb_usd:.1f}",
                flush=True,
            )

    recovery_rate = planted_passes / LAW.planted_trials
    null_upper = wilson_upper(null_passes, LAW.null_trials)
    if null_upper > LAW.null_false_pass_wilson_upper_limit:
        verdict = "PREFLIGHT_FAILED_NULL_FALSE_PASS"
    elif recovery_rate < LAW.recovery_rate_required:
        verdict = "PREFLIGHT_FAILED_UNDERPOWERED"
    else:
        verdict = "PREFLIGHT_PASSED"

    receipt = {
        "schema_version": "v5.drawdown-preflight-receipt.v1",
        "declaration": {
            "path": str(args.declaration),
            "sha256": declaration["declaration_sha256"],
        },
        "law_sha256": LAW.sha256(),
        "calibration": {
            "path": str(CALIBRATION_PATH),
            "sha256": CALIBRATION_SHA256,
        },
        "planted_trials": LAW.planted_trials,
        "planted_passes": planted_passes,
        "recovery_rate": recovery_rate,
        "recovery_required": LAW.recovery_rate_required,
        "null_trials": LAW.null_trials,
        "null_passes": null_passes,
        "null_false_pass_wilson_upper": null_upper,
        "verdict": verdict,
        "trials": trials,
    }
    payload = canonical_json(receipt)
    receipt["receipt_sha256"] = hashlib.sha256(payload).hexdigest()
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    print(f"verdict: {verdict}")
    print(f"recovery: {planted_passes}/{LAW.planted_trials} = {recovery_rate:.2f}")
    print(
        f"null false passes: {null_passes}/{LAW.null_trials}"
        f" (Wilson upper {null_upper:.4f})"
    )
    print(f"receipt: {receipt_path}")


if __name__ == "__main__":
    main()
