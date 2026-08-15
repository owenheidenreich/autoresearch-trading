"""Freeze Job 39's selector correction without changing its valid fit."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from v5.ops.build_causal_day_dataset import canonical_json, file_sha256
from v5.research.causal_day_architectures import computed_parameter_counts
from v5.research.causal_day_magnitude import TARGET_CLIP_POINTS
from v5.research.causal_day_policy_gate import (
    REOPENED_CORPUS,
    REOPENED_LABEL,
    REQUIRED_KILL_CONDITIONS,
    assert_fit_permitted,
    load_reopening,
)
from v5.research.causal_day_selection import (
    SelectorSpecificationError,
    assert_absolute_threshold_attainable,
)


DEFAULT_V4 = Path("v5/work/entry-exit-attribution/DECLARATION_V4.json")
DEFAULT_FIT_RECEIPT = Path(
    "v4/audit/autoresearch/causal_day_magnitude_fit_2026_08_14_attempt001/receipt.json"
)
PRIMARY_ARCHITECTURE = "neural_four_head"
PRIMARY_HORIZON = 120
TARGET_SIGNAL_MINUTES_PER_SESSION = 2


def _verified(path: Path, schema: str) -> dict:
    value = json.loads(path.read_text())
    expected = value.get("receipt_sha256")
    unsigned = dict(value)
    unsigned.pop("receipt_sha256", None)
    if hashlib.sha256(canonical_json(unsigned)).hexdigest() != expected:
        raise RuntimeError(f"self-hash mismatch: {path}")
    if value.get("schema_version") != schema:
        raise RuntimeError(f"unexpected schema: {path}")
    return value


def build_declaration(
    v4_path: Path = DEFAULT_V4,
    fit_receipt_path: Path = DEFAULT_FIT_RECEIPT,
) -> dict:
    v4 = _verified(v4_path, "v5.causal-day-trader-fit-declaration.v4")
    fit = _verified(fit_receipt_path, "v5.causal-day-magnitude-fit.v1")
    if fit.get("declaration", {}).get("sha256") != file_sha256(v4_path):
        raise RuntimeError("the valid fit is not bound to the supplied V4 declaration")
    if fit.get("economics_read") is not False or fit.get("threshold_tuned") is not False:
        raise RuntimeError("the fit receipt does not preserve the pre-economics firewall")

    counts = computed_parameter_counts()
    reopening = load_reopening()
    assert_fit_permitted(
        PRIMARY_ARCHITECTURE,
        sessions=int(v4["source_population"]["sessions"]),
        trainable_parameters=counts[PRIMARY_ARCHITECTURE],
        reopening=reopening,
        label=REOPENED_LABEL,
        horizon=PRIMARY_HORIZON,
        corpus=REOPENED_CORPUS,
        declared_kill_conditions=REQUIRED_KILL_CONDITIONS,
    )

    old_threshold = float(
        v4["primary_kill_cell"]["predicted_depth_threshold_points"]
    )
    try:
        assert_absolute_threshold_attainable(
            old_threshold,
            label_clip_ceiling_points=float(TARGET_CLIP_POINTS[1]),
        )
    except SelectorSpecificationError as error:
        defect_proof = str(error)
    else:
        raise RuntimeError("V4 selector defect no longer reproduces")

    value = copy.deepcopy(v4)
    value["schema_version"] = "v5.causal-day-trader-fit-declaration.v5"
    value["supersedes_declaration_sha256"] = file_sha256(v4_path)
    value["purpose"] = (
        "Correct the pre-outcome V4 selector defect while reusing the unchanged valid "
        "width-3 fit and preserving all seven signed kill conditions"
    )
    value["v4_interpretation"] = {
        "status": "INCONCLUSIVE_SPECIFICATION_DEFECT",
        "negative_result_void": True,
        "defect_proof": defect_proof,
        "zero_trade_receipt_is_audit_history_only": True,
    }
    value["fit_reuse"] = {
        "model_refit": False,
        "reason": "The model and out-of-fold ranking remain valid; only activation was defective",
        "v4_declaration": {"path": str(v4_path), "sha256": file_sha256(v4_path)},
        "fit_receipt": {
            "path": str(fit_receipt_path),
            "sha256": file_sha256(fit_receipt_path),
        },
        "architecture": PRIMARY_ARCHITECTURE,
        "horizon_minutes": PRIMARY_HORIZON,
        "computed_trainable_parameters": counts[PRIMARY_ARCHITECTURE],
        "policy_fit_gate": "PERMITTED through load_reopening() and assert_fit_permitted() with all seven required kill conditions",
    }
    primary = value["primary_kill_cell"]
    primary.pop("predicted_depth_threshold_points", None)
    primary["selector"] = {
        "mode": "causal_training_prefix_rank",
        "score_unit": "maximum predicted 120-minute ITM depth among eligible contracts in each minute",
        "target_signal_minutes_per_training_session": TARGET_SIGNAL_MINUTES_PER_SESSION,
        "fold_cutoff": "k-th largest training-minute maximum where k = 2 * number of strictly earlier training sessions",
        "application": "freeze one cutoff per fold, then walk each scored session minute by minute under the existing occupancy and trade-cap state",
        "current_scored_session_statistics_used": False,
        "whole_day_top_n_on_scored_session": False,
        "outcomes_or_pnl_used_for_cutoff": False,
        "tie_break": value["action_family"]["tie_break"],
    }
    primary["pass_rule"] = "at least one selected trade and mean gross mid-to-mid dollars per trade strictly greater than zero"
    primary["failure_consequence"] = (
        "If measured gross mid-to-mid is non-positive, record a negative result and stop "
        "before matched control, shuffled economics, bid economics or exit fitting"
    )
    value["selector_attainability_gate"] = {
        "implementation": "v5/research/causal_day_selection.py",
        "absolute_rule": "refuse every absolute prediction threshold greater than or equal to the label clip ceiling",
        "label_clip_ceiling_points": float(TARGET_CLIP_POINTS[1]),
        "voided_v4_threshold_points": old_threshold,
        "v4_defect_reproduced_before_economics": True,
        "rank_rule": "require a finite interior training-prefix rank cutoff strictly below the observed training maximum and strictly before every score session",
    }
    value["ranking_diagnostics_before_economics"] = [
        "Pearson correlation of real OOF predictions with actual 120-minute ITM depth",
        "actual 120-minute depth mean and count in ten equal-count prediction ranks",
        "real and shuffled counts with predicted depth at or above 25 points",
        "real and shuffled prediction standard deviations",
    ]
    value["controls_and_reporting"]["v5_primary_family"] = (
        "One corrected primary selector fixed before economics; its matched and shuffled "
        "controls remain conditional on kill condition 1"
    )
    value["controls_and_reporting"]["v5_composition_match_law"] = {
        "cells": "same scored session x entry regime x option side x absolute-delta quintile x entry-ask-premium quintile",
        "quintile_source": "causal values from the initial 93-session training prefix only, frozen before economics",
        "sampling": "deterministic hash order without replacement, excluding model trades",
        "occupancy": "exact 120-minute clock intervals must not overlap; same-minute re-entry remains forbidden",
        "required_completeness": "every model trade must receive an exact matched trade or kill condition 2 fails",
        "outcome_blind": True,
    }
    value["controls_and_reporting"]["v5_corrected_bootstrap"] = {
        "method": "moving whole-session block bootstrap",
        "one_sided_level": 0.95,
        "declared_family_size": 648,
        "family_choice": "retain V4's complete pre-exit family size rather than narrowing multiplicity after fit",
        "fold_rule": "absolute real and paired deltas versus both controls must each be positive in at least 4 of 5 chronological folds",
    }
    value["forbidden"] = list(
        dict.fromkeys(
            [
                *value["forbidden"],
                "absolute prediction threshold at or above the label clip ceiling",
                "whole-session top-N or percentile computed from the scored session",
                "cutoff calibration from scored-session predictions",
                "cutoff calibration from labels, P&L, spreads or trade outcomes",
            ]
        )
    )
    implementations = (
        Path("v5/research/causal_day_selection.py"),
        Path("v5/ops/write_causal_day_fit_declaration_v5.py"),
        Path("v5/ops/calibrate_causal_day_rank_selector.py"),
        Path("v5/ops/evaluate_causal_day_magnitude_v5.py"),
    )
    value["v5_implementation_hashes"] = {
        str(path): file_sha256(path) if path.is_file() else "PENDING_BEFORE_EXECUTION"
        for path in implementations
    }
    value.pop("receipt_sha256", None)
    value["receipt_sha256"] = hashlib.sha256(canonical_json(value)).hexdigest()
    return value


def run(*, out_path: Path, v4_path: Path, fit_receipt_path: Path) -> dict:
    if out_path.exists():
        raise RuntimeError(f"refusing to overwrite declaration: {out_path}")
    value = build_declaration(v4_path, fit_receipt_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(out_path)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--v4", type=Path, default=DEFAULT_V4)
    parser.add_argument("--fit-receipt", type=Path, default=DEFAULT_FIT_RECEIPT)
    args = parser.parse_args()
    run(out_path=args.out, v4_path=args.v4, fit_receipt_path=args.fit_receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
