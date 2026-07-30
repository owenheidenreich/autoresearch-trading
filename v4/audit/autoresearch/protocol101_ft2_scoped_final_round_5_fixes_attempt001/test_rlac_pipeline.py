#!/usr/bin/env python3
"""Synthetic executable proof that RLAC target construction is one-way."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np


HERE = Path(__file__).resolve().parent


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def ridge_fit(features: np.ndarray, targets: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(features)), features])
    penalty = np.eye(design.shape[1]) * 1e-6
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ targets)


def predict(features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(features)), features]) @ coefficients


def build_rlac_targets(
    *,
    quality: np.ndarray,
    upside: np.ndarray,
    fee_cleared: np.ndarray,
    intent_eligible: np.ndarray,
    quality_threshold: float,
) -> dict[str, Any]:
    """Build targets without accepting or importing a runtime composer."""
    wait_rows: list[dict[str, Any]] = []
    regret_rows: list[dict[str, Any]] = []
    for minute in range(quality.shape[0]):
        eligible = np.flatnonzero(intent_eligible[minute])
        if len(eligible) == 0:
            continue
        passing = [
            int(contract)
            for contract in eligible
            if quality[minute, contract] >= quality_threshold
            and fee_cleared[minute, contract]
        ]
        wait = len(passing) == 0
        c_star: int | None = None
        if passing:
            c_star = sorted(
                passing,
                key=lambda contract: (
                    -float(upside[minute, contract]),
                    -float(quality[minute, contract]),
                    int(contract),
                ),
            )[0]
        c_star_regret = sorted(
            [int(contract) for contract in eligible],
            key=lambda contract: (
                -float(upside[minute, contract]),
                -float(quality[minute, contract]),
                int(contract),
            ),
        )[0]
        wait_rows.append(
            {
                "minute": minute,
                "y_wait": int(wait),
                "eligible_count": int(len(eligible)),
                "c_star": c_star,
                "c_star_regret": c_star_regret,
            }
        )
        for contract in eligible:
            regret_rows.append(
                {
                    "minute": minute,
                    "contract": int(contract),
                    "regret": float(
                        upside[minute, c_star_regret]
                        - upside[minute, contract]
                    ),
                }
            )
    return {
        "wait_rows": wait_rows,
        "regret_rows": regret_rows,
        "zero_eligible_minutes": int(
            np.sum(~np.any(intent_eligible, axis=1))
        ),
    }


def finite_sample_qhat(scores: np.ndarray, coverage: float) -> float:
    ordered = np.sort(np.asarray(scores, dtype=float))
    rank = min(
        len(ordered),
        int(math.ceil((len(ordered) + 1) * coverage)),
    )
    return float(ordered[rank - 1])


def run_pipeline(
    composer_implementation: Callable[..., Any],
) -> dict[str, Any]:
    rng = np.random.default_rng(20260730)
    minutes = 18
    contracts = 5
    minute_feature = np.linspace(-1.0, 1.0, minutes)
    contract_feature = np.linspace(-0.8, 0.8, contracts)
    quality = (
        0.55 * minute_feature[:, None]
        - 0.25 * np.abs(contract_feature)[None, :]
        + rng.normal(0.0, 0.08, size=(minutes, contracts))
    )
    upside = np.clip(
        0.45
        + 0.18 * minute_feature[:, None]
        - 0.12 * contract_feature[None, :]
        + rng.normal(0.0, 0.04, size=(minutes, contracts)),
        0.0,
        1.0,
    )
    fee_cleared = upside > 0.42
    intent_eligible = np.ones((minutes, contracts), dtype=bool)
    intent_eligible[0, :] = False
    intent_eligible[5, 4] = False

    # composer_implementation is deliberately never passed to the target builder.
    targets = build_rlac_targets(
        quality=quality,
        upside=upside,
        fee_cleared=fee_cleared,
        intent_eligible=intent_eligible,
        quality_threshold=-0.05,
    )
    wait_rows = targets["wait_rows"]
    regret_rows = targets["regret_rows"]

    wait_x = np.asarray(
        [
            [
                minute_feature[row["minute"]],
                row["eligible_count"] / contracts,
            ]
            for row in wait_rows
        ],
        dtype=float,
    )
    wait_y = np.asarray([row["y_wait"] for row in wait_rows], dtype=float)
    wait_fit = np.arange(len(wait_rows)) < 11
    wait_coef = ridge_fit(wait_x[wait_fit], wait_y[wait_fit])
    wait_prob = sigmoid(predict(wait_x, wait_coef))

    regret_x = np.asarray(
        [
            [
                minute_feature[row["minute"]],
                contract_feature[row["contract"]],
            ]
            for row in regret_rows
        ],
        dtype=float,
    )
    regret_y = np.asarray(
        [row["regret"] for row in regret_rows],
        dtype=float,
    )
    regret_fit = np.asarray(
        [row["minute"] < 11 for row in regret_rows],
        dtype=bool,
    )
    regret_cal = ~regret_fit
    regret_coef = ridge_fit(regret_x[regret_fit], regret_y[regret_fit])
    regret_pred = predict(regret_x, regret_coef)
    qhat = finite_sample_qhat(
        np.maximum(
            0.0,
            regret_y[regret_cal] - regret_pred[regret_cal],
        ),
        0.90,
    )
    regret_upper = np.minimum(1.0, regret_pred + qhat)

    freeze_payload = {
        "wait_coefficients": wait_coef.tolist(),
        "regret_coefficients": regret_coef.tolist(),
        "qhat": qhat,
    }
    freeze_hash = hashlib.sha256(canonical_bytes(freeze_payload)).hexdigest()

    actions: list[dict[str, Any]] = []
    regret_lookup = {
        (row["minute"], row["contract"]): float(regret_upper[index])
        for index, row in enumerate(regret_rows)
    }
    wait_lookup = {
        row["minute"]: float(wait_prob[index])
        for index, row in enumerate(wait_rows)
    }
    for minute in range(minutes):
        eligible = np.flatnonzero(intent_eligible[minute])
        if len(eligible) == 0:
            actions.append({"minute": minute, "action": "WAIT_ZERO_ELIGIBLE"})
            continue
        action = composer_implementation(
            minute=minute,
            eligible=[int(value) for value in eligible],
            p_wait=wait_lookup[minute],
            regret_upper=regret_lookup,
        )
        actions.append({"minute": minute, **action})

    calibration_mask = ~wait_fit
    wait_ece = float(
        abs(
            wait_prob[calibration_mask].mean()
            - wait_y[calibration_mask].mean()
        )
    )
    marginal_coverage = float(
        np.mean(regret_y[regret_cal] <= regret_upper[regret_cal])
    )
    return {
        "targets": targets,
        "target_sha256": hashlib.sha256(
            canonical_bytes(targets)
        ).hexdigest(),
        "freeze_sha256": freeze_hash,
        "gate_metrics": {
            "wait_ECE_one_bin_fixture": wait_ece,
            "regret_marginal_coverage": marginal_coverage,
            "regret_calibration_rows": int(regret_cal.sum()),
        },
        "actions": actions,
    }


def composer_a(
    *,
    minute: int,
    eligible: list[int],
    p_wait: float,
    regret_upper: dict[tuple[int, int], float],
) -> dict[str, Any]:
    candidates = [
        contract
        for contract in eligible
        if regret_upper.get((minute, contract), math.inf) <= 0.10
    ]
    if p_wait >= 0.65 or not candidates:
        return {"action": "WAIT"}
    return {"action": "ENTER", "contract": min(candidates)}


def composer_b(**_: Any) -> dict[str, Any]:
    return {"action": "WAIT_ALTERNATE_IMPLEMENTATION"}


def load_checker_module() -> Any:
    path = HERE / "check_cross_contract_consistency_v3.py"
    spec = importlib.util.spec_from_file_location("consistency_v3", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load checker v3")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "t1_rlac_pipeline_output.json",
    )
    args = parser.parse_args()
    primary = run_pipeline(composer_a)
    swapped = run_pipeline(composer_b)
    repeated = run_pipeline(composer_a)
    checker = load_checker_module()
    positive_provenance = {
        "allowed_ancestors": [
            "frozen_FT2_04_labels",
            "canonical_time_t_intent_masks",
            "fit_role_CDFs",
        ],
        "runtime_or_model_ancestors": [],
    }
    circular_negative_control = {
        "allowed_ancestors": ["frozen_FT2_04_labels"],
        "runtime_or_model_ancestors": [
            "runtime_composer_selection",
        ],
    }
    assertions = {
        "targets_compute_with_composer_stubbed": bool(primary["targets"]),
        "targets_byte_identical_under_composer_swap": (
            primary["target_sha256"] == swapped["target_sha256"]
        ),
        "gate_metrics_decidable_and_reproducible": (
            primary["gate_metrics"] == repeated["gate_metrics"]
            and all(
                math.isfinite(float(value))
                for value in primary["gate_metrics"].values()
            )
        ),
        "positive_target_provenance_accepted": (
            checker.validate_target_provenance(positive_provenance)
        ),
        "circular_negative_control_detected": (
            not checker.validate_target_provenance(
                circular_negative_control
            )
        ),
    }
    payload = {
        "schema_version": "Protocol101FT2ScopedRoundRLACPipelineTestV1",
        "fixture_only": True,
        "model_training_scope": "tiny_synthetic_fixture_only",
        "assertions": assertions,
        "target_sha256": primary["target_sha256"],
        "freeze_sha256": primary["freeze_sha256"],
        "gate_metrics": primary["gate_metrics"],
        "composer_action_count": len(primary["actions"]),
        "outcome": "pass" if all(assertions.values()) else "fail",
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"outcome": payload["outcome"], "assertions": assertions}))
    return 0 if payload["outcome"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
