"""Run the predeclared outcome-blind attribution of V5's instability."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from v5.ops.build_causal_day_dataset import canonical_json, file_sha256
from v5.ops.train_causal_day_magnitude import (
    BATCH_MINUTES,
    _action_minute_indices,
    _device_batch,
)
from v5.research.causal_day_architectures import (
    computed_parameter_counts,
    declared_dimensions,
)
from v5.research.causal_day_attribution import (
    activation_drift,
    classify_attribution,
    contract_proxy_metrics,
    decompose_contract_scores,
    score_decomposition_metrics,
    state_channel_metrics,
)
from v5.research.causal_day_fit_cache import (
    FeatureScaler,
    cache_path,
    collate_minutes,
    load_cached_session,
    verify_cache_index,
)
from v5.research.causal_day_magnitude import (
    TARGET_SCALE_POINTS,
    build_magnitude_policy,
    chronological_folds,
)


PRIMARY_ARCHITECTURE = "neural_four_head"
PRIMARY_HORIZON = 120
SCORE_COLUMN = "predicted_depth_120m"
CANDIDATE_COLUMNS = (
    "session",
    "entry_minute",
    "contract_id",
    "right",
    "self_delta",
    "entry_ask_usd",
    "spread_usd",
    "moneyness_itm_points",
    "self_iv",
    "self_gamma",
    "self_theta_per_minute",
    "self_vega",
    "volume",
    "open_interest",
)
PROXIES = (
    "is_call",
    "abs_delta",
    "entry_ask_usd",
    "spread_usd",
    "moneyness_itm_points",
    "self_iv",
    "self_gamma",
    "self_theta_per_minute",
    "self_vega",
    "volume",
    "open_interest",
)


def _verified(path: Path, schema: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    expected = value.get("receipt_sha256")
    unsigned = dict(value)
    unsigned.pop("receipt_sha256", None)
    if hashlib.sha256(canonical_json(unsigned)).hexdigest() != expected:
        raise RuntimeError(f"receipt self-hash mismatch: {path}")
    if value.get("schema_version") != schema:
        raise RuntimeError(f"unexpected receipt schema at {path}")
    return value


def _load_scaler(path: Path) -> FeatureScaler:
    value = json.loads(path.read_text())
    return FeatureScaler(
        candle_mean=np.asarray(value["candle_mean"], dtype=np.float32),
        candle_scale=np.asarray(value["candle_scale"], dtype=np.float32),
        ladder_mean=np.asarray(value["ladder_mean"], dtype=np.float32),
        ladder_scale=np.asarray(value["ladder_scale"], dtype=np.float32),
    )


def _fit_prediction(fit: dict) -> dict:
    matches = [
        row
        for row in fit["predictions"]
        if row["architecture"] == PRIMARY_ARCHITECTURE
        and row["horizon_minutes"] == PRIMARY_HORIZON
        and row["shuffled_label_null"] is False
    ]
    if len(matches) != 1:
        raise RuntimeError("fit receipt does not identify one real V5 prediction member")
    value = matches[0]
    if file_sha256(Path(value["path"])) != value["sha256"]:
        raise RuntimeError("V5 prediction hash mismatch")
    return value


def _checkpoint_by_fold(calibration: dict) -> dict[int, dict]:
    values = {
        int(row["fold"]): row
        for row in calibration["checkpoints"]
        if row["null"] is False
    }
    if set(values) != set(range(1, 6)):
        raise RuntimeError("calibration does not identify five real checkpoints")
    for row in values.values():
        if file_sha256(Path(row["path"])) != row["sha256"]:
            raise RuntimeError("V5 checkpoint hash mismatch")
    return values


def _fold_cutoffs(calibration: dict) -> dict[int, float]:
    values = {
        int(row["fold"]): float(row["cutoff_points"])
        for row in calibration["selector"]["calibrations"]["real"]
    }
    if set(values) != set(range(1, 6)):
        raise RuntimeError("calibration does not identify five real cutoffs")
    return values


@torch.no_grad()
def score_decomposed_population(
    *,
    cache_root: Path,
    folds: tuple,
    checkpoints: dict[int, dict],
    scaler: FeatureScaler,
    device: torch.device,
) -> tuple[pd.DataFrame, float]:
    rows: list[dict[str, object]] = []
    worst_reconstruction = 0.0
    for fold in folds:
        model = build_magnitude_policy(
            PRIMARY_ARCHITECTURE, declared_dimensions()
        ).to(device)
        state = torch.load(
            Path(checkpoints[fold.number]["path"]),
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state)
        model.eval()
        for session in fold.score_sessions:
            cached = load_cached_session(cache_path(cache_root, session))
            indices = _action_minute_indices(cached)
            for start in range(0, len(indices), BATCH_MINUTES):
                chosen = indices[start : start + BATCH_MINUTES]
                batch, _, metadata = collate_minutes(cached, chosen, scaler)
                device_batch = _device_batch(batch, device)
                parts = decompose_contract_scores(model, device_batch)
                worst_reconstruction = max(
                    worst_reconstruction, parts.reconstruction_max_abs_error
                )
                action_mask = batch.entry_action_mask.numpy()
                cursor = 0
                for batch_row in range(len(chosen)):
                    offsets = np.flatnonzero(action_mask[batch_row])
                    for node_index in offsets:
                        key = metadata[cursor]
                        cursor += 1
                        rows.append(
                            {
                                "session": key[0],
                                "entry_minute": key[1],
                                "contract_id": key[2],
                                "fold": fold.number,
                                "full_score": float(
                                    parts.full[batch_row, node_index].cpu()
                                    * TARGET_SCALE_POINTS
                                ),
                                "state_offset": float(
                                    parts.state_offset[batch_row].cpu()
                                    * TARGET_SCALE_POINTS
                                ),
                                "contract_node_contribution": float(
                                    parts.contract_node[batch_row, node_index].cpu()
                                    * TARGET_SCALE_POINTS
                                ),
                                "no_candle_state_offset": float(
                                    parts.no_candle_state_offset[batch_row].cpu()
                                    * TARGET_SCALE_POINTS
                                ),
                                "no_ladder_summary_offset": float(
                                    parts.no_ladder_summary_offset[batch_row].cpu()
                                    * TARGET_SCALE_POINTS
                                ),
                                "neutral_clock_offset": float(
                                    parts.neutral_clock_offset[batch_row].cpu()
                                    * TARGET_SCALE_POINTS
                                ),
                            }
                        )
                if cursor != len(metadata):
                    raise RuntimeError("decomposition metadata order drift")
        print(
            f"attributed fold={fold.number} sessions={len(fold.score_sessions)}",
            flush=True,
        )
    return pd.DataFrame(rows), worst_reconstruction


def _join_predictions(
    decomposed: pd.DataFrame,
    predictions_path: Path,
) -> tuple[pd.DataFrame, float]:
    keys = ["session", "entry_minute", "contract_id", "fold"]
    prediction = pd.read_parquet(predictions_path, columns=[*keys, SCORE_COLUMN])
    if decomposed.duplicated(keys).any() or prediction.duplicated(keys).any():
        raise RuntimeError("decomposition/prediction key is not one-to-one")
    merged = decomposed.merge(
        prediction, on=keys, how="left", validate="one_to_one"
    )
    if len(merged) != len(prediction) or merged[SCORE_COLUMN].isna().any():
        raise RuntimeError("decomposition population differs from frozen OOF predictions")
    error = float(
        np.max(
            np.abs(
                merged["full_score"].to_numpy(float)
                - merged[SCORE_COLUMN].to_numpy(float)
            )
        )
    )
    if error > 1e-5:
        raise RuntimeError(f"checkpoint reconstruction differs from OOF predictions: {error}")
    return merged.drop(columns=[SCORE_COLUMN]), error


def _join_causal_proxies(
    scores: pd.DataFrame,
    candidates_path: Path,
) -> pd.DataFrame:
    keys = ["session", "entry_minute", "contract_id"]
    candidates = pd.read_parquet(candidates_path, columns=list(CANDIDATE_COLUMNS))
    if candidates.duplicated(keys).any():
        raise RuntimeError("causal candidate proxy key is not one-to-one")
    merged = scores.merge(candidates, on=keys, how="left", validate="one_to_one")
    if len(merged) != len(scores) or merged["right"].isna().any():
        raise RuntimeError("a decomposed score has no causal proxy row")
    merged["is_call"] = merged["right"].astype(str).eq("C").astype(float)
    merged["abs_delta"] = pd.to_numeric(
        merged["self_delta"], errors="coerce"
    ).abs()
    return merged


def run(
    *,
    declaration_path: Path,
    fit_receipt_path: Path,
    calibration_path: Path,
    cache_root: Path,
    candidates_path: Path,
    output_root: Path,
    evidence_dir: Path,
    device_name: str,
) -> dict[str, Any]:
    if output_root.exists() or evidence_dir.exists():
        raise RuntimeError("refusing to overwrite V5 attribution outputs")
    declaration = _verified(
        declaration_path, "v5.causal-day-v5-instability-attribution-declaration.v2"
    )
    fit = _verified(fit_receipt_path, "v5.causal-day-magnitude-fit.v1")
    calibration = _verified(
        calibration_path, "v5.causal-day-rank-selector-calibration.v1"
    )
    expected_inputs = declaration["immutable_inputs"]
    for name, path in (
        ("fit_receipt", fit_receipt_path),
        ("rank_calibration", calibration_path),
        ("candidate_table", candidates_path),
        ("cache_receipt", cache_root / "receipt.json"),
    ):
        if expected_inputs[name]["sha256"] != file_sha256(path):
            raise RuntimeError(f"attribution input changed after declaration: {name}")
    for path, expected in declaration["implementation_hashes"].items():
        if file_sha256(Path(path)) != expected:
            raise RuntimeError(f"attribution implementation changed after declaration: {path}")
    if calibration.get("economics_read") is not False:
        raise RuntimeError("rank calibration is not pre-economics")

    cache_receipt, manifest = verify_cache_index(cache_root)
    sessions = sorted(manifest["session"].astype(str))
    folds = chronological_folds(sessions)
    scaler_path = Path(fit["scaler"]["path"])
    if file_sha256(scaler_path) != fit["scaler"]["sha256"]:
        raise RuntimeError("frozen scaler hash mismatch")
    scaler = _load_scaler(scaler_path)
    checkpoints = _checkpoint_by_fold(calibration)
    prediction_info = _fit_prediction(fit)
    if Path(calibration["candidate_table"]["path"]) != candidates_path:
        raise RuntimeError("candidate path differs from the pre-economics calibration")

    device = torch.device(device_name)
    if device.type == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    torch.use_deterministic_algorithms(True)
    decomposed, reconstruction_error = score_decomposed_population(
        cache_root=cache_root,
        folds=folds,
        checkpoints=checkpoints,
        scaler=scaler,
        device=device,
    )
    scores, prediction_error = _join_predictions(
        decomposed, Path(prediction_info["path"])
    )
    scored_proxies = _join_causal_proxies(scores, candidates_path)

    activation = activation_drift(scores, fold_cutoffs=_fold_cutoffs(calibration))
    decomposition = score_decomposition_metrics(scores)
    channels = state_channel_metrics(scores)
    proxies = contract_proxy_metrics(scored_proxies, PROXIES)
    classification = classify_attribution(
        activation=activation,
        decomposition=decomposition,
        channels=channels,
        proxies=proxies,
    )

    output_root.mkdir(parents=True, exist_ok=False)
    evidence_dir.mkdir(parents=True, exist_ok=False)
    scores_path = output_root / "decomposed_oof_scores.parquet"
    scores.to_parquet(scores_path, index=False)
    metrics_path = output_root / "attribution_metrics.json"
    metrics = {
        "activation_drift": activation,
        "score_decomposition": decomposition,
        "state_channel_attribution": channels,
        "contract_proxy_attribution": proxies,
        "classification": classification,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    receipt: dict[str, Any] = {
        "schema_version": "v5.causal-day-v5-instability-attribution.v1",
        "created_on": "2026-08-14",
        "status": "COMPLETE_OUTCOME_BLIND_ATTRIBUTION",
        "declaration": {
            "path": str(declaration_path),
            "sha256": file_sha256(declaration_path),
        },
        "immutable_inputs": expected_inputs,
        "population": {
            "sessions": int(cache_receipt["sessions"]),
            "scored_sessions": sum(len(fold.score_sessions) for fold in folds),
            "oof_contract_scores": len(scores),
            "first_session": str(scores["session"].min()),
            "last_session": str(scores["session"].max()),
            "reserved_sessions_present": bool(
                scores["session"].astype(str).ge("2026-08-06").any()
            ),
        },
        "integrity": {
            "checkpoint_score_reconstruction_max_abs_error": reconstruction_error,
            "stored_oof_prediction_max_abs_error": prediction_error,
            "candidate_columns_read": list(CANDIDATE_COLUMNS),
            "pnl_columns_read": [],
            "target_columns_used": [],
            "new_selector_or_threshold_evaluated": False,
            "economic_subgroup_selected": False,
        },
        "metrics": metrics,
        "artifacts": {
            "decomposed_scores": {
                "path": str(scores_path),
                "sha256": file_sha256(scores_path),
                "rows": len(scores),
            },
            "metrics": {
                "path": str(metrics_path),
                "sha256": file_sha256(metrics_path),
            },
        },
        "model_parameters": computed_parameter_counts()[PRIMARY_ARCHITECTURE],
        "conservative_parameter_budget_range": [29, 50],
        "implementation_hashes": declaration["implementation_hashes"],
    }
    receipt["receipt_sha256"] = hashlib.sha256(canonical_json(receipt)).hexdigest()
    receipt_path = evidence_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(receipt_path)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--fit-receipt", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    run(
        declaration_path=args.declaration,
        fit_receipt_path=args.fit_receipt,
        calibration_path=args.calibration,
        cache_root=args.cache_root,
        candidates_path=args.candidates,
        output_root=args.output_root,
        evidence_dir=args.evidence_dir,
        device_name=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
