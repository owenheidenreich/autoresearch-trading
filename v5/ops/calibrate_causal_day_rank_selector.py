"""Calibrate V5 rank cutoffs from fold-training prefixes, before economics."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

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
from v5.research.causal_day_fit_cache import (
    FeatureScaler,
    cache_path,
    collate_minutes,
    load_cached_session,
    verify_cache_index,
)
from v5.research.causal_day_magnitude import (
    build_magnitude_policy,
    chronological_folds,
)
from v5.research.causal_day_policy_gate import (
    REOPENED_CORPUS,
    REOPENED_LABEL,
    REQUIRED_KILL_CONDITIONS,
    assert_fit_permitted,
    load_reopening,
)
from v5.research.causal_day_selection import (
    assert_cutoff_precedes_score_sessions,
    calibrate_rank_cutoff,
)


PRIMARY_ARCHITECTURE = "neural_four_head"
PRIMARY_HORIZON = 120
TARGET_SIGNAL_MINUTES_PER_SESSION = 2
DIAGNOSTIC_SCORE_POINTS = 25.0


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


def _load_scaler(path: Path) -> FeatureScaler:
    value = json.loads(path.read_text())
    return FeatureScaler(
        candle_mean=np.asarray(value["candle_mean"], dtype=np.float32),
        candle_scale=np.asarray(value["candle_scale"], dtype=np.float32),
        ladder_mean=np.asarray(value["ladder_mean"], dtype=np.float32),
        ladder_scale=np.asarray(value["ladder_scale"], dtype=np.float32),
    )


@torch.no_grad()
def score_training_minute_maxima(
    model: torch.nn.Module,
    *,
    paths: list[Path],
    scaler: FeatureScaler,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, object]] = []
    for path in paths:
        cached = load_cached_session(path)
        indices = _action_minute_indices(cached)
        for start in range(0, len(indices), BATCH_MINUTES):
            chosen = indices[start : start + BATCH_MINUTES]
            batch, _, _ = collate_minutes(cached, chosen, scaler)
            device_batch = _device_batch(batch, device)
            predictions = model(device_batch).contract_logits.detach().cpu().numpy() * 30.0
            action_mask = batch.entry_action_mask.numpy()
            for row, minute_index in enumerate(chosen):
                values = predictions[row][action_mask[row]]
                if not len(values) or not np.isfinite(values).all():
                    raise RuntimeError("training minute has no finite eligible score")
                rows.append(
                    {
                        "session": cached.session,
                        "entry_minute": str(cached.minutes[int(minute_index)]),
                        "predicted_depth_120m": float(values.max()),
                    }
                )
    return pd.DataFrame(rows)


def _prediction_member(fit: dict, *, null: bool) -> dict:
    matches = [
        value
        for value in fit["predictions"]
        if value["architecture"] == PRIMARY_ARCHITECTURE
        and value["horizon_minutes"] == PRIMARY_HORIZON
        and value["shuffled_label_null"] is null
    ]
    if len(matches) != 1:
        raise RuntimeError("fit receipt does not contain exactly one primary prediction member")
    value = matches[0]
    if file_sha256(Path(value["path"])) != value["sha256"]:
        raise RuntimeError("OOF prediction hash mismatch")
    return value


def ranking_diagnostics(
    *,
    real_predictions: pd.DataFrame,
    shuffled_predictions: pd.DataFrame,
    candidates_path: Path,
) -> dict:
    keys = ["session", "entry_minute", "contract_id"]
    score = "predicted_depth_120m"
    if not real_predictions[keys].equals(shuffled_predictions[keys]):
        raise RuntimeError("real and shuffled OOF prediction populations differ")
    labels = pd.read_parquet(
        candidates_path, columns=[*keys, "maximum_itm_depth_120m"]
    )
    if labels.duplicated(keys).any():
        raise RuntimeError("magnitude label key is not one-to-one")
    joined = real_predictions[keys + [score]].merge(
        labels, on=keys, how="left", validate="one_to_one"
    )
    target = pd.to_numeric(joined["maximum_itm_depth_120m"], errors="raise")
    prediction = pd.to_numeric(joined[score], errors="raise")
    if target.isna().any() or not np.isfinite(prediction.to_numpy(float)).all():
        raise RuntimeError("ranking diagnostic population is incomplete")
    correlation = float(np.corrcoef(prediction.to_numpy(float), target.to_numpy(float))[0, 1])
    order = np.argsort(prediction.to_numpy(float), kind="mergesort")
    deciles = []
    for index, locations in enumerate(np.array_split(order, 10), start=1):
        deciles.append(
            {
                "decile_low_to_high": index,
                "rows": int(len(locations)),
                "mean_prediction_points": float(prediction.iloc[locations].mean()),
                "mean_actual_depth_points": float(target.iloc[locations].mean()),
            }
        )
    real_score = pd.to_numeric(real_predictions[score], errors="raise").to_numpy(float)
    shuffled_score = pd.to_numeric(shuffled_predictions[score], errors="raise").to_numpy(float)
    return {
        "population_rows": len(joined),
        "pearson_prediction_vs_actual_depth": correlation,
        "prediction_rank_deciles": deciles,
        "diagnostic_point_25": {
            "threshold_is_not_a_trading_selector": True,
            "real_predictions_at_or_above": int(np.count_nonzero(real_score >= DIAGNOSTIC_SCORE_POINTS)),
            "shuffled_predictions_at_or_above": int(
                np.count_nonzero(shuffled_score >= DIAGNOSTIC_SCORE_POINTS)
            ),
        },
        "prediction_sample_standard_deviation": {
            "real": float(np.std(real_score, ddof=1)),
            "shuffled": float(np.std(shuffled_score, ddof=1)),
        },
        "columns_read_from_candidate_table": [*keys, "maximum_itm_depth_120m"],
        "pnl_or_trade_outcome_used_for_selector": False,
    }


def _quintile_edges(candidates_path: Path, initial_sessions: tuple[str, ...]) -> dict:
    frame = pd.read_parquet(
        candidates_path,
        columns=["session", "self_delta", "entry_ask_usd"],
        filters=[("session", "in", list(initial_sessions))],
    )
    frame = frame[frame["session"].astype(str).isin(initial_sessions)]
    delta = pd.to_numeric(frame["self_delta"], errors="raise").abs().to_numpy(float)
    premium = pd.to_numeric(frame["entry_ask_usd"], errors="raise").to_numpy(float)
    if not len(frame) or not np.isfinite(delta).all() or not np.isfinite(premium).all():
        raise RuntimeError("matched-control training-prefix strata are incomplete")
    return {
        "source_sessions": len(initial_sessions),
        "source_first": initial_sessions[0],
        "source_last": initial_sessions[-1],
        "delta_definition": "absolute contemporaneous Black-Scholes delta",
        "premium_definition": "contemporaneous entry ask dollars for one contract",
        "delta_internal_edges": np.quantile(delta, [0.2, 0.4, 0.6, 0.8]).tolist(),
        "premium_internal_edges_usd": np.quantile(
            premium, [0.2, 0.4, 0.6, 0.8]
        ).tolist(),
        "outcome_columns_read": False,
    }


def run(
    *,
    declaration_path: Path,
    fit_receipt_path: Path,
    cache_root: Path,
    candidates_path: Path,
    out_dir: Path,
    device_name: str,
) -> dict:
    if out_dir.exists():
        raise RuntimeError(f"refusing to overwrite calibration: {out_dir}")
    declaration = _verified(
        declaration_path, "v5.causal-day-trader-fit-declaration.v5"
    )
    fit = _verified(fit_receipt_path, "v5.causal-day-magnitude-fit.v1")
    if declaration["fit_reuse"]["fit_receipt"]["sha256"] != file_sha256(
        fit_receipt_path
    ):
        raise RuntimeError("V5 declaration is not bound to the supplied fit receipt")
    if fit.get("economics_read") is not False or fit.get("threshold_tuned") is not False:
        raise RuntimeError("fit receipt crossed the pre-economics firewall")

    cache_receipt, manifest = verify_cache_index(cache_root)
    sessions = sorted(manifest["session"].astype(str))
    folds = chronological_folds(sessions)
    counts = computed_parameter_counts()
    assert_fit_permitted(
        PRIMARY_ARCHITECTURE,
        sessions=len(sessions),
        trainable_parameters=counts[PRIMARY_ARCHITECTURE],
        reopening=load_reopening(),
        label=REOPENED_LABEL,
        horizon=PRIMARY_HORIZON,
        corpus=REOPENED_CORPUS,
        declared_kill_conditions=REQUIRED_KILL_CONDITIONS,
    )
    if declaration["fit_reuse"]["computed_trainable_parameters"] != counts[PRIMARY_ARCHITECTURE]:
        raise RuntimeError("V5 parameter count differs from the built model")

    scaler_path = Path(fit["scaler"]["path"])
    if file_sha256(scaler_path) != fit["scaler"]["sha256"]:
        raise RuntimeError("frozen initial-prefix scaler hash mismatch")
    scaler = _load_scaler(scaler_path)
    device = torch.device(device_name)
    if device.type == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    torch.use_deterministic_algorithms(True)
    checkpoint_root = Path(_prediction_member(fit, null=False)["path"]).parent / "checkpoints"
    calibrations: dict[str, list[dict]] = {"real": [], "shuffled": []}
    checkpoint_hashes = []
    for null_name in ("real", "shuffled"):
        for fold in folds:
            checkpoint = (
                checkpoint_root
                / PRIMARY_ARCHITECTURE
                / f"{PRIMARY_HORIZON}m"
                / null_name
                / f"fold{fold.number}.pt"
            )
            if not checkpoint.is_file():
                raise RuntimeError(f"missing fit checkpoint: {checkpoint}")
            model = build_magnitude_policy(PRIMARY_ARCHITECTURE, declared_dimensions()).to(device)
            state = torch.load(checkpoint, map_location=device, weights_only=True)
            model.load_state_dict(state)
            training_scores = score_training_minute_maxima(
                model,
                paths=[cache_path(cache_root, value) for value in fold.train_sessions],
                scaler=scaler,
                device=device,
            )
            cutoff = calibrate_rank_cutoff(
                training_scores,
                fold=fold.number,
                score_column="predicted_depth_120m",
                target_signal_minutes_per_session=TARGET_SIGNAL_MINUTES_PER_SESSION,
            )
            assert_cutoff_precedes_score_sessions(cutoff, fold.score_sessions)
            calibrations[null_name].append(asdict(cutoff))
            checkpoint_hashes.append(
                {
                    "null": null_name == "shuffled",
                    "fold": fold.number,
                    "path": str(checkpoint),
                    "sha256": file_sha256(checkpoint),
                }
            )
            print(
                f"calibrated {null_name} fold={fold.number} "
                f"train={len(fold.train_sessions)} cutoff={cutoff.cutoff_points:.6f}",
                flush=True,
            )

    real_info = _prediction_member(fit, null=False)
    shuffled_info = _prediction_member(fit, null=True)
    real_predictions = pd.read_parquet(real_info["path"])
    shuffled_predictions = pd.read_parquet(shuffled_info["path"])
    diagnostics = ranking_diagnostics(
        real_predictions=real_predictions,
        shuffled_predictions=shuffled_predictions,
        candidates_path=candidates_path,
    )
    matched_edges = _quintile_edges(candidates_path, folds[0].train_sessions)

    out_dir.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema_version": "v5.causal-day-rank-selector-calibration.v1",
        "created_on": "2026-08-14",
        "status": "PASS_BEFORE_ECONOMICS",
        "declaration": {
            "path": str(declaration_path),
            "sha256": file_sha256(declaration_path),
        },
        "fit_receipt": {
            "path": str(fit_receipt_path),
            "sha256": file_sha256(fit_receipt_path),
        },
        "cache_receipt": {
            "path": str(cache_root / "receipt.json"),
            "sha256": file_sha256(cache_root / "receipt.json"),
            "verified_sessions": int(cache_receipt["sessions"]),
        },
        "candidate_table": {
            "path": str(candidates_path),
            "sha256": file_sha256(candidates_path),
        },
        "selector": {
            "mode": "causal_training_prefix_rank",
            "target_signal_minutes_per_training_session": TARGET_SIGNAL_MINUTES_PER_SESSION,
            "contract_scores_collapsed_to_minute_maximum": True,
            "current_score_session_used_for_cutoff": False,
            "label_or_pnl_used_for_cutoff": False,
            "calibrations": calibrations,
        },
        "ranking_diagnostics": diagnostics,
        "matched_control_quintiles": matched_edges,
        "checkpoints": checkpoint_hashes,
        "economics_read": False,
        "bid_prices_read": False,
        "fit_gate": {
            "status": "PERMITTED",
            "architecture": PRIMARY_ARCHITECTURE,
            "horizon_minutes": PRIMARY_HORIZON,
            "parameters": counts[PRIMARY_ARCHITECTURE],
            "kill_conditions": list(REQUIRED_KILL_CONDITIONS),
        },
        "implementation_sha256": file_sha256(Path(__file__)),
    }
    receipt["receipt_sha256"] = hashlib.sha256(canonical_json(receipt)).hexdigest()
    receipt_path = out_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(receipt_path)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--fit-receipt", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    run(
        declaration_path=args.declaration,
        fit_receipt_path=args.fit_receipt,
        cache_root=args.cache_root,
        candidates_path=args.candidates,
        out_dir=args.out_dir,
        device_name=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
