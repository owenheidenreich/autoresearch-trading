"""Run the one reopened chronological multi-horizon magnitude fit.

This command deliberately stops at out-of-fold predictions.  It never reads a
P&L column.  Economics are a separate command that first verifies the feature
timestamp audit and all seven signed kill conditions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from v5.ops.build_causal_day_dataset import canonical_json, file_sha256
from v5.research.causal_day_architectures import (
    CausalPolicyBatch,
    computed_parameter_counts,
    declared_dimensions,
)
from v5.research.causal_day_fit_cache import (
    CachedSession,
    FeatureScaler,
    cache_path,
    collate_minutes,
    fit_initial_scaler,
    load_cached_session,
    verify_cache_index,
)
from v5.research.causal_day_magnitude import (
    HORIZONS,
    PERMITTED_ARCHITECTURES,
    build_magnitude_policy,
    chronological_folds,
    magnitude_loss,
    parameter_count,
    stratum_weights,
)
from v5.research.causal_day_policy_gate import (
    REOPENED_CORPUS,
    REOPENED_LABEL,
    REQUIRED_KILL_CONDITIONS,
    assert_fit_permitted,
    load_reopening,
)


SEED = 39_202_608_14
BATCH_MINUTES = 16
INITIAL_EPOCHS = 3
UPDATE_EPOCHS = 1
LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.0001
GRADIENT_CLIP = 5.0


def _stable_seed(*values: object) -> int:
    digest = hashlib.sha256("|".join(map(str, values)).encode()).digest()
    return int.from_bytes(digest[:4], "little")


def _device_batch(batch: CausalPolicyBatch, device: torch.device) -> CausalPolicyBatch:
    return CausalPolicyBatch(
        candles=batch.candles.to(device),
        candle_mask=batch.candle_mask.to(device),
        ladder=batch.ladder.to(device),
        ladder_mask=batch.ladder_mask.to(device),
        entry_action_mask=batch.entry_action_mask.to(device),
        account=batch.account.to(device),
        position=batch.position.to(device),
        clock=batch.clock.to(device),
        roles=batch.roles,
    )


def _action_minute_indices(cached: CachedSession) -> np.ndarray:
    counts = np.add.reduceat(
        cached.action_mask.astype(np.int16), cached.ladder_offsets[:-1]
    )
    return np.flatnonzero(counts > 0)


def _training_targets(paths: Iterable[Path]) -> np.ndarray:
    values = []
    for path in paths:
        cached = load_cached_session(path)
        values.append(cached.targets[cached.action_mask])
    return np.concatenate(values, axis=0)


def shuffled_target_assignments(
    paths: list[Path], seed: int
) -> dict[str, np.ndarray]:
    """Shuffle outcome vectors within side and 5-point moneyness strata.

    The three horizons move together so the null preserves their nesting.  It
    also preserves side and contract geometry, removing only the association
    between a causal market state and the later magnitude path.
    """

    cached_values = [load_cached_session(path) for path in paths]
    targets = np.concatenate(
        [value.targets[value.action_mask] for value in cached_values], axis=0
    )
    is_call = np.concatenate(
        [value.ladder[value.action_mask, 21] for value in cached_values]
    ).astype(int)
    moneyness_node = np.concatenate(
        [np.round(value.ladder[value.action_mask, 20] / 5.0).astype(int) for value in cached_values]
    )
    strata = is_call * 100 + moneyness_node
    rng = np.random.default_rng(seed)
    shuffled = targets.copy()
    for stratum in np.unique(strata):
        locations = np.flatnonzero(strata == stratum)
        shuffled[locations] = targets[rng.permutation(locations)]
    result = {}
    cursor = 0
    for cached in cached_values:
        count = int(cached.action_mask.sum())
        result[cached.session] = shuffled[cursor : cursor + count]
        cursor += count
    return result


def _override_targets(cached: CachedSession, action_targets: np.ndarray | None) -> np.ndarray | None:
    if action_targets is None:
        return None
    if len(action_targets) != int(cached.action_mask.sum()):
        raise RuntimeError("shuffled target assignment count drift")
    values = cached.targets.copy()
    values[cached.action_mask] = action_targets
    return values


def train_stage(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    paths: list[Path],
    scaler: FeatureScaler,
    weights: torch.Tensor,
    device: torch.device,
    architecture: str,
    horizon: int,
    null: bool,
    stage: int,
    epochs: int,
) -> list[float]:
    assignments = (
        shuffled_target_assignments(paths, _stable_seed(SEED, architecture, "null", stage))
        if null
        else {}
    )
    losses = []
    model.train()
    for epoch in range(epochs):
        rng = np.random.default_rng(_stable_seed(SEED, architecture, null, stage, epoch))
        order = rng.permutation(len(paths))
        running_loss = 0.0
        running_batches = 0
        for path_index in order:
            cached = load_cached_session(paths[int(path_index)])
            target_override = _override_targets(cached, assignments.get(cached.session))
            indices = _action_minute_indices(cached)
            indices = rng.permutation(indices)
            for start in range(0, len(indices), BATCH_MINUTES):
                chosen = indices[start : start + BATCH_MINUTES]
                batch, targets, _ = collate_minutes(
                    cached, chosen, scaler, targets_override=target_override
                )
                batch = _device_batch(batch, device)
                targets = targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                horizon_index = HORIZONS.index(horizon)
                predictions = model(batch).contract_logits
                loss = magnitude_loss(
                    predictions,
                    targets[:, :, horizon_index],
                    batch.entry_action_mask,
                    weights,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                optimizer.step()
                running_loss += float(loss.detach().cpu())
                running_batches += 1
        losses.append(running_loss / max(running_batches, 1))
    return losses


@torch.no_grad()
def score_sessions(
    model: torch.nn.Module,
    *,
    paths: list[Path],
    scaler: FeatureScaler,
    device: torch.device,
    architecture: str,
    horizon: int,
    null: bool,
    fold: int,
) -> pd.DataFrame:
    model.eval()
    rows = []
    for path in paths:
        cached = load_cached_session(path)
        indices = _action_minute_indices(cached)
        for start in range(0, len(indices), BATCH_MINUTES):
            chosen = indices[start : start + BATCH_MINUTES]
            batch, _, metadata = collate_minutes(cached, chosen, scaler)
            batch = _device_batch(batch, device)
            values = model(batch).contract_logits[batch.entry_action_mask]
            values = values.detach().cpu().numpy() * 30.0
            if len(values) != len(metadata):
                raise RuntimeError("prediction metadata order drift")
            for key, prediction in zip(metadata, values, strict=True):
                rows.append(
                    {
                        "session": key[0],
                        "entry_minute": key[1],
                        "contract_id": key[2],
                        "architecture": architecture,
                        "horizon_minutes": horizon,
                        "shuffled_label_null": null,
                        "fold": fold,
                        f"predicted_depth_{horizon}m": float(prediction),
                    }
                )
    return pd.DataFrame(rows)


def run(
    *,
    cache_root: Path,
    declaration_path: Path,
    out_root: Path,
    evidence_dir: Path,
    device_name: str,
) -> dict:
    if out_root.exists() or evidence_dir.exists():
        raise RuntimeError("refusing to overwrite fit outputs or evidence")
    declaration = json.loads(declaration_path.read_text())
    expected = declaration.get("receipt_sha256")
    unsigned = dict(declaration)
    unsigned.pop("receipt_sha256", None)
    if hashlib.sha256(canonical_json(unsigned)).hexdigest() != expected:
        raise RuntimeError("fit declaration self-hash mismatch")
    if declaration.get("schema_version") != "v5.causal-day-trader-fit-declaration.v4":
        raise RuntimeError("the re-ruled fit requires declaration v4")

    cache_receipt, manifest = verify_cache_index(cache_root)
    sessions = sorted(manifest["session"].astype(str))
    reopening = load_reopening()
    dimensions = declared_dimensions()
    computed_counts = computed_parameter_counts()
    declared_architectures = declaration.get("architectures", {})
    if declared_architectures.get("hidden_size") != dimensions.hidden_size:
        raise RuntimeError("declaration hidden size differs from the frozen built width")
    if declared_architectures.get("permitted") != {
        name: computed_counts[name] for name in PERMITTED_ARCHITECTURES
    }:
        raise RuntimeError("declaration parameter counts differ from built-model counts")
    # Fail before creating an output directory if any architecture/horizon is
    # outside the signed scope or its actual built count exceeds the budget.
    for architecture in PERMITTED_ARCHITECTURES:
        for horizon in HORIZONS:
            assert_fit_permitted(
                architecture,
                sessions=len(sessions),
                trainable_parameters=computed_counts[architecture],
                reopening=reopening,
                label=REOPENED_LABEL,
                horizon=horizon,
                corpus=REOPENED_CORPUS,
                declared_kill_conditions=REQUIRED_KILL_CONDITIONS,
            )
    folds = chronological_folds(sessions)
    initial_paths = [cache_path(cache_root, value) for value in folds[0].train_sessions]
    scaler = fit_initial_scaler(initial_paths)
    weights_array = stratum_weights(_training_targets(initial_paths))

    device = torch.device(device_name)
    if device.type == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    torch.manual_seed(SEED)
    np.random.seed(SEED % (2**32))
    torch.use_deterministic_algorithms(True)
    out_root.mkdir(parents=True, exist_ok=False)
    evidence_dir.mkdir(parents=True, exist_ok=False)
    prediction_artifacts = []
    fit_rows = []

    for architecture in PERMITTED_ARCHITECTURES:
        for horizon_index, horizon in enumerate(HORIZONS):
            for null in (False, True):
                torch.manual_seed(_stable_seed(SEED, architecture, horizon, null))
                model = build_magnitude_policy(architecture, dimensions).to(device)
                parameters = parameter_count(model)
                if parameters != computed_counts[architecture]:
                    raise RuntimeError("fit model count drifted after gate preflight")
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
                )
                weights = torch.from_numpy(weights_array[horizon_index]).to(device)
                predictions = []
                history = []
                for fold in folds:
                    if fold.number == 1:
                        train_sessions = list(fold.train_sessions)
                        epochs = INITIAL_EPOCHS
                    else:
                        train_sessions = list(folds[fold.number - 2].score_sessions)
                        epochs = UPDATE_EPOCHS
                    losses = train_stage(
                        model,
                        optimizer,
                        paths=[cache_path(cache_root, value) for value in train_sessions],
                        scaler=scaler,
                        weights=weights,
                        device=device,
                        architecture=architecture,
                        horizon=horizon,
                        null=null,
                        stage=fold.number,
                        epochs=epochs,
                    )
                    history.append(
                        {
                            "fold": fold.number,
                            "incremental_train_sessions": train_sessions,
                            "epochs": epochs,
                            "losses": losses,
                        }
                    )
                    predictions.append(
                        score_sessions(
                            model,
                            paths=[
                                cache_path(cache_root, value)
                                for value in fold.score_sessions
                            ],
                            scaler=scaler,
                            device=device,
                            architecture=architecture,
                            horizon=horizon,
                            null=null,
                            fold=fold.number,
                        )
                    )
                    checkpoint = (
                        out_root
                        / "checkpoints"
                        / architecture
                        / f"{horizon}m"
                        / ("shuffled" if null else "real")
                        / f"fold{fold.number}.pt"
                    )
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), checkpoint)
                prediction = pd.concat(predictions, ignore_index=True)
                prediction_path = out_root / (
                    f"predictions_{architecture}_{horizon}m_"
                    f"{'shuffled' if null else 'real'}.parquet"
                )
                prediction.to_parquet(prediction_path, index=False)
                prediction_artifacts.append(
                    {
                        "architecture": architecture,
                        "horizon_minutes": horizon,
                        "shuffled_label_null": null,
                        "path": str(prediction_path),
                        "sha256": file_sha256(prediction_path),
                        "rows": len(prediction),
                    }
                )
                fit_rows.append(
                    {
                        "architecture": architecture,
                        "horizon_minutes": horizon,
                        "shuffled_label_null": null,
                        "parameters": parameters,
                        "training_history": history,
                    }
                )
                print(
                    f"fit {architecture} horizon={horizon} null={null}; "
                    f"predictions={len(prediction):,}",
                    flush=True,
                )

    scaler_path = out_root / "initial_prefix_scaler.json"
    scaler_path.write_text(json.dumps(scaler.to_json(), indent=2, sort_keys=True) + "\n")
    payload = {
        "schema_version": "v5.causal-day-magnitude-fit.v1",
        "created_on": "2026-08-14",
        "declaration": {"path": str(declaration_path), "sha256": file_sha256(declaration_path)},
        "reopening": {
            "path": str(Path("v5/governance/CAUSAL_DAY_FIT_REOPENING_2026_08_14.md")),
            "sha256": reopening.document_sha256,
            "kill_conditions": list(REQUIRED_KILL_CONDITIONS),
        },
        "cache_receipt": {"path": str(cache_root / "receipt.json"), "sha256": file_sha256(cache_root / "receipt.json")},
        "device": str(device),
        "seed": SEED,
        "folds": [
            {
                "number": value.number,
                "train_first": min(value.train_sessions),
                "train_last": max(value.train_sessions),
                "train_sessions": len(value.train_sessions),
                "score_first": min(value.score_sessions),
                "score_last": max(value.score_sessions),
                "score_sessions": len(value.score_sessions),
            }
            for value in folds
        ],
        "fit_configuration": {
            "batch_minutes": BATCH_MINUTES,
            "initial_epochs": INITIAL_EPOCHS,
            "update_epochs": UPDATE_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRADIENT_CLIP,
            "target": "maximum ITM depth clipped [-25,30] and scaled by 30",
            "horizon_models": "one canonical scalar contract-head model per declared horizon",
            "loss": "stratum-weighted smooth-L1 beta=0.25",
            "null": "training-only outcome-vector permutation within side and 5-point moneyness node",
            "normalization": "mean/std on the initial 93-session prefix only, frozen forward, clipped [-10,10]",
        },
        "scaler": {"path": str(scaler_path), "sha256": file_sha256(scaler_path)},
        "stratum_weights": weights_array.tolist(),
        "fits": fit_rows,
        "predictions": prediction_artifacts,
        "economics_read": False,
        "threshold_tuned": False,
        "four_independent_fit": False,
        "implementation_hashes": {
            str(Path(__file__)): file_sha256(Path(__file__)),
            "v5/research/causal_day_magnitude.py": file_sha256(Path("v5/research/causal_day_magnitude.py")),
            "v5/research/causal_day_fit_cache.py": file_sha256(Path("v5/research/causal_day_fit_cache.py")),
            "v5/research/causal_day_policy_gate.py": file_sha256(Path("v5/research/causal_day_policy_gate.py")),
        },
    }
    payload["receipt_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    receipt_path = evidence_dir / "receipt.json"
    receipt_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(receipt_path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    run(
        cache_root=args.cache_root,
        declaration_path=args.declaration,
        out_root=args.out_root,
        evidence_dir=args.evidence_dir,
        device_name=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
