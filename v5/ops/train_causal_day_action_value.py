"""Fit the single compact Q(WAIT)-versus-Q(ENTER) policy when authorized.

The current gate refuses this architecture and label.  This command therefore
performs every integrity check and calls the unchanged gate before loading a
target session, creating output, or constructing an optimizer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from v5.ops.build_causal_day_dataset import canonical_json, file_sha256
from v5.research.causal_day_action_advantage import HORIZON_MINUTES, LABEL_NAME
from v5.research.causal_day_action_value_targets import (
    TARGET_SCALE_USD,
    action_value_loss,
    collate_action_values,
    load_target_session,
    shuffled_session_surface,
    target_path,
)
from v5.research.causal_day_action_value_attainability import (
    action_value_selector_attainability,
)
from v5.research.causal_day_architectures import (
    CausalPolicyBatch,
    computed_parameter_counts,
)
from v5.research.causal_day_compact_interaction import (
    ARCHITECTURE_NAME,
    CompactInteractionEntryPolicy,
)
from v5.research.causal_day_declaration_reseal import (
    assert_mechanical_reseal,
    load_declaration as load_reseal_declaration,
    load_pinned_source_pair,
)
from v5.research.causal_day_fit_cache import (
    FeatureScaler,
    cache_path,
    collate_minutes,
    fit_initial_scaler,
    load_cached_session,
    verify_cache_index,
)
from v5.research.causal_day_magnitude import chronological_folds
from v5.research.causal_day_policy_gate import (
    REOPENED_CORPUS,
    REOPENED_RESEARCH_LAW_SHA256,
    REQUIRED_KILL_CONDITIONS,
    assert_fit_permitted,
    load_reopening,
)


SEED = 39_301_202_608_14
BATCH_MINUTES = 16
INITIAL_EPOCHS = 3
UPDATE_EPOCHS = 1
LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.0001
GRADIENT_CLIP = 5.0
DECLARATION_SCHEMA = "v5.causal-day-action-value-fit-declaration.v3"


def canonical_parameter_count() -> int:
    """Return the registry-built count used by both governance and receipts."""

    return computed_parameter_counts()[ARCHITECTURE_NAME]


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


def train_stage(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    feature_paths: list[Path],
    target_root: Path,
    scaler: FeatureScaler,
    device: torch.device,
    null: bool,
    stage: int,
    epochs: int,
) -> list[float]:
    losses = []
    model.train()
    for epoch in range(epochs):
        rng = np.random.default_rng(_stable_seed(SEED, null, stage, epoch))
        order = rng.permutation(len(feature_paths))
        running = 0.0
        batches = 0
        for path_index in order:
            cached = load_cached_session(feature_paths[int(path_index)])
            targets = load_target_session(target_path(target_root, cached.session))
            if null:
                targets = shuffled_session_surface(
                    cached,
                    targets,
                    seed=_stable_seed(SEED, "null", stage, cached.session),
                )
            indices = rng.permutation(np.arange(cached.minute_count))
            for start in range(0, len(indices), BATCH_MINUTES):
                chosen = indices[start : start + BATCH_MINUTES]
                batch, _, _ = collate_minutes(cached, chosen, scaler)
                q_enter, q_wait = collate_action_values(cached, targets, chosen)
                batch = _device_batch(batch, device)
                q_enter = q_enter.to(device)
                q_wait = q_wait.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = action_value_loss(model(batch), q_enter, q_wait, batch.entry_action_mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                optimizer.step()
                running += float(loss.detach().cpu())
                batches += 1
        losses.append(running / max(batches, 1))
    return losses


@torch.no_grad()
def score_sessions(
    model: torch.nn.Module,
    *,
    feature_paths: list[Path],
    scaler: FeatureScaler,
    device: torch.device,
    null: bool,
    fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model.eval()
    candidate_rows = []
    minute_rows = []
    for path in feature_paths:
        cached = load_cached_session(path)
        all_indices = np.arange(cached.minute_count)
        for start in range(0, len(all_indices), BATCH_MINUTES):
            chosen = all_indices[start : start + BATCH_MINUTES]
            batch, _, metadata = collate_minutes(cached, chosen, scaler)
            batch = _device_batch(batch, device)
            scores = model(batch)
            values = (
                scores.contract_logits[batch.entry_action_mask].detach().cpu().numpy()
                * TARGET_SCALE_USD
            )
            if len(values) != len(metadata):
                raise RuntimeError("action-value prediction metadata order drift")
            for key, prediction in zip(metadata, values, strict=True):
                candidate_rows.append(
                    {
                        "session": key[0],
                        "entry_minute": key[1],
                        "contract_id": key[2],
                        "architecture": ARCHITECTURE_NAME,
                        "horizon_minutes": HORIZON_MINUTES,
                        "shuffled_label_null": null,
                        "fold": fold,
                        "predicted_q_enter_bid_120m_usd": float(prediction),
                    }
                )
            wait = scores.abstain_logits.detach().cpu().numpy() * TARGET_SCALE_USD
            for row, minute_index in enumerate(chosen):
                minute_rows.append(
                    {
                        "session": cached.session,
                        "entry_minute": str(cached.minutes[minute_index]),
                        "architecture": ARCHITECTURE_NAME,
                        "horizon_minutes": HORIZON_MINUTES,
                        "shuffled_label_null": null,
                        "fold": fold,
                        "predicted_q_wait_bid_120m_usd": float(wait[row]),
                    }
                )
    return pd.DataFrame(candidate_rows), pd.DataFrame(minute_rows)


def _verified_declaration(path: Path) -> dict:
    value = json.loads(path.read_text())
    expected = value.get("receipt_sha256")
    unsigned = dict(value)
    unsigned.pop("receipt_sha256", None)
    if hashlib.sha256(canonical_json(unsigned)).hexdigest() != expected:
        raise RuntimeError("action-value fit declaration self-hash mismatch")
    if value.get("schema_version") != DECLARATION_SCHEMA:
        raise RuntimeError("action-value fit declaration schema drift")
    for relative, digest in value.get("implementation_hashes", {}).items():
        path_value = Path(relative)
        if not path_value.is_file() or file_sha256(path_value) != digest:
            raise RuntimeError(f"action-value fit implementation hash mismatch: {relative}")
    return value


def run(
    *,
    feature_cache_root: Path,
    target_cache_root: Path,
    declaration_path: Path,
    out_root: Path,
    evidence_dir: Path,
    device_name: str,
) -> dict:
    if out_root.exists() or evidence_dir.exists():
        raise RuntimeError("refusing to overwrite action-value fit output or evidence")
    declaration = _verified_declaration(declaration_path)
    source_fit, source_evaluation = load_pinned_source_pair()
    paired_evaluation = load_reseal_declaration(
        Path(declaration["mechanical_reseal"]["paired_evaluation_declaration"])
    )
    reseal_hash = assert_mechanical_reseal(
        source_fit,
        source_evaluation,
        declaration,
        paired_evaluation,
    )
    if reseal_hash != REOPENED_RESEARCH_LAW_SHA256:
        raise RuntimeError("fit declaration is outside the signed research law")
    parameter_counts = computed_parameter_counts()
    parameters = parameter_counts[ARCHITECTURE_NAME]
    attainability = action_value_selector_attainability()
    if declaration["architecture"]["computed_parameters"] != parameters:
        raise RuntimeError("compact action-value canonical parameter count drift")
    if declaration["architecture"]["canonical_parameter_counts"] != parameter_counts:
        raise RuntimeError("declared canonical architecture family drift")
    if declaration["selector_attainability"] != attainability.to_dict():
        raise RuntimeError("selector attainability proof drift")
    _, feature_manifest = verify_cache_index(feature_cache_root)
    target_receipt_path = target_cache_root / "receipt.json"
    target_receipt = json.loads(target_receipt_path.read_text())
    target_expected = target_receipt.get("receipt_sha256")
    target_unsigned = dict(target_receipt)
    target_unsigned.pop("receipt_sha256", None)
    if hashlib.sha256(canonical_json(target_unsigned)).hexdigest() != target_expected:
        raise RuntimeError("target-cache receipt self-hash mismatch")
    if file_sha256(target_receipt_path) != declaration["inputs"]["target_cache_receipt_sha256"]:
        raise RuntimeError("target cache differs from fit declaration")
    sessions = sorted(feature_manifest["session"].astype(str))
    if len(sessions) != 243:
        raise RuntimeError("action-value fit requires all 243 declared sessions")
    # Binding fail-closed boundary: no target session, output directory,
    # optimizer, checkpoint, or fit exists before this call succeeds.
    reopening = load_reopening()
    assert_fit_permitted(
        ARCHITECTURE_NAME,
        selector_attainability=attainability,
        sessions=len(sessions),
        trainable_parameters=parameters,
        reopening=reopening,
        label=LABEL_NAME,
        horizon=HORIZON_MINUTES,
        corpus=REOPENED_CORPUS,
        declared_kill_conditions=REQUIRED_KILL_CONDITIONS,
    )

    folds = chronological_folds(sessions)
    initial_paths = [cache_path(feature_cache_root, value) for value in folds[0].train_sessions]
    scaler = fit_initial_scaler(initial_paths)
    device = torch.device(device_name)
    if device.type == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    torch.manual_seed(SEED)
    np.random.seed(SEED % (2**32))
    torch.use_deterministic_algorithms(True)
    out_root.mkdir(parents=True, exist_ok=False)
    evidence_dir.mkdir(parents=True, exist_ok=False)
    fit_rows = []
    artifacts = []
    for null in (False, True):
        torch.manual_seed(_stable_seed(SEED, null))
        model = CompactInteractionEntryPolicy().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        candidate_predictions = []
        minute_predictions = []
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
                feature_paths=[cache_path(feature_cache_root, value) for value in train_sessions],
                target_root=target_cache_root,
                scaler=scaler,
                device=device,
                null=null,
                stage=fold.number,
                epochs=epochs,
            )
            history.append(
                {"fold": fold.number, "incremental_train_sessions": train_sessions, "epochs": epochs, "losses": losses}
            )
            candidate, minute = score_sessions(
                model,
                feature_paths=[cache_path(feature_cache_root, value) for value in fold.score_sessions],
                scaler=scaler,
                device=device,
                null=null,
                fold=fold.number,
            )
            candidate_predictions.append(candidate)
            minute_predictions.append(minute)
            checkpoint = out_root / "checkpoints" / ("shuffled" if null else "real") / f"fold{fold.number}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint)
        candidate = pd.concat(candidate_predictions, ignore_index=True)
        minute = pd.concat(minute_predictions, ignore_index=True)
        suffix = "shuffled" if null else "real"
        candidate_path_out = out_root / f"candidate_predictions_{suffix}.parquet"
        minute_path_out = out_root / f"minute_predictions_{suffix}.parquet"
        candidate.to_parquet(candidate_path_out, index=False)
        minute.to_parquet(minute_path_out, index=False)
        for kind, path_value, rows in (
            ("candidate", candidate_path_out, len(candidate)),
            ("minute", minute_path_out, len(minute)),
        ):
            artifacts.append(
                {"null": null, "kind": kind, "path": str(path_value), "rows": rows, "sha256": file_sha256(path_value)}
            )
        fit_rows.append({"null": null, "parameters": parameters, "history": history})
    scaler_path = out_root / "initial_prefix_scaler.json"
    scaler_path.write_text(json.dumps(scaler.to_json(), indent=2, sort_keys=True) + "\n")
    payload = {
        "schema_version": "v5.causal-day-action-value-fit.v1",
        "created_on": "2026-08-14",
        "declaration": {"path": str(declaration_path), "sha256": file_sha256(declaration_path)},
        "architecture": ARCHITECTURE_NAME,
        "parameters": parameters,
        "parameter_count_source": "computed_parameter_counts()['compact_interaction_entry']",
        "canonical_parameter_counts": parameter_counts,
        "selector_attainability": attainability.to_dict(),
        "mechanical_reseal": {
            "status": "PASS_ZERO_RESEARCH_LAW_DIFFERENCES",
            "research_law_sha256": reseal_hash,
            "source_fit_declaration": str(
                declaration["mechanical_reseal"]["source_fit_declaration"]
            ),
            "source_evaluation_declaration": str(
                declaration["mechanical_reseal"]["source_evaluation_declaration"]
            ),
        },
        "label": LABEL_NAME,
        "horizon_minutes": HORIZON_MINUTES,
        "seed": SEED,
        "device": str(device),
        "fits": fit_rows,
        "predictions": artifacts,
        "scaler": {"path": str(scaler_path), "sha256": file_sha256(scaler_path)},
        "economics_read": False,
        "operating_point_tuned": False,
        "receipt_sha256": None,
    }
    payload.pop("receipt_sha256")
    payload["receipt_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    (evidence_dir / "receipt.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache-root", type=Path, required=True)
    parser.add_argument("--target-cache-root", type=Path, required=True)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    run(
        feature_cache_root=args.feature_cache_root,
        target_cache_root=args.target_cache_root,
        declaration_path=args.declaration,
        out_root=args.out_root,
        evidence_dir=args.evidence_dir,
        device_name=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
