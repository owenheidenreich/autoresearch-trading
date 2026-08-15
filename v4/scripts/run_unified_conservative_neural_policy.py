"""Train the first preregistered unified conservative neural policy.

This is a bounded training run on the frozen serial-DP oracle scope. It does
not change the paper default, does not call broker endpoints, and does not make
a better-than-Protocol101 claim.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v4.model.supervised_pilot import FeatureScaler
from v4.model.unified_conservative_neural_policy import (
    ROLE_LABEL,
    TRAINING_SPEC_LABEL,
    ConservativeAdvantageMLP,
    ConservativeNeuralPolicyConfig,
    conservative_action_allowed,
    loss_for_batch,
)
from v4.model.unified_conservative_policy import validate_no_future_feature_columns
from v4.model.unified_policy_trajectory import FLAT_TRAJECTORY_FEATURE_COLUMNS, HOLDING_TRAJECTORY_FEATURE_COLUMNS


DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/unified_conservative_neural_policy_v1")
DEFAULT_DOC_PATH = Path("v4/docs/UNIFIED_CONSERVATIVE_NEURAL_POLICY_V1.md")
DEFAULT_FLAT_DATASET = Path("v4/audit/autoresearch/v4_aplus_hypothesis_270_full_surface_action_advantage_dataset/full_surface_action_advantage.parquet")
DEFAULT_HOLDING_DATASET = Path("v4/audit/autoresearch/v4_aplus_hypothesis_274_position_state_action_advantage_dataset/position_state_action_advantage.parquet")
DEFAULT_SESSION_MANIFEST = Path("v4/audit/autoresearch/unified_serial_dp_oracle/serial_dp_session_manifest.csv")
DEFAULT_ORACLE_SUMMARY = Path("v4/audit/autoresearch/unified_serial_dp_oracle/summary.json")

TRAIN_SPLITS = ("q3_2025", "q4_2025")
VALIDATION_SPLIT = "q1_2026"
DIAGNOSTIC_SPLIT = "recent_2026"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flat-dataset", type=Path, default=DEFAULT_FLAT_DATASET)
    parser.add_argument("--holding-dataset", type=Path, default=DEFAULT_HOLDING_DATASET)
    parser.add_argument("--session-manifest", type=Path, default=DEFAULT_SESSION_MANIFEST)
    parser.add_argument("--oracle-summary", type=Path, default=DEFAULT_ORACLE_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC_PATH)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-flat-train-rows", type=int, default=250_000)
    parser.add_argument("--max-flat-eval-rows", type=int, default=120_000)
    parser.add_argument("--max-holding-train-rows", type=int, default=300_000)
    parser.add_argument("--max-holding-eval-rows", type=int, default=150_000)
    parser.add_argument("--flat-train-positive-fraction", type=float, default=0.0)
    parser.add_argument("--flat-positive-class-weight", type=float, default=1.0)
    parser.add_argument("--flat-positive-regression-weight", type=float, default=1.0)
    parser.add_argument("--flat-tail-class-weight", type=float, default=1.0)
    parser.add_argument("--skip-doc", action="store_true")
    parser.add_argument("--skip-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = ConservativeNeuralPolicyConfig(epochs=int(args.epochs), batch_size=int(args.batch_size), hidden_dim=int(args.hidden_dim))
    flat_config = replace(
        config,
        positive_class_weight=float(args.flat_positive_class_weight),
        positive_regression_weight=float(args.flat_positive_regression_weight),
        tail_class_weight=float(args.flat_tail_class_weight),
    )
    holding_config = config
    oracle = load_json(args.oracle_summary)
    session_manifest = pd.read_csv(args.session_manifest)
    included_sessions = included_session_keys(session_manifest)
    flat_feature_columns = existing_columns(args.flat_dataset, FLAT_TRAJECTORY_FEATURE_COLUMNS)
    holding_feature_columns = existing_columns(args.holding_dataset, HOLDING_TRAJECTORY_FEATURE_COLUMNS)
    validate_no_future_feature_columns(flat_feature_columns)
    validate_no_future_feature_columns(holding_feature_columns)

    flat = load_flat(args.flat_dataset, flat_feature_columns, included_sessions)
    holding = load_holding(args.holding_dataset, holding_feature_columns, included_sessions)

    flat_train, flat_val, flat_diag = split_and_sample(
        flat,
        train_limit=int(args.max_flat_train_rows),
        eval_limit=int(args.max_flat_eval_rows),
        seed=int(args.seed),
        positive_column="target_positive",
        train_positive_fraction=float(args.flat_train_positive_fraction),
    )
    hold_train, hold_val, hold_diag = split_and_sample(
        holding,
        train_limit=int(args.max_holding_train_rows),
        eval_limit=int(args.max_holding_eval_rows),
        seed=int(args.seed) + 97,
    )

    flat_model, flat_scaler, flat_history = train_head(
        flat_train,
        flat_val,
        feature_columns=flat_feature_columns,
        target_column="a_enter",
        positive_column="target_positive",
        tail_column="target_tail",
        seed=int(args.seed),
        config=flat_config,
    )
    holding_model, holding_scaler, holding_history = train_head(
        hold_train,
        hold_val,
        feature_columns=holding_feature_columns,
        target_column="a_hold",
        positive_column="target_positive",
        tail_column="target_tail",
        seed=int(args.seed) + 1,
        config=holding_config,
    )

    artifact_dir = args.out_dir / "model_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(flat_model.state_dict(), artifact_dir / "flat_entry_model.pt")
    torch.save(holding_model.state_dict(), artifact_dir / "holding_lifecycle_model.pt")
    write_json(artifact_dir / "flat_scaler.json", flat_scaler.to_dict())
    write_json(artifact_dir / "holding_scaler.json", holding_scaler.to_dict())
    manifest = {
        "role_label": ROLE_LABEL,
        "training_spec": TRAINING_SPEC_LABEL,
        "seed": int(args.seed),
        "config": config.to_dict(),
        "flat_training_config": flat_config.to_dict(),
        "holding_training_config": holding_config.to_dict(),
        "flat_train_positive_fraction": float(args.flat_train_positive_fraction),
        "train_splits": list(TRAIN_SPLITS),
        "validation_split": VALIDATION_SPLIT,
        "diagnostic_split": DIAGNOSTIC_SPLIT,
        "flat_feature_columns": flat_feature_columns,
        "holding_feature_columns": holding_feature_columns,
        "oracle_summary": str(args.oracle_summary),
    }
    write_json(artifact_dir / "manifest.json", manifest)

    payload = {
        "role_label": ROLE_LABEL,
        "training_spec": TRAINING_SPEC_LABEL,
        "what_is_this": "preregistered model training / conservative unified neural policy",
        "changes_paper_default": False,
        "paper_default_baseline": "PAPER_DEFAULT_PROTOCOL101",
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": True,
        "decision": "conservative_neural_policy_trained_replay_and_challenge_still_blocked",
        "challenge_allowed": False,
        "challenge_blockers": [
            "strict serial replay not yet run for this model",
            "calibrated stochastic fill model unavailable",
            "untouched holdout data pending",
            "live no-order full-action parity pending",
            "formal validation controls pending",
        ],
        "data_scope": {
            "oracle_decision": oracle.get("decision", "missing"),
            "train_splits": list(TRAIN_SPLITS),
            "validation_split": VALIDATION_SPLIT,
            "diagnostic_split": DIAGNOSTIC_SPLIT,
            "included_sessions": int(len(included_sessions)),
            "flat_rows_loaded": int(len(flat)),
            "holding_rows_loaded": int(len(holding)),
        },
        "training_rows": {
            "flat_train": int(len(flat_train)),
            "flat_validation": int(len(flat_val)),
            "flat_diagnostic": int(len(flat_diag)),
            "holding_train": int(len(hold_train)),
            "holding_validation": int(len(hold_val)),
            "holding_diagnostic": int(len(hold_diag)),
        },
        "label_balance": {
            "flat_train_positive_rate": positive_rate(flat_train, "target_positive"),
            "flat_validation_positive_rate": positive_rate(flat_val, "target_positive"),
            "flat_diagnostic_positive_rate": positive_rate(flat_diag, "target_positive"),
            "holding_train_positive_rate": positive_rate(hold_train, "target_positive"),
            "holding_validation_positive_rate": positive_rate(hold_val, "target_positive"),
            "holding_diagnostic_positive_rate": positive_rate(hold_diag, "target_positive"),
        },
        "flat_entry": {
            "feature_count": len(flat_feature_columns),
            "history": flat_history,
            "metrics": {
                "train": evaluate_head(flat_train, flat_model, flat_scaler, flat_feature_columns, "a_enter", flat_config),
                "validation": evaluate_head(flat_val, flat_model, flat_scaler, flat_feature_columns, "a_enter", flat_config),
                "diagnostic_recent": evaluate_head(flat_diag, flat_model, flat_scaler, flat_feature_columns, "a_enter", flat_config),
            },
        },
        "holding_lifecycle": {
            "feature_count": len(holding_feature_columns),
            "history": holding_history,
            "metrics": {
                "train": evaluate_head(hold_train, holding_model, holding_scaler, holding_feature_columns, "a_hold", holding_config),
                "validation": evaluate_head(hold_val, holding_model, holding_scaler, holding_feature_columns, "a_hold", holding_config),
                "diagnostic_recent": evaluate_head(hold_diag, holding_model, holding_scaler, holding_feature_columns, "a_hold", holding_config),
            },
        },
        "artifacts": {
            "manifest": str(artifact_dir / "manifest.json"),
            "flat_entry_model": str(artifact_dir / "flat_entry_model.pt"),
            "holding_lifecycle_model": str(artifact_dir / "holding_lifecycle_model.pt"),
            "flat_scaler": str(artifact_dir / "flat_scaler.json"),
            "holding_scaler": str(artifact_dir / "holding_scaler.json"),
        },
        "next_required_evidence": [
            "Run strict one-account serial replay with the trained flat-entry and holding-lifecycle heads.",
            "Keep Protocol101 as paper default until replay, fill, untouched holdout, live parity, and validation controls pass.",
        ],
        "outputs": {
            "summary": str(args.out_dir / "summary.json"),
            "report": str(args.out_dir / "report.md"),
            "doc": None if args.skip_doc else str(args.doc_path),
        },
    }
    write_json(args.out_dir / "summary.json", payload)
    report = render_report(payload)
    (args.out_dir / "report.md").write_text(report)
    if not args.skip_doc:
        args.doc_path.parent.mkdir(parents=True, exist_ok=True)
        args.doc_path.write_text(report)
    if not args.skip_ledger:
        append_ledger(payload, args.out_dir)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def existing_columns(path: Path, requested: tuple[str, ...]) -> list[str]:
    names = set(pq.ParquetFile(path).schema_arrow.names)
    return [column for column in requested if column in names]


def included_session_keys(frame: pd.DataFrame) -> set[tuple[str, str]]:
    included = frame[frame["included"].astype(bool)].copy()
    return set(zip(included["split"].astype(str), included["session"].astype(str)))


def load_flat(path: Path, feature_columns: list[str], sessions: set[tuple[str, str]]) -> pd.DataFrame:
    columns = list(dict.fromkeys(["split", "session", "candidate_uid", "a_enter", "candidate_pnl", *feature_columns]))
    frame = pd.read_parquet(path, columns=columns)
    frame = filter_sessions(frame, sessions)
    frame["a_enter"] = pd.to_numeric(frame["a_enter"], errors="coerce")
    frame["candidate_pnl"] = pd.to_numeric(frame["candidate_pnl"], errors="coerce")
    frame["target_positive"] = frame["a_enter"].gt(0.0).astype(float)
    frame["target_tail"] = frame["candidate_pnl"].le(-500.0).astype(float)
    return frame.dropna(subset=["a_enter", "candidate_pnl"]).reset_index(drop=True)


def load_holding(path: Path, feature_columns: list[str], sessions: set[tuple[str, str]]) -> pd.DataFrame:
    columns = list(dict.fromkeys(["split", "session", "candidate_uid", "a_hold", "current_pnl", *feature_columns]))
    frame = pd.read_parquet(path, columns=columns)
    frame = filter_sessions(frame, sessions)
    frame["a_hold"] = pd.to_numeric(frame["a_hold"], errors="coerce")
    frame["current_pnl"] = pd.to_numeric(frame["current_pnl"], errors="coerce")
    frame["target_positive"] = frame["a_hold"].gt(0.0).astype(float)
    frame["target_tail"] = frame["current_pnl"].le(-500.0).astype(float)
    return frame.dropna(subset=["a_hold", "current_pnl"]).reset_index(drop=True)


def filter_sessions(frame: pd.DataFrame, sessions: set[tuple[str, str]]) -> pd.DataFrame:
    if frame.empty or not sessions:
        return frame.iloc[0:0].copy()
    allowed = pd.MultiIndex.from_tuples(sessions, names=["split", "session"])
    current = pd.MultiIndex.from_frame(frame[["split", "session"]].astype(str))
    return frame.loc[current.isin(allowed)].copy()


def split_and_sample(
    frame: pd.DataFrame,
    *,
    train_limit: int,
    eval_limit: int,
    seed: int,
    positive_column: str | None = None,
    train_positive_fraction: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_source = frame[frame["split"].isin(TRAIN_SPLITS)]
    train = sample_training_rows(
        train_source,
        train_limit,
        seed=seed,
        positive_column=positive_column,
        positive_fraction=float(train_positive_fraction),
    )
    validation = sample_rows(frame[frame["split"].eq(VALIDATION_SPLIT)], eval_limit, seed=seed + 1)
    diagnostic = sample_rows(frame[frame["split"].eq(DIAGNOSTIC_SPLIT)], eval_limit, seed=seed + 2)
    return train, validation, diagnostic


def sample_training_rows(
    frame: pd.DataFrame,
    limit: int,
    *,
    seed: int,
    positive_column: str | None,
    positive_fraction: float,
) -> pd.DataFrame:
    if positive_column is None or positive_fraction <= 0.0 or positive_column not in frame.columns:
        return sample_rows(frame, limit, seed=seed)
    if limit <= 0 or frame.empty:
        return frame.copy().reset_index(drop=True)
    positive = frame[frame[positive_column].astype(float).gt(0.5)]
    negative = frame[~frame.index.isin(positive.index)]
    positive_target = min(len(positive), int(round(float(limit) * min(max(float(positive_fraction), 0.0), 1.0))))
    positive_sample = sample_rows(positive, positive_target, seed=seed) if positive_target > 0 else positive.iloc[0:0].copy()
    negative_target = max(0, int(limit) - len(positive_sample))
    negative_sample = sample_rows(negative, negative_target, seed=seed + 1)
    out = pd.concat([positive_sample, negative_sample], ignore_index=True, sort=False)
    if len(out) > limit:
        out = out.sample(n=int(limit), random_state=int(seed) + 2)
    return out.sample(frac=1.0, random_state=int(seed) + 3).reset_index(drop=True)


def sample_rows(frame: pd.DataFrame, limit: int, *, seed: int) -> pd.DataFrame:
    if limit <= 0 or len(frame) <= limit:
        return frame.copy().reset_index(drop=True)
    return frame.sample(n=int(limit), random_state=int(seed)).reset_index(drop=True)


def positive_rate(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    return float(frame[column].astype(float).gt(0.5).mean())


def train_head(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    positive_column: str,
    tail_column: str,
    seed: int,
    config: ConservativeNeuralPolicyConfig,
) -> tuple[ConservativeAdvantageMLP, FeatureScaler, list[dict[str, Any]]]:
    validate_training_inputs(train, validation, feature_columns, target_column)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    scaler = FeatureScaler.fit(train[feature_columns].to_numpy(dtype=np.float32))
    x_train, adv_train, pos_train, tail_train = tensors(train, scaler, feature_columns, target_column, positive_column, tail_column, config)
    x_val, adv_val, pos_val, tail_val = tensors(validation, scaler, feature_columns, target_column, positive_column, tail_column, config)
    model = ConservativeAdvantageMLP(input_dim=len(feature_columns), hidden_dim=config.hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(adv_train), torch.from_numpy(pos_train), torch.from_numpy(tail_train)),
        batch_size=config.batch_size,
        shuffle=True,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for bx, ba, bp, bt in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, parts = loss_for_batch(model(bx), advantage_target=ba, positive_target=bp, tail_target=bt, config=config)
            loss.backward()
            optimizer.step()
            losses.append(parts)
        val_loss = validation_loss(model, x_val, adv_val, pos_val, tail_val, config)
        is_best = bool(val_loss < best_val)
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(np.mean([row["loss"] for row in losses])) if losses else 0.0,
                "validation_loss": float(val_loss),
                "is_best": is_best,
            }
        )
    model.load_state_dict(best_state)
    return model, scaler, history


def validate_training_inputs(train: pd.DataFrame, validation: pd.DataFrame, feature_columns: list[str], target_column: str) -> None:
    if not feature_columns:
        raise ValueError("no feature columns are available for conservative neural policy training")
    if train.empty:
        raise ValueError(f"no training rows are available for target {target_column}")
    if validation.empty:
        raise ValueError(f"no validation rows are available for target {target_column}")


def tensors(
    frame: pd.DataFrame,
    scaler: FeatureScaler,
    feature_columns: list[str],
    target_column: str,
    positive_column: str,
    tail_column: str,
    config: ConservativeNeuralPolicyConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = scaler.transform(frame[feature_columns].to_numpy(dtype=np.float32))
    advantage = np.clip(frame[target_column].to_numpy(dtype=np.float32), -config.target_clip, config.target_clip) / config.target_scale
    positive = frame[positive_column].to_numpy(dtype=np.float32)
    tail = frame[tail_column].to_numpy(dtype=np.float32)
    return x, advantage.astype(np.float32), positive.astype(np.float32), tail.astype(np.float32)


def validation_loss(
    model: ConservativeAdvantageMLP,
    x: np.ndarray,
    advantage: np.ndarray,
    positive: np.ndarray,
    tail: np.ndarray,
    config: ConservativeNeuralPolicyConfig,
) -> float:
    model.eval()
    with torch.no_grad():
        loss, _ = loss_for_batch(
            model(torch.from_numpy(x)),
            advantage_target=torch.from_numpy(advantage),
            positive_target=torch.from_numpy(positive),
            tail_target=torch.from_numpy(tail),
            config=config,
        )
    return float(loss.detach().cpu())


def evaluate_head(
    frame: pd.DataFrame,
    model: ConservativeAdvantageMLP,
    scaler: FeatureScaler,
    feature_columns: list[str],
    target_column: str,
    config: ConservativeNeuralPolicyConfig,
) -> dict[str, Any]:
    if frame.empty:
        return {"rows": 0}
    x = scaler.transform(frame[feature_columns].to_numpy(dtype=np.float32))
    preds = predict(model, x, config)
    target = frame[target_column].to_numpy(dtype=float)
    allowed = np.asarray(
        [
            conservative_action_allowed(
                predicted_advantage=float(a),
                positive_probability=float(p),
                tail_probability=float(t),
                config=config,
            )["allowed"]
            for a, p, t in zip(preds["predicted_advantage"], preds["positive_probability"], preds["tail_probability"])
        ],
        dtype=bool,
    )
    pred_positive = preds["positive_probability"] >= 0.5
    true_positive = target > 0.0
    corr = float(np.corrcoef(preds["predicted_advantage"], target)[0, 1]) if len(frame) > 1 else 0.0
    if not np.isfinite(corr):
        corr = 0.0
    return {
        "rows": int(len(frame)),
        "target_mean": float(np.mean(target)),
        "target_median": float(np.median(target)),
        "mae": float(np.mean(np.abs(preds["predicted_advantage"] - target))),
        "corr": corr,
        "positive_rate": float(np.mean(true_positive)),
        "positive_accuracy": float(np.mean(pred_positive == true_positive)),
        "conservative_allowed_rows": int(allowed.sum()),
        "conservative_allowed_fraction": float(allowed.mean()),
        "allowed_true_advantage_mean": float(np.mean(target[allowed])) if allowed.any() else 0.0,
        "allowed_true_positive_rate": float(np.mean(true_positive[allowed])) if allowed.any() else 0.0,
        "predicted_advantage_mean": float(np.mean(preds["predicted_advantage"])),
        "positive_probability_mean": float(np.mean(preds["positive_probability"])),
        "tail_probability_mean": float(np.mean(preds["tail_probability"])),
    }


def predict(model: ConservativeAdvantageMLP, x: np.ndarray, config: ConservativeNeuralPolicyConfig) -> dict[str, np.ndarray]:
    model.eval()
    adv: list[np.ndarray] = []
    pos: list[np.ndarray] = []
    tail: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), 16_384):
            out = model(torch.from_numpy(x[start : start + 16_384]))
            adv.append(out["advantage"].cpu().numpy() * config.target_scale)
            pos.append(torch.sigmoid(out["positive_logit"]).cpu().numpy())
            tail.append(torch.sigmoid(out["tail_logit"]).cpu().numpy())
    return {
        "predicted_advantage": np.concatenate(adv),
        "positive_probability": np.concatenate(pos),
        "tail_probability": np.concatenate(tail),
    }


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        f"Training spec: `{payload['training_spec']}`",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        f"Paper default baseline: `{payload['paper_default_baseline']}`",
        "Paid data downloaded: no",
        "Broker endpoint called: no",
        "Model training: yes",
        f"Decision: `{payload['decision']}`",
        f"Challenge allowed: `{payload['challenge_allowed']}`",
        "",
        "## Scope",
        "",
        f"- Oracle decision: `{payload['data_scope']['oracle_decision']}`",
        f"- Train splits: `{payload['data_scope']['train_splits']}`",
        f"- Validation split: `{payload['data_scope']['validation_split']}`",
        f"- Diagnostic split: `{payload['data_scope']['diagnostic_split']}`",
        f"- Flat rows loaded: `{payload['data_scope']['flat_rows_loaded']}`",
        f"- Holding rows loaded: `{payload['data_scope']['holding_rows_loaded']}`",
        f"- Flat train positive rate: `{payload['label_balance']['flat_train_positive_rate']:.4f}`",
        f"- Holding train positive rate: `{payload['label_balance']['holding_train_positive_rate']:.4f}`",
        "",
        "## Metrics",
        "",
        "| head | split | rows | MAE | corr | allowed rows | allowed true adv mean | allowed true positive rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for head_name in ["flat_entry", "holding_lifecycle"]:
        for split_name, metrics in payload[head_name]["metrics"].items():
            lines.append(
                f"| {head_name} | {split_name} | {metrics.get('rows', 0)} | {metrics.get('mae', 0.0):.2f} | "
                f"{metrics.get('corr', 0.0):.3f} | {metrics.get('conservative_allowed_rows', 0)} | "
                f"{metrics.get('allowed_true_advantage_mean', 0.0):.2f} | {metrics.get('allowed_true_positive_rate', 0.0):.3f} |"
            )
    lines.extend(["", "## Challenge Blockers", ""])
    lines.extend(f"- {item}" for item in payload["challenge_blockers"])
    lines.extend(["", "## Next Required Evidence", ""])
    lines.extend(f"{idx}. {item}" for idx, item in enumerate(payload["next_required_evidence"], start=1))
    lines.extend(["", "## Artifacts", ""])
    for name, path in payload["artifacts"].items():
        lines.append(f"- {name}: `{path}`")
    lines.extend(["", "## Outputs", ""])
    for name, path in payload["outputs"].items():
        lines.append(f"- {name}: `{path}`")
    return "\n".join(lines) + "\n"


def append_ledger(payload: dict[str, Any], out_dir: Path) -> None:
    ledger = Path("v4/ledger/RESEARCH_LEDGER.md")
    if not ledger.exists():
        return
    marker = f"## {ROLE_LABEL}"
    text = ledger.read_text()
    if marker in text:
        return
    with ledger.open("a") as handle:
        handle.write(
            "\n".join(
                [
                    "",
                    marker,
                    "",
                    f"- What is this: {payload['what_is_this']}",
                    "- Changes paper default: no",
                    "- Paid data downloaded: no",
                    "- Broker endpoint called: no",
                    "- Model training: yes",
                    f"- Decision: `{payload['decision']}`",
                    f"- Report: `{out_dir / 'report.md'}`",
                    "- Result: Trained only; Protocol101 challenge remains blocked.",
                ]
            )
            + "\n"
        )


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
