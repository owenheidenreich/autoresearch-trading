"""CHALLENGER_POSITION_STATE_LIFECYCLE_POLICY_V1.

Train a research-only lifecycle model on DATASET_POSITION_STATE_ACTION_ADVANTAGE_V1.
This model learns holding-state `hold` versus `exit now` advantages; it does
not replace the paper default and does not submit broker orders.

No paid data is downloaded. No broker endpoint is called.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from v4.model.supervised_pilot import FeatureScaler


ROLE_LABEL = "CHALLENGER_POSITION_STATE_LIFECYCLE_POLICY_V1"
HISTORICAL_ID = "Protocol275"
DEFAULT_DATASET = Path("v4/audit/autoresearch/v4_aplus_hypothesis_274_position_state_action_advantage_dataset/position_state_action_advantage.parquet")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_275_position_state_lifecycle_policy")
MODEL_SEEDS = [1, 2, 3, 4, 5]
FEATURE_COLUMNS = [
    "minutes_since_entry",
    "minutes_since_open",
    "minutes_to_forced_flat",
    "entry_premium",
    "entry_a_enter",
    "entry_q_wait",
    "entry_q_enter",
    "entry_underlying_price",
    "entry_iv",
    "entry_delta",
    "entry_gamma",
    "entry_theta",
    "bid",
    "ask",
    "mid",
    "spread",
    "spread_frac",
    "bid_size",
    "ask_size",
    "quote_gap_seconds",
    "underlying_price",
    "iv",
    "delta",
    "gamma",
    "theta",
    "vega",
    "gamma_theta_ratio",
    "theta_over_mid",
    "current_pnl",
    "mfe_to_now",
    "mae_to_now",
    "giveback_from_mfe",
    "giveback_fraction",
    "pnl_velocity_1",
    "pnl_velocity_3",
    "pnl_velocity_5",
    "time_since_mfe_minutes",
]
FOLDS = [
    {
        "name": "fold1_train_q1_validate_q2_test_q3",
        "train_splits": ["q1_2025"],
        "validation_split": "q2_2025",
        "test_splits": ["q3_2025"],
    },
    {
        "name": "fold2_train_q1_q2_validate_q3_test_q4",
        "train_splits": ["q1_2025", "q2_2025"],
        "validation_split": "q3_2025",
        "test_splits": ["q4_2025"],
    },
    {
        "name": "fold3_train_q1_q2_q3_validate_q4_test_q1_2026",
        "train_splits": ["q1_2025", "q2_2025", "q3_2025"],
        "validation_split": "q4_2025",
        "test_splits": ["q1_2026"],
    },
    {
        "name": "fold4_train_2025_validate_q1_2026_test_recent",
        "train_splits": ["q1_2025", "q2_2025", "q3_2025", "q4_2025"],
        "validation_split": "q1_2026",
        "test_splits": ["recent_2026"],
    },
]


@dataclass(frozen=True)
class LifecyclePolicyConfig:
    epochs: int = 6
    batch_size: int = 4096
    hidden_dim: int = 128
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    advantage_scale: float = 500.0
    advantage_clip: float = 5000.0
    regression_weight: float = 0.15
    max_train_rows_per_fold: int = 800_000
    max_validation_rows: int = 300_000
    max_test_rows_per_split: int = 300_000
    min_validation_exit_recall: float = 0.35


class PositionLifecycleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.action_head = nn.Linear(hidden_dim, 2)
        self.advantage_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.body(features)
        return self.action_head(encoded), self.advantage_head(encoded).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seeds", nargs="*", type=int, default=MODEL_SEEDS)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-train-rows-per-fold", type=int, default=800_000)
    parser.add_argument("--max-validation-rows", type=int, default=300_000)
    parser.add_argument("--max-test-rows-per-split", type=int, default=300_000)
    parser.add_argument("--skip-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = load_dataset(args.dataset)
    config = LifecyclePolicyConfig(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        hidden_dim=int(args.hidden_dim),
        max_train_rows_per_fold=int(args.max_train_rows_per_fold),
        max_validation_rows=int(args.max_validation_rows),
        max_test_rows_per_split=int(args.max_test_rows_per_split),
    )
    fold_results = []
    for fold in FOLDS:
        train = sample_rows(data[data["split"].isin(fold["train_splits"])], int(config.max_train_rows_per_fold), seed=11)
        validation = sample_rows(data[data["split"].eq(fold["validation_split"])], int(config.max_validation_rows), seed=17)
        if train.empty or validation.empty:
            fold_results.append({"fold": fold["name"], "skipped": True, "reason": "missing_train_or_validation_rows", "splits": {}})
            continue
        for seed in args.seeds:
            model, scaler, history = train_model(train, validation, seed=int(seed), config=config)
            threshold = select_threshold(validation, model, scaler)
            artifact_dir = args.out_dir / "model_artifacts" / fold["name"] / f"seed_{seed}"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), artifact_dir / "model.pt")
            (artifact_dir / "scaler.json").write_text(json.dumps(scaler.to_dict(), indent=2, sort_keys=True) + "\n")
            (artifact_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "role_label": ROLE_LABEL,
                        "historical_protocol": HISTORICAL_ID,
                        "fold": fold["name"],
                        "seed": int(seed),
                        "feature_columns": FEATURE_COLUMNS,
                        "threshold": threshold,
                        "config": asdict(config),
                        "history": history,
                    },
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
                + "\n"
            )
            result = {"fold": fold["name"], "seed": int(seed), "threshold": threshold, "splits": {}}
            for split in fold["test_splits"]:
                test = sample_rows(data[data["split"].eq(split)], int(config.max_test_rows_per_split), seed=23)
                result["splits"][split] = evaluate(test, model, scaler, threshold=float(threshold["threshold"]))
                if split == "q1_2026":
                    march = test[pd.to_datetime(test["state_time"], utc=True).dt.strftime("%Y-%m").eq("2026-03")]
                    result["splits"]["march_2026"] = evaluate(march, model, scaler, threshold=float(threshold["threshold"]))
            fold_results.append(result)
    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "model / holding-state lifecycle advantage policy",
        "changes_paper_default": False,
        "candidate_label": ROLE_LABEL,
        "paper_default_label": "PAPER_DEFAULT_PROTOCOL101",
        "data_used": str(args.dataset),
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "feature_columns": FEATURE_COLUMNS,
        "config": asdict(config),
        "dataset_summary": summarize_dataset(data),
        "fold_results": fold_results,
        "aggregate": aggregate(fold_results),
        "decision": decide(fold_results),
        "next_experiment": "Wire this lifecycle policy into the unified entry simulation; do not promote until strict serial PnL beats PAPER_DEFAULT_PROTOCOL101.",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    if not args.skip_ledger:
        append_ledger(payload, args.out_dir)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def load_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=[*["split", "session", "state_time", "oracle_holding_action", "q_exit", "q_hold", "a_hold"], *FEATURE_COLUMNS])
    for column in FEATURE_COLUMNS + ["q_exit", "q_hold", "a_hold"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["target_hold"] = frame["oracle_holding_action"].astype(str).eq("hold").astype(int)
    return frame.dropna(subset=["split", "session", "state_time", "q_exit", "q_hold", "a_hold", "target_hold"]).reset_index(drop=True)


def sample_rows(frame: pd.DataFrame, limit: int, *, seed: int) -> pd.DataFrame:
    if limit <= 0 or len(frame) <= limit:
        return frame.copy()
    return frame.sample(n=limit, random_state=seed).sort_index().reset_index(drop=True)


def train_model(train: pd.DataFrame, validation: pd.DataFrame, *, seed: int, config: LifecyclePolicyConfig) -> tuple[PositionLifecycleMLP, FeatureScaler, list[dict[str, Any]]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    scaler = FeatureScaler.fit(train[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
    x_train, y_train, adv_train = tensors(train, scaler, config)
    x_val, y_val, adv_val = tensors(validation, scaler, config)
    model = PositionLifecycleMLP(input_dim=len(FEATURE_COLUMNS), hidden_dim=config.hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    weights = class_weights(y_train)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(adv_train)), batch_size=config.batch_size, shuffle=True)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for bx, by, ba in loader:
            optimizer.zero_grad(set_to_none=True)
            logits, pred_adv = model(bx)
            loss = nn.functional.cross_entropy(logits, by, weight=weights)
            loss = loss + config.regression_weight * nn.functional.huber_loss(pred_adv, ba, delta=1.0)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val = validation_loss(model, x_val, y_val, adv_val, weights, config)
        if val < best_val:
            best_val = val
            best_state = copy.deepcopy(model.state_dict())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": float(val), "is_best": val <= best_val})
    model.load_state_dict(best_state)
    return model, scaler, history


def tensors(frame: pd.DataFrame, scaler: FeatureScaler, config: LifecyclePolicyConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = scaler.transform(frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
    y = frame["target_hold"].to_numpy(dtype=np.int64)
    adv = np.clip(frame["a_hold"].to_numpy(dtype=np.float32), -config.advantage_clip, config.advantage_clip) / config.advantage_scale
    return x, y, adv.astype(np.float32)


def class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=2).astype(float)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def validation_loss(model: PositionLifecycleMLP, x: np.ndarray, y: np.ndarray, adv: np.ndarray, weights: torch.Tensor, config: LifecyclePolicyConfig) -> float:
    model.eval()
    with torch.no_grad():
        logits, pred_adv = model(torch.from_numpy(x))
        loss = nn.functional.cross_entropy(logits, torch.from_numpy(y), weight=weights)
        loss = loss + config.regression_weight * nn.functional.huber_loss(pred_adv, torch.from_numpy(adv), delta=1.0)
    return float(loss.detach().cpu())


def select_threshold(validation: pd.DataFrame, model: PositionLifecycleMLP, scaler: FeatureScaler) -> dict[str, Any]:
    scores = hold_scores(validation, model, scaler)
    finite = scores[np.isfinite(scores)]
    thresholds = sorted(set(np.quantile(finite, [0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95]).round(4).tolist() + [0.0])) if len(finite) else [0.0]
    sweep = []
    for threshold in thresholds:
        metrics = evaluate_scores(validation, scores, threshold=float(threshold))
        sweep.append({"threshold": float(threshold), **metrics})
    eligible = [row for row in sweep if row["exit_recall"] >= 0.35]
    pool = eligible if eligible else sweep
    best = max(pool, key=lambda row: (row["captured_advantage_vs_exit"], row["balanced_accuracy"], row["exit_recall"]))
    return {"threshold": float(best["threshold"]), "objective": "validation captured hold/exit advantage with exit-recall floor", "min_exit_recall": 0.35, "selected": best, "sweep": sweep}


def hold_scores(frame: pd.DataFrame, model: PositionLifecycleMLP, scaler: FeatureScaler) -> np.ndarray:
    if frame.empty:
        return np.asarray([], dtype=float)
    x = scaler.transform(frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(x), 65536):
            logits, pred_adv = model(torch.from_numpy(x[start : start + 65536]))
            prob_margin = (logits[:, 1] - logits[:, 0]).cpu().numpy()
            scores.append(prob_margin + pred_adv.cpu().numpy() * 0.25)
    return np.concatenate(scores)


def evaluate(frame: pd.DataFrame, model: PositionLifecycleMLP, scaler: FeatureScaler, *, threshold: float) -> dict[str, Any]:
    return evaluate_scores(frame, hold_scores(frame, model, scaler), threshold=threshold)


def evaluate_scores(frame: pd.DataFrame, scores: np.ndarray, *, threshold: float) -> dict[str, Any]:
    if frame.empty or len(scores) == 0:
        return {"rows": 0, "status": "empty"}
    true_hold = frame["target_hold"].to_numpy(dtype=bool)
    pred_hold = scores > float(threshold)
    q_exit = frame["q_exit"].to_numpy(dtype=float)
    q_hold = frame["q_hold"].to_numpy(dtype=float)
    chosen = np.where(pred_hold, q_hold, q_exit)
    oracle = np.maximum(q_hold, q_exit)
    baseline = q_exit
    tp = int((pred_hold & true_hold).sum())
    tn = int((~pred_hold & ~true_hold).sum())
    fp = int((pred_hold & ~true_hold).sum())
    fn = int((~pred_hold & true_hold).sum())
    hold_recall = tp / max(tp + fn, 1)
    exit_recall = tn / max(tn + fp, 1)
    return {
        "rows": int(len(frame)),
        "threshold": float(threshold),
        "accuracy": float((pred_hold == true_hold).mean()),
        "balanced_accuracy": float((hold_recall + exit_recall) / 2.0),
        "hold_recall": float(hold_recall),
        "exit_recall": float(exit_recall),
        "predicted_hold_fraction": float(pred_hold.mean()),
        "true_hold_fraction": float(true_hold.mean()),
        "captured_advantage_vs_exit": float((chosen - baseline).sum()),
        "oracle_advantage_vs_exit": float((oracle - baseline).sum()),
        "capture_fraction": float((chosen - baseline).sum() / max((oracle - baseline).sum(), 1e-9)),
        "mean_chosen_minus_oracle": float((chosen - oracle).mean()),
        "confusion": {"tp_hold": tp, "tn_exit": tn, "fp_bad_hold": fp, "fn_missed_hold": fn},
        "status": "pass",
    }


def summarize_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    out = {}
    for split, group in frame.groupby("split", sort=True):
        out[str(split)] = {"rows": int(len(group)), "hold_fraction": float(group["target_hold"].mean()), "sessions": int(group["session"].nunique())}
    return out


def aggregate(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for split in ["q3_2025", "q4_2025", "q1_2026", "march_2026", "recent_2026"]:
        rows = [result["splits"][split] for result in fold_results if not result.get("skipped") and split in result.get("splits", {}) and result["splits"][split].get("status") == "pass"]
        if not rows:
            out[split] = {"seeds": 0}
            continue
        out[split] = {
            "seeds": len(rows),
            "median_balanced_accuracy": float(np.median([row["balanced_accuracy"] for row in rows])),
            "median_capture_fraction": float(np.median([row["capture_fraction"] for row in rows])),
            "median_captured_advantage_vs_exit": float(np.median([row["captured_advantage_vs_exit"] for row in rows])),
            "median_exit_recall": float(np.median([row["exit_recall"] for row in rows])),
            "median_predicted_hold_fraction": float(np.median([row["predicted_hold_fraction"] for row in rows])),
        }
    return out


def decide(fold_results: list[dict[str, Any]]) -> str:
    aggregate_payload = aggregate(fold_results)
    scored = [item for item in aggregate_payload.values() if item.get("seeds", 0) > 0]
    if not scored:
        return "blocked_position_lifecycle_policy_no_scored_splits"
    if all(item["median_capture_fraction"] > 0.50 and item["median_balanced_accuracy"] > 0.60 for item in scored):
        return "position_lifecycle_policy_has_learned_signal_needs_serial_integration"
    return "research_only_position_lifecycle_policy_signal_insufficient_or_mixed"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        f"Candidate: `{payload['candidate_label']}`",
        f"Paper default baseline: `{payload['paper_default_label']}`",
        "Paid data downloaded: no",
        "Broker endpoint called: no",
        f"Decision: `{payload['decision']}`",
        f"Next experiment: {payload['next_experiment']}",
        "",
        "## Aggregate",
        "",
        "| split | seeds | balanced acc | capture fraction | captured adv vs exit | exit recall | predicted hold frac |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, row in payload["aggregate"].items():
        if row.get("seeds", 0) == 0:
            continue
        lines.append(
            f"| {split} | {row['seeds']} | {row['median_balanced_accuracy']:.3f} | {row['median_capture_fraction']:.3f} | "
            f"{row['median_captured_advantage_vs_exit']:.2f} | {row['median_exit_recall']:.3f} | {row['median_predicted_hold_fraction']:.3f} |"
        )
    lines.extend(["", "This is lifecycle-label learning only. It is not yet strict serial trading PnL versus Protocol101.", "", f"- Summary: `{path.parent / 'summary.json'}`"])
    path.write_text("\n".join(lines) + "\n")


def append_ledger(payload: dict[str, Any], out_dir: Path) -> None:
    ledger = Path("v4/ledger/RESEARCH_LEDGER.md")
    if not ledger.exists():
        return
    marker = f"## {HISTORICAL_ID} - {ROLE_LABEL}"
    if marker in ledger.read_text():
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
                    f"- Candidate: `{payload['candidate_label']}`",
                    "- Paid data downloaded: no",
                    "- Broker endpoint called: no",
                    f"- Decision: `{payload['decision']}`",
                    f"- Report: `{out_dir / 'report.md'}`",
                ]
            )
            + "\n"
        )


if __name__ == "__main__":
    raise SystemExit(main())
