"""EXP_ENTRY_QUALITY_CALIBRATOR_V1.

Historically Protocol249. This is a second-stage neural gate over the frozen
premium-leaning blended utility challenger. It does not replace
PAPER_DEFAULT_PROTOCOL101, download data, or call any broker endpoint.

The hypothesis is narrow: the current challenger finds more opportunity than
Protocol101, but accepts too many low-quality entries. This runner keeps the
challenger's full-action candidate choice fixed, then trains a causal entry
quality calibrator to decide whether that proposed trade is worth occupying the
single account slot.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from v4.model.supervised_pilot import FeatureScaler
from v4.scripts.run_protocol165_full_action_space_policy import (
    DEFAULT_PROTOCOL101_SUMMARY,
    DEFAULT_RECENT_BASELINE,
    FOLDS,
    STARTING_CASH,
    aggregate,
    fmt,
    load_dataset,
    load_protocol101_baselines,
    reported_slices,
    simulation_result,
    smoke_folds,
    trade_from_row,
)
from v4.scripts.run_protocol172_full_action_value_policy import build_events, summarize_events
import v4.scripts.run_protocol183_two_stage_full_action_policy as p183
import v4.scripts.run_protocol221_return_on_premium_policy as p221
import v4.scripts.run_protocol248_challenger_failure_surface as p248


ROLE_LABEL = "EXP_ENTRY_QUALITY_CALIBRATOR_V1"
HISTORICAL_ID = "Protocol249"
DEFAULT_DATASET = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_211_full_action_history_feature_repair/full_action_surface_edge_with_history.parquet"
)
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_249_entry_quality_calibrator")
BASE_ARTIFACT_ROOTS = [
    Path("v4/audit/autoresearch/v4_aplus_hypothesis_238_premium_leaning_blend_seed_stability/model_artifacts"),
    Path("v4/audit/autoresearch/v4_aplus_hypothesis_239_premium_leaning_blend_seed_holdout_4_5/model_artifacts"),
]
MODEL_SEEDS = [1, 2, 3, 4, 5]
FORBIDDEN_FEATURE_SUBSTRINGS = (
    "candidate_pnl",
    "raw_candidate_pnl",
    "realized_pnl",
    "pnl",
    "profit",
    "loss",
    "candidate_exit",
    "exit_time",
    "exit_reason",
    "path_",
    "mfe",
    "mae",
    "future",
    "label",
)


@dataclass(frozen=True)
class EntryQualityConfig:
    epochs: int = 12
    batch_size: int = 512
    hidden_dim: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    target_scale: float = 300.0
    target_clip: float = 1200.0
    min_validation_trades: int = 10


class EntryQualityCalibrator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--protocol101-summary", type=Path, default=DEFAULT_PROTOCOL101_SUMMARY)
    parser.add_argument("--recent-baseline-summary", type=Path, default=DEFAULT_RECENT_BASELINE)
    parser.add_argument("--seeds", nargs="*", type=int, default=MODEL_SEEDS)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--starting-cash", type=float, default=STARTING_CASH)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-smoke-sessions", type=int, default=3)
    parser.add_argument("--skip-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.dataset)
    events = build_events(dataset, starting_cash=float(args.starting_cash))
    if args.smoke:
        events = smoke_events(events, max_sessions=int(args.max_smoke_sessions))
    folds = smoke_folds(events) if args.smoke else FOLDS
    config = EntryQualityConfig(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        hidden_dim=int(args.hidden_dim),
    )

    fold_results: list[dict[str, Any]] = []
    calibrated_trades: list[dict[str, Any]] = []
    base_trades: list[dict[str, Any]] = []
    opportunity_counts: list[dict[str, Any]] = []
    artifact_failures: list[dict[str, Any]] = []
    for fold in folds:
        print(json.dumps({"stage": "fold_start", "fold": fold["name"]}), flush=True)
        fold_events = {
            "train": [event for event in events if event["split"] in set(fold["train_splits"])],
            "validation": [event for event in events if event["split"] == fold["validation_split"]],
        }
        if not fold_events["train"] or not fold_events["validation"]:
            fold_results.append({"fold": fold["name"], "skipped": True, "reason": "missing_train_or_validation_events", "splits": {}})
            continue
        for seed in args.seeds:
            print(json.dumps({"stage": "seed_start", "fold": fold["name"], "seed": int(seed)}), flush=True)
            try:
                base = load_base_artifact(fold["name"], int(seed), dataset)
            except FileNotFoundError as exc:
                artifact_failures.append({"fold": fold["name"], "seed": int(seed), "reason": str(exc)})
                continue

            train_pred = predict_base_actions(fold_events["train"], base)
            validation_pred = predict_base_actions(fold_events["validation"], base)
            train_opportunities = collect_base_opportunities(fold_events["train"], train_pred, base)
            validation_opportunities = collect_base_opportunities(fold_events["validation"], validation_pred, base)
            opportunity_counts.append(
                {
                    "fold": fold["name"],
                    "seed": int(seed),
                    "train_opportunities": int(len(train_opportunities)),
                    "validation_opportunities": int(len(validation_opportunities)),
                }
            )
            print(
                json.dumps(
                    {
                        "stage": "base_opportunities",
                        "fold": fold["name"],
                        "seed": int(seed),
                        "train": int(len(train_opportunities)),
                        "validation": int(len(validation_opportunities)),
                    }
                ),
                flush=True,
            )
            if len(train_opportunities) < 20 or len(validation_opportunities) < 5:
                fold_results.append(
                    {
                        "fold": fold["name"],
                        "seed": int(seed),
                        "skipped": True,
                        "reason": "insufficient_base_opportunities_for_calibration",
                        "splits": {},
                    }
                )
                continue

            feature_columns = calibrator_feature_columns(base.feature_columns)
            assert_no_leakage_features(feature_columns)
            model, scaler, history = train_calibrator(
                train_opportunities,
                validation_opportunities,
                seed=int(seed),
                feature_columns=feature_columns,
                config=config,
            )
            threshold = select_gate_threshold(
                fold_events["validation"],
                validation_pred,
                base,
                model,
                scaler,
                feature_columns=feature_columns,
                config=config,
                starting_cash=float(args.starting_cash),
            )

            model_dir = args.out_dir / "model_artifacts" / fold["name"] / f"seed_{seed}"
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_dir / "model.pt")
            (model_dir / "scaler.json").write_text(json.dumps(scaler.to_dict(), indent=2, sort_keys=True) + "\n")
            (model_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "role_label": ROLE_LABEL,
                        "historical_protocol": HISTORICAL_ID,
                        "fold": fold["name"],
                        "seed": int(seed),
                        "base_candidate": "CHALLENGER_PREMIUM_LEANING_BLENDED_UTILITY_V1",
                        "base_artifact_dir": str(base.artifact_dir),
                        "base_threshold": float(base.threshold),
                        "feature_columns": feature_columns,
                        "config": asdict(config),
                        "threshold_selection": threshold,
                        "history": history,
                    },
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
                + "\n"
            )

            result = {"fold": fold["name"], "seed": int(seed), "threshold": float(threshold["threshold"]), "splits": {}}
            for split_name, split_events in reported_slices(events, fold).items():
                split_pred = predict_base_actions(split_events, base)
                split_gate_scores = precompute_gate_scores(split_events, split_pred, base, model, scaler, feature_columns)
                calibrated = simulate_calibrated(
                    split_events,
                    split_pred,
                    base,
                    model,
                    scaler,
                    gate_threshold=float(threshold["threshold"]),
                    feature_columns=feature_columns,
                    gate_scores_by_index=split_gate_scores,
                    slippage_per_side=0.0,
                    starting_cash=float(args.starting_cash),
                    strategy="entry_quality_calibrator",
                )
                calibrated_10 = simulate_calibrated(
                    split_events,
                    split_pred,
                    base,
                    model,
                    scaler,
                    gate_threshold=float(threshold["threshold"]),
                    feature_columns=feature_columns,
                    gate_scores_by_index=split_gate_scores,
                    slippage_per_side=0.10,
                    starting_cash=float(args.starting_cash),
                    strategy="entry_quality_calibrator_stress10",
                )
                calibrated_25 = simulate_calibrated(
                    split_events,
                    split_pred,
                    base,
                    model,
                    scaler,
                    gate_threshold=float(threshold["threshold"]),
                    feature_columns=feature_columns,
                    gate_scores_by_index=split_gate_scores,
                    slippage_per_side=0.25,
                    starting_cash=float(args.starting_cash),
                    strategy="entry_quality_calibrator_stress25",
                )
                base_result = simulate_base_from_predictions(
                    split_events,
                    split_pred,
                    base,
                    slippage_per_side=0.0,
                    starting_cash=float(args.starting_cash),
                    strategy="premium_blend_base",
                )
                result["splits"][split_name] = {
                    "model": add_quality_metrics(calibrated.summary, calibrated.trades),
                    "model_stress_0_10": add_quality_metrics(calibrated_10.summary, calibrated_10.trades),
                    "model_stress_0_25": add_quality_metrics(calibrated_25.summary, calibrated_25.trades),
                    "base_challenger": add_quality_metrics(base_result.summary, base_result.trades),
                    "quality_delta_vs_base": quality_delta(calibrated.summary, base_result.summary),
                }
                calibrated_trades.extend(
                    {**trade, "fold": fold["name"], "seed": int(seed), "reported_split": split_name}
                    for trade in calibrated.trades
                )
                base_trades.extend(
                    {**trade, "fold": fold["name"], "seed": int(seed), "reported_split": split_name}
                    for trade in base_result.trades
                )
            fold_results.append(result)
            print(json.dumps({"stage": "seed_done", "fold": fold["name"], "seed": int(seed), "threshold": float(threshold["threshold"])}), flush=True)

    frozen_protocol101 = load_protocol101_baselines(args.protocol101_summary, args.recent_baseline_summary)
    calibrated_frame = pd.DataFrame(calibrated_trades)
    base_frame = pd.DataFrame(base_trades)
    if not calibrated_frame.empty:
        calibrated_frame.to_csv(args.out_dir / "model_trades.csv", index=False)
    if not base_frame.empty:
        base_frame.to_csv(args.out_dir / "base_challenger_trades.csv", index=False)
    aggregate_payload = aggregate(fold_results, frozen_protocol101)
    aggregate_payload = add_required_split_and_seed_checks(aggregate_payload, required_seed_count=len(MODEL_SEEDS))
    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "experiment / model change",
        "changes_paper_default": False,
        "candidate_label": "CHALLENGER_ENTRY_QUALITY_CALIBRATED_PREMIUM_BLEND_V1",
        "paper_default_label": "PAPER_DEFAULT_PROTOCOL101",
        "baseline_challenger_label": "CHALLENGER_PREMIUM_LEANING_BLENDED_UTILITY_V1",
        "baseline_to_beat": "PAPER_DEFAULT_PROTOCOL101 strict one-account serial replay",
        "data_used": str(args.dataset),
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": True,
        "event_summary": summarize_events(events),
        "fold_results": fold_results,
        "aggregate": aggregate_payload,
        "frozen_protocol101_baselines": frozen_protocol101,
        "opportunity_counts": opportunity_counts,
        "artifact_failures": artifact_failures,
        "trade_profile": {
            "calibrated": trade_profile(calibrated_frame),
            "base_challenger": trade_profile(base_frame),
        },
        "quality_summary": quality_summary(fold_results),
        "decision": "",
        "next_experiment": (
            "If the calibrator improves quality but fails the Protocol101 gate, move to "
            "EXP_CONTRACT_VALUE_SELECTION_HEAD_V1 or lifecycle continuation depending on the attribution failure."
        ),
    }
    payload["decision"] = decide(payload)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    if not args.skip_ledger:
        append_ledger(payload, args.out_dir)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


@dataclass
class BaseArtifact:
    artifact_dir: Path
    feature_columns: list[str]
    threshold: float
    model: p183.TwoStageFullActionPolicy
    scaler: FeatureScaler


def load_base_artifact(fold_name: str, seed: int, dataset: pd.DataFrame) -> BaseArtifact:
    for root in BASE_ARTIFACT_ROOTS:
        artifact_dir = root / fold_name / f"seed_{seed}"
        manifest_path = artifact_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            feature_columns = [str(column) for column in manifest["feature_columns"]]
            p183.set_active_feature_columns(feature_columns, dataset)
            scaler = scaler_from_json(artifact_dir / "scaler.json")
            hidden_dim = int(manifest.get("config", {}).get("hidden_dim", 96))
            model = p183.TwoStageFullActionPolicy(input_dim=len(feature_columns), hidden_dim=hidden_dim)
            model.load_state_dict(torch.load(artifact_dir / "model.pt", map_location="cpu"))
            model.eval()
            return BaseArtifact(
                artifact_dir=artifact_dir,
                feature_columns=feature_columns,
                threshold=float(manifest.get("threshold_selection", {}).get("threshold", 0.0)),
                model=model,
                scaler=scaler,
            )
    raise FileNotFoundError(f"missing base artifact for fold={fold_name} seed={seed}")


def scaler_from_json(path: Path) -> FeatureScaler:
    payload = json.loads(path.read_text())
    return FeatureScaler(
        fill=np.asarray(payload["fill"], dtype=np.float32),
        mean=np.asarray(payload["mean"], dtype=np.float32),
        std=np.asarray(payload["std"], dtype=np.float32),
    )


def predict_base_actions(events: list[dict[str, Any]], base: BaseArtifact, *, batch_size: int = 4096) -> list[dict[str, Any]]:
    p183.set_active_feature_columns(base.feature_columns)
    predictions: list[dict[str, Any]] = []
    base.model.eval()
    with torch.no_grad():
        for start in range(0, len(events), batch_size):
            batch = events[start : start + batch_size]
            x, mask, _, _, _, _ = p183.tensors(batch, base.scaler, p183.TwoStageFullActionConfig())
            event_logit, candidate_scores = base.model(torch.from_numpy(x), torch.from_numpy(mask))
            score_np = event_logit.cpu().numpy()
            action_np = candidate_scores.cpu().numpy().argmax(axis=1) + 1
            mask_np = mask
            for local, event in enumerate(batch):
                action = int(action_np[local])
                valid_action = bool(0 < action <= len(event["candidates"]) and mask_np[local, action - 1])
                predictions.append({"action": action if valid_action else 0, "score": float(score_np[local]), "valid_action": valid_action})
    return predictions


def collect_base_opportunities(
    events: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    base: BaseArtifact,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event, prediction in zip(events, predictions):
        if not prediction["valid_action"] or float(prediction["score"]) < base.threshold:
            continue
        action = int(prediction["action"])
        if action <= 0 or action > len(event["candidates"]):
            continue
        candidate = event["candidates"].iloc[action - 1]
        record = opportunity_record(event, candidate, prediction, base)
        if record is not None:
            rows.append(record)
    return pd.DataFrame(rows)


def opportunity_record(
    event: dict[str, Any],
    row: pd.Series,
    prediction: dict[str, Any],
    base: BaseArtifact,
) -> dict[str, Any] | None:
    entry_ask = finite(row.get("entry_ask"))
    pnl = finite(row.get("candidate_pnl"))
    if entry_ask <= 0.0 or not math.isfinite(pnl):
        return None
    record: dict[str, Any] = {
        "split": str(event["split"]),
        "session": str(event["session"]),
        "decision_dt": pd.Timestamp(event["decision_dt"]),
        "candidate_exit_dt": pd.Timestamp(row["candidate_exit_dt"]),
        "candidate_uid": str(row.get("candidate_uid", "")),
        "trade_uid": str(row.get("trade_uid", "")),
        "contract_id": str(row.get("contract_id", "")),
        "right": str(row.get("right", "")),
        "offset": finite(row.get("offset")),
        "base_score": float(prediction["score"]),
        "base_threshold": float(base.threshold),
        "base_score_margin": float(prediction["score"]) - float(base.threshold),
        "entry_ask": float(entry_ask),
        "entry_premium": float(entry_ask * 100.0),
        "candidate_pnl": float(pnl),
    }
    for column in base.feature_columns:
        record[column] = finite(row.get(column), 0.0)
    record.update(moneyness_flags(record["right"], record["offset"]))
    return record


def calibrator_feature_columns(base_feature_columns: list[str]) -> list[str]:
    extras = [
        "base_score",
        "base_score_margin",
        "entry_premium",
        "is_call",
        "is_put",
        "is_itm",
        "is_atm",
        "is_otm",
    ]
    return list(dict.fromkeys([*extras, *base_feature_columns]))


def assert_no_leakage_features(feature_columns: Iterable[str]) -> None:
    offenders = [
        column
        for column in feature_columns
        if any(token in str(column).lower() for token in FORBIDDEN_FEATURE_SUBSTRINGS)
        and str(column) not in {"entry_premium"}
    ]
    if offenders:
        raise ValueError(f"calibrator feature leakage risk: {offenders}")


def moneyness_flags(right: Any, offset: Any) -> dict[str, float]:
    bucket = p248.moneyness(right, offset)
    side = str(right)
    return {
        "is_call": 1.0 if side == "C" else 0.0,
        "is_put": 1.0 if side == "P" else 0.0,
        "is_itm": 1.0 if bucket == "ITM" else 0.0,
        "is_atm": 1.0 if bucket == "ATM" else 0.0,
        "is_otm": 1.0 if bucket == "OTM" else 0.0,
    }


def train_calibrator(
    train_opportunities: pd.DataFrame,
    validation_opportunities: pd.DataFrame,
    *,
    seed: int,
    feature_columns: list[str],
    config: EntryQualityConfig,
) -> tuple[EntryQualityCalibrator, FeatureScaler, list[dict[str, Any]]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    x_train_raw, y_train, w_train = opportunity_tensors(train_opportunities, feature_columns, config, scaler=None)
    scaler = FeatureScaler.fit(x_train_raw)
    x_train, y_train, w_train = opportunity_tensors(train_opportunities, feature_columns, config, scaler=scaler)
    x_val, y_val, w_val = opportunity_tensors(validation_opportunities, feature_columns, config, scaler=scaler)
    model = EntryQualityCalibrator(input_dim=len(feature_columns), hidden_dim=config.hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(w_train)),
        batch_size=config.batch_size,
        shuffle=True,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for bx, by, bw in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(bx)
            loss = nn.functional.huber_loss(pred, by, delta=1.0, reduction="none")
            loss = (loss * bw).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            pred_val = model(torch.from_numpy(x_val))
            val_loss = nn.functional.huber_loss(pred_val, torch.from_numpy(y_val), delta=1.0, reduction="none")
            val_loss = float((val_loss * torch.from_numpy(w_val)).mean().detach().cpu())
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": val_loss, "is_best": val_loss <= best_val})
    model.load_state_dict(best_state)
    return model, scaler, history


def opportunity_tensors(
    opportunities: pd.DataFrame,
    feature_columns: list[str],
    config: EntryQualityConfig,
    *,
    scaler: FeatureScaler | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = opportunities[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    y = np.clip(pd.to_numeric(opportunities["candidate_pnl"], errors="coerce").to_numpy(dtype=np.float32), -config.target_clip, config.target_clip)
    y = (y / float(config.target_scale)).astype(np.float32)
    weights = (1.0 + np.minimum(np.abs(y) * float(config.target_scale) / 300.0, 5.0)).astype(np.float32)
    if scaler is not None:
        x = scaler.transform(x)
    return x.astype(np.float32), y, weights


def calibrator_scores(
    opportunities: pd.DataFrame,
    model: EntryQualityCalibrator,
    scaler: FeatureScaler,
    feature_columns: list[str],
    *,
    batch_size: int = 4096,
) -> np.ndarray:
    if opportunities.empty:
        return np.asarray([], dtype=float)
    raw = opportunities[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    x = scaler.transform(raw)
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            out.append(model(torch.from_numpy(x[start : start + batch_size])).cpu().numpy())
    return np.concatenate(out).astype(float)


def select_gate_threshold(
    events: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    base: BaseArtifact,
    model: EntryQualityCalibrator,
    scaler: FeatureScaler,
    *,
    feature_columns: list[str],
    config: EntryQualityConfig,
    starting_cash: float,
) -> dict[str, Any]:
    opportunities = collect_base_opportunities(events, predictions, base)
    scores = calibrator_scores(opportunities, model, scaler, feature_columns)
    finite_scores = scores[np.isfinite(scores)]
    thresholds = [float("-inf"), float("inf")]
    if len(finite_scores):
        thresholds.extend(np.quantile(finite_scores, [0, .1, .2, .35, .5, .65, .8, .9, .95]).round(4).tolist())
        thresholds.append(float(finite_scores.min()) - 1e-3)
    thresholds = sorted(set(float(item) for item in thresholds))
    gate_scores_by_index = precompute_gate_scores(events, predictions, base, model, scaler, feature_columns)
    base_result = simulate_base_from_predictions(
        events,
        predictions,
        base,
        slippage_per_side=0.0,
        starting_cash=starting_cash,
        strategy="premium_blend_base_validation",
    )
    sweep = []
    for threshold in thresholds:
        base_sim = simulate_calibrated(
            events,
            predictions,
            base,
            model,
            scaler,
            gate_threshold=float(threshold),
            feature_columns=feature_columns,
            gate_scores_by_index=gate_scores_by_index,
            slippage_per_side=0.0,
            starting_cash=starting_cash,
            strategy="entry_quality_validation",
        )
        stress = simulate_calibrated(
            events,
            predictions,
            base,
            model,
            scaler,
            gate_threshold=float(threshold),
            feature_columns=feature_columns,
            gate_scores_by_index=gate_scores_by_index,
            slippage_per_side=0.10,
            starting_cash=starting_cash,
            strategy="entry_quality_validation_stress10",
        )
        sweep.append(
            {
                "threshold": float(threshold),
                "model": add_quality_metrics(base_sim.summary, base_sim.trades),
                "model_stress_0_10": add_quality_metrics(stress.summary, stress.trades),
                "base_challenger": add_quality_metrics(base_result.summary, base_result.trades),
                "quality_delta_vs_base": quality_delta(base_sim.summary, base_result.summary),
            }
        )
    eligible = [
        row
        for row in sweep
        if row["model"]["trades"] >= config.min_validation_trades
        and row["model_stress_0_10"]["total_pnl"] > 0.0
    ]
    pool = eligible if eligible else sweep
    best = max(
        pool,
        key=lambda row: (
            row["model_stress_0_10"]["total_pnl"],
            row["model"]["profit_factor"],
            row["model"]["avg_pnl"],
            -abs(float(row["model"].get("max_drawdown", 0.0))),
            row["model"]["win_rate"],
        ),
    )
    return {
        "threshold": float(best["threshold"]),
        "objective": "validation-only: stress_0_10 PnL, then PF, avg PnL, drawdown, win rate",
        "selected": best,
        "sweep": sweep,
    }


def simulate_base_from_predictions(
    events: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    base: BaseArtifact,
    *,
    slippage_per_side: float,
    starting_cash: float,
    strategy: str,
) -> Any:
    trades: list[dict[str, Any]] = []
    equity = float(starting_cash)
    open_until: dict[str, pd.Timestamp] = {}
    skipped = {"overlap": 0, "threshold": 0, "unaffordable": 0, "invalid": 0, "gate": 0}
    ordered = sorted(zip(events, predictions), key=lambda item: (item[0]["session"], item[0]["decision_dt"]))
    for event, prediction in ordered:
        if open_until.get(event["session"]) is not None and event["decision_dt"] < open_until[event["session"]]:
            skipped["overlap"] += len(event["candidates"])
            continue
        if not prediction["valid_action"] or float(prediction["score"]) < base.threshold:
            skipped["threshold"] += len(event["candidates"])
            continue
        action = int(prediction["action"])
        trade = trade_from_row(
            event["candidates"].iloc[action - 1],
            score=float(prediction["score"]),
            threshold=float(base.threshold),
            slippage_per_side=slippage_per_side,
            equity=equity,
            strategy=strategy,
        )
        if trade is None:
            skipped["invalid"] += 1
            continue
        trade["base_score"] = float(prediction["score"])
        trade["base_threshold"] = float(base.threshold)
        if trade["entry_premium_with_slippage"] > equity:
            skipped["unaffordable"] += 1
            continue
        trades.append(trade)
        equity += trade["pnl"]
        trades[-1]["account_equity_after"] = equity
        open_until[event["session"]] = pd.Timestamp(trade["exit_time"])
    return simulation_result(trades, events, skipped, starting_cash=starting_cash, strategy=strategy)


def simulate_calibrated(
    events: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    base: BaseArtifact,
    model: EntryQualityCalibrator,
    scaler: FeatureScaler,
    *,
    gate_threshold: float,
    feature_columns: list[str],
    gate_scores_by_index: dict[int, float] | None = None,
    slippage_per_side: float,
    starting_cash: float,
    strategy: str,
) -> Any:
    trades: list[dict[str, Any]] = []
    equity = float(starting_cash)
    open_until: dict[str, pd.Timestamp] = {}
    skipped = {"overlap": 0, "threshold": 0, "unaffordable": 0, "invalid": 0, "gate": 0}
    if gate_scores_by_index is None:
        gate_scores_by_index = precompute_gate_scores(events, predictions, base, model, scaler, feature_columns)
    ordered = sorted(enumerate(zip(events, predictions)), key=lambda item: (item[1][0]["session"], item[1][0]["decision_dt"]))
    for event_idx, (event, prediction) in ordered:
        if open_until.get(event["session"]) is not None and event["decision_dt"] < open_until[event["session"]]:
            skipped["overlap"] += len(event["candidates"])
            continue
        if not prediction["valid_action"] or float(prediction["score"]) < base.threshold:
            skipped["threshold"] += len(event["candidates"])
            continue
        action = int(prediction["action"])
        candidate = event["candidates"].iloc[action - 1]
        if event_idx not in gate_scores_by_index:
            skipped["invalid"] += 1
            continue
        gate_score = float(gate_scores_by_index[event_idx])
        if gate_score < gate_threshold:
            skipped["gate"] += 1
            continue
        trade = trade_from_row(
            candidate,
            score=gate_score,
            threshold=gate_threshold,
            slippage_per_side=slippage_per_side,
            equity=equity,
            strategy=strategy,
        )
        if trade is None:
            skipped["invalid"] += 1
            continue
        trade["base_score"] = float(prediction["score"])
        trade["base_threshold"] = float(base.threshold)
        trade["gate_score"] = gate_score
        trade["gate_threshold"] = float(gate_threshold)
        if trade["entry_premium_with_slippage"] > equity:
            skipped["unaffordable"] += 1
            continue
        trades.append(trade)
        equity += trade["pnl"]
        trades[-1]["account_equity_after"] = equity
        open_until[event["session"]] = pd.Timestamp(trade["exit_time"])
    return simulation_result(trades, events, skipped, starting_cash=starting_cash, strategy=strategy)


def precompute_gate_scores(
    events: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    base: BaseArtifact,
    model: EntryQualityCalibrator,
    scaler: FeatureScaler,
    feature_columns: list[str],
) -> dict[int, float]:
    records: list[dict[str, Any]] = []
    indices: list[int] = []
    for idx, (event, prediction) in enumerate(zip(events, predictions)):
        if not prediction["valid_action"] or float(prediction["score"]) < base.threshold:
            continue
        action = int(prediction["action"])
        if action <= 0 or action > len(event["candidates"]):
            continue
        record = opportunity_record(event, event["candidates"].iloc[action - 1], prediction, base)
        if record is None:
            continue
        records.append(record)
        indices.append(idx)
    if not records:
        return {}
    scores = calibrator_scores(pd.DataFrame(records), model, scaler, feature_columns)
    return {int(idx): float(score) for idx, score in zip(indices, scores)}


def add_quality_metrics(summary: dict[str, Any], trades: list[dict[str, Any]]) -> dict[str, Any]:
    out = dict(summary)
    out["pnl_per_premium"] = (
        float(sum(float(trade["pnl"]) for trade in trades) / sum(float(trade["entry_premium"]) for trade in trades))
        if trades and sum(float(trade["entry_premium"]) for trade in trades) > 0.0
        else 0.0
    )
    out["median_entry_premium"] = float(np.median([float(trade["entry_premium"]) for trade in trades])) if trades else 0.0
    out["gross_premium"] = float(sum(float(trade["entry_premium"]) for trade in trades))
    return out


def quality_delta(model: dict[str, Any], base: dict[str, Any]) -> dict[str, float]:
    return {
        "total_pnl": finite(model.get("total_pnl"), 0.0) - finite(base.get("total_pnl"), 0.0),
        "win_rate": finite(model.get("win_rate"), 0.0) - finite(base.get("win_rate"), 0.0),
        "profit_factor": finite(model.get("profit_factor"), 0.0) - finite(base.get("profit_factor"), 0.0),
        "avg_pnl": finite(model.get("avg_pnl"), 0.0) - finite(base.get("avg_pnl"), 0.0),
        "max_drawdown": finite(model.get("max_drawdown"), 0.0) - finite(base.get("max_drawdown"), 0.0),
        "trades": finite(model.get("trades"), 0.0) - finite(base.get("trades"), 0.0),
    }


def quality_summary(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ["q3_2025", "q4_2025", "q1_2026", "march_2026", "recent_2026"]:
        rows = []
        for result in fold_results:
            if result.get("skipped") or split not in result.get("splits", {}):
                continue
            rows.append(result["splits"][split])
        if not rows:
            continue
        dimensions = []
        for metric in ["win_rate", "profit_factor", "avg_pnl", "max_drawdown"]:
            deltas = np.asarray([row["quality_delta_vs_base"].get(metric, 0.0) for row in rows], dtype=float)
            dimensions.append({"metric": metric, "median_delta_vs_base": float(np.median(deltas)), "improved": bool(np.median(deltas) > 0.0)})
        out[split] = {
            "quality_dimensions": dimensions,
            "improved_dimension_count": int(sum(1 for item in dimensions if item["improved"])),
            "median_delta_pnl_vs_base": float(np.median([row["quality_delta_vs_base"].get("total_pnl", 0.0) for row in rows])),
        }
    return out


def add_required_split_and_seed_checks(aggregate_payload: dict[str, Any], *, required_seed_count: int) -> dict[str, Any]:
    checks = list(aggregate_payload.get("promotion_checks", []))
    for split in ["q3_2025", "q4_2025", "q1_2026", "march_2026", "recent_2026"]:
        item = aggregate_payload.get(split, {})
        checks.append(
            {
                "split": split,
                "name": "required_seed_count",
                "pass": int(item.get("seeds", 0)) >= int(required_seed_count),
                "value": int(item.get("seeds", 0)),
                "required": int(required_seed_count),
            }
        )
    q3 = aggregate_payload.get("q3_2025", {})
    if q3.get("seeds", 0) == 0:
        checks.append({"split": "q3_2025", "name": "split_available", "pass": False, "value": 0})
    else:
        checks.extend(
            [
                {"split": "q3_2025", "name": "positive_median_pnl", "pass": q3["median_total_pnl"] > 0, "value": q3["median_total_pnl"]},
                {"split": "q3_2025", "name": "positive_seed_fraction_ge_0_80", "pass": q3["positive_seed_fraction"] >= 0.80, "value": q3["positive_seed_fraction"]},
                {"split": "q3_2025", "name": "median_pf_ge_1_15", "pass": q3["median_profit_factor"] >= 1.15, "value": q3["median_profit_factor"]},
                {"split": "q3_2025", "name": "stress_0_10_positive", "pass": q3["median_stress_0_10_total_pnl"] > 0, "value": q3["median_stress_0_10_total_pnl"]},
                {"split": "q3_2025", "name": "beats_frozen_protocol101", "pass": bool(q3["beats_frozen_protocol101"]), "value": q3["median_delta_vs_frozen_protocol101"]},
            ]
        )
    aggregate_payload["promotion_checks"] = checks
    aggregate_payload["promotion_ready"] = bool(checks and all(check["pass"] for check in checks))
    return aggregate_payload


def trade_profile(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"rows": 0}
    out = {
        "rows": int(len(frame)),
        "total_pnl": float(pd.to_numeric(frame["pnl"], errors="coerce").fillna(0.0).sum()),
        "win_rate": float((pd.to_numeric(frame["pnl"], errors="coerce").fillna(0.0) > 0.0).mean()),
        "median_entry_premium": float(pd.to_numeric(frame["entry_premium"], errors="coerce").median()),
        "by_side": [],
        "by_moneyness": [],
    }
    tmp = frame.copy()
    tmp["moneyness"] = [p248.moneyness(right, offset) for right, offset in zip(tmp["right"], tmp["offset"])]
    for group_cols, key in [(["right"], "by_side"), (["moneyness"], "by_moneyness")]:
        rows = []
        for group_key, group in tmp.groupby(group_cols, sort=True, observed=False):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            pnl = pd.to_numeric(group["pnl"], errors="coerce").fillna(0.0)
            rows.append(
                {
                    group_cols[0]: str(group_key[0]),
                    "trades": int(len(group)),
                    "pnl": float(pnl.sum()),
                    "win_rate": float((pnl > 0.0).mean()),
                    "median_entry_premium": float(pd.to_numeric(group["entry_premium"], errors="coerce").median()),
                }
            )
        out[key] = rows
    return out


def decide(payload: dict[str, Any]) -> str:
    aggregate_payload = payload.get("aggregate", {})
    if aggregate_payload.get("promotion_ready"):
        quality = payload.get("quality_summary", {})
        required = ["q4_2025", "q1_2026", "march_2026", "recent_2026"]
        if all(quality.get(split, {}).get("improved_dimension_count", 0) >= 2 for split in required if split in quality):
            return "research_candidate_improves_entry_quality_and_clears_protocol101_gate"
    if any(item.get("median_delta_pnl_vs_base", 0.0) > 0.0 for item in payload.get("quality_summary", {}).values()):
        return "research_only_entry_quality_calibrator_partial_improvement_not_replacement"
    return "rejected_entry_quality_calibrator_no_clear_improvement"


def smoke_events(events: list[dict[str, Any]], *, max_sessions: int) -> list[dict[str, Any]]:
    sessions = sorted({event["session"] for event in events})[:max_sessions]
    return [event for event in events if event["session"] in set(sessions)]


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        f"Candidate being tested: {payload['candidate_label']}",
        f"Paper default baseline: {payload['paper_default_label']}",
        f"Baseline challenger: {payload['baseline_challenger_label']}",
        f"Data used: `{payload['data_used']}`",
        f"Paid data downloaded: {payload['paid_data_downloaded_by_runner']}",
        f"Broker endpoint called: {payload['broker_endpoint_called']}",
        f"Historical ID: `{payload['historical_protocol']}`",
        f"Decision: `{payload['decision']}`",
        f"Next experiment: {payload['next_experiment']}",
        "",
        "## Aggregate Vs Protocol101",
        "",
        "| split | seeds | median PnL | Protocol101 | delta | PF | stress 0.10 | stress 0.25 | trades |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ["q3_2025", "q4_2025", "q1_2026", "march_2026", "recent_2026"]:
        item = payload["aggregate"].get(split, {})
        if not item or item.get("seeds", 0) == 0:
            continue
        lines.append(
            f"| {split} | {item['seeds']} | {fmt(item['median_total_pnl'])} | "
            f"{fmt(item.get('frozen_protocol101_total_pnl'))} | {fmt(item.get('median_delta_vs_frozen_protocol101'))} | "
            f"{fmt(item['median_profit_factor'])} | {fmt(item['median_stress_0_10_total_pnl'])} | "
            f"{fmt(item['median_stress_0_25_total_pnl'])} | {fmt(item['median_trades'])} |"
        )
    lines.extend(["", "## Quality Delta Vs Current Challenger", ""])
    for split, item in payload.get("quality_summary", {}).items():
        lines.append(f"- {split}: {item['improved_dimension_count']} improved dimensions, median PnL delta vs base {fmt(item['median_delta_pnl_vs_base'])}")
    lines.extend(["", "## Trade Profile", ""])
    for label, profile in payload["trade_profile"].items():
        lines.append(f"- {label}: {profile.get('rows', 0)} rows, PnL {fmt(profile.get('total_pnl'))}, win rate {finite(profile.get('win_rate'), 0.0):.3f}, median premium {fmt(profile.get('median_entry_premium'))}")
    lines.extend(["", "## Outputs", "", f"- Summary: `{path.parent / 'summary.json'}`", f"- Model trades: `{path.parent / 'model_trades.csv'}`", f"- Base challenger trades: `{path.parent / 'base_challenger_trades.csv'}`"])
    path.write_text("\n".join(lines) + "\n")


def append_ledger(payload: dict[str, Any], out_dir: Path) -> None:
    ledger = Path("v4/ledger/RESEARCH_LEDGER.md")
    if not ledger.exists():
        return
    lines = [
        "",
        f"## {HISTORICAL_ID} - {ROLE_LABEL}",
        "",
        f"- What is this: {payload['what_is_this']}",
        "- Changes paper default: no",
        f"- Candidate: {payload['candidate_label']}",
        f"- Baseline: {payload['paper_default_label']}",
        f"- Data used: `{payload['data_used']}`",
        "- Paid data downloaded: false",
        "- Broker endpoint called: false",
        f"- Decision: `{payload['decision']}`",
        f"- Report: `{out_dir / 'report.md'}`",
    ]
    with ledger.open("a") as handle:
        handle.write("\n".join(lines) + "\n")


def finite(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


if __name__ == "__main__":
    raise SystemExit(main())
