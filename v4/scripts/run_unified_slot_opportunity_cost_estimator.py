"""Train the causal slot opportunity-cost estimator.

This is not a trading policy. It estimates the Protocol101 opportunity cost a
challenger may block when it consumes the single serial slot.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v4.model.unified_policy_trajectory import FLAT_TRAJECTORY_FEATURE_COLUMNS
from v4.model.unified_slot_opportunity_cost_estimator import (
    ROLE_LABEL,
    TRAINING_SPEC_LABEL,
    SlotOpportunityCostEstimatorConfig,
    build_slot_cost_targets,
    coerce_feature_matrix,
    estimator_predictions,
    sample_training_rows,
    validate_slot_cost_feature_columns,
)


DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/unified_slot_opportunity_cost_estimator")
DEFAULT_DOC_PATH = Path("v4/docs/UNIFIED_SLOT_OPPORTUNITY_COST_ESTIMATOR.md")
DEFAULT_FLAT_DATASET = Path("v4/audit/autoresearch/v4_aplus_hypothesis_270_full_surface_action_advantage_dataset/full_surface_action_advantage.parquet")
DEFAULT_LABELS = Path("v4/audit/autoresearch/unified_slot_opportunity_cost_label_dataset/slot_opportunity_cost_labels.parquet")
DEFAULT_LABEL_SUMMARY = Path("v4/audit/autoresearch/unified_slot_opportunity_cost_label_dataset/summary.json")
DEFAULT_SESSION_MANIFEST = Path("v4/audit/autoresearch/unified_serial_dp_oracle/serial_dp_session_manifest.csv")
KEY_COLUMNS = ["split", "session", "decision_dt", "candidate_uid"]
LABEL_COLUMNS = [
    *KEY_COLUMNS,
    "blocked_protocol101_entries",
    "blocked_protocol101_pnl_0_00",
    "blocked_protocol101_pnl_0_10",
    "blocked_protocol101_pnl_0_25",
    "has_blocked_protocol101_entry",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flat-dataset", type=Path, default=DEFAULT_FLAT_DATASET)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--label-summary", type=Path, default=DEFAULT_LABEL_SUMMARY)
    parser.add_argument("--session-manifest", type=Path, default=DEFAULT_SESSION_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC_PATH)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-train-rows", type=int, default=350_000)
    parser.add_argument("--max-eval-rows-per-split", type=int, default=120_000)
    parser.add_argument("--train-positive-fraction", type=float, default=0.40)
    parser.add_argument("--positive-sample-weight", type=float, default=3.0)
    parser.add_argument("--skip-doc", action="store_true")
    parser.add_argument("--skip-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = SlotOpportunityCostEstimatorConfig(
        max_train_rows=int(args.max_train_rows),
        max_eval_rows_per_split=int(args.max_eval_rows_per_split),
        train_positive_fraction=float(args.train_positive_fraction),
        positive_sample_weight=float(args.positive_sample_weight),
        random_seed=int(args.seed),
    )
    feature_columns = existing_columns(args.flat_dataset, FLAT_TRAJECTORY_FEATURE_COLUMNS)
    validate_slot_cost_feature_columns(feature_columns)
    sessions = included_session_keys(pd.read_csv(args.session_manifest))
    labels = load_labels(args.labels, sessions)
    frame = attach_features(labels, args.flat_dataset, feature_columns, sessions)
    targets = build_slot_cost_targets(frame, config)
    frame = pd.concat([frame.reset_index(drop=True), targets.reset_index(drop=True)], axis=1)

    train_frame = frame[frame["split"].astype(str).isin(config.train_splits)].copy()
    train_frame = sample_training_rows(
        train_frame,
        positive_column="target_positive_cost",
        limit=int(config.max_train_rows),
        positive_fraction=float(config.train_positive_fraction),
        seed=int(config.random_seed),
    )
    eval_frames = split_eval_frames(frame, config)

    bundle = train_estimator(train_frame, feature_columns, config)
    metrics = {
        "train_sample": evaluate_estimator(train_frame, bundle, feature_columns),
        **{
            split: evaluate_estimator(split_frame, bundle, feature_columns)
            for split, split_frame in eval_frames.items()
        },
    }
    validation_metrics = metrics.get(config.validation_split, {})
    global_p90 = float(validation_metrics.get("cost_p90_abs_error", metrics["train_sample"].get("cost_p90_abs_error", 0.0)))
    bundle["global_p90_abs_error"] = global_p90

    artifact_dir = args.out_dir / "model_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "slot_opportunity_cost_estimator.joblib"
    joblib.dump(bundle, model_path)
    manifest = {
        "role_label": ROLE_LABEL,
        "training_spec": TRAINING_SPEC_LABEL,
        "feature_columns": feature_columns,
        "config": config.to_dict(),
        "global_p90_abs_error": global_p90,
    }
    write_json(artifact_dir / "manifest.json", manifest)

    payload = {
        "role_label": ROLE_LABEL,
        "training_spec": TRAINING_SPEC_LABEL,
        "what_is_this": "foundation estimator / causal Protocol101 slot opportunity-cost model",
        "changes_paper_default": False,
        "paper_default_baseline": "PAPER_DEFAULT_PROTOCOL101",
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": True,
        "decision": decide(metrics, train_frame),
        "challenge_allowed": False,
        "label_summary_decision": load_json(args.label_summary).get("decision", "missing"),
        "data_scope": {
            "rows_loaded": int(len(frame)),
            "train_rows": int(len(train_frame)),
            "feature_count": int(len(feature_columns)),
            "train_splits": list(config.train_splits),
            "validation_split": config.validation_split,
            "diagnostic_split": config.diagnostic_split,
            "included_sessions": int(len(sessions)),
        },
        "label_balance": label_balance(frame, train_frame, eval_frames),
        "metrics": metrics,
        "feature_contract": {
            "status": "pass",
            "feature_count": int(len(feature_columns)),
            "forbidden_label_columns": [
                "blocked_protocol101_entries",
                "blocked_protocol101_pnl_0_00",
                "blocked_protocol101_pnl_0_10",
                "blocked_protocol101_pnl_0_25",
                "candidate_exit_dt",
            ],
        },
        "artifacts": {
            "manifest": str(artifact_dir / "manifest.json"),
            "model": str(model_path),
        },
        "next_required_evidence": [
            "Replay the learned estimator as a strict defer overlay on flat-calibrated challenger overrides.",
            "Reject challenger slot consumption unless predicted advantage clears estimated cost plus uncertainty.",
            "Require Q1/Q3 nonnegative same-scope deltas under $0.00/$0.10/$0.25 stress before another neural policy run.",
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


def load_labels(path: Path, sessions: set[tuple[str, str]]) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=LABEL_COLUMNS)
    frame = filter_sessions(frame, sessions)
    frame["decision_dt"] = pd.to_datetime(frame["decision_dt"], utc=True, errors="coerce")
    return frame[frame["decision_dt"].notna()].reset_index(drop=True)


def attach_features(
    labels: pd.DataFrame,
    flat_dataset: Path,
    feature_columns: list[str],
    sessions: set[tuple[str, str]],
) -> pd.DataFrame:
    flat_columns = list(dict.fromkeys([*KEY_COLUMNS, *feature_columns]))
    flat = pd.read_parquet(flat_dataset, columns=flat_columns)
    flat = filter_sessions(flat, sessions)
    flat["decision_dt"] = pd.to_datetime(flat["decision_dt"], utc=True, errors="coerce")
    frame = labels.merge(flat, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    for column in feature_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.reset_index(drop=True)


def filter_sessions(frame: pd.DataFrame, sessions: set[tuple[str, str]]) -> pd.DataFrame:
    if frame.empty or not sessions:
        return frame.iloc[0:0].copy()
    allowed = pd.MultiIndex.from_tuples(sessions, names=["split", "session"])
    current = pd.MultiIndex.from_frame(frame[["split", "session"]].astype(str))
    return frame.loc[current.isin(allowed)].copy()


def train_estimator(
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    config: SlotOpportunityCostEstimatorConfig,
) -> dict[str, Any]:
    x = coerce_feature_matrix(train_frame, feature_columns)
    y_cost_log = train_frame["target_log_cost"].to_numpy(dtype=float)
    y_count = train_frame["target_blocked_entries"].to_numpy(dtype=float)
    y_positive = train_frame["target_positive_cost"].astype(bool).to_numpy()
    weights = np.where(y_positive, float(config.positive_sample_weight), 1.0)

    cost_model = HistGradientBoostingRegressor(
        max_iter=160,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=int(config.random_seed),
    )
    count_model = HistGradientBoostingRegressor(
        max_iter=120,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=int(config.random_seed) + 11,
    )
    if len(np.unique(y_positive)) > 1:
        positive_model: Any = HistGradientBoostingClassifier(
            max_iter=140,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=int(config.random_seed) + 23,
        )
        positive_model.fit(x, y_positive, sample_weight=weights)
    else:
        positive_model = DummyClassifier(strategy="constant", constant=bool(y_positive[0]) if len(y_positive) else False)
        positive_model.fit(x, y_positive)
    cost_model.fit(x, y_cost_log, sample_weight=weights)
    count_model.fit(x, y_count, sample_weight=weights)
    return {
        "role_label": ROLE_LABEL,
        "training_spec": TRAINING_SPEC_LABEL,
        "config": config.to_dict(),
        "feature_columns": feature_columns,
        "cost_model": cost_model,
        "positive_model": positive_model,
        "count_model": count_model,
    }


def split_eval_frames(
    frame: pd.DataFrame,
    config: SlotOpportunityCostEstimatorConfig,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for split, group in frame.groupby("split", sort=True):
        if int(config.max_eval_rows_per_split) > 0 and len(group) > int(config.max_eval_rows_per_split):
            group = group.sample(n=int(config.max_eval_rows_per_split), random_state=int(config.random_seed) + len(out) + 101)
        out[str(split)] = group.reset_index(drop=True)
    return out


def evaluate_estimator(frame: pd.DataFrame, bundle: dict[str, Any], feature_columns: list[str]) -> dict[str, Any]:
    if frame.empty:
        return {"rows": 0}
    predictions = estimator_predictions(bundle, frame)
    y_cost = frame["target_cost"].to_numpy(dtype=float)
    y_positive = frame["target_positive_cost"].astype(bool).to_numpy()
    y_count = frame["target_blocked_entries"].to_numpy(dtype=float)
    pred_cost = predictions["estimated_blocked_protocol101_cost"].to_numpy(dtype=float)
    pred_positive = predictions["blocked_cost_positive_probability"].to_numpy(dtype=float)
    pred_count = predictions["estimated_blocked_entries"].to_numpy(dtype=float)
    abs_error = np.abs(pred_cost - y_cost)
    positive_mask = y_positive.astype(bool)
    return {
        "rows": int(len(frame)),
        "positive_cost_rate": float(np.mean(y_positive)),
        "mean_actual_cost": float(np.mean(y_cost)),
        "mean_predicted_cost": float(np.mean(pred_cost)),
        "cost_mae": float(mean_absolute_error(y_cost, pred_cost)),
        "cost_rmse": float(mean_squared_error(y_cost, pred_cost) ** 0.5),
        "cost_p90_abs_error": float(np.quantile(abs_error, 0.90)),
        "positive_cost_mae": float(mean_absolute_error(y_cost[positive_mask], pred_cost[positive_mask])) if positive_mask.any() else 0.0,
        "count_mae": float(mean_absolute_error(y_count, pred_count)),
        "positive_auc": safe_auc(y_positive, pred_positive),
        "positive_brier": safe_brier(y_positive, pred_positive),
        "positive_log_loss": safe_log_loss(y_positive, pred_positive),
        "calibration_bins": calibration_bins(y_positive, y_cost, pred_positive, pred_cost),
    }


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def safe_brier(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(y_true) == 0:
        return None
    return float(brier_score_loss(y_true, np.clip(y_score, 0.0, 1.0)))


def safe_log_loss(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(log_loss(y_true, np.clip(y_score, 1e-6, 1.0 - 1e-6)))


def calibration_bins(
    y_positive: np.ndarray,
    y_cost: np.ndarray,
    pred_positive: np.ndarray,
    pred_cost: np.ndarray,
    *,
    bins: int = 5,
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(
        {
            "actual_positive": y_positive.astype(float),
            "actual_cost": y_cost,
            "predicted_positive": pred_positive,
            "predicted_cost": pred_cost,
        }
    ).sort_values("predicted_positive")
    if frame.empty:
        return []
    frame["bin"] = pd.qcut(np.arange(len(frame)), q=min(int(bins), len(frame)), labels=False)
    rows = []
    for bin_id, group in frame.groupby("bin", sort=True):
        rows.append(
            {
                "bin": int(bin_id),
                "rows": int(len(group)),
                "mean_predicted_positive": float(group["predicted_positive"].mean()),
                "actual_positive_rate": float(group["actual_positive"].mean()),
                "mean_predicted_cost": float(group["predicted_cost"].mean()),
                "mean_actual_cost": float(group["actual_cost"].mean()),
            }
        )
    return rows


def label_balance(
    frame: pd.DataFrame,
    train_frame: pd.DataFrame,
    eval_frames: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    payload = {
        "all_rows": int(len(frame)),
        "all_positive_cost_rate": float(frame["target_positive_cost"].mean()) if len(frame) else 0.0,
        "train_rows": int(len(train_frame)),
        "train_positive_cost_rate": float(train_frame["target_positive_cost"].mean()) if len(train_frame) else 0.0,
    }
    for split, group in eval_frames.items():
        payload[f"{split}_rows"] = int(len(group))
        payload[f"{split}_positive_cost_rate"] = float(group["target_positive_cost"].mean()) if len(group) else 0.0
    return payload


def decide(metrics: dict[str, Any], train_frame: pd.DataFrame) -> str:
    validation = metrics.get("q1_2026", {})
    if train_frame.empty:
        return "slot_opportunity_cost_estimator_blocked_no_training_rows"
    if validation.get("rows", 0) <= 0:
        return "slot_opportunity_cost_estimator_blocked_no_validation_rows"
    if validation.get("positive_auc") is None:
        return "slot_opportunity_cost_estimator_blocked_validation_has_single_class"
    return "slot_opportunity_cost_estimator_ready_for_defer_overlay_replay"


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        f"Paper default baseline: `{payload['paper_default_baseline']}`",
        "Paid data downloaded: no",
        "Broker endpoint called: no",
        "Model training: yes",
        f"Decision: `{payload['decision']}`",
        "",
        "## Scope",
        "",
        f"- Rows loaded: `{payload['data_scope']['rows_loaded']}`",
        f"- Train rows: `{payload['data_scope']['train_rows']}`",
        f"- Feature count: `{payload['data_scope']['feature_count']}`",
        f"- Train splits: `{payload['data_scope']['train_splits']}`",
        f"- Validation split: `{payload['data_scope']['validation_split']}`",
        "",
        "## Metrics",
        "",
        table(metrics_rows(payload["metrics"]), ["split", "rows", "positive_cost_rate", "cost_mae", "cost_p90_abs_error", "positive_auc", "positive_brier", "count_mae"]),
        "",
        "## Feature Contract",
        "",
        f"- Status: `{payload['feature_contract']['status']}`",
        "- Runtime-forbidden label/exit/oracle columns are excluded from the feature set.",
        "",
        "## Next Required Evidence",
        "",
    ]
    lines.extend(f"{idx}. {item}" for idx, item in enumerate(payload["next_required_evidence"], start=1))
    lines.extend(["", "## Artifacts", ""])
    for name, path in payload["artifacts"].items():
        lines.append(f"- {name}: `{path}`")
    lines.extend(["", "## Outputs", ""])
    for name, path in payload["outputs"].items():
        lines.append(f"- {name}: `{path}`")
    return "\n".join(lines) + "\n"


def metrics_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for split, values in metrics.items():
        row = {"split": split}
        row.update(values)
        rows.append(row)
    return rows


def table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if value is None:
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def append_ledger(payload: dict[str, Any], out_dir: Path) -> None:
    ledger = Path("v4/ledger/RESEARCH_LEDGER.md")
    if not ledger.exists():
        return
    marker = f"## {ROLE_LABEL}"
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
                    "- Paid data downloaded: no",
                    "- Broker endpoint called: no",
                    "- Model training: yes",
                    f"- Decision: `{payload['decision']}`",
                    f"- Report: `{out_dir / 'report.md'}`",
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
