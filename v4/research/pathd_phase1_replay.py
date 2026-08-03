"""Strict-serial Phase-1 replay and bounded negative-control packet."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from v4.research.phase1_exit_model import stable_hash
from v4.research.phase1_exit_model import sensitivity_label_path


SCHEMA_VERSION = "pathd.phase1-four-box-replay.v1"
CONTROL_EXIT_POLICY = "hold_to_1555"


class Phase1ReplayError(RuntimeError):
    """Replay inputs violate the frozen campaign contract."""


def _load_campaign(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    semantic = dict(payload)
    expected = semantic.pop("campaign_sha256", None)
    if expected != stable_hash(semantic):
        raise Phase1ReplayError(f"campaign hash drift: {path}")
    if payload.get("protected_holdout_opened") is not False:
        raise Phase1ReplayError("protected holdout state is not closed")
    return payload


def _prediction_path(exit_root: Path, fold: int, session: str, trajectory_id: str) -> Path:
    return (
        exit_root
        / "oof_predictions"
        / f"fold={fold}"
        / f"session={session}"
        / f"{trajectory_id}.parquet"
    )


def _first_value(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    prediction: pd.DataFrame,
    *,
    utility: np.ndarray,
) -> tuple[float, int, str]:
    premium = float(features["entry_fill_option_price"].iloc[0]) * 100.0
    floor = features["current_net_pnl_dollars"].to_numpy(float) <= -0.50 * premium
    exit_request = np.asarray(utility, dtype=float) <= 0.0
    indexes = np.flatnonzero(floor | exit_request)
    if len(indexes):
        index = int(indexes[0])
        reason = "CATASTROPHIC_FLOOR" if floor[index] else "LEARNED_EXIT"
        return float(labels["exit_until_filled_value_dollars"].iloc[index]), index, reason
    return float(labels["hold_to_1555_value_dollars"].iloc[0]), len(labels) - 1, "FORCED_FLAT"


def _horizon_value(labels: pd.DataFrame, features: pd.DataFrame, seconds: int) -> float:
    index = np.flatnonzero(features["seconds_held"].to_numpy(float) >= seconds)
    if not len(index):
        return float(labels["hold_to_1555_value_dollars"].iloc[0])
    return float(labels["exit_until_filled_value_dollars"].iloc[int(index[0])])


def _stop_target_value(
    labels: pd.DataFrame, features: pd.DataFrame, *, stop: float, target: float
) -> float:
    returns = features["current_return_on_entry_premium"].to_numpy(float)
    indexes = np.flatnonzero((returns <= stop) | (returns >= target))
    if not len(indexes):
        return float(labels["hold_to_1555_value_dollars"].iloc[0])
    return float(labels["exit_until_filled_value_dollars"].iloc[int(indexes[0])])


def _random_value(labels: pd.DataFrame, *, seed: int, trajectory_id: str) -> float:
    digest = hashlib.sha256(f"{seed}|{trajectory_id}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    index = min(len(labels) - 1, int(math.floor(fraction * len(labels))))
    return float(labels["exit_until_filled_value_dollars"].iloc[index])


def _bootstrap_lcb(session_values: pd.Series, *, seed: int = 901, draws: int = 10_000) -> float:
    values = session_values.to_numpy(float)
    if not len(values):
        return float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=float)
    for start in range(0, draws, 1_000):
        count = min(1_000, draws - start)
        indexes = rng.integers(0, len(values), size=(count, len(values)))
        means[start : start + count] = values[indexes].mean(axis=1)
    return float(np.quantile(means, 0.05))


def _summary(rows: pd.DataFrame, value_column: str) -> dict[str, Any]:
    session = rows.groupby("session", sort=True)[value_column].sum()
    return {
        "trajectories": int(len(rows)),
        "sessions": int(len(session)),
        "pooled_net_pnl_dollars": float(rows[value_column].sum()),
        "mean_session_pnl_dollars": float(session.mean()) if len(session) else None,
        "one_sided_95pct_session_bootstrap_lcb_dollars": _bootstrap_lcb(session),
        "positive_sessions": int((session > 0.0).sum()),
    }


def run_four_box_replay(
    *,
    scratch_root: Path,
    entry_campaign_path: Path,
    exit_campaign_path: Path,
    report_root: Path,
) -> dict[str, Any]:
    """Replay learned/control entries against hold/learned exits identically."""

    entry = _load_campaign(entry_campaign_path)
    exit_campaign = _load_campaign(exit_campaign_path)
    index = pd.read_parquet(
        entry.get("evaluation_trajectory_index_path", entry["trajectory_index_path"])
    )
    if index["trajectory_id"].duplicated().any():
        raise Phase1ReplayError("trajectory index identity is not unique")
    exit_root = Path(exit_campaign_path).parent
    rows: list[dict[str, Any]] = []
    exit_value_curves: dict[str, np.ndarray] = {}
    for item in index.sort_values(["session", "trajectory_id"]).to_dict("records"):
        session = str(item["session"])
        trajectory_id = str(item["trajectory_id"])
        fold = int(item["outer_fold"])
        feature_path = Path(scratch_root) / "exit_features" / f"session={session}" / f"{trajectory_id}.parquet"
        label_path = Path(scratch_root) / "exit_labels" / f"session={session}" / f"{trajectory_id}.parquet"
        prediction_path = _prediction_path(exit_root, fold, session, trajectory_id)
        for path in (feature_path, label_path, prediction_path):
            if not path.is_file():
                raise Phase1ReplayError(f"missing replay partition: {path}")
        features = pd.read_parquet(feature_path)
        labels = pd.read_parquet(label_path)
        predictions = pd.read_parquet(prediction_path)
        identity = ["session", "trajectory_id", "decision_time_ns"]
        if not features[identity].equals(labels[identity]) or not features[identity].equals(
            predictions[identity]
        ):
            raise Phase1ReplayError(f"replay identity drift: {trajectory_id}")
        learned_value, exit_index, exit_reason = _first_value(
            features,
            labels,
            predictions,
            utility=predictions["utility_hold"].to_numpy(float),
        )
        row: dict[str, Any] = {
            **item,
            "control_exit_value": float(labels["hold_to_1555_value_dollars"].iloc[0]),
            "learned_exit_value": learned_value,
            "learned_exit_index": exit_index,
            "learned_exit_reason": exit_reason,
            "trajectory_rows": len(labels),
            "learned_exit_fraction": exit_index / max(1, len(labels) - 1),
            "exit_immediate": float(labels["exit_until_filled_value_dollars"].iloc[0]),
            "exit_60s": _horizon_value(labels, features, 60),
            "exit_300s": _horizon_value(labels, features, 300),
            "exit_900s": _horizon_value(labels, features, 900),
            "stop50_target100": _stop_target_value(labels, features, stop=-0.50, target=1.00),
            "stop25_target50": _stop_target_value(labels, features, stop=-0.25, target=0.50),
        }
        exit_value_curves[trajectory_id] = labels[
            "exit_until_filled_value_dollars"
        ].to_numpy(float)
        for name, utility in (
            ("constant", np.zeros(len(predictions))),
            ("sign_reversed", -predictions["utility_hold"].to_numpy(float)),
            ("time_shifted", predictions["utility_hold"].shift(-1).fillna(0.0).to_numpy(float)),
        ):
            row[f"negative_{name}"], _, _ = _first_value(
                features, labels, predictions, utility=utility
            )
        rng = np.random.default_rng(
            int(hashlib.sha256(trajectory_id.encode()).hexdigest()[:16], 16)
        )
        shuffled = predictions["utility_hold"].to_numpy(float).copy()
        rng.shuffle(shuffled)
        row["negative_shuffled"], _, _ = _first_value(
            features, labels, predictions, utility=shuffled
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    for role, role_frame in frame.groupby("entry_policy_role", sort=True):
        fractions = role_frame["learned_exit_fraction"].to_numpy(float)
        target_indexes = role_frame.index.to_numpy()
        for seed in range(8):
            rng = np.random.default_rng(
                int(hashlib.sha256(f"{role}|matched-rate|{seed}".encode()).hexdigest()[:16], 16)
            )
            assigned = fractions.copy()
            rng.shuffle(assigned)
            for frame_index, fraction in zip(target_indexes, assigned, strict=True):
                trajectory_id = str(frame.at[frame_index, "trajectory_id"])
                curve = exit_value_curves[trajectory_id]
                curve_index = min(len(curve) - 1, int(round(fraction * max(0, len(curve) - 1))))
                frame.at[frame_index, f"matched_random_{seed}"] = float(curve[curve_index])
                frame.at[frame_index, f"matched_random_{seed}_fraction"] = float(fraction)
    boxes: dict[str, Any] = {}
    roles = {
        "learned_entry": "LEARNED_OOF",
        "control_entry": "DETERMINISTIC_CONTROL_OOF",
    }
    for entry_name, role in roles.items():
        selected = frame[frame["entry_policy_role"].eq(role)]
        boxes[f"{entry_name}/control_exit"] = _summary(selected, "control_exit_value")
        boxes[f"{entry_name}/learned_exit"] = _summary(selected, "learned_exit_value")
    learned_rows = frame[frame["entry_policy_role"].eq("LEARNED_OOF")]
    comparator_columns = [
        "exit_immediate", "control_exit_value", "exit_60s", "exit_300s", "exit_900s",
        "stop50_target100", "stop25_target50", *[f"matched_random_{seed}" for seed in range(8)],
    ]
    comparators = {name: _summary(learned_rows, name) for name in comparator_columns}
    negatives = {
        name.removeprefix("negative_"): _summary(learned_rows, name)
        for name in frame.columns
        if name.startswith("negative_")
    }
    learned_integrated = boxes["learned_entry/learned_exit"]
    best_comparator_name = max(
        comparator_columns,
        key=lambda name: comparators[name]["pooled_net_pnl_dollars"],
    )
    best_comparator = comparators[best_comparator_name]
    sensitivity_results: dict[str, Any] = {}
    for fee_per_side in (1.5, 2.0):
        for latency_seconds in (0, 1, 2, 5):
            integrated_rows = []
            comparator_rows = []
            session_rows = []
            for item in learned_rows.to_dict("records"):
                labels = pd.read_parquet(sensitivity_label_path(
                    scratch_root,
                    session=str(item["session"]),
                    trajectory_id=str(item["trajectory_id"]),
                    fee_per_side_dollars=fee_per_side,
                    latency_seconds=latency_seconds,
                ))
                features = pd.read_parquet(
                    Path(scratch_root)
                    / "exit_features"
                    / f"session={item['session']}"
                    / f"{item['trajectory_id']}.parquet"
                )
                if item["learned_exit_reason"] == "FORCED_FLAT":
                    integrated = float(labels["hold_to_1555_value_dollars"].iloc[0])
                else:
                    integrated = float(
                        labels["exit_until_filled_value_dollars"].iloc[
                            int(item["learned_exit_index"])
                        ]
                    )
                if best_comparator_name == "control_exit_value":
                    comparator = float(labels["hold_to_1555_value_dollars"].iloc[0])
                elif best_comparator_name == "exit_immediate":
                    comparator = float(labels["exit_until_filled_value_dollars"].iloc[0])
                elif best_comparator_name.startswith("exit_"):
                    comparator = _horizon_value(
                        labels, features, int(best_comparator_name.removeprefix("exit_").removesuffix("s"))
                    )
                elif best_comparator_name == "stop50_target100":
                    comparator = _stop_target_value(labels, features, stop=-0.50, target=1.00)
                elif best_comparator_name == "stop25_target50":
                    comparator = _stop_target_value(labels, features, stop=-0.25, target=0.50)
                elif best_comparator_name.startswith("matched_random_"):
                    fraction = float(item[f"{best_comparator_name}_fraction"])
                    comparator_index = min(
                        len(labels) - 1,
                        int(round(fraction * max(0, len(labels) - 1))),
                    )
                    comparator = float(
                        labels["exit_until_filled_value_dollars"].iloc[comparator_index]
                    )
                else:
                    raise Phase1ReplayError(f"unsupported comparator: {best_comparator_name}")
                integrated_rows.append(integrated)
                comparator_rows.append(comparator)
                session_rows.append(str(item["session"]))
            delta = np.asarray(integrated_rows) - np.asarray(comparator_rows)
            session_delta = pd.DataFrame(
                {"session": session_rows, "delta": delta}
            ).groupby("session")["delta"].sum()
            key = f"roundtrip_fee={fee_per_side*2.0:.2f}|latency={latency_seconds}s"
            sensitivity_results[key] = {
                "integrated_net_pnl_dollars": float(np.sum(integrated_rows)),
                "best_comparator_net_pnl_dollars": float(np.sum(comparator_rows)),
                "delta_dollars": float(np.sum(delta)),
                "one_sided_95pct_session_bootstrap_delta_lcb_dollars": _bootstrap_lcb(
                    session_delta
                ),
                "directionally_positive": bool(np.sum(delta) > 0.0),
            }
    fold_delta = {}
    for fold, fold_rows in learned_rows.groupby("outer_fold", sort=True):
        fold_delta[str(int(fold))] = float(
            fold_rows["learned_exit_value"].sum() - fold_rows[best_comparator_name].sum()
        )
    positive_skill = int(exit_campaign.get("positive_target_skill_folds", 0)) > 0
    negative_accepted = {
        name: value["pooled_net_pnl_dollars"] > best_comparator["pooled_net_pnl_dollars"]
        for name, value in negatives.items()
    }
    underpowered = bool(
        learned_integrated["sessions"] < 30 or learned_rows["outer_fold"].nunique() < 5
    )
    tier_s_supported = bool(
        not underpowered
        and
        learned_integrated["pooled_net_pnl_dollars"]
        > best_comparator["pooled_net_pnl_dollars"]
        and learned_integrated["one_sided_95pct_session_bootstrap_lcb_dollars"] > 0.0
        and sum(value > 0.0 for value in fold_delta.values()) >= 4
        and positive_skill
        and all(value["directionally_positive"] for value in sensitivity_results.values())
        and not any(negative_accepted.values())
    )
    if any(negative_accepted.values()):
        verdict = "INVALID"
    elif underpowered:
        verdict = "UNDERPOWERED"
    else:
        verdict = "TIER_S_SUPPORTED" if tier_s_supported else "TIER_S_NOT_SUPPORTED"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "verdict": verdict,
        "four_boxes": boxes,
        "comparators": comparators,
        "best_comparator": best_comparator_name,
        "paired_fold_delta_vs_best_comparator": fold_delta,
        "exit_target_positive_skill_folds": exit_campaign.get("positive_target_skill_folds"),
        "negative_controls": negatives,
        "negative_control_accepted": negative_accepted,
        "legacy_p5": "UNAVAILABLE_UNLESS_SEPARATELY_PROVEN_LAWFUL",
        "latency_and_fee_sensitivities": sensitivity_results,
        "protected_holdout_opened": False,
        "paper_order_submitted": False,
    }
    payload["replay_sha256"] = stable_hash(payload)
    report_root = Path(report_root)
    report_root.mkdir(parents=True, exist_ok=False)
    frame.to_parquet(report_root / "trajectory_outcomes.parquet", index=False, compression="zstd")
    (report_root / "replay.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


__all__ = ["Phase1ReplayError", "run_four_box_replay"]
