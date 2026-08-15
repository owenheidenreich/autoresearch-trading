"""Validation-calibrated abstention gate for the SPXW 0DTE pilot.

This evaluator asks a narrow question before any broader data purchase:

    Can the current action model learn when *not* to trade?

The model is still trained on January. February is split by session into a
calibration half and a selection half. Calibration learns how raw model scores
map to realized executable PnL; selection chooses abstention/risk controls; and
March remains the untouched holdout.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np

from v4.model.action_pilot import (
    ActionDecision,
    load_action_decisions,
    predict_actions,
    train_action_model,
)
from v4.model.environment_diagnostics import time_bucket
from v4.model.supervised_pilot import PilotConfig, Trade, session_from_path, split_name
from v4.scripts.evaluate_broad_data_purchase_signal import PurchaseGate
from v4.scripts.evaluate_risk_controlled_purchase_signal import (
    DAILY_LOSS_STOPS,
    MAX_TRADES_PER_DAY,
    metrics_with_concentration,
)
from v4.scripts.evaluate_timeaware_action_filters import FILTERS
from v4.scripts.train_spxw_supervised_pilot import POLICY_META


CALIBRATION_SCORE_NAMES = (
    "top_score",
    "edge_vs_no_trade",
    "directional_margin",
    "edge_plus_directional_margin",
    "worst_case_margin",
)
CALIBRATED_PNL_FLOORS = (0.0, 25.0, 50.0, 100.0, 150.0)
CALIBRATED_WIN_RATE_FLOORS = (0.0, 0.50, 0.55)
RAW_EDGE_FLOORS = (None, 0.0)
CALIBRATION_BINS = 8


@dataclass(frozen=True)
class ActionPredictionRecord:
    """One model-selected long-call or long-put candidate at a decision minute."""

    session: str
    decision_time: datetime
    action: int
    right: str
    offset: float
    pnl: float
    top_score: float
    no_trade_score: float
    edge_vs_no_trade: float
    directional_margin: float
    edge_plus_directional_margin: float
    worst_case_margin: float


@dataclass(frozen=True)
class ScoreCalibration:
    """Quantile-bin score calibration fitted on a validation calibration split."""

    score_name: str
    edges: tuple[float, ...]
    expected_pnl_by_bin: tuple[float, ...]
    win_rate_by_bin: tuple[float, ...]
    count_by_bin: tuple[int, ...]

    def _bin_index(self, score: float) -> int | None:
        if not np.isfinite(score) or len(self.expected_pnl_by_bin) == 0:
            return None
        idx = int(np.searchsorted(self.edges[1:-1], score, side="right"))
        return min(max(idx, 0), len(self.expected_pnl_by_bin) - 1)

    def expected_pnl(self, score: float) -> float:
        idx = self._bin_index(score)
        if idx is None:
            return 0.0
        return float(self.expected_pnl_by_bin[idx])

    def win_rate(self, score: float) -> float:
        idx = self._bin_index(score)
        if idx is None:
            return 0.0
        return float(self.win_rate_by_bin[idx])

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CalibratedConfig:
    """Locked abstention and one-contract risk controls."""

    score_name: str
    min_calibrated_pnl: float
    min_calibrated_win_rate: float
    min_raw_edge: float | None
    time_filter: str
    allowed_buckets: tuple[str, ...]
    max_trades_per_day: int
    daily_loss_stop: float | None

    @property
    def name(self) -> str:
        raw_edge = "none" if self.min_raw_edge is None else f"{self.min_raw_edge:.0f}"
        stop = "none" if self.daily_loss_stop is None else f"{self.daily_loss_stop:.0f}"
        max_trades = "unlimited" if self.max_trades_per_day >= 99 else str(self.max_trades_per_day)
        return (
            f"{self.time_filter}|score={self.score_name}|cal_pnl>="
            f"{self.min_calibrated_pnl:.0f}|win>={self.min_calibrated_win_rate:.2f}|"
            f"raw_edge={raw_edge}|max_trades={max_trades}|daily_stop={stop}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_derived"))
    p.add_argument("--out-dir", type=Path, default=Path("v4/audit/calibrated_abstention_signal"))
    p.add_argument("--seeds", nargs="*", type=int, default=[11, 22, 33, 44, 55, 66, 77, 88, 99, 111])
    p.add_argument("--policy-indexes", nargs="*", type=int, default=[0, 1, 2], choices=sorted(POLICY_META))
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--min-selection-trades", type=int, default=12)
    return p.parse_args()


def _paths_by_split(data_dir: Path) -> dict[str, list[Path]]:
    out = {"train": [], "validation": [], "test": []}
    for path in sorted(data_dir.glob("*.pkl")):
        out[split_name(session_from_path(path))].append(path)
    return out


def split_validation_by_session(
    decisions: Sequence[ActionDecision],
) -> tuple[list[ActionDecision], list[ActionDecision]]:
    """Split validation sessions into calibration and selection halves."""
    sessions = sorted({d.session for d in decisions})
    if len(sessions) < 2:
        midpoint = max(1, len(decisions) // 2)
        return list(decisions[:midpoint]), list(decisions[midpoint:])
    split_at = max(1, len(sessions) // 2)
    calibration_sessions = set(sessions[:split_at])
    calibration = [d for d in decisions if d.session in calibration_sessions]
    selection = [d for d in decisions if d.session not in calibration_sessions]
    return calibration, selection


def prediction_records(
    decisions: Sequence[ActionDecision],
    predictions: np.ndarray,
) -> list[ActionPredictionRecord]:
    records: list[ActionPredictionRecord] = []
    for decision, pred in zip(decisions, predictions):
        if len(pred) < 3:
            continue
        action = int(np.argmax(pred[1:]) + 1)
        top_score = float(pred[action])
        other_action = 2 if action == 1 else 1
        no_trade_score = float(pred[0])
        directional_margin = top_score - float(pred[other_action])
        edge_vs_no_trade = top_score - no_trade_score
        pnl = float(decision.labels[action])
        if not all(
            np.isfinite(x)
            for x in (top_score, no_trade_score, directional_margin, edge_vs_no_trade, pnl)
        ):
            continue
        records.append(
            ActionPredictionRecord(
                session=decision.session,
                decision_time=decision.decision_time,
                action=action,
                right="C" if action == 1 else "P",
                offset=float(decision.offsets[action]),
                pnl=pnl,
                top_score=top_score,
                no_trade_score=no_trade_score,
                edge_vs_no_trade=edge_vs_no_trade,
                directional_margin=directional_margin,
                edge_plus_directional_margin=edge_vs_no_trade + 0.25 * directional_margin,
                worst_case_margin=min(edge_vs_no_trade, directional_margin),
            )
        )
    return records


def _weighted_isotonic_non_decreasing(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators fit for a short nondecreasing sequence."""
    blocks: list[list[float]] = []
    for idx, (value, weight) in enumerate(zip(values, weights)):
        if weight <= 0:
            continue
        blocks.append([float(idx), float(idx), float(value), float(weight)])
        while len(blocks) >= 2 and blocks[-2][2] > blocks[-1][2]:
            right = blocks.pop()
            left = blocks.pop()
            merged_weight = left[3] + right[3]
            merged_value = (
                (left[2] * left[3] + right[2] * right[3]) / merged_weight
                if merged_weight > 0
                else 0.0
            )
            blocks.append([left[0], right[1], merged_value, merged_weight])

    fitted = np.zeros_like(values, dtype=float)
    if not blocks:
        return fitted
    for start, end, level, _ in blocks:
        fitted[int(start) : int(end) + 1] = level
    empty = weights <= 0
    if empty.any():
        observed = np.where(~empty)[0]
        for idx in np.where(empty)[0]:
            nearest = observed[np.argmin(np.abs(observed - idx))]
            fitted[idx] = fitted[nearest]
    return fitted


def fit_score_calibration(
    records: Sequence[ActionPredictionRecord],
    score_name: str,
    *,
    bins: int = CALIBRATION_BINS,
) -> ScoreCalibration:
    scores = np.asarray([getattr(r, score_name) for r in records], dtype=float)
    pnls = np.asarray([r.pnl for r in records], dtype=float)
    valid = np.isfinite(scores) & np.isfinite(pnls)
    scores = scores[valid]
    pnls = pnls[valid]
    if len(scores) == 0:
        return ScoreCalibration(score_name, (0.0, 1.0), (0.0,), (0.0,), (0,))

    raw_edges = np.quantile(scores, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(raw_edges.astype(float))
    if len(edges) < 2:
        span = max(abs(float(scores[0])) * 1e-6, 1e-6)
        edges = np.asarray([float(scores[0]) - span, float(scores[0]) + span], dtype=float)
    else:
        span = max((float(edges[-1]) - float(edges[0])) * 1e-6, 1e-6)
        edges[0] = float(edges[0]) - span
        edges[-1] = float(edges[-1]) + span

    bin_index = np.searchsorted(edges[1:-1], scores, side="right")
    bin_count = len(edges) - 1
    expected = np.zeros(bin_count, dtype=float)
    win_rate = np.zeros(bin_count, dtype=float)
    counts = np.zeros(bin_count, dtype=float)
    global_mean = float(pnls.mean())
    global_win_rate = float((pnls > 0).mean())
    for idx in range(bin_count):
        mask = bin_index == idx
        counts[idx] = float(mask.sum())
        if mask.any():
            expected[idx] = float(pnls[mask].mean())
            win_rate[idx] = float((pnls[mask] > 0).mean())
        else:
            expected[idx] = global_mean
            win_rate[idx] = global_win_rate

    expected = _weighted_isotonic_non_decreasing(expected, counts)
    win_rate = _weighted_isotonic_non_decreasing(win_rate, counts)
    return ScoreCalibration(
        score_name=score_name,
        edges=tuple(float(x) for x in edges),
        expected_pnl_by_bin=tuple(float(x) for x in expected),
        win_rate_by_bin=tuple(float(x) for x in win_rate),
        count_by_bin=tuple(int(x) for x in counts),
    )


def fit_calibrations(
    records: Sequence[ActionPredictionRecord],
) -> dict[str, ScoreCalibration]:
    return {
        score_name: fit_score_calibration(records, score_name)
        for score_name in CALIBRATION_SCORE_NAMES
    }


def simulate_calibrated_policy(
    records: Sequence[ActionPredictionRecord],
    calibrations: dict[str, ScoreCalibration],
    *,
    config: CalibratedConfig,
    cooldown_minutes: int,
    strategy: str,
) -> list[Trade]:
    trades: list[Trade] = []
    next_time_by_session: dict[str, datetime] = {}
    trades_by_session: dict[str, int] = {}
    pnl_by_session: dict[str, float] = {}
    halted_sessions: set[str] = set()
    allowed = set(config.allowed_buckets)
    calibration = calibrations[config.score_name]

    for record in records:
        if time_bucket(record.decision_time) not in allowed:
            continue
        if record.session in halted_sessions:
            continue
        if trades_by_session.get(record.session, 0) >= config.max_trades_per_day:
            continue
        next_time = next_time_by_session.get(record.session)
        if next_time is not None and record.decision_time < next_time:
            continue
        if config.min_raw_edge is not None and record.edge_vs_no_trade < config.min_raw_edge:
            continue

        score = float(getattr(record, config.score_name))
        calibrated_pnl = calibration.expected_pnl(score)
        calibrated_win_rate = calibration.win_rate(score)
        if calibrated_pnl < config.min_calibrated_pnl:
            continue
        if calibrated_win_rate < config.min_calibrated_win_rate:
            continue
        if not np.isfinite(record.pnl):
            continue

        trades.append(
            Trade(
                session=record.session,
                decision_time=record.decision_time.isoformat(),
                pnl=float(record.pnl),
                score=calibrated_pnl,
                right=record.right,
                offset=float(record.offset),
                strategy=strategy,
            )
        )
        trades_by_session[record.session] = trades_by_session.get(record.session, 0) + 1
        pnl_by_session[record.session] = pnl_by_session.get(record.session, 0.0) + record.pnl
        next_time_by_session[record.session] = record.decision_time + timedelta(
            minutes=cooldown_minutes
        )
        if (
            config.daily_loss_stop is not None
            and pnl_by_session[record.session] <= config.daily_loss_stop
        ):
            halted_sessions.add(record.session)

    return trades


def _calibrated_config_grid(
    calibrations: dict[str, ScoreCalibration],
) -> list[CalibratedConfig]:
    configs = []
    daily_loss_stops = (None, -1_000.0)
    max_trades_per_day = tuple(x for x in MAX_TRADES_PER_DAY if x in (2, 3, 4, 99))
    for score_name in calibrations:
        for min_pnl in CALIBRATED_PNL_FLOORS:
            for min_win_rate in CALIBRATED_WIN_RATE_FLOORS:
                for min_raw_edge in RAW_EDGE_FLOORS:
                    for filter_name, allowed_buckets in FILTERS.items():
                        for max_trades in max_trades_per_day:
                            for daily_stop in daily_loss_stops:
                                configs.append(
                                    CalibratedConfig(
                                        score_name=score_name,
                                        min_calibrated_pnl=float(min_pnl),
                                        min_calibrated_win_rate=float(min_win_rate),
                                        min_raw_edge=min_raw_edge,
                                        time_filter=filter_name,
                                        allowed_buckets=tuple(allowed_buckets),
                                        max_trades_per_day=int(max_trades),
                                        daily_loss_stop=daily_stop,
                                    )
                                )
    return configs


def _selection_key(row: dict) -> tuple[float, float, float, float, float, float]:
    metrics = row["metrics"]
    profit_factor = float(metrics["profit_factor"])
    if not np.isfinite(profit_factor):
        profit_factor = 999.0
    return (
        float(metrics["total_pnl"]),
        min(profit_factor, 5.0),
        float(metrics["positive_day_fraction"]),
        -float(metrics["top_day_profit_share"]),
        float(metrics["max_drawdown"]),
        float(metrics["trades"]),
    )


def select_calibrated_config(
    records: Sequence[ActionPredictionRecord],
    calibrations: dict[str, ScoreCalibration],
    *,
    cooldown_minutes: int,
    min_selection_trades: int,
) -> tuple[CalibratedConfig, list[dict]]:
    """Choose abstention/risk controls from the selection validation split only."""
    sweep = []
    for config in _calibrated_config_grid(calibrations):
        trades = simulate_calibrated_policy(
            records,
            calibrations,
            config=config,
            cooldown_minutes=cooldown_minutes,
            strategy="validation_calibrated_sweep",
        )
        sweep.append(
            {
                "config": asdict(config) | {"name": config.name},
                "metrics": metrics_with_concentration(trades),
            }
        )

    eligible = [
        row
        for row in sweep
        if row["metrics"]["trades"] >= min_selection_trades
        and row["metrics"]["total_pnl"] > 0
        and row["metrics"]["profit_factor"] >= 1.10
        and row["metrics"]["positive_day_fraction"] >= 0.45
        and row["metrics"]["max_drawdown"] >= -5_000
        and row["metrics"]["top_day_profit_share"] <= 0.75
    ]
    pool = eligible if eligible else sweep
    best = max(pool, key=_selection_key)
    cfg = best["config"]
    selected = CalibratedConfig(
        score_name=str(cfg["score_name"]),
        min_calibrated_pnl=float(cfg["min_calibrated_pnl"]),
        min_calibrated_win_rate=float(cfg["min_calibrated_win_rate"]),
        min_raw_edge=cfg["min_raw_edge"],
        time_filter=str(cfg["time_filter"]),
        allowed_buckets=tuple(cfg["allowed_buckets"]),
        max_trades_per_day=int(cfg["max_trades_per_day"]),
        daily_loss_stop=cfg["daily_loss_stop"],
    )
    return selected, sweep


def _run_one(
    *,
    paths: dict[str, list[Path]],
    policy_index: int,
    seed: int,
    epochs: int,
    batch_size: int,
    min_selection_trades: int,
) -> dict:
    policy_name, cooldown = POLICY_META[policy_index]
    config = PilotConfig(
        policy_index=policy_index,
        policy_name=policy_name,
        cooldown_minutes=cooldown,
        epochs=epochs,
        batch_size=batch_size,
        hidden_dim=128,
        seed=seed,
    )
    split_decisions = {
        split: load_action_decisions(files, policy_index=policy_index)
        for split, files in paths.items()
    }
    calibration_decisions, selection_decisions = split_validation_by_session(
        split_decisions["validation"]
    )
    model, scaler, history = train_action_model(
        split_decisions["train"],
        calibration_decisions,
        config=config,
    )
    decision_sets = {
        "train": split_decisions["train"],
        "calibration": calibration_decisions,
        "validation_selection": selection_decisions,
        "test": split_decisions["test"],
    }
    predictions = {
        split: predict_actions(model, scaler, decisions, target_scale=config.target_scale)
        for split, decisions in decision_sets.items()
    }
    records = {
        split: prediction_records(decision_sets[split], predictions[split])
        for split in decision_sets
    }
    calibrations = fit_calibrations(records["calibration"])
    selected_config, selection_sweep = select_calibrated_config(
        records["validation_selection"],
        calibrations,
        cooldown_minutes=cooldown,
        min_selection_trades=min_selection_trades,
    )
    trades_by_split = {
        split: simulate_calibrated_policy(
            split_records,
            calibrations,
            config=selected_config,
            cooldown_minutes=cooldown,
            strategy="action_neural_calibrated_abstention",
        )
        for split, split_records in records.items()
    }
    return {
        "policy_index": policy_index,
        "policy_name": policy_name,
        "seed": seed,
        "selected_config": asdict(selected_config) | {"name": selected_config.name},
        "best_epoch": next((x["epoch"] for x in history if x["is_best"]), None),
        "split_counts": {
            split: {"decisions": len(decision_sets[split]), "records": len(records[split])}
            for split in decision_sets
        },
        "calibrations": {name: cal.to_dict() for name, cal in calibrations.items()},
        "selected": {
            split: metrics_with_concentration(trades)
            for split, trades in trades_by_split.items()
        },
        "selection_sweep_top10": sorted(selection_sweep, key=_selection_key, reverse=True)[:10],
    }


def _count_values(values: Sequence[object]) -> dict[str, int]:
    return {
        str(value): values.count(value)
        for value in sorted(set(values), key=lambda x: str(x))
    }


def _aggregate(results: list[dict], gate: PurchaseGate) -> dict:
    by_policy = {}
    for policy_index in sorted({r["policy_index"] for r in results}):
        rows = [r for r in results if r["policy_index"] == policy_index]
        test = [r["selected"]["test"] for r in rows]
        configs = [r["selected_config"]["name"] for r in rows]
        filters = [r["selected_config"]["time_filter"] for r in rows]
        score_names = [r["selected_config"]["score_name"] for r in rows]
        max_trades = [r["selected_config"]["max_trades_per_day"] for r in rows]
        daily_stops = [r["selected_config"]["daily_loss_stop"] for r in rows]
        raw_edges = [r["selected_config"]["min_raw_edge"] for r in rows]
        arr = {
            "total_pnl": np.asarray([m["total_pnl"] for m in test], dtype=float),
            "profit_factor": np.asarray([m["profit_factor"] for m in test], dtype=float),
            "max_drawdown": np.asarray([m["max_drawdown"] for m in test], dtype=float),
            "trades": np.asarray([m["trades"] for m in test], dtype=float),
            "positive_day_fraction": np.asarray(
                [m["positive_day_fraction"] for m in test], dtype=float
            ),
            "top_day_profit_share": np.asarray(
                [m["top_day_profit_share"] for m in test], dtype=float
            ),
        }
        summary = {
            "policy_name": POLICY_META[policy_index][0],
            "runs": len(rows),
            "selected_config_counts": _count_values(configs),
            "selected_filter_counts": _count_values(filters),
            "selected_score_counts": _count_values(score_names),
            "selected_max_trades_counts": _count_values(max_trades),
            "selected_daily_stop_counts": _count_values(daily_stops),
            "selected_raw_edge_counts": _count_values(raw_edges),
            "test_pnl_median": float(np.median(arr["total_pnl"])),
            "test_pnl_mean": float(arr["total_pnl"].mean()),
            "test_profit_factor_median": float(np.median(arr["profit_factor"])),
            "test_max_drawdown_median": float(np.median(arr["max_drawdown"])),
            "test_trades_median": float(np.median(arr["trades"])),
            "positive_seed_fraction": float((arr["total_pnl"] > 0).mean()),
            "positive_day_fraction_median": float(np.median(arr["positive_day_fraction"])),
            "top_day_profit_share_median": float(np.median(arr["top_day_profit_share"])),
        }
        summary["passes_broad_purchase_gate"] = bool(
            summary["test_profit_factor_median"] >= gate.min_test_profit_factor_median
            and summary["test_pnl_median"] >= gate.min_test_pnl_median
            and summary["positive_seed_fraction"] >= gate.min_positive_seed_fraction
            and summary["positive_day_fraction_median"] >= gate.min_positive_day_fraction_median
            and summary["test_max_drawdown_median"] >= gate.max_drawdown_floor_median
            and summary["test_trades_median"] >= gate.min_test_trades_median
            and summary["top_day_profit_share_median"] <= gate.max_top_day_profit_share_median
        )
        by_policy[str(policy_index)] = summary

    any_pass = any(x["passes_broad_purchase_gate"] for x in by_policy.values())
    recommendation = (
        "broad_purchase_supported"
        if any_pass
        else "no_broad_purchase_yet_continue_existing_data_iteration"
    )
    return {
        "gate": asdict(gate),
        "recommendation": recommendation,
        "by_policy": by_policy,
    }


def _write_report(out_dir: Path, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")

    aggregate = payload["aggregate"]
    lines = [
        "# Calibrated Abstention Broad Data Purchase Signal",
        "",
        payload["framing"],
        "",
        f"Recommendation: **{aggregate['recommendation']}**",
        "",
        "## Gate",
        "",
        "| Criterion | Value |",
        "|---|---:|",
    ]
    for key, value in aggregate["gate"].items():
        lines.append(f"| {key} | {value} |")
    lines += [
        "",
        "## Policy Robustness",
        "",
        "| Policy | Runs | Filters | Scores | Max Trades | Daily Stops | Median PnL | Median PF | Median DD | Trades | Positive Seeds | Positive Days | Top-Day Share | Pass |",
        "|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, summary in aggregate["by_policy"].items():
        lines.append(
            f"| {summary['policy_name']} | {summary['runs']} | "
            f"{summary['selected_filter_counts']} | {summary['selected_score_counts']} | "
            f"{summary['selected_max_trades_counts']} | {summary['selected_daily_stop_counts']} | "
            f"{summary['test_pnl_median']:.0f} | {summary['test_profit_factor_median']:.3f} | "
            f"{summary['test_max_drawdown_median']:.0f} | {summary['test_trades_median']:.0f} | "
            f"{summary['positive_seed_fraction']:.2f} | "
            f"{summary['positive_day_fraction_median']:.2f} | "
            f"{summary['top_day_profit_share_median']:.2f} | "
            f"{summary['passes_broad_purchase_gate']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        (
            "Calibration is fitted on the first half of February only. Abstention and "
            "risk controls are selected on the second half of February only. March is "
            "held out for the purchase decision. A fail here means the project should "
            "continue improving on the existing pilot before buying a broad history."
        ),
    ]
    md_path = out_dir / "report.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(json_path)
    print(md_path)


def main() -> int:
    args = parse_args()
    paths = _paths_by_split(args.data_dir)
    gate = PurchaseGate()
    results = []
    for policy_index in args.policy_indexes:
        for seed in args.seeds:
            print(f"running calibrated policy={policy_index} seed={seed}", flush=True)
            results.append(
                _run_one(
                    paths=paths,
                    policy_index=policy_index,
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    min_selection_trades=args.min_selection_trades,
                )
            )

    aggregate = _aggregate(results, gate)
    payload = {
        "framing": (
            "This evaluator uses only the existing January-March pilot data. The model "
            "is trained on January, score calibration is fitted on the first half of "
            "February, abstention/risk controls are selected on the second half of "
            "February, and March remains the untouched holdout. No new paid data is "
            "used or purchased."
        ),
        "seeds": args.seeds,
        "policy_indexes": args.policy_indexes,
        "calibration_grid": {
            "score_names": CALIBRATION_SCORE_NAMES,
            "calibrated_pnl_floors": CALIBRATED_PNL_FLOORS,
            "calibrated_win_rate_floors": CALIBRATED_WIN_RATE_FLOORS,
            "raw_edge_floors": RAW_EDGE_FLOORS,
            "time_filters": FILTERS,
            "max_trades_per_day": tuple(x for x in MAX_TRADES_PER_DAY if x in (2, 3, 4, 99)),
            "daily_loss_stops": tuple(x for x in DAILY_LOSS_STOPS if x in (None, -1_000.0)),
            "calibration_bins": CALIBRATION_BINS,
        },
        "aggregate": aggregate,
        "runs": results,
    }
    _write_report(args.out_dir, payload)
    print(json.dumps(aggregate, indent=2, allow_nan=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
