"""Apply frozen Protocol 081 sequence artifacts to Q4 2024 external paths.

Inputs are the Protocol 104/105 Q4 2024 external candidate lifecycle tables.
This script does not train a model. It loads frozen Protocol 081 artifacts and
emits the same ``selected_trades_sequence_exits.json`` shape used by Protocol
092/101 candidate tables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v4.live.protocol066_inference import (
    OVERRIDE_THRESHOLD_EPSILON,
    load_protocol066_artifact,
    predict_protocol066_sequence,
)
from v4.model.hypothesis_protocol import stress_trades
from v4.model.supervised_pilot import Trade
from v4.scripts.evaluate_risk_controlled_purchase_signal import metrics_with_concentration


DEFAULT_SEQUENCE_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_105_q4_2024_external_lifecycle_sequence")
DEFAULT_PROTOCOL081_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_081_q4start_residual_sequence_deterministic_artifacts")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_106_q4_2024_external_protocol081_exits")
DEFAULT_FOLD = "train_q1_2025_q2_2025_q3_2025_q4_2025_test_q1_2026"
PROTOCOL_SEEDS = [1, 2, 3, 4, 5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE_DIR)
    parser.add_argument("--protocol081-dir", type=Path, default=DEFAULT_PROTOCOL081_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--artifact-fold", default=DEFAULT_FOLD)
    parser.add_argument("--protocol-seeds", nargs="*", type=int, default=PROTOCOL_SEEDS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trades = pd.read_parquet(args.sequence_dir / "protocol054_lifecycle_trades.parquet")
    steps = pd.read_parquet(args.sequence_dir / "protocol054_lifecycle_steps.parquet")
    trades["trade_uid"] = trades["trade_uid"].astype(str)
    steps["trade_uid"] = steps["trade_uid"].astype(str)
    grouped_steps = {
        str(uid): frame.sort_values("step_idx").reset_index(drop=True)
        for uid, frame in steps.groupby("trade_uid", sort=False)
    }

    selected_rows: list[dict[str, Any]] = []
    seed_summaries: list[dict[str, Any]] = []
    for seed in args.protocol_seeds:
        manifest = (
            args.protocol081_dir
            / "model_artifacts"
            / args.artifact_fold
            / f"seed_{int(seed)}"
            / "manifest.json"
        )
        artifact = load_protocol066_artifact(manifest)
        seed_rows: list[dict[str, Any]] = []
        candidate_trades: list[Trade] = []
        p054_trades: list[Trade] = []
        for _, trade in trades.sort_values(["session", "seed", "decision_time", "trade_uid"]).iterrows():
            uid = str(trade["trade_uid"])
            step_frame = grouped_steps.get(uid)
            if step_frame is None or step_frame.empty:
                continue
            value, recovery, decay = predict_protocol066_sequence(artifact, step_frame)
            row = _sequence_exit_row(
                trade=trade,
                steps=step_frame,
                value=value,
                recovery=recovery,
                decay=decay,
                protocol_seed=int(seed),
                override_threshold=artifact.selected_override_threshold,
            )
            seed_rows.append(row)
            candidate_trades.append(
                Trade(
                    session=str(row["session"]),
                    decision_time=str(row["decision_time"]),
                    pnl=float(row["candidate_pnl"]),
                    score=float(row["predicted_continuation_value"]),
                    right=str(row["right"]),
                    offset=float(row["offset"]),
                    strategy=f"protocol081_q4_2024_external:seed{seed}:{row['candidate_exit_reason']}",
                )
            )
            p054_trades.append(
                Trade(
                    session=str(row["session"]),
                    decision_time=str(row["decision_time"]),
                    pnl=float(row["protocol054_pnl"]),
                    score=None,
                    right=str(row["right"]),
                    offset=float(row["offset"]),
                    strategy="protocol054_external",
                )
            )
        selected_rows.extend(seed_rows)
        metrics = _metrics(candidate_trades)
        p054_metrics = _metrics(p054_trades)
        seed_summaries.append(
            {
                "protocol_seed": int(seed),
                "rows": len(seed_rows),
                "candidate_metrics": metrics,
                "protocol054_metrics": p054_metrics,
                "delta_vs_protocol054": float(metrics["total_pnl"] - p054_metrics["total_pnl"]),
                "stress_50": _metrics(stress_trades(candidate_trades, extra_cost_per_trade=50.0)),
                "stress_100": _metrics(stress_trades(candidate_trades, extra_cost_per_trade=100.0)),
            }
        )

    selected_path = args.out_dir / "selected_trades_sequence_exits.json"
    selected_path.write_text(json.dumps(selected_rows, indent=2, sort_keys=True, allow_nan=False) + "\n")
    payload = {
        "protocol": "106_q4_2024_external_protocol081_exits",
        "paid_data_downloaded": False,
        "live_orders": False,
        "model_training": False,
        "source_sequence_dir": str(args.sequence_dir),
        "source_protocol081_dir": str(args.protocol081_dir),
        "artifact_fold": str(args.artifact_fold),
        "protocol_seeds": [int(seed) for seed in args.protocol_seeds],
        "selected_trades_file": str(selected_path),
        "selected_rows": len(selected_rows),
        "seed_summaries": seed_summaries,
        "decision": "built_q4_2024_protocol081_external_candidate_exits",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    _write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "selected_rows": len(selected_rows)}, indent=2, sort_keys=True))
    print(args.out_dir / "report.md")
    return 0


def _sequence_exit_row(
    *,
    trade: pd.Series,
    steps: pd.DataFrame,
    value: np.ndarray,
    recovery: np.ndarray,
    decay: np.ndarray,
    protocol_seed: int,
    override_threshold: float,
) -> dict[str, Any]:
    protocol054_exit_step = int(trade["protocol054_exit_step"])
    exit_idx = min(protocol054_exit_step, len(steps) - 1)
    reason = "protocol054_fallback"
    max_idx = min(protocol054_exit_step, len(steps) - 1)
    for idx in range(max_idx + 1):
        row = steps.iloc[idx]
        baseline_reason = str(row.get("baseline_exit_reason", ""))
        if bool(row.get("is_baseline_exit_step", False)) and baseline_reason in {"hard_stop", "target"}:
            exit_idx = int(idx)
            reason = baseline_reason
            break
        if (
            int(idx) < protocol054_exit_step
            and np.isfinite(override_threshold)
            and float(value[int(idx)]) > float(override_threshold) + OVERRIDE_THRESHOLD_EPSILON
        ):
            exit_idx = int(idx)
            reason = "sequence_residual_override"
            break
    exit_row = steps.iloc[exit_idx]
    p054_pnl = float(trade["protocol054_pnl"])
    pnl = float(exit_row["current_pnl"])
    return {
        "trade_uid": str(trade["trade_uid"]),
        "canonical_entry_uid": str(trade["canonical_entry_uid"]),
        "split": "q4_2024_external",
        "seed": int(protocol_seed),
        "entry_seed": int(trade["seed"]),
        "session": str(trade["session"]),
        "decision_time": str(trade["decision_time"]),
        "contract_id": str(trade["contract_id"]),
        "right": str(trade["right"]),
        "offset": float(trade["offset"]),
        "candidate_pnl": pnl,
        "protocol054_pnl": p054_pnl,
        "delta_vs_protocol054": pnl - p054_pnl,
        "candidate_exit_reason": reason,
        "candidate_exit_step": int(exit_idx),
        "candidate_exit_time": str(exit_row["quote_time"]),
        "protocol054_exit_reason": str(trade["protocol054_exit_reason"]),
        "protocol054_exit_step": protocol054_exit_step,
        "predicted_continuation_value": float(value[exit_idx]),
        "override_threshold": float(override_threshold) if np.isfinite(override_threshold) else "inf",
        "predicted_recovery_probability": float(recovery[exit_idx]),
        "predicted_decay_probability": float(decay[exit_idx]),
        "current_pnl_at_exit": pnl,
        "mfe_to_exit": float(exit_row["mfe_to_now"]),
        "mae_to_exit": float(exit_row["mae_to_now"]),
        "future_max_delta_at_exit": float(exit_row["future_max_delta"]),
        "future_min_delta_at_exit": float(exit_row["future_min_delta"]),
    }


def _metrics(trades: list[Trade]) -> dict[str, Any]:
    metrics = metrics_with_concentration(trades)
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.generic)):
            value = float(value)
            if np.isfinite(value):
                out[key] = value
            elif value > 0:
                out[key] = 999.0
            elif value < 0:
                out[key] = -999.0
            else:
                out[key] = 0.0
        else:
            out[key] = value
    return out


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol 106: Q4 2024 External Protocol 081 Exits",
        "",
        "No paid market data was downloaded. No live broker data or order endpoint was used. No model was trained.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Selected rows: `{payload['selected_rows']}`",
        f"- Output: `{payload['selected_trades_file']}`",
        "",
        "## Seed Summary",
        "",
        "| protocol_seed | rows | sequence_pnl | protocol054_pnl | delta | pf | +50 | +100 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["seed_summaries"]:
        metrics = row["candidate_metrics"]
        p054 = row["protocol054_metrics"]
        stress50 = row["stress_50"]
        stress100 = row["stress_100"]
        lines.append(
            f"| {row['protocol_seed']} | {row['rows']} | {metrics['total_pnl']:.0f} | "
            f"{p054['total_pnl']:.0f} | {row['delta_vs_protocol054']:.0f} | "
            f"{metrics['profit_factor']:.3f} | {stress50['total_pnl']:.0f} | {stress100['total_pnl']:.0f} |"
        )
    lines += [
        "",
        "## Next",
        "",
        "Build a Protocol 092-compatible candidate table for `q4_2024_external`, then score frozen Protocol 101 artifacts as a temporal-regime stress audit.",
    ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
