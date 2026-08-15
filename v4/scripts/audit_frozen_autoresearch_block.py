"""Audit frozen autoresearch champions on a new out-of-sample data block."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from v4.model.action_pilot import (
    ACTION_DECISION_LOSS_CONFIG,
    action_feature_version,
    load_action_decisions,
    predict_actions,
    train_action_model,
)
from v4.model.supervised_pilot import PilotConfig, session_from_path, split_name
from v4.scripts.evaluate_calibrated_abstention_signal import split_validation_by_session
from v4.scripts.evaluate_risk_controlled_purchase_signal import metrics_with_concentration
from v4.scripts.run_autoresearch_loop import AutoresearchTrial, simulate_true_no_trade_policy
from v4.scripts.train_spxw_supervised_pilot import POLICY_META


@dataclass(frozen=True)
class FrozenChampion:
    name: str
    source_report: str
    policy_index: int
    loss_mode: str
    trial: AutoresearchTrial


FROZEN_CHAMPIONS = (
    FrozenChampion(
        name="huber_q_autoresearch_001_champion",
        source_report="v4/audit/autoresearch/v4_autoresearch_001/report.md",
        policy_index=1,
        loss_mode="huber",
        trial=AutoresearchTrial(
            name="skip_first30_edge0_max4_stop1000",
            time_filter="skip_first_30",
            allowed_buckets=("post_open_morning", "midday", "late_afternoon"),
            min_edge_vs_no_trade=0.0,
            max_trades_per_day=4,
            daily_loss_stop=-1_000.0,
        ),
    ),
    FrozenChampion(
        name="decision_aware_v1_champion",
        source_report="v4/audit/autoresearch/v4_autoresearch_001_decision_loss/report.md",
        policy_index=1,
        loss_mode="decision_aware",
        trial=AutoresearchTrial(
            name="post_open_late_edge25_max4",
            time_filter="post_open_and_late",
            allowed_buckets=("post_open_morning", "late_afternoon"),
            min_edge_vs_no_trade=25.0,
            max_trades_per_day=4,
            daily_loss_stop=None,
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_derived"))
    p.add_argument("--audit-data-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seeds", nargs="*", type=int, default=[11, 22, 33])
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4096)
    return p.parse_args()


def _q1_paths_by_split(data_dir: Path) -> dict[str, list[Path]]:
    out = {"train": [], "validation": [], "test": []}
    for path in sorted(data_dir.glob("*.pkl")):
        out[split_name(session_from_path(path))].append(path)
    return out


def _audit_paths(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("*.pkl"))


def _aggregate_metrics(metrics: Sequence[dict]) -> dict:
    arr = {
        "total_pnl": np.asarray([m["total_pnl"] for m in metrics], dtype=float),
        "profit_factor": np.asarray([m["profit_factor"] for m in metrics], dtype=float),
        "max_drawdown": np.asarray([m["max_drawdown"] for m in metrics], dtype=float),
        "trades": np.asarray([m["trades"] for m in metrics], dtype=float),
        "positive_day_fraction": np.asarray([m["positive_day_fraction"] for m in metrics], dtype=float),
        "top_day_profit_share": np.asarray([m["top_day_profit_share"] for m in metrics], dtype=float),
    }
    return {
        "runs": len(metrics),
        "pnl_median": float(np.median(arr["total_pnl"])),
        "pnl_mean": float(arr["total_pnl"].mean()),
        "profit_factor_median": float(np.median(arr["profit_factor"])),
        "max_drawdown_median": float(np.median(arr["max_drawdown"])),
        "trades_median": float(np.median(arr["trades"])),
        "positive_seed_fraction": float((arr["total_pnl"] > 0).mean()),
        "positive_day_fraction_median": float(np.median(arr["positive_day_fraction"])),
        "top_day_profit_share_median": float(np.median(arr["top_day_profit_share"])),
    }


def _run_one(
    *,
    champion: FrozenChampion,
    train_paths: dict[str, list[Path]],
    audit_paths: Sequence[Path],
    seed: int,
    epochs: int,
    batch_size: int,
) -> dict:
    policy_name, cooldown = POLICY_META[champion.policy_index]
    config = PilotConfig(
        policy_index=champion.policy_index,
        policy_name=policy_name,
        cooldown_minutes=cooldown,
        epochs=epochs,
        batch_size=batch_size,
        hidden_dim=128,
        seed=seed,
    )
    q1_train = load_action_decisions(train_paths["train"], policy_index=champion.policy_index)
    q1_validation = load_action_decisions(
        train_paths["validation"], policy_index=champion.policy_index
    )
    calibration_decisions, _ = split_validation_by_session(q1_validation)
    audit_decisions = load_action_decisions(audit_paths, policy_index=champion.policy_index)
    model, scaler, history = train_action_model(
        q1_train,
        calibration_decisions,
        config=config,
        loss_mode=champion.loss_mode,
    )
    predictions = predict_actions(
        model,
        scaler,
        audit_decisions,
        target_scale=config.target_scale,
    )
    trades = simulate_true_no_trade_policy(
        audit_decisions,
        predictions,
        trial=champion.trial,
        cooldown_minutes=cooldown,
        strategy=f"frozen_audit:{champion.name}",
    )
    metrics = metrics_with_concentration(trades)
    return {
        "champion": champion.name,
        "seed": seed,
        "policy_index": champion.policy_index,
        "policy_name": policy_name,
        "loss_mode": champion.loss_mode,
        "feature_version": action_feature_version(champion.loss_mode),
        "trial": asdict(champion.trial) | {"config_id": champion.trial.config_id},
        "best_epoch": next((x["epoch"] for x in history if x["is_best"]), None),
        "audit_decisions": len(audit_decisions),
        "audit_metrics": metrics,
    }


def main() -> int:
    args = parse_args()
    train_paths = _q1_paths_by_split(args.train_data_dir)
    audit_paths = _audit_paths(args.audit_data_dir)
    if not audit_paths:
        raise SystemExit(f"no audit pkl files found under {args.audit_data_dir}")

    runs = []
    for champion in FROZEN_CHAMPIONS:
        for seed in args.seeds:
            print(f"frozen audit champion={champion.name} seed={seed}", flush=True)
            runs.append(
                _run_one(
                    champion=champion,
                    train_paths=train_paths,
                    audit_paths=audit_paths,
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                )
            )

    by_champion = {}
    for champion in FROZEN_CHAMPIONS:
        rows = [r for r in runs if r["champion"] == champion.name]
        by_champion[champion.name] = {
            "source_report": champion.source_report,
            "policy_index": champion.policy_index,
            "policy_name": POLICY_META[champion.policy_index][0],
            "loss_mode": champion.loss_mode,
            "feature_version": action_feature_version(champion.loss_mode),
            "loss_config": ACTION_DECISION_LOSS_CONFIG
            if champion.loss_mode == "decision_aware"
            else None,
            "trial": asdict(champion.trial) | {"config_id": champion.trial.config_id},
            "summary": _aggregate_metrics([r["audit_metrics"] for r in rows]),
        }

    payload = {
        "framing": (
            "Frozen out-of-sample audit on a newly downloaded Q4 2025 block. "
            "Champions, loss modes, policies, and risk controls were fixed from "
            "earlier Jan-Mar 2026 research reports before scoring Q4. Q4 metrics "
            "are audit-only and must not be used to retune the trial surface."
        ),
        "train_data_dir": str(args.train_data_dir),
        "audit_data_dir": str(args.audit_data_dir),
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "audit_files": len(audit_paths),
        "by_champion": by_champion,
        "runs": runs,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "report.json"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")

    lines = [
        "# Frozen Q4 2025 Autoresearch Audit",
        "",
        payload["framing"],
        "",
        f"Audit files: `{len(audit_paths)}`",
        "",
        "| Champion | Objective | Trial | Q4 PnL | Q4 PF | Q4 DD | Trades | Positive Seeds | Positive Days | Top-Day Share |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in by_champion.items():
        s = row["summary"]
        lines.append(
            f"| {name} | {row['feature_version']} | {row['trial']['name']} | "
            f"{s['pnl_median']:.0f} | {s['profit_factor_median']:.3f} | "
            f"{s['max_drawdown_median']:.0f} | {s['trades_median']:.0f} | "
            f"{s['positive_seed_fraction']:.2f} | {s['positive_day_fraction_median']:.2f} | "
            f"{s['top_day_profit_share_median']:.2f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        (
            "This report is a holdout survival check. A good Q4 result can justify "
            "continued modeling, but it is not a live-trading approval. A weak Q4 "
            "result means the Jan-Mar signal is not yet broad enough."
        ),
    ]
    md_path = args.out_dir / "report.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(json_path)
    print(md_path)
    print(json.dumps(by_champion, indent=2, allow_nan=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
