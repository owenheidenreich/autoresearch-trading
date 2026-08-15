"""Protocol 036: entry-side contract-quality reranking.

This keeps the Protocol 024 entry timing model frozen, then changes only the
contract selection within a decision minute. If several SPXW contracts are near
the model's preferred action score, choose the one with better contract
economics instead of skipping the trade.

No paid data is downloaded here.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np

from v4.model.environment_diagnostics import time_bucket
from v4.model.hypothesis_protocol import (
    MarketStructureCache,
    ProtocolTrial,
    SurfaceDecision,
    SurfaceVariant,
    predict_surface_actions,
    registered_aplus_surface_variants,
    registered_protocol_trials,
    selection_reward,
    stress_trades,
    summarize_random_baseline,
    token_feature_names,
    train_surface_model,
    window_seed,
)
from v4.model.supervised_pilot import PilotConfig, Trade
from v4.scripts.evaluate_calibrated_abstention_signal import split_validation_by_session
from v4.scripts.evaluate_risk_controlled_purchase_signal import metrics_with_concentration
from v4.scripts.run_aplus_neural_protocol import (
    _load_surface_decisions_cached,
    _paths_by_split,
    _protocol_window,
)
from v4.scripts.train_spxw_supervised_pilot import POLICY_META


LOOP_ID = "v4_aplus_hypothesis_036_contract_quality_rerank"
BASELINE = {
    "selection": {"pnl": 6460.0, "trades": 16.0, "stress50": 6010.0},
    "march_2026": {"pnl": 11300.0, "trades": 20.0, "stress50": 10300.0},
    "q1_2025": {"pnl": 7170.0, "trades": 106.0, "stress50": 1870.0},
    "q2_2025": {"pnl": 8270.0, "trades": 104.0, "stress50": 2870.0},
    "q3_2025": {"pnl": 13470.0, "trades": 127.0, "stress50": 7340.0},
    "q4_2025": {"pnl": 9210.0, "trades": 111.0, "stress50": 3110.0},
}


@dataclass(frozen=True)
class RerankConfig:
    name: str
    near_top_band: float
    value_weight: float
    spread_penalty: float
    theta_penalty: float
    breakeven_penalty: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_derived_nofee"))
    parser.add_argument("--q1-data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q1_2025_nofee"))
    parser.add_argument("--q2-data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q2_2025_nofee"))
    parser.add_argument("--q3-data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q3_2025_nofee"))
    parser.add_argument("--q4-data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q4_2025_nofee"))
    parser.add_argument("--seed-q4-data-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("v4/audit/autoresearch/v4_aplus_hypothesis_036_contract_quality_rerank"))
    parser.add_argument("--decision-cache-dir", type=Path, default=Path("data/cache/v4_aplus_surface_decisions_nofee"))
    parser.add_argument("--no-decision-cache", action="store_true")
    parser.add_argument("--seeds", nargs="*", type=int, default=[11, 22, 33])
    parser.add_argument("--policy-index", type=int, default=1, choices=sorted(POLICY_META))
    parser.add_argument("--variant-name", default="surface_structure_aplus_side_value_multitask")
    parser.add_argument("--trial-name", default="post_open_late_edge25_max2")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def _paths(data_dir: Path) -> list[Path]:
    paths = sorted(data_dir.glob("*.pkl"))
    if not paths:
        raise SystemExit(f"no pkl files found under {data_dir}")
    return paths


def _find_variant(name: str) -> SurfaceVariant:
    for variant in registered_aplus_surface_variants():
        if variant.name == name:
            return variant
    raise SystemExit(f"unknown A+ variant: {name}")


def _find_trial(name: str) -> ProtocolTrial:
    for trial in registered_protocol_trials():
        if trial.name == name:
            return trial
    raise SystemExit(f"unknown trial: {name}")


def _rerank_grid() -> list[RerankConfig]:
    configs = [RerankConfig("baseline_no_rerank", 0.0, 0.0, 0.0, 0.0, 0.0)]
    for band in (25.0, 50.0, 75.0, 100.0):
        for value_weight in (25.0, 50.0, 75.0):
            configs.append(
                RerankConfig(
                    name=f"band{int(band)}_value{int(value_weight)}",
                    near_top_band=band,
                    value_weight=value_weight,
                    spread_penalty=0.0,
                    theta_penalty=0.0,
                    breakeven_penalty=0.0,
                )
            )
            configs.append(
                RerankConfig(
                    name=f"band{int(band)}_value{int(value_weight)}_costpen",
                    near_top_band=band,
                    value_weight=value_weight,
                    spread_penalty=650.0,
                    theta_penalty=90.0,
                    breakeven_penalty=8.0,
                )
            )
    return configs


def _finite(value: object, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _token_values(feature_names: Sequence[str], token: np.ndarray) -> dict[str, float]:
    return {name: _finite(value) for name, value in zip(feature_names, token)}


def _quality_adjustment(values: dict[str, float], config: RerankConfig) -> float:
    return (
        config.value_weight * values.get("contract_value_score", 0.0)
        - config.spread_penalty * max(values.get("spread_tax", 0.0), 0.0)
        - config.theta_penalty * max(values.get("theta_burden_hold", 0.0), 0.0)
        - config.breakeven_penalty * max(values.get("breakeven_atr", 0.0), 0.0)
    )


def _choose_action(
    decision: SurfaceDecision,
    pred: np.ndarray,
    *,
    trial: ProtocolTrial,
    config: RerankConfig,
    feature_names: Sequence[str],
) -> tuple[int | None, float, bool]:
    action_mask = np.concatenate([[True], decision.token_mask])
    masked = np.asarray(pred, dtype=float).copy()
    masked[~action_mask] = -np.inf
    if not np.isfinite(masked).any():
        return None, 0.0, False
    best_action = int(np.nanargmax(masked))
    if best_action == 0:
        return None, 0.0, False
    flat_score = float(masked[0])
    best_edge = float(masked[best_action] - flat_score)
    if not np.isfinite(best_edge) or best_edge < trial.min_edge_vs_no_trade:
        return None, best_edge, False
    if config.near_top_band <= 0.0 or config.value_weight == 0.0:
        return best_action, best_edge, False

    candidate_actions = []
    for action in np.where(action_mask)[0]:
        if action == 0:
            continue
        if masked[action] < masked[best_action] - config.near_top_band:
            continue
        if masked[action] - flat_score < trial.min_edge_vs_no_trade:
            continue
        candidate_actions.append(int(action))
    if not candidate_actions:
        return best_action, best_edge, False
    adjusted = []
    for action in candidate_actions:
        token_idx = action - 1
        values = _token_values(feature_names, decision.token_features[token_idx])
        adjusted.append(float(masked[action]) + _quality_adjustment(values, config))
    chosen = int(candidate_actions[int(np.argmax(adjusted))])
    return chosen, float(masked[chosen] - flat_score), chosen != best_action


def _simulate(
    decisions: Sequence[SurfaceDecision],
    predictions: np.ndarray,
    *,
    trial: ProtocolTrial,
    cooldown_minutes: int,
    config: RerankConfig,
    feature_names: Sequence[str],
) -> tuple[list[Trade], list[dict]]:
    trades: list[Trade] = []
    trace: list[dict] = []
    next_time_by_session: dict[str, datetime] = {}
    trades_by_session: dict[str, int] = {}
    pnl_by_session: dict[str, float] = {}
    halted_sessions: set[str] = set()
    allowed = set(trial.allowed_buckets)
    for decision, pred in zip(decisions, predictions):
        if time_bucket(decision.decision_time) not in allowed:
            continue
        if decision.session in halted_sessions:
            continue
        if trades_by_session.get(decision.session, 0) >= trial.max_trades_per_day:
            continue
        next_time = next_time_by_session.get(decision.session)
        if next_time is not None and decision.decision_time < next_time:
            continue
        action, edge, reranked = _choose_action(
            decision,
            pred,
            trial=trial,
            config=config,
            feature_names=feature_names,
        )
        if action is None:
            continue
        token_idx = action - 1
        pnl = float(decision.labels[token_idx])
        if not np.isfinite(pnl):
            continue
        trade = Trade(
            session=decision.session,
            decision_time=decision.decision_time.isoformat(),
            pnl=pnl,
            score=edge,
            right=str(decision.rights[token_idx]),
            offset=float(decision.offsets[token_idx]),
            strategy=f"{LOOP_ID}:{config.name}",
        )
        trades.append(trade)
        trace.append(
            {
                "session": decision.session,
                "decision_time": decision.decision_time.isoformat(),
                "contract_id": str(decision.contract_ids[token_idx]),
                "right": trade.right,
                "offset": trade.offset,
                "pnl": trade.pnl,
                "score": edge,
                "reranked": bool(reranked),
            }
        )
        trades_by_session[decision.session] = trades_by_session.get(decision.session, 0) + 1
        pnl_by_session[decision.session] = pnl_by_session.get(decision.session, 0.0) + pnl
        next_time_by_session[decision.session] = decision.decision_time + timedelta(minutes=cooldown_minutes)
        if trial.daily_loss_stop is not None and pnl_by_session[decision.session] <= trial.daily_loss_stop:
            halted_sessions.add(decision.session)
    return trades, trace


def _metrics(trades: Sequence[Trade]) -> dict:
    return metrics_with_concentration(trades)


def _summary(rows: Sequence[dict], split_order: Sequence[str], metric_key: str) -> list[dict]:
    out = []
    for split in split_order:
        metrics = [row[metric_key] for row in rows if row["split"] == split]
        stress50 = [row["stress50"] for row in rows if row["split"] == split]
        stress100 = [row["stress100"] for row in rows if row["split"] == split]
        if not metrics:
            continue
        out.append(
            {
                "split": split,
                "pnl_median": float(np.median([m["total_pnl"] for m in metrics])),
                "pf_median": float(np.median([m["profit_factor"] for m in metrics])),
                "trades_median": float(np.median([m["trades"] for m in metrics])),
                "positive_seed_fraction": float(np.mean([m["total_pnl"] > 0.0 for m in metrics])),
                "stress50_pnl_median": float(np.median([m["total_pnl"] for m in stress50])),
                "stress100_pnl_median": float(np.median([m["total_pnl"] for m in stress100])),
            }
        )
    return out


def _write_report(path: Path, payload: dict) -> None:
    lines = [
        "# Protocol 036: Contract-Quality Rerank",
        "",
        payload["framing"],
        "",
        f"Selected config: `{payload['selected_config']['name']}`",
        "",
        "| Split | Candidate PnL | PF | Trades | +50 PnL | Protocol 024 PnL | Delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        base = BASELINE[row["split"]]["pnl"]
        lines.append(
            f"| {row['split']} | {row['pnl_median']:.0f} | {row['pf_median']:.3f} | "
            f"{row['trades_median']:.0f} | {row['stress50_pnl_median']:.0f} | "
            f"{base:.0f} | {row['pnl_median'] - base:.0f} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        payload["decision"],
        "",
        "This is not paper/live approval.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    variant = _find_variant(args.variant_name)
    trial = _find_trial(args.trial_name)
    policy_name, cooldown = POLICY_META[args.policy_index]
    decision_cache_dir = None if args.no_decision_cache else args.decision_cache_dir
    market_cache = MarketStructureCache()
    token_names = token_feature_names(variant.token_mode)
    train_paths = _paths_by_split(args.data_dir)
    seed_q4_dir = args.seed_q4_data_dir or args.q4_data_dir
    window = _protocol_window(train_paths, _paths(seed_q4_dir))
    train_decisions = _load_surface_decisions_cached(
        train_paths["train"],
        policy_index=args.policy_index,
        variant=variant,
        market_cache=market_cache,
        split="train",
        cache_dir=decision_cache_dir,
    )
    validation_decisions = _load_surface_decisions_cached(
        train_paths["validation"],
        policy_index=args.policy_index,
        variant=variant,
        market_cache=market_cache,
        split="validation",
        cache_dir=decision_cache_dir,
    )
    calibration_decisions, selection_decisions = split_validation_by_session(validation_decisions)
    decision_sets = {
        "selection": selection_decisions,
        "march_2026": _load_surface_decisions_cached(
            train_paths["test"],
            policy_index=args.policy_index,
            variant=variant,
            market_cache=market_cache,
            split="march",
            cache_dir=decision_cache_dir,
        ),
        "q1_2025": _load_surface_decisions_cached(
            _paths(args.q1_data_dir),
            policy_index=args.policy_index,
            variant=variant,
            market_cache=market_cache,
            split="q1_2025",
            cache_dir=decision_cache_dir,
        ),
        "q2_2025": _load_surface_decisions_cached(
            _paths(args.q2_data_dir),
            policy_index=args.policy_index,
            variant=variant,
            market_cache=market_cache,
            split="q2_2025",
            cache_dir=decision_cache_dir,
        ),
        "q3_2025": _load_surface_decisions_cached(
            _paths(args.q3_data_dir),
            policy_index=args.policy_index,
            variant=variant,
            market_cache=market_cache,
            split="q3_2025",
            cache_dir=decision_cache_dir,
        ),
        "q4_2025": _load_surface_decisions_cached(
            _paths(args.q4_data_dir),
            policy_index=args.policy_index,
            variant=variant,
            market_cache=market_cache,
            split="q4_2025",
            cache_dir=decision_cache_dir,
        ),
    }
    grid = _rerank_grid()
    config_scores: dict[str, list[float]] = {config.name: [] for config in grid}
    config_by_name = {config.name: config for config in grid}
    artifacts = []
    for seed in args.seeds:
        effective_seed = window_seed(seed, window.window_id)
        print(f"{LOOP_ID} seed={seed}", flush=True)
        config = PilotConfig(
            policy_index=args.policy_index,
            policy_name=policy_name,
            cooldown_minutes=cooldown,
            epochs=args.epochs,
            batch_size=args.batch_size,
            hidden_dim=128,
            seed=effective_seed,
        )
        model, standardizer, history = train_surface_model(
            train_decisions,
            calibration_decisions,
            config=config,
            variant=variant,
        )
        predictions = {
            split: predict_surface_actions(model, standardizer, decisions, target_scale=config.target_scale)
            for split, decisions in decision_sets.items()
        }
        for rerank_config in grid:
            trades, _ = _simulate(
                selection_decisions,
                predictions["selection"],
                trial=trial,
                cooldown_minutes=cooldown,
                config=rerank_config,
                feature_names=token_names,
            )
            config_scores[rerank_config.name].append(selection_reward(_metrics(trades)))
        artifacts.append(
            {
                "seed": seed,
                "effective_seed": effective_seed,
                "history": history,
                "predictions": predictions,
            }
        )
    selected_name = max(config_scores, key=lambda name: float(np.median(config_scores[name])))
    selected_config = config_by_name[selected_name]
    print(f"{LOOP_ID} selected {selected_name}", flush=True)

    split_order = ["selection", "march_2026", "q1_2025", "q2_2025", "q3_2025", "q4_2025"]
    rows = []
    selected_trades: list[dict] = []
    for artifact in artifacts:
        for split in split_order:
            trades, trace = _simulate(
                decision_sets[split],
                artifact["predictions"][split],
                trial=trial,
                cooldown_minutes=cooldown,
                config=selected_config,
                feature_names=token_names,
            )
            for row in trace:
                selected_trades.append({"seed": artifact["seed"], "split": split, **row})
            rows.append(
                {
                    "seed": artifact["seed"],
                    "split": split,
                    "metrics": _metrics(trades),
                    "stress50": _metrics(stress_trades(trades, extra_cost_per_trade=50.0)),
                    "stress100": _metrics(stress_trades(trades, extra_cost_per_trade=100.0)),
                    "random_baseline": summarize_random_baseline(
                        decision_sets[split],
                        trial=trial,
                        cooldown_minutes=cooldown,
                        seed=artifact["effective_seed"],
                        target_trade_count=len(trades),
                    ),
                }
            )
    summary = _summary(rows, split_order, "metrics")
    lookup = {row["split"]: row for row in summary}
    beats_baseline = all(
        lookup[split]["pnl_median"] > BASELINE[split]["pnl"]
        for split in ("selection", "march_2026", "q1_2025", "q2_2025", "q3_2025", "q4_2025")
    )
    survives = all(
        lookup[split]["pnl_median"] > 0.0
        and lookup[split]["stress50_pnl_median"] > 0.0
        and lookup[split]["positive_seed_fraction"] >= 2 / 3
        for split in ("march_2026", "q1_2025", "q2_2025", "q3_2025", "q4_2025")
    )
    decision = "Reject Protocol 036 as a replacement for Protocol 024."
    if beats_baseline and survives:
        decision = "Keep Protocol 036 as the new research baseline candidate; it beats Protocol 024 on all locked splits and survives +50 stress."
    payload = {
        "loop_id": LOOP_ID,
        "framing": (
            "Freeze Protocol 024 entry timing and rerank only the contract choice among near-tied actions "
            "using contract value, spread, theta, and breakeven economics. This tests entry-side contract "
            "quality without filtering away the trade opportunity."
        ),
        "pre_registration": [
            "No paid data.",
            "Do not alter Protocol 024 variant, policy, trial, seeds, labels, or split protocol.",
            "Select one rerank config on February selection only.",
            "Score March 2026 and Q1/Q2/Q3/Q4 2025 once with the selected config.",
        ],
        "baseline": BASELINE,
        "selected_config": asdict(selected_config),
        "grid": [asdict(config) for config in grid],
        "selection_scores": {
            name: {"seed_rewards": scores, "median_reward": float(np.median(scores))}
            for name, scores in config_scores.items()
        },
        "summary": summary,
        "seed_rows": rows,
        "selected_trades": selected_trades,
        "decision": decision,
        "beats_baseline": bool(beats_baseline),
        "survives": bool(survives),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    _write_report(args.out_dir / "report.md", payload)
    print(args.out_dir / "report.json")
    print(args.out_dir / "report.md")
    print(decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
