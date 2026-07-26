"""Run gate-only and random-in-gate null baselines for Protocol101 fair rows.

This is an offline diagnostic requested after the fair-contract synchronization
review. It does not train, tune thresholds, touch broker endpoints, download
data, change defaults, or promote a model.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from v4.model.supervised_pilot import (
    SELECTION_MODE_STABLE_ABS_OFFSET_20,
    DecisionCandidates,
    Trade,
    entry_filter_mask,
    load_decisions,
    metrics_for_trades,
    simulate_model_policy,
    top_prediction,
)
from v4.scripts.run_protocol101_fair_contract_training_runner import (
    POLICY_META,
    load_json,
    paths_by_split,
)


DEFAULT_DESIGN = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_expanded_jul_dec2025_128_q1_design/summary.json"
)
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_gate_null_baselines"
)
PRIMARY_ENTRY_FILTER = "put_near_after_0940_vwap_m2_10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--policy-index", type=int, choices=sorted(POLICY_META), default=0)
    parser.add_argument(
        "--entry-filter",
        action="append",
        default=None,
        help=(
            "Gate to evaluate. May be repeated. Defaults to the primary broad "
            "base gate put_near_after_0940_vwap_m2_10."
        ),
    )
    parser.add_argument("--selection-mode", default=SELECTION_MODE_STABLE_ABS_OFFSET_20)
    parser.add_argument("--max-trades-per-session", type=int, default=3)
    parser.add_argument("--max-daily-loss", type=float, default=500.0)
    parser.add_argument("--starting-cash", type=float, default=10_000.0)
    parser.add_argument("--stress-per-trade", type=float, default=20.0)
    parser.add_argument("--null-seeds", type=int, default=1000)
    parser.add_argument("--null-seed-start", type=int, default=0)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def stress_trades(trades: Sequence[Trade], *, stress_per_trade: float) -> list[Trade]:
    if stress_per_trade <= 0.0:
        return list(trades)
    return [replace(trade, pnl=float(trade.pnl) - float(stress_per_trade)) for trade in trades]


def prediction_ones(decisions: Sequence[DecisionCandidates]) -> list[np.ndarray]:
    return [np.ones(len(decision.labels), dtype=np.float32) for decision in decisions]


def prediction_for_selected_keys(
    decisions: Sequence[DecisionCandidates],
    selected_keys: set[tuple[str, str]],
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for decision in decisions:
        key = (decision.session, decision.decision_time.isoformat())
        if key in selected_keys:
            out.append(np.ones(len(decision.labels), dtype=np.float32))
        else:
            out.append(np.full(len(decision.labels), -np.inf, dtype=np.float32))
    return out


def trade_session_counts(trades: Sequence[Trade]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for trade in trades:
        counts[str(trade.session)] += 1
    return dict(counts)


def trade_rows(trades: Sequence[Trade], *, stress_per_trade: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trade in sorted(trades, key=lambda item: (item.session, item.decision_time, item.strategy)):
        rows.append(
            {
                "session": trade.session,
                "decision_time": trade.decision_time,
                "strategy": trade.strategy,
                "right": trade.right,
                "offset": float(trade.offset),
                "score": trade.score,
                "raw_pnl": float(trade.pnl),
                "stressed_pnl": float(trade.pnl) - float(stress_per_trade),
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def simulate_gate_only(
    decisions: Sequence[DecisionCandidates],
    *,
    entry_filter: str,
    selection_mode: str,
    cooldown_minutes: int,
    max_trades_per_session: int,
    max_daily_loss: float,
    starting_cash: float,
    stress_per_trade: float,
    strategy: str,
) -> list[Trade]:
    return simulate_model_policy(
        decisions,
        prediction_ones(decisions),
        threshold=0.0,
        cooldown_minutes=cooldown_minutes,
        strategy=strategy,
        entry_filter=entry_filter,
        min_score_margin=0.0,
        max_score_ceiling=0.0,
        max_trades_per_session=max_trades_per_session,
        max_daily_loss=max_daily_loss,
        selection_mode=selection_mode,
        starting_cash=starting_cash,
        cash_pnl_adjustment=-float(stress_per_trade),
    )


def decision_key(decision: DecisionCandidates) -> tuple[str, str]:
    return (str(decision.session), decision.decision_time.isoformat())


def eligible_decisions(
    decisions: Sequence[DecisionCandidates],
    *,
    entry_filter: str,
    selection_mode: str,
) -> list[DecisionCandidates]:
    eligible: list[DecisionCandidates] = []
    for decision in decisions:
        pred = np.ones(len(decision.labels), dtype=np.float32)
        if top_prediction(
            decision,
            pred,
            entry_filter=entry_filter,
            selection_mode=selection_mode,
        ) is not None:
            eligible.append(decision)
    return eligible


def eligible_grouped_decisions(
    decisions: Sequence[DecisionCandidates],
    *,
    entry_filter: str,
    selection_mode: str,
) -> dict[str, list[DecisionCandidates]]:
    grouped: dict[str, list[DecisionCandidates]] = defaultdict(list)
    for decision in eligible_decisions(
        decisions,
        entry_filter=entry_filter,
        selection_mode=selection_mode,
    ):
        grouped[decision.session].append(decision)
    return grouped


def greedy_random_keys_for_session(
    session_decisions: Sequence[DecisionCandidates],
    *,
    rng: np.random.Generator,
    cooldown_minutes: int,
    max_count: int,
) -> set[tuple[str, str]]:
    if max_count <= 0:
        return set()
    shuffled = list(session_decisions)
    rng.shuffle(shuffled)
    accepted: list[DecisionCandidates] = []
    cooldown = timedelta(minutes=cooldown_minutes)
    for decision in shuffled:
        if len(accepted) >= max_count:
            break
        if all(abs(decision.decision_time - prior.decision_time) >= cooldown for prior in accepted):
            accepted.append(decision)
    return {decision_key(decision) for decision in accepted}


def random_in_gate_trades(
    decisions: Sequence[DecisionCandidates],
    *,
    grouped_eligible: dict[str, list[DecisionCandidates]],
    entry_filter: str,
    selection_mode: str,
    cooldown_minutes: int,
    max_trades_per_session: int,
    max_daily_loss: float,
    starting_cash: float,
    stress_per_trade: float,
    seed: int,
    match_counts: dict[str, int] | None,
    strategy: str,
) -> list[Trade]:
    rng = np.random.default_rng(seed)
    selected: set[tuple[str, str]] = set()
    for session, session_decisions in grouped_eligible.items():
        limit = int(max_trades_per_session) if max_trades_per_session > 0 else len(session_decisions)
        if match_counts is not None:
            limit = min(limit, int(match_counts.get(session, 0)))
        selected.update(
            greedy_random_keys_for_session(
                session_decisions,
                rng=rng,
                cooldown_minutes=cooldown_minutes,
                max_count=limit,
            )
        )

    return simulate_model_policy(
        decisions,
        prediction_for_selected_keys(decisions, selected),
        threshold=0.0,
        cooldown_minutes=cooldown_minutes,
        strategy=strategy,
        entry_filter=entry_filter,
        min_score_margin=0.0,
        max_score_ceiling=0.0,
        max_trades_per_session=max_trades_per_session,
        max_daily_loss=max_daily_loss,
        selection_mode=selection_mode,
        starting_cash=starting_cash,
        cash_pnl_adjustment=-float(stress_per_trade),
    )


def percentile_summary(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {key: 0.0 for key in ("mean", "std", "p5", "p25", "p50", "p75", "p95", "p99")}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def empirical_p_value(*, candidate_value: float, null_values: Sequence[float]) -> float:
    arr = np.asarray(null_values, dtype=float)
    if arr.size == 0:
        return 1.0
    return float((1 + np.sum(arr >= float(candidate_value))) / (1 + arr.size))


def null_result(
    decisions: Sequence[DecisionCandidates],
    *,
    grouped_eligible: dict[str, list[DecisionCandidates]],
    entry_filter: str,
    selection_mode: str,
    cooldown_minutes: int,
    max_trades_per_session: int,
    max_daily_loss: float,
    starting_cash: float,
    stress_per_trade: float,
    seed_start: int,
    null_seeds: int,
    match_counts: dict[str, int] | None,
    strategy_prefix: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total_pnls: list[float] = []
    profit_factors: list[float] = []
    trades_counts: list[float] = []
    avg_pnls: list[float] = []
    for offset in range(null_seeds):
        seed = int(seed_start + offset)
        trades = random_in_gate_trades(
            decisions,
            grouped_eligible=grouped_eligible,
            entry_filter=entry_filter,
            selection_mode=selection_mode,
            cooldown_minutes=cooldown_minutes,
            max_trades_per_session=max_trades_per_session,
            max_daily_loss=max_daily_loss,
            starting_cash=starting_cash,
            stress_per_trade=stress_per_trade,
            seed=seed,
            match_counts=match_counts,
            strategy=f"{strategy_prefix}_seed{seed}",
        )
        stressed_metrics = metrics_for_trades(
            stress_trades(trades, stress_per_trade=stress_per_trade)
        )
        total_pnls.append(float(stressed_metrics["total_pnl"]))
        profit_factors.append(float(stressed_metrics["profit_factor"]))
        trades_counts.append(float(stressed_metrics["trades"]))
        avg_pnls.append(float(stressed_metrics["avg_pnl"]))
        rows.append(
            {
                "seed": seed,
                "trades": int(stressed_metrics["trades"]),
                "stressed_total_pnl": float(stressed_metrics["total_pnl"]),
                "stressed_avg_pnl": float(stressed_metrics["avg_pnl"]),
                "stressed_profit_factor": float(stressed_metrics["profit_factor"]),
                "stressed_max_drawdown": float(stressed_metrics["max_drawdown"]),
            }
        )
    return {
        "runs": int(null_seeds),
        "rows": rows,
        "stressed_total_pnl_distribution": percentile_summary(total_pnls),
        "stressed_profit_factor_distribution": percentile_summary(profit_factors),
        "trades_distribution": percentile_summary(trades_counts),
        "stressed_avg_pnl_distribution": percentile_summary(avg_pnls),
        "_stressed_total_pnl_values": total_pnls,
    }


def parallel_label_values(
    decisions: Sequence[DecisionCandidates],
    *,
    entry_filter: str,
    selection_mode: str,
) -> dict[str, Any]:
    values: list[float] = []
    no_label = 0
    for decision in decisions:
        pred = np.ones(len(decision.labels), dtype=np.float32)
        top = top_prediction(
            decision,
            pred,
            entry_filter=entry_filter,
            selection_mode=selection_mode,
        )
        if top is None:
            no_label += 1
            continue
        idx, _score, _margin = top
        values.append(float(decision.labels[idx]))
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {
            "gated_minutes_with_selected_label": 0,
            "no_label_minutes": int(no_label),
            "mean": 0.0,
            "winsor600_mean": 0.0,
            "trim10_mean": 0.0,
            "median": 0.0,
            "share_positive": 0.0,
            "sum": 0.0,
        }
    winsor = np.clip(arr, -600.0, 600.0)
    sorted_arr = np.sort(arr)
    trim = sorted_arr
    trim_n = int(math.floor(0.10 * len(sorted_arr)))
    if len(sorted_arr) > 2 * trim_n and trim_n > 0:
        trim = sorted_arr[trim_n:-trim_n]
    return {
        "gated_minutes_with_selected_label": int(arr.size),
        "no_label_minutes": int(no_label),
        "mean": float(np.mean(arr)),
        "winsor600_mean": float(np.mean(winsor)),
        "trim10_mean": float(np.mean(trim)) if len(trim) else 0.0,
        "median": float(np.median(arr)),
        "share_positive": float(np.mean(arr > 0.0)),
        "sum": float(np.sum(arr)),
    }


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Protocol101 Gate-Only And Null Baseline Diagnostics",
        "",
        f"Generated: {summary['generated_at_utc']}",
        "",
        "This is an offline diagnostic. It did not train, tune thresholds, contact brokers/vendors, "
        "download data, change defaults, promote a model, or enable paper-submit.",
        "",
        "## Configuration",
        "",
        f"- Design: `{summary['design']}`",
        f"- Feature contract: `{summary['selected_feature_contract']}`",
        f"- Policy: `{summary['policy_name']}`",
        f"- Selection mode: `{summary['selection_mode']}`",
        f"- Stress per trade: `${summary['stress_per_trade']:.2f}`",
        f"- Null seeds: `{summary['null_seeds']}`",
        "",
        "## Results",
        "",
    ]
    for filter_name, filter_payload in summary["entry_filters"].items():
        lines.extend([f"### `{filter_name}`", ""])
        lines.append("| Split | Gate Trades | Gate Stressed PnL | Gate PF | Null-U p95 | Null-M p95 | Gate p vs Null-U | Gate p vs Null-M |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for split, split_payload in filter_payload["splits"].items():
            gate = split_payload["gate_only"]["stressed_metrics"]
            null_u = split_payload["null_u"]
            null_m = split_payload["null_m"]
            lines.append(
                "| {split} | {trades} | ${pnl:,.0f} | {pf:.3f} | ${u95:,.0f} | ${m95:,.0f} | {pu:.4f} | {pm:.4f} |".format(
                    split=split,
                    trades=int(gate["trades"]),
                    pnl=float(gate["total_pnl"]),
                    pf=float(gate["profit_factor"]),
                    u95=float(null_u["stressed_total_pnl_distribution"]["p95"]),
                    m95=float(null_m["stressed_total_pnl_distribution"]["p95"]),
                    pu=float(split_payload["gate_only"]["p_value_vs_null_u"]),
                    pm=float(split_payload["gate_only"]["p_value_vs_null_m"]),
                )
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "Gate-only is a possible candidate shape only if it is profitable after stress and beats the "
            "random-in-gate null distribution. A weak or null-like result means the current pocket should "
            "not be treated as proven edge, even if oracle labels look positive.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    design = load_json(args.design)
    manifest = load_json(Path(design["allowed_data"]["canonical_manifest"]))
    paths, blockers = paths_by_split(design, manifest)
    if blockers:
        raise SystemExit(f"split path blockers: {blockers}")

    policy_name, cooldown_minutes = POLICY_META[int(args.policy_index)]
    loaded = {
        split: load_decisions(split_paths, policy_index=int(args.policy_index))
        for split, split_paths in paths.items()
        if split in {"validation", "diagnostic_test"}
    }

    entry_filters = args.entry_filter or [PRIMARY_ENTRY_FILTER]
    summary: dict[str, Any] = {
        "schema_version": "Protocol101GateNullBaselinesV1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "design": str(args.design),
        "selected_feature_contract": design.get("selected_feature_contract"),
        "policy_index": int(args.policy_index),
        "policy_name": policy_name,
        "cooldown_minutes": int(cooldown_minutes),
        "selection_mode": str(args.selection_mode),
        "max_trades_per_session": int(args.max_trades_per_session),
        "max_daily_loss": float(args.max_daily_loss),
        "starting_cash": float(args.starting_cash),
        "stress_per_trade": float(args.stress_per_trade),
        "null_seeds": int(args.null_seeds),
        "null_seed_start": int(args.null_seed_start),
        "broker_endpoint_called": False,
        "paper_submit_allowed": False,
        "model_training_executed": False,
        "threshold_selection_executed": False,
        "paid_data_downloaded": False,
        "recorder_days_used_for_selection": False,
        "evidence_grade": "provisional_current_march_splits_only",
        "entry_filters": {},
    }

    for entry_filter in entry_filters:
        filter_payload: dict[str, Any] = {"splits": {}}
        for split, decisions in loaded.items():
            split_out_dir = out_dir / entry_filter / split
            grouped_eligible = eligible_grouped_decisions(
                decisions,
                entry_filter=entry_filter,
                selection_mode=str(args.selection_mode),
            )
            gate_trades = simulate_gate_only(
                decisions,
                entry_filter=entry_filter,
                selection_mode=str(args.selection_mode),
                cooldown_minutes=cooldown_minutes,
                max_trades_per_session=int(args.max_trades_per_session),
                max_daily_loss=float(args.max_daily_loss),
                starting_cash=float(args.starting_cash),
                stress_per_trade=float(args.stress_per_trade),
                strategy=f"gate_only_{entry_filter}",
            )
            gate_stressed = metrics_for_trades(
                stress_trades(gate_trades, stress_per_trade=float(args.stress_per_trade))
            )
            gate_counts = trade_session_counts(gate_trades)
            null_u = null_result(
                decisions,
                grouped_eligible=grouped_eligible,
                entry_filter=entry_filter,
                selection_mode=str(args.selection_mode),
                cooldown_minutes=cooldown_minutes,
                max_trades_per_session=int(args.max_trades_per_session),
                max_daily_loss=float(args.max_daily_loss),
                starting_cash=float(args.starting_cash),
                stress_per_trade=float(args.stress_per_trade),
                seed_start=int(args.null_seed_start),
                null_seeds=int(args.null_seeds),
                match_counts=None,
                strategy_prefix=f"null_u_{entry_filter}",
            )
            null_m = null_result(
                decisions,
                grouped_eligible=grouped_eligible,
                entry_filter=entry_filter,
                selection_mode=str(args.selection_mode),
                cooldown_minutes=cooldown_minutes,
                max_trades_per_session=int(args.max_trades_per_session),
                max_daily_loss=float(args.max_daily_loss),
                starting_cash=float(args.starting_cash),
                stress_per_trade=float(args.stress_per_trade),
                seed_start=int(args.null_seed_start),
                null_seeds=int(args.null_seeds),
                match_counts=gate_counts,
                strategy_prefix=f"null_m_{entry_filter}",
            )
            null_u_rows = null_u.pop("rows")
            null_m_rows = null_m.pop("rows")
            null_u_pnl_values = null_u.pop("_stressed_total_pnl_values")
            null_m_pnl_values = null_m.pop("_stressed_total_pnl_values")
            write_csv(split_out_dir / "gate_only_trades.csv", trade_rows(gate_trades, stress_per_trade=float(args.stress_per_trade)))
            write_csv(split_out_dir / "null_u_runs.csv", null_u_rows)
            write_csv(split_out_dir / "null_m_runs.csv", null_m_rows)
            filter_payload["splits"][split] = {
                "decision_count": int(len(decisions)),
                "candidate_count": int(sum(len(decision.labels) for decision in decisions)),
                "parallel_label_values": parallel_label_values(
                    decisions,
                    entry_filter=entry_filter,
                    selection_mode=str(args.selection_mode),
                ),
                "gate_only": {
                    "raw_metrics": metrics_for_trades(gate_trades),
                    "stressed_metrics": gate_stressed,
                    "session_trade_counts": gate_counts,
                    "p_value_vs_null_u": empirical_p_value(
                        candidate_value=float(gate_stressed["total_pnl"]),
                        null_values=null_u_pnl_values,
                    ),
                    "p_value_vs_null_m": empirical_p_value(
                        candidate_value=float(gate_stressed["total_pnl"]),
                        null_values=null_m_pnl_values,
                    ),
                    "trades_csv": str(split_out_dir / "gate_only_trades.csv"),
                },
                "null_u": {
                    **null_u,
                    "runs_csv": str(split_out_dir / "null_u_runs.csv"),
                },
                "null_m": {
                    **null_m,
                    "runs_csv": str(split_out_dir / "null_m_runs.csv"),
                },
            }
        summary["entry_filters"][entry_filter] = filter_payload

    write_json(out_dir / "summary.json", summary)
    (out_dir / "report.md").write_text(build_report(summary) + "\n")


if __name__ == "__main__":
    main()
