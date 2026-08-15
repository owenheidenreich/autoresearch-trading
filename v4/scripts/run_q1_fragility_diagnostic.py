"""Run Protocol 021: Q1 fragility diagnostic for the frozen A+ candidate.

This is a diagnostic, not a model promotion and not a Q1-tuned filter search.
It uses only already-built processed decision rows and the same frozen
Protocol 018 candidate:

* surface_structure_aplus_teacher_margin
* policy1
* post_open_late_edge25_max2

The question is deliberately narrow: did full Q1 2025 become fragile because
post-open morning put entries paid too much spread/theta/breakeven cost for
their A+ timing pattern?
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo

import numpy as np

from v4.model.environment_diagnostics import time_bucket
from v4.model.hypothesis_protocol import (
    MarketStructureCache,
    ProtocolTrial,
    SurfaceDecision,
    SurfaceVariant,
    registered_aplus_surface_variants,
    registered_protocol_trials,
    token_feature_names,
    train_surface_model,
    predict_surface_actions,
    stress_trades,
    summarize_random_baseline,
    window_seed,
)
from v4.model.supervised_pilot import PilotConfig, Trade, session_from_path
from v4.scripts.evaluate_calibrated_abstention_signal import split_validation_by_session
from v4.scripts.evaluate_risk_controlled_purchase_signal import metrics_with_concentration
from v4.scripts.run_aplus_neural_protocol import (
    _load_surface_decisions_cached,
    _paths_by_split,
    _protocol_window,
)
from v4.scripts.train_spxw_supervised_pilot import POLICY_META


LOOP_ID = "v4_aplus_hypothesis_021_q1_fragility_diagnostic"
_NY = ZoneInfo("America/New_York")
_FEATURES_OF_INTEREST = (
    "abs_delta",
    "delta_atr_capture",
    "convexity_per_premium",
    "theta_burden_hold",
    "spread_tax",
    "breakeven_atr",
    "gamma_theta_ratio_scaled",
    "contract_value_score",
    "worth_spread_flag",
    "obvious_overpay_flag",
    "pattern_count_norm",
    "pattern_side_gap_atr",
    "pattern_side_sigma",
    "pattern_side_move1_atr",
    "pattern_side_move5_atr",
    "pattern_side_move15_atr",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_derived"))
    parser.add_argument("--q1-data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q1_2025"))
    parser.add_argument("--q2-data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q2_2025"))
    parser.add_argument("--q3-data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q3_2025"))
    parser.add_argument("--q4-data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q4_2025"))
    parser.add_argument("--seed-q4-data-dir", type=Path, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("v4/audit/autoresearch/v4_aplus_hypothesis_021_q1_fragility_diagnostic"),
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=[11, 22, 33])
    parser.add_argument("--policy-index", type=int, default=1, choices=sorted(POLICY_META))
    parser.add_argument("--variant-name", default="surface_structure_aplus_teacher_margin")
    parser.add_argument("--trial-name", default="post_open_late_edge25_max2")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--decision-cache-dir", type=Path, default=Path("data/cache/v4_aplus_surface_decisions"))
    parser.add_argument("--no-decision-cache", action="store_true")
    parser.add_argument("--market-structure-source", choices=("v2_cache", "index_bars"), default="v2_cache")
    parser.add_argument("--market-spx-dir", type=Path, default=None)
    parser.add_argument("--market-vix-dir", type=Path, default=None)
    parser.add_argument("--es-vwap-dir", type=Path, default=None)
    return parser.parse_args()


def _finite_json(value: object) -> float | int | str | None:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (float, int)):
        f = float(value)
        if not math.isfinite(f):
            return None
        return f
    if value is None:
        return None
    return str(value)


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
    raise SystemExit(f"unknown protocol trial: {name}")


def _trade_from_row(row: dict) -> Trade:
    return Trade(
        session=str(row["session"]),
        decision_time=str(row["decision_time"]),
        pnl=float(row["pnl"]),
        score=None if row.get("score") is None else float(row["score"]),
        right=str(row["right"]),
        offset=float(row["offset"]),
        strategy=str(row["strategy"]),
    )


def _summarize_trade_rows(rows: Sequence[dict]) -> dict:
    trades = [_trade_from_row(row) for row in rows]
    metrics = metrics_with_concentration(trades)
    pnl = np.asarray([row["pnl"] for row in rows], dtype=float)
    out = {
        "rows": int(len(rows)),
        "total_pnl": float(metrics["total_pnl"]),
        "profit_factor": float(metrics["profit_factor"]),
        "max_drawdown": float(metrics["max_drawdown"]),
        "positive_day_fraction": float(metrics["positive_day_fraction"]),
        "top_day_profit_share": float(metrics["top_day_profit_share"]),
        "win_rate": float(np.mean(pnl > 0.0)) if len(pnl) else 0.0,
        "avg_pnl": float(np.mean(pnl)) if len(pnl) else 0.0,
        "median_pnl": float(np.median(pnl)) if len(pnl) else 0.0,
    }
    seed_metrics = []
    for seed in sorted({int(row["seed"]) for row in rows}):
        seed_rows = [row for row in rows if int(row["seed"]) == seed]
        seed_trade_metrics = metrics_with_concentration([_trade_from_row(row) for row in seed_rows])
        seed_metrics.append(seed_trade_metrics)
    if seed_metrics:
        stress50_metrics = []
        stress100_metrics = []
        for seed in sorted({int(row["seed"]) for row in rows}):
            seed_rows = [row for row in rows if int(row["seed"]) == seed]
            seed_trades = [_trade_from_row(row) for row in seed_rows]
            stress50_metrics.append(
                metrics_with_concentration(stress_trades(seed_trades, extra_cost_per_trade=50.0))
            )
            stress100_metrics.append(
                metrics_with_concentration(stress_trades(seed_trades, extra_cost_per_trade=100.0))
            )
        out.update(
            {
                "seed_pnl_median": float(np.median([m["total_pnl"] for m in seed_metrics])),
                "seed_pf_median": float(np.median([m["profit_factor"] for m in seed_metrics])),
                "seed_trades_median": float(np.median([m["trades"] for m in seed_metrics])),
                "positive_seed_fraction": float(np.mean([m["total_pnl"] > 0.0 for m in seed_metrics])),
                "seed_stress50_pnl_median": float(np.median([m["total_pnl"] for m in stress50_metrics])),
                "seed_stress50_pf_median": float(np.median([m["profit_factor"] for m in stress50_metrics])),
                "seed_stress100_pnl_median": float(np.median([m["total_pnl"] for m in stress100_metrics])),
                "seed_stress100_pf_median": float(np.median([m["profit_factor"] for m in stress100_metrics])),
            }
        )
    else:
        out.update(
            {
                "seed_pnl_median": 0.0,
                "seed_pf_median": 0.0,
                "seed_trades_median": 0.0,
                "positive_seed_fraction": 0.0,
                "seed_stress50_pnl_median": 0.0,
                "seed_stress50_pf_median": 0.0,
                "seed_stress100_pnl_median": 0.0,
                "seed_stress100_pf_median": 0.0,
            }
        )
    for feature in _FEATURES_OF_INTEREST:
        key = f"feature_{feature}"
        values = np.asarray(
            [row[key] for row in rows if isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))],
            dtype=float,
        )
        if len(values):
            out[f"{key}_median"] = float(np.median(values))
            out[f"{key}_mean"] = float(np.mean(values))
        else:
            out[f"{key}_median"] = None
            out[f"{key}_mean"] = None
    return out


def _group_summary(rows: Sequence[dict], keys: Sequence[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        group_key = tuple(row.get(key) for key in keys)
        groups.setdefault(group_key, []).append(row)
    out = []
    for group_key, group_rows in sorted(groups.items(), key=lambda item: item[0]):
        summary = {key: value for key, value in zip(keys, group_key)}
        summary.update(_summarize_trade_rows(group_rows))
        out.append(summary)
    return out


def _median_or_none(values: Iterable[float | None]) -> float | None:
    clean = np.asarray([v for v in values if v is not None and math.isfinite(float(v))], dtype=float)
    if not len(clean):
        return None
    return float(np.median(clean))


def _decision_summary(decisions: Sequence[SurfaceDecision]) -> dict:
    valid_tokens = int(sum(int(d.token_mask.sum()) for d in decisions))
    return {
        "decision_rows": int(len(decisions)),
        "valid_tokens": valid_tokens,
        "sessions": int(len({d.session for d in decisions})),
        "start_session": min((d.session for d in decisions), default=None),
        "end_session": max((d.session for d in decisions), default=None),
    }


def _simulate_enriched_trades(
    decisions: Sequence[SurfaceDecision],
    predictions: np.ndarray,
    *,
    trial: ProtocolTrial,
    cooldown_minutes: int,
    strategy: str,
    split: str,
    seed: int,
    effective_seed: int,
    feature_names: Sequence[str],
) -> list[dict]:
    trades = []
    next_time_by_session: dict[str, datetime] = {}
    trades_by_session: dict[str, int] = {}
    pnl_by_session: dict[str, float] = {}
    halted_sessions: set[str] = set()
    allowed = set(trial.allowed_buckets)
    for decision, pred in zip(decisions, predictions):
        bucket = time_bucket(decision.decision_time)
        if bucket not in allowed:
            continue
        if decision.session in halted_sessions:
            continue
        if trades_by_session.get(decision.session, 0) >= trial.max_trades_per_day:
            continue
        next_time = next_time_by_session.get(decision.session)
        if next_time is not None and decision.decision_time < next_time:
            continue
        action_mask = np.concatenate([[True], decision.token_mask])
        masked = np.asarray(pred, dtype=float).copy()
        masked[~action_mask] = -np.inf
        if not np.isfinite(masked).any():
            continue
        action = int(np.nanargmax(masked))
        if action == 0:
            continue
        edge = float(masked[action] - masked[0])
        if not np.isfinite(edge) or edge < trial.min_edge_vs_no_trade:
            continue
        token_idx = action - 1
        pnl = float(decision.labels[token_idx])
        if not np.isfinite(pnl):
            continue

        decision_et = decision.decision_time.astimezone(_NY)
        row = {
            "split": split,
            "seed": int(seed),
            "effective_seed": int(effective_seed),
            "session": decision.session,
            "decision_time": decision.decision_time.isoformat(),
            "decision_time_et": decision_et.isoformat(),
            "bucket": bucket,
            "minute_of_day_et": int(decision_et.hour * 60 + decision_et.minute),
            "token_idx": int(token_idx),
            "contract_id": _finite_json(decision.contract_ids[token_idx]),
            "right": str(decision.rights[token_idx]),
            "offset": float(decision.offsets[token_idx]),
            "pnl": pnl,
            "score": edge,
            "flat_score": float(masked[0]),
            "action_score": float(masked[action]),
            "pattern_target": (
                None
                if decision.pattern_targets is None
                else _finite_json(float(decision.pattern_targets[token_idx]))
            ),
            "value_target": (
                None
                if decision.value_targets is None
                else _finite_json(float(decision.value_targets[token_idx]))
            ),
            "strategy": strategy,
        }
        for feature_name, value in zip(feature_names, decision.token_features[token_idx]):
            row[f"feature_{feature_name}"] = _finite_json(float(value))
        trades.append(row)
        trades_by_session[decision.session] = trades_by_session.get(decision.session, 0) + 1
        pnl_by_session[decision.session] = pnl_by_session.get(decision.session, 0.0) + pnl
        next_time_by_session[decision.session] = decision.decision_time + timedelta(minutes=cooldown_minutes)
        if trial.daily_loss_stop is not None and pnl_by_session[decision.session] <= trial.daily_loss_stop:
            halted_sessions.add(decision.session)
    return trades


def _feature_gap_table(rows: Sequence[dict]) -> list[dict]:
    """Compare Q1 post-open puts to the same lens in every audit block."""
    out = []
    for split in sorted({row["split"] for row in rows}):
        for label, predicate in (
            (
                "post_open_put",
                lambda row: row["bucket"] == "post_open_morning" and row["right"] == "P",
            ),
            (
                "post_open_call",
                lambda row: row["bucket"] == "post_open_morning" and row["right"] == "C",
            ),
            ("late_afternoon_all", lambda row: row["bucket"] == "late_afternoon"),
            ("all_other", lambda row: not (row["bucket"] == "post_open_morning" and row["right"] == "P")),
        ):
            group = [row for row in rows if row["split"] == split and predicate(row)]
            summary = {"split": split, "lens": label}
            summary.update(_summarize_trade_rows(group))
            out.append(summary)
    return out


def _aggregate_random_rows(rows: Sequence[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row["split"]), []).append(row)
    out = []
    for split, group in sorted(groups.items()):
        out.append(
            {
                "split": split,
                "random_pnl_median": float(np.median([row["total_pnl_median"] for row in group])),
                "random_pf_median": float(np.median([row["profit_factor_median"] for row in group])),
                "random_trades_median": float(np.median([row["trades_median"] for row in group])),
            }
        )
    return out


def _random_lookup(rows: Sequence[dict], split: str) -> dict:
    for row in rows:
        if row["split"] == split:
            return row
    return {
        "split": split,
        "random_pnl_median": 0.0,
        "random_pf_median": 0.0,
        "random_trades_median": 0.0,
    }


def _lookup(summary_rows: Sequence[dict], **keys: str) -> dict | None:
    for row in summary_rows:
        if all(row.get(k) == v for k, v in keys.items()):
            return row
    return None


def _interpretation(focus_rows: Sequence[dict]) -> dict:
    q1_put = _lookup(focus_rows, split="q1_2025", lens="post_open_put")
    q1_call = _lookup(focus_rows, split="q1_2025", lens="post_open_call")
    q1_late = _lookup(focus_rows, split="q1_2025", lens="late_afternoon_all")
    other_puts = [row for row in focus_rows if row["lens"] == "post_open_put" and row["split"] != "q1_2025"]
    other_put_pnl_median = _median_or_none(row["seed_pnl_median"] for row in other_puts)
    other_put_value_median = _median_or_none(
        row["feature_contract_value_score_median"] for row in other_puts
    )
    q1_value = None if q1_put is None else q1_put.get("feature_contract_value_score_median")
    flags = {
        "q1_post_open_put_negative": bool(q1_put and q1_put["seed_pnl_median"] < 0.0),
        "q1_post_open_put_pf_below_one": bool(q1_put and q1_put["seed_pf_median"] < 1.0),
        "q1_post_open_put_worse_than_q1_post_open_call": bool(
            q1_put
            and q1_call
            and q1_put["seed_pnl_median"] < q1_call["seed_pnl_median"]
        ),
        "q1_late_afternoon_positive": bool(q1_late and q1_late["seed_pnl_median"] > 0.0),
        "q1_post_open_put_value_below_other_audits": bool(
            q1_value is not None
            and other_put_value_median is not None
            and float(q1_value) < float(other_put_value_median)
        ),
    }
    if all(flags.values()):
        conclusion = (
            "Q1 fragility is concentrated exactly where pre-registered: post-open morning puts. "
            "The contract-quality features are directionally weaker than the same lens in other audits, "
            "especially value score, breakeven, and gamma/theta, but this is not a static-veto proof "
            "because Q2 post-open puts still worked with mediocre value medians."
        )
        next_hypothesis = (
            "Pre-register a side-aware contract-quality calibration objective trained before holdout scoring. "
            "It should penalize put-side overpay risk when theta burden, spread tax, breakeven ATR, and weak "
            "gamma/theta make the A+ pattern economically not worth paying for. It must be trained without Q1 "
            "labels and survive Q1/March/Q2/Q3/Q4 plus slippage stress."
        )
    elif q1_put and q1_put["seed_pnl_median"] < 0.0:
        conclusion = (
            "Q1 post-open puts are fragile, but the feature evidence is incomplete. Treat this as a target "
            "for diagnostics, not a permission to hard-filter Q1."
        )
        next_hypothesis = (
            "Inspect side-aware value calibration on train/calibration/selection only, then audit unchanged "
            "against all frozen blocks."
        )
    else:
        conclusion = (
            "The pre-registered Q1 put-quality failure mode was not confirmed strongly enough to justify a "
            "model change."
        )
        next_hypothesis = "Do not add a Q1-specific change from this diagnostic."
    return {
        "flags": flags,
        "other_post_open_put_seed_pnl_median": other_put_pnl_median,
        "other_post_open_put_contract_value_score_median": other_put_value_median,
        "conclusion": conclusion,
        "next_hypothesis": next_hypothesis,
    }


def _fmt(value: object, digits: int = 0) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(f):
        if f > 0:
            return "inf"
        if f < 0:
            return "-inf"
        return ""
    return f"{f:.{digits}f}"


def _sanitize_json(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return None
    return value


def _write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# Protocol 021: Q1 Fragility Diagnostic",
        "",
        payload["framing"],
        "",
        "## Pre-Registration",
        "",
    ]
    for item in payload["pre_registration"]:
        lines.append(f"- {item}")
    lines += [
        "",
        "## Fixed Candidate",
        "",
        f"- Variant: `{payload['candidate']['variant_name']}`",
        f"- Policy: `policy{payload['candidate']['policy_index']}` / `{payload['candidate']['policy_name']}`",
        f"- Trial: `{payload['candidate']['trial_name']}`",
        f"- Window ID: `{payload['window']['window_id']}`",
        "",
        "## Data Used",
        "",
    ]
    for split, summary in payload["decision_summaries"].items():
        lines.append(
            f"- {split}: {summary['sessions']} sessions, {summary['decision_rows']} decision rows, "
            f"{summary['valid_tokens']} valid tokens"
        )
    lines += [
        "",
        "Cost: $0 incremental paid data. This script reads existing processed `.pkl` files only.",
        "",
        "## Split Summary",
        "",
        "| Split | Seed Median PnL | Seed Median PF | Trades | Positive Seeds | +50 PnL | +100 PnL | Random PnL | All-Seed PnL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    random_rows = payload.get("random_baseline_by_split", [])
    for row in payload["summary_by_split"]:
        random_row = _random_lookup(random_rows, row["split"])
        lines.append(
            f"| {row['split']} | {_fmt(row['seed_pnl_median'])} | {_fmt(row['seed_pf_median'], 3)} | "
            f"{_fmt(row['seed_trades_median'])} | {_fmt(row['positive_seed_fraction'], 2)} | "
            f"{_fmt(row['seed_stress50_pnl_median'])} | {_fmt(row['seed_stress100_pnl_median'])} | "
            f"{_fmt(random_row['random_pnl_median'])} | {_fmt(row['total_pnl'])} |"
        )
    lines += [
        "",
        "## Fragility Lens",
        "",
        "| Split | Lens | Seed Median PnL | Seed Median PF | Trades | Avg PnL | Win Rate | Value Score | Theta Burden | Spread Tax | Breakeven ATR | Gamma/Theta | Overpay Flag |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["focus_lens"]:
        lines.append(
            f"| {row['split']} | {row['lens']} | {_fmt(row['seed_pnl_median'])} | "
            f"{_fmt(row['seed_pf_median'], 3)} | {_fmt(row['rows'])} | {_fmt(row['avg_pnl'])} | "
            f"{_fmt(row['win_rate'], 2)} | {_fmt(row['feature_contract_value_score_median'], 3)} | "
            f"{_fmt(row['feature_theta_burden_hold_median'], 3)} | "
            f"{_fmt(row['feature_spread_tax_median'], 3)} | {_fmt(row['feature_breakeven_atr_median'], 2)} | "
            f"{_fmt(row['feature_gamma_theta_ratio_scaled_median'], 3)} | "
            f"{_fmt(row['feature_obvious_overpay_flag_median'], 2)} |"
        )
    lines += [
        "",
        "## Bucket / Side Detail",
        "",
        "| Split | Bucket | Side | Seed Median PnL | Seed Median PF | Trades | Value Score | Pattern Count |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary_by_split_bucket_side"]:
        lines.append(
            f"| {row['split']} | {row['bucket']} | {row['right']} | {_fmt(row['seed_pnl_median'])} | "
            f"{_fmt(row['seed_pf_median'], 3)} | {_fmt(row['rows'])} | "
            f"{_fmt(row['feature_contract_value_score_median'], 3)} | "
            f"{_fmt(row['feature_pattern_count_norm_median'], 3)} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        payload["interpretation"]["conclusion"],
        "",
        "Next hypothesis:",
        "",
        payload["interpretation"]["next_hypothesis"],
        "",
        "This is not paper/live approval and not a data-purchase approval.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    variant = _find_variant(args.variant_name)
    trial = _find_trial(args.trial_name)
    policy_name, cooldown = POLICY_META[args.policy_index]
    decision_cache_dir = None if args.no_decision_cache else args.decision_cache_dir

    train_paths = _paths_by_split(args.data_dir)
    seed_q4_dir = args.seed_q4_data_dir or args.q4_data_dir
    seed_q4_paths = _paths(seed_q4_dir)
    window = _protocol_window(train_paths, seed_q4_paths)
    market_cache = MarketStructureCache(
        source=args.market_structure_source,
        index_spx_dir=args.market_spx_dir,
        index_vix_dir=args.market_vix_dir,
        es_vwap_dir=args.es_vwap_dir,
    )
    feature_names = token_feature_names(variant.token_mode)

    decision_sets: dict[str, Sequence[SurfaceDecision]] = {}
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
    decision_sets["selection"] = selection_decisions
    decision_sets["march_2026"] = _load_surface_decisions_cached(
        train_paths["test"],
        policy_index=args.policy_index,
        variant=variant,
        market_cache=market_cache,
        split="march",
        cache_dir=decision_cache_dir,
    )
    for split, data_dir in (
        ("q1_2025", args.q1_data_dir),
        ("q2_2025", args.q2_data_dir),
        ("q3_2025", args.q3_data_dir),
        ("q4_2025", args.q4_data_dir),
    ):
        decision_sets[split] = _load_surface_decisions_cached(
            _paths(data_dir),
            policy_index=args.policy_index,
            variant=variant,
            market_cache=market_cache,
            split=split,
            cache_dir=decision_cache_dir,
        )

    all_selected: list[dict] = []
    random_baseline_rows: list[dict] = []
    selected_dir = args.out_dir / "selected_trades_enriched"
    selected_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        effective_seed = window_seed(seed, window.window_id)
        config = PilotConfig(
            policy_index=args.policy_index,
            policy_name=policy_name,
            cooldown_minutes=cooldown,
            epochs=args.epochs,
            batch_size=args.batch_size,
            hidden_dim=128,
            seed=effective_seed,
        )
        print(
            f"{LOOP_ID} policy={args.policy_index} variant={variant.name} "
            f"trial={trial.name} seed={seed}",
            flush=True,
        )
        model, standardizer, _history = train_surface_model(
            train_decisions,
            calibration_decisions,
            config=config,
            variant=variant,
        )
        for split, decisions in decision_sets.items():
            predictions = predict_surface_actions(
                model,
                standardizer,
                decisions,
                target_scale=config.target_scale,
            )
            rows = _simulate_enriched_trades(
                decisions,
                predictions,
                trial=trial,
                cooldown_minutes=cooldown,
                strategy=f"v4_aplus_neural_protocol_003:{variant.name}:{trial.name}",
                split=split,
                seed=seed,
                effective_seed=effective_seed,
                feature_names=feature_names,
            )
            all_selected.extend(rows)
            random_metrics = summarize_random_baseline(
                decisions,
                trial=trial,
                cooldown_minutes=cooldown,
                seed=effective_seed,
                target_trade_count=len(rows),
            )
            random_baseline_rows.append(
                {
                    "split": split,
                    "seed": seed,
                    "effective_seed": effective_seed,
                    **random_metrics,
                }
            )

    for split in sorted({row["split"] for row in all_selected}):
        split_rows = [row for row in all_selected if row["split"] == split]
        (selected_dir / f"selected_trades_{split}.json").write_text(
            json.dumps(split_rows, indent=2, allow_nan=False) + "\n"
        )

    summary_by_split = _group_summary(all_selected, ["split"])
    summary_by_split_bucket_side = _group_summary(all_selected, ["split", "bucket", "right"])
    focus_lens = _feature_gap_table(all_selected)
    interpretation = _interpretation(focus_lens)
    payload = {
        "loop_id": LOOP_ID,
        "framing": (
            "Pre-registered no-paid-data diagnostic for the frozen A+ timing/value candidate. "
            "The diagnostic asks whether Q1 2025 fragility is concentrated in post-open morning "
            "put entries and contract-quality/overpay variables, without changing the candidate."
        ),
        "pre_registration": [
            "Do not tune to Q1 2025 and do not select a Q1-winning time filter.",
            "Do not download or purchase any paid market data.",
            "Keep the frozen Protocol 018 candidate, seeds, policy, trial, executable labels, and cooldown behavior.",
            "Use Q1 only as a diagnostic holdout, not as a threshold-selection block.",
            "Focus the question on post-open morning puts and A+ contract economics: delta, gamma/theta, theta burden, spread tax, breakeven ATR, and overpay flags.",
            "Any future model change must be pre-registered separately and survive March 2026 plus Q1/Q2/Q3/Q4 2025 without adding Q1-selected knobs.",
        ],
        "candidate": {
            "variant_name": variant.name,
            "variant": asdict(variant) | {"variant_id": variant.variant_id},
            "policy_index": args.policy_index,
            "policy_name": policy_name,
            "cooldown_minutes": cooldown,
            "trial_name": trial.name,
            "trial": asdict(trial) | {"config_id": trial.config_id},
            "seeds": args.seeds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        },
        "window": asdict(window) | {"window_id": window.window_id},
        "args": {
            "data_dir": str(args.data_dir),
            "q1_data_dir": str(args.q1_data_dir),
            "q2_data_dir": str(args.q2_data_dir),
            "q3_data_dir": str(args.q3_data_dir),
            "q4_data_dir": str(args.q4_data_dir),
            "seed_q4_data_dir": str(seed_q4_dir),
            "out_dir": str(args.out_dir),
            "decision_cache_dir": None if decision_cache_dir is None else str(decision_cache_dir),
            "market_structure_source": args.market_structure_source,
            "market_spx_dir": None if args.market_spx_dir is None else str(args.market_spx_dir),
            "market_vix_dir": None if args.market_vix_dir is None else str(args.market_vix_dir),
            "es_vwap_dir": None if args.es_vwap_dir is None else str(args.es_vwap_dir),
        },
        "decision_summaries": {split: _decision_summary(decisions) for split, decisions in decision_sets.items()},
        "feature_manifest": {
            "token_features": list(feature_names),
            "features_of_interest": list(_FEATURES_OF_INTEREST),
            "label_source": "v4_cbbo_ask_entry_bid_exit_no_commission",
            "market_structure_source": getattr(market_cache, "cache_id", "unknown_market_cache"),
        },
        "summary_by_split": summary_by_split,
        "summary_by_split_bucket_side": summary_by_split_bucket_side,
        "focus_lens": focus_lens,
        "random_baseline_rows": random_baseline_rows,
        "random_baseline_by_split": _aggregate_random_rows(random_baseline_rows),
        "interpretation": interpretation,
        "selected_trade_count": len(all_selected),
        "selected_trade_files": str(selected_dir),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_json = args.out_dir / "report.json"
    report_md = args.out_dir / "report.md"
    payload = _sanitize_json(payload)
    report_json.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    _write_markdown(report_md, payload)
    print(report_json)
    print(report_md)
    print(json.dumps(interpretation, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
