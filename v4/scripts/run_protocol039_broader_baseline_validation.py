"""Protocol 039: no-new-model validation of the broader baseline.

This protocol validates the Protocol 024-family model under the broader
expanding walk-forward setup introduced by Protocol 038. It intentionally adds
no new model, loss, filter, selection knob, or paid data.

Outputs:

* more-seed fold stability
* side/time/contract exposure summaries
* trade-set attribution versus frozen Protocol 024 selected trades
* Q2 2025 seed fragility diagnostics
"""
from __future__ import annotations

import argparse
import json
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
from v4.scripts.evaluate_risk_controlled_purchase_signal import metrics_with_concentration
from v4.scripts.run_aplus_neural_protocol import _load_surface_decisions_cached
from v4.scripts.run_soft_quality_walkforward_protocol import FoldSpec, _folds
from v4.scripts.train_spxw_supervised_pilot import POLICY_META


LOOP_ID = "v4_aplus_hypothesis_039_broader_baseline_validation"
BASELINE_VARIANT = "surface_structure_aplus_side_value_multitask"
DEFAULT_TRIAL = "post_open_late_edge25_max2"
_NY = ZoneInfo("America/New_York")

FOLD_TO_FROZEN_SPLIT = {
    "train_through_q1_2025_test_q2_2025": "q2_2025",
    "train_through_q2_2025_test_q3_2025": "q3_2025",
    "train_through_q3_2025_test_q4_2025": "q4_2025",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--q1-2025-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q1_2025_nofee"))
    p.add_argument("--q2-2025-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q2_2025_nofee"))
    p.add_argument("--q3-2025-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q3_2025_nofee"))
    p.add_argument("--q4-2025-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q4_2025_nofee"))
    p.add_argument("--q1-2026-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_derived_nofee"))
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("v4/audit/autoresearch/v4_aplus_hypothesis_039_broader_baseline_validation"),
    )
    p.add_argument(
        "--decision-cache-dir",
        type=Path,
        default=Path("data/cache/v4_aplus_surface_decisions_nofee"),
    )
    p.add_argument(
        "--frozen-protocol024-dir",
        type=Path,
        default=Path(
            "v4/audit/autoresearch/v4_aplus_hypothesis_024_side_value_multitask_nofee/selected_trades_enriched"
        ),
    )
    p.add_argument("--no-decision-cache", action="store_true")
    p.add_argument("--seeds", nargs="*", type=int, default=[11, 22, 33, 44, 55, 66, 77, 88, 99, 111])
    p.add_argument("--variant-name", default=BASELINE_VARIANT)
    p.add_argument("--loop-id", default=LOOP_ID)
    p.add_argument(
        "--purpose",
        default="validate broader baseline stability and diagnose fragility before adding another model knob",
    )
    p.add_argument("--policy-index", type=int, default=1, choices=sorted(POLICY_META))
    p.add_argument("--trial-name", default=DEFAULT_TRIAL)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--validation-days", type=int, default=10)
    p.add_argument("--max-folds", type=int, default=0, help="debug limit; 0 means all folds")
    p.add_argument("--random-runs", type=int, default=20)
    p.add_argument(
        "--market-structure-source",
        choices=("v2_cache", "index_bars"),
        default="v2_cache",
    )
    p.add_argument("--market-spx-dir", type=Path, default=None)
    p.add_argument("--market-vix-dir", type=Path, default=None)
    p.add_argument("--es-vwap-dir", type=Path, default=None)
    return p.parse_args()


def _find_variant(name: str) -> SurfaceVariant:
    for variant in registered_aplus_surface_variants():
        if variant.name == name:
            return variant
    raise SystemExit(f"unknown variant: {name}")


def _find_trial(name: str) -> ProtocolTrial:
    for trial in registered_protocol_trials():
        if trial.name == name:
            return trial
    raise SystemExit(f"unknown trial: {name}")


def _minute_of_day_et(value: datetime) -> int:
    local = value.astimezone(_NY)
    return local.hour * 60 + local.minute


def _offset_bucket(offset: float) -> str:
    abs_offset = abs(float(offset))
    if abs_offset < 2.5:
        return "atm"
    if abs_offset <= 10.0:
        return "near_5_10"
    if abs_offset <= 25.0:
        return "mid_15_25"
    return "far_30_50"


def _moneyness_bucket(right: str, offset: float) -> str:
    if abs(float(offset)) < 2.5:
        return "atm"
    if right == "C":
        return "itm" if offset < 0.0 else "otm"
    if right == "P":
        return "itm" if offset > 0.0 else "otm"
    return "unknown"


def _finite(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _trade_key(row: dict) -> tuple:
    return (
        int(row["seed"]),
        str(row["session"]),
        str(row["decision_time"]),
        str(row.get("contract_id", "")),
    )


def _minute_key(row: dict) -> tuple:
    return (int(row["seed"]), str(row["session"]), str(row["decision_time"]))


def _rows_to_trades(rows: Sequence[dict], *, strategy: str = "protocol039_group") -> list[Trade]:
    return [
        Trade(
            session=str(row["session"]),
            decision_time=str(row["decision_time"]),
            pnl=float(row["pnl"]),
            score=None if row.get("score") is None else float(row["score"]),
            right=str(row["right"]),
            offset=float(row["offset"]),
            strategy=str(row.get("strategy") or strategy),
        )
        for row in rows
    ]


def _metrics_for_rows(rows: Sequence[dict]) -> dict:
    metrics = metrics_with_concentration(_rows_to_trades(rows))
    pnl = np.asarray([float(row["pnl"]) for row in rows], dtype=float)
    scores = np.asarray([_finite(row.get("score"), np.nan) for row in rows], dtype=float)
    return {
        **metrics,
        "avg_pnl": float(np.mean(pnl)) if len(pnl) else 0.0,
        "median_pnl": float(np.median(pnl)) if len(pnl) else 0.0,
        "win_rate": float(np.mean(pnl > 0.0)) if len(pnl) else 0.0,
        "avg_score": float(np.nanmean(scores)) if np.isfinite(scores).any() else 0.0,
    }


def _group_summary(rows: Sequence[dict], keys: Sequence[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key, "unknown") for key in keys), []).append(row)
    out = []
    for group_key, group_rows in sorted(groups.items(), key=lambda item: item[0]):
        record = {key: value for key, value in zip(keys, group_key)}
        record.update(_metrics_for_rows(group_rows))
        out.append(record)
    return sorted(out, key=lambda row: (row["total_pnl"], row["trades"]), reverse=True)


def _enriched_trade_rows(
    decisions: Sequence[SurfaceDecision],
    predictions: np.ndarray,
    *,
    trial: ProtocolTrial,
    cooldown_minutes: int,
    strategy: str,
    split: str,
    fold_name: str,
    seed: int,
    effective_seed: int,
    variant: SurfaceVariant,
) -> list[dict]:
    rows = []
    feature_names = token_feature_names(variant.token_mode)
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
        token_features = np.asarray(decision.token_features[token_idx], dtype=float)
        local_time = decision.decision_time.astimezone(_NY)
        right = str(decision.rights[token_idx])
        offset = float(decision.offsets[token_idx])
        row = {
            "loop_id": LOOP_ID,
            "split": split,
            "fold_name": fold_name,
            "seed": int(seed),
            "effective_seed": int(effective_seed),
            "session": decision.session,
            "decision_time": decision.decision_time.isoformat(),
            "decision_time_et": local_time.isoformat(),
            "minute_of_day_et": _minute_of_day_et(decision.decision_time),
            "bucket": bucket,
            "token_idx": int(token_idx),
            "contract_id": str(decision.contract_ids[token_idx]),
            "right": right,
            "offset": offset,
            "offset_bucket": _offset_bucket(offset),
            "moneyness_bucket": _moneyness_bucket(right, offset),
            "pnl": pnl,
            "score": edge,
            "flat_score": float(masked[0]),
            "action_score": float(masked[action]),
            "pattern_target": (
                None
                if getattr(decision, "pattern_targets", None) is None
                else float(decision.pattern_targets[token_idx])
            ),
            "value_target": (
                None
                if getattr(decision, "value_targets", None) is None
                else float(decision.value_targets[token_idx])
            ),
            "quality_target": (
                None
                if getattr(decision, "quality_targets", None) is None
                else float(decision.quality_targets[token_idx])
            ),
            "strategy": strategy,
        }
        for name, value in zip(feature_names, token_features):
            row[f"feature_{name}"] = _finite(value)
        rows.append(row)
        trades_by_session[decision.session] = trades_by_session.get(decision.session, 0) + 1
        pnl_by_session[decision.session] = pnl_by_session.get(decision.session, 0.0) + pnl
        next_time_by_session[decision.session] = decision.decision_time + timedelta(minutes=cooldown_minutes)
        if trial.daily_loss_stop is not None and pnl_by_session[decision.session] <= trial.daily_loss_stop:
            halted_sessions.add(decision.session)
    return rows


def _run_fold_seed(
    *,
    fold: FoldSpec,
    variant: SurfaceVariant,
    seed: int,
    policy_index: int,
    trial: ProtocolTrial,
    market_cache: MarketStructureCache,
    decision_cache_dir: Path | None,
    epochs: int,
    batch_size: int,
    random_runs: int,
) -> dict:
    policy_name, cooldown = POLICY_META[policy_index]
    effective_seed = window_seed(seed, fold.window_id)
    config = PilotConfig(
        policy_index=policy_index,
        policy_name=policy_name,
        cooldown_minutes=cooldown,
        epochs=epochs,
        batch_size=batch_size,
        hidden_dim=128,
        seed=effective_seed,
    )
    train_decisions = _load_surface_decisions_cached(
        fold.train_paths,
        policy_index=policy_index,
        variant=variant,
        market_cache=market_cache,
        split=f"{fold.name}_train",
        cache_dir=decision_cache_dir,
    )
    validation_decisions = _load_surface_decisions_cached(
        fold.validation_paths,
        policy_index=policy_index,
        variant=variant,
        market_cache=market_cache,
        split=f"{fold.name}_validation",
        cache_dir=decision_cache_dir,
    )
    test_decisions = _load_surface_decisions_cached(
        fold.test_paths,
        policy_index=policy_index,
        variant=variant,
        market_cache=market_cache,
        split=f"{fold.name}_test",
        cache_dir=decision_cache_dir,
    )
    model, standardizer, history = train_surface_model(
        train_decisions,
        validation_decisions,
        config=config,
        variant=variant,
    )
    predictions = predict_surface_actions(model, standardizer, test_decisions, target_scale=config.target_scale)
    split = FOLD_TO_FROZEN_SPLIT.get(fold.name, "q1_2026")
    rows = _enriched_trade_rows(
        test_decisions,
        predictions,
        trial=trial,
        cooldown_minutes=cooldown,
        strategy=f"{LOOP_ID}:{variant.name}:{trial.name}:{fold.name}",
        split=split,
        fold_name=fold.name,
        seed=seed,
        effective_seed=effective_seed,
        variant=variant,
    )
    metrics = _metrics_for_rows(rows)
    stress = {
        str(extra): metrics_with_concentration(stress_trades(_rows_to_trades(rows), extra_cost_per_trade=float(extra)))
        for extra in (25, 50, 100)
    }
    random = summarize_random_baseline(
        test_decisions,
        trial=trial,
        cooldown_minutes=cooldown,
        seed=effective_seed,
        target_trade_count=len(rows),
        runs=random_runs,
    )
    march_rows = [row | {"split": "march_2026"} for row in rows if row["session"] >= "2026-03-01"]
    return {
        "fold": fold.summary(),
        "fold_name": fold.name,
        "split": split,
        "policy_index": policy_index,
        "policy_name": policy_name,
        "trial": asdict(trial) | {"config_id": trial.config_id},
        "seed": seed,
        "effective_seed": effective_seed,
        "variant": asdict(variant) | {"variant_id": variant.variant_id},
        "best_epoch": next((row["epoch"] for row in history if row["is_best"]), None),
        "history": history,
        "decision_counts": {
            "train": len(train_decisions),
            "validation": len(validation_decisions),
            "test": len(test_decisions),
        },
        "metrics": metrics,
        "stress": stress,
        "random_baseline": random,
        "selection_reward": selection_reward(metrics),
        "selected_trades": rows,
        "march_2026_trades": march_rows,
        "march_2026_metrics": _metrics_for_rows(march_rows),
        "march_2026_stress": {
            str(extra): metrics_with_concentration(stress_trades(_rows_to_trades(march_rows), extra_cost_per_trade=float(extra)))
            for extra in (25, 50, 100)
        },
    }


def _median(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.median(arr)) if len(arr) else 0.0


def _summarize_fold_stability(rows: Sequence[dict]) -> list[dict]:
    out = []
    for fold_name in sorted({row["fold_name"] for row in rows}):
        group = [row for row in rows if row["fold_name"] == fold_name]
        metrics = [row["metrics"] for row in group]
        stress50 = [row["stress"]["50"]["total_pnl"] for row in group]
        random_pnl = [row["random_baseline"]["total_pnl_median"] for row in group]
        out.append(
            {
                "fold_name": fold_name,
                "split": group[0]["split"],
                "runs": len(group),
                "pnl_median": _median(m["total_pnl"] for m in metrics),
                "pnl_mean": float(np.mean([m["total_pnl"] for m in metrics])),
                "pnl_min": float(np.min([m["total_pnl"] for m in metrics])),
                "pnl_max": float(np.max([m["total_pnl"] for m in metrics])),
                "pf_median": _median(m["profit_factor"] for m in metrics),
                "trades_median": _median(m["trades"] for m in metrics),
                "positive_seed_fraction": float(np.mean([m["total_pnl"] > 0.0 for m in metrics])),
                "stress50_pnl_median": _median(stress50),
                "stress50_positive_seed_fraction": float(np.mean([value > 0.0 for value in stress50])),
                "random_pnl_median": _median(random_pnl),
                "beats_random_seed_fraction": float(
                    np.mean([m["total_pnl"] > r for m, r in zip(metrics, random_pnl)])
                ),
                "top_day_share_median": _median(m["top_day_profit_share"] for m in metrics),
                "positive_day_fraction_median": _median(m["positive_day_fraction"] for m in metrics),
            }
        )
    return out


def _load_frozen_trades(directory: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for path in sorted(directory.glob("selected_trades_*.json")):
        split = path.stem.removeprefix("selected_trades_")
        rows = json.loads(path.read_text())
        for row in rows:
            row.setdefault("split", split)
            row["source"] = "frozen_protocol024"
        out[split] = rows
    return out


def _attribution_for_split(*, broad_rows: Sequence[dict], frozen_rows: Sequence[dict], split: str) -> dict:
    common_seeds = sorted({int(row["seed"]) for row in broad_rows} & {int(row["seed"]) for row in frozen_rows})
    broad = [row for row in broad_rows if int(row["seed"]) in common_seeds]
    frozen = [row for row in frozen_rows if int(row["seed"]) in common_seeds]
    broad_by_exact = {_trade_key(row): row for row in broad}
    frozen_by_exact = {_trade_key(row): row for row in frozen}
    exact_keys = set(broad_by_exact) & set(frozen_by_exact)
    broad_only = [row for key, row in broad_by_exact.items() if key not in frozen_by_exact]
    frozen_only = [row for key, row in frozen_by_exact.items() if key not in broad_by_exact]
    broad_minutes = {_minute_key(row) for row in broad}
    frozen_minutes = {_minute_key(row) for row in frozen}
    same_minute_changed = [
        row for row in broad_only if _minute_key(row) in frozen_minutes
    ]
    return {
        "split": split,
        "common_seeds": common_seeds,
        "broad": _metrics_for_rows(broad),
        "frozen": _metrics_for_rows(frozen),
        "exact_overlap_trades": len(exact_keys),
        "exact_overlap_pnl": float(sum(float(broad_by_exact[key]["pnl"]) for key in exact_keys)),
        "broad_only": _metrics_for_rows(broad_only),
        "frozen_only": _metrics_for_rows(frozen_only),
        "same_minute_changed": _metrics_for_rows(same_minute_changed),
        "frozen_only_missed_winners": float(sum(max(float(row["pnl"]), 0.0) for row in frozen_only)),
        "frozen_only_avoided_losers": float(sum(min(float(row["pnl"]), 0.0) for row in frozen_only)),
        "broad_only_new_winners": float(sum(max(float(row["pnl"]), 0.0) for row in broad_only)),
        "broad_only_new_losers": float(sum(min(float(row["pnl"]), 0.0) for row in broad_only)),
        "broad_only_side_bucket": _group_summary(broad_only, ["bucket", "right"]),
        "frozen_only_side_bucket": _group_summary(frozen_only, ["bucket", "right"]),
    }


def _all_trade_rows(rows: Sequence[dict]) -> list[dict]:
    out = []
    for row in rows:
        out.extend(row["selected_trades"])
    return out


def _march_trade_rows(rows: Sequence[dict]) -> list[dict]:
    out = []
    for row in rows:
        out.extend(row["march_2026_trades"])
    return out


def _q2_fragility(rows: Sequence[dict]) -> dict:
    q2_runs = [row for row in rows if row["split"] == "q2_2025"]
    seed_rows = []
    for row in q2_runs:
        record = {
            "seed": row["seed"],
            "effective_seed": row["effective_seed"],
            "best_epoch": row["best_epoch"],
            **row["metrics"],
            "stress50_pnl": row["stress"]["50"]["total_pnl"],
            "random_pnl_median": row["random_baseline"]["total_pnl_median"],
        }
        seed_rows.append(record)
    seed_rows = sorted(seed_rows, key=lambda item: item["total_pnl"])
    all_q2_trades = [trade for row in q2_runs for trade in row["selected_trades"]]
    losing_seed_values = {row["seed"] for row in seed_rows if row["total_pnl"] <= 0.0}
    winning_seed_values = {row["seed"] for row in seed_rows if row["total_pnl"] > 0.0}
    losing_trades = [row for row in all_q2_trades if row["seed"] in losing_seed_values]
    winning_trades = [row for row in all_q2_trades if row["seed"] in winning_seed_values]
    return {
        "seed_rows": seed_rows,
        "positive_seed_fraction": float(np.mean([row["total_pnl"] > 0.0 for row in seed_rows])) if seed_rows else 0.0,
        "stress50_positive_seed_fraction": float(np.mean([row["stress50_pnl"] > 0.0 for row in seed_rows])) if seed_rows else 0.0,
        "worst_seed": seed_rows[0] if seed_rows else None,
        "best_seed": seed_rows[-1] if seed_rows else None,
        "all_q2_exposure": {
            "side": _group_summary(all_q2_trades, ["right"]),
            "time": _group_summary(all_q2_trades, ["bucket"]),
            "side_time": _group_summary(all_q2_trades, ["bucket", "right"]),
            "moneyness": _group_summary(all_q2_trades, ["right", "moneyness_bucket", "offset_bucket"]),
        },
        "losing_seed_exposure": {
            "side_time": _group_summary(losing_trades, ["bucket", "right"]),
            "moneyness": _group_summary(losing_trades, ["right", "moneyness_bucket", "offset_bucket"]),
        },
        "winning_seed_exposure": {
            "side_time": _group_summary(winning_trades, ["bucket", "right"]),
            "moneyness": _group_summary(winning_trades, ["right", "moneyness_bucket", "offset_bucket"]),
        },
        "worst_trades": sorted(all_q2_trades, key=lambda row: float(row["pnl"]))[:20],
        "best_trades": sorted(all_q2_trades, key=lambda row: float(row["pnl"]), reverse=True)[:20],
    }


def _build_summary(rows: Sequence[dict], frozen: dict[str, list[dict]]) -> dict:
    trades = _all_trade_rows(rows)
    march_trades = _march_trade_rows(rows)
    attribution = []
    for fold_name, frozen_split in FOLD_TO_FROZEN_SPLIT.items():
        attribution.append(
            _attribution_for_split(
                broad_rows=[row for row in trades if row["fold_name"] == fold_name],
                frozen_rows=frozen.get(frozen_split, []),
                split=frozen_split,
            )
        )
    attribution.append(
        _attribution_for_split(
            broad_rows=march_trades,
            frozen_rows=frozen.get("march_2026", []),
            split="march_2026",
        )
    )
    return {
        "fold_stability": _summarize_fold_stability(rows),
        "overall_exposure": {
            "side": _group_summary(trades, ["right"]),
            "time": _group_summary(trades, ["bucket"]),
            "side_time": _group_summary(trades, ["bucket", "right"]),
            "moneyness": _group_summary(trades, ["right", "moneyness_bucket", "offset_bucket"]),
            "fold_side_time": _group_summary(trades, ["split", "bucket", "right"]),
        },
        "march_exposure": {
            "side_time": _group_summary(march_trades, ["bucket", "right"]),
            "moneyness": _group_summary(march_trades, ["right", "moneyness_bucket", "offset_bucket"]),
        },
        "trade_set_attribution_vs_frozen_protocol024": attribution,
        "q2_seed_fragility": _q2_fragility(rows),
    }


def _write_markdown(path: Path, payload: dict) -> None:
    summary = payload["summary"]
    variant_name = payload["args"].get("variant_name", BASELINE_VARIANT)
    lines = [
        f"# {payload.get('loop_id', LOOP_ID)} Validation",
        "",
        "No paid data was downloaded. The expanding walk-forward, policy, trial, and selection rules are fixed.",
        "",
        f"Variant: `{variant_name}`",
        "",
        "## Fold Stability",
        "",
        "| Fold | Runs | PnL Median | PnL Min | PnL Max | PF Median | Trades | +50 Median | Positive Seeds | Beats Random |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["fold_stability"]:
        lines.append(
            f"| {row['split']} | {row['runs']} | {row['pnl_median']:.0f} | {row['pnl_min']:.0f} | "
            f"{row['pnl_max']:.0f} | {row['pf_median']:.3f} | {row['trades_median']:.0f} | "
            f"{row['stress50_pnl_median']:.0f} | {row['positive_seed_fraction']:.2f} | "
            f"{row['beats_random_seed_fraction']:.2f} |"
        )
    lines += [
        "",
        "## Side And Time Exposure",
        "",
        "| Bucket | Right | PnL | PF | Trades | Win Rate | Avg PnL |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["overall_exposure"]["side_time"][:20]:
        lines.append(
            f"| {row['bucket']} | {row['right']} | {row['total_pnl']:.0f} | {row['profit_factor']:.3f} | "
            f"{row['trades']} | {row['win_rate']:.2f} | {row['avg_pnl']:.1f} |"
        )
    lines += [
        "",
        "## Attribution Versus Frozen Protocol 024",
        "",
        "Attribution is restricted to common base seeds shared with the frozen Protocol 024 selected-trade files.",
        "",
        "| Split | Broad PnL | Frozen PnL | Delta | Exact Overlap | Broad-Only PnL | Frozen-Only PnL | Missed Frozen Winners | Avoided Frozen Losers |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["trade_set_attribution_vs_frozen_protocol024"]:
        broad = row["broad"]
        frozen = row["frozen"]
        broad_only = row["broad_only"]
        frozen_only = row["frozen_only"]
        lines.append(
            f"| {row['split']} | {broad['total_pnl']:.0f} | {frozen['total_pnl']:.0f} | "
            f"{broad['total_pnl'] - frozen['total_pnl']:.0f} | {row['exact_overlap_trades']} | "
            f"{broad_only['total_pnl']:.0f} | {frozen_only['total_pnl']:.0f} | "
            f"{row['frozen_only_missed_winners']:.0f} | {row['frozen_only_avoided_losers']:.0f} |"
        )
    q2 = summary["q2_seed_fragility"]
    lines += [
        "",
        "## Q2 2025 Seed Fragility",
        "",
        f"- Positive seed fraction: `{q2['positive_seed_fraction']:.2f}`",
        f"- +50 stress positive seed fraction: `{q2['stress50_positive_seed_fraction']:.2f}`",
    ]
    if q2["worst_seed"] is not None:
        lines.append(
            f"- Worst seed: `{q2['worst_seed']['seed']}` PnL `{q2['worst_seed']['total_pnl']:.0f}`, "
            f"PF `{q2['worst_seed']['profit_factor']:.3f}`, trades `{q2['worst_seed']['trades']}`"
        )
    if q2["best_seed"] is not None:
        lines.append(
            f"- Best seed: `{q2['best_seed']['seed']}` PnL `{q2['best_seed']['total_pnl']:.0f}`, "
            f"PF `{q2['best_seed']['profit_factor']:.3f}`, trades `{q2['best_seed']['trades']}`"
        )
    lines += [
        "",
        "### Q2 Losing-Seed Exposure",
        "",
        "| Bucket | Right | PnL | PF | Trades | Win Rate |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in q2["losing_seed_exposure"]["side_time"]:
        lines.append(
            f"| {row['bucket']} | {row['right']} | {row['total_pnl']:.0f} | "
            f"{row['profit_factor']:.3f} | {row['trades']} | {row['win_rate']:.2f} |"
        )
    lines += [
        "",
        "| Seed | PnL | PF | Trades | +50 PnL | Random PnL | Top-Day Share |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in q2["seed_rows"]:
        lines.append(
            f"| {row['seed']} | {row['total_pnl']:.0f} | {row['profit_factor']:.3f} | {row['trades']} | "
            f"{row['stress50_pnl']:.0f} | {row['random_pnl_median']:.0f} | {row['top_day_profit_share']:.2f} |"
        )
    lines += [
        "",
        "## Decision",
        "",
    ]
    q2_ok = q2["positive_seed_fraction"] >= 0.80 and q2["stress50_positive_seed_fraction"] >= 0.70
    all_positive = all(row["positive_seed_fraction"] >= 0.80 for row in summary["fold_stability"])
    all_stress = all(row["stress50_positive_seed_fraction"] >= 0.70 for row in summary["fold_stability"])
    if all_positive and all_stress and q2_ok:
        lines.append(
            "The variant passes this standalone stability audit as a research candidate. This is still not paper/live approval; the next gate is comparison against the frozen incumbent plus path-level 1s/tick verification where applicable."
        )
    else:
        lines.append(
            "The variant is promising but not promotion-grade. Diagnose the fragile fold/seed exposures before treating it as a replacement for the frozen incumbent."
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    global LOOP_ID
    LOOP_ID = args.loop_id
    variant = _find_variant(args.variant_name)
    trial = _find_trial(args.trial_name)
    folds = _folds(args)
    market_cache = MarketStructureCache(
        source=args.market_structure_source,
        index_spx_dir=args.market_spx_dir,
        index_vix_dir=args.market_vix_dir,
        es_vwap_dir=args.es_vwap_dir,
    )
    decision_cache_dir = None if args.no_decision_cache else args.decision_cache_dir
    rows = []
    for fold in folds:
        for seed in args.seeds:
            print(f"{LOOP_ID} fold={fold.name} seed={seed}", flush=True)
            rows.append(
                _run_fold_seed(
                    fold=fold,
                    variant=variant,
                    seed=seed,
                    policy_index=args.policy_index,
                    trial=trial,
                    market_cache=market_cache,
                    decision_cache_dir=decision_cache_dir,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    random_runs=args.random_runs,
                )
            )
    frozen = _load_frozen_trades(args.frozen_protocol024_dir)
    summary = _build_summary(rows, frozen)
    selected_dir = args.out_dir / "selected_trades"
    selected_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        (selected_dir / f"{row['split']}_seed{row['seed']}.json").write_text(
            json.dumps(row["selected_trades"], indent=2, sort_keys=True)
        )
    payload = {
        "loop_id": LOOP_ID,
        "args": {
            "q1_2025_dir": str(args.q1_2025_dir),
            "q2_2025_dir": str(args.q2_2025_dir),
            "q3_2025_dir": str(args.q3_2025_dir),
            "q4_2025_dir": str(args.q4_2025_dir),
            "q1_2026_dir": str(args.q1_2026_dir),
            "out_dir": str(args.out_dir),
            "decision_cache_dir": None if decision_cache_dir is None else str(decision_cache_dir),
            "frozen_protocol024_dir": str(args.frozen_protocol024_dir),
            "seeds": args.seeds,
            "policy_index": args.policy_index,
            "trial_name": args.trial_name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "validation_days": args.validation_days,
            "random_runs": args.random_runs,
            "market_structure_source": args.market_structure_source,
            "variant_name": args.variant_name,
            "loop_id": args.loop_id,
            "purpose": args.purpose,
        },
        "pre_registration": {
            "paid_data_downloaded": False,
            "new_model_added": bool(args.variant_name != BASELINE_VARIANT),
            "new_loss_added": bool(variant.loss_mode != "aplus_side_value_multitask"),
            "new_selection_knob_added": False,
            "variant": asdict(variant) | {"variant_id": variant.variant_id},
            "folds": [fold.summary() for fold in folds],
            "purpose": args.purpose,
        },
        "runs": [
            {k: v for k, v in row.items() if k not in {"selected_trades", "march_2026_trades"}}
            for row in rows
        ],
        "summary": summary,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    _write_markdown(args.out_dir / "report.md", payload)
    print(json.dumps(summary["fold_stability"], indent=2, sort_keys=True))
    print(f"wrote {args.out_dir / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
