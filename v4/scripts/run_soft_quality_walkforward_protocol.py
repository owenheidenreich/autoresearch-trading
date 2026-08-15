"""Protocol 038: soft contract quality inside broader walk-forward training.

This protocol answers the question raised by Protocol 024 vs 034 attribution:
can the model learn contract quality as confidence shaping without hard-vetoing
March convex winners?

No paid data is downloaded. The comparison is deliberately small:

* baseline: frozen Protocol 024 model family
* candidate: same features/trial, with a soft quality objective
* folds: expanding 2025 training history, next-quarter tests, then Q1 2026
* trial: policy1 / post_open_late_edge25_max2
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from v4.model.hypothesis_protocol import (
    MarketStructureCache,
    ProtocolTrial,
    SurfaceDecision,
    SurfaceVariant,
    predict_surface_actions,
    registered_aplus_surface_variants,
    registered_protocol_trials,
    selection_reward,
    simulate_surface_policy,
    stress_trades,
    summarize_random_baseline,
    train_surface_model,
    window_seed,
)
from v4.model.supervised_pilot import PilotConfig, session_from_path
from v4.scripts.evaluate_risk_controlled_purchase_signal import metrics_with_concentration
from v4.scripts.run_aplus_neural_protocol import _load_surface_decisions_cached
from v4.scripts.train_spxw_supervised_pilot import POLICY_META


LOOP_ID = "v4_aplus_hypothesis_038_soft_quality_walkforward"
BASELINE_VARIANT = "surface_structure_aplus_side_value_multitask"
CANDIDATE_VARIANT = "surface_structure_aplus_soft_quality_confidence"
DEFAULT_TRIAL = "post_open_late_edge25_max2"


@dataclass(frozen=True)
class FoldSpec:
    name: str
    train_paths: tuple[Path, ...]
    validation_paths: tuple[Path, ...]
    test_paths: tuple[Path, ...]

    @property
    def window_id(self) -> str:
        payload = {
            "name": self.name,
            "train_start": session_from_path(self.train_paths[0]),
            "train_end": session_from_path(self.train_paths[-1]),
            "validation_start": session_from_path(self.validation_paths[0]),
            "validation_end": session_from_path(self.validation_paths[-1]),
            "test_start": session_from_path(self.test_paths[0]),
            "test_end": session_from_path(self.test_paths[-1]),
        }
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def summary(self) -> dict:
        return {
            "name": self.name,
            "train_sessions": [session_from_path(self.train_paths[0]), session_from_path(self.train_paths[-1])],
            "validation_sessions": [
                session_from_path(self.validation_paths[0]),
                session_from_path(self.validation_paths[-1]),
            ],
            "test_sessions": [session_from_path(self.test_paths[0]), session_from_path(self.test_paths[-1])],
            "train_days": len(self.train_paths),
            "validation_days": len(self.validation_paths),
            "test_days": len(self.test_paths),
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--q4-2024-dir", type=Path, default=None)
    p.add_argument("--q1-2025-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q1_2025_nofee"))
    p.add_argument("--q2-2025-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q2_2025_nofee"))
    p.add_argument("--q3-2025-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q3_2025_nofee"))
    p.add_argument("--q4-2025-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q4_2025_nofee"))
    p.add_argument("--q1-2026-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_derived_nofee"))
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("v4/audit/autoresearch/v4_aplus_hypothesis_038_soft_quality_walkforward"),
    )
    p.add_argument(
        "--decision-cache-dir",
        type=Path,
        default=Path("data/cache/v4_aplus_surface_decisions_nofee"),
    )
    p.add_argument("--no-decision-cache", action="store_true")
    p.add_argument("--seeds", nargs="*", type=int, default=[11, 22, 33])
    p.add_argument("--policy-index", type=int, default=1, choices=sorted(POLICY_META))
    p.add_argument("--trial-name", default=DEFAULT_TRIAL)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--validation-days", type=int, default=10)
    p.add_argument("--max-folds", type=int, default=0, help="debug limit; 0 means all folds")
    p.add_argument(
        "--market-structure-source",
        choices=("v2_cache", "index_bars"),
        default="v2_cache",
    )
    p.add_argument("--market-spx-dir", type=Path, default=None)
    p.add_argument("--market-vix-dir", type=Path, default=None)
    p.add_argument("--es-vwap-dir", type=Path, default=None)
    return p.parse_args()


def _paths(data_dir: Path) -> tuple[Path, ...]:
    paths = tuple(sorted(data_dir.glob("*.pkl")))
    if not paths:
        raise SystemExit(f"no pkl files found under {data_dir}")
    return paths


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


def _folds(args: argparse.Namespace) -> list[FoldSpec]:
    blocks = []
    q4_2024_dir = getattr(args, "q4_2024_dir", None)
    if q4_2024_dir is not None:
        blocks.append(("q4_2024", _paths(q4_2024_dir)))
    blocks.extend([
        ("q1_2025", _paths(args.q1_2025_dir)),
        ("q2_2025", _paths(args.q2_2025_dir)),
        ("q3_2025", _paths(args.q3_2025_dir)),
        ("q4_2025", _paths(args.q4_2025_dir)),
        ("q1_2026", _paths(args.q1_2026_dir)),
    ])
    folds: list[FoldSpec] = []
    history: list[Path] = []
    for idx in range(len(blocks) - 1):
        history.extend(blocks[idx][1])
        test_name, test_paths = blocks[idx + 1]
        if len(history) <= args.validation_days:
            raise SystemExit("not enough history for requested validation-days")
        folds.append(
            FoldSpec(
                name=f"train_through_{blocks[idx][0]}_test_{test_name}",
                train_paths=tuple(history[: -args.validation_days]),
                validation_paths=tuple(history[-args.validation_days :]),
                test_paths=tuple(test_paths),
            )
        )
    if args.max_folds:
        folds = folds[: args.max_folds]
    return folds


def _slice_march(decisions: Sequence[SurfaceDecision], predictions: np.ndarray) -> tuple[list[SurfaceDecision], np.ndarray]:
    keep = [idx for idx, decision in enumerate(decisions) if decision.session >= "2026-03-01"]
    if not keep:
        return [], np.empty((0, predictions.shape[1] if predictions.ndim == 2 else 0), dtype=np.float32)
    idxs = np.asarray(keep, dtype=int)
    return [decisions[idx] for idx in keep], predictions[idxs]


def _metrics_row(trades: Sequence, *, decisions: Sequence[SurfaceDecision], trial: ProtocolTrial, cooldown: int, seed: int) -> dict:
    metrics = metrics_with_concentration(trades)
    stress = {
        str(extra): metrics_with_concentration(stress_trades(trades, extra_cost_per_trade=float(extra)))
        for extra in (25, 50, 100)
    }
    random = summarize_random_baseline(
        decisions,
        trial=trial,
        cooldown_minutes=cooldown,
        seed=seed,
        target_trade_count=len(trades),
        runs=20,
    )
    return {
        "metrics": metrics,
        "stress": stress,
        "random_baseline": random,
        "selection_reward": selection_reward(metrics),
    }


def _run_variant_fold(
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
    test_predictions = predict_surface_actions(
        model,
        standardizer,
        test_decisions,
        target_scale=config.target_scale,
    )
    test_trades = simulate_surface_policy(
        test_decisions,
        test_predictions,
        trial=trial,
        cooldown_minutes=cooldown,
        strategy=f"{LOOP_ID}:{variant.name}:{trial.name}:{fold.name}",
    )
    split_results = {
        "test": _metrics_row(
            test_trades,
            decisions=test_decisions,
            trial=trial,
            cooldown=cooldown,
            seed=effective_seed,
        )
    }
    if fold.name.endswith("test_q1_2026"):
        march_decisions, march_predictions = _slice_march(test_decisions, test_predictions)
        march_trades = simulate_surface_policy(
            march_decisions,
            march_predictions,
            trial=trial,
            cooldown_minutes=cooldown,
            strategy=f"{LOOP_ID}:{variant.name}:{trial.name}:march_2026_slice",
        )
        split_results["march_2026_slice"] = _metrics_row(
            march_trades,
            decisions=march_decisions,
            trial=trial,
            cooldown=cooldown,
            seed=effective_seed,
        )
    return {
        "loop_id": LOOP_ID,
        "fold": fold.summary(),
        "fold_name": fold.name,
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
        "split_results": split_results,
    }


def _median(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=float))) if values else 0.0


def _summarize(rows: Sequence[dict]) -> dict:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["fold_name"], row["variant"]["name"]), []).append(row)

    fold_variant = []
    for (fold_name, variant_name), group in sorted(groups.items()):
        test = [row["split_results"]["test"]["metrics"] for row in group]
        stress50 = [row["split_results"]["test"]["stress"]["50"]["total_pnl"] for row in group]
        random_pnl = [row["split_results"]["test"]["random_baseline"]["total_pnl_median"] for row in group]
        summary = {
            "fold_name": fold_name,
            "variant": variant_name,
            "runs": len(group),
            "pnl_median": _median([m["total_pnl"] for m in test]),
            "pf_median": _median([m["profit_factor"] for m in test]),
            "trades_median": _median([m["trades"] for m in test]),
            "dd_median": _median([m["max_drawdown"] for m in test]),
            "positive_seed_fraction": float(np.mean([m["total_pnl"] > 0.0 for m in test])),
            "positive_day_fraction_median": _median([m["positive_day_fraction"] for m in test]),
            "top_day_share_median": _median([m["top_day_profit_share"] for m in test]),
            "stress50_pnl_median": _median(stress50),
            "random_pnl_median": _median(random_pnl),
            "beats_random_median": bool(_median([m["total_pnl"] for m in test]) > _median(random_pnl)),
        }
        if "march_2026_slice" in group[0]["split_results"]:
            march = [row["split_results"]["march_2026_slice"]["metrics"] for row in group]
            march_stress50 = [
                row["split_results"]["march_2026_slice"]["stress"]["50"]["total_pnl"]
                for row in group
            ]
            summary |= {
                "march_pnl_median": _median([m["total_pnl"] for m in march]),
                "march_pf_median": _median([m["profit_factor"] for m in march]),
                "march_trades_median": _median([m["trades"] for m in march]),
                "march_stress50_pnl_median": _median(march_stress50),
                "march_positive_seed_fraction": float(np.mean([m["total_pnl"] > 0.0 for m in march])),
            }
        fold_variant.append(summary)

    by_fold = {
        row["fold_name"]: {item["variant"]: item for item in fold_variant if item["fold_name"] == row["fold_name"]}
        for row in fold_variant
    }
    deltas = []
    for fold_name, variants in sorted(by_fold.items()):
        if BASELINE_VARIANT not in variants or CANDIDATE_VARIANT not in variants:
            continue
        base = variants[BASELINE_VARIANT]
        cand = variants[CANDIDATE_VARIANT]
        delta = {
            "fold_name": fold_name,
            "pnl_delta": cand["pnl_median"] - base["pnl_median"],
            "pf_delta": cand["pf_median"] - base["pf_median"],
            "trades_delta": cand["trades_median"] - base["trades_median"],
            "stress50_delta": cand["stress50_pnl_median"] - base["stress50_pnl_median"],
        }
        if "march_pnl_median" in cand and "march_pnl_median" in base:
            delta |= {
                "march_pnl_delta": cand["march_pnl_median"] - base["march_pnl_median"],
                "march_stress50_delta": cand["march_stress50_pnl_median"] - base["march_stress50_pnl_median"],
            }
        deltas.append(delta)

    candidate_rows = [row for row in fold_variant if row["variant"] == CANDIDATE_VARIANT]
    baseline_rows = [row for row in fold_variant if row["variant"] == BASELINE_VARIANT]
    soft_beats = sum(1 for row in deltas if row["pnl_delta"] > 0.0)
    candidate_positive = all(row["pnl_median"] > 0.0 for row in candidate_rows)
    candidate_stress_positive = all(row["stress50_pnl_median"] > 0.0 for row in candidate_rows)
    candidate_beats_random = all(row["beats_random_median"] for row in candidate_rows)
    march_delta = next((row.get("march_pnl_delta") for row in deltas if "march_pnl_delta" in row), None)
    march_not_damaged = march_delta is None or march_delta >= 0.0

    return {
        "fold_variant": fold_variant,
        "deltas": deltas,
        "decision": {
            "soft_beats_baseline_folds": soft_beats,
            "folds_compared": len(deltas),
            "candidate_positive_all_folds": candidate_positive,
            "candidate_stress50_positive_all_folds": candidate_stress_positive,
            "candidate_beats_random_all_folds": candidate_beats_random,
            "march_not_damaged": march_not_damaged,
            "candidate_mean_pnl": float(np.mean([row["pnl_median"] for row in candidate_rows])) if candidate_rows else 0.0,
            "baseline_mean_pnl": float(np.mean([row["pnl_median"] for row in baseline_rows])) if baseline_rows else 0.0,
            "keep_candidate": bool(
                len(deltas) > 0
                and soft_beats >= max(1, int(np.ceil(0.75 * len(deltas))))
                and candidate_positive
                and candidate_stress_positive
                and candidate_beats_random
                and march_not_damaged
            ),
        },
    }


def _write_markdown(path: Path, payload: dict) -> None:
    summary = payload["summary"]
    decision = summary["decision"]
    lines = [
        "# Protocol 038 Soft Quality Walk-Forward",
        "",
        "No paid data was downloaded. This is a pre-registered comparison of soft in-network contract-quality learning versus the Protocol 024 model family under broader expanding walk-forward training.",
        "",
        "## Fixed Setup",
        "",
        f"- Policy/trial: `policy{payload['args']['policy_index']} / {payload['args']['trial_name']}`",
        f"- Variants: `{BASELINE_VARIANT}` vs `{CANDIDATE_VARIANT}`",
        f"- Seeds: `{payload['args']['seeds']}`",
        f"- Epochs: `{payload['args']['epochs']}`",
        f"- Validation days per fold: `{payload['args']['validation_days']}`",
        "",
        "## Fold Results",
        "",
        "| Fold | Variant | PnL | PF | Trades | +50 Stress | Random PnL | Positive Seeds |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["fold_variant"]:
        lines.append(
            f"| {row['fold_name']} | {row['variant']} | "
            f"{row['pnl_median']:.0f} | {row['pf_median']:.3f} | "
            f"{row['trades_median']:.0f} | {row['stress50_pnl_median']:.0f} | "
            f"{row['random_pnl_median']:.0f} | {row['positive_seed_fraction']:.2f} |"
        )
    lines += [
        "",
        "## Candidate Minus Baseline",
        "",
        "| Fold | PnL Delta | PF Delta | Trades Delta | +50 Stress Delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["deltas"]:
        lines.append(
            f"| {row['fold_name']} | {row['pnl_delta']:.0f} | {row['pf_delta']:.3f} | "
            f"{row['trades_delta']:.0f} | {row['stress50_delta']:.0f} |"
        )
    lines += [
        "",
        "## March 2026 Slice",
        "",
        "| Variant | PnL | PF | Trades | +50 Stress | Positive Seeds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["fold_variant"]:
        if "march_pnl_median" not in row:
            continue
        lines.append(
            f"| {row['variant']} | {row['march_pnl_median']:.0f} | "
            f"{row['march_pf_median']:.3f} | {row['march_trades_median']:.0f} | "
            f"{row['march_stress50_pnl_median']:.0f} | {row['march_positive_seed_fraction']:.2f} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        f"- Soft beats baseline folds: `{decision['soft_beats_baseline_folds']} / {decision['folds_compared']}`",
        f"- Candidate positive all folds: `{decision['candidate_positive_all_folds']}`",
        f"- Candidate +50 stress positive all folds: `{decision['candidate_stress50_positive_all_folds']}`",
        f"- Candidate beats matched random all folds: `{decision['candidate_beats_random_all_folds']}`",
        f"- March not damaged: `{decision['march_not_damaged']}`",
        f"- Keep candidate: `{decision['keep_candidate']}`",
        "",
    ]
    if decision["keep_candidate"]:
        lines.append(
            "Interpretation: the soft quality objective is a credible next baseline candidate, but it is still a research signal rather than live-trading approval."
        )
    else:
        lines.append(
            "Interpretation: do not replace Protocol 024 yet. Use this result to decide the next single hypothesis without adding a new selection grid."
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    variants = [_find_variant(BASELINE_VARIANT), _find_variant(CANDIDATE_VARIANT)]
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
        for variant in variants:
            for seed in args.seeds:
                print(
                    f"{LOOP_ID} fold={fold.name} variant={variant.name} seed={seed}",
                    flush=True,
                )
                rows.append(
                    _run_variant_fold(
                        fold=fold,
                        variant=variant,
                        seed=seed,
                        policy_index=args.policy_index,
                        trial=trial,
                        market_cache=market_cache,
                        decision_cache_dir=decision_cache_dir,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                    )
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
            "seeds": args.seeds,
            "policy_index": args.policy_index,
            "trial_name": args.trial_name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "validation_days": args.validation_days,
            "market_structure_source": args.market_structure_source,
            "market_spx_dir": None if args.market_spx_dir is None else str(args.market_spx_dir),
            "market_vix_dir": None if args.market_vix_dir is None else str(args.market_vix_dir),
            "es_vwap_dir": None if args.es_vwap_dir is None else str(args.es_vwap_dir),
        },
        "pre_registration": {
            "paid_data_downloaded": False,
            "model_change": "soft quality confidence loss only",
            "selection_knobs_added": False,
            "folds": [fold.summary() for fold in folds],
            "comparison_rule": (
                "candidate must beat baseline in at least 75% of folds, stay positive under +50 stress "
                "in every fold, beat matched random in every fold, and not damage the March 2026 slice"
            ),
        },
        "rows": rows,
        "summary": _summarize(rows),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    _write_markdown(args.out_dir / "report.md", payload)
    print(json.dumps(payload["summary"]["decision"], indent=2, sort_keys=True))
    print(f"wrote {args.out_dir / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
