"""Protocol 044: put-pattern recall loss screen on official context.

This is a one-change experiment after Protocol 043. It keeps the Protocol 039
walk-forward, trial, policy, data, and selection rules fixed, and only compares
the frozen baseline loss against a soft put-pattern recall loss.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import v4.scripts.run_soft_quality_walkforward_protocol as walk
from v4.model.hypothesis_protocol import MarketStructureCache, SurfaceVariant
from v4.scripts.train_spxw_supervised_pilot import POLICY_META


LOOP_ID = "v4_aplus_hypothesis_044_put_pattern_recall"
BASELINE_VARIANT = "surface_structure_aplus_side_value_multitask"
CANDIDATE_VARIANT = "surface_structure_aplus_put_pattern_recall"
DEFAULT_TRIAL = "post_open_late_edge25_max2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q1-2025-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q1_2025_official_context"))
    parser.add_argument("--q2-2025-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q2_2025_official_context"))
    parser.add_argument("--q3-2025-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q3_2025_official_context"))
    parser.add_argument("--q4-2025-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q4_2025_official_context"))
    parser.add_argument("--q1-2026-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q1_2026_official_context"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("v4/audit/autoresearch/v4_aplus_hypothesis_044_put_pattern_recall"),
    )
    parser.add_argument(
        "--decision-cache-dir",
        type=Path,
        default=Path("data/cache/v4_aplus_surface_decisions_official_context"),
    )
    parser.add_argument("--no-decision-cache", action="store_true")
    parser.add_argument("--seeds", nargs="*", type=int, default=[11, 22, 33])
    parser.add_argument("--policy-index", type=int, default=1, choices=sorted(POLICY_META))
    parser.add_argument("--trial-name", default=DEFAULT_TRIAL)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--validation-days", type=int, default=10)
    parser.add_argument("--max-folds", type=int, default=0, help="debug limit; 0 means all folds")
    parser.add_argument("--market-structure-source", choices=("v2_cache", "index_bars"), default="index_bars")
    parser.add_argument("--market-spx-dir", type=Path, default=Path("data/vendor/thetadata/index/spx_1m"))
    parser.add_argument("--market-vix-dir", type=Path, default=Path("data/vendor/thetadata/index/vix_1m"))
    parser.add_argument("--es-vwap-dir", type=Path, default=None)
    return parser.parse_args()


def _write_markdown(path: Path, payload: dict) -> None:
    summary = payload["summary"]
    decision = summary["decision"]
    lines = [
        "# Protocol 044 Put-Pattern Recall Screen",
        "",
        "No paid data was downloaded. This is a pre-registered one-change comparison on official-context data.",
        "",
        "## Fixed Setup",
        "",
        f"- Policy/trial: `policy{payload['args']['policy_index']} / {payload['args']['trial_name']}`",
        f"- Baseline: `{BASELINE_VARIANT}`",
        f"- Candidate: `{CANDIDATE_VARIANT}`",
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
            f"| {row['fold_name']} | {row['variant']} | {row['pnl_median']:.0f} | "
            f"{row['pf_median']:.3f} | {row['trades_median']:.0f} | "
            f"{row['stress50_pnl_median']:.0f} | {row['random_pnl_median']:.0f} | "
            f"{row['positive_seed_fraction']:.2f} |"
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
            f"| {row['variant']} | {row['march_pnl_median']:.0f} | {row['march_pf_median']:.3f} | "
            f"{row['march_trades_median']:.0f} | {row['march_stress50_pnl_median']:.0f} | "
            f"{row['march_positive_seed_fraction']:.2f} |"
        )
    lines += [
        "",
        "## Decision Rule",
        "",
        f"- Candidate beats baseline folds: `{decision['soft_beats_baseline_folds']} / {decision['folds_compared']}`",
        f"- Candidate positive all folds: `{decision['candidate_positive_all_folds']}`",
        f"- Candidate +50 stress positive all folds: `{decision['candidate_stress50_positive_all_folds']}`",
        f"- Candidate beats matched random all folds: `{decision['candidate_beats_random_all_folds']}`",
        f"- March not damaged: `{decision['march_not_damaged']}`",
        f"- Keep candidate: `{decision['keep_candidate']}`",
        "",
        "Interpretation: keep the candidate only if it clears the rule above. Otherwise reject this one-change hypothesis and preserve Protocol 039 official context as the frozen baseline.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    walk.LOOP_ID = LOOP_ID
    walk.BASELINE_VARIANT = BASELINE_VARIANT
    walk.CANDIDATE_VARIANT = CANDIDATE_VARIANT

    variants = [
        walk._find_variant(BASELINE_VARIANT),
        SurfaceVariant(
            name=CANDIDATE_VARIANT,
            action_space="surface",
            market_mode="structure",
            token_mode="aplus",
            loss_mode="aplus_put_pattern_recall",
        ),
    ]
    trial = walk._find_trial(args.trial_name)
    folds = walk._folds(args)
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
                print(f"{LOOP_ID} fold={fold.name} variant={variant.name} seed={seed}", flush=True)
                rows.append(
                    walk._run_variant_fold(
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
            "model_change": "soft put-pattern recall loss only",
            "selection_knobs_added": False,
            "baseline_variant": BASELINE_VARIANT,
            "candidate_variant": CANDIDATE_VARIANT,
            "folds": [fold.summary() for fold in folds],
            "comparison_rule": (
                "candidate must beat baseline in at least 75% of folds, stay positive under +50 stress "
                "in every fold, beat matched random in every fold, and not damage the March 2026 slice"
            ),
        },
        "rows": rows,
        "summary": walk._summarize(rows),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_markdown(args.out_dir / "report.md", payload)
    print(json.dumps(payload["summary"]["decision"], indent=2, sort_keys=True))
    print(f"wrote {args.out_dir / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
