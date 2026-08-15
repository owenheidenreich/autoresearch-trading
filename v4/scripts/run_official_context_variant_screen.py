"""Official-context one-change variant screen.

Runs the Protocol 039 expanding walk-forward setup against one candidate
variant. This is intended for autoresearch-style screens where the data,
policy, trial, folds, and selection rule stay fixed and only the model variant
changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import v4.scripts.run_soft_quality_walkforward_protocol as walk
from v4.model.hypothesis_protocol import MarketStructureCache
from v4.scripts.train_spxw_supervised_pilot import POLICY_META


DEFAULT_BASELINE = "surface_structure_aplus_side_value_multitask"
DEFAULT_CANDIDATE = "surface_structure_aplus_interactions_side_value_multitask"
DEFAULT_LOOP_ID = "v4_aplus_hypothesis_045_interactions_official_context"
DEFAULT_TRIAL = "post_open_late_edge25_max2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop-id", default=DEFAULT_LOOP_ID)
    parser.add_argument("--baseline-variant", default=DEFAULT_BASELINE)
    parser.add_argument("--candidate-variant", default=DEFAULT_CANDIDATE)
    parser.add_argument("--model-change", default="A+ pattern/value interaction token features only")
    parser.add_argument("--q1-2025-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q1_2025_official_context"))
    parser.add_argument("--q2-2025-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q2_2025_official_context"))
    parser.add_argument("--q3-2025-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q3_2025_official_context"))
    parser.add_argument("--q4-2025-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q4_2025_official_context"))
    parser.add_argument("--q1-2026-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q1_2026_official_context"))
    parser.add_argument("--out-dir", type=Path, default=Path("v4/audit/autoresearch/v4_aplus_hypothesis_045_interactions_official_context"))
    parser.add_argument("--decision-cache-dir", type=Path, default=Path("data/cache/v4_aplus_surface_decisions_official_context"))
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
    baseline = payload["args"]["baseline_variant"]
    candidate = payload["args"]["candidate_variant"]
    lines = [
        f"# {payload['loop_id']} Variant Screen",
        "",
        "No paid data was downloaded. This is a pre-registered one-change comparison on official-context data.",
        "",
        "## Fixed Setup",
        "",
        f"- Policy/trial: `policy{payload['args']['policy_index']} / {payload['args']['trial_name']}`",
        f"- Baseline: `{baseline}`",
        f"- Candidate: `{candidate}`",
        f"- Model change: `{payload['pre_registration']['model_change']}`",
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
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    walk.LOOP_ID = args.loop_id
    walk.BASELINE_VARIANT = args.baseline_variant
    walk.CANDIDATE_VARIANT = args.candidate_variant

    variants = [walk._find_variant(args.baseline_variant), walk._find_variant(args.candidate_variant)]
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
                print(f"{args.loop_id} fold={fold.name} variant={variant.name} seed={seed}", flush=True)
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
        "loop_id": args.loop_id,
        "args": {
            "baseline_variant": args.baseline_variant,
            "candidate_variant": args.candidate_variant,
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
            "model_change": args.model_change,
            "selection_knobs_added": False,
            "baseline_variant": args.baseline_variant,
            "candidate_variant": args.candidate_variant,
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
