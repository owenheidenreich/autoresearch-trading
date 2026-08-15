"""Run Protocol 003: A+ pattern/value-aware neural surface policy.

This is the neural follow-up to the non-neural A+ contract audit. It keeps the
same train/calibration/selection/audit split discipline as Protocol 002:

* January 2026 trains the model.
* Early February calibrates/early-stops.
* Late February selects the champion.
* March 2026 and frozen Q4 2025 are audit-only survival checks.

No paid data is downloaded here. The only new model idea is explicit A+
contract context: transferable timing patterns plus current contract value
features, with an optional multitask loss for pattern presence and
"worth paying the spread" classification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from v4.model.hypothesis_protocol import (
    GeneralizationWindow,
    MarketStructureCache,
    SurfaceVariant,
    aggregate_protocol_rows,
    bootstrap_trade_pnl,
    load_surface_decisions,
    predict_surface_actions,
    registered_aplus_surface_variants,
    registered_protocol_trials,
    selection_reward,
    simulate_surface_policy,
    stress_trades,
    summarize_random_baseline,
    structural_feature_names,
    token_feature_names,
    train_surface_model,
    window_seed,
)
from v4.model.supervised_pilot import PilotConfig, session_from_path, split_name
from v4.scripts.evaluate_calibrated_abstention_signal import split_validation_by_session
from v4.scripts.evaluate_risk_controlled_purchase_signal import metrics_with_concentration
from v4.scripts.train_spxw_supervised_pilot import POLICY_META


LOOP_ID = "v4_aplus_neural_protocol_003"
CACHE_VERSION = "aplus_surface_decisions_v3_no_commission_context"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_derived"))
    parser.add_argument("--q4-data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q4_2025"))
    parser.add_argument(
        "--seed-q4-data-dir",
        type=Path,
        default=None,
        help="optional external-holdout directory used only for stable window/seed derivation",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("v4/audit/autoresearch/v4_aplus_neural_protocol_003"))
    parser.add_argument("--seeds", nargs="*", type=int, default=[11, 22, 33])
    parser.add_argument("--policy-indexes", nargs="*", type=int, default=sorted(POLICY_META), choices=sorted(POLICY_META))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--decision-cache-dir",
        type=Path,
        default=Path("data/cache/v4_aplus_surface_decisions"),
    )
    parser.add_argument("--no-decision-cache", action="store_true")
    parser.add_argument("--max-variants", type=int, default=0, help="debug limit; 0 means all variants")
    parser.add_argument(
        "--variant-names",
        nargs="*",
        default=[],
        help="optional explicit variant names; empty means all registered Protocol 003 variants",
    )
    parser.add_argument("--market-structure-source", choices=("v2_cache", "index_bars"), default="v2_cache")
    parser.add_argument("--market-spx-dir", type=Path, default=None)
    parser.add_argument("--market-vix-dir", type=Path, default=None)
    parser.add_argument("--es-vwap-dir", type=Path, default=None)
    return parser.parse_args()


def _paths_by_split(data_dir: Path) -> dict[str, list[Path]]:
    out = {"train": [], "validation": [], "test": []}
    for path in sorted(data_dir.glob("*.pkl")):
        out[split_name(session_from_path(path))].append(path)
    return out


def _sessions(paths: Sequence[Path]) -> list[str]:
    return [session_from_path(path) for path in sorted(paths)]


def _paths_fingerprint(
    paths: Sequence[Path],
    *,
    policy_index: int,
    variant: SurfaceVariant,
    split: str,
    market_cache_id: str,
) -> str:
    payload = {
        "cache_version": CACHE_VERSION,
        "policy_index": int(policy_index),
        "variant": asdict(variant) | {"variant_id": variant.variant_id},
        "split": split,
        "market_cache_id": market_cache_id,
        "paths": [
            {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in sorted(paths)
        ],
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def _load_surface_decisions_cached(
    paths: Sequence[Path],
    *,
    policy_index: int,
    variant: SurfaceVariant,
    market_cache: MarketStructureCache,
    split: str,
    cache_dir: Path | None,
) -> list:
    if cache_dir is None:
        return load_surface_decisions(
            paths,
            policy_index=policy_index,
            variant=variant,
            market_cache=market_cache,
        )
    key = _paths_fingerprint(
        paths,
        policy_index=policy_index,
        variant=variant,
        split=split,
        market_cache_id=getattr(market_cache, "cache_id", "unknown_market_cache"),
    )
    cache_path = cache_dir / f"{split}.policy{policy_index}.{variant.name}.{key}.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as handle:
            decisions = pickle.load(handle)
        print(f"decision cache hit {split} policy={policy_index} variant={variant.name} rows={len(decisions)}", flush=True)
        return decisions
    decisions = load_surface_decisions(
        paths,
        policy_index=policy_index,
        variant=variant,
        market_cache=market_cache,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(decisions, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(cache_path)
    print(f"decision cache write {split} policy={policy_index} variant={variant.name} rows={len(decisions)}", flush=True)
    return decisions


def _protocol_window(paths: dict[str, list[Path]], q4_paths: Sequence[Path]) -> GeneralizationWindow:
    validation_sessions = _sessions(paths["validation"])
    calibration_sessions = validation_sessions[: len(validation_sessions) // 2]
    selection_sessions = validation_sessions[len(validation_sessions) // 2 :]
    train_sessions = _sessions(paths["train"])
    march_sessions = _sessions(paths["test"])
    q4_sessions = _sessions(q4_paths)
    return GeneralizationWindow(
        train_start=train_sessions[0],
        train_end=train_sessions[-1],
        calibration_start=calibration_sessions[0],
        calibration_end=calibration_sessions[-1],
        selection_start=selection_sessions[0],
        selection_end=selection_sessions[-1],
        audit_start=f"{march_sessions[0]}+{q4_sessions[0]}",
        audit_end=f"{march_sessions[-1]}+{q4_sessions[-1]}",
    )


def _run_one_variant(
    *,
    paths: dict[str, list[Path]],
    q4_paths: Sequence[Path],
    policy_index: int,
    seed: int,
    variant: SurfaceVariant,
    market_cache: MarketStructureCache,
    epochs: int,
    batch_size: int,
    stable_window_id: str,
    decision_cache_dir: Path | None,
) -> list[dict]:
    policy_name, cooldown = POLICY_META[policy_index]
    effective_seed = window_seed(seed, stable_window_id)
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
        paths["train"],
        policy_index=policy_index,
        variant=variant,
        market_cache=market_cache,
        split="train",
        cache_dir=decision_cache_dir,
    )
    validation_decisions = _load_surface_decisions_cached(
        paths["validation"],
        policy_index=policy_index,
        variant=variant,
        market_cache=market_cache,
        split="validation",
        cache_dir=decision_cache_dir,
    )
    calibration_decisions, selection_decisions = split_validation_by_session(validation_decisions)
    march_decisions = _load_surface_decisions_cached(
        paths["test"],
        policy_index=policy_index,
        variant=variant,
        market_cache=market_cache,
        split="march",
        cache_dir=decision_cache_dir,
    )
    q4_decisions = _load_surface_decisions_cached(
        q4_paths,
        policy_index=policy_index,
        variant=variant,
        market_cache=market_cache,
        split="q4",
        cache_dir=decision_cache_dir,
    )

    model, standardizer, history = train_surface_model(
        train_decisions,
        calibration_decisions,
        config=config,
        variant=variant,
    )
    decision_sets = {
        "train": train_decisions,
        "calibration": calibration_decisions,
        "selection": selection_decisions,
        "march": march_decisions,
        "q4": q4_decisions,
    }
    predictions = {
        split: predict_surface_actions(model, standardizer, decisions, target_scale=config.target_scale)
        for split, decisions in decision_sets.items()
    }
    rows = []
    for trial in registered_protocol_trials():
        metrics_by_split = {}
        bootstrap_by_split = {}
        random_baseline_by_split = {}
        slippage_stress_by_split = {}
        for split, decisions in decision_sets.items():
            trades = simulate_surface_policy(
                decisions,
                predictions[split],
                trial=trial,
                cooldown_minutes=cooldown,
                strategy=f"{LOOP_ID}:{variant.name}:{trial.name}",
            )
            metrics_by_split[split] = metrics_with_concentration(trades)
            if split in {"selection", "march", "q4"}:
                random_baseline_by_split[split] = summarize_random_baseline(
                    decisions,
                    trial=trial,
                    cooldown_minutes=cooldown,
                    seed=effective_seed,
                    target_trade_count=len(trades),
                )
            if split in {"march", "q4"}:
                bootstrap_by_split[split] = bootstrap_trade_pnl(trades, seed=effective_seed)
                slippage_stress_by_split[split] = {
                    str(extra_cost): metrics_with_concentration(
                        stress_trades(trades, extra_cost_per_trade=float(extra_cost))
                    )
                    for extra_cost in (25, 50, 100)
                }
        rows.append(
            {
                "loop_id": LOOP_ID,
                "policy_index": policy_index,
                "policy_name": policy_name,
                "seed": seed,
                "effective_seed": effective_seed,
                "variant": asdict(variant) | {"variant_id": variant.variant_id},
                "trial": asdict(trial) | {"config_id": trial.config_id},
                "feature_manifest": {
                    "scalar_features": structural_feature_names(variant.market_mode),
                    "token_features": token_feature_names(variant.token_mode),
                    "label_source": "v4_cbbo_ask_entry_bid_exit_no_commission",
                    "market_structure_source": getattr(market_cache, "cache_id", "unknown_market_cache"),
                    "auxiliary_targets": {
                        "aplus_multitask": ["pattern_present", "contract_worth_paying_spread"],
                        "aplus_teacher_margin": [
                            "pattern_present",
                            "contract_worth_paying_spread",
                            "positive_pattern_value_teacher_margin",
                        ],
                        "aplus_side_quality_margin": [
                            "pattern_present",
                            "contract_worth_paying_spread",
                            "side_aware_put_contract_quality_margin",
                        ],
                        "aplus_side_value_multitask": [
                            "pattern_present",
                            "side_weighted_contract_worth_paying_spread",
                            "positive_pattern_value_teacher_margin",
                        ],
                        "aplus_balanced_value_multitask": [
                            "pattern_present",
                            "light_side_weighted_contract_worth_paying_spread",
                            "positive_pattern_value_teacher_margin",
                        ],
                        "aplus_soft_quality_confidence": [
                            "pattern_present",
                            "soft_contract_quality_probability",
                            "quality_weighted_positive_margin",
                            "soft_bad_quality_loser_margin",
                        ],
                    }.get(variant.loss_mode, []),
                },
                "best_epoch": next((x["epoch"] for x in history if x["is_best"]), None),
                "history": history,
                "split_counts": {split: len(decisions) for split, decisions in decision_sets.items()},
                "metrics_by_split": metrics_by_split,
                "random_baseline_by_split": random_baseline_by_split,
                "slippage_stress_by_split": slippage_stress_by_split,
                "bootstrap_by_split": bootstrap_by_split,
                "selection_reward": selection_reward(metrics_by_split["selection"]),
            }
        )
    return rows


def _variant_comparison(aggregate: dict) -> list[dict]:
    out = []
    for variant in sorted({row["variant"] for row in aggregate["by_combo"].values()}):
        rows = [row for row in aggregate["by_combo"].values() if row["variant"] == variant]
        best = max(
            rows,
            key=lambda row: (
                row["passes_protocol_002_gate"],
                row["selection_floor"],
                row["march_pnl_median"] + row["q4_pnl_median"],
                row["march_pf_median"],
                row["q4_pf_median"],
            ),
        )
        out.append(best)
    return sorted(
        out,
        key=lambda row: (
            row["passes_protocol_002_gate"],
            row["selection_floor"],
            row["march_pnl_median"] + row["q4_pnl_median"],
            row["q4_pf_median"],
        ),
        reverse=True,
    )


def _write_markdown(path: Path, payload: dict) -> None:
    aggregate = payload["aggregate"]
    champion = aggregate["generalization_champion"]
    lines = [
        "# v4 A+ Neural Protocol 003",
        "",
        payload["framing"],
        "",
        f"Window ID: `{payload['window']['window_id']}`",
        f"Protocol pass count: `{aggregate['protocol_pass_count']}`",
        "",
        "## Pre-Registered Model Change",
        "",
    ]
    for item in payload["pre_registered_model_change"]:
        lines.append(f"- {item}")
    lines += [
        "",
        "## Generalization Champion",
        "",
    ]
    if champion is None:
        lines.append("No champion was produced.")
    else:
        lines += [
            f"Best row: **{champion['variant']} / policy{champion['policy_index']} / {champion['trial']}**",
            "",
            "| Split | Median PnL | Median PF | Median DD | Trades | Positive Seeds | Positive Days | Top-Day Share |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            f"| Selection | {champion['selection_pnl_median']:.0f} | {champion['selection_pf_median']:.3f} |  | {champion['selection_trades_median']:.0f} |  | {champion['selection_positive_days_median']:.2f} | {champion['selection_top_day_share_median']:.2f} |",
            f"| March | {champion['march_pnl_median']:.0f} | {champion['march_pf_median']:.3f} | {champion['march_dd_median']:.0f} | {champion['march_trades_median']:.0f} | {champion['march_positive_seed_fraction']:.2f} | {champion['march_positive_days_median']:.2f} | {champion['march_top_day_share_median']:.2f} |",
            f"| Q4 | {champion['q4_pnl_median']:.0f} | {champion['q4_pf_median']:.3f} | {champion['q4_dd_median']:.0f} | {champion['q4_trades_median']:.0f} | {champion['q4_positive_seed_fraction']:.2f} | {champion['q4_positive_days_median']:.2f} | {champion['q4_top_day_share_median']:.2f} |",
        ]
    lines += [
        "",
        "## Best Row Per Variant",
        "",
        "| Variant | Policy | Trial | Sel PnL/PF | March PnL/PF | Q4 PnL/PF | Pass |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for row in payload["variant_comparison"]:
        lines.append(
            f"| {row['variant']} | {row['policy_index']} | {row['trial']} | "
            f"{row['selection_pnl_median']:.0f}/{row['selection_pf_median']:.3f} | "
            f"{row['march_pnl_median']:.0f}/{row['march_pf_median']:.3f} | "
            f"{row['q4_pnl_median']:.0f}/{row['q4_pf_median']:.3f} | "
            f"{row['passes_protocol_002_gate']} |"
        )
    lines += [
        "",
        "## Top Rows",
        "",
        "| Rank | Variant | Policy | Trial | Sel PnL | March PnL/PF | Q4 PnL/PF | Pass |",
        "|---:|---|---:|---|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(aggregate["ranked_generalization"][:20], start=1):
        lines.append(
            f"| {rank} | {row['variant']} | {row['policy_index']} | {row['trial']} | "
            f"{row['selection_pnl_median']:.0f} | {row['march_pnl_median']:.0f}/{row['march_pf_median']:.3f} | "
            f"{row['q4_pnl_median']:.0f}/{row['q4_pf_median']:.3f} | {row['passes_protocol_002_gate']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        (
            "A passing row is a research promotion signal only. It means the A+ pattern/value "
            "framing survived the frozen audits well enough to justify narrower validation, "
            "not live trading and not an automatic broad data purchase."
        ),
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    paths = _paths_by_split(args.data_dir)
    q4_paths = sorted(args.q4_data_dir.glob("*.pkl"))
    if not q4_paths:
        raise SystemExit(f"no Q4 pkl files found under {args.q4_data_dir}")
    seed_q4_dir = args.seed_q4_data_dir or args.q4_data_dir
    seed_q4_paths = sorted(seed_q4_dir.glob("*.pkl"))
    if not seed_q4_paths:
        raise SystemExit(f"no seed-window pkl files found under {seed_q4_dir}")
    variants = list(registered_aplus_surface_variants())
    if args.variant_names:
        requested = set(args.variant_names)
        variants = [variant for variant in variants if variant.name in requested]
        missing = requested - {variant.name for variant in variants}
        if missing:
            raise SystemExit(f"unknown Protocol 003 variant(s): {sorted(missing)}")
    if args.max_variants:
        variants = variants[: args.max_variants]
    window = _protocol_window(paths, seed_q4_paths)
    market_cache = MarketStructureCache(
        source=args.market_structure_source,
        index_spx_dir=args.market_spx_dir,
        index_vix_dir=args.market_vix_dir,
        es_vwap_dir=args.es_vwap_dir,
    )
    decision_cache_dir = None if args.no_decision_cache else args.decision_cache_dir
    rows = []
    for policy_index in args.policy_indexes:
        for variant in variants:
            for seed in args.seeds:
                print(
                    f"{LOOP_ID} policy={policy_index} variant={variant.name} seed={seed}",
                    flush=True,
                )
                rows.extend(
                    _run_one_variant(
                        paths=paths,
                        q4_paths=q4_paths,
                        policy_index=policy_index,
                        seed=seed,
                        variant=variant,
                        market_cache=market_cache,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        stable_window_id=window.window_id,
                        decision_cache_dir=decision_cache_dir,
                    )
                )
    aggregate = aggregate_protocol_rows(rows)
    payload = {
        "loop_id": LOOP_ID,
        "framing": (
            "Pre-registered neural follow-up to the A+ contract value audit. "
            "The model still chooses one long SPXW contract or flat. The new "
            "hypothesis is that explicit entry-pattern context plus contract "
            "value/overpay context improves generalization."
        ),
        "window": asdict(window) | {"window_id": window.window_id},
        "args": {
            "data_dir": str(args.data_dir),
            "q4_data_dir": str(args.q4_data_dir),
            "seed_q4_data_dir": str(seed_q4_dir),
            "out_dir": str(args.out_dir),
            "seeds": args.seeds,
            "policy_indexes": args.policy_indexes,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "decision_cache_dir": None if decision_cache_dir is None else str(decision_cache_dir),
            "variant_names": args.variant_names,
            "variants": [asdict(v) | {"variant_id": v.variant_id} for v in variants],
            "market_structure_source": args.market_structure_source,
            "market_spx_dir": None if args.market_spx_dir is None else str(args.market_spx_dir),
            "market_vix_dir": None if args.market_vix_dir is None else str(args.market_vix_dir),
            "es_vwap_dir": None if args.es_vwap_dir is None else str(args.es_vwap_dir),
        },
        "pre_registered_model_change": [
            "Keep the Protocol 002 fixed trial grid, split protocol, seeds, and executable labels.",
            "Add A+ token features: timing pattern flags, side-specific movement, delta, convexity, theta burden, spread tax, breakeven ATR, and value score.",
            "Test one feature-only A+ model and one multitask A+ model with pattern-present and contract-worth-paying auxiliary losses.",
            "Test one teacher-margin model that pushes profitable A+ train/calibration tokens above flat without using holdout labels.",
            "Test one side-quality margin model that penalizes marginal put-side overpay risk without using holdout labels.",
            "Test one side-weighted value multitask model that shapes contract-quality representation without hard-suppressing actions.",
            "Do not select or tune on March 2026 or frozen Q4 2025.",
            "Do not download paid data.",
        ],
        "protocol_rules": [
            "No paid data download.",
            "No Q4 tuning inside the loop.",
            "Every variant/trial/seed is logged.",
            "Selection reward is February-selection only.",
            "March and Q4 survival decide whether the hypothesis is worth promoting.",
            "v4 CBBO ask-entry/bid-exit labels remain executable truth.",
        ],
        "aggregate": aggregate,
        "variant_comparison": _variant_comparison(aggregate),
        "rows": rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "report.json"
    md_path = args.out_dir / "report.md"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    _write_markdown(md_path, payload)
    print(json_path)
    print(md_path)
    print(json.dumps(aggregate["generalization_champion"], indent=2, allow_nan=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
