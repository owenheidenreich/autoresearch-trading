"""Run Generalization Protocol 002 over the 30-item hypothesis backlog."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np

from v4.model.hypothesis_protocol import (
    GeneralizationWindow,
    MarketStructureCache,
    SurfaceVariant,
    aggregate_protocol_rows,
    bootstrap_trade_pnl,
    load_surface_decisions,
    predict_surface_actions,
    registered_protocol_trials,
    registered_surface_variants,
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


LOOP_ID = "v4_generalization_protocol_002"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_derived"))
    parser.add_argument("--q4-data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q4_2025"))
    parser.add_argument("--out-dir", type=Path, default=Path("v4/audit/autoresearch/v4_generalization_protocol_002"))
    parser.add_argument("--seeds", nargs="*", type=int, default=[11, 22, 33])
    parser.add_argument("--policy-indexes", nargs="*", type=int, default=sorted(POLICY_META), choices=sorted(POLICY_META))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-variants", type=int, default=0, help="debug limit; 0 means all variants")
    return parser.parse_args()


def _paths_by_split(data_dir: Path) -> dict[str, list[Path]]:
    out = {"train": [], "validation": [], "test": []}
    for path in sorted(data_dir.glob("*.pkl")):
        out[split_name(session_from_path(path))].append(path)
    return out


def _sessions(paths: Sequence[Path]) -> list[str]:
    return [session_from_path(path) for path in sorted(paths)]


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
    train_decisions = load_surface_decisions(paths["train"], policy_index=policy_index, variant=variant, market_cache=market_cache)
    validation_decisions = load_surface_decisions(paths["validation"], policy_index=policy_index, variant=variant, market_cache=market_cache)
    calibration_decisions, selection_decisions = split_validation_by_session(validation_decisions)
    march_decisions = load_surface_decisions(paths["test"], policy_index=policy_index, variant=variant, market_cache=market_cache)
    q4_decisions = load_surface_decisions(q4_paths, policy_index=policy_index, variant=variant, market_cache=market_cache)

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
                    "label_source": "v4_cbbo_ask_entry_bid_exit",
                    "market_structure_source": "v2_spx_spy_vix_cache_causal_context",
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


def _delta(a: dict | None, b: dict | None, key: str) -> float | None:
    if a is None or b is None:
        return None
    return float(a.get(key, 0.0) - b.get(key, 0.0))


def _best_by_variant(aggregate: dict, variant: str) -> dict | None:
    rows = [r for r in aggregate["by_combo"].values() if r["variant"] == variant]
    if not rows:
        return None
    return max(
        rows,
        key=lambda r: (
            r["selection_floor"],
            r["selection_trades_median"] > 0,
            r["q4_pnl_median"] + r["march_pnl_median"],
            r["q4_pf_median"],
            r["march_pf_median"],
        ),
    )


def _hypothesis_report(aggregate: dict) -> list[dict]:
    atm_current = _best_by_variant(aggregate, "atm_current_huber")
    atm_structure = _best_by_variant(aggregate, "atm_structure_pressure_huber")
    surface_current = _best_by_variant(aggregate, "surface_current_huber")
    surface_structure_base = _best_by_variant(aggregate, "surface_structure_base_huber")
    surface_structure_pressure = _best_by_variant(aggregate, "surface_structure_pressure_huber")
    sidecontrast = _best_by_variant(aggregate, "surface_structure_pressure_sidecontrast")
    structure_lift = _delta(surface_structure_base, surface_current, "q4_pnl_median")
    pressure_lift = _delta(surface_structure_pressure, surface_structure_base, "q4_pnl_median")
    surface_lift = _delta(surface_current, atm_current, "q4_pnl_median")
    side_lift = _delta(sidecontrast, surface_structure_pressure, "q4_pnl_median")
    pass_count = aggregate["protocol_pass_count"]

    def status_from_lift(
        lift: float | None,
        candidate: dict | None = None,
        *,
        threshold: float = 0.0,
    ) -> str:
        if lift is None:
            return "defer_missing_comparator"
        if candidate is not None and candidate.get("passes_protocol_002_gate"):
            return "promote_candidate"
        if candidate is not None and (candidate.get("march_survives") or candidate.get("q4_survives")):
            return "mixed_continue_diagnostics"
        if lift > threshold:
            return "mixed_no_promotion"
        if lift < -threshold:
            return "reject_for_now"
        return "neutral"

    return [
        {"id": 1, "topic": "stable walk-forward identity", "status": "adopt_protocol", "evidence": "Protocol window id seeds every run through window_seed()."},
        {"id": 2, "topic": "scope-separated reporting", "status": "adopt_protocol", "evidence": "Report separates selection, March audit, Q4 audit, concentration, bootstrap, and per-combo stability."},
        {"id": 3, "topic": "rolling-window design", "status": "defer_until_more_clean_months", "evidence": "Q1 + Q4 are not enough for many disjoint rolling windows; keep expanding-window shape for the next data block."},
        {"id": 4, "topic": "frozen March/Q4 audit", "status": "adopt_protocol", "evidence": "March and Q4 are scored after pre-registration; Q4 is not used inside training."},
        {"id": 5, "topic": "trial manifest", "status": "adopt_protocol", "evidence": "Every row records variant, trial, feature manifest, policy, seed, split counts, and history."},
        {"id": 6, "topic": "multiple-testing protection", "status": "adopt_protocol", "evidence": f"{len(aggregate['by_combo'])} combos are logged and protocol_pass_count={pass_count}; no winner is hidden."},
        {"id": 7, "topic": "prior-window calibration", "status": "adopt_protocol", "evidence": "Training uses January; early February is calibration; late February selection is separate."},
        {"id": 8, "topic": "bootstrap confidence intervals", "status": "adopt_protocol", "evidence": "March and Q4 trade PnL bootstrap intervals are saved for every run."},
        {"id": 9, "topic": "matched/random baselines", "status": "adopt_protocol", "evidence": "Every selection/March/Q4 row includes a randomized same-trial baseline thinned toward the neural trade count."},
        {"id": 10, "topic": "slippage stress grid", "status": "adopt_protocol", "evidence": "Every March/Q4 row includes extra $25/$50/$100 per-trade stress metrics."},
        {"id": 11, "topic": "concentration gates", "status": "adopt_protocol", "evidence": "Top-day share is a gate input for selection, March, and Q4."},
        {"id": 12, "topic": "feature provenance audit", "status": "adopt_protocol", "evidence": "Feature manifest separates v4 label truth from v2/v3 causal context."},
        {"id": 13, "topic": "true VWAP sigma position", "status": status_from_lift(structure_lift, surface_structure_base), "evidence": f"Structure-vs-current Q4 median PnL lift: {structure_lift}."},
        {"id": 14, "topic": "true OMAR fields", "status": status_from_lift(structure_lift, surface_structure_base), "evidence": "OMAR high/low/mid/range are bundled in the structure comparator."},
        {"id": 15, "topic": "first-15 structure", "status": status_from_lift(structure_lift, surface_structure_base), "evidence": "first15 range/position/acceptance are bundled in the structure comparator."},
        {"id": 16, "topic": "last-10 structure", "status": status_from_lift(structure_lift, surface_structure_base), "evidence": "last10 range over OMAR and break state are bundled in the structure comparator."},
        {"id": 17, "topic": "sigma x IV/cell awareness", "status": "partial_next", "evidence": "Sigma is included; explicit IV-percentile cells need official cross-day IV percentile construction."},
        {"id": 18, "topic": "time-of-day interactions", "status": "adopt_protocol", "evidence": "Time bucket one-hots are scalar features and fixed trials reuse prior windows."},
        {"id": 19, "topic": "ATR-15 scale with OMAR as location", "status": status_from_lift(structure_lift, surface_structure_base), "evidence": "ATR15_pct and OMAR location fields are included in the structure comparator."},
        {"id": 20, "topic": "option quality features", "status": status_from_lift(pressure_lift, surface_structure_pressure), "evidence": f"Pressure-vs-base Q4 median PnL lift: {pressure_lift}."},
        {"id": 21, "topic": "Greek pressure features", "status": status_from_lift(pressure_lift, surface_structure_pressure), "evidence": "gamma/theta pressure features are part of the pressure comparator."},
        {"id": 22, "topic": "full action surface", "status": status_from_lift(surface_lift, surface_current), "evidence": f"Surface-vs-ATM Q4 median PnL lift: {surface_lift}."},
        {"id": 23, "topic": "unified policy architecture", "status": status_from_lift(surface_lift, surface_current), "evidence": "Surface model uses scalar state plus contract-token scoring and explicit flat action."},
        {"id": 24, "topic": "side-contrastive training", "status": status_from_lift(side_lift, sidecontrast), "evidence": f"Sidecontrast-vs-surface-pressure Q4 median PnL lift: {side_lift}."},
        {"id": 25, "topic": "side/cell balance", "status": status_from_lift(side_lift, sidecontrast), "evidence": "Side ranking is tested; explicit cell-balance awaits IV-percentile cells."},
        {"id": 26, "topic": "clean-entry and stopout heads", "status": "defer", "evidence": "No clean/stopout auxiliary labels were added; keep entry model first."},
        {"id": 27, "topic": "separate entry and exit research", "status": "adopt_protocol", "evidence": "This run remains entry/side/strike-only."},
        {"id": 28, "topic": "H3a in-trade exit features", "status": "defer", "evidence": "Exit features stay deferred until entry survives Q4."},
        {"id": 29, "topic": "regime-stratified diagnostics", "status": "partial_next", "evidence": "Time/concentration diagnostics are present; full trend x vol and sigma x IV tables should be the next report layer."},
        {"id": 30, "topic": "broad-purchase signal", "status": "pass" if pass_count else "not_cleared", "evidence": f"Protocol 002 pass count: {pass_count}."},
    ]


def _write_markdown(path: Path, payload: dict) -> None:
    aggregate = payload["aggregate"]
    champion = aggregate["generalization_champion"]
    lines = [
        "# v4 Generalization Protocol 002",
        "",
        payload["framing"],
        "",
        f"Window ID: `{payload['window']['window_id']}`",
        f"Protocol pass count: `{aggregate['protocol_pass_count']}`",
        "",
        "## Generalization Champion",
        "",
    ]
    if champion is None:
        lines.append("No champion was produced.")
    else:
        lines += [
            f"Best generalization row: **{champion['variant']} / policy{champion['policy_index']} / {champion['trial']}**",
            "",
            "| Split | Median PnL | Median PF | Median DD | Trades | Positive Seeds | Positive Days | Top-Day Share |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            f"| Selection | {champion['selection_pnl_median']:.0f} | {champion['selection_pf_median']:.3f} |  | {champion['selection_trades_median']:.0f} |  | {champion['selection_positive_days_median']:.2f} | {champion['selection_top_day_share_median']:.2f} |",
            f"| March | {champion['march_pnl_median']:.0f} | {champion['march_pf_median']:.3f} | {champion['march_dd_median']:.0f} | {champion['march_trades_median']:.0f} | {champion['march_positive_seed_fraction']:.2f} | {champion['march_positive_days_median']:.2f} | {champion['march_top_day_share_median']:.2f} |",
            f"| Q4 | {champion['q4_pnl_median']:.0f} | {champion['q4_pf_median']:.3f} | {champion['q4_dd_median']:.0f} | {champion['q4_trades_median']:.0f} | {champion['q4_positive_seed_fraction']:.2f} | {champion['q4_positive_days_median']:.2f} | {champion['q4_top_day_share_median']:.2f} |",
        ]
    lines += [
        "",
        "## Top Rows By Generalization",
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
        "## Hypothesis Decisions",
        "",
        "| ID | Topic | Status | Evidence |",
        "|---:|---|---|---|",
    ]
    for row in payload["hypotheses"]:
        lines.append(f"| {row['id']} | {row['topic']} | {row['status']} | {row['evidence']} |")
    lines += [
        "",
        "## Interpretation",
        "",
        (
            "A passing row would justify the next narrow data audit, not live trading. "
            "A failing row means the tested combination is not yet general enough, even if it looks good in February."
        ),
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    paths = _paths_by_split(args.data_dir)
    q4_paths = sorted(args.q4_data_dir.glob("*.pkl"))
    if not q4_paths:
        raise SystemExit(f"no Q4 pkl files found under {args.q4_data_dir}")
    variants = list(registered_surface_variants())
    if args.max_variants:
        variants = variants[: args.max_variants]
    window = _protocol_window(paths, q4_paths)
    market_cache = MarketStructureCache()
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
                    )
                )
    aggregate = aggregate_protocol_rows(rows)
    payload = {
        "loop_id": LOOP_ID,
        "framing": (
            "Pre-registered hypothesis screen for the next SPXW 0DTE generalization protocol. "
            "Models train on January, early February is calibration, late February is selection, "
            "and March plus frozen Q4 2025 are audit-only survival checks."
        ),
        "window": asdict(window) | {"window_id": window.window_id},
        "args": {
            "data_dir": str(args.data_dir),
            "q4_data_dir": str(args.q4_data_dir),
            "out_dir": str(args.out_dir),
            "seeds": args.seeds,
            "policy_indexes": args.policy_indexes,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "variants": [asdict(v) | {"variant_id": v.variant_id} for v in variants],
        },
        "protocol_rules": [
            "No paid data download.",
            "No Q4 tuning inside the loop.",
            "Every variant/trial/seed is logged.",
            "Selection reward is February-selection only.",
            "March and Q4 survival decide whether a hypothesis is worth promoting.",
            "v4 CBBO ask-entry/bid-exit labels remain executable truth.",
        ],
        "aggregate": aggregate,
        "hypotheses": _hypothesis_report(aggregate),
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
