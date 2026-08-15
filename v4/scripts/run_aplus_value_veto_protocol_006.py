"""Run Protocol 006: deterministic A+ contract-value vetoes.

Protocol 004/005 found that the learned A+ permission model has directional
signal, but its raw entry stream is still too fragile under +$50/trade stress.
This protocol does not retrain the neural model. It screens a fixed set of
domain vetoes over the already-simulated one-contract permission policies and
locks the champion using only the selection split with +$50 stress.

The holdout result is still only a broad-data-purchase signal. It is not live
trading approval.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from v4.model.hypothesis_protocol import (
    MarketStructureCache,
    stress_trades,
    summarize_random_baseline,
    window_seed,
)
from v4.model.supervised_pilot import Trade
from v4.scripts.evaluate_calibrated_abstention_signal import split_validation_by_session
from v4.scripts.evaluate_risk_controlled_purchase_signal import metrics_with_concentration
from v4.scripts.run_aplus_neural_protocol import (
    _load_surface_decisions_cached,
    _paths_by_split,
    _protocol_window,
)
from v4.scripts.run_aplus_permission_protocol import PERMISSION_TRIAL, _variant
from v4.scripts.train_spxw_supervised_pilot import POLICY_META


LOOP_ID = "v4_aplus_value_veto_protocol_006"
SPLITS = ("selection", "march", "q4")
SEEDS = (11, 22, 33)


@dataclass(frozen=True)
class ValueVetoSpec:
    source_variant: str
    name: str
    feature: str
    op: str
    threshold: float | None = None
    low: float | None = None
    high: float | None = None

    @property
    def spec_id(self) -> str:
        raw = f"{self.source_variant}__{self.name}"
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selected-trades-dir",
        type=Path,
        default=Path("v4/audit/autoresearch/v4_aplus_permission_protocol_005_stress50_select/selected_trades"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_derived"))
    parser.add_argument("--q4-data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q4_2025"))
    parser.add_argument("--decision-cache-dir", type=Path, default=Path("data/cache/v4_aplus_surface_decisions"))
    parser.add_argument("--no-decision-cache", action="store_true")
    parser.add_argument("--policy-index", type=int, default=1, choices=sorted(POLICY_META))
    parser.add_argument("--out-dir", type=Path, default=Path("v4/audit/autoresearch/v4_aplus_value_veto_protocol_006"))
    parser.add_argument("--selection-mode", choices=("stress-score", "domain-priority"), default="stress-score")
    parser.add_argument(
        "--locked-spec-id",
        default=None,
        help="force the locked champion to a pre-registered spec id instead of selecting from the screen",
    )
    return parser.parse_args()


def registered_value_veto_specs() -> tuple[ValueVetoSpec, ...]:
    """Fixed candidate set chosen from contract-value/Greek economics."""

    specs: list[ValueVetoSpec] = []
    for source in ("bce_cost25", "bce_cost50"):
        specs.extend(
            [
                ValueVetoSpec(source, "none", "", "none"),
                ValueVetoSpec(source, "theta_le_0.15", "feature_theta_burden_hold", "le", threshold=0.15),
                ValueVetoSpec(source, "theta_le_0.25", "feature_theta_burden_hold", "le", threshold=0.25),
                ValueVetoSpec(source, "gamma_theta_ge_0.0025", "feature_gamma_theta_ratio_scaled", "ge", threshold=0.0025),
                ValueVetoSpec(source, "gamma_theta_ge_0.004", "feature_gamma_theta_ratio_scaled", "ge", threshold=0.004),
                ValueVetoSpec(source, "spread_le_0.015", "feature_spread_tax", "le", threshold=0.015),
                ValueVetoSpec(source, "spread_le_0.02", "feature_spread_tax", "le", threshold=0.02),
                ValueVetoSpec(source, "spread_le_0.03", "feature_spread_tax", "le", threshold=0.03),
                ValueVetoSpec(source, "breakeven_le_1.25", "feature_breakeven_atr", "le", threshold=1.25),
                ValueVetoSpec(source, "breakeven_le_1.5", "feature_breakeven_atr", "le", threshold=1.5),
                ValueVetoSpec(source, "breakeven_le_2.0", "feature_breakeven_atr", "le", threshold=2.0),
                ValueVetoSpec(source, "contract_value_ge_-0.5", "feature_contract_value_score", "ge", threshold=-0.5),
                ValueVetoSpec(source, "contract_value_ge_0.0", "feature_contract_value_score", "ge", threshold=0.0),
                ValueVetoSpec(source, "contract_value_ge_0.25", "feature_contract_value_score", "ge", threshold=0.25),
                ValueVetoSpec(source, "delta_0.25_0.85", "feature_abs_delta", "between", low=0.25, high=0.85),
                ValueVetoSpec(source, "delta_0.30_0.80", "feature_abs_delta", "between", low=0.30, high=0.80),
                ValueVetoSpec(source, "delta_0.50_0.85", "feature_abs_delta", "between", low=0.50, high=0.85),
                ValueVetoSpec(source, "base_edge_ge_0.25", "base_edge", "ge", threshold=0.25),
                ValueVetoSpec(source, "base_edge_ge_0.55", "base_edge", "ge", threshold=0.55),
                ValueVetoSpec(source, "base_edge_ge_0.65", "base_edge", "ge", threshold=0.65),
            ]
        )
    return tuple(specs)


def _passes_spec(row: dict, spec: ValueVetoSpec) -> bool:
    if spec.op == "none":
        return True
    value = float(row.get(spec.feature, float("nan")))
    if not np.isfinite(value):
        return False
    if spec.op == "le":
        return value <= float(spec.threshold)
    if spec.op == "ge":
        return value >= float(spec.threshold)
    if spec.op == "between":
        return float(spec.low) <= value <= float(spec.high)
    raise ValueError(f"unknown veto op: {spec.op}")


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _trade_from_row(row: dict, spec: ValueVetoSpec) -> Trade:
    return Trade(
        session=str(row["session"]),
        decision_time=str(row["decision_time"]),
        pnl=float(row["pnl"]),
        score=float(row.get("permission_probability", row.get("score", 0.0))),
        right=str(row["right"]),
        offset=float(row["offset"]),
        strategy=f"{LOOP_ID}:{spec.spec_id}",
    )


def _load_selected_rows(selected_trades_dir: Path, specs: Sequence[ValueVetoSpec]) -> dict[str, dict[str, list[dict]]]:
    sources = sorted({spec.source_variant for spec in specs})
    out: dict[str, dict[str, list[dict]]] = {}
    for source in sources:
        out[source] = {}
        for split in SPLITS:
            path = selected_trades_dir / source / f"selected_trades_{split}.json"
            if not path.exists():
                raise SystemExit(f"missing selected-trade file: {path}")
            out[source][split] = json.loads(path.read_text())
    return out


def _decision_sets(args: argparse.Namespace) -> tuple[dict[str, list], object]:
    paths = _paths_by_split(args.data_dir)
    q4_paths = sorted(args.q4_data_dir.glob("*.pkl"))
    if not q4_paths:
        raise SystemExit(f"no Q4 pkl files found under {args.q4_data_dir}")
    window = _protocol_window(paths, q4_paths)
    variant = _variant()
    cache_dir = None if args.no_decision_cache else args.decision_cache_dir
    market_cache = MarketStructureCache()
    validation = _load_surface_decisions_cached(
        paths["validation"],
        policy_index=args.policy_index,
        variant=variant,
        market_cache=market_cache,
        split="validation",
        cache_dir=cache_dir,
    )
    calibration, selection = split_validation_by_session(validation)
    return (
        {
            "selection": selection,
            "march": _load_surface_decisions_cached(
                paths["test"],
                policy_index=args.policy_index,
                variant=variant,
                market_cache=market_cache,
                split="march",
                cache_dir=cache_dir,
            ),
            "q4": _load_surface_decisions_cached(
                q4_paths,
                policy_index=args.policy_index,
                variant=variant,
                market_cache=market_cache,
                split="q4",
                cache_dir=cache_dir,
            ),
        },
        window,
    )


def _score_selection(summary: dict) -> float:
    stress = summary["stress"]["selection"]["50"]
    if (
        summary.get("selection_trades_median", 0.0) < 8.0
        or summary.get("selection_positive_seed_fraction", 0.0) < 2 / 3
        or stress["pnl_median"] <= 0.0
        or stress["profit_factor_median"] < 1.05
    ):
        return -1_000_000.0
    return (
        stress["pnl_median"]
        + 250.0 * min(stress["profit_factor_median"], 3.0)
        + 25.0 * summary.get("selection_trades_median", 0.0)
    )


def _domain_priority(spec: dict) -> int:
    name = str(spec["name"])
    if name.startswith("gamma_theta"):
        return 5
    if name.startswith("spread"):
        return 4
    if name.startswith("delta"):
        return 3
    if name.startswith("theta"):
        return 2
    if name.startswith("contract_value"):
        return 1
    return 0


def _domain_priority_eligible(summary: dict) -> bool:
    stress = summary["stress"]["selection"]["50"]
    return bool(
        summary["selection_score"] > -1_000_000.0
        and stress["survives"]
        and summary.get("selection_positive_day_fraction_median", 0.0) >= 0.60
        and summary.get("selection_top_day_share_median", 1.0) <= 0.45
    )


def _summarize_split(seed_rows: Sequence[dict], split: str) -> dict:
    metrics = [row["metrics_by_split"][split] for row in seed_rows]
    random_pnl = [float(row["random_baseline_by_split"][split]["total_pnl_median"]) for row in seed_rows]
    pnl = [float(row["total_pnl"]) for row in metrics]
    pf = [float(row["profit_factor"]) for row in metrics]
    trades = [float(row["trades"]) for row in metrics]
    return {
        "pnl_by_seed": pnl,
        "pnl_median": _median(pnl),
        "profit_factor_by_seed": pf,
        "profit_factor_median": _median(pf),
        "trades_by_seed": trades,
        "trades_median": _median(trades),
        "positive_seed_fraction": float(np.mean([x > 0 for x in pnl])) if pnl else 0.0,
        "positive_day_fraction_median": _median([float(row["positive_day_fraction"]) for row in metrics]),
        "top_day_share_median": _median([float(row["top_day_profit_share"]) for row in metrics]),
        "random_pnl_by_seed": random_pnl,
        "random_pnl_median": _median(random_pnl),
        "edge_vs_random_median": _median(pnl) - _median(random_pnl),
    }


def _summarize_stress(seed_rows: Sequence[dict], split: str) -> dict:
    out = {}
    for cost in ("25", "50", "100"):
        metrics = [row["slippage_stress_by_split"][split][cost] for row in seed_rows]
        pnl = [float(row["total_pnl"]) for row in metrics]
        pf = [float(row["profit_factor"]) for row in metrics]
        out[cost] = {
            "pnl_by_seed": pnl,
            "pnl_median": _median(pnl),
            "profit_factor_by_seed": pf,
            "profit_factor_median": _median(pf),
            "survives": bool(_median(pnl) > 0.0 and _median(pf) >= 1.05),
        }
    return out


def _spec_summary(seed_rows: Sequence[dict], spec: ValueVetoSpec) -> dict:
    summary = {"spec": asdict(spec) | {"spec_id": spec.spec_id}, "runs": len(seed_rows)}
    for split in SPLITS:
        split_summary = _summarize_split(seed_rows, split)
        for key, value in split_summary.items():
            if isinstance(value, (int, float, bool)):
                summary[f"{split}_{key}"] = value
            else:
                summary[f"{split}_{key}"] = value
    stress = {split: _summarize_stress(seed_rows, split) for split in SPLITS}
    summary["stress"] = stress
    summary["selection_score"] = _score_selection(summary)
    summary["passes_broad_data_purchase_gate"] = bool(
        summary.get("selection_pnl_median", 0.0) > 0.0
        and summary.get("selection_trades_median", 0.0) >= 8.0
        and summary.get("march_pnl_median", 0.0) > 0.0
        and summary.get("march_profit_factor_median", 0.0) >= 1.05
        and summary.get("march_positive_seed_fraction", 0.0) >= 2 / 3
        and summary.get("march_trades_median", 0.0) >= 8.0
        and summary.get("march_edge_vs_random_median", 0.0) > 0.0
        and summary.get("q4_pnl_median", 0.0) > 0.0
        and summary.get("q4_profit_factor_median", 0.0) >= 1.05
        and summary.get("q4_positive_seed_fraction", 0.0) >= 2 / 3
        and summary.get("q4_trades_median", 0.0) >= 8.0
        and summary.get("q4_edge_vs_random_median", 0.0) > 0.0
        and stress["march"]["50"]["survives"]
        and stress["q4"]["50"]["survives"]
    )
    summary["purchase_gate_score"] = (
        summary.get("march_pnl_median", 0.0)
        + summary.get("q4_pnl_median", 0.0)
        + summary["stress"]["march"]["50"]["pnl_median"]
        + summary["stress"]["q4"]["50"]["pnl_median"]
    )
    return summary


def _run_spec(
    *,
    spec: ValueVetoSpec,
    selected_rows: dict[str, dict[str, list[dict]]],
    decision_sets: dict[str, list],
    window_id: str,
    cooldown: int,
) -> tuple[list[dict], dict[str, list[dict]]]:
    seed_rows = []
    exported_rows = {split: [] for split in SPLITS}
    for seed in SEEDS:
        effective_seed = window_seed(seed, window_id)
        metrics_by_split = {}
        random_by_split = {}
        stress_by_split = {}
        for split in SPLITS:
            rows = [
                row
                for row in selected_rows[spec.source_variant][split]
                if int(row["seed"]) == seed and _passes_spec(row, spec)
            ]
            trades = [_trade_from_row(row, spec) for row in rows]
            metrics_by_split[split] = metrics_with_concentration(trades)
            random_by_split[split] = summarize_random_baseline(
                decision_sets[split],
                trial=PERMISSION_TRIAL,
                cooldown_minutes=cooldown,
                seed=effective_seed,
                target_trade_count=len(trades),
            )
            stress_by_split[split] = {
                str(cost): metrics_with_concentration(stress_trades(trades, extra_cost_per_trade=float(cost)))
                for cost in (25, 50, 100)
            }
            for row in rows:
                exported_rows[split].append({"value_veto_spec": spec.spec_id, **row})
        seed_rows.append(
            {
                "seed": int(seed),
                "effective_seed": int(effective_seed),
                "spec": asdict(spec) | {"spec_id": spec.spec_id},
                "metrics_by_split": metrics_by_split,
                "random_baseline_by_split": random_by_split,
                "slippage_stress_by_split": stress_by_split,
            }
        )
    return seed_rows, exported_rows


def _write_markdown(path: Path, payload: dict) -> None:
    champion = payload["locked_champion"]
    lines = [
        "# A+ Value Veto Protocol 006",
        "",
        payload["framing"],
        "",
        f"Locked champion: `{champion['spec']['spec_id']}`",
        f"Selection score: `{champion['selection_score']:.1f}`",
        f"Passes broad-data-purchase gate: `{champion['passes_broad_data_purchase_gate']}`",
        f"Passing candidates in full screen: `{payload['aggregate']['pass_count']}`",
        "",
        "## Locked Champion",
        "",
        "| Split | Median PnL | Median PF | Trades | Positive Seeds | Edge vs Random | +50 PnL/PF |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        stress = champion["stress"][split]["50"]
        lines.append(
            f"| {split} | {champion[f'{split}_pnl_median']:.0f} | "
            f"{champion[f'{split}_profit_factor_median']:.3f} | "
            f"{champion[f'{split}_trades_median']:.0f} | "
            f"{champion[f'{split}_positive_seed_fraction']:.2f} | "
            f"{champion[f'{split}_edge_vs_random_median']:.0f} | "
            f"{stress['pnl_median']:.0f}/{stress['profit_factor_median']:.3f} |"
        )
    lines += [
        "",
        "## Top Selection-Locked Candidates",
        "",
        "| Rank | Spec | Pass | Selection Score | March +50 | Q4 +50 | March PnL/PF | Q4 PnL/PF |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(payload["aggregate"]["selection_ranked"][:15], start=1):
        lines.append(
            f"| {idx} | `{row['spec']['spec_id']}` | {row['passes_broad_data_purchase_gate']} | "
            f"{row['selection_score']:.1f} | "
            f"{row['stress']['march']['50']['pnl_median']:.0f} | "
            f"{row['stress']['q4']['50']['pnl_median']:.0f} | "
            f"{row['march_pnl_median']:.0f}/{row['march_profit_factor_median']:.3f} | "
            f"{row['q4_pnl_median']:.0f}/{row['q4_profit_factor_median']:.3f} |"
        )
    lines += ["", "## Interpretation", "", payload["interpretation"]]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    specs = registered_value_veto_specs()
    selected_rows = _load_selected_rows(args.selected_trades_dir, specs)
    decision_sets, window = _decision_sets(args)
    _, cooldown = POLICY_META[args.policy_index]

    rows = []
    summaries = []
    exported_by_spec: dict[str, dict[str, list[dict]]] = {}
    for spec in specs:
        seed_rows, exported = _run_spec(
            spec=spec,
            selected_rows=selected_rows,
            decision_sets=decision_sets,
            window_id=window.window_id,
            cooldown=cooldown,
        )
        rows.extend(seed_rows)
        summaries.append(_spec_summary(seed_rows, spec))
        exported_by_spec[spec.spec_id] = exported

    if args.selection_mode == "domain-priority":
        selection_ranked = sorted(
            summaries,
            key=lambda row: (
                _domain_priority_eligible(row),
                _domain_priority(row["spec"]),
                row["selection_score"],
                row.get("selection_positive_day_fraction_median", 0.0),
                -row.get("selection_top_day_share_median", 1.0),
                row["spec"]["spec_id"],
            ),
            reverse=True,
        )
    else:
        selection_ranked = sorted(
            summaries,
            key=lambda row: (
                row["selection_score"],
                row.get("selection_pnl_median", 0.0),
                row.get("selection_profit_factor_median", 0.0),
                row["spec"]["spec_id"],
            ),
            reverse=True,
        )
    if args.locked_spec_id:
        matches = [row for row in summaries if row["spec"]["spec_id"] == args.locked_spec_id]
        if not matches:
            raise SystemExit(f"unknown locked spec id: {args.locked_spec_id}")
        locked_champion = matches[0]
    else:
        locked_champion = selection_ranked[0]
    holdout_ranked = sorted(
        summaries,
        key=lambda row: (
            row["passes_broad_data_purchase_gate"],
            row["purchase_gate_score"],
            row["selection_score"],
            row["spec"]["spec_id"],
        ),
        reverse=True,
    )
    payload = {
        "loop_id": LOOP_ID,
        "framing": (
            "Protocol 006 locks a deterministic Greek/contract-value veto using only the selection split "
            "under +$50/trade stress, then audits the locked policy on March 2026 and frozen Q4 2025."
        ),
        "window": asdict(window) | {"window_id": window.window_id},
        "policy_index": int(args.policy_index),
        "permission_trial": asdict(PERMISSION_TRIAL) | {"config_id": PERMISSION_TRIAL.config_id},
        "source_selected_trades_dir": str(args.selected_trades_dir),
        "selection_mode": args.selection_mode,
        "locked_spec_id": args.locked_spec_id,
        "registered_value_veto_specs": [asdict(spec) | {"spec_id": spec.spec_id} for spec in specs],
        "promotion_gate": [
            "locked champion is selected by selection-split +$50 stress score only",
            "positive selection, March, and frozen Q4 medians",
            "March and Q4 beat matched random by median",
            "March and Q4 survive +$50/trade stress",
            "March and Q4 have at least 2/3 positive seeds",
            "at least 8 median trades in selection, March, and Q4",
        ],
        "locked_champion": locked_champion,
        "aggregate": {
            "pass_count": int(sum(row["passes_broad_data_purchase_gate"] for row in summaries)),
            "selection_ranked": selection_ranked,
            "holdout_ranked": holdout_ranked,
            "by_spec": {row["spec"]["spec_id"]: row for row in summaries},
        },
        "rows": rows,
        "interpretation": (
            "If the locked champion passes, the evidence supports buying a modest additional validation block, "
            "not scaling directly to live trading. Thin +50 margins should be treated as fragile."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_json = args.out_dir / "report.json"
    report_md = args.out_dir / "report.md"
    report_json.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    _write_markdown(report_md, payload)

    champion_dir = args.out_dir / "selected_trades" / locked_champion["spec"]["spec_id"]
    champion_dir.mkdir(parents=True, exist_ok=True)
    for split, trade_rows in exported_by_spec[locked_champion["spec"]["spec_id"]].items():
        (champion_dir / f"selected_trades_{split}.json").write_text(
            json.dumps(trade_rows, indent=2, allow_nan=True) + "\n"
        )
    print(report_json)
    print(report_md)
    print(json.dumps(locked_champion, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
