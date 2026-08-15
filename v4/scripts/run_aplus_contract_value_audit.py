"""A+ contract value audit for SPXW 0DTE timing patterns.

Pickles' "A+ setup" framing is not just a chart pattern. For long 0DTE
contracts, the expression has to be worth paying the spread for. This audit
adds Greek/value buckets on top of the transferable timing patterns:

* sufficient delta for the intended directional expression
* enough gamma/convexity relative to premium and theta burden
* spread tax low enough to overcome
* breakeven distance reasonable versus current movement scale
* IV/premium not obviously overpriced relative to the local sample

The audit remains non-neural and keeps March/Q4 frozen.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from v4.model.hypothesis_protocol import MarketStructureCache
from v4.model.supervised_pilot import session_from_path
from v4.scripts.run_edge_existence_audit import (
    _load_1s_audit,
    _normalized_audit,
    _quantile_bucket,
    _safe_float,
)
from v4.scripts.run_entry_timing_pattern_audit import (
    RULE_TEMPLATES as PATTERN_RULE_TEMPLATES,
    _discover_rules as _discover_pattern_rules,
    _load_pattern_candidates,
    _pattern_summary,
    _simulate_rules,
)
from v4.scripts.train_spxw_supervised_pilot import POLICY_META


VALUE_RULE_TEMPLATES: tuple[tuple[str, ...], ...] = (
    ("side", "entry_pattern", "value_grade"),
    ("side", "entry_pattern", "delta_bucket"),
    ("side", "entry_pattern", "convexity_bucket"),
    ("side", "entry_pattern", "theta_burden_bucket"),
    ("side", "entry_pattern", "spread_tax_bucket"),
    ("side", "entry_pattern", "breakeven_atr_bucket"),
    ("side", "entry_pattern", "iv_relative_bucket"),
    ("side", "entry_pattern", "moneyness_bucket", "value_grade"),
    ("side", "entry_pattern", "moneyness_bucket", "delta_bucket"),
    ("side", "entry_pattern", "moneyness_bucket", "convexity_bucket"),
    ("side", "entry_pattern", "moneyness_bucket", "theta_burden_bucket"),
    ("side", "entry_pattern", "moneyness_bucket", "spread_tax_bucket"),
    ("side", "entry_pattern", "premium_bucket", "delta_bucket"),
    ("side", "entry_pattern", "premium_bucket", "value_grade"),
    ("side", "entry_pattern", "value_grade", "spread_tax_bucket"),
    ("side", "entry_pattern", "value_grade", "theta_burden_bucket"),
    ("side", "entry_pattern", "value_grade", "breakeven_atr_bucket"),
    ("side", "entry_pattern", "moneyness_bucket", "value_grade", "spread_tax_bucket"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_derived"))
    parser.add_argument("--q4-data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q4_2025"))
    parser.add_argument("--normalized-dir", type=Path, default=Path("v4/normalized"))
    parser.add_argument("--cbbo-1s-audit", type=Path, default=Path("v4/audit/cbbo_1m_vs_1s_audit_summary.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("v4/audit/aplus_contract_value"))
    parser.add_argument("--policy-indexes", nargs="*", type=int, default=sorted(POLICY_META), choices=sorted(POLICY_META))
    parser.add_argument("--min-discovery-candidates", type=int, default=80)
    parser.add_argument("--min-selection-candidates", type=int, default=20)
    parser.add_argument("--max-rules-per-policy", type=int, default=180)
    parser.add_argument("--max-simulated-rules", type=int, default=420)
    parser.add_argument("--max-trades-per-day", type=int, default=1)
    parser.add_argument("--random-runs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260430)
    return parser.parse_args()


def _paths(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("*.pkl"))


def _policy_hold_minutes(policy_index: int) -> int:
    name = POLICY_META[int(policy_index)][0]
    # Names are ask_to_bid_stopXX_targetYY_holdNNm.
    try:
        return int(name.rsplit("hold", 1)[1].removesuffix("m"))
    except (IndexError, ValueError):
        return int(POLICY_META[int(policy_index)][1])


def _bucket_fixed(values: pd.Series, bins: Sequence[float], labels: Sequence[str]) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce")
    out = pd.Series(np.full(len(arr), labels[-1], dtype=object), index=values.index)
    lower = -np.inf
    for edge, label in zip(bins, labels):
        out[(arr > lower) & (arr <= edge)] = label
        lower = edge
    out[arr > lower] = labels[-1]
    out[arr.isna()] = "unknown"
    return out


def _reference(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["split"].isin(["train", "calibration"])].copy()


def _add_contract_value_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    ref = _reference(frame)
    ask = pd.to_numeric(frame["ask"], errors="coerce").clip(lower=0.01)
    spread = pd.to_numeric(frame["spread"], errors="coerce").clip(lower=0.0)
    iv = pd.to_numeric(frame["iv"], errors="coerce")
    abs_delta = pd.to_numeric(frame["delta"], errors="coerce").abs()
    gamma = pd.to_numeric(frame["gamma"], errors="coerce").abs()
    theta = pd.to_numeric(frame["theta"], errors="coerce").abs()
    breakeven = pd.to_numeric(frame.get("breakeven_distance", pd.Series(np.nan, index=frame.index)), errors="coerce")
    spx = pd.to_numeric(frame["spx_close"], errors="coerce").abs().clip(lower=1.0)
    atr_points = (pd.to_numeric(frame["atr15_pct"], errors="coerce").abs() * spx).clip(lower=0.50)
    hold_minutes = frame["policy_index"].map(lambda x: _policy_hold_minutes(int(x))).astype(float)

    frame["abs_delta"] = abs_delta
    frame["spread_tax"] = spread / ask
    frame["theta_burden_hold"] = (theta * (hold_minutes / 390.0)) / ask
    frame["convexity_per_premium"] = (0.5 * gamma * (atr_points**2)) / ask
    frame["delta_atr_capture"] = (abs_delta * atr_points) / ask
    frame["breakeven_atr"] = (breakeven / atr_points).replace([np.inf, -np.inf], np.nan)
    frame["gamma_theta_ratio"] = gamma / (theta / 390.0 + 1e-6)

    ref_with = _reference(frame)
    frame["iv_relative_bucket"] = _quantile_bucket(frame["iv"], ref_with["iv"], "iv")
    frame["convexity_bucket"] = _quantile_bucket(frame["convexity_per_premium"], ref_with["convexity_per_premium"], "convexity")
    frame["delta_capture_bucket"] = _quantile_bucket(frame["delta_atr_capture"], ref_with["delta_atr_capture"], "delta_capture")
    frame["gamma_theta_bucket"] = _quantile_bucket(frame["gamma_theta_ratio"], ref_with["gamma_theta_ratio"], "gamma_theta")

    # For costs, lower is better. Use explicit labels so the report reads like a trader would read it.
    frame["theta_burden_bucket"] = _bucket_fixed(
        frame["theta_burden_hold"],
        bins=(0.015, 0.040, 0.080),
        labels=("theta_light", "theta_ok", "theta_heavy", "theta_extreme"),
    )
    frame["spread_tax_bucket"] = _bucket_fixed(
        frame["spread_tax"],
        bins=(0.030, 0.070, 0.140),
        labels=("spread_tight", "spread_ok", "spread_expensive", "spread_bad"),
    )
    frame["breakeven_atr_bucket"] = _bucket_fixed(
        frame["breakeven_atr"],
        bins=(0.50, 1.25, 2.50),
        labels=("breakeven_near", "breakeven_reachable", "breakeven_stretched", "breakeven_far"),
    )
    frame["delta_bucket"] = _bucket_fixed(
        frame["abs_delta"],
        bins=(0.15, 0.30, 0.55, 0.75),
        labels=("delta_lottery", "delta_low", "delta_balanced", "delta_high", "delta_heavy"),
    )

    # A plain current-feature value score. It is not a label; it only ranks
    # same-minute contract expressions by whether the premium is worth paying.
    breakeven_cost = frame["breakeven_atr"].fillna(frame["breakeven_atr"].median()).clip(0, 6.0)
    cost = (
        1.4 * frame["spread_tax"].clip(0, 1.0)
        + 1.2 * frame["theta_burden_hold"].clip(0, 1.0)
        + 0.25 * breakeven_cost
    )
    benefit = (
        0.65 * frame["delta_atr_capture"].clip(0, 4.0)
        + 0.85 * frame["convexity_per_premium"].clip(0, 4.0)
        + 0.15 * frame["gamma_theta_ratio"].clip(0, 20.0) / 20.0
    )
    frame["contract_value_score"] = (benefit - cost).replace([np.inf, -np.inf], np.nan).fillna(-9.0)
    ref_score = _reference(frame)["contract_value_score"]
    q40, q70, q88 = np.nanquantile(ref_score[np.isfinite(ref_score)], [0.40, 0.70, 0.88])
    frame["value_grade"] = np.select(
        [
            frame["contract_value_score"] >= q88,
            frame["contract_value_score"] >= q70,
            frame["contract_value_score"] >= q40,
        ],
        ["A_plus_value", "A_value", "B_value"],
        default="C_or_overpay",
    )
    frame["overpay_flag"] = (
        (frame["spread_tax_bucket"].isin(["spread_expensive", "spread_bad"]))
        | (frame["theta_burden_bucket"].isin(["theta_heavy", "theta_extreme"]))
        | (frame["breakeven_atr_bucket"].isin(["breakeven_far"]))
    )
    frame["quality_score"] = frame["quality_score"] + 0.35 * frame["contract_value_score"]
    return frame


def _group_metrics(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    work = frame[list(columns) + ["pnl"]].copy()
    work["gross_profit"] = work["pnl"].clip(lower=0)
    work["gross_loss"] = -work["pnl"].clip(upper=0)
    work["wins"] = (work["pnl"] > 0).astype(float)
    grouped = work.groupby(list(columns), observed=True, dropna=False).agg(
        count=("pnl", "size"),
        total=("pnl", "sum"),
        avg=("pnl", "mean"),
        median=("pnl", "median"),
        gross_profit=("gross_profit", "sum"),
        gross_loss=("gross_loss", "sum"),
        win_rate=("wins", "mean"),
    )
    grouped["pf"] = grouped["gross_profit"] / grouped["gross_loss"].replace(0.0, np.nan)
    grouped["pf"] = grouped["pf"].replace([np.inf, -np.inf], np.nan).fillna(999.0)
    return grouped.reset_index()


def _discover_value_rules(
    frame: pd.DataFrame,
    *,
    min_discovery_candidates: int,
    min_selection_candidates: int,
    max_per_policy: int,
) -> list[dict]:
    discovery = frame[frame["split"].isin(["train", "calibration"])]
    selection = frame[frame["split"] == "selection"]
    rows: list[dict] = []
    templates = VALUE_RULE_TEMPLATES + PATTERN_RULE_TEMPLATES
    for policy_index in sorted(frame["policy_index"].unique()):
        policy_rows: list[dict] = []
        d_policy = discovery[discovery["policy_index"] == policy_index]
        s_policy = selection[selection["policy_index"] == policy_index]
        for columns in templates:
            d_metrics = _group_metrics(d_policy, columns)
            s_metrics = _group_metrics(s_policy, columns)
            if d_metrics.empty or s_metrics.empty:
                continue
            merged = d_metrics.merge(s_metrics, on=list(columns), suffixes=("_discovery", "_selection"))
            merged = merged[
                (merged["count_discovery"] >= min_discovery_candidates)
                & (merged["count_selection"] >= min_selection_candidates)
                & (merged["avg_discovery"] > 0)
                & (merged["avg_selection"] > 0)
                & (merged["pf_discovery"] >= 1.05)
                & (merged["pf_selection"] >= 1.05)
            ]
            for record in merged.to_dict("records"):
                values = tuple(str(record[col]) for col in columns)
                raw = f"policy{int(policy_index)}|" + "|".join(f"{c}={v}" for c, v in zip(columns, values))
                rule_id = __import__("hashlib").sha1(raw.encode("utf-8")).hexdigest()[:12]
                policy_rows.append(
                    {
                        "rule_id": rule_id,
                        "rule": raw,
                        "policy_index": int(policy_index),
                        "columns": list(columns),
                        "values": list(values),
                        "discovery_candidates": int(record["count_discovery"]),
                        "selection_candidates": int(record["count_selection"]),
                        "discovery_avg_candidate_pnl": float(record["avg_discovery"]),
                        "selection_avg_candidate_pnl": float(record["avg_selection"]),
                        "discovery_pf_candidate": float(record["pf_discovery"]),
                        "selection_pf_candidate": float(record["pf_selection"]),
                        "selection_total_candidate_pnl": float(record["total_selection"]),
                    }
                )
        deduped = {}
        for row in sorted(
            policy_rows,
            key=lambda r: (
                any(col in r["columns"] for col in ("value_grade", "theta_burden_bucket", "spread_tax_bucket", "convexity_bucket")),
                r["selection_avg_candidate_pnl"],
                r["selection_pf_candidate"],
                r["selection_candidates"],
            ),
            reverse=True,
        ):
            deduped.setdefault(row["rule_id"], row)
        rows.extend(list(deduped.values())[:max_per_policy])
    return rows


def _value_summary(frame: pd.DataFrame) -> dict:
    rows = []
    for grade, group in frame.groupby("value_grade", observed=True):
        rows.append(
            {
                "value_grade": str(grade),
                "rows": int(len(group)),
                "median_contract_value_score": float(group["contract_value_score"].median()),
                "median_abs_delta": float(group["abs_delta"].median()),
                "median_theta_burden_hold": float(group["theta_burden_hold"].median()),
                "median_spread_tax": float(group["spread_tax"].median()),
                "median_breakeven_atr": float(group["breakeven_atr"].median()),
            }
        )
    rows.sort(key=lambda r: r["median_contract_value_score"], reverse=True)
    return {"grades": rows}


def _write_markdown(path: Path, payload: dict) -> None:
    champion = payload["champion"]
    lines = [
        "# A+ Contract Value Audit",
        "",
        "Greek-aware audit for whether timing patterns improve when the contract is worth paying the spread for.",
        "",
        f"Transfer gate pass count: `{payload['transfer_gate_pass_count']}`",
        f"Discovered value rules: `{payload['discovered_rule_count']}`",
        f"Simulated value rules: `{payload['simulated_rule_count']}`",
        "",
        "## Champion",
        "",
    ]
    if champion:
        lines.extend(
            [
                f"Best value-aware rule: `{champion['rule']}`",
                f"Cross-regime positive bucket fraction: `{champion['cross_regime_positive_bucket_fraction']:.2f}`",
                "",
                "| Split | Trades | PnL | PF | DD | Positive Days | Random Same-Time PnL |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for split in ("selection", "march", "q4"):
            metrics = champion["metrics_by_split"][split]
            random = champion["random_same_time_by_split"][split]
            lines.append(
                f"| {split} | {metrics['trades']} | {metrics['total_pnl']:.0f} | "
                f"{metrics['profit_factor']:.3f} | {metrics['max_drawdown']:.0f} | "
                f"{metrics['positive_day_fraction']:.2f} | {random['total_pnl_median']:.0f} |"
            )
    lines.extend(
        [
            "",
            "## Top Value-Aware Rules",
            "",
            "| Rank | Rule | Sel PnL/PF | March PnL/PF | Q4 PnL/PF | Bucket+ | Pass |",
            "|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    for idx, row in enumerate(payload["top_simulated_rules"][:25], start=1):
        sel = row["metrics_by_split"]["selection"]
        march = row["metrics_by_split"]["march"]
        q4 = row["metrics_by_split"]["q4"]
        lines.append(
            f"| {idx} | `{row['rule']}` | {sel['total_pnl']:.0f}/{sel['profit_factor']:.3f} | "
            f"{march['total_pnl']:.0f}/{march['profit_factor']:.3f} | "
            f"{q4['total_pnl']:.0f}/{q4['profit_factor']:.3f} | "
            f"{row['cross_regime_positive_bucket_fraction']:.2f} | {row['transfer_gate_pass']} |"
        )
    lines.extend(
        [
            "",
            "## Value Grades",
            "",
            "| Grade | Rows | Median Score | Median Delta | Theta Burden | Spread Tax | Breakeven ATR |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["value_summary"]["grades"]:
        lines.append(
            f"| {row['value_grade']} | {row['rows']} | {row['median_contract_value_score']:.3f} | "
            f"{row['median_abs_delta']:.3f} | {row['median_theta_burden_hold']:.3f} | "
            f"{row['median_spread_tax']:.3f} | {row['median_breakeven_atr']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            payload["interpretation"],
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    market_cache = MarketStructureCache()
    all_paths = _paths(args.data_dir) + _paths(args.q4_data_dir)
    sessions = [session_from_path(path) for path in all_paths]
    frame = _load_pattern_candidates(all_paths, policies=args.policy_indexes, market_cache=market_cache)
    frame = _add_contract_value_features(frame)
    discovered = _discover_value_rules(
        frame,
        min_discovery_candidates=args.min_discovery_candidates,
        min_selection_candidates=args.min_selection_candidates,
        max_per_policy=args.max_rules_per_policy,
    )
    simulated = _simulate_rules(
        frame,
        discovered,
        max_rules=args.max_simulated_rules,
        max_trades_per_day=args.max_trades_per_day,
        random_runs=args.random_runs,
        seed=args.seed,
    )
    pass_count = sum(1 for row in simulated if row["transfer_gate_pass"])
    champion = simulated[0] if simulated else None
    payload = {
        "audit_id": "v4_aplus_contract_value_audit_001",
        "framing": "Test whether Greek/value-aware contract expression improves transferable timing patterns.",
        "args": {
            "data_dir": str(args.data_dir),
            "q4_data_dir": str(args.q4_data_dir),
            "policy_indexes": args.policy_indexes,
            "min_discovery_candidates": args.min_discovery_candidates,
            "min_selection_candidates": args.min_selection_candidates,
            "max_trades_per_day": args.max_trades_per_day,
            "random_runs": args.random_runs,
        },
        "contract_value_features": [
            "abs_delta",
            "delta_atr_capture",
            "convexity_per_premium",
            "theta_burden_hold",
            "spread_tax",
            "breakeven_atr",
            "gamma_theta_ratio",
            "iv_relative_bucket",
            "value_grade",
        ],
        "pattern_summary": _pattern_summary(frame),
        "value_summary": _value_summary(frame),
        "data_summary": {
            "normalized": _normalized_audit(args.normalized_dir, sessions),
            "cbbo_1m_vs_1s": _load_1s_audit(args.cbbo_1s_audit),
        },
        "discovered_rule_count": len(discovered),
        "simulated_rule_count": len(simulated),
        "transfer_gate_pass_count": pass_count,
        "champion": champion,
        "top_simulated_rules": simulated[:60],
        "top_discovered_rules": discovered[:60],
        "interpretation": (
            "A positive pass count means the timing patterns are improved or confirmed by contract value conditions. "
            "These A+ filters should become explicit model targets: the network should learn pattern quality and contract "
            "value separately, then only trade when both agree. Results remain research leads, not live-trading permission."
            if pass_count
            else "No Greek/value filter survived the transfer gate. Treat contract value features as model inputs but not hard filters yet."
        ),
    }
    (args.out_dir / "report.json").write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    _write_markdown(args.out_dir / "report.md", payload)
    print(
        json.dumps(
            {
                "discovered_rule_count": len(discovered),
                "simulated_rule_count": len(simulated),
                "transfer_gate_pass_count": pass_count,
                "champion": champion["rule"] if champion else None,
                "champion_selection_pnl": champion["metrics_by_split"]["selection"]["total_pnl"] if champion else None,
                "champion_march_pnl": champion["metrics_by_split"]["march"]["total_pnl"] if champion else None,
                "champion_q4_pnl": champion["metrics_by_split"]["q4"]["total_pnl"] if champion else None,
            },
            indent=2,
        )
    )
    print(f"WROTE {args.out_dir / 'report.json'}")
    print(f"WROTE {args.out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
