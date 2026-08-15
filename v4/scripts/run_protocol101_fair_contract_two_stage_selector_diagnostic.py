"""Diagnose compound live-causal selector gates before training more models.

The candidate-edge stability audit showed broad univariate pockets of causal
opportunity, but previous learned attempts were still too fragile. This packet
tests predeclared compound first-stage gates using labels only as an offline
diagnostic. It does not train, tune thresholds, contact brokers/vendors,
download data, change defaults, submit orders, or promote a model.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from v4.model.supervised_pilot import (
    DecisionCandidates,
    Trade,
    entry_filter_mask,
    load_decisions,
    metrics_for_trades,
)
from v4.scripts.run_protocol101_fair_contract_candidate_edge_stability import (
    LabelStats,
    affordable_mask,
    candidate_buckets,
)
from v4.scripts.run_protocol101_fair_contract_training_runner import (
    load_json,
    paths_by_split,
)


DEFAULT_DESIGN = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_expanded_jul_dec2025_128_q1_design/summary.json"
)
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_two_stage_selector_diagnostic_jul_dec128_q1"
)
BASE_ENTRY_FILTER = "put_near_after_0940_vwap_m2_10"


@dataclass(frozen=True)
class GateSpec:
    """A live-causal first-stage selector gate evaluated before model scoring."""

    name: str
    description: str
    entry_filter: str = BASE_ENTRY_FILTER
    predicates: tuple[str, ...] = ()


def default_gate_specs() -> tuple[GateSpec, ...]:
    """Return predeclared gates from stable label buckets, not fitted models."""
    return (
        GateSpec(
            name="base_put_near_after0940_vwap_m2_10",
            description="Existing put-near 09:40+ VWAP -2/+10 causal pocket.",
        ),
        GateSpec(
            name="base_plus_omar_neg",
            description="Base pocket only when OMAR is negative.",
            predicates=("omar_neg",),
        ),
        GateSpec(
            name="base_plus_range_20_45",
            description="Base pocket only during moderate 20-45 point intraday range.",
            predicates=("range_20_45",),
        ),
        GateSpec(
            name="base_plus_midday",
            description="Base pocket only from 11:30-13:29 ET.",
            predicates=("midday_1130_1329",),
        ),
        GateSpec(
            name="base_plus_morning",
            description="Base pocket only from 10:00-11:29 ET.",
            predicates=("morning_1000_1129",),
        ),
        GateSpec(
            name="base_plus_premium_7_5_to_15",
            description="Base pocket only for 7.5-15.0 ask premium candidates.",
            predicates=("premium_7_5_to_15",),
        ),
        GateSpec(
            name="base_plus_premium_gte_15",
            description="Base pocket only for 15.0+ ask premium candidates.",
            predicates=("premium_gte_15",),
        ),
        GateSpec(
            name="base_plus_premium_gte_7_5",
            description="Base pocket only for 7.5+ ask premium candidates.",
            predicates=("premium_gte_7_5",),
        ),
        GateSpec(
            name="base_plus_near_vwap",
            description="Base pocket only when SPX is within +/-2 points of VWAP.",
            predicates=("near_vwap",),
        ),
        GateSpec(
            name="base_plus_above_vwap_2_10",
            description="Base pocket only when SPX is 2-10 points above VWAP.",
            predicates=("above_vwap_2_10",),
        ),
        GateSpec(
            name="base_plus_omar_neg_range_20_45",
            description="Base pocket with negative OMAR and moderate range.",
            predicates=("omar_neg", "range_20_45"),
        ),
        GateSpec(
            name="base_plus_omar_neg_midday",
            description="Base pocket with negative OMAR during 11:30-13:29 ET.",
            predicates=("omar_neg", "midday_1130_1329"),
        ),
        GateSpec(
            name="base_plus_omar_neg_premium_7_5_to_15",
            description="Base pocket with negative OMAR and mid-premium candidates.",
            predicates=("omar_neg", "premium_7_5_to_15"),
        ),
        GateSpec(
            name="base_plus_midday_premium_7_5_to_15",
            description="Base pocket during midday with mid-premium candidates.",
            predicates=("midday_1130_1329", "premium_7_5_to_15"),
        ),
        GateSpec(
            name="base_plus_range_20_45_premium_7_5_to_15",
            description="Base pocket with moderate range and mid-premium candidates.",
            predicates=("range_20_45", "premium_7_5_to_15"),
        ),
    )


def gate_specs_by_name() -> dict[str, GateSpec]:
    return {spec.name: spec for spec in default_gate_specs()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--policy-index", type=int, default=0)
    parser.add_argument("--positive-label-threshold", type=float, default=20.0)
    parser.add_argument("--min-candidates", type=int, default=100)
    parser.add_argument("--oracle-max-trades-per-session", type=int, default=3)
    parser.add_argument("--oracle-cooldown-minutes", type=int, default=10)
    parser.add_argument(
        "--gates",
        default="",
        help="Optional comma-separated gate names. Defaults to all predeclared gates.",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def predicate_matches(predicate: str, buckets: dict[str, str]) -> bool:
    mapping = {
        "omar_neg": ("omar_side", "omar_neg"),
        "omar_pos": ("omar_side", "omar_pos"),
        "range_20_45": ("range_bucket", "range_20_45"),
        "range_45_80": ("range_bucket", "range_45_80"),
        "morning_1000_1129": ("time_bucket", "morning_1000_1129"),
        "midday_1130_1329": ("time_bucket", "midday_1130_1329"),
        "near_vwap": ("vwap_gap_bucket", "near_vwap"),
        "above_vwap_2_10": ("vwap_gap_bucket", "above_vwap_2_10"),
        "premium_7_5_to_15": ("premium_bucket", "7_5_to_15"),
        "premium_gte_15": ("premium_bucket", "gte_15"),
        "mom15_pos": ("momentum15_side", "mom15_pos"),
        "mom15_neg": ("momentum15_side", "mom15_neg"),
    }
    if predicate == "premium_gte_7_5":
        return str(buckets.get("premium_bucket")) in {"7_5_to_15", "gte_15"}
    if predicate not in mapping:
        raise ValueError(f"unknown gate predicate: {predicate}")
    key, expected = mapping[predicate]
    return str(buckets.get(key)) == expected


def candidate_allowed_by_gate(decision: DecisionCandidates, idx: int, spec: GateSpec) -> bool:
    buckets = candidate_buckets(decision, int(idx))
    return all(predicate_matches(predicate, buckets) for predicate in spec.predicates)


def gate_mask(decision: DecisionCandidates, spec: GateSpec) -> np.ndarray:
    base = entry_filter_mask(decision, spec.entry_filter) & affordable_mask(decision)
    if not spec.predicates:
        return np.asarray(base, dtype=bool)
    allowed = np.zeros(len(decision.labels), dtype=bool)
    for idx in np.flatnonzero(base):
        allowed[int(idx)] = candidate_allowed_by_gate(decision, int(idx), spec)
    return allowed


def aggregate_gate_labels(
    decisions_by_split: dict[str, list[DecisionCandidates]],
    *,
    gates: Sequence[GateSpec],
    positive_label_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, decisions in decisions_by_split.items():
        for spec in gates:
            stats = LabelStats()
            decisions_with_candidates = 0
            for decision in decisions:
                labels = np.asarray(decision.labels, dtype=np.float32)
                allowed = gate_mask(decision, spec) & np.isfinite(labels)
                if not allowed.any():
                    continue
                decisions_with_candidates += 1
                for idx in np.flatnonzero(allowed):
                    stats.add(float(labels[int(idx)]), positive_threshold=positive_label_threshold)
            rows.append(
                {
                    "split": split,
                    "gate": spec.name,
                    "description": spec.description,
                    "entry_filter": spec.entry_filter,
                    "predicates": ",".join(spec.predicates),
                    "decision_count": len(decisions),
                    "decisions_with_candidates": decisions_with_candidates,
                    **stats.to_row(),
                }
            )
    return rows


def oracle_trades_for_gate(
    decisions: Sequence[DecisionCandidates],
    *,
    gate: GateSpec,
    positive_label_threshold: float,
    cooldown_minutes: int,
    max_trades_per_session: int,
) -> list[Trade]:
    trades: list[Trade] = []
    next_time_by_session: dict[str, Any] = {}
    trades_by_session: dict[str, int] = {}
    for decision in decisions:
        next_time = next_time_by_session.get(decision.session)
        if next_time is not None and decision.decision_time < next_time:
            continue
        if max_trades_per_session > 0 and trades_by_session.get(decision.session, 0) >= max_trades_per_session:
            continue
        labels = np.asarray(decision.labels, dtype=np.float32)
        allowed = gate_mask(decision, gate) & np.isfinite(labels)
        if not allowed.any():
            continue
        allowed_idx = np.flatnonzero(allowed)
        best_local = int(allowed_idx[int(np.argmax(labels[allowed_idx]))])
        best_label = float(labels[best_local])
        if best_label <= positive_label_threshold:
            continue
        trades.append(
            Trade(
                session=decision.session,
                decision_time=decision.decision_time.isoformat(),
                pnl=best_label,
                score=None,
                right=str(decision.rights[best_local]),
                offset=float(decision.offsets[best_local]),
                strategy=f"oracle_gate_{gate.name}",
            )
        )
        trades_by_session[decision.session] = trades_by_session.get(decision.session, 0) + 1
        next_time_by_session[decision.session] = decision.decision_time + timedelta(minutes=cooldown_minutes)
    return trades


def build_oracle_rows(
    decisions_by_split: dict[str, list[DecisionCandidates]],
    *,
    gates: Sequence[GateSpec],
    positive_label_threshold: float,
    cooldown_minutes: int,
    max_trades_per_session: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, decisions in decisions_by_split.items():
        for spec in gates:
            trades = oracle_trades_for_gate(
                decisions,
                gate=spec,
                positive_label_threshold=positive_label_threshold,
                cooldown_minutes=cooldown_minutes,
                max_trades_per_session=max_trades_per_session,
            )
            rows.append(
                {
                    "split": split,
                    "gate": spec.name,
                    "oracle_kind": "top_label_positive_only_serial",
                    "cooldown_minutes": cooldown_minutes,
                    "max_trades_per_session": max_trades_per_session,
                    **metrics_for_trades(trades),
                }
            )
    return rows


def stability_rows(
    label_rows: Sequence[dict[str, Any]],
    oracle_rows: Sequence[dict[str, Any]],
    *,
    min_candidates: int,
) -> list[dict[str, Any]]:
    labels = {(row["gate"], row["split"]): row for row in label_rows}
    oracles = {(row["gate"], row["split"]): row for row in oracle_rows}
    gates = sorted({str(row["gate"]) for row in label_rows})
    rows: list[dict[str, Any]] = []
    for gate in gates:
        val = labels.get((gate, "validation"))
        diag = labels.get((gate, "diagnostic_test"))
        val_oracle = oracles.get((gate, "validation"), {})
        diag_oracle = oracles.get((gate, "diagnostic_test"), {})
        if not val or not diag:
            continue
        val_count = int(val.get("candidate_count") or 0)
        diag_count = int(diag.get("candidate_count") or 0)
        val_avg = float(val.get("avg_label_pnl") or 0.0)
        diag_avg = float(diag.get("avg_label_pnl") or 0.0)
        val_oracle_pnl = float(val_oracle.get("total_pnl") or 0.0)
        diag_oracle_pnl = float(diag_oracle.get("total_pnl") or 0.0)
        if val_count < min_candidates or diag_count < min_candidates:
            status = "too_sparse"
        elif val_avg > 0 and diag_avg > 0 and val_oracle_pnl > 0 and diag_oracle_pnl > 0:
            status = "stable_positive_compound_gate"
        elif val_avg > 0 and diag_avg < 0:
            status = "validation_positive_diagnostic_negative"
        elif val_avg < 0 and diag_avg > 0:
            status = "validation_negative_diagnostic_positive"
        else:
            status = "weak_or_mixed"
        rows.append(
            {
                "gate": gate,
                "status": status,
                "validation_candidate_count": val_count,
                "diagnostic_candidate_count": diag_count,
                "validation_avg_label_pnl": val_avg,
                "diagnostic_avg_label_pnl": diag_avg,
                "validation_profit_factor": val.get("profit_factor"),
                "diagnostic_profit_factor": diag.get("profit_factor"),
                "validation_oracle_trades": val_oracle.get("trades", 0),
                "diagnostic_oracle_trades": diag_oracle.get("trades", 0),
                "validation_oracle_pnl": val_oracle_pnl,
                "diagnostic_oracle_pnl": diag_oracle_pnl,
                "validation_oracle_profit_factor": val_oracle.get("profit_factor", 0.0),
                "diagnostic_oracle_profit_factor": diag_oracle.get("profit_factor", 0.0),
            }
        )
    rows.sort(
        key=lambda row: (
            row["status"] == "stable_positive_compound_gate",
            min(float(row["validation_avg_label_pnl"]), float(row["diagnostic_avg_label_pnl"])),
            min(float(row["validation_oracle_pnl"]), float(row["diagnostic_oracle_pnl"])),
        ),
        reverse=True,
    )
    return rows


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Protocol101 Fair-Contract Two-Stage Selector Diagnostic",
        "",
        "## Decision",
        "",
        f"- Status: `{payload['status']}`",
        f"- Design: `{payload['design']}`",
        f"- Policy index: `{payload['policy_index']}`",
        f"- Model training executed: `false`",
        f"- Broker endpoint called: `false`",
        f"- Paper-submit allowed: `false`",
        "",
        "## Top Compound Gates",
        "",
        "| Gate | Status | Val Avg | Diag Avg | Val Candidates | Diag Candidates | Val Oracle PnL | Diag Oracle PnL |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("stability_rows", [])[:20]:
        lines.append(
            f"| `{row['gate']}` | `{row['status']}` | "
            f"{float(row['validation_avg_label_pnl']):.2f} | "
            f"{float(row['diagnostic_avg_label_pnl']):.2f} | "
            f"{row['validation_candidate_count']} | {row['diagnostic_candidate_count']} | "
            f"{float(row['validation_oracle_pnl']):.2f} | "
            f"{float(row['diagnostic_oracle_pnl']):.2f} |"
        )
    lines.extend(["", "## Interpretation", ""])
    for item in payload.get("interpretation", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Outputs", ""])
    for key, value in payload.get("outputs", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def selected_gates(gates_arg: str) -> list[GateSpec]:
    specs = gate_specs_by_name()
    if not gates_arg.strip():
        return list(specs.values())
    names = [item.strip() for item in gates_arg.split(",") if item.strip()]
    missing = [name for name in names if name not in specs]
    if missing:
        raise ValueError(f"unknown gate names: {','.join(missing)}")
    return [specs[name] for name in names]


def main() -> int:
    args = parse_args()
    design = load_json(args.design)
    manifest_path = Path(str((design.get("allowed_data") or {}).get("canonical_manifest") or ""))
    manifest = load_json(manifest_path)
    paths, blockers = paths_by_split(design, manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gates = selected_gates(str(args.gates or ""))
    decisions_by_split = {
        split: load_decisions(split_paths, policy_index=int(args.policy_index))
        for split, split_paths in paths.items()
    }
    label_rows = aggregate_gate_labels(
        decisions_by_split,
        gates=gates,
        positive_label_threshold=float(args.positive_label_threshold),
    )
    oracle_rows = build_oracle_rows(
        decisions_by_split,
        gates=gates,
        positive_label_threshold=float(args.positive_label_threshold),
        cooldown_minutes=int(args.oracle_cooldown_minutes),
        max_trades_per_session=int(args.oracle_max_trades_per_session),
    )
    stable_rows = stability_rows(
        label_rows,
        oracle_rows,
        min_candidates=int(args.min_candidates),
    )
    stable_positive = [
        row for row in stable_rows if row.get("status") == "stable_positive_compound_gate"
    ]
    outputs = {
        "gate_label_summary_csv": str(args.out_dir / "gate_label_summary.csv"),
        "oracle_upper_bound_csv": str(args.out_dir / "oracle_upper_bound.csv"),
        "stability_rows_csv": str(args.out_dir / "stability_rows.csv"),
        "summary_json": str(args.out_dir / "summary.json"),
        "report_md": str(args.out_dir / "report.md"),
    }
    write_csv(Path(outputs["gate_label_summary_csv"]), label_rows)
    write_csv(Path(outputs["oracle_upper_bound_csv"]), oracle_rows)
    write_csv(Path(outputs["stability_rows_csv"]), stable_rows)
    interpretation = [
        "This uses future labels only to audit first-stage selector repairability; it is not a deployable policy.",
        "A stable compound gate can justify a preregistered learned-model attempt, but the learned scorer must still pass strict replay and feature-jitter gates.",
        "Sparse high-PnL gates are treated as hypothesis generators, not paper-readiness evidence.",
    ]
    if stable_positive:
        interpretation.append(
            f"Top stable compound gate is {stable_positive[0]['gate']} with validation/diagnostic average labels "
            f"{float(stable_positive[0]['validation_avg_label_pnl']):.2f}/"
            f"{float(stable_positive[0]['diagnostic_avg_label_pnl']):.2f}."
        )
    payload = {
        "schema_version": "Protocol101FairContractTwoStageSelectorDiagnosticV1",
        "status": "pass" if not blockers else "partial",
        "design": str(args.design),
        "manifest": str(manifest_path),
        "policy_index": int(args.policy_index),
        "positive_label_threshold": float(args.positive_label_threshold),
        "min_candidates": int(args.min_candidates),
        "gates": [spec.__dict__ for spec in gates],
        "blockers": blockers,
        "model_training_executed": False,
        "threshold_tuning_executed": False,
        "broker_endpoint_called": False,
        "paper_submit_allowed": False,
        "paid_data_downloaded": False,
        "split_decision_counts": {split: len(decisions) for split, decisions in decisions_by_split.items()},
        "stable_positive_gate_count": len(stable_positive),
        "stability_rows": stable_rows,
        "interpretation": interpretation,
        "outputs": outputs,
    }
    Path(outputs["summary_json"]).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n")
    Path(outputs["report_md"]).write_text(render_report(payload))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "stable_positive_gate_count": payload["stable_positive_gate_count"],
                "top_gate": stable_positive[0]["gate"] if stable_positive else None,
                "report": outputs["report_md"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
