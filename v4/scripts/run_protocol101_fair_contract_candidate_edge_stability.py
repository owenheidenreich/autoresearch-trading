"""Audit fair-contract candidate-label stability before training more models.

This offline diagnostic answers a narrower question than model search:
do protocol101-live-v1 rows contain stable causal candidate-level edge across
validation and diagnostic splits? It uses the approved design manifest and
does not train, tune thresholds, contact brokers/vendors, download data, change
defaults, or promote a model.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np

from v4.model.supervised_pilot import (
    DecisionCandidates,
    Trade,
    entry_filter_mask,
    load_decisions,
    metrics_for_trades,
)
from v4.scripts.run_protocol101_fair_contract_failure_diagnostic import (
    offset_bucket,
    premium_bucket,
    time_bucket,
)
from v4.scripts.run_protocol101_fair_contract_training_runner import (
    load_json,
    paths_by_split,
)


DEFAULT_DESIGN = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_expanded_jul_sep2025_64_q1_design/summary.json"
)
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_candidate_edge_stability"
)
DEFAULT_ENTRY_FILTERS = (
    "none,"
    "near_10_20_offset,"
    "put_near_10_20_offset,"
    "put_near_after_0940,"
    "put_near_after_0940_vwap_m2_10,"
    "vwap_aligned,"
    "above_vwap_omar_pos_after_open"
)
DIMENSIONS = (
    "overall",
    "right",
    "time_bucket",
    "offset_bucket",
    "premium_bucket",
    "right_time_bucket",
    "right_offset_bucket",
    "vwap_side",
    "omar_side",
    "momentum15_side",
    "vwap_gap_bucket",
    "range_bucket",
)


@dataclass
class LabelStats:
    count: int = 0
    total: float = 0.0
    wins: float = 0.0
    losses: float = 0.0
    positive_count: int = 0
    best: float = float("-inf")
    worst: float = float("inf")

    def add(self, value: float, *, positive_threshold: float) -> None:
        if not math.isfinite(value):
            return
        self.count += 1
        self.total += float(value)
        if value > 0:
            self.wins += float(value)
        elif value < 0:
            self.losses += abs(float(value))
        if value > positive_threshold:
            self.positive_count += 1
        self.best = max(self.best, float(value))
        self.worst = min(self.worst, float(value))

    def to_row(self) -> dict[str, Any]:
        if self.count == 0:
            return {
                "candidate_count": 0,
                "avg_label_pnl": 0.0,
                "positive_rate": 0.0,
                "profit_factor": 0.0,
                "best_label_pnl": 0.0,
                "worst_label_pnl": 0.0,
                "total_label_pnl": 0.0,
            }
        return {
            "candidate_count": self.count,
            "avg_label_pnl": self.total / self.count,
            "positive_rate": self.positive_count / self.count,
            "profit_factor": self.wins / self.losses if self.losses > 0 else (float("inf") if self.wins > 0 else 0.0),
            "best_label_pnl": self.best,
            "worst_label_pnl": self.worst,
            "total_label_pnl": self.total,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--policy-index", type=int, default=0)
    parser.add_argument("--entry-filters", default=DEFAULT_ENTRY_FILTERS)
    parser.add_argument("--positive-label-threshold", type=float, default=20.0)
    parser.add_argument("--min-candidates", type=int, default=100)
    parser.add_argument("--oracle-max-trades-per-session", type=int, default=3)
    parser.add_argument("--oracle-cooldown-minutes", type=int, default=10)
    return parser.parse_args()


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def market_context(decision: DecisionCandidates) -> dict[str, str]:
    last = np.asarray(decision.market_last, dtype=np.float32)
    if last.shape[0] < 7:
        return {
            "vwap_side": "unknown",
            "omar_side": "unknown",
            "momentum15_side": "unknown",
            "vwap_gap_bucket": "unknown",
            "range_bucket": "unknown",
        }
    spx_close = safe_float(last[0])
    spx_vwap = safe_float(last[2])
    omar = safe_float(last[3])
    session_range = safe_float(last[4])
    momentum15 = safe_float(last[6])
    if spx_close is None or spx_vwap is None:
        vwap_side = "unknown"
        vwap_gap_bucket = "unknown"
    else:
        vwap_gap = spx_close - spx_vwap
        vwap_side = "above_vwap" if vwap_gap > 0 else "below_vwap" if vwap_gap < 0 else "at_vwap"
        if vwap_gap <= -10:
            vwap_gap_bucket = "below_vwap_gte_10"
        elif vwap_gap < -2:
            vwap_gap_bucket = "below_vwap_2_10"
        elif vwap_gap <= 2:
            vwap_gap_bucket = "near_vwap"
        elif vwap_gap < 10:
            vwap_gap_bucket = "above_vwap_2_10"
        else:
            vwap_gap_bucket = "above_vwap_gte_10"
    omar_side = "unknown" if omar is None else "omar_pos" if omar > 0 else "omar_neg" if omar < 0 else "omar_flat"
    momentum15_side = (
        "unknown"
        if momentum15 is None
        else "mom15_pos"
        if momentum15 > 0
        else "mom15_neg"
        if momentum15 < 0
        else "mom15_flat"
    )
    if session_range is None:
        range_bucket = "unknown"
    elif session_range < 20:
        range_bucket = "range_lt_20"
    elif session_range < 45:
        range_bucket = "range_20_45"
    elif session_range < 80:
        range_bucket = "range_45_80"
    else:
        range_bucket = "range_gte_80"
    return {
        "vwap_side": vwap_side,
        "omar_side": omar_side,
        "momentum15_side": momentum15_side,
        "vwap_gap_bucket": vwap_gap_bucket,
        "range_bucket": range_bucket,
    }


def candidate_buckets(decision: DecisionCandidates, idx: int) -> dict[str, str]:
    right = str(decision.rights[idx])
    time = time_bucket(decision.decision_time)
    offset = offset_bucket(decision.offsets[idx])
    premium = "unknown"
    if decision.entry_asks is not None and idx < len(decision.entry_asks):
        premium = premium_bucket(decision.entry_asks[idx])
    context = market_context(decision)
    return {
        "overall": "all",
        "right": right,
        "time_bucket": time,
        "offset_bucket": offset,
        "premium_bucket": premium,
        "right_time_bucket": f"{right}_{time}",
        "right_offset_bucket": f"{right}_{offset}",
        **context,
    }


def affordable_mask(decision: DecisionCandidates, *, starting_cash: float = 10_000.0) -> np.ndarray:
    if decision.entry_asks is None:
        return np.ones(len(decision.labels), dtype=bool)
    asks = np.asarray(decision.entry_asks, dtype=np.float32)
    return np.asarray(
        np.isfinite(asks) & (asks > 0.0) & ((asks * 100.0) <= float(starting_cash) + 1e-9),
        dtype=bool,
    )


def aggregate_candidate_labels(
    decisions_by_split: dict[str, list[DecisionCandidates]],
    *,
    entry_filters: list[str],
    positive_label_threshold: float,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str, str], LabelStats]]:
    stats: dict[tuple[str, str, str, str], LabelStats] = {}
    split_rows: list[dict[str, Any]] = []
    for split, decisions in decisions_by_split.items():
        for entry_filter in entry_filters:
            decisions_with_candidates = 0
            candidate_count = 0
            for decision in decisions:
                allowed = entry_filter_mask(decision, entry_filter)
                allowed = allowed & affordable_mask(decision)
                labels = np.asarray(decision.labels, dtype=np.float32)
                finite_allowed = allowed & np.isfinite(labels)
                if not finite_allowed.any():
                    continue
                decisions_with_candidates += 1
                for idx in np.flatnonzero(finite_allowed):
                    candidate_count += 1
                    label = float(labels[int(idx)])
                    buckets = candidate_buckets(decision, int(idx))
                    for dimension in DIMENSIONS:
                        key = (entry_filter, split, dimension, str(buckets[dimension]))
                        stats.setdefault(key, LabelStats()).add(
                            label,
                            positive_threshold=positive_label_threshold,
                        )
            split_rows.append(
                {
                    "entry_filter": entry_filter,
                    "split": split,
                    "decision_count": len(decisions),
                    "decisions_with_candidates": decisions_with_candidates,
                    "candidate_count": candidate_count,
                }
            )
    rows: list[dict[str, Any]] = []
    for (entry_filter, split, dimension, bucket), metric in sorted(stats.items()):
        rows.append(
            {
                "entry_filter": entry_filter,
                "split": split,
                "dimension": dimension,
                "bucket": bucket,
                **metric.to_row(),
            }
        )
    return split_rows + rows, stats


def oracle_trades_for_filter(
    decisions: list[DecisionCandidates],
    *,
    entry_filter: str,
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
        allowed = entry_filter_mask(decision, entry_filter) & affordable_mask(decision) & np.isfinite(labels)
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
                strategy=f"oracle_label_top_{entry_filter}",
            )
        )
        trades_by_session[decision.session] = trades_by_session.get(decision.session, 0) + 1
        next_time_by_session[decision.session] = decision.decision_time + timedelta(
            minutes=int(cooldown_minutes)
        )
    return trades


def build_oracle_rows(
    decisions_by_split: dict[str, list[DecisionCandidates]],
    *,
    entry_filters: list[str],
    positive_label_threshold: float,
    cooldown_minutes: int,
    max_trades_per_session: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, decisions in decisions_by_split.items():
        for entry_filter in entry_filters:
            trades = oracle_trades_for_filter(
                decisions,
                entry_filter=entry_filter,
                positive_label_threshold=positive_label_threshold,
                cooldown_minutes=cooldown_minutes,
                max_trades_per_session=max_trades_per_session,
            )
            rows.append(
                {
                    "split": split,
                    "entry_filter": entry_filter,
                    "oracle_kind": "top_label_positive_only_serial",
                    "cooldown_minutes": cooldown_minutes,
                    "max_trades_per_session": max_trades_per_session,
                    **metrics_for_trades(trades),
                }
            )
    return rows


def stability_pairs(
    stats: dict[tuple[str, str, str, str], LabelStats],
    *,
    min_candidates: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    validation_keys = {
        (entry_filter, dimension, bucket)
        for (entry_filter, split, dimension, bucket), metric in stats.items()
        if split == "validation" and metric.count >= min_candidates
    }
    diagnostic_keys = {
        (entry_filter, dimension, bucket)
        for (entry_filter, split, dimension, bucket), metric in stats.items()
        if split == "diagnostic_test" and metric.count >= min_candidates
    }
    for entry_filter, dimension, bucket in sorted(validation_keys & diagnostic_keys):
        val = stats[(entry_filter, "validation", dimension, bucket)]
        diag = stats[(entry_filter, "diagnostic_test", dimension, bucket)]
        val_row = val.to_row()
        diag_row = diag.to_row()
        val_avg = float(val_row["avg_label_pnl"])
        diag_avg = float(diag_row["avg_label_pnl"])
        if val_avg > 0 and diag_avg > 0:
            status = "stable_positive_label_bucket"
        elif val_avg < 0 and diag_avg < 0:
            status = "stable_negative_label_bucket"
        elif val_avg > 0 and diag_avg < 0:
            status = "validation_positive_diagnostic_negative"
        elif val_avg < 0 and diag_avg > 0:
            status = "validation_negative_diagnostic_positive"
        else:
            status = "flat_or_mixed"
        rows.append(
            {
                "entry_filter": entry_filter,
                "dimension": dimension,
                "bucket": bucket,
                "status": status,
                "validation_candidate_count": val.count,
                "diagnostic_candidate_count": diag.count,
                "validation_avg_label_pnl": val_avg,
                "diagnostic_avg_label_pnl": diag_avg,
                "validation_positive_rate": val_row["positive_rate"],
                "diagnostic_positive_rate": diag_row["positive_rate"],
                "validation_profit_factor": val_row["profit_factor"],
                "diagnostic_profit_factor": diag_row["profit_factor"],
            }
        )
    return rows


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Protocol101 Fair-Contract Candidate Edge Stability",
        "",
        "## Decision",
        "",
        f"- Status: `{payload['status']}`",
        f"- Design: `{payload['design']}`",
        f"- Policy index: `{payload['policy_index']}`",
        f"- Broker endpoint called: `false`",
        f"- Paper-submit allowed: `false`",
        f"- Model training executed: `false`",
        "",
        "## Oracle Upper Bound",
        "",
        "| Filter | Split | Trades | PnL | PF | Max DD |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in payload.get("oracle_rows", []):
        lines.append(
            f"| `{row['entry_filter']}` | `{row['split']}` | {row['trades']} | "
            f"{float(row['total_pnl']):.2f} | {float(row['profit_factor']):.3f} | "
            f"{float(row['max_drawdown']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Stable Positive Label Buckets",
            "",
            "| Filter | Dimension | Bucket | Validation Avg | Diagnostic Avg | Validation Count | Diagnostic Count |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload.get("stable_positive_buckets", [])[:20]:
        lines.append(
            f"| `{row['entry_filter']}` | `{row['dimension']}` | `{row['bucket']}` | "
            f"{float(row['validation_avg_label_pnl']):.2f} | "
            f"{float(row['diagnostic_avg_label_pnl']):.2f} | "
            f"{row['validation_candidate_count']} | {row['diagnostic_candidate_count']} |"
        )
    lines.extend(["", "## Interpretation", ""])
    for item in payload.get("interpretation", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Outputs", ""])
    for key, value in payload.get("outputs", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    design = load_json(args.design)
    manifest_path = Path(str((design.get("allowed_data") or {}).get("canonical_manifest") or ""))
    manifest = load_json(manifest_path)
    paths, blockers = paths_by_split(design, manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    entry_filters = [item.strip() for item in str(args.entry_filters).split(",") if item.strip()]
    decisions_by_split = {
        split: load_decisions(split_paths, policy_index=int(args.policy_index))
        for split, split_paths in paths.items()
    }
    candidate_rows, stats = aggregate_candidate_labels(
        decisions_by_split,
        entry_filters=entry_filters,
        positive_label_threshold=float(args.positive_label_threshold),
    )
    pairs = stability_pairs(stats, min_candidates=int(args.min_candidates))
    oracle_rows = build_oracle_rows(
        decisions_by_split,
        entry_filters=entry_filters,
        positive_label_threshold=float(args.positive_label_threshold),
        cooldown_minutes=int(args.oracle_cooldown_minutes),
        max_trades_per_session=int(args.oracle_max_trades_per_session),
    )
    stable_positive = [
        row for row in pairs if row.get("status") == "stable_positive_label_bucket"
    ]
    stable_positive.sort(
        key=lambda row: (
            min(float(row["validation_avg_label_pnl"]), float(row["diagnostic_avg_label_pnl"])),
            min(int(row["validation_candidate_count"]), int(row["diagnostic_candidate_count"])),
        ),
        reverse=True,
    )
    outputs = {
        "candidate_bucket_summary_csv": str(args.out_dir / "candidate_bucket_summary.csv"),
        "stability_pairs_csv": str(args.out_dir / "stability_pairs.csv"),
        "stable_positive_buckets_csv": str(args.out_dir / "stable_positive_buckets.csv"),
        "oracle_upper_bound_csv": str(args.out_dir / "oracle_upper_bound.csv"),
        "summary_json": str(args.out_dir / "summary.json"),
        "report_md": str(args.out_dir / "report.md"),
    }
    write_csv(Path(outputs["candidate_bucket_summary_csv"]), candidate_rows)
    write_csv(Path(outputs["stability_pairs_csv"]), pairs)
    write_csv(Path(outputs["stable_positive_buckets_csv"]), stable_positive)
    write_csv(Path(outputs["oracle_upper_bound_csv"]), oracle_rows)
    interpretation = [
        "This diagnostic uses future labels only to audit whether causal feature buckets contain stable opportunity; it is not a deployable policy.",
        "If oracle upper bounds are strong but learned models fail, the next branch should focus on model objective/ranking/calibration rather than raw feature-contract repair.",
        "If stable positive buckets are sparse or contradictory, more narrow hand filters are likely overfit.",
    ]
    payload = {
        "schema_version": "Protocol101FairContractCandidateEdgeStabilityV1",
        "status": "pass" if not blockers else "partial",
        "design": str(args.design),
        "manifest": str(manifest_path),
        "policy_index": int(args.policy_index),
        "entry_filters": entry_filters,
        "positive_label_threshold": float(args.positive_label_threshold),
        "min_candidates": int(args.min_candidates),
        "blockers": blockers,
        "model_training_executed": False,
        "threshold_tuning_executed": False,
        "broker_endpoint_called": False,
        "paper_submit_allowed": False,
        "paid_data_downloaded": False,
        "split_decision_counts": {
            split: len(decisions) for split, decisions in decisions_by_split.items()
        },
        "oracle_rows": oracle_rows,
        "stable_positive_bucket_count": len(stable_positive),
        "stable_positive_buckets": stable_positive[:50],
        "interpretation": interpretation,
        "outputs": outputs,
    }
    Path(outputs["summary_json"]).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n")
    Path(outputs["report_md"]).write_text(render_report(payload))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "stable_positive_bucket_count": payload["stable_positive_bucket_count"],
                "report": outputs["report_md"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
