"""Attribute attempt-level IBKR-vs-historical fair-contract replay drift.

This audit reads paired ``Protocol101DecisionTraceV1`` rows produced by the
fair-contract IBKR capture and historical dataset replayers. It focuses on the
candidate-specific failure cases that block paper readiness: action drift caused
by the score ceiling, and selected-contract ranking drift.

The script is offline-only. It does not contact brokers or vendors, train,
tune thresholds, change defaults, submit orders, or promote a model.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from v4.dataset.spxw_0dte_neural import MARKET_FEATURE_NAMES, OPTION_FEATURE_NAMES
from v4.live.ibkr_market_capture import clean_json
from v4.scripts.run_protocol101_fair_contract_ibkr_capture_replay import DEFAULT_TRAINING_RESULT
from v4.scripts.run_protocol101_fair_contract_selected_candidate_export import (
    load_json_optional,
    load_model,
    score_candidate_records,
)


DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/protocol101_fair_contract_attempt107_pair_attribution")
SCHEMA_VERSION = "Protocol101FairContractPairAttributionV1"


@dataclass(frozen=True)
class AttributionCase:
    name: str
    session: str
    decision_ts: str
    live_traces: Path
    historical_traces: Path
    contracts: tuple[str, ...]
    reason: str


DEFAULT_CASES = (
    AttributionCase(
        name="july1_score_ceiling_action_drift",
        session="2026-07-01",
        decision_ts="2026-07-01T14:36:00+00:00",
        live_traces=Path(
            "v4/audit/autoresearch/protocol101_fair_contract_attempt107_ibkr_capture_replay_2026_07_01"
            "/decision_traces.jsonl"
        ),
        historical_traces=Path(
            "v4/audit/autoresearch/protocol101_fair_contract_attempt107_historical_replay_2026_07_01"
            "/decision_traces.jsonl"
        ),
        contracts=("SPXW-20260701-07510.000-P",),
        reason="Historical score crosses max_score_ceiling=50 while IBKR score remains below the ceiling.",
    ),
    AttributionCase(
        name="july2_selected_contract_ranking_drift",
        session="2026-07-02",
        decision_ts="2026-07-02T13:57:00+00:00",
        live_traces=Path(
            "v4/audit/autoresearch/protocol101_fair_contract_attempt107_ibkr_capture_replay_2026_07_02"
            "/decision_traces.jsonl"
        ),
        historical_traces=Path(
            "v4/audit/autoresearch/protocol101_fair_contract_attempt107_historical_replay_2026_07_02"
            "/decision_traces.jsonl"
        ),
        contracts=("SPXW-20260702-07545.000-P", "SPXW-20260702-07520.000-P"),
        reason="Both feeds enter at the same minute, but cross-vendor feature drift flips the selected put.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-result", type=Path, default=DEFAULT_TRAINING_RESULT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-n", type=int, default=15)
    return parser.parse_args()


def candidate_feature_names() -> list[str]:
    names: list[str] = []
    names.extend(f"option.{name}" for name in OPTION_FEATURE_NAMES)
    for prefix in ("market_last", "market_mean", "market_std", "market_delta"):
        names.extend(f"{prefix}.{name}" for name in MARKET_FEATURE_NAMES)
    names.extend(("side.is_call", "side.is_put", "shape.offset_norm", "shape.abs_offset_norm"))
    names.extend(
        (
            "env.vwap_gap_over_range",
            "env.range_pct",
            "env.above_vwap",
            "env.below_vwap",
            "env.omar_pos",
            "env.omar_neg",
            "env.mom5_pos",
            "env.mom5_neg",
            "env.mom15_pos",
            "env.mom15_neg",
            "env.vwap_trend_aligned",
            "env.vwap_mean_reversion_side",
            "env.omar_aligned",
            "env.omar_counter",
            "env.momentum15_aligned",
            "env.momentum15_counter",
        )
    )
    names.extend(
        (
            "time.progress",
            "time.remaining",
            "time.sin",
            "time.cos",
            "time.first_30",
            "time.post_open_morning",
            "time.midday",
            "time.late_afternoon",
        )
    )
    return names


def feature_family(name: str) -> str:
    if name.startswith("option."):
        return "option"
    if name.startswith("market_last."):
        return "market_last"
    if name.startswith("market_mean."):
        return "market_mean"
    if name.startswith("market_std."):
        return "market_std"
    if name.startswith("market_delta."):
        return "market_delta"
    if name.startswith("env."):
        return "environment"
    if name.startswith("time."):
        return "time"
    if name.startswith("shape."):
        return "shape"
    if name.startswith("side."):
        return "side"
    return "other"


def load_trace(path: Path, decision_ts: str) -> dict[str, Any]:
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if str(row.get("decision_ts")) == decision_ts:
                return row
    raise ValueError(f"decision_ts {decision_ts} not found in {path}")


def payload(row: dict[str, Any]) -> dict[str, Any]:
    inner = row.get("payload")
    return inner if isinstance(inner, dict) else row


def by_contract(items: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item.get("contract_id") or ""): item for item in items if item.get("contract_id")}


def trace_maps(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    body = payload(row)
    features = body.get("features") if isinstance(body.get("features"), dict) else {}
    scores = body.get("model_scores") if isinstance(body.get("model_scores"), dict) else {}
    return {
        "candidate_universe": by_contract(body.get("candidate_universe") or []),
        "token_features": by_contract(features.get("token_features") or []),
        "candidate_scores": by_contract(scores.get("candidate_scores") or []),
    }


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def vector_for(row: dict[str, Any], contract_id: str) -> np.ndarray:
    maps = trace_maps(row)
    token = maps["token_features"].get(contract_id)
    if not token:
        raise ValueError(f"contract {contract_id} missing token features at {row.get('decision_ts')}")
    return np.asarray(token.get("features") or [], dtype=np.float32)


def score_for(row: dict[str, Any], contract_id: str) -> float | None:
    maps = trace_maps(row)
    score = finite((maps["candidate_scores"].get(contract_id) or {}).get("score"))
    if score is not None:
        return score
    return finite((maps["candidate_universe"].get(contract_id) or {}).get("score"))


def right_for(contract_id: str) -> str:
    return "P" if contract_id.endswith("-P") else "C" if contract_id.endswith("-C") else ""


def model_score(loaded: Any, vector: np.ndarray, contract_id: str) -> float:
    records = [{"_features": np.asarray(vector, dtype=np.float32), "right": right_for(contract_id)}]
    score_candidate_records(records, loaded)
    return float(records[0]["score"])


def group_indices(names: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for idx, name in enumerate(names):
        groups.setdefault(feature_family(name), []).append(idx)
    return groups


def top_feature_diffs(
    *,
    case: AttributionCase,
    contract_id: str,
    live_vector: np.ndarray,
    historical_vector: np.ndarray,
    live_score: float,
    historical_score: float,
    names: list[str],
    loaded: Any,
    top_n: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for idx, name in enumerate(names):
        live_value = finite(live_vector[idx]) if idx < len(live_vector) else None
        historical_value = finite(historical_vector[idx]) if idx < len(historical_vector) else None
        if live_value is None or historical_value is None:
            continue
        swapped = np.array(live_vector, dtype=np.float32, copy=True)
        swapped[idx] = historical_value
        swapped_score = model_score(loaded, swapped, contract_id)
        rows.append(
            {
                "case": case.name,
                "session": case.session,
                "decision_ts": case.decision_ts,
                "contract_id": contract_id,
                "feature": name,
                "family": feature_family(name),
                "live_value": live_value,
                "historical_value": historical_value,
                "delta_historical_minus_live": historical_value - live_value,
                "abs_delta": abs(historical_value - live_value),
                "live_score": live_score,
                "historical_score": historical_score,
                "score_delta_historical_minus_live": historical_score - live_score,
                "single_feature_live_to_historical_score": swapped_score,
                "single_feature_score_delta": swapped_score - live_score,
                "abs_single_feature_score_delta": abs(swapped_score - live_score),
            }
        )
    top_raw = sorted(rows, key=lambda row: row["abs_delta"], reverse=True)[:top_n]
    top_score = sorted(rows, key=lambda row: row["abs_single_feature_score_delta"], reverse=True)[:top_n]
    return top_raw, top_score


def group_swaps(
    *,
    case: AttributionCase,
    contract_id: str,
    live_vector: np.ndarray,
    historical_vector: np.ndarray,
    live_score: float,
    historical_score: float,
    groups: dict[str, list[int]],
    loaded: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, indices in sorted(groups.items()):
        swapped = np.array(live_vector, dtype=np.float32, copy=True)
        for idx in indices:
            if idx < len(swapped) and idx < len(historical_vector):
                swapped[idx] = historical_vector[idx]
        swapped_score = model_score(loaded, swapped, contract_id)
        rows.append(
            {
                "case": case.name,
                "session": case.session,
                "decision_ts": case.decision_ts,
                "contract_id": contract_id,
                "family": family,
                "feature_count": len(indices),
                "live_score": live_score,
                "historical_score": historical_score,
                "score_delta_historical_minus_live": historical_score - live_score,
                "group_live_to_historical_score": swapped_score,
                "group_score_delta": swapped_score - live_score,
                "abs_group_score_delta": abs(swapped_score - live_score),
            }
        )
    return sorted(rows, key=lambda row: row["abs_group_score_delta"], reverse=True)


def score_table(row: dict[str, Any]) -> list[dict[str, Any]]:
    maps = trace_maps(row)
    out: list[dict[str, Any]] = []
    for contract_id, score_row in maps["candidate_scores"].items():
        score = finite(score_row.get("score"))
        candidate = maps["candidate_universe"].get(contract_id) or {}
        out.append(
            {
                "contract_id": contract_id,
                "score": score,
                "right": candidate.get("right"),
                "offset": candidate.get("offset"),
                "entry_bid": candidate.get("entry_bid"),
                "entry_ask": candidate.get("entry_ask"),
                "quote_age_ms": candidate.get("quote_age_ms"),
            }
        )
    return sorted(out, key=lambda item: float(item["score"] if item["score"] is not None else -1e9), reverse=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(clean_json(row))


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Protocol101 Fair-Contract Pair Attribution",
        "",
        "## Decision",
        "",
        f"- Status: `{payload['status']}`",
        f"- Candidate: `{payload['candidate']}`",
        "- Broker endpoint called: `false`",
        "- Paper-submit allowed: `false`",
        "- Model training executed here: `false`",
        "- Threshold tuning executed here: `false`",
        "",
        "## Main Finding",
        "",
        payload["main_finding"],
        "",
        "## Cases",
        "",
    ]
    for case in payload["cases"]:
        lines.extend(
            [
                f"### {case['name']}",
                "",
                f"- Session: `{case['session']}`",
                f"- Decision timestamp: `{case['decision_ts']}`",
                f"- Live action: `{case['live_action']}` selected `{case['live_selected_contract_id']}` score `{case['live_selected_score']}`",
                f"- Historical action: `{case['historical_action']}` selected `{case['historical_selected_contract_id']}` score `{case['historical_selected_score']}`",
                f"- Reason inspected: {case['reason']}",
                "",
                "| Contract | Live Score | Historical Score | Delta | Top Group Swaps |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for contract in case["contracts"]:
            groups = ", ".join(
                f"{row['family']} {row['group_score_delta']:+.2f}"
                for row in contract["top_group_swaps"][:4]
            )
            lines.append(
                f"| `{contract['contract_id']}` | {contract['live_score']:.4f} | "
                f"{contract['historical_score']:.4f} | {contract['score_delta_historical_minus_live']:+.4f} | {groups} |"
            )
        if case.get("ranking"):
            lines.extend(["", "Top live scores:", ""])
            for row in case["ranking"]["live_top"][:5]:
                lines.append(f"- `{row['contract_id']}` score `{row['score']}` offset `{row.get('offset')}`")
            lines.extend(["", "Top historical scores:", ""])
            for row in case["ranking"]["historical_top"][:5]:
                lines.append(f"- `{row['contract_id']}` score `{row['score']}` offset `{row.get('offset')}`")
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "- If a single vendor-side feature group moves a score across the ceiling or flips the top rank, the candidate is data-plane-sensitive at that timestamp.",
            "- This audit does not tune the threshold or edit the candidate. It only identifies whether the current failure is likely repairable feature semantics or candidate brittleness.",
            "",
            "## Outputs",
            "",
        ]
    )
    for key, value in payload.get("outputs", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    training_result = load_json_optional(args.training_result)
    if not training_result:
        raise FileNotFoundError(f"training result missing or invalid: {args.training_result}")
    loaded = load_model(dict(training_result))
    names = candidate_feature_names()
    groups = group_indices(names)
    feature_diff_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    case_payloads: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []

    for case in DEFAULT_CASES:
        live = load_trace(case.live_traces, case.decision_ts)
        historical = load_trace(case.historical_traces, case.decision_ts)
        live_body = payload(live)
        historical_body = payload(historical)
        case_payload: dict[str, Any] = {
            "name": case.name,
            "session": case.session,
            "decision_ts": case.decision_ts,
            "reason": case.reason,
            "live_action": live.get("selected_action"),
            "historical_action": historical.get("selected_action"),
            "live_selected_contract_id": live.get("selected_contract_id"),
            "historical_selected_contract_id": historical.get("selected_contract_id"),
            "live_selected_score": finite(live.get("selected_score")),
            "historical_selected_score": finite(historical.get("selected_score")),
            "live_block_reasons": live.get("block_reasons") or [],
            "historical_block_reasons": historical.get("block_reasons") or [],
            "live_candidate_count": live_body.get("candidate_count"),
            "historical_candidate_count": historical_body.get("candidate_count"),
            "contracts": [],
        }
        live_top = score_table(live)
        historical_top = score_table(historical)
        case_payload["ranking"] = {"live_top": live_top[:10], "historical_top": historical_top[:10]}
        for side, rows in (("live", live_top[:20]), ("historical", historical_top[:20])):
            for rank, row in enumerate(rows, start=1):
                ranking_rows.append({"case": case.name, "side": side, "rank": rank, **row})
        for contract_id in case.contracts:
            live_vector = vector_for(live, contract_id)
            historical_vector = vector_for(historical, contract_id)
            if len(live_vector) != len(names) or len(historical_vector) != len(names):
                raise ValueError(
                    f"{case.name} {contract_id} vector length mismatch: "
                    f"live={len(live_vector)} historical={len(historical_vector)} names={len(names)}"
                )
            live_score = score_for(live, contract_id)
            historical_score = score_for(historical, contract_id)
            if live_score is None or historical_score is None:
                raise ValueError(f"{case.name} {contract_id} missing score")
            top_raw, top_score = top_feature_diffs(
                case=case,
                contract_id=contract_id,
                live_vector=live_vector,
                historical_vector=historical_vector,
                live_score=live_score,
                historical_score=historical_score,
                names=names,
                loaded=loaded,
                top_n=max(int(args.top_n), 1),
            )
            family_rows = group_swaps(
                case=case,
                contract_id=contract_id,
                live_vector=live_vector,
                historical_vector=historical_vector,
                live_score=live_score,
                historical_score=historical_score,
                groups=groups,
                loaded=loaded,
            )
            feature_diff_rows.extend(top_raw)
            feature_diff_rows.extend(top_score)
            group_rows.extend(family_rows)
            case_payload["contracts"].append(
                {
                    "contract_id": contract_id,
                    "live_score": live_score,
                    "historical_score": historical_score,
                    "score_delta_historical_minus_live": historical_score - live_score,
                    "top_raw_feature_diffs": top_raw[:5],
                    "top_single_feature_score_swaps": top_score[:5],
                    "top_group_swaps": family_rows[:5],
                }
            )
        case_payloads.append(case_payload)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "fail_not_paper_ready",
        "candidate": "attempt107",
        "training_result": str(args.training_result),
        "cases": case_payloads,
        "main_finding": (
            "Attempt107 still fails paired cross-vendor paper-readiness: same-input replay is deterministic, "
            "but candidate scores and rankings are sensitive to IBKR-vs-historical feature deltas at threshold-crossing moments."
        ),
        "broker_endpoint_called": False,
        "paper_submit_allowed": False,
        "model_training_executed_here": False,
        "threshold_tuning_executed_here": False,
    }
    feature_csv = args.out_dir / "feature_diffs.csv"
    group_csv = args.out_dir / "group_swaps.csv"
    ranking_csv = args.out_dir / "rankings.csv"
    summary_json = args.out_dir / "summary.json"
    report_md = args.out_dir / "report.md"
    write_csv(feature_csv, feature_diff_rows)
    write_csv(group_csv, group_rows)
    write_csv(ranking_csv, ranking_rows)
    payload_out = {
        **summary,
        "outputs": {
            "feature_diffs_csv": str(feature_csv),
            "group_swaps_csv": str(group_csv),
            "rankings_csv": str(ranking_csv),
            "summary_json": str(summary_json),
            "report_md": str(report_md),
        },
    }
    summary_json.write_text(json.dumps(clean_json(payload_out), indent=2, sort_keys=True) + "\n")
    report_md.write_text(render_report(clean_json(payload_out)))
    print(
        json.dumps(
            {
                "status": payload_out["status"],
                "cases": [case["name"] for case in case_payloads],
                "report": str(report_md),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
