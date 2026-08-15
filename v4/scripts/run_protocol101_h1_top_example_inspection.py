"""Inspect top Q1 lost legacy trades at the actual decision-trace level.

The H1 route summary says pattern/token semantics are the first repair
hypothesis. This script makes that concrete by comparing the same contract in
legacy-feature and live-v1 traces for the highest-PnL lost trades.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from v4.model.hypothesis_protocol import token_feature_names
from v4.scripts.run_protocol101_h1_canonical_semantics_audit import classify_h1_feature_route


DEFAULT_TRADE_ATTRIBUTION = Path(
    "v4/audit/autoresearch/protocol101_q1_2026_trade_pnl_attribution"
)
DEFAULT_Q1_ROOT = Path("v4/audit/autoresearch/protocol101_q1_2026_contract_comparison")
DEFAULT_OUT = Path("v4/audit/autoresearch/protocol101_h1_top_example_inspection")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trade-attribution-dir", type=Path, default=DEFAULT_TRADE_ATTRIBUTION)
    parser.add_argument("--q1-root", type=Path, default=DEFAULT_Q1_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--token-mode", default="aplus")
    return parser.parse_args()


def finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if pd.notna(number) else default


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def trace_key(session: Any, decision_time: Any) -> tuple[str, str]:
    return str(session), pd.Timestamp(decision_time).tz_convert("UTC").isoformat()


def load_target_traces(path: Path, keys: set[tuple[str, str]]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            key = trace_key(row.get("session"), row.get("decision_ts"))
            if key in keys:
                out[key] = row
    return out


def find_candidate(trace: dict[str, Any] | None, contract_id: str) -> dict[str, Any] | None:
    if not trace:
        return None
    for candidate in trace.get("candidate_universe") or []:
        if str(candidate.get("contract_id")) == str(contract_id):
            return candidate
    return None


def abs_delta(a: Any, b: Any) -> float:
    return abs(finite(a) - finite(b))


def example_signature(
    *,
    legacy_trace: dict[str, Any] | None,
    live_trace: dict[str, Any] | None,
    legacy_candidate: dict[str, Any] | None,
    live_candidate: dict[str, Any] | None,
    pattern_diff_count: int,
    pressure_zeroed: bool,
) -> str:
    if legacy_trace is None or live_trace is None:
        return "missing_trace"
    if legacy_candidate is None:
        return "legacy_candidate_missing"
    if live_candidate is None:
        return "live_contract_not_in_universe"
    quote_delta = max(
        abs_delta(legacy_candidate.get("bid"), live_candidate.get("bid")),
        abs_delta(legacy_candidate.get("ask"), live_candidate.get("ask")),
        abs_delta(legacy_candidate.get("spread"), live_candidate.get("spread")),
    )
    if quote_delta <= 1e-4 and pressure_zeroed and pattern_diff_count > 0:
        return "same_quote_pressure_zeroed_and_pattern_drift"
    if quote_delta <= 1e-4 and pattern_diff_count > 0:
        return "same_quote_pattern_drift"
    if quote_delta <= 1e-4 and pressure_zeroed:
        return "same_quote_pressure_zeroed"
    if quote_delta > 1e-4:
        return "quote_or_greek_path_drift"
    return "other_semantic_drift"


def token_diffs(
    *,
    names: list[str],
    legacy_candidate: dict[str, Any] | None,
    live_candidate: dict[str, Any] | None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    if legacy_candidate is None or live_candidate is None:
        return []
    legacy_values = legacy_candidate.get("token_features") or []
    live_values = live_candidate.get("token_features") or []
    rows = []
    for idx, name in enumerate(names[: min(len(names), len(legacy_values), len(live_values))]):
        raw_delta = finite(legacy_values[idx]) - finite(live_values[idx])
        if abs(raw_delta) <= 1e-9:
            continue
        route = classify_h1_feature_route(name)
        rows.append(
            {
                "feature": name,
                "feature_index": idx,
                "legacy_value": finite(legacy_values[idx]),
                "live_value": finite(live_values[idx]),
                "raw_delta_legacy_minus_live": raw_delta,
                "abs_delta": abs(raw_delta),
                **route,
            }
        )
    return sorted(rows, key=lambda item: float(item["abs_delta"]), reverse=True)[:limit]


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    names = token_feature_names(args.token_mode)
    lost = pd.read_csv(args.trade_attribution_dir / "lost_legacy_trades_attribution.csv")
    targets = lost.sort_values("pnl", ascending=False).head(int(args.top_n)).copy()
    keys = {trace_key(row.session, row.decision_time) for row in targets.itertuples(index=False)}
    legacy_traces = load_target_traces(args.q1_root / "legacy_protocol161" / "decision_traces.jsonl", keys)
    live_traces = load_target_traces(args.q1_root / "live_protocol161" / "decision_traces.jsonl", keys)

    example_rows: list[dict[str, Any]] = []
    diff_rows: list[dict[str, Any]] = []
    for raw in targets.itertuples(index=False):
        key = trace_key(raw.session, raw.decision_time)
        legacy_trace = legacy_traces.get(key)
        live_trace = live_traces.get(key)
        legacy_candidate = find_candidate(legacy_trace, raw.contract_id)
        live_candidate = find_candidate(live_trace, raw.contract_id)
        diffs = token_diffs(
            names=names,
            legacy_candidate=legacy_candidate,
            live_candidate=live_candidate,
            limit=12,
        )
        pattern_diff_count = sum(
            1 for item in diffs if str(item["repair_route"]) == "canonical_pattern_semantics"
        )
        pressure_zeroed = False
        if legacy_candidate and live_candidate:
            for name in ("option_ohlcv_volume", "stat_open_interest", "log_option_volume", "log_open_interest"):
                if name in names:
                    idx = names.index(name)
                    legacy_values = legacy_candidate.get("token_features") or []
                    live_values = live_candidate.get("token_features") or []
                    if idx < len(legacy_values) and idx < len(live_values):
                        if finite(legacy_values[idx]) > 0 and abs(finite(live_values[idx])) <= 1e-9:
                            pressure_zeroed = True
                            break
        signature = example_signature(
            legacy_trace=legacy_trace,
            live_trace=live_trace,
            legacy_candidate=legacy_candidate,
            live_candidate=live_candidate,
            pattern_diff_count=pattern_diff_count,
            pressure_zeroed=pressure_zeroed,
        )
        for item in diffs:
            diff_rows.append(
                {
                    "session": raw.session,
                    "decision_time": raw.decision_time,
                    "contract_id": raw.contract_id,
                    "pnl": finite(raw.pnl),
                    "signature": signature,
                    **item,
                }
            )
        quote_delta = None
        if legacy_candidate and live_candidate:
            quote_delta = max(
                abs_delta(legacy_candidate.get("bid"), live_candidate.get("bid")),
                abs_delta(legacy_candidate.get("ask"), live_candidate.get("ask")),
                abs_delta(legacy_candidate.get("spread"), live_candidate.get("spread")),
            )
        example_rows.append(
            {
                "session": raw.session,
                "decision_time": raw.decision_time,
                "contract_id": raw.contract_id,
                "right": raw.right,
                "pnl": finite(raw.pnl),
                "classification": raw.classification,
                "dominant_gap_closure_group": raw.dominant_gap_closure_group,
                "legacy_action": (legacy_trace or {}).get("selected_action"),
                "live_action": (live_trace or {}).get("selected_action"),
                "live_block_reasons": json.dumps((live_trace or {}).get("block_reasons") or []),
                "legacy_trace_present": legacy_trace is not None,
                "live_trace_present": live_trace is not None,
                "legacy_candidate_present": legacy_candidate is not None,
                "live_candidate_present": live_candidate is not None,
                "legacy_candidate_count": (legacy_trace or {}).get("candidate_count"),
                "live_candidate_count": (live_trace or {}).get("candidate_count"),
                "legacy_edge": (legacy_candidate or {}).get("edge"),
                "live_edge": (live_candidate or {}).get("edge"),
                "edge_delta_legacy_minus_live": finite((legacy_candidate or {}).get("edge"))
                - finite((live_candidate or {}).get("edge")),
                "legacy_flat_score": (legacy_candidate or {}).get("surface_flat_score"),
                "live_flat_score": (live_candidate or {}).get("surface_flat_score"),
                "legacy_action_score": (legacy_candidate or {}).get("surface_action_score"),
                "live_action_score": (live_candidate or {}).get("surface_action_score"),
                "max_quote_delta": quote_delta,
                "legacy_source_context_ts": (legacy_trace or {}).get("source_context_ts")
                or ((legacy_trace or {}).get("metadata") or {}).get("source_context_time"),
                "live_source_context_ts": (live_trace or {}).get("source_context_ts")
                or ((live_trace or {}).get("metadata") or {}).get("source_context_time"),
                "pressure_zeroed": pressure_zeroed,
                "pattern_diff_count_in_top_diffs": pattern_diff_count,
                "top_diff_features": ",".join(str(item["feature"]) for item in diffs[:8]),
                "signature": signature,
            }
        )

    write_csv(args.out_dir / "top_example_inspection.csv", example_rows)
    write_csv(args.out_dir / "token_diff_long.csv", diff_rows)
    by_signature = pd.DataFrame(example_rows).groupby("signature").agg(
        examples=("signature", "size"),
        pnl=("pnl", "sum"),
        mean_edge_delta=("edge_delta_legacy_minus_live", "mean"),
    ).reset_index().to_dict("records")
    by_route = pd.DataFrame(diff_rows).groupby("repair_route").agg(
        diff_rows=("repair_route", "size"),
        examples=("contract_id", "nunique"),
        mean_abs_delta=("abs_delta", "mean"),
        max_abs_delta=("abs_delta", "max"),
    ).reset_index().to_dict("records") if diff_rows else []
    summary = {
        "schema_version": "Protocol101H1TopExampleInspectionV1",
        "top_n": int(args.top_n),
        "token_mode": args.token_mode,
        "signatures": by_signature,
        "diff_routes": by_route,
        "no_training": True,
        "no_threshold_tuning": True,
        "broker_endpoint_called": False,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# Protocol101 H1 Top Example Inspection",
        "",
        "## Scope",
        "",
        "- Offline trace inspection only.",
        "- Compares the same contract in Q1 legacy-feature replay versus live-v1 replay for top lost legacy trades.",
        "",
        "## Signatures",
        "",
    ]
    for row in by_signature:
        lines.append(
            f"- `{row['signature']}`: examples=`{row['examples']}`, "
            f"pnl-exposure=`${float(row['pnl']):,.0f}`, "
            f"mean-edge-delta=`{float(row['mean_edge_delta']):.2f}`."
        )
    lines.extend(["", "## Diff Routes", ""])
    for row in by_route:
        lines.append(
            f"- `{row['repair_route']}`: diff-rows=`{row['diff_rows']}`, "
            f"examples=`{row['examples']}`, "
            f"mean-abs-delta=`{float(row['mean_abs_delta']):.2f}`, "
            f"max-abs-delta=`{float(row['max_abs_delta']):.2f}`."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If the signature is `same_quote_pressure_zeroed_and_pattern_drift`, the contract and quote were present; the old and live-v1 games diverged in token semantics.",
            "- This does not justify copying historical volume/OI into live. It tells us which token families must either be made causally reproducible or abandoned/retrained.",
            "- The artifacts here are inspection aids for H1; they do not change Protocol101 behavior.",
        ]
    )
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
