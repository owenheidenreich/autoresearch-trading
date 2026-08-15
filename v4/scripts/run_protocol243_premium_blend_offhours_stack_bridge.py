"""RUNTIME_PREMIUM_BLEND_OFFHOURS_STACK_BRIDGE_V1.

Historically Protocol243. This off-hours bridge audit compares how
PAPER_DEFAULT_PROTOCOL101 and CHALLENGER_PREMIUM_LEANING_BLENDED_UTILITY_V1 are
constructed, then tests whether the challenger can build its trained
full-action feature surface from live-style quote objects without broker calls.

It does not change the paper default, download paid data, place orders, or call
IBKR. It is a workaround for market-closed hours: prove the adapter and feature
contract now, then leave only true live quote breadth/freshness for the next
market session.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v4.live.full_action_challenger_adapter import (
    ALL_FEATURE_COLUMNS,
    FullActionHistoryState,
    build_full_action_candidates_from_quotes,
    feature_source_coverage,
)
from v4.scripts.run_protocol217_full_action_history_runtime_parity import DEFAULT_DATASET
from v4.scripts.run_protocol241_premium_blend_runtime_parity import DEFAULT_ARTIFACT_MANIFEST


ROLE_LABEL = "RUNTIME_PREMIUM_BLEND_OFFHOURS_STACK_BRIDGE_V1"
HISTORICAL_ID = "Protocol243"
CHALLENGER_LABEL = "CHALLENGER_PREMIUM_LEANING_BLENDED_UTILITY_V1"
PAPER_DEFAULT_LABEL = "PAPER_DEFAULT_PROTOCOL101"
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_243_premium_blend_offhours_stack_bridge")
DEFAULT_RECORDED_LIVE_GLOBS = [
    "v4/audit/autoresearch/v4_aplus_hypothesis_147_protocol101_morning_session/2026-05-19/protocol101_no-order-shadow_2026-05-19/cycle_*/live_capture/live_router_shadow_observations.jsonl",
    "v4/audit/autoresearch/v4_aplus_hypothesis_147_protocol101_morning_session/2026-05-20/protocol101_no-order-shadow_2026-05-20/cycle_*/live_capture/live_router_shadow_observations.jsonl",
]
STACK_ROWS = [
    {
        "layer": "operational role",
        "protocol101": "current paper default",
        "challenger": "frozen research challenger",
        "gap": "status, not mechanics",
    },
    {
        "layer": "broker/live adapter",
        "protocol101": "implemented in run_protocol160 persistent IBKR trader",
        "challenger": "off-hours full-action adapter added here; not yet wired into broker loop",
        "gap": "needs live-paper runner switch/adapter path before replacement",
    },
    {
        "layer": "candidate surface",
        "protocol101": "Protocol051 surface candidates after min_edge/time/gate logic",
        "challenger": "full SPXW PM 0DTE ATM +/- $50 surface, calls and puts",
        "gap": "challenger must see broader surface live, not only Protocol101 narrowed samples",
    },
    {
        "layer": "entry model",
        "protocol101": "Protocol101 event-history policy",
        "challenger": "two-stage full-action neural scorer with 45/55 dollar/premium utility",
        "gap": "different tensor shape and threshold",
    },
    {
        "layer": "exit/lifecycle",
        "protocol101": "frozen Protocol081/066 lifecycle stack",
        "challenger": "same frozen exit assumptions in historical replay",
        "gap": "paper replacement can reuse lifecycle/order plumbing",
    },
    {
        "layer": "paper order/account guard",
        "protocol101": "implemented",
        "challenger": "should reuse the Protocol101 paper executor and risk guard",
        "gap": "wire selected challenger contract into same order intent path",
    },
    {
        "layer": "already proven off-hours",
        "protocol101": "live/paper plumbing previously exercised",
        "challenger": "historical no-order runtime parity passed in Protocol241",
        "gap": "recorded-live/full-surface coverage is the remaining off-hours blocker",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--artifact-manifest", type=Path, default=DEFAULT_ARTIFACT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sessions", nargs="*", default=["2026-05-19", "2026-05-20"])
    parser.add_argument("--max-events-per-session", type=int, default=60)
    parser.add_argument("--recorded-live-glob", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.artifact_manifest.read_text())
    feature_columns = list(manifest["feature_columns"])
    coverage = feature_source_coverage(feature_columns)
    historical_surface = historical_surface_summary(args.dataset, sessions=list(args.sessions))
    recorded_globs = args.recorded_live_glob or DEFAULT_RECORDED_LIVE_GLOBS
    recorded_live = recorded_live_summary(recorded_globs)
    adapter_parity = run_historical_adapter_parity(
        args.dataset,
        feature_columns=feature_columns,
        sessions=list(args.sessions),
        max_events_per_session=int(args.max_events_per_session),
    )
    stack = pd.DataFrame(STACK_ROWS)
    stack_path = args.out_dir / "stack_comparison.csv"
    stack.to_csv(stack_path, index=False)
    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "runtime / off-hours stack bridge and promotion-workaround audit",
        "changes_paper_default": False,
        "candidate_label": CHALLENGER_LABEL,
        "paper_default_label": PAPER_DEFAULT_LABEL,
        "other_baseline_label": "Protocol101 live/paper stack construction",
        "data_used": {
            "historical_full_action_dataset": str(args.dataset),
            "challenger_artifact_manifest": str(args.artifact_manifest),
            "recorded_live_globs": recorded_globs,
        },
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "feature_source_coverage": coverage,
        "historical_full_action_surface": historical_surface,
        "recorded_protocol101_live_surface": recorded_live,
        "historical_adapter_parity": adapter_parity,
        "stack_comparison_csv": str(stack_path),
        "decision": decide(coverage, historical_surface, recorded_live, adapter_parity),
        "next_experiment": (
            "Wire the challenger adapter into a no-order/paper-dry-run variant of the persistent trader, "
            "then require an automatic premarket live-surface breadth/freshness gate before any paper-default switch."
        ),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def historical_surface_summary(path: Path, *, sessions: list[str]) -> dict[str, Any]:
    columns = ["session", "decision_dt", "contract_id", "right", "offset", "entry_premium", "root", "settlement_style"]
    frame = pd.read_parquet(path, columns=columns)
    frame = frame[frame["session"].astype(str).isin(sessions)].copy()
    if frame.empty:
        return {"status": "fail", "reason": "no_historical_rows_for_sessions", "sessions": sessions}
    grouped = frame.groupby(["session", "decision_dt"], sort=True)
    counts = grouped.agg(
        candidate_count=("contract_id", "nunique"),
        call_count=("right", lambda x: int((x.astype(str) == "C").sum())),
        put_count=("right", lambda x: int((x.astype(str) == "P").sum())),
        max_abs_offset=("offset", lambda x: float(pd.to_numeric(x, errors="coerce").abs().max())),
    ).reset_index()
    return {
        "status": "pass",
        "sessions": sorted(frame["session"].astype(str).unique().tolist()),
        "event_count": int(len(counts)),
        "candidate_count_min": int(counts["candidate_count"].min()),
        "candidate_count_median": float(counts["candidate_count"].median()),
        "candidate_count_max": int(counts["candidate_count"].max()),
        "median_call_count": float(counts["call_count"].median()),
        "median_put_count": float(counts["put_count"].median()),
        "median_max_abs_offset": float(counts["max_abs_offset"].median()),
        "spxw_pm_only": bool(frame["root"].astype(str).eq("SPXW").all() and frame["settlement_style"].astype(str).eq("PM").all()),
    }


def recorded_live_summary(patterns: list[str]) -> dict[str, Any]:
    paths = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    event_rows = []
    feature_keys: set[str] = set()
    for path in paths:
        rows = []
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                rows.append(item)
                feature_keys.update((item.get("features") or {}).keys())
        if not rows:
            continue
        decision_time = str(rows[0].get("decision_time") or "")
        contracts = {str(row.get("contract_id")) for row in rows if row.get("contract_id")}
        event_rows.append(
            {
                "path": path,
                "session": str(rows[0].get("session") or ""),
                "decision_time": decision_time,
                "contract_count": len(contracts),
                "spx": _finite_float((rows[0].get("context") or {}).get("spx"), math.nan),
                "vix": _finite_float((rows[0].get("context") or {}).get("vix"), math.nan),
            }
        )
    if not event_rows:
        return {"status": "fail", "reason": "no_recorded_live_rows", "paths": 0}
    frame = pd.DataFrame(event_rows)
    return {
        "status": "pass",
        "paths": int(len(paths)),
        "event_count": int(len(frame)),
        "sessions": sorted(frame["session"].astype(str).unique().tolist()),
        "contract_count_min": int(frame["contract_count"].min()),
        "contract_count_median": float(frame["contract_count"].median()),
        "contract_count_max": int(frame["contract_count"].max()),
        "feature_keys_seen": sorted(feature_keys),
        "surface_breadth_note": "These recorded Protocol101/081 shadow logs contain sampled lifecycle contracts, not the full challenger ATM +/- $50 action surface.",
        "usable_for_full_challenger_promotion": bool(frame["contract_count"].median() >= 40),
    }


def run_historical_adapter_parity(
    path: Path,
    *,
    feature_columns: list[str],
    sessions: list[str],
    max_events_per_session: int,
) -> dict[str, Any]:
    columns = list(
        dict.fromkeys(
            [
                "session",
                "decision_dt",
                "contract_id",
                "root",
                "settlement_style",
                "right",
                "offset",
                "entry_bid",
                "entry_ask",
                "entry_mid",
                "entry_spread",
                "entry_bid_size",
                "entry_ask_size",
                "entry_underlying_price",
                "entry_iv",
                "entry_delta",
                "entry_gamma",
                "entry_theta",
                "surface_edge",
                "edge",
                *feature_columns,
            ]
        )
    )
    frame = pd.read_parquet(path, columns=columns)
    frame = frame[frame["session"].astype(str).isin(sessions)].copy()
    frame["decision_dt"] = pd.to_datetime(frame["decision_dt"], utc=True, errors="coerce")
    frame = frame.sort_values(["session", "decision_dt", "contract_id"]).reset_index(drop=True)
    comparable = [column for column in feature_columns if column in ALL_FEATURE_COLUMNS]
    diffs: list[float] = []
    event_count = 0
    row_count = 0
    failed_events: list[dict[str, Any]] = []
    session_event_counts: dict[str, int] = {}
    for session, group in frame.groupby("session", sort=True):
        history = FullActionHistoryState()
        for _, event in group.groupby("decision_dt", sort=True):
            seen_for_session = session_event_counts.get(str(session), 0)
            if seen_for_session >= max_events_per_session:
                break
            session_event_counts[str(session)] = seen_for_session + 1
            quotes = [_quote_from_historical_row(row) for _, row in event.iterrows()]
            first = event.iloc[0]
            market = {column: _finite_float(first.get(column), 0.0) for column in feature_columns if column.startswith("market_")}
            adapter = build_full_action_candidates_from_quotes(
                decision_time=first["decision_dt"],
                spx=_finite_float(first["entry_underlying_price"], 0.0),
                vix=_finite_float(first.get("market_vix_close"), 0.0),
                option_quotes=quotes,
                history=history,
                market_features=market,
                session=str(session),
            )
            merged = event[["contract_id", *comparable]].merge(
                adapter[["contract_id", *comparable]],
                on="contract_id",
                suffixes=("_historical", "_adapter"),
                how="inner",
            )
            if len(merged) != len(event):
                failed_events.append(
                    {
                        "session": str(session),
                        "decision_dt": pd.Timestamp(first["decision_dt"]).isoformat(),
                        "reason": "contract_mismatch",
                        "historical_rows": int(len(event)),
                        "adapter_rows": int(len(adapter)),
                        "matched_rows": int(len(merged)),
                    }
                )
            else:
                for column in comparable:
                    a = pd.to_numeric(merged[f"{column}_historical"], errors="coerce").to_numpy(dtype=float)
                    b = pd.to_numeric(merged[f"{column}_adapter"], errors="coerce").to_numpy(dtype=float)
                    delta = np.abs(a - b)
                    finite_delta = delta[np.isfinite(delta)]
                    diffs.append(float(finite_delta.max()) if len(finite_delta) else 0.0)
            history.update(adapter, decision_time=first["decision_dt"])
            event_count += 1
            row_count += int(len(adapter))
    max_diff = max(diffs) if diffs else math.inf
    return {
        "status": "pass" if not failed_events and max_diff <= 1e-6 else "fail",
        "events_checked": int(event_count),
        "candidate_rows_checked": int(row_count),
        "feature_columns_checked": int(len(comparable)),
        "max_abs_feature_diff": float(max_diff),
        "failed_events": failed_events[:20],
        "note": "Historical full-action rows were converted to live-style quote objects and rebuilt through the adapter without path/exit labels.",
    }


def _quote_from_historical_row(row: pd.Series) -> dict[str, Any]:
    return {
        "contract_id": str(row["contract_id"]),
        "root": str(row.get("root") or "SPXW"),
        "settlement_style": str(row.get("settlement_style") or "PM"),
        "strike": _strike_from_contract_id(str(row["contract_id"]), fallback=_finite_float(row["entry_underlying_price"], 0.0) + _finite_float(row["offset"], 0.0)),
        "right": str(row["right"]),
        "bid": _finite_float(row["entry_bid"], 0.0),
        "ask": _finite_float(row["entry_ask"], 0.0),
        "mid": _finite_float(row["entry_mid"], 0.0),
        "spread": _finite_float(row["entry_spread"], 0.0),
        "bid_size": _finite_float(row["entry_bid_size"], 0.0),
        "ask_size": _finite_float(row["entry_ask_size"], 0.0),
        "underlying_price": _finite_float(row["entry_underlying_price"], 0.0),
        "iv": _finite_float(row["entry_iv"], 0.0),
        "delta": _finite_float(row["entry_delta"], 0.0),
        "gamma": _finite_float(row["entry_gamma"], 0.0),
        "theta": _finite_float(row["entry_theta"], 0.0),
        "surface_edge": _finite_float(row.get("surface_edge", row.get("edge", 0.0)), 0.0),
        "edge": _finite_float(row.get("edge", row.get("surface_edge", 0.0)), 0.0),
    }


def decide(
    coverage: dict[str, Any],
    historical_surface: dict[str, Any],
    recorded_live: dict[str, Any],
    adapter_parity: dict[str, Any],
) -> str:
    if coverage.get("status") != "pass":
        return "blocked_challenger_feature_source_gap"
    if adapter_parity.get("status") != "pass":
        return "blocked_offhours_adapter_does_not_match_training_features"
    if historical_surface.get("status") != "pass":
        return "blocked_missing_historical_surface_for_bridge"
    if not bool(recorded_live.get("usable_for_full_challenger_promotion")):
        return "conditional_bridge_ready_but_recorded_live_logs_too_narrow_for_promotion"
    return "conditional_bridge_ready_recorded_live_surface_sufficient"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {payload['role_label']}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        f"Candidate being tested: {payload['candidate_label']}",
        f"Baseline: {payload['paper_default_label']}",
        f"Data used: {payload['data_used']['historical_full_action_dataset']}",
        f"Paid data downloaded: {payload['paid_data_downloaded_by_runner']}",
        f"Broker endpoint called: {payload['broker_endpoint_called']}",
        f"Next experiment: {payload['next_experiment']}",
        "",
        f"Historical ID: `{payload['historical_protocol']}`",
        f"Decision: `{payload['decision']}`",
        "",
        "## Construction Comparison",
        "",
        "| layer | Protocol101 | Challenger | Gap |",
        "|---|---|---|---|",
    ]
    for row in STACK_ROWS:
        lines.append(f"| {row['layer']} | {row['protocol101']} | {row['challenger']} | {row['gap']} |")
    historical = payload["historical_full_action_surface"]
    recorded = payload["recorded_protocol101_live_surface"]
    parity = payload["historical_adapter_parity"]
    coverage = payload["feature_source_coverage"]
    lines.extend(
        [
            "",
            "## Off-Hours Bridge Result",
            "",
            f"- Challenger feature source coverage: `{coverage['status']}` ({coverage['mapped_feature_count']}/{coverage['feature_count']} mapped)",
            f"- Historical adapter parity: `{parity['status']}` across `{parity['events_checked']}` events and `{parity['candidate_rows_checked']}` candidate rows",
            f"- Max adapter-vs-training feature diff: `{parity['max_abs_feature_diff']}`",
            f"- Historical full-action median candidates/event: `{historical.get('candidate_count_median')}`",
            f"- Recorded Protocol101 live median contracts/event: `{recorded.get('contract_count_median')}`",
            f"- Recorded logs usable for full challenger promotion: `{recorded.get('usable_for_full_challenger_promotion')}`",
            "",
            "## Interpretation",
            "",
            "The challenger can now build the same trained feature surface from live-style quote objects off-hours. "
            "That is the main workaround we can prove while the market is closed. The remaining blocker is not model construction; "
            "it is live quote breadth/freshness for the broader full-action ladder. The existing recorded Protocol101 logs are too narrow "
            "to prove that because they contain sampled lifecycle contracts, not the whole ATM +/- $50 challenger surface.",
            "",
            "## Outputs",
            "",
            f"- Summary: `{path.parent / 'summary.json'}`",
            f"- Stack comparison: `{path.parent / 'stack_comparison.csv'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _strike_from_contract_id(contract_id: str, *, fallback: float) -> float:
    parts = contract_id.split("-")
    if len(parts) >= 3:
        return _finite_float(parts[2], fallback)
    return fallback


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


if __name__ == "__main__":
    raise SystemExit(main())
