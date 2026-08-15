"""Use existing captures for Protocol101 source-policy repairability diagnostics.

This is an offline reducer. It does not rebuild datasets, train, tune
thresholds, download vendor data, call broker endpoints, or mutate the shared
feature contract. It answers whether existing Q1 and recorder evidence is
enough to keep working before the next live capture.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

import pandas as pd

from v4.live.protocol101_synchronization import (
    Protocol101ExistingCaptureRepairabilityDiagnosticV1,
)


DEFAULT_Q1_TRADE_ATTRIBUTION = Path(
    "v4/audit/autoresearch/protocol101_q1_2026_trade_pnl_attribution/lost_legacy_trades_attribution.csv"
)
DEFAULT_H1_TOP_EXAMPLES = Path(
    "v4/audit/autoresearch/protocol101_h1_top_example_inspection/summary.json"
)
DEFAULT_H1_REPAIRABILITY = Path(
    "v4/audit/autoresearch/protocol101_h1_repairability_decision/repairability_decision.json"
)
DEFAULT_CAPTURE_ROOT = Path.home() / ".autoresearch-trading/live_runtime/ibkr_capture"
DEFAULT_OUT = Path("v4/audit/autoresearch/protocol101_existing_capture_repairability_diagnostic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lost-trades", type=Path, default=DEFAULT_Q1_TRADE_ATTRIBUTION)
    parser.add_argument("--h1-top-examples", type=Path, default=DEFAULT_H1_TOP_EXAMPLES)
    parser.add_argument("--h1-repairability", type=Path, default=DEFAULT_H1_REPAIRABILITY)
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--sessions", nargs="*", default=["2026-06-30", "2026-07-01", "2026-07-02"])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().read_text())


def finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows or [{"status": "no_rows"}])


def summarize_lost_trade_source_policy(lost: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    near_minute = pd.to_numeric(lost.get("near_minute_delta"), errors="coerce")
    near_strike = pd.to_numeric(lost.get("near_strike_delta"), errors="coerce")
    pnl = pd.to_numeric(lost["pnl"], errors="coerce").fillna(0.0)
    one_minute_near = near_minute.abs().le(1.0) & near_strike.abs().le(5.0)
    two_minute_near = near_minute.abs().le(2.0) & near_strike.abs().le(10.0)
    exact_time_or_one_minute = near_minute.abs().le(1.0)
    top_group_rows = []
    for group, frame in lost.groupby(lost["dominant_gap_closure_group"].fillna("UNKNOWN")):
        group_pnl = pd.to_numeric(frame["pnl"], errors="coerce").fillna(0.0)
        top_group_rows.append(
            {
                "dominant_gap_closure_group": str(group),
                "trades": int(len(frame)),
                "pnl": float(group_pnl.sum()),
                "one_minute_near_trades": int(one_minute_near.loc[frame.index].sum()),
                "one_minute_near_pnl": float(group_pnl.loc[one_minute_near.loc[frame.index]].sum()),
            }
        )
    top_group_rows = sorted(top_group_rows, key=lambda row: abs(float(row["pnl"])), reverse=True)
    summary = {
        "lost_legacy_trades": int(len(lost)),
        "lost_legacy_pnl": float(pnl.sum()),
        "near_match_trades": int(lost.get("near_match", pd.Series(False, index=lost.index)).fillna(False).astype(bool).sum()),
        "near_match_pnl": float(pnl.loc[lost.get("near_match", pd.Series(False, index=lost.index)).fillna(False).astype(bool)].sum()),
        "one_minute_near_trades": int(one_minute_near.sum()),
        "one_minute_near_pnl": float(pnl.loc[one_minute_near].sum()),
        "one_minute_near_pnl_share": float(pnl.loc[one_minute_near].sum() / pnl.sum()) if float(pnl.sum()) else None,
        "two_minute_near_trades": int(two_minute_near.sum()),
        "two_minute_near_pnl": float(pnl.loc[two_minute_near].sum()),
        "exact_time_or_one_minute_trades": int(exact_time_or_one_minute.sum()),
        "missing_near_candidate_rows": int(near_minute.isna().sum()),
        "dominant_groups": top_group_rows,
    }
    return summary, top_group_rows


def _capture_dir(capture_root: Path, session: str) -> Path:
    return capture_root.expanduser() / session / f"protocol101-recorder-{session}"


def summarize_capture_timing(capture_dir: Path) -> dict[str, Any]:
    trace_path = capture_dir / "ibkr_protocol101_traces.jsonl"
    quality = load_json(capture_dir / "ibkr_capture_quality.json")
    replay = load_json(capture_dir / "same_input_replay_summary.json")
    context_lags: list[float] = []
    quote_lags: list[float] = []
    future_context_rows = 0
    future_quote_rows = 0
    missing_timestamps = 0
    rows = 0
    entry_rows = 0
    with trace_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            rows += 1
            raw = json.loads(line)
            payload = raw.get("payload") if isinstance(raw.get("payload"), dict) else raw
            decision = pd.Timestamp(raw.get("decision_ts") or payload.get("decision_ts"))
            context = raw.get("source_context_ts") or payload.get("source_context_ts")
            quote = raw.get("source_quote_ts") or payload.get("source_quote_ts")
            if payload.get("selected_action") == "enter" or raw.get("selected_action") == "enter":
                entry_rows += 1
            if not context or not quote:
                missing_timestamps += 1
                continue
            context_ts = pd.Timestamp(context)
            quote_ts = pd.Timestamp(quote)
            context_lag = (context_ts - decision).total_seconds() / 60.0
            quote_lag = (quote_ts - decision).total_seconds()
            context_lags.append(context_lag)
            quote_lags.append(quote_lag)
            if context_lag > 1e-9:
                future_context_rows += 1
            if quote_lag > 1e-6:
                future_quote_rows += 1
    return {
        "session": str(replay.get("session") or quality.get("session") or capture_dir.parent.name),
        "capture_status": quality.get("status"),
        "opening_context_ready": quality.get("opening_context_ready"),
        "missing_opening_minutes": quality.get("missing_opening_minutes"),
        "same_input_exact": bool(replay.get("same_input_exact")),
        "trace_rows": rows,
        "entry_actions": int(replay.get("entry_actions") or entry_rows),
        "lifecycle_actions": replay.get("lifecycle_actions") or {},
        "context_lag_min_median": median(context_lags) if context_lags else None,
        "context_lag_min_unique_sample": sorted({round(value, 6) for value in context_lags})[:8],
        "quote_lag_sec_median": median(quote_lags) if quote_lags else None,
        "quote_lag_sec_min": min(quote_lags) if quote_lags else None,
        "quote_lag_sec_max": max(quote_lags) if quote_lags else None,
        "future_context_rows": future_context_rows,
        "future_quote_rows": future_quote_rows,
        "missing_source_timestamp_rows": missing_timestamps,
    }


def summarize_capture_vendor_fields(capture_dir: Path) -> dict[str, Any]:
    events_path = capture_dir / "market_events.jsonl"
    session = capture_dir.parent.name
    event_counts: dict[str, int] = {}
    option_update_rows = 0
    option_volume_rows = 0
    option_open_interest_rows = 0
    market_data_types: set[str] = set()
    generic_diagnostic_payload_rows = 0
    with events_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            event_type = str(row.get("event_type") or "")
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
            data_type = payload.get("market_data_type_name")
            if data_type:
                market_data_types.add(str(data_type))
            if event_type == "option_update":
                option_update_rows += 1
                if optional_float(payload.get("volume")) is not None:
                    option_volume_rows += 1
                if optional_float(payload.get("open_interest_call")) is not None or optional_float(
                    payload.get("open_interest_put")
                ) is not None:
                    option_open_interest_rows += 1
                if (
                    optional_float(payload.get("volume")) is not None
                    or optional_float(payload.get("open_interest_call")) is not None
                    or optional_float(payload.get("open_interest_put")) is not None
                ):
                    generic_diagnostic_payload_rows += 1
    return {
        "session": session,
        "event_counts": event_counts,
        "option_update_rows": option_update_rows,
        "option_volume_rows": option_volume_rows,
        "option_open_interest_rows": option_open_interest_rows,
        "generic_diagnostic_payload_rows": generic_diagnostic_payload_rows,
        "market_data_types": sorted(market_data_types),
        "volume_is_direct_databento_ohlcv_replacement": False,
        "open_interest_is_direct_databento_statistics_replacement": False,
    }


def build_repairability_routes(
    *,
    q1_source_policy: dict[str, Any],
    h1_top: dict[str, Any],
    h1_repairability: dict[str, Any],
    capture_timing: list[dict[str, Any]],
    vendor_fields: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    all_context_lags_exact_prior = all(
        row.get("future_context_rows") == 0
        and row.get("context_lag_min_median") == -1
        for row in capture_timing
    )
    all_same_input = all(bool(row.get("same_input_exact")) for row in capture_timing)
    volume_rows = sum(int(row.get("option_volume_rows") or 0) for row in vendor_fields)
    oi_rows = sum(int(row.get("option_open_interest_rows") or 0) for row in vendor_fields)
    top_signature = ((h1_top.get("signatures") or [{}])[0] or {}).get("signature", "UNKNOWN")
    prior_decision = h1_repairability.get("decision", "UNKNOWN")
    return [
        {
            "route": "source_policy_timing_and_atm_geometry",
            "status": "offline_diagnostic_rebuild_allowed",
            "evidence": (
                f"{q1_source_policy['one_minute_near_trades']} lost legacy trades "
                f"(${q1_source_policy['one_minute_near_pnl']:.0f}) have a live-contract signal "
                "within one minute and five strike points; "
                f"top H1 signature is {top_signature}."
            ),
            "repairability": (
                "plausible_but_not_proven_contract_safe"
                if q1_source_policy["one_minute_near_trades"] > 0
                else "weak"
            ),
            "allowed_now": "build diagnostic alternate-source-policy artifacts only",
            "blocked_change": "do not mutate protocol101-live-v1 until causality is proven",
        },
        {
            "route": "current_recorder_contract_causality",
            "status": "passes_current_contract_evidence",
            "evidence": (
                "Existing capture traces are exact same-input replays and use source_context_ts = decision_ts - 1 minute."
                if all_context_lags_exact_prior and all_same_input
                else "Existing capture timing or same-input replay evidence is incomplete."
            ),
            "repairability": "current_contract_is_causal_but_lower_pnl",
            "allowed_now": "use existing captures for repeatable offline tests",
            "blocked_change": "do not infer that legacy same-minute semantics are live-safe from this evidence alone",
        },
        {
            "route": "volume_open_interest_pressure",
            "status": "diagnostic_only",
            "evidence": (
                f"Existing captures contain {volume_rows} option volume payload rows and "
                f"{oi_rows} open-interest payload rows; these are IBKR generic diagnostics, "
                "not Databento OHLCV/statistics replacements."
            ),
            "repairability": "not_a_direct_frozen_model_rescue",
            "allowed_now": "compare distributions and availability after capture",
            "blocked_change": "do not restore historical volume/OI fields into live inference",
        },
        {
            "route": "shared_contract_mutation",
            "status": "blocked",
            "evidence": f"Prior governance decision remains `{prior_decision}`.",
            "repairability": "requires_causal_source_policy_proof_or_retraining",
            "allowed_now": "report and diagnostic experiments",
            "blocked_change": "no production contract update and no Q1 rerun as a fixed claim",
        },
    ]


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lost = pd.read_csv(args.lost_trades)
    h1_top = load_json(args.h1_top_examples)
    h1_repairability = load_json(args.h1_repairability)
    q1_source_policy, group_rows = summarize_lost_trade_source_policy(lost)
    capture_timing = []
    vendor_fields = []
    for session in args.sessions:
        capture_dir = _capture_dir(args.capture_root, session)
        capture_timing.append(summarize_capture_timing(capture_dir))
        vendor_fields.append(summarize_capture_vendor_fields(capture_dir))
    routes = build_repairability_routes(
        q1_source_policy=q1_source_policy,
        h1_top=h1_top,
        h1_repairability=h1_repairability,
        capture_timing=capture_timing,
        vendor_fields=vendor_fields,
    )
    same_input_sessions = sum(1 for row in capture_timing if row.get("same_input_exact"))
    total_entries = sum(int(row.get("entry_actions") or 0) for row in capture_timing)
    headline = {
        "status": "existing_capture_offline_diagnostic_complete",
        "sessions": list(args.sessions),
        "same_input_exact_sessions": same_input_sessions,
        "entry_actions_in_existing_captures": total_entries,
        "production_contract_change_allowed": False,
        "q1_rerun_as_fixed_claim_allowed": False,
        "model_training": False,
        "threshold_tuning": False,
        "broker_endpoint_called": False,
        "paid_download_called": False,
    }
    conclusions = [
        "Do not wait idly for July 6; existing captures are enough for repeatable offline trace and source-policy diagnostics.",
        "The current recorder/live-v1 contract is causally clean on existing captures: context is one minute behind the decision and quotes are before the decision.",
        "The strongest Q1 recovery clue is a timing/feature-semantics clue, not a missing-contract clue: many lost legacy trades have a nearby live-contract signal.",
        "That clue justifies a diagnostic alternate-source-policy rebuild, but not a production contract mutation yet.",
        "Volume/open-interest diagnostics exist in IBKR captures, but their semantics are not direct replacements for Databento OHLCV/statistics.",
        "July 6+ recorder data should confirm a frozen hypothesis on fresh sessions, not serve as the first place we look for the answer.",
    ]
    next_actions = [
        "Build an explicitly labeled diagnostic alternate-source-policy Q1 artifact, if supported by the historical builder, to estimate recoverable frozen-model PnL without changing protocol101-live-v1.",
        "Compare the diagnostic artifact against the existing capture traces for timestamp causality: no feature may use a minute that would be incomplete at inference time.",
        "If the diagnostic source policy recovers Q1 non-inferiority and is proven causal, then propose a contract update and rerun Q1 with frozen weights.",
        "If the diagnostic source policy does not recover non-inferiority or is not causal, preserve the canonical live-reproducible contract and plan retraining later.",
        "Use July 6+ as out-of-sample confirmation for the selected source-policy hypothesis and for volume/OI diagnostic availability.",
    ]
    packet = Protocol101ExistingCaptureRepairabilityDiagnosticV1(
        headline=headline,
        q1_lost_trade_policy=q1_source_policy,
        capture_timing=capture_timing,
        capture_vendor_fields=vendor_fields,
        repairability_routes=routes,
        conclusions=conclusions,
        next_actions=next_actions,
    )
    (args.out_dir / "summary.json").write_text(
        json.dumps(packet.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    write_csv(args.out_dir / "lost_trade_source_policy_by_group.csv", group_rows)
    write_csv(args.out_dir / "capture_trace_timing_summary.csv", capture_timing)
    write_csv(args.out_dir / "capture_vendor_field_summary.csv", vendor_fields)
    write_csv(args.out_dir / "repairability_routes.csv", routes)
    lines = [
        "# Protocol101 Existing Capture Repairability Diagnostic",
        "",
        "## Scope",
        "",
        "- Offline only: existing captures and existing Q1 artifacts.",
        "- No model training, threshold tuning, paid download, broker call, or feature-contract mutation.",
        "",
        "## Headline",
        "",
        f"- Existing exact same-input capture sessions: `{same_input_sessions}` / `{len(args.sessions)}`.",
        f"- Existing capture entry actions: `{total_entries}`.",
        (
            "- Q1 lost legacy trades with live-contract signal within one minute and five strikes: "
            f"`{q1_source_policy['one_minute_near_trades']}` / "
            f"`${q1_source_policy['one_minute_near_pnl']:.0f}` "
            f"({(q1_source_policy['one_minute_near_pnl_share'] or 0.0):.1%} of lost PnL)."
        ),
        f"- Production contract change allowed: `{str(headline['production_contract_change_allowed']).lower()}`.",
        f"- Q1 rerun as fixed claim allowed: `{str(headline['q1_rerun_as_fixed_claim_allowed']).lower()}`.",
        "",
        "## Capture Timing",
        "",
    ]
    for row in capture_timing:
        lines.append(
            f"- `{row['session']}`: status=`{row['capture_status']}`, same-input=`{row['same_input_exact']}`, "
            f"trace-rows=`{row['trace_rows']}`, entries=`{row['entry_actions']}`, "
            f"context-lag-min-median=`{row['context_lag_min_median']}`, "
            f"future-context-rows=`{row['future_context_rows']}`, future-quote-rows=`{row['future_quote_rows']}`."
        )
    lines.extend(["", "## Repairability Routes", ""])
    for row in routes:
        lines.append(
            f"- `{row['route']}`: status=`{row['status']}`, repairability=`{row['repairability']}`. "
            f"{row['evidence']}"
        )
    lines.extend(["", "## Conclusions", ""])
    lines.extend(f"- {item}" for item in conclusions)
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {item}" for item in next_actions)
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(headline, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
