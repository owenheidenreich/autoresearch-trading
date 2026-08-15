"""Protocol 159: live/training feature-parity audit.

This is an observability gate for the paper trader. It does not download data
and does not call a broker endpoint. It checks that the live Protocol101 path is
building the same feature shapes and causal context expected by the frozen
training artifacts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, time as wall_time
import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from v4.dataset.spxw_0dte_neural import MARKET_FEATURE_NAMES, OPTION_FEATURE_NAMES
from v4.live.paper_trade_log import DEFAULT_TRADE_LOG_ROOT, load_trade_log
from v4.live.protocol101_entry import Protocol101HistoryState, protocol101_candidate_frame_from_surface
from v4.live.protocol101_live_entry import LiveIndexState, build_live_surface_row, live_surface_decision
from v4.model.hypothesis_protocol import (
    registered_aplus_surface_variants,
    structural_feature_names,
    token_feature_names,
)
from v4.scripts.run_protocol101_event_history_policy import FEATURE_COLUMNS
from v4.scripts.run_protocol119_protocol101_live_readiness import DEFAULT_PROTOCOL101_MANIFEST
from v4.scripts.run_protocol121_protocol101_entry_router_smoke import DEFAULT_SURFACE_MANIFEST
from v4.scripts.run_protocol158_protocol101_live_entry_paper_bridge import variant_for


NY = ZoneInfo("America/New_York")
DEFAULT_OUT_ROOT = Path("v4/audit/autoresearch/v4_aplus_hypothesis_159_live_training_parity_audit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trade-log", type=Path, default=None)
    parser.add_argument("--trade-log-root", type=Path, default=DEFAULT_TRADE_LOG_ROOT)
    parser.add_argument("--surface-manifest", type=Path, default=DEFAULT_SURFACE_MANIFEST)
    parser.add_argument("--protocol101-manifest", type=Path, default=DEFAULT_PROTOCOL101_MANIFEST)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--min-context-minutes", type=float, default=30.0)
    parser.add_argument("--max-open-start-lag-minutes", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trade_log = args.trade_log or latest_trade_log(args.trade_log_root)
    rows = load_trade_log(trade_log)
    if not rows:
        raise SystemExit(f"no rows found in {trade_log}")
    session = str(rows[-1].get("session") or datetime.now(NY).date().isoformat())
    run_id = str(rows[-1].get("run_id") or trade_log.stem)
    out_dir = args.out_root / session / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = build_audit_payload(
        rows=rows,
        trade_log=trade_log,
        surface_manifest=args.surface_manifest,
        protocol101_manifest=args.protocol101_manifest,
        min_context_minutes=float(args.min_context_minutes),
        max_open_start_lag_minutes=float(args.max_open_start_lag_minutes),
    )
    payload["outputs"] = {
        "summary": str(out_dir / "summary.json"),
        "report": str(out_dir / "report.md"),
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "report": str(out_dir / "report.md"), "summary": str(out_dir / "summary.json")}, indent=2))
    return 0 if not payload["decision"].startswith("fail_") else 1


def build_audit_payload(
    *,
    rows: list[dict[str, Any]],
    trade_log: Path,
    surface_manifest: Path,
    protocol101_manifest: Path,
    min_context_minutes: float,
    max_open_start_lag_minutes: float,
) -> dict[str, Any]:
    surface = load_json(surface_manifest)
    protocol101 = load_json(protocol101_manifest)
    variant_name = str(surface.get("variant_name") or "")
    variant = variant_for(variant_name)
    synthetic = synthetic_live_schema_check(variant)
    latest_market = latest_row(rows, "market_snapshot")
    latest_candidate = latest_row(rows, "candidate_set")
    live_context = object_or_empty(latest_candidate.get("live_index_context"))
    gate = object_or_empty(latest_candidate.get("candidate_gate_diagnostics"))
    market_snapshot = object_or_empty(latest_market.get("market_snapshot"))
    market_context = object_or_empty(market_snapshot.get("context"))
    market_underlying = object_or_empty(market_snapshot.get("underlying"))
    spx_market_data_type = market_underlying.get("spx_market_data_type") or market_context.get("spx_market_data_type")
    vix_market_data_type = market_underlying.get("vix_market_data_type") or market_context.get("vix_market_data_type")

    checks: list[dict[str, Any]] = []
    warnings: list[str] = []
    add_check(checks, "surface_manifest_exists", bool(surface), {"path": str(surface_manifest)})
    add_check(checks, "protocol101_manifest_exists", bool(protocol101), {"path": str(protocol101_manifest)})
    add_check(checks, "surface_variant_registered", variant_name in {item.name for item in registered_aplus_surface_variants()}, {"variant_name": variant_name})
    add_check(checks, "surface_policy_index_matches_training", int(surface.get("policy_index", -1)) == 1, {"policy_index": surface.get("policy_index")})
    add_check(checks, "option_feature_names_match_training", synthetic["option_feature_names_match"], synthetic["option_schema"])
    add_check(checks, "market_feature_names_match_training", synthetic["market_feature_names_match"], synthetic["market_schema"])
    add_check(checks, "surface_scalar_shape_matches_training", synthetic["scalar_dim"] == len(structural_feature_names(variant.market_mode)), {"live_scalar_dim": synthetic["scalar_dim"], "expected_scalar_dim": len(structural_feature_names(variant.market_mode))})
    add_check(checks, "surface_token_shape_matches_training", synthetic["token_dim"] == len(token_feature_names(variant.token_mode)), {"live_token_dim": synthetic["token_dim"], "expected_token_dim": len(token_feature_names(variant.token_mode))})
    add_check(checks, "protocol101_feature_columns_match_code", tuple(protocol101.get("feature_columns") or []) == tuple(FEATURE_COLUMNS), {"manifest_feature_count": len(protocol101.get("feature_columns") or []), "code_feature_count": len(FEATURE_COLUMNS)})
    add_check(checks, "protocol101_candidate_features_complete", synthetic["candidate_features_complete"], {"missing": synthetic["missing_candidate_features"]})
    add_check(checks, "latest_live_market_snapshot_present", bool(latest_market), {"timestamp": latest_market.get("timestamp")})
    add_check(
        checks,
        "latest_market_data_is_live",
        spx_market_data_type == "live" and vix_market_data_type == "live",
        {
            "spx_market_data_type": spx_market_data_type,
            "vix_market_data_type": vix_market_data_type,
            "context": market_context,
        },
    )
    add_check(checks, "candidate_gate_diagnostics_present", bool(gate), {"timestamp": latest_candidate.get("timestamp")})
    add_check(checks, "live_index_context_present", bool(live_context), live_context)

    context_span = number(live_context.get("span_minutes"))
    minute_rows = int(number(live_context.get("minute_row_count")) or 0)
    add_check(
        checks,
        "live_context_has_minimum_span",
        context_span is not None and context_span >= min_context_minutes and minute_rows >= int(min_context_minutes),
        {"span_minutes": context_span, "minute_row_count": minute_rows, "required_minutes": min_context_minutes},
    )
    start_lag = market_open_lag_minutes(live_context.get("first_timestamp"))
    if start_lag is not None and start_lag > max_open_start_lag_minutes:
        warnings.append(
            f"live context started {start_lag:.1f} minutes after 09:30 ET; today's early context is incomplete"
        )
    if synthetic["known_approximations"]:
        warnings.extend(synthetic["known_approximations"])

    failed = [row for row in checks if row["status"] == "fail"]
    if failed:
        decision = "fail_live_training_parity"
    elif warnings:
        decision = "pass_live_training_parity_with_warnings"
    else:
        decision = "pass_live_training_parity"
    return {
        "protocol": "159_live_training_parity_audit",
        "decision": decision,
        "paid_data_downloaded": False,
        "broker_order_endpoint_called": False,
        "real_money_trading": False,
        "source_trade_log": str(trade_log),
        "surface_manifest": str(surface_manifest),
        "protocol101_manifest": str(protocol101_manifest),
        "surface_variant": variant_name,
        "checks": checks,
        "warnings": warnings,
        "latest_live_market": summarize_market(latest_market),
        "latest_candidate_gate": gate,
        "latest_live_index_context": live_context,
        "synthetic_schema_check": synthetic,
    }


def synthetic_live_schema_check(variant: Any) -> dict[str, Any]:
    state = LiveIndexState()
    base_ts = datetime(2026, 1, 2, 15, 0, tzinfo=ZoneInfo("UTC"))
    for idx in range(45):
        state.add(timestamp=base_ts + timedelta(minutes=idx), spx=6000.0 + idx * 0.25, vix=18.0)
    row, _ = build_live_surface_row(
        decision_time=datetime(2026, 1, 2, 15, 44, tzinfo=ZoneInfo("UTC")),
        spx=6011.0,
        vix=18.0,
        option_quotes=synthetic_quotes(),
        index_state=state,
    )
    decision = live_surface_decision(session="2026-01-02", row=row, variant=variant, policy_index=1, index_state=state)
    candidate_frame = protocol101_candidate_frame_from_surface(
        decision,
        np.asarray([0.0] + [30.0] * len(decision.token_mask), dtype=np.float32),
        Protocol101HistoryState(),
        min_edge=25.0,
    )
    missing = [column for column in FEATURE_COLUMNS if column not in candidate_frame.columns]
    return {
        "option_feature_names_match": tuple(row["feature_names"]) == tuple(OPTION_FEATURE_NAMES),
        "market_feature_names_match": tuple(row["market_feature_names"]) == tuple(MARKET_FEATURE_NAMES),
        "option_schema": {"live": list(row["feature_names"]), "training": list(OPTION_FEATURE_NAMES)},
        "market_schema": {"live": list(row["market_feature_names"]), "training": list(MARKET_FEATURE_NAMES)},
        "market_window_shape": list(np.asarray(row["market_window"]).shape),
        "option_ladder_shape": list(np.asarray(row["option_ladder"]).shape),
        "scalar_dim": int(len(decision.scalar_features)),
        "token_dim": int(decision.token_features.shape[1]),
        "candidate_features_complete": not missing and not candidate_frame.empty,
        "missing_candidate_features": missing,
        "known_approximations": [
            "Live SPX structure high/low/ATR are reconstructed from one-minute IBKR index snapshots, not vendor OHLC bars.",
            "Live SPX VWAP falls back to one-minute SPX mean because SPX index has no true trade volume.",
        ],
    }


def synthetic_quotes() -> list[dict[str, Any]]:
    out = []
    for strike in (6000.0, 6005.0, 6010.0):
        for right in ("C", "P"):
            out.append(
                {
                    "contract_id": f"SPXW-20260102-{strike:09.3f}-{right}",
                    "symbol": "SPX",
                    "trading_class": "SPXW",
                    "expiry": "20260102",
                    "strike": strike,
                    "right": right,
                    "bid": 9.8,
                    "ask": 10.0,
                    "bid_size": 20,
                    "ask_size": 25,
                    "iv": 0.20,
                    "delta": 0.50 if right == "C" else -0.50,
                    "gamma": 0.01,
                    "theta": -0.30,
                    "quote_age_ms": 100,
                }
            )
    return out


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol 159 Live/Training Parity Audit",
        "",
        f"Decision: `{payload['decision']}`",
        "",
        "## Checks",
        "",
        "| Check | Status | Detail |",
        "|---|---:|---|",
    ]
    for row in payload["checks"]:
        lines.append(f"| `{row['name']}` | `{row['status']}` | `{json.dumps(row.get('detail') or {}, sort_keys=True, default=str)}` |")
    if payload["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {item}" for item in payload["warnings"])
    lines.extend(
        [
            "",
            "## Latest Context",
            "",
            "```json",
            json.dumps(payload["latest_live_index_context"], indent=2, sort_keys=True, default=str),
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def latest_trade_log(root: Path) -> Path:
    files = sorted(root.rglob("*.jsonl"), key=lambda path: path.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"no paper trade logs found under {root}")
    return files[-1]


def latest_row(rows: list[dict[str, Any]], event_type: str) -> dict[str, Any]:
    for row in reversed(rows):
        if row.get("event_type") == event_type:
            return row
    return {}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: dict[str, Any] | None = None) -> None:
    checks.append({"name": name, "status": "pass" if passed else "fail", "detail": detail or {}})


def object_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def number(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def market_open_lag_minutes(timestamp: Any) -> float | None:
    if not timestamp:
        return None
    ts = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00")).astimezone(NY)
    open_dt = datetime.combine(ts.date(), wall_time(9, 30), tzinfo=NY)
    return float((ts - open_dt).total_seconds() / 60.0)


def summarize_market(row: dict[str, Any]) -> dict[str, Any]:
    snapshot = object_or_empty(row.get("market_snapshot"))
    return {
        "timestamp": row.get("timestamp"),
        "underlying": object_or_empty(snapshot.get("underlying")),
        "option_nbbo": object_or_empty(snapshot.get("option_nbbo")),
        "context": object_or_empty(snapshot.get("context")),
    }


if __name__ == "__main__":
    raise SystemExit(main())
