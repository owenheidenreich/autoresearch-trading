"""Protocol 124: Protocol101 live-data parity checkpoint.

This no-order/no-download checkpoint consolidates the current broker-data
state. Delayed IBKR rows can prove plumbing, but only live SPX/VIX plus live
SPXW OPRA NBBO can clear Protocol101 no-order shadow capture.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_124_protocol101_live_data_parity_checkpoint")
DEFAULT_LEDGER = Path("v4/ledger/RESEARCH_LEDGER.md")
DEFAULT_IBKR_SUMMARY = Path("v4/audit/ibkr_live_data_entitlements/summary.json")
DEFAULT_PROTOCOL119_SUMMARY = Path("v4/audit/autoresearch/v4_aplus_hypothesis_119_protocol101_live_readiness/summary.json")
DEFAULT_PROTOCOL121_SUMMARY = Path("v4/audit/autoresearch/v4_aplus_hypothesis_121_protocol101_entry_router_smoke/summary.json")
DEFAULT_DELAYED_CAPTURE_SUMMARY = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_088_protocol081_live_shadow_router/ibkr-live-capture_summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--ibkr-summary", type=Path, default=DEFAULT_IBKR_SUMMARY)
    parser.add_argument("--protocol119-summary", type=Path, default=DEFAULT_PROTOCOL119_SUMMARY)
    parser.add_argument("--protocol121-summary", type=Path, default=DEFAULT_PROTOCOL121_SUMMARY)
    parser.add_argument("--delayed-capture-summary", type=Path, default=DEFAULT_DELAYED_CAPTURE_SUMMARY)
    parser.add_argument("--no-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ibkr = load_json(args.ibkr_summary)
    p119 = load_json(args.protocol119_summary)
    p121 = load_json(args.protocol121_summary)
    delayed = load_json(args.delayed_capture_summary)
    payload = build_payload(ibkr=ibkr, protocol119=p119, protocol121=p121, delayed_capture=delayed)
    summary_path = args.out_dir / "summary.json"
    report_path = args.out_dir / "report.md"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_report(report_path, payload)
    if not args.no_ledger:
        append_ledger(args.ledger, payload, report_path)
    print(json.dumps({"decision": payload["decision"], "report": str(report_path)}, indent=2, sort_keys=True))
    return 0 if payload["decision"].startswith("ready_") else 1


def build_payload(
    *,
    ibkr: dict[str, Any],
    protocol119: dict[str, Any],
    protocol121: dict[str, Any],
    delayed_capture: dict[str, Any],
) -> dict[str, Any]:
    feed = ibkr.get("feed_status", {})
    spx = feed.get("spx", {})
    vix = feed.get("vix", {})
    options = feed.get("spxw_options", {})
    delayed_probe = delayed_capture.get("feed_probe", {})
    delayed_rows = int(delayed_capture.get("captured_rows", 0) or 0)
    delayed_parity = (delayed_capture.get("shadow_parity") or {}).get("status")
    checks = {
        "protocol101_entry_router_wired": protocol121.get("decision") == "pass_protocol101_entry_router_edge_wired",
        "protocol119_feature_status_pass": (protocol119.get("feature_dependency_audit") or {}).get("status") == "pass",
        "ibkr_connected": bool(ibkr.get("ibkr_connected")),
        "spx_live_price": bool(spx.get("live_price_available")),
        "vix_live_price": bool(vix.get("live_price_available")),
        "spxw_live_nbbo_rows": int(options.get("live_nbbo_rows", 0) or 0),
        "delayed_plumbing_rows": delayed_rows,
        "delayed_plumbing_parity_pass": delayed_parity == "pass",
        "delayed_market_data_observed": "delayed" in str(delayed_probe.get("observed_option_market_data_types", {})).lower()
        or "delayed" in str(delayed_probe.get("spx_market_data_type", "")).lower(),
    }
    decision = decide(checks=checks, protocol119=protocol119, delayed_capture=delayed_capture)
    return {
        "protocol": "124_protocol101_live_data_parity_checkpoint",
        "decision": decision,
        "paid_data_downloaded": False,
        "live_orders": False,
        "broker_order_endpoint_called": False,
        "checks": checks,
        "ibkr_blocked_reason": ibkr.get("blocked_reason"),
        "protocol119_decision": protocol119.get("decision"),
        "delayed_capture_decision": delayed_capture.get("decision"),
        "delayed_capture_blocked_reason": delayed_capture.get("blocked_reason"),
        "required_before_protocol101_live_shadow": required_actions(checks),
    }


def decide(*, checks: dict[str, Any], protocol119: dict[str, Any], delayed_capture: dict[str, Any]) -> str:
    if (
        protocol119.get("decision") == "ready_for_protocol101_no_order_live_capture"
        and checks["spx_live_price"]
        and checks["vix_live_price"]
        and checks["spxw_live_nbbo_rows"] > 0
        and checks["protocol101_entry_router_wired"]
    ):
        return "ready_for_protocol101_no_order_live_capture"
    if checks["ibkr_connected"] and checks["delayed_plumbing_rows"] > 0 and checks["delayed_plumbing_parity_pass"]:
        return "blocked_live_subscriptions_delayed_plumbing_passed"
    if checks["ibkr_connected"]:
        return "blocked_live_subscriptions_or_nbbo"
    return "blocked_ibkr_connection"


def required_actions(checks: dict[str, Any]) -> list[str]:
    actions = []
    if not checks["ibkr_connected"]:
        actions.append("Start IB Gateway/TWS with API enabled on the configured paper-trading port.")
    if not checks["spx_live_price"] or not checks["vix_live_price"]:
        actions.append("Enable live Cboe index market data for SPX and VIX in the IBKR API session.")
    if checks["spxw_live_nbbo_rows"] <= 0:
        actions.append("Enable live OPRA top-of-book data so SPXW option NBBO is available to the API session.")
    if not checks["protocol101_entry_router_wired"]:
        actions.append("Keep Protocol121 passing so the Protocol101 live router uses real surface edge features.")
    if not actions:
        actions.append("Run no-order Protocol101 live shadow capture; order placement must remain disabled.")
    return actions


def write_report(path: Path, payload: dict[str, Any]) -> None:
    checks = payload["checks"]
    lines = [
        "# Protocol 124: Protocol101 Live-Data Parity Checkpoint",
        "",
        "No paid data was downloaded. No broker order endpoint was called. No orders were placed.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Protocol119 decision: `{payload['protocol119_decision']}`",
        f"- IBKR blocked reason: `{payload['ibkr_blocked_reason']}`",
        f"- Delayed capture decision: `{payload['delayed_capture_decision']}`",
        f"- Delayed capture blocker: `{payload['delayed_capture_blocked_reason']}`",
        "",
        "## Checks",
        "",
        "| check | value |",
        "| --- | ---: |",
    ]
    for key, value in checks.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines += [
        "",
        "## Required Before Protocol101 Live Shadow",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["required_before_protocol101_live_shadow"])
    lines += [
        "",
        "Delayed rows prove that the local IBKR pipe, SPXW chain qualification, quote parsing, and no-order safety checks can run. They do not prove promotion-grade live-data parity.",
    ]
    path.write_text("\n".join(lines) + "\n")


def append_ledger(ledger: Path, payload: dict[str, Any], report_path: Path) -> None:
    marker = "## 2026-05-14 Protocol 124 Protocol101 Live-Data Parity Checkpoint"
    entry = f"""

{marker}

```text
Date: 2026-05-14
Decision / Experiment: Consolidated Protocol101 live-data parity state after delayed IBKR plumbing succeeded but live subscriptions remained unavailable.
Reason: The current work queue is live-data parity, no-order Protocol101 shadow capture, and order-state rehearsal around a $10,000 paper account. This checkpoint keeps delayed plumbing evidence separate from promotion-grade live-data evidence.
Data Used: Existing IBKR entitlement summary, Protocol119 readiness summary, Protocol121 entry-router smoke summary, and latest no-order delayed capture summary. No paid data was downloaded, no broker order endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision {payload['decision']}. Report: {report_path}
Next Gate: {'; '.join(payload['required_before_protocol101_live_shadow'])}
Owner: Codex
```
"""
    existing = ledger.read_text() if ledger.exists() else ""
    if marker not in existing:
        ledger.write_text(existing.rstrip() + entry + "\n")
        return
    start = existing.index(marker)
    next_start = existing.find("\n## ", start + len(marker))
    replacement = entry.strip() + "\n"
    if next_start == -1:
        ledger.write_text(existing[:start].rstrip() + "\n\n" + replacement)
    else:
        ledger.write_text(existing[:start].rstrip() + "\n\n" + replacement + existing[next_start:])


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
