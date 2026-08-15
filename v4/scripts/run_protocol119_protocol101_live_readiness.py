"""Protocol 119: Protocol 101 no-order live readiness gate.

This is a no-download, no-order readiness check. It does not call IBKR by
itself; it reads the most recent no-order entitlement probe output and combines
that with Protocol 101 artifact/loadability checks and a live-feature
dependency audit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from v4.model.supervised_pilot import FeatureScaler
from v4.scripts.run_protocol097_sequential_event_policy import EventSetPolicy


DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_119_protocol101_live_readiness")
DEFAULT_PROTOCOL101_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy")
DEFAULT_PROTOCOL101_MANIFEST = (
    DEFAULT_PROTOCOL101_DIR
    / "model_artifacts/fold3_train_q1_q2_q3_validate_q4_test_q1_2026/seed_1/manifest.json"
)
DEFAULT_PROTOCOL101_SUMMARY = DEFAULT_PROTOCOL101_DIR / "summary.json"
DEFAULT_PROTOCOL118_SUMMARY = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_118_protocol101_shadow_rehearsal/summary.json"
)
DEFAULT_PROTOCOL120_SUMMARY = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_120_surface_edge_portability/summary.json"
)
DEFAULT_PROTOCOL121_SUMMARY = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_121_protocol101_entry_router_smoke/summary.json"
)
DEFAULT_IBKR_ENTITLEMENT_SUMMARY = Path("v4/audit/ibkr_live_data_entitlements/summary.json")
DEFAULT_LEDGER = Path("v4/ledger/RESEARCH_LEDGER.md")

LIVE_DERIVABLE_FEATURES = {
    "entry_minutes_since_open",
    "entry_minutes_to_forced_flat",
    "entry_progress",
    "entry_progress_sin",
    "entry_progress_cos",
    "entry_is_first_30m",
    "entry_is_post_open_morning",
    "entry_is_midday",
    "entry_is_late_afternoon",
    "right_is_call",
    "right_is_put",
    "offset",
    "abs_offset",
    "entry_bid",
    "entry_ask",
    "entry_mid",
    "entry_spread",
    "entry_spread_frac",
    "entry_bid_size",
    "entry_ask_size",
    "entry_underlying_price",
    "entry_iv",
    "entry_delta",
    "entry_abs_delta",
    "entry_gamma",
    "entry_theta",
    "entry_abs_theta",
    "entry_gamma_theta_ratio",
    "entry_theta_over_mid",
    "entry_theta_burden",
    "entry_gamma_per_premium",
    "entry_premium_over_underlying",
    "entry_spread_over_mid",
    "entry_size_imbalance",
    "entry_call_delta_signed",
    "entry_put_delta_signed",
    "hist_events_seen",
    "hist_minutes_since_prev_event",
    "hist_prev_candidate_count",
    "hist_prev_max_gamma",
    "hist_prev_mean_theta_burden",
    "hist_prev_min_spread_over_mid",
    "hist_prev_call_count",
    "hist_prev_put_count",
    "hist_roll3_candidate_count_mean",
    "hist_roll3_max_gamma",
    "hist_roll3_mean_theta_burden",
    "hist_roll3_min_spread_over_mid",
}
UPSTREAM_EDGE_FEATURES = {
    "edge",
    "hist_prev_max_edge",
    "hist_prev_mean_edge",
    "hist_prev_call_minus_put_edge",
    "hist_roll3_max_edge",
    "hist_roll3_mean_edge",
    "hist_roll3_call_minus_put_edge",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--protocol101-manifest", type=Path, default=DEFAULT_PROTOCOL101_MANIFEST)
    parser.add_argument("--protocol101-summary", type=Path, default=DEFAULT_PROTOCOL101_SUMMARY)
    parser.add_argument("--protocol118-summary", type=Path, default=DEFAULT_PROTOCOL118_SUMMARY)
    parser.add_argument("--protocol120-summary", type=Path, default=DEFAULT_PROTOCOL120_SUMMARY)
    parser.add_argument("--protocol121-summary", type=Path, default=DEFAULT_PROTOCOL121_SUMMARY)
    parser.add_argument("--ibkr-entitlement-summary", type=Path, default=DEFAULT_IBKR_ENTITLEMENT_SUMMARY)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--no-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    artifact = load_protocol101_artifact(args.protocol101_manifest, args.protocol101_summary)
    protocol118 = _load_json(args.protocol118_summary)
    protocol120 = _load_json(args.protocol120_summary)
    protocol121 = _load_json(args.protocol121_summary)
    ibkr = _load_json(args.ibkr_entitlement_summary)
    feature_audit = audit_live_feature_dependencies(artifact["feature_columns"])
    feature_audit = apply_surface_edge_portability(feature_audit, protocol120)
    feature_audit = apply_protocol101_entry_router_smoke(feature_audit, protocol121)
    decision = decide_live_readiness(
        artifact=artifact,
        protocol118=protocol118,
        ibkr=ibkr,
        feature_audit=feature_audit,
    )
    payload = {
        "protocol": "119_protocol101_live_readiness",
        "decision": decision,
        "paid_data_downloaded": False,
        "live_orders": False,
        "broker_order_endpoint_called": False,
        "market_data_endpoint_called_by_this_script": False,
        "protocol101_artifact": artifact,
        "protocol118_decision": protocol118.get("decision"),
        "protocol120_decision": protocol120.get("decision"),
        "protocol121_decision": protocol121.get("decision"),
        "ibkr_entitlement": {
            "summary_path": str(args.ibkr_entitlement_summary),
            "decision": ibkr.get("decision"),
            "blocked_reason": ibkr.get("blocked_reason"),
            "ibkr_connected": ibkr.get("ibkr_connected"),
            "regular_market_hours": ibkr.get("regular_market_hours"),
            "connection_attempts": ibkr.get("connection_attempts", []),
        },
        "feature_dependency_audit": feature_audit,
        "next_requirements": next_requirements(decision, feature_audit, ibkr),
    }
    summary_path = args.out_dir / "summary.json"
    report_path = args.out_dir / "report.md"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_report(report_path, payload)
    if not args.no_ledger:
        append_ledger(args.ledger, payload, report_path)
    print(json.dumps({"decision": decision, "report": str(report_path)}, indent=2, sort_keys=True))
    return 0 if decision == "ready_for_protocol101_no_order_live_capture" else 1


def load_protocol101_artifact(manifest_path: Path, summary_path: Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    model_path = manifest_path.parent / "model.pt"
    scaler_path = manifest_path.parent / "scaler.json"
    missing = [str(path) for path in (manifest_path, model_path, scaler_path, summary_path) if not path.exists()]
    feature_columns = list(manifest.get("feature_columns", []))
    loadable = not missing and bool(feature_columns)
    threshold = None
    if summary_path.exists():
        threshold = _threshold_from_summary(summary_path, str(manifest.get("fold")), int(manifest.get("seed", -1)))
    error = None
    if loadable:
        try:
            hidden_dim = int(manifest.get("config", {}).get("hidden_dim", 96))
            model = EventSetPolicy(input_dim=len(feature_columns), hidden_dim=hidden_dim)
            state = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state)
            scaler_payload = _load_json(scaler_path)
            scaler = FeatureScaler(
                fill=np.asarray(scaler_payload["fill"], dtype=np.float32),
                mean=np.asarray(scaler_payload["mean"], dtype=np.float32),
                std=np.asarray(scaler_payload["std"], dtype=np.float32),
            )
            dummy = np.zeros((1, len(feature_columns)), dtype=np.float32)
            _ = scaler.transform(dummy)
        except Exception as exc:  # pragma: no cover - exact torch failure text is environment-specific.
            loadable = False
            error = str(exc)
    return {
        "manifest_path": str(manifest_path),
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "fold": manifest.get("fold"),
        "seed": manifest.get("seed"),
        "threshold": threshold,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "loadable": bool(loadable),
        "missing_files": missing,
        "load_error": error,
    }


def audit_live_feature_dependencies(feature_columns: list[str]) -> dict[str, Any]:
    columns = set(feature_columns)
    derivable = sorted(columns & LIVE_DERIVABLE_FEATURES)
    upstream_edge = sorted(columns & UPSTREAM_EDGE_FEATURES)
    unknown = sorted(columns - LIVE_DERIVABLE_FEATURES - UPSTREAM_EDGE_FEATURES)
    return {
        "status": "blocked_missing_upstream_edge_generator" if upstream_edge or unknown else "pass",
        "feature_count": len(feature_columns),
        "live_derivable_features": derivable,
        "upstream_edge_features": upstream_edge,
        "unknown_features": unknown,
        "why_it_matters": (
            "Protocol 101 was trained with the upstream A+ surface edge and edge-history features. "
            "Filling these with zero in live trading would change the model's input distribution, so a live "
            "router must port the frozen Protocol 051/A+ surface scorer or retrain without those features."
        ),
    }


def apply_surface_edge_portability(feature_audit: dict[str, Any], protocol120: dict[str, Any]) -> dict[str, Any]:
    """Refine the feature audit when the frozen edge scorer has been proven loadable."""

    out = dict(feature_audit)
    if (
        out.get("status") == "blocked_missing_upstream_edge_generator"
        and protocol120.get("decision") == "pass_surface_edge_generator_loads_offline_live_router_next"
    ):
        out["status"] = "blocked_upstream_edge_generator_not_wired"
        out["surface_edge_generator_loadable"] = True
        out["surface_edge_generator_summary"] = protocol120.get("summary", {})
        out["why_it_matters"] = (
            "Protocol 120 proved the frozen Protocol 051/A+ surface scorer can produce finite edge values offline. "
            "The remaining blocker is wiring that scorer into the Protocol 101 live entry router so `edge` and "
            "edge-history features are produced causally from the live candidate ladder."
        )
    else:
        out["surface_edge_generator_loadable"] = False
    return out


def apply_protocol101_entry_router_smoke(feature_audit: dict[str, Any], protocol121: dict[str, Any]) -> dict[str, Any]:
    """Mark edge features as live-router wired once Protocol 121 passes."""

    out = dict(feature_audit)
    if protocol121.get("decision") == "pass_protocol101_entry_router_edge_wired":
        out["status"] = "pass"
        out["protocol101_entry_router_edge_wired"] = True
        out["protocol101_entry_router_summary"] = protocol121.get("summary", {})
        out["why_it_matters"] = (
            "Protocol 121 proved the frozen Protocol 051/A+ surface edge scorer feeds the frozen Protocol 101 "
            "event-policy feature path without zero-filling edge features. The remaining readiness blocker is live "
            "market-data availability, not local edge-feature wiring."
        )
    else:
        out["protocol101_entry_router_edge_wired"] = False
    return out


def decide_live_readiness(
    *,
    artifact: dict[str, Any],
    protocol118: dict[str, Any],
    ibkr: dict[str, Any],
    feature_audit: dict[str, Any],
) -> str:
    if not artifact.get("loadable"):
        return "blocked_protocol101_artifact_not_loadable"
    if protocol118.get("decision") != "pass_historical_no_order_shadow_rehearsal_live_capture_next":
        return "blocked_protocol118_rehearsal_not_passed"
    if ibkr.get("decision") != "pass":
        reason = str(ibkr.get("blocked_reason") or "ibkr_market_data_not_ready")
        return f"blocked_{reason}"
    if feature_audit.get("status") != "pass":
        return str(feature_audit["status"])
    return "ready_for_protocol101_no_order_live_capture"


def next_requirements(decision: str, feature_audit: dict[str, Any], ibkr: dict[str, Any]) -> list[str]:
    requirements: list[str] = []
    if decision.startswith("blocked_ibkr") or ibkr.get("decision") != "pass":
        errors = ibkr.get("ibkr_errors", [])
        has_competing_session = any(int(event.get("error_code", 0)) == 10197 for event in errors if isinstance(event, dict))
        if ibkr.get("ibkr_connected") and has_competing_session:
            requirements.append(
                "IB Gateway is reachable, but IBKR returned error 10197: no market data during a competing live session. Close or log out other IBKR live-market-data sessions, then rerun the no-order entitlement check."
            )
        elif ibkr.get("ibkr_connected"):
            requirements.append(
                "IB Gateway is reachable, but live SPX/VIX and OPRA/SPXW market data are not available to this API session. Check market-data subscriptions and API market-data acknowledgement."
            )
        else:
            requirements.append(
                "Start IB Gateway/TWS with API enabled and a reachable local port. The attempted ports were 7497, 4002, 7496, and 4001."
            )
        requirements.append(
            "Rerun the no-order entitlement check and require live SPX, VIX, and OPRA/SPXW NBBO market data before Protocol 101 no-order live capture."
        )
    if feature_audit.get("status") == "blocked_upstream_edge_generator_not_wired":
        requirements.append(
            "Wire the loadable Protocol 051/A+ surface edge scorer into the Protocol 101 no-order live entry router."
        )
        requirements.append(
            "Use the live scorer output to populate `edge` and edge-history features causally; do not substitute zeros."
        )
    elif feature_audit.get("status") != "pass":
        requirements.append(
            "Port the frozen Protocol 051/A+ surface scorer into the live entry router so `edge` and edge-history features are produced causally."
        )
        requirements.append(
            "Do not fill Protocol 101 edge features with zero for live capture; that would be an out-of-distribution model input."
        )
    if not requirements:
        requirements.append("Run Protocol 101 no-order live capture with order placement disabled.")
    return requirements


def _threshold_from_summary(path: Path, fold: str, seed: int) -> float | None:
    summary = _load_json(path)
    for row in summary.get("fold_results", []):
        if str(row.get("fold")) == fold and int(row.get("seed", -1)) == int(seed):
            try:
                return float(row.get("threshold"))
            except (TypeError, ValueError):
                return None
    return None


def write_report(path: Path, payload: dict[str, Any]) -> None:
    artifact = payload["protocol101_artifact"]
    ibkr = payload["ibkr_entitlement"]
    feature_audit = payload["feature_dependency_audit"]
    lines = [
        "# Protocol 119: Protocol101 Live Readiness",
        "",
        "No paid data was downloaded. This script did not call IBKR and did not place orders.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Protocol 101 artifact loadable: `{artifact['loadable']}`",
        f"- Protocol 101 feature count: `{artifact['feature_count']}`",
        f"- Protocol 118 rehearsal decision: `{payload['protocol118_decision']}`",
        f"- IBKR entitlement decision: `{ibkr['decision']}`",
        f"- IBKR blocked reason: `{ibkr['blocked_reason']}`",
        f"- Feature dependency status: `{feature_audit['status']}`",
        "",
        "## Next Requirements",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["next_requirements"])
    lines += [
        "",
        "## Feature Dependency",
        "",
        f"- Live-derivable features: `{len(feature_audit['live_derivable_features'])}`",
        f"- Surface edge generator loadable: `{feature_audit.get('surface_edge_generator_loadable')}`",
        f"- Protocol 101 entry router edge wired: `{feature_audit.get('protocol101_entry_router_edge_wired')}`",
        f"- Upstream edge features: `{feature_audit['upstream_edge_features']}`",
        f"- Unknown features: `{feature_audit['unknown_features']}`",
        "",
        feature_audit["why_it_matters"],
        "",
        "## IBKR Attempts",
        "",
        "```json",
        json.dumps(ibkr.get("connection_attempts", []), indent=2, sort_keys=True),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n")


def append_ledger(ledger: Path, payload: dict[str, Any], report_path: Path) -> None:
    marker = "## 2026-05-14 Protocol 119 Protocol101 Live Readiness"
    entry = f"""

{marker}

```text
Date: 2026-05-14
Decision / Experiment: Checked whether frozen Protocol 101 is ready for no-order live shadow capture.
Reason: Protocol 118 passed historical no-order rehearsal, so the next gate is live-readiness rather than another historical model tweak.
Data Used: Existing Protocol 101 artifacts, Protocol 118 summary, and the no-order IBKR entitlement check output only. No paid data was downloaded, this script did not call IBKR, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision {payload['decision']}. Report: {report_path}
Next Gate: Resolve the listed blockers before Protocol 101 no-order live capture: {', '.join(payload['next_requirements'])}
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


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
