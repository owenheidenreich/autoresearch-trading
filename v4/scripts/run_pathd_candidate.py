"""Separate Path-D candidate runner with no submission-capable mode."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

from v4.checks.paid_data_guard import add_paid_data_approval_args
from v4.path_d.contracts import (
    BrokerStateSnapshotV1,
    ExecutionIntentV1,
    PositionV1,
)
from v4.path_d.risk.governor import FeedHealthV1
from v4.path_d.runtime.ibkr_paper_dry_run import qualify_and_preview
from v4.research.pathd_phase1_entry import load_entry_artifact
from v4.research.phase1_exit_model import load_exit_artifact


# Frozen in v4/audit/autoresearch/thetadata_completed_minute_timing_2026_08_03/
# shared_emission_lag.json (status FROZEN, L = ceil(max observed) over five
# samples spanning 608-2335 ms). Thin and dispersed; re-derive from the
# Track-A union and only ever raise it.
FROZEN_EMISSION_LAG_MS = 2336

MODES = ("databento-no-order", "ibkr-paper-dry-run", "joint-live-no-order")

# Path-D's production target is "Databento Live OPRA for decisions, IBKR for
# execution only", but the two adapters had never run in one process: the modes
# above were mutually exclusive, so nothing measured the clock that actually
# governs legality -- OPRA completed-interval end to IBKR order-ready.
#
# Training emits a decision at interval end plus a frozen emission lag L. If
# the real end-to-end path is slower than L, the model is deciding earlier than
# is live-possible. That is the signed18 look-ahead class, and no offline
# artifact can detect it.
JOINT_SCHEMA = "pathd.candidate-joint-live-no-order.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Path-D development candidate; broker submission is structurally unavailable."
    )
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--entry-manifest", type=Path, required=True)
    parser.add_argument("--exit-manifest", type=Path, required=True)
    parser.add_argument("--session-date")
    parser.add_argument("--definition-duration-seconds", type=float, default=30.0)
    parser.add_argument("--quote-duration-seconds", type=float, default=300.0)
    parser.add_argument("--schemas", nargs="+", default=("cbbo-1m", "cbbo-1s"))
    parser.add_argument("--env-file", type=Path, default=Path("v4/.env"))
    add_paid_data_approval_args(parser)
    parser.add_argument("--intent", type=Path, action="append", default=[])
    parser.add_argument("--ibkr-host", default="127.0.0.1")
    parser.add_argument("--ibkr-port", type=int, default=4002)
    parser.add_argument("--ibkr-client-id", type=int, default=191)
    parser.add_argument("--sealed-shadow-position", action="store_true")
    return parser.parse_args()


def _artifact_bindings(args: argparse.Namespace) -> dict[str, Any]:
    entry = load_entry_artifact(args.entry_manifest)
    exit_model = load_exit_artifact(args.exit_manifest)
    entry_manifest = json.loads(args.entry_manifest.read_text())
    exit_manifest = json.loads(args.exit_manifest.read_text())
    if entry.role != "FULL_DEVELOPMENT_SHADOW_ONLY":
        raise RuntimeError("candidate runner requires the entry shadow-only artifact")
    if exit_manifest.get("role") != "FULL_DEVELOPMENT_SHADOW_ONLY":
        raise RuntimeError("candidate runner requires the exit shadow-only artifact")
    return {
        "entry_manifest": str(args.entry_manifest.resolve()),
        "entry_manifest_sha256": entry_manifest["manifest_sha256"],
        "entry_feature_names": list(entry.feature_names),
        "exit_manifest": str(args.exit_manifest.resolve()),
        "exit_manifest_sha256": exit_manifest["manifest_sha256"],
        "exit_feature_names": list(exit_model.feature_names),
        "entry_artifact_may_generate_exit_training": False,
    }


def _approval_args(args: argparse.Namespace) -> list[str]:
    result = [
        "--approval-manifest", str(args.approval_manifest),
        "--approval-env-var", str(args.approval_env_var),
    ]
    if args.approval_text is not None:
        result.extend(("--approval-text", str(args.approval_text)))
    return result


def _databento_no_order(args: argparse.Namespace) -> dict[str, Any]:
    if not args.session_date:
        raise SystemExit("--session-date is required in databento-no-order mode")
    artifacts = _artifact_bindings(args)
    definition_dir = args.output_dir / "definitions"
    quote_dir = args.output_dir / "quotes"
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"output directory must be absent or empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common = ["--session-date", args.session_date, "--env-file", str(args.env_file)] + _approval_args(args)
    subprocess.run(
        [
            sys.executable,
            "-m", "v4.scripts.capture_databento_live_opra_definitions",
            *common,
            "--duration-seconds", str(args.definition_duration_seconds),
            "--output-dir", str(definition_dir),
        ],
        check=True,
    )
    definition_summary = json.loads(
        (definition_dir / "definition_capture_summary.json").read_text()
    )
    definition_path = Path(definition_summary["current_session_parquet"]["path"])
    subprocess.run(
        [
            sys.executable,
            "-m", "v4.scripts.capture_databento_live_opra_training_twin",
            *common,
            "--definition-path", str(definition_path),
            "--duration-seconds", str(args.quote_duration_seconds),
            "--schemas", *args.schemas,
            "--output-dir", str(quote_dir),
        ],
        check=True,
    )
    quote_summary = json.loads((quote_dir / "capture_summary.json").read_text())
    return {
        "schema_version": "pathd.candidate-databento-no-order.v1",
        "status": "CAPTURED_NO_ORDER",
        "mode": "databento-no-order",
        "definitions": definition_summary,
        "quotes": quote_summary,
        "artifacts": artifacts,
        "ibkr_accessed": False,
        "broker_submit_endpoint_called": False,
        "paper_order_submitted": False,
    }


def _account_value(values: list[Any], tag: str, *, default: float | None = None) -> float:
    matches = [float(value.value) for value in values if value.tag == tag]
    if matches:
        return matches[-1]
    if default is not None:
        return default
    raise RuntimeError(f"IBKR account summary missing {tag}")


def _ibkr_dry_run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.intent:
        raise SystemExit("at least one --intent is required in ibkr-paper-dry-run mode")
    artifacts = _artifact_bindings(args)
    from ib_insync import IB, LimitOrder, Option

    intents = [ExecutionIntentV1.from_dict(json.loads(path.read_text())) for path in args.intent]
    ib = IB()
    ib.connect(
        args.ibkr_host,
        args.ibkr_port,
        clientId=args.ibkr_client_id,
        readonly=True,
    )
    try:
        accounts = [value for value in ib.managedAccounts() if str(value).startswith("DU")]
        if len(accounts) != 1:
            raise RuntimeError(f"exactly one DU paper account required, observed {len(accounts)}")
        account_id = accounts[0]
        values = list(ib.accountSummary(account_id) or [])
        available = _account_value(values, "AvailableFunds")
        daily_pnl = _account_value(values, "RealizedPnL", default=0.0)
        captured = _utc_now()
        positions: list[PositionV1] = []
        for row in ib.positions(account_id):
            contract = row.contract
            symbol = str(getattr(contract, "localSymbol", "") or "")
            if float(row.position) <= 0 or len(symbol) != 21 or not symbol.startswith("SPXW  "):
                continue
            positions.append(
                PositionV1(
                    osi_symbol=symbol,
                    quantity=int(row.position),
                    average_cost_micros=max(0, int(round(float(row.avgCost) * 1_000_000))),
                    opened_at_utc=captured,
                )
            )
        previews = []
        for intent in intents:
            snapshot_version = intent.state_precondition.position_snapshot_version
            intent_positions = list(positions)
            if args.sealed_shadow_position and intent.state_precondition.expected_position == "LONG_ONE":
                intent_positions = [
                    PositionV1(
                        osi_symbol=intent.contract.osi_symbol,
                        quantity=intent.decision.quantity,
                        average_cost_micros=intent.price_budget.reference_ask_micros,
                        opened_at_utc=intent.clocks.event_interval_end_utc,
                    )
                ]
            state = BrokerStateSnapshotV1(
                schema_version="pathd.broker_state_snapshot.v1",
                snapshot_version=snapshot_version,
                captured_at_utc=captured,
                account_id_redacted="DU…" + account_id[-2:],
                available_funds_micros=max(0, int(round(available * 1_000_000))),
                daily_pnl_micros=int(round(daily_pnl * 1_000_000)),
                open_positions=tuple(intent_positions),
                connectivity="CONNECTED",
                source="IBKR",
            )
            feed = FeedHealthV1(
                option_received_timestamp_utc=intent.clocks.option_received_watermark_utc,
                spx_received_timestamp_utc=intent.clocks.spx_received_watermark_utc,
            )
            previews.append(asdict(qualify_and_preview(
                ib=ib,
                option_cls=Option,
                order_cls=LimitOrder,
                account_id=account_id,
                intent=intent,
                broker_state=state,
                feed_health=feed,
                now_utc=_utc_now(),
            )))
        return {
            "schema_version": "pathd.candidate-ibkr-paper-dry-run.v1",
            "status": "DRY_RUN_COMPLETE",
            "mode": "ibkr-paper-dry-run",
            "ibkr_readonly_connection": True,
            "account_id_redacted": "DU…" + account_id[-2:],
            "snapshot_versions": [
                intent.state_precondition.position_snapshot_version for intent in intents
            ],
            "previews": previews,
            "artifacts": artifacts,
            "broker_submit_endpoint_called": False,
            "paper_order_submitted": False,
        }
    finally:
        ib.disconnect()


def joint_latency_ledger(
    quote_summary: Mapping[str, Any],
    *,
    ibkr_ready_ns: int,
    emission_lag_ms: int = FROZEN_EMISSION_LAG_MS,
) -> dict[str, Any]:
    """Tie the OPRA arrival clock to the IBKR order-ready clock.

    ``receipt_minus_interval_end_ns`` is the only measurement that answers "how
    stale is the quote when we first hold it"; the broker leg is measured here
    as wall clock from the last captured record to a qualified, previewed
    order. Their sum is the earliest a decision on that interval could reach
    the broker, and it must not exceed the emission lag the model trained on.

    Returns a verdict rather than raising: a slow run is evidence, and the
    caller records it. Only a decision made FASTER than reality is a defect.
    """

    families = quote_summary.get("receipt_minus_interval_end_ns") or {}
    worst_p99_ns = 0
    per_family: dict[str, Any] = {}
    for name, quantiles in families.items():
        if not isinstance(quantiles, Mapping):
            continue
        p99 = int(quantiles.get("p99") or 0)
        per_family[str(name)] = {"p99_ns": p99, "p99_ms": round(p99 / 1e6, 3)}
        worst_p99_ns = max(worst_p99_ns, p99)
    total_ns = worst_p99_ns + max(0, int(ibkr_ready_ns))
    budget_ns = int(emission_lag_ms) * 1_000_000
    return {
        "schema_version": "pathd.joint-latency-ledger.v1",
        "opra_receipt_minus_interval_end_p99_by_family": per_family,
        "opra_worst_p99_ns": worst_p99_ns,
        "ibkr_qualify_and_preview_ns": int(ibkr_ready_ns),
        "end_to_end_ns": total_ns,
        "end_to_end_ms": round(total_ns / 1e6, 3),
        "frozen_emission_lag_ms": int(emission_lag_ms),
        "within_emission_lag_budget": total_ns <= budget_ns,
        "verdict": (
            "END_TO_END_WITHIN_TRAINED_EMISSION_LAG"
            if total_ns <= budget_ns
            else "END_TO_END_EXCEEDS_TRAINED_EMISSION_LAG"
        ),
        "interpretation": (
            "The trained emission lag is an upper bound on how late a decision "
            "may be. Exceeding it means the live path cannot reproduce the "
            "training clock and the lag must be re-frozen upward, never the "
            "measurement discarded."
        ),
    }


def _joint_live_no_order(args: argparse.Namespace) -> dict[str, Any]:
    """One process: live OPRA capture, then IBKR readonly preview. No submit."""

    if not args.session_date:
        raise SystemExit("--session-date is required in joint-live-no-order mode")
    if not args.intent:
        raise SystemExit("at least one --intent is required in joint-live-no-order mode")
    captured = _databento_no_order(args)
    broker_started_ns = time.time_ns()
    previewed = _ibkr_dry_run(args)
    ibkr_ready_ns = time.time_ns() - broker_started_ns
    latency = joint_latency_ledger(captured["quotes"], ibkr_ready_ns=ibkr_ready_ns)
    # Both legs assert this independently; re-assert on the joined payload so a
    # future edit to either cannot quietly relax it here.
    if previewed["broker_submit_endpoint_called"] or captured["broker_submit_endpoint_called"]:
        raise RuntimeError("joint mode observed a broker submit endpoint call")
    return {
        "schema_version": JOINT_SCHEMA,
        "status": "JOINT_LIVE_NO_ORDER_COMPLETE",
        "mode": "joint-live-no-order",
        "session_date": args.session_date,
        "definitions": captured["definitions"],
        "quotes": captured["quotes"],
        "previews": previewed["previews"],
        "account_id_redacted": previewed["account_id_redacted"],
        "snapshot_versions": previewed["snapshot_versions"],
        "latency": latency,
        "artifacts": previewed["artifacts"],
        "vendors_exercised_in_one_process": ["DATABENTO_LIVE_OPRA", "IBKR"],
        "ibkr_readonly_connection": True,
        "broker_submit_endpoint_called": False,
        "paper_order_submitted": False,
    }


def main() -> int:
    args = parse_args()
    if args.mode == "databento-no-order":
        payload = _databento_no_order(args)
    elif args.mode == "ibkr-paper-dry-run":
        payload = _ibkr_dry_run(args)
    else:
        payload = _joint_live_no_order(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "candidate_run.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
