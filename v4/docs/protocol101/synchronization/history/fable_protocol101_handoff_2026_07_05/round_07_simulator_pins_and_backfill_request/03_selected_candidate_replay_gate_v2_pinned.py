"""Strict one-account replay gate for fair-contract selected candidates.

This is the bridge between selected entry export and any later lifecycle/shadow
validation. It uses only exported selected-candidate rows and the canonical
Protocol101 serial simulator, enforces one account, one open position, and
ask-entry affordability, and stays in a waiting state until a future
owner-approved candidate export exists.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from v4.model.protocol101_serial_simulator import (
    ACCOUNT_CONTINUITY,
    CASH_BASIS,
    COOLDOWN_ANCHOR,
    DAILY_LOSS_BASIS,
    EXIT_TIME_SEMANTICS,
    FEE_MODEL,
    NO_NEW_ENTRIES_AFTER_ET,
    PROTOCOL101_SERIAL_SIMULATOR_VERSION,
    STRESS_APPLICATION,
    SerialCandidate,
    SerialSimulatorConfig,
    simulate_serial_candidates,
)


DEFAULT_SELECTED_EXPORT = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_selected_candidate_export/summary.json"
)
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_selected_candidate_replay_gate"
)
STRICT_REPLAY_IMPLEMENTATION_VERSION = "selected_candidate_replay_gate_v2"


@dataclass(frozen=True)
class ReplayTrade:
    split: str
    session: str
    decision_time: str
    synthetic_exit_time: str
    contract_id: str
    right: str
    offset: float
    entry_ask: float
    premium_at_risk: float
    score: float
    raw_label_pnl: float
    stressed_pnl: float
    cash_before: float
    cash_after: float
    feature_hash: str
    source_quote_time: str
    source_context_time: str
    realized_pnl_for_daily_loss: float
    cash_pnl_for_account_state: float
    simulator_version: str
    daily_loss_basis: str
    cash_basis: str
    stress_application: str
    exit_time_semantics: str
    cooldown_anchor: str
    no_new_entries_after: str
    fee_model: str
    account_continuity: str
    stress_per_trade_dollars: float
    candidate_stream_hash: str
    simulator_semantics_hash: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-export", type=Path, default=DEFAULT_SELECTED_EXPORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--starting-cash", type=float, default=10_000.0)
    parser.add_argument("--contract-multiplier", type=float, default=100.0)
    parser.add_argument("--stress-per-side", type=float, default=0.10)
    parser.add_argument("--min-validation-trades", type=int, default=20)
    parser.add_argument("--min-diagnostic-trades", type=int, default=20)
    parser.add_argument("--min-profit-factor", type=float, default=1.25)
    parser.add_argument("--max-drawdown-pct-of-start", type=float, default=0.35)
    parser.add_argument("--max-trades-per-session", type=int, default=0)
    parser.add_argument("--max-daily-loss", type=float, default=0.0)
    return parser.parse_args()


def load_json_optional(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def parse_time(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_selected_candidates(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def waiting_payload(reason: str) -> dict[str, Any]:
    return {
        "schema_version": "Protocol101FairContractSelectedCandidateReplayGateV1",
        "implementation_version": STRICT_REPLAY_IMPLEMENTATION_VERSION,
        "status": "waiting_for_owner_approved_training_result",
        "decision": "no_selected_candidates_available_for_strict_replay",
        "strict_replay_executed": False,
        "model_training_executed_here": False,
        "threshold_tuning_executed_here": False,
        "broker_endpoint_called": False,
        "paper_submit_allowed": False,
        "simulator_version": PROTOCOL101_SERIAL_SIMULATOR_VERSION,
        "daily_loss_basis": DAILY_LOSS_BASIS,
        "cash_basis": CASH_BASIS,
        "stress_application": STRESS_APPLICATION,
        "exit_time_semantics": EXIT_TIME_SEMANTICS,
        "cooldown_anchor": COOLDOWN_ANCHOR,
        "no_new_entries_after": NO_NEW_ENTRIES_AFTER_ET,
        "fee_model": FEE_MODEL,
        "account_continuity": ACCOUNT_CONTINUITY,
        "blockers": [reason],
        "checks": {},
        "metrics": {},
        "outputs": {},
    }


def _maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return safe_float(value)


def _max_hold_from_row(row: dict[str, Any]) -> float | None:
    for key in ("max_hold_minutes", "policy_max_hold_minutes", "hold_minutes"):
        value = _maybe_float(row.get(key))
        if value is not None:
            return value
    policy_name = str(row.get("policy_name") or "")
    match = re.search(r"hold(\d+(?:\.\d+)?)m", policy_name)
    if match:
        return float(match.group(1))
    return None


def replay_selected_candidates(
    rows: list[dict[str, Any]],
    *,
    starting_cash: float,
    contract_multiplier: float,
    stress_per_side: float,
    max_trades_per_session: int = 0,
    max_daily_loss: float = 0.0,
) -> tuple[list[ReplayTrade], dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row.get("split")),
            str(row.get("session")),
            str(row.get("decision_time")),
        ),
    )
    skipped = {
        "missing_required_field": 0,
    }
    candidates: list[SerialCandidate] = []
    required = (
        "split",
        "session",
        "decision_time",
        "contract_id",
        "right",
        "offset",
        "entry_ask",
        "score",
        "label_net_pnl",
        "feature_hash",
        "source_quote_time",
        "source_context_time",
        "cooldown_minutes",
    )
    for row in ordered:
        if any(str(row.get(key) or "") == "" for key in required):
            skipped["missing_required_field"] += 1
            continue
        decision_time = parse_time(row.get("decision_time"))
        if decision_time is None:
            skipped["missing_required_field"] += 1
            continue
        entry_ask = safe_float(row.get("entry_ask"))
        label_pnl = safe_float(row.get("label_net_pnl"))
        score = safe_float(row.get("score"))
        offset = safe_float(row.get("offset"))
        cooldown = safe_float(row.get("cooldown_minutes"))
        if any(value is None for value in (entry_ask, label_pnl, score, offset, cooldown)):
            skipped["missing_required_field"] += 1
            continue
        candidates.append(
            SerialCandidate(
                split=str(row["split"]),
                session=str(row["session"]),
                decision_time=decision_time,
                contract_id=str(row["contract_id"]),
                right=str(row["right"]),
                offset=float(offset),
                entry_ask=float(entry_ask),
                score=float(score),
                raw_label_pnl=float(label_pnl),
                cooldown_minutes=float(cooldown),
                max_hold_minutes=_max_hold_from_row(row),
                feature_hash=str(row["feature_hash"]),
                source_quote_time=str(row["source_quote_time"]),
                source_context_time=str(row["source_context_time"]),
            )
        )
    simulator_trades, simulator_state = simulate_serial_candidates(
        candidates,
        config=SerialSimulatorConfig(
            starting_cash=float(starting_cash),
            contract_multiplier=float(contract_multiplier),
            max_trades_per_session=max(int(max_trades_per_session), 0),
            max_daily_loss=max(float(max_daily_loss), 0.0),
            stress_per_trade=2.0 * float(stress_per_side) * float(contract_multiplier),
        ),
    )
    for name, count in simulator_state.skipped.items():
        skipped[name] = int(count)
    trades = [
        ReplayTrade(
            split=trade.split,
            session=trade.session,
            decision_time=trade.decision_time,
            synthetic_exit_time=trade.synthetic_exit_time,
            contract_id=trade.contract_id,
            right=trade.right,
            offset=trade.offset,
            entry_ask=trade.entry_ask,
            premium_at_risk=trade.premium_at_risk,
            score=trade.score,
            raw_label_pnl=trade.raw_label_pnl,
            stressed_pnl=trade.stressed_pnl,
            cash_before=trade.cash_before,
            cash_after=trade.cash_after,
            feature_hash=trade.feature_hash,
            source_quote_time=trade.source_quote_time,
            source_context_time=trade.source_context_time,
            realized_pnl_for_daily_loss=trade.realized_pnl_for_daily_loss,
            cash_pnl_for_account_state=trade.cash_pnl_for_account_state,
            simulator_version=trade.simulator_version,
            daily_loss_basis=trade.daily_loss_basis,
            cash_basis=trade.cash_basis,
            stress_application=trade.stress_application,
            exit_time_semantics=trade.exit_time_semantics,
            cooldown_anchor=trade.cooldown_anchor,
            no_new_entries_after=trade.no_new_entries_after,
            fee_model=trade.fee_model,
            account_continuity=trade.account_continuity,
            stress_per_trade_dollars=trade.stress_per_trade_dollars,
            candidate_stream_hash=trade.candidate_stream_hash,
            simulator_semantics_hash=trade.simulator_semantics_hash,
        )
        for trade in simulator_trades
    ]
    return trades, {
        "skipped": skipped,
        "cash_by_split": simulator_state.cash_by_account,
        "equity_by_split": simulator_state.equity_by_account,
        "realized_raw_pnl_by_split_session": simulator_state.realized_raw_pnl_by_split_session,
        "simulator_version": PROTOCOL101_SERIAL_SIMULATOR_VERSION,
        "daily_loss_basis": DAILY_LOSS_BASIS,
        "cash_basis": CASH_BASIS,
        "stress_application": STRESS_APPLICATION,
        "exit_time_semantics": EXIT_TIME_SEMANTICS,
        "cooldown_anchor": COOLDOWN_ANCHOR,
        "no_new_entries_after": NO_NEW_ENTRIES_AFTER_ET,
        "fee_model": FEE_MODEL,
        "account_continuity": ACCOUNT_CONTINUITY,
        "stress_per_trade_dollars": 2.0 * float(stress_per_side) * float(contract_multiplier),
        "candidate_stream_hash": simulator_state.candidate_stream_hash,
        "simulator_semantics_hash": simulator_state.simulator_semantics_hash,
        "simulator_semantics": simulator_state.semantics,
    }


def metrics_for_trades(trades: list[ReplayTrade], *, starting_cash: float) -> dict[str, Any]:
    if not trades:
        return {
            "trades": 0,
            "total_pnl": 0.0,
            "ending_equity": float(starting_cash),
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct_of_start": 0.0,
            "sessions_traded": 0,
        }
    pnl = np.asarray([trade.stressed_pnl for trade in trades], dtype=float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    equity = starting_cash + np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate([[starting_cash], equity]))[1:]
    drawdown = equity - peak
    gross_loss = abs(float(losses.sum()))
    return {
        "trades": int(len(trades)),
        "total_pnl": float(pnl.sum()),
        "ending_equity": float(starting_cash + pnl.sum()),
        "profit_factor": float(wins.sum() / gross_loss) if gross_loss > 0 else (float("inf") if wins.sum() > 0 else 0.0),
        "win_rate": float((pnl > 0).mean()),
        "avg_pnl": float(pnl.mean()),
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "max_drawdown_pct_of_start": float(drawdown.min() / starting_cash) if len(drawdown) and starting_cash > 0 else 0.0,
        "sessions_traded": int(len({trade.session for trade in trades})),
    }


def build_checks(
    *,
    metrics_by_split: dict[str, dict[str, Any]],
    skipped: dict[str, int],
    selected_export: dict[str, Any],
    min_validation_trades: int,
    min_diagnostic_trades: int,
    min_profit_factor: float,
    max_drawdown_pct_of_start: float,
) -> dict[str, dict[str, Any]]:
    validation = metrics_by_split.get("validation", {})
    diagnostic = metrics_by_split.get("diagnostic_test", {})
    values = {
        "selected_export_passed": (
            selected_export.get("status"),
            "pass",
            selected_export.get("status") == "pass",
        ),
        "broker_endpoint_called": (
            bool(selected_export.get("broker_endpoint_called")),
            False,
            not bool(selected_export.get("broker_endpoint_called")),
        ),
        "paper_submit_allowed": (
            bool(selected_export.get("paper_submit_allowed")),
            False,
            not bool(selected_export.get("paper_submit_allowed")),
        ),
        "no_missing_required_fields": (
            int(skipped.get("missing_required_field", 0)),
            0,
            int(skipped.get("missing_required_field", 0)) == 0,
        ),
        "no_overlap_skips": (
            int(skipped.get("overlap", 0)),
            0,
            int(skipped.get("overlap", 0)) == 0,
        ),
        "no_unaffordable_skips": (
            int(skipped.get("unaffordable", 0)),
            0,
            int(skipped.get("unaffordable", 0)) == 0,
        ),
        "validation_trade_count": (
            int(validation.get("trades") or 0),
            min_validation_trades,
            int(validation.get("trades") or 0) >= min_validation_trades,
        ),
        "diagnostic_trade_count": (
            int(diagnostic.get("trades") or 0),
            min_diagnostic_trades,
            int(diagnostic.get("trades") or 0) >= min_diagnostic_trades,
        ),
        "validation_positive_pnl": (
            float(validation.get("total_pnl") or 0.0),
            ">0",
            float(validation.get("total_pnl") or 0.0) > 0.0,
        ),
        "diagnostic_positive_pnl": (
            float(diagnostic.get("total_pnl") or 0.0),
            ">0",
            float(diagnostic.get("total_pnl") or 0.0) > 0.0,
        ),
        "validation_profit_factor": (
            float(validation.get("profit_factor") or 0.0),
            min_profit_factor,
            float(validation.get("profit_factor") or 0.0) >= min_profit_factor,
        ),
        "diagnostic_profit_factor": (
            float(diagnostic.get("profit_factor") or 0.0),
            min_profit_factor,
            float(diagnostic.get("profit_factor") or 0.0) >= min_profit_factor,
        ),
        "validation_drawdown": (
            abs(float(validation.get("max_drawdown_pct_of_start") or 0.0)),
            max_drawdown_pct_of_start,
            abs(float(validation.get("max_drawdown_pct_of_start") or 0.0)) <= max_drawdown_pct_of_start,
        ),
        "diagnostic_drawdown": (
            abs(float(diagnostic.get("max_drawdown_pct_of_start") or 0.0)),
            max_drawdown_pct_of_start,
            abs(float(diagnostic.get("max_drawdown_pct_of_start") or 0.0)) <= max_drawdown_pct_of_start,
        ),
    }
    return {
        name: {"value": value, "required": required, "pass": passed}
        for name, (value, required, passed) in values.items()
    }


def write_csv(path: Path, trades: list[ReplayTrade]) -> None:
    fields = list(asdict(trades[0]).keys()) if trades else [field for field in ReplayTrade.__dataclass_fields__]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for trade in trades:
            writer.writerow(asdict(trade))


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Protocol101 Fair-Contract Selected Candidate Replay Gate",
        "",
        "## Decision",
        "",
        f"- Status: `{payload['status']}`",
        f"- Decision: `{payload['decision']}`",
        f"- Strict replay executed: `{str(payload['strict_replay_executed']).lower()}`",
        f"- Model training executed here: `{str(payload['model_training_executed_here']).lower()}`",
        f"- Threshold tuning executed here: `{str(payload['threshold_tuning_executed_here']).lower()}`",
        f"- Broker endpoint called: `{str(payload['broker_endpoint_called']).lower()}`",
        f"- Paper-submit allowed: `{str(payload['paper_submit_allowed']).lower()}`",
        f"- Simulator version: `{payload.get('simulator_version', '')}`",
        f"- Daily-loss basis: `{payload.get('daily_loss_basis', '')}`",
        f"- Cash basis: `{payload.get('cash_basis', '')}`",
        f"- Stress application: `{payload.get('stress_application', '')}`",
        f"- Exit-time semantics: `{payload.get('exit_time_semantics', '')}`",
        f"- Cooldown anchor: `{payload.get('cooldown_anchor', '')}`",
        f"- No new entries after: `{payload.get('no_new_entries_after', '')}`",
        f"- Fee model: `{payload.get('fee_model', '')}`",
        f"- Account continuity: `{payload.get('account_continuity', '')}`",
        f"- Stress per trade dollars: `{payload.get('stress_per_trade_dollars', '')}`",
        f"- Candidate stream hash: `{payload.get('candidate_stream_hash', '')}`",
        f"- Simulator semantics hash: `{payload.get('simulator_semantics_hash', '')}`",
        f"- Max trades per session: `{payload.get('max_trades_per_session', 0)}`",
        f"- Max daily loss: `{payload.get('max_daily_loss', 0.0)}`",
        "",
        "## Metrics",
        "",
    ]
    for split, metrics in (payload.get("metrics") or {}).items():
        lines.append(
            f"- `{split}`: trades=`{metrics.get('trades')}`, total_pnl=`{metrics.get('total_pnl')}`, "
            f"profit_factor=`{metrics.get('profit_factor')}`, max_dd_pct_start=`{metrics.get('max_drawdown_pct_of_start')}`."
        )
    if payload.get("checks"):
        lines.extend(["", "## Checks", ""])
        for name, check in payload["checks"].items():
            lines.append(
                f"- `{name}`: value=`{check.get('value')}`, required=`{check.get('required')}`, pass=`{str(check.get('pass')).lower()}`."
            )
    if payload.get("blockers"):
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- `{item}`" for item in payload["blockers"])
    lines.extend(["", "## Outputs", ""])
    for name, path in (payload.get("outputs") or {}).items():
        lines.append(f"- `{name}`: `{path}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected_export = load_json_optional(args.selected_export)
    if not selected_export:
        payload = waiting_payload("selected_export_summary_missing")
    elif selected_export.get("status") != "pass":
        payload = waiting_payload(f"selected_export_not_pass:{selected_export.get('status')}")
    else:
        csv_path = Path(str((selected_export.get("outputs") or {}).get("selected_candidates_csv") or ""))
        if not csv_path.exists():
            payload = waiting_payload("selected_candidates_csv_missing")
        else:
            rows = load_selected_candidates(csv_path)
            trades, replay_state = replay_selected_candidates(
                rows,
                starting_cash=float(args.starting_cash),
                contract_multiplier=float(args.contract_multiplier),
                stress_per_side=float(args.stress_per_side),
                max_trades_per_session=max(int(args.max_trades_per_session), 0),
                max_daily_loss=max(float(args.max_daily_loss), 0.0),
            )
            trades_by_split: dict[str, list[ReplayTrade]] = {}
            for trade in trades:
                trades_by_split.setdefault(trade.split, []).append(trade)
            metrics = {
                split: metrics_for_trades(split_trades, starting_cash=float(args.starting_cash))
                for split, split_trades in sorted(trades_by_split.items())
            }
            checks = build_checks(
                metrics_by_split=metrics,
                skipped=replay_state["skipped"],
                selected_export=selected_export,
                min_validation_trades=int(args.min_validation_trades),
                min_diagnostic_trades=int(args.min_diagnostic_trades),
                min_profit_factor=float(args.min_profit_factor),
                max_drawdown_pct_of_start=float(args.max_drawdown_pct_of_start),
            )
            passed = all(check["pass"] for check in checks.values())
            trades_csv = args.out_dir / "strict_replay_trades.csv"
            write_csv(trades_csv, trades)
            payload = {
                "schema_version": "Protocol101FairContractSelectedCandidateReplayGateV1",
                "implementation_version": STRICT_REPLAY_IMPLEMENTATION_VERSION,
                "status": "pass" if passed else "fail",
                "decision": (
                    "selected_candidates_pass_strict_replay_gate"
                    if passed
                    else "selected_candidates_fail_strict_replay_gate"
                ),
                "strict_replay_executed": True,
                "model_training_executed_here": False,
                "threshold_tuning_executed_here": False,
                "broker_endpoint_called": False,
                "paper_submit_allowed": False,
                "starting_cash": float(args.starting_cash),
                "contract_multiplier": float(args.contract_multiplier),
                "stress_per_side": float(args.stress_per_side),
                "max_trades_per_session": max(int(args.max_trades_per_session), 0),
                "max_daily_loss": max(float(args.max_daily_loss), 0.0),
                "simulator_version": replay_state["simulator_version"],
                "daily_loss_basis": replay_state["daily_loss_basis"],
                "cash_basis": replay_state["cash_basis"],
                "stress_application": replay_state["stress_application"],
                "exit_time_semantics": replay_state["exit_time_semantics"],
                "cooldown_anchor": replay_state["cooldown_anchor"],
                "no_new_entries_after": replay_state["no_new_entries_after"],
                "fee_model": replay_state["fee_model"],
                "account_continuity": replay_state["account_continuity"],
                "stress_per_trade_dollars": replay_state["stress_per_trade_dollars"],
                "candidate_stream_hash": replay_state["candidate_stream_hash"],
                "simulator_semantics_hash": replay_state["simulator_semantics_hash"],
                "simulator_semantics": replay_state["simulator_semantics"],
                "checks": checks,
                "metrics": metrics,
                "skipped": replay_state["skipped"],
                "blockers": [
                    name for name, check in checks.items() if not bool(check.get("pass"))
                ],
                "outputs": {
                    "strict_replay_trades_csv": str(trades_csv),
                },
            }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "report.md").write_text(render_report(payload))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "report": str(args.out_dir / "report.md"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
