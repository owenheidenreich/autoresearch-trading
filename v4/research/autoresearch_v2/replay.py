"""Exact two-clock serial replay and matched lockstep admission."""
from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from v4.model.protocol101_serial_simulator_v5 import (
    PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
    SerialCandidateV5,
    SerialReplayTradeV5,
    SerialSimulatorV5Config,
    simulate_serial_candidates_v5,
)

from .dataset import POLICY_INDEX, label_columns


NY = ZoneInfo("America/New_York")


class PairedReplayMismatch(RuntimeError):
    pass


def candidate_from_row(
    row: dict[str, Any],
    *,
    policy: str,
    score: float,
    strategy: str,
    split: str = "development",
) -> SerialCandidateV5:
    columns = label_columns(policy)
    entry_ask = float(row["entry_ask"])
    exit_bid = float(row[columns["exit_bid"]])
    return SerialCandidateV5(
        split=split,
        fold=str(row.get("fold", "")),
        session=str(row["session"]),
        decision_time_ns=int(row["decision_time_ns"]),
        contract_id=str(row["contract_id"]),
        right=str(row["right"]),
        canonical_strike_slot=int(row["canonical_strike_slot"]),
        policy_index=int(POLICY_INDEX[policy]),
        entry_ask=entry_ask,
        score=float(score),
        raw_label_pnl_after_campaign_fee=(exit_bid - entry_ask) * 100.0 - 3.0,
        label_mid_pnl_before_campaign_fee=float(row[columns["mid_pnl"]]),
        label_realized_exit_time_ns=int(row[columns["exit_ns"]]),
        label_source_exit_quote_time_ns=int(row[columns["source_exit_ns"]]),
        label_exit_quote_age_ms=float(row[columns["quote_age_ms"]]),
        label_exit_reason_code=int(row[columns["reason"]]),
        label_executable_exit_bid=exit_bid,
        label_policy_deadline_ns=int(row[columns["deadline_ns"]]),
        label_invalid_reason_code=int(row[columns["invalid"]]),
        feature_hash=str(row.get("feature_hash", "autoresearch_v2_signed17")),
        source_quote_time_ns=int(row["source_quote_time_ns"]),
        source_context_time_ns=int(row["source_context_time_ns"]),
        strategy=strategy,
        source_simulator_version=PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
        metadata={
            "candidate_uid": str(row["candidate_uid"]),
            "offset": int(row["offset"]),
            "entry_mid": float(row["entry_mid"]),
        },
    )


def _identity(item: SerialCandidateV5) -> tuple[str, int, str]:
    return item.session, item.decision_time_ns, item.contract_id


def _before_cutoff(item: SerialCandidateV5, cutoff: str) -> bool:
    hour, minute = [int(value) for value in cutoff.split(":")]
    local = pd.Timestamp(item.decision_time_ns, unit="ns", tz="UTC").tz_convert(NY)
    return (local.hour, local.minute) <= (hour, minute)


def valid_for_policy(row: dict[str, Any], policy: str) -> bool:
    columns = label_columns(policy)
    return (
        int(row[columns["invalid"]]) == 0
        and int(row[columns["reason"]]) > 0
        and int(row[columns["exit_ns"]]) > int(row["decision_time_ns"])
        and math.isfinite(float(row[columns["pnl"]]))
        and math.isfinite(float(row[columns["mid_pnl"]]))
        and math.isfinite(float(row[columns["exit_bid"]]))
        and math.isfinite(float(row[columns["quote_age_ms"]]))
    )


def build_candidates(
    rows: Iterable[dict[str, Any]], *, policy: str, strategy: str
) -> list[SerialCandidateV5]:
    return [
        candidate_from_row(
            row,
            policy=policy,
            score=float(row["_pred"]),
            strategy=strategy,
        )
        for row in rows
        if valid_for_policy(row, policy)
    ]


def replay_absolute(
    rows: Iterable[dict[str, Any]], *, policy: str, strategy: str
) -> tuple[list[SerialReplayTradeV5], Any]:
    return simulate_serial_candidates_v5(
        build_candidates(rows, policy=policy, strategy=strategy),
        config=SerialSimulatorV5Config(),
    )


def paired_lockstep_exit_replay(
    rows: Iterable[dict[str, Any]],
    *,
    arm_policy: str,
    baseline_policy: str,
) -> dict[str, Any]:
    """Admit the same intents only when both one-account arms can trade.

    Each arm maintains its own cash, daily loss state, and realized occupancy.
    The shared admission rule guarantees exact entry identity/count/side/premium/
    moneyness while allowing the tested exit policy to change holding time.
    The resulting streams are then reproduced by simulator v5 independently.
    """

    cfg = SerialSimulatorV5Config()
    paired = []
    for row in rows:
        if not valid_for_policy(row, arm_policy) or not valid_for_policy(row, baseline_policy):
            continue
        paired.append(
            (
                candidate_from_row(
                    row,
                    policy=arm_policy,
                    score=float(row["_pred"]),
                    strategy=f"autoresearch_v2:{arm_policy}",
                ),
                candidate_from_row(
                    row,
                    policy=baseline_policy,
                    score=float(row["_pred"]),
                    strategy=f"autoresearch_v2:{baseline_policy}",
                ),
            )
        )
    paired.sort(key=lambda pair: (pair[0].session, pair[0].decision_time_ns, pair[0].contract_id))
    cash = {"arm": cfg.starting_cash, "baseline": cfg.starting_cash}
    pending: dict[str, SerialCandidateV5 | None] = {"arm": None, "baseline": None}
    active_session: str | None = None
    session_start = dict(cash)
    session_pnl = {"arm": 0.0, "baseline": 0.0}
    admitted_arm: list[SerialCandidateV5] = []
    admitted_baseline: list[SerialCandidateV5] = []
    skipped = {"invalid_or_unpaired": 0, "cutoff": 0, "occupancy": 0, "risk": 0}

    def realize(name: str) -> None:
        item = pending[name]
        if item is None:
            return
        cash[name] += item.raw_label_pnl_after_campaign_fee
        session_pnl[name] += item.raw_label_pnl_after_campaign_fee
        pending[name] = None

    for arm, baseline in paired:
        decision = arm.decision_time_ns
        if active_session != arm.session:
            for name in ("arm", "baseline"):
                realize(name)
            active_session = arm.session
            session_start = dict(cash)
            session_pnl = {"arm": 0.0, "baseline": 0.0}
        for name in ("arm", "baseline"):
            item = pending[name]
            if item is not None and item.label_realized_exit_time_ns <= decision:
                realize(name)
        if not _before_cutoff(arm, cfg.no_new_entries_after_et):
            skipped["cutoff"] += 1
            continue
        if pending["arm"] is not None or pending["baseline"] is not None:
            skipped["occupancy"] += 1
            continue
        checks = []
        for name, item in (("arm", arm), ("baseline", baseline)):
            required_cash = item.entry_ask * cfg.contract_multiplier + cfg.affordability_reserve_per_trade
            daily_limit = cfg.max_daily_loss_fraction_of_session_start_equity * session_start[name]
            checks.append(
                required_cash <= cash[name] + 1e-9
                and not (daily_limit > 0 and session_pnl[name] <= -daily_limit)
            )
        if not all(checks):
            skipped["risk"] += 1
            continue
        admitted_arm.append(arm)
        admitted_baseline.append(baseline)
        pending["arm"] = arm
        pending["baseline"] = baseline
    for name in ("arm", "baseline"):
        realize(name)

    arm_trades, arm_state = simulate_serial_candidates_v5(admitted_arm, config=cfg)
    baseline_trades, baseline_state = simulate_serial_candidates_v5(admitted_baseline, config=cfg)
    arm_ids = [_identity(item) for item in arm_trades]
    baseline_ids = [_identity(item) for item in baseline_trades]
    if arm_ids != baseline_ids or len(arm_ids) != len(admitted_arm):
        raise PairedReplayMismatch("simulator v5 did not reproduce matched lockstep admissions")
    arm_session: dict[str, float] = {}
    baseline_session: dict[str, float] = {}
    for trade in arm_trades:
        arm_session[trade.session] = arm_session.get(trade.session, 0.0) + trade.raw_label_pnl_after_campaign_fee
    for trade in baseline_trades:
        baseline_session[trade.session] = baseline_session.get(trade.session, 0.0) + trade.raw_label_pnl_after_campaign_fee
    deltas = {
        session: arm_session.get(session, 0.0) - baseline_session.get(session, 0.0)
        for session in sorted(set(arm_session) | set(baseline_session))
    }
    return {
        "arm_policy": arm_policy,
        "baseline_policy": baseline_policy,
        "matched_trade_count": len(arm_trades),
        "matched_identity": arm_ids == baseline_ids,
        "skipped": skipped,
        "session_deltas": deltas,
        "arm_total_pnl": float(sum(arm_session.values())),
        "baseline_total_pnl": float(sum(baseline_session.values())),
        "arm_final_cash": float(arm_state.cash_by_account["development"]),
        "baseline_final_cash": float(baseline_state.cash_by_account["development"]),
        "arm_state": {
            "skipped": arm_state.skipped,
            "trade_identity_hash": arm_state.trade_identity_hash,
            "simulator_config_hash": arm_state.simulator_config_hash,
        },
        "baseline_state": {
            "skipped": baseline_state.skipped,
            "trade_identity_hash": baseline_state.trade_identity_hash,
            "simulator_config_hash": baseline_state.simulator_config_hash,
        },
        "trade_rows": [
            {
                "session": arm_trade.session,
                "decision_time_ns": arm_trade.decision_time_ns,
                "contract_id": arm_trade.contract_id,
                "right": arm_trade.right,
                "entry_ask": arm_trade.entry_ask,
                "arm_exit_time_ns": arm_trade.label_realized_exit_time_ns,
                "baseline_exit_time_ns": baseline_trade.label_realized_exit_time_ns,
                "arm_pnl": arm_trade.raw_label_pnl_after_campaign_fee,
                "baseline_pnl": baseline_trade.raw_label_pnl_after_campaign_fee,
            }
            for arm_trade, baseline_trade in zip(arm_trades, baseline_trades)
        ],
    }


def paired_lockstep_component_replay(
    pairs: Iterable[tuple[str, dict[str, Any], dict[str, Any]]],
    *,
    policy: str,
    arm_name: str,
    baseline_name: str,
) -> dict[str, Any]:
    """Strict paired replay for differing timing/direction/contract choices."""

    cfg = SerialSimulatorV5Config()
    candidates = []
    for signal_id, arm_row, baseline_row in pairs:
        if not valid_for_policy(arm_row, policy) or not valid_for_policy(baseline_row, policy):
            continue
        candidates.append(
            (
                signal_id,
                candidate_from_row(
                    arm_row,
                    policy=policy,
                    score=float(arm_row["_pred"]),
                    strategy=f"autoresearch_v2:{arm_name}",
                    split="arm",
                ),
                candidate_from_row(
                    baseline_row,
                    policy=policy,
                    score=float(baseline_row["_pred"]),
                    strategy=f"autoresearch_v2:{baseline_name}",
                    split="baseline",
                ),
            )
        )
    candidates.sort(
        key=lambda item: (
            item[1].session,
            min(item[1].decision_time_ns, item[2].decision_time_ns),
            item[0],
        )
    )
    cash = {"arm": cfg.starting_cash, "baseline": cfg.starting_cash}
    pending: dict[str, SerialCandidateV5 | None] = {"arm": None, "baseline": None}
    active_session: str | None = None
    session_start = dict(cash)
    session_pnl = {"arm": 0.0, "baseline": 0.0}
    admitted_arm: list[SerialCandidateV5] = []
    admitted_baseline: list[SerialCandidateV5] = []
    admitted_signals: list[str] = []
    skipped = {"cutoff": 0, "occupancy": 0, "risk": 0}

    def realize(name: str) -> None:
        item = pending[name]
        if item is not None:
            cash[name] += item.raw_label_pnl_after_campaign_fee
            session_pnl[name] += item.raw_label_pnl_after_campaign_fee
            pending[name] = None

    for signal_id, arm, baseline in candidates:
        if arm.session != baseline.session:
            raise PairedReplayMismatch("paired component rows cross sessions")
        if active_session != arm.session:
            for name in ("arm", "baseline"):
                realize(name)
            active_session = arm.session
            session_start = dict(cash)
            session_pnl = {"arm": 0.0, "baseline": 0.0}
        for name, item in (("arm", arm), ("baseline", baseline)):
            current = pending[name]
            if current is not None and current.label_realized_exit_time_ns <= item.decision_time_ns:
                realize(name)
        if not _before_cutoff(arm, cfg.no_new_entries_after_et) or not _before_cutoff(
            baseline, cfg.no_new_entries_after_et
        ):
            skipped["cutoff"] += 1
            continue
        if pending["arm"] is not None or pending["baseline"] is not None:
            skipped["occupancy"] += 1
            continue
        risk_ok = []
        for name, item in (("arm", arm), ("baseline", baseline)):
            required = item.entry_ask * cfg.contract_multiplier + cfg.affordability_reserve_per_trade
            limit = cfg.max_daily_loss_fraction_of_session_start_equity * session_start[name]
            risk_ok.append(
                required <= cash[name] + 1e-9
                and not (limit > 0.0 and session_pnl[name] <= -limit)
            )
        if not all(risk_ok):
            skipped["risk"] += 1
            continue
        admitted_signals.append(signal_id)
        admitted_arm.append(arm)
        admitted_baseline.append(baseline)
        pending["arm"] = arm
        pending["baseline"] = baseline
    for name in ("arm", "baseline"):
        realize(name)

    arm_trades, arm_state = simulate_serial_candidates_v5(admitted_arm, config=cfg)
    baseline_trades, baseline_state = simulate_serial_candidates_v5(admitted_baseline, config=cfg)
    if len(arm_trades) != len(admitted_signals) or len(baseline_trades) != len(admitted_signals):
        raise PairedReplayMismatch("simulator v5 failed to reproduce component admissions")
    arm_session: dict[str, float] = {}
    baseline_session: dict[str, float] = {}
    for trade in arm_trades:
        arm_session[trade.session] = arm_session.get(trade.session, 0.0) + trade.raw_label_pnl_after_campaign_fee
    for trade in baseline_trades:
        baseline_session[trade.session] = baseline_session.get(trade.session, 0.0) + trade.raw_label_pnl_after_campaign_fee
    deltas = {
        session: arm_session.get(session, 0.0) - baseline_session.get(session, 0.0)
        for session in sorted(set(arm_session) | set(baseline_session))
    }
    return {
        "arm": arm_name,
        "baseline": baseline_name,
        "policy": policy,
        "matched_trade_count": len(admitted_signals),
        "matched_entry_count": len(arm_trades) == len(baseline_trades),
        "skipped": skipped,
        "session_deltas": deltas,
        "arm_total_pnl": float(sum(arm_session.values())),
        "baseline_total_pnl": float(sum(baseline_session.values())),
        "arm_final_cash": float(arm_state.cash_by_account.get("arm", cfg.starting_cash)),
        "baseline_final_cash": float(
            baseline_state.cash_by_account.get("baseline", cfg.starting_cash)
        ),
        "arm_simulator_hash": arm_state.simulator_config_hash,
        "baseline_simulator_hash": baseline_state.simulator_config_hash,
        "admitted_signal_hash": __import__("hashlib").sha256(
            "\n".join(admitted_signals).encode()
        ).hexdigest(),
    }


def serial_summary(trades: Iterable[SerialReplayTradeV5], state: Any) -> dict[str, Any]:
    items = list(trades)
    return {
        "trade_count": len(items),
        "total_pnl": float(sum(item.raw_label_pnl_after_campaign_fee for item in items)),
        "final_cash": {key: float(value) for key, value in state.cash_by_account.items()},
        "skipped": dict(state.skipped),
        "trade_identity_hash": state.trade_identity_hash,
        "simulator_config_hash": state.simulator_config_hash,
        "simulator_semantics": dict(state.semantics),
    }
