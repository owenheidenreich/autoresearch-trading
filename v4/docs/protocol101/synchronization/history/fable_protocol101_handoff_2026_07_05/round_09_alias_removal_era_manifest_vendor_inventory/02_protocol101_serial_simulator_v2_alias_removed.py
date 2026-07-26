"""Canonical Protocol101 serial replay simulator.

This module is intentionally small and model-agnostic. It defines the account
state semantics shared by fair-contract folds going forward. Archived March
diagnostics remain frozen under their original v1 replay artifacts; this v2
simulator is for future fold/backfill work and scripts that explicitly record
``simulator_version``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable
from zoneinfo import ZoneInfo


PROTOCOL101_SERIAL_SIMULATOR_VERSION = "protocol101_serial_simulator_v2"
DAILY_LOSS_BASIS = "raw_realized_net_pnl"
CASH_BASIS = "raw_realized_net_pnl"
STRESS_APPLICATION = "metrics_only"
EXIT_TIME_SEMANTICS = "synthetic_exit_at_entry_plus_cooldown"
COOLDOWN_ANCHOR = "entry"
NO_NEW_ENTRIES_AFTER_ET = "15:30"
FORCED_FLAT_BEFORE_ET = "15:55"
FEE_MODEL = "none_in_state"
ACCOUNT_CONTINUITY = "cash_compounds_across_sessions_within_split"


@dataclass(frozen=True)
class SerialSimulatorConfig:
    """Versioned one-account serial simulator settings."""

    starting_cash: float = 10_000.0
    contract_multiplier: float = 100.0
    max_trades_per_session: int = 0
    max_daily_loss: float = 0.0
    stress_per_trade: float = 0.0
    enforce_affordability: bool = True
    no_new_entries_after_et: str = NO_NEW_ENTRIES_AFTER_ET
    forced_flat_before_et: str = FORCED_FLAT_BEFORE_ET
    exit_time_semantics: str = EXIT_TIME_SEMANTICS
    cooldown_anchor: str = COOLDOWN_ANCHOR
    fee_model: str = FEE_MODEL
    account_continuity: str = ACCOUNT_CONTINUITY
    require_cooldown_equals_max_hold_when_present: bool = True
    simulator_version: str = PROTOCOL101_SERIAL_SIMULATOR_VERSION
    daily_loss_basis: str = DAILY_LOSS_BASIS
    cash_basis: str = CASH_BASIS
    stress_application: str = STRESS_APPLICATION

    def effective_simulator_version(self) -> str:
        if self.require_cooldown_equals_max_hold_when_present:
            return str(self.simulator_version)
        return f"{self.simulator_version}_cooldown_hold_guard_disabled"

    def semantics(self) -> dict[str, Any]:
        return {
            "simulator_version": self.effective_simulator_version(),
            "daily_loss_basis": self.daily_loss_basis,
            "cash_basis": self.cash_basis,
            "stress_application": self.stress_application,
            "starting_cash": float(self.starting_cash),
            "contract_multiplier": float(self.contract_multiplier),
            "max_trades_per_session": int(self.max_trades_per_session),
            "max_daily_loss": float(self.max_daily_loss),
            "stress_per_trade": float(self.stress_per_trade),
            "stress_per_trade_dollars": float(self.stress_per_trade),
            "enforce_affordability": bool(self.enforce_affordability),
            "no_new_entries_after": self.no_new_entries_after_et,
            "forced_flat_before": self.forced_flat_before_et,
            "exit_time_semantics": self.exit_time_semantics,
            "cooldown_anchor": self.cooldown_anchor,
            "fee_model": self.fee_model,
            "account_continuity": self.account_continuity,
            "require_cooldown_equals_max_hold_when_present": bool(
                self.require_cooldown_equals_max_hold_when_present
            ),
        }


@dataclass(frozen=True)
class SerialCandidate:
    """One selected entry intent before serial account constraints."""

    split: str
    session: str
    decision_time: datetime
    contract_id: str
    right: str
    offset: float
    entry_ask: float
    score: float
    raw_label_pnl: float
    cooldown_minutes: float
    max_hold_minutes: float | None = None
    feature_hash: str = ""
    source_quote_time: str = ""
    source_context_time: str = ""
    strategy: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SerialReplayTrade:
    """One accepted serial replay trade under canonical state semantics."""

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
    forced_flat_before: str
    fee_model: str
    account_continuity: str
    stress_per_trade_dollars: float
    simulator_config_hash: str
    candidate_stream_hash: str
    candidate_payload_hash: str


@dataclass(frozen=True)
class SerialReplayState:
    """State and skipped counters after canonical serial replay."""

    skipped: dict[str, int]
    cash_by_account: dict[str, float]
    equity_by_account: dict[str, list[float]]
    realized_raw_pnl_by_split_session: dict[str, float]
    semantics: dict[str, Any]
    simulator_config_hash: str
    candidate_stream_hash: str
    candidate_payload_hash: str


def _account_key(candidate: SerialCandidate) -> str:
    return str(candidate.split)


def _session_key(candidate: SerialCandidate) -> tuple[str, str]:
    return (str(candidate.split), str(candidate.session))


def _json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _candidate_stream_hash(candidates: Iterable[SerialCandidate]) -> str:
    keys = [
        {
            "split": str(candidate.split),
            "session": str(candidate.session),
            "decision_time": candidate.decision_time.isoformat(),
            "contract_id": str(candidate.contract_id),
            "right": str(candidate.right),
            "offset": float(candidate.offset),
        }
        for candidate in candidates
    ]
    return _json_hash(keys)


def _candidate_payload_hash(candidates: Iterable[SerialCandidate]) -> str:
    payload = [
        {
            "split": str(candidate.split),
            "session": str(candidate.session),
            "decision_time": candidate.decision_time.isoformat(),
            "contract_id": str(candidate.contract_id),
            "right": str(candidate.right),
            "offset": float(candidate.offset),
            "entry_ask": float(candidate.entry_ask),
            "raw_label_pnl": float(candidate.raw_label_pnl),
            "cooldown_minutes": float(candidate.cooldown_minutes),
            "max_hold_minutes": (
                None if candidate.max_hold_minutes is None else float(candidate.max_hold_minutes)
            ),
        }
        for candidate in candidates
    ]
    return _json_hash(payload)


def _et_minutes(value: str) -> int | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    hour_text, minute_text = raw.split(":", 1)
    return int(hour_text) * 60 + int(minute_text[:2])


def _is_after_entry_cutoff(decision_time: datetime, cutoff: str) -> bool:
    cutoff_minutes = _et_minutes(cutoff)
    if cutoff_minutes is None:
        return False
    local = decision_time.astimezone(ZoneInfo("America/New_York"))
    decision_minutes = local.hour * 60 + local.minute
    return decision_minutes > cutoff_minutes


def _is_tz_naive(decision_time: datetime) -> bool:
    return decision_time.tzinfo is None or decision_time.utcoffset() is None


def _local_forced_flat_time(decision_time: datetime, forced_flat_before: str) -> datetime:
    forced_minutes = _et_minutes(forced_flat_before)
    if forced_minutes is None:
        return decision_time + timedelta(minutes=10_000)
    local = decision_time.astimezone(ZoneInfo("America/New_York"))
    return local.replace(
        hour=forced_minutes // 60,
        minute=forced_minutes % 60,
        second=0,
        microsecond=0,
    )


def _capped_exit_time(
    decision_time: datetime,
    *,
    cooldown_minutes: float,
    forced_flat_before: str,
) -> datetime:
    synthetic_exit = decision_time + timedelta(minutes=float(cooldown_minutes))
    forced_flat = _local_forced_flat_time(decision_time, forced_flat_before)
    if synthetic_exit.astimezone(ZoneInfo("America/New_York")) > forced_flat:
        return forced_flat
    return synthetic_exit


def _cooldown_hold_mismatch(candidate: SerialCandidate, config: SerialSimulatorConfig) -> bool:
    if not config.require_cooldown_equals_max_hold_when_present:
        return False
    if candidate.max_hold_minutes is None:
        return False
    return abs(float(candidate.cooldown_minutes) - float(candidate.max_hold_minutes)) > 1e-9


def _realize_pending(
    *,
    pending: SerialReplayTrade | None,
    config: SerialSimulatorConfig,
    account: str,
    session_key: tuple[str, str],
    cash_by_account: dict[str, float],
    equity_by_account: dict[str, list[float]],
    realized_raw_pnl_by_session: dict[tuple[str, str], float],
) -> None:
    if pending is None:
        return
    raw_pnl = float(pending.raw_label_pnl)
    realized_raw_pnl_by_session[session_key] = (
        realized_raw_pnl_by_session.get(session_key, 0.0) + raw_pnl
    )
    cash = cash_by_account.setdefault(account, float(config.starting_cash))
    cash += raw_pnl
    cash_by_account[account] = cash
    equity_by_account.setdefault(account, [float(config.starting_cash)]).append(cash)


def simulate_serial_candidates(
    candidates: Iterable[SerialCandidate],
    *,
    config: SerialSimulatorConfig | None = None,
) -> tuple[list[SerialReplayTrade], SerialReplayState]:
    """Replay selected candidates through one-account serial constraints.

    v2 semantics:
    - daily-loss guards use raw realized net PnL, matching live-observable state;
    - account cash/affordability state uses raw realized net PnL;
    - stress is a metrics/reporting haircut only and never drives state.
    """

    cfg = config or SerialSimulatorConfig()
    ordered = sorted(
        candidates,
        key=lambda item: (
            str(item.split),
            str(item.session),
            item.decision_time,
            str(item.contract_id),
        ),
    )
    stream_hash = _candidate_stream_hash(ordered)
    payload_hash = _candidate_payload_hash(ordered)
    semantics = dict(cfg.semantics())
    config_hash = _json_hash(semantics)
    semantics["simulator_config_hash"] = config_hash
    semantics["candidate_stream_hash"] = stream_hash
    semantics["candidate_payload_hash"] = payload_hash
    skipped = {
        "tz_naive_decision_time": 0,
        "after_entry_cutoff": 0,
        "cooldown_hold_mismatch": 0,
        "overlap": 0,
        "unaffordable": 0,
        "nonpositive_ask": 0,
        "session_trade_cap": 0,
        "daily_loss_stop": 0,
    }
    trades: list[SerialReplayTrade] = []
    cash_by_account: dict[str, float] = {}
    equity_by_account: dict[str, list[float]] = {}
    pending_by_session: dict[tuple[str, str], SerialReplayTrade | None] = {}
    trades_by_session: dict[tuple[str, str], int] = {}
    realized_raw_pnl_by_session: dict[tuple[str, str], float] = {}

    for candidate in ordered:
        account = _account_key(candidate)
        session_key = _session_key(candidate)
        cash = cash_by_account.setdefault(account, float(cfg.starting_cash))
        equity_by_account.setdefault(account, [float(cfg.starting_cash)])
        if _is_tz_naive(candidate.decision_time):
            skipped["tz_naive_decision_time"] += 1
            continue
        if _is_after_entry_cutoff(candidate.decision_time, cfg.no_new_entries_after_et):
            skipped["after_entry_cutoff"] += 1
            continue
        if _cooldown_hold_mismatch(candidate, cfg):
            skipped["cooldown_hold_mismatch"] += 1
            continue
        pending = pending_by_session.get(session_key)
        if pending is not None:
            pending_exit = datetime.fromisoformat(pending.synthetic_exit_time)
            if candidate.decision_time >= pending_exit:
                _realize_pending(
                    pending=pending,
                    config=cfg,
                    account=account,
                    session_key=session_key,
                    cash_by_account=cash_by_account,
                    equity_by_account=equity_by_account,
                    realized_raw_pnl_by_session=realized_raw_pnl_by_session,
                )
                pending = None
                pending_by_session[session_key] = None
                cash = cash_by_account.setdefault(account, float(cfg.starting_cash))
            else:
                skipped["overlap"] += 1
                continue
        if (
            cfg.max_trades_per_session > 0
            and trades_by_session.get(session_key, 0) >= int(cfg.max_trades_per_session)
        ):
            skipped["session_trade_cap"] += 1
            continue
        if (
            cfg.max_daily_loss > 0.0
            and realized_raw_pnl_by_session.get(session_key, 0.0) <= -float(cfg.max_daily_loss)
        ):
            skipped["daily_loss_stop"] += 1
            continue
        if candidate.entry_ask <= 0.0:
            skipped["nonpositive_ask"] += 1
            continue
        premium = float(candidate.entry_ask) * float(cfg.contract_multiplier)
        if cfg.enforce_affordability and premium > cash + 1e-9:
            skipped["unaffordable"] += 1
            continue
        raw_pnl = float(candidate.raw_label_pnl)
        stressed_pnl = raw_pnl - float(cfg.stress_per_trade)
        exit_time = _capped_exit_time(
            candidate.decision_time,
            cooldown_minutes=float(candidate.cooldown_minutes),
            forced_flat_before=cfg.forced_flat_before_et,
        )
        trade = SerialReplayTrade(
            split=str(candidate.split),
            session=str(candidate.session),
            decision_time=candidate.decision_time.isoformat(),
            synthetic_exit_time=exit_time.astimezone(timezone.utc).isoformat(),
            contract_id=str(candidate.contract_id),
            right=str(candidate.right),
            offset=float(candidate.offset),
            entry_ask=float(candidate.entry_ask),
            premium_at_risk=float(premium),
            score=float(candidate.score),
            raw_label_pnl=raw_pnl,
            stressed_pnl=stressed_pnl,
            cash_before=float(cash),
            cash_after=float(cash + raw_pnl),
            feature_hash=str(candidate.feature_hash),
            source_quote_time=str(candidate.source_quote_time),
            source_context_time=str(candidate.source_context_time),
            realized_pnl_for_daily_loss=raw_pnl,
            cash_pnl_for_account_state=raw_pnl,
            simulator_version=cfg.effective_simulator_version(),
            daily_loss_basis=cfg.daily_loss_basis,
            cash_basis=cfg.cash_basis,
            stress_application=cfg.stress_application,
            exit_time_semantics=cfg.exit_time_semantics,
            cooldown_anchor=cfg.cooldown_anchor,
            no_new_entries_after=cfg.no_new_entries_after_et,
            forced_flat_before=cfg.forced_flat_before_et,
            fee_model=cfg.fee_model,
            account_continuity=cfg.account_continuity,
            stress_per_trade_dollars=float(cfg.stress_per_trade),
            simulator_config_hash=config_hash,
            candidate_stream_hash=stream_hash,
            candidate_payload_hash=payload_hash,
        )
        trades.append(trade)
        trades_by_session[session_key] = trades_by_session.get(session_key, 0) + 1
        pending_by_session[session_key] = trade

    for session_key, pending in list(pending_by_session.items()):
        if pending is None:
            continue
        split, _session = session_key
        _realize_pending(
            pending=pending,
            config=cfg,
            account=split,
            session_key=session_key,
            cash_by_account=cash_by_account,
            equity_by_account=equity_by_account,
            realized_raw_pnl_by_session=realized_raw_pnl_by_session,
        )
        pending_by_session[session_key] = None

    realized_out = {
        f"{split}:{session}": float(value)
        for (split, session), value in sorted(realized_raw_pnl_by_session.items())
    }
    return trades, SerialReplayState(
        skipped=skipped,
        cash_by_account={key: float(value) for key, value in cash_by_account.items()},
        equity_by_account={
            key: [float(item) for item in values]
            for key, values in equity_by_account.items()
        },
        realized_raw_pnl_by_split_session=realized_out,
        semantics=semantics,
        simulator_config_hash=config_hash,
        candidate_stream_hash=stream_hash,
        candidate_payload_hash=payload_hash,
    )
