"""Canonical Protocol101 serial replay simulator.

This module is intentionally small and model-agnostic. It defines the account
state semantics shared by fair-contract folds going forward. Archived March
diagnostics remain frozen under their original v1 replay artifacts; this v2
simulator is for future fold/backfill work and scripts that explicitly record
``simulator_version``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable


PROTOCOL101_SERIAL_SIMULATOR_VERSION = "protocol101_serial_simulator_v2"
DAILY_LOSS_BASIS = "raw_realized_net_pnl"
CASH_BASIS = "raw_realized_net_pnl"
STRESS_APPLICATION = "metrics_only"


@dataclass(frozen=True)
class SerialSimulatorConfig:
    """Versioned one-account serial simulator settings."""

    starting_cash: float = 10_000.0
    contract_multiplier: float = 100.0
    max_trades_per_session: int = 0
    max_daily_loss: float = 0.0
    stress_per_trade: float = 0.0
    enforce_affordability: bool = True
    simulator_version: str = PROTOCOL101_SERIAL_SIMULATOR_VERSION
    daily_loss_basis: str = DAILY_LOSS_BASIS
    cash_basis: str = CASH_BASIS
    stress_application: str = STRESS_APPLICATION

    def semantics(self) -> dict[str, Any]:
        return {
            "simulator_version": self.simulator_version,
            "daily_loss_basis": self.daily_loss_basis,
            "cash_basis": self.cash_basis,
            "stress_application": self.stress_application,
            "starting_cash": float(self.starting_cash),
            "contract_multiplier": float(self.contract_multiplier),
            "max_trades_per_session": int(self.max_trades_per_session),
            "max_daily_loss": float(self.max_daily_loss),
            "stress_per_trade": float(self.stress_per_trade),
            "enforce_affordability": bool(self.enforce_affordability),
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


@dataclass(frozen=True)
class SerialReplayState:
    """State and skipped counters after canonical serial replay."""

    skipped: dict[str, int]
    cash_by_account: dict[str, float]
    equity_by_account: dict[str, list[float]]
    realized_raw_pnl_by_split_session: dict[str, float]
    semantics: dict[str, Any]


def _account_key(candidate: SerialCandidate) -> str:
    return str(candidate.split)


def _session_key(candidate: SerialCandidate) -> tuple[str, str]:
    return (str(candidate.split), str(candidate.session))


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
    skipped = {
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
        exit_time = candidate.decision_time + timedelta(minutes=float(candidate.cooldown_minutes))
        trade = SerialReplayTrade(
            split=str(candidate.split),
            session=str(candidate.session),
            decision_time=candidate.decision_time.isoformat(),
            synthetic_exit_time=exit_time.isoformat(),
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
            simulator_version=cfg.simulator_version,
            daily_loss_basis=cfg.daily_loss_basis,
            cash_basis=cfg.cash_basis,
            stress_application=cfg.stress_application,
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
        semantics=cfg.semantics(),
    )

