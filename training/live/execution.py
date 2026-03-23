from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import asdict
from typing import Any

from ib_insync import IB, Contract, MarketOrder, Order

from training.live.contracts import DecisionIntent, ExecutionState, RiskUpdateIntent


class OCOExecutionEngine:
    """Execution layer that mirrors decision intents and enforces hard invariants."""

    def __init__(
        self,
        ib: IB | None = None,
        dry_run: bool = True,
        max_position_size: int = 1,
        daily_loss_limit_pct: float = 0.05,
        kill_switch_path: str | None = None,
        audit_path: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self.ib = ib
        self.dry_run = dry_run
        self.max_position_size = max(1, int(max_position_size))
        self.daily_loss_limit_pct = float(daily_loss_limit_pct)
        self.kill_switch_path = kill_switch_path
        self.audit_path = audit_path
        self.session_id = session_id
        self.positions: dict[str, ExecutionState] = {}
        self._next_local_id = 1_000_000
        self._live_orders: dict[str, dict[str, Any]] = {}
        self._trade_by_order_id: dict[int, dict[str, Any]] = {}
        self._ib_callbacks_installed = False
        self.realized_pnl_pct: float = 0.0
        self.starting_capital: float = 10_000.0  # Must match STARTING_CAPITAL in train.py
        self.session_pnl_dollars: float = 0.0
        self.trades_closed: int = 0

    def _ts(self) -> str:
        return dt.datetime.utcnow().isoformat()

    def _check_kill_switch(self) -> bool:
        if not self.kill_switch_path:
            return False
        if not os.path.exists(self.kill_switch_path):
            return False
        try:
            content = open(self.kill_switch_path).read().strip().lower()
        except OSError:
            return False
        return content in {"1", "on", "true", "stop", "kill"}

    def _audit(self, event: str, payload: dict[str, Any]) -> None:
        if not self.audit_path:
            return
        os.makedirs(os.path.dirname(self.audit_path), exist_ok=True)
        row = {
            "ts": self._ts(),
            "event": event,
            "session_id": self.session_id,
            "payload": payload,
        }
        with open(self.audit_path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def _contract_payload(self, contract: Any) -> dict[str, Any]:
        return {
            "symbol": getattr(contract, "symbol", None),
            "secType": getattr(contract, "secType", None),
            "exchange": getattr(contract, "exchange", None),
            "currency": getattr(contract, "currency", None),
            "tradingClass": getattr(contract, "tradingClass", None),
            "lastTradeDateOrContractMonth": getattr(contract, "lastTradeDateOrContractMonth", None),
            "strike": getattr(contract, "strike", None),
            "right": getattr(contract, "right", None),
            "conId": getattr(contract, "conId", None),
        }

    def _order_payload(self, order: Order) -> dict[str, Any]:
        return {
            "orderId": int(getattr(order, "orderId", 0)),
            "parentId": int(getattr(order, "parentId", 0)),
            "action": getattr(order, "action", None),
            "orderType": getattr(order, "orderType", None),
            "totalQuantity": int(getattr(order, "totalQuantity", 0)),
            "lmtPrice": getattr(order, "lmtPrice", None),
            "auxPrice": getattr(order, "auxPrice", None),
            "ocaGroup": getattr(order, "ocaGroup", None),
        }

    def _ensure_ib_callbacks(self) -> None:
        if self.dry_run or not self.ib or self._ib_callbacks_installed:
            return

        def _on_error(req_id: int, error_code: int, error_string: str, contract: Any) -> None:
            self._audit(
                "ib_error_event",
                {
                    "req_id": int(req_id),
                    "error_code": int(error_code),
                    "error_string": str(error_string),
                    "contract": self._contract_payload(contract),
                },
            )

        self.ib.errorEvent += _on_error
        self._ib_callbacks_installed = True

    def _track_trade(self, *, position_id: str, role: str, intent: DecisionIntent, trade: Any) -> None:
        order = getattr(trade, "order", None)
        if order is None:
            return
        order_id = int(getattr(order, "orderId", 0))
        self._trade_by_order_id[order_id] = {
            "position_id": position_id,
            "role": role,
            "intent_id": intent.intent_id,
            "decision_id": intent.decision_id,
        }

        def _status_handler(trade_obj: Any) -> None:
            state = self.positions.get(position_id)
            status_obj = getattr(trade_obj, "orderStatus", None)
            status = str(getattr(status_obj, "status", "")) if status_obj is not None else ""
            avg_fill = getattr(status_obj, "avgFillPrice", None) if status_obj is not None else None
            filled = getattr(status_obj, "filled", None) if status_obj is not None else None
            remaining = getattr(status_obj, "remaining", None) if status_obj is not None else None
            perm_id = None
            if status_obj is not None:
                try:
                    perm_id = int(getattr(status_obj, "permId", 0) or 0) or None
                except Exception:
                    perm_id = None

            payload = {
                "position_id": position_id,
                "order_role": role,
                "intent_id": intent.intent_id,
                "decision_id": intent.decision_id,
                "order_id": order_id,
                "status": status,
                "filled": filled,
                "remaining": remaining,
                "avg_fill_price": avg_fill,
                "perm_id": perm_id,
            }
            self._audit("ib_order_status", payload)
            if state is None:
                return
            state.updated_at = self._ts()
            if role == "parent" and status.lower() in {"filled", "partiallyfilled"}:
                state.fill_status = status.upper()
                try:
                    fill_px = float(avg_fill) if avg_fill is not None else None
                except Exception:
                    fill_px = None
                if fill_px is not None and fill_px > 0:
                    state.fill_price = fill_px
                    state.fill_time = self._ts()
                    if state.entry_price_reference and state.entry_price_reference > 0:
                        state.slippage_bps = (
                            (fill_px - state.entry_price_reference)
                            / state.entry_price_reference
                            * 10000.0
                        )
                if perm_id is not None:
                    state.ib_perm_id_entry = perm_id
            # Stop or TP filled — position is closed, update P&L
            if role in ("stop", "take_profit") and status.lower() == "filled":
                try:
                    exit_px = float(avg_fill) if avg_fill is not None else None
                except Exception:
                    exit_px = None
                if state.status == "OPEN":
                    state.status = "CLOSED"
                    state.notes.append(f"{role}_filled")
                    self._update_realized_pnl(state, exit_px, reason=f"{role}_filled")

        def _fill_handler(trade_obj: Any, fill_obj: Any) -> None:
            state = self.positions.get(position_id)
            execution = getattr(fill_obj, "execution", None)
            exec_id = getattr(execution, "execId", None) if execution is not None else None
            price = getattr(execution, "price", None) if execution is not None else None
            shares = getattr(execution, "shares", None) if execution is not None else None
            exec_time = getattr(execution, "time", None) if execution is not None else None
            self._audit(
                "ib_exec_details",
                {
                    "position_id": position_id,
                    "order_role": role,
                    "intent_id": intent.intent_id,
                    "decision_id": intent.decision_id,
                    "order_id": order_id,
                    "exec_id": exec_id,
                    "price": price,
                    "shares": shares,
                    "time": str(exec_time) if exec_time is not None else None,
                },
            )
            if state is not None and role == "parent":
                state.last_exec_id = str(exec_id) if exec_id else state.last_exec_id
                state.fill_status = "FILLED"
                try:
                    fill_px = float(price) if price is not None else None
                except Exception:
                    fill_px = None
                if fill_px is not None and fill_px > 0:
                    state.fill_price = fill_px
                    state.fill_time = self._ts()
                    if state.entry_price_reference and state.entry_price_reference > 0:
                        state.slippage_bps = (
                            (fill_px - state.entry_price_reference)
                            / state.entry_price_reference
                            * 10000.0
                        )

        def _cancel_handler(trade_obj: Any) -> None:
            self._audit(
                "ib_order_cancelled",
                {
                    "position_id": position_id,
                    "order_role": role,
                    "intent_id": intent.intent_id,
                    "decision_id": intent.decision_id,
                    "order_id": order_id,
                },
            )

        if hasattr(trade, "statusEvent"):
            trade.statusEvent += _status_handler
        if hasattr(trade, "fillEvent"):
            trade.fillEvent += _fill_handler
        if hasattr(trade, "cancelledEvent"):
            trade.cancelledEvent += _cancel_handler

    def _next_order_id(self) -> int:
        if self.ib and self.ib.isConnected() and not self.dry_run:
            return int(self.ib.client.getReqId())
        self._next_local_id += 1
        return self._next_local_id

    def _validate_new_entry(self, intent: DecisionIntent) -> None:
        if self._check_kill_switch():
            raise RuntimeError("Kill switch is active; entry blocked")
        if self.realized_pnl_pct <= -abs(self.daily_loss_limit_pct):
            raise RuntimeError("Daily loss limit reached; entry blocked")
        if intent.qty < 1 or intent.qty > self.max_position_size:
            raise ValueError(f"Invalid qty {intent.qty}; max_position_size={self.max_position_size}")
        if intent.stop_price <= 0 or intent.take_profit_price <= 0:
            raise ValueError("stop_price and take_profit_price must be > 0")

    def place_entry(self, intent: DecisionIntent) -> ExecutionState:
        self._validate_new_entry(intent)
        now = self._ts()
        parent_id = self._next_order_id()
        stop_id = self._next_order_id()
        tp_id = self._next_order_id()
        position_id = f"pos-{parent_id}"

        if self.dry_run:
            state = ExecutionState(
                position_id=position_id,
                contract=intent.contract,
                qty=intent.qty,
                status="OPEN",
                entry_order_id=parent_id,
                stop_order_id=stop_id,
                take_profit_order_id=tp_id,
                current_stop=float(intent.stop_price),
                current_take_profit=float(intent.take_profit_price),
                created_at=now,
                updated_at=now,
                entry_price_reference=intent.reference_price,
                fill_status="FILLED",
                fill_price=float(intent.reference_price) if intent.reference_price else None,
                fill_time=now,
                slippage_bps=0.0 if intent.reference_price else None,
                session_id=self.session_id,
                decision_id=intent.decision_id,
                intent_id=intent.intent_id,
                metadata={
                    "dry_run": True,
                    "entry_order": intent.entry_order,
                    "reason_codes": list(intent.reason_codes),
                    **(intent.metadata or {}),
                },
            )
            self.positions[position_id] = state
            self._audit(
                "entry_dry_run",
                {
                    "position_id": position_id,
                    "decision_id": intent.decision_id,
                    "intent_id": intent.intent_id,
                    "intent": asdict(intent),
                    "state": asdict(state),
                    "contract": self._contract_payload(intent.contract),
                },
            )
            return state

        if not self.ib or not self.ib.isConnected():
            raise RuntimeError("IB is not connected for live order placement")
        self._ensure_ib_callbacks()

        contract = intent.contract
        if not isinstance(contract, Contract):
            raise ValueError("DecisionIntent.contract must be an IB Contract in live mode")

        group = f"OCO-{parent_id}"
        parent = Order(
            orderId=parent_id,
            action="BUY",
            totalQuantity=intent.qty,
            orderType=intent.entry_order.upper(),
            transmit=False,
        )
        if parent.orderType == "LMT":
            if intent.entry_limit_price is None:
                raise ValueError("LMT entry requires entry_limit_price")
            parent.lmtPrice = float(intent.entry_limit_price)

        stop = Order(
            orderId=stop_id,
            parentId=parent_id,
            action="SELL",
            totalQuantity=intent.qty,
            orderType="STP",
            auxPrice=float(intent.stop_price),
            transmit=False,
            ocaGroup=group,
            ocaType=1,
        )
        take_profit = Order(
            orderId=tp_id,
            parentId=parent_id,
            action="SELL",
            totalQuantity=intent.qty,
            orderType="LMT",
            lmtPrice=float(intent.take_profit_price),
            transmit=True,
            ocaGroup=group,
            ocaType=1,
        )

        parent_trade = self.ib.placeOrder(contract, parent)
        stop_trade = self.ib.placeOrder(contract, stop)
        tp_trade = self.ib.placeOrder(contract, take_profit)

        state = ExecutionState(
            position_id=position_id,
            contract=contract,
            qty=intent.qty,
            status="OPEN",
            entry_order_id=parent_id,
            stop_order_id=stop_id,
            take_profit_order_id=tp_id,
            current_stop=float(intent.stop_price),
            current_take_profit=float(intent.take_profit_price),
            created_at=now,
            updated_at=now,
            entry_price_reference=intent.reference_price,
            session_id=self.session_id,
            decision_id=intent.decision_id,
            intent_id=intent.intent_id,
            metadata={
                "entry_order": intent.entry_order,
                "reason_codes": list(intent.reason_codes),
                **(intent.metadata or {}),
            },
        )
        self.positions[position_id] = state
        self._live_orders[position_id] = {
            "contract": contract,
            "parent": parent,
            "stop": stop,
            "take_profit": take_profit,
            "parent_trade": parent_trade,
            "stop_trade": stop_trade,
            "take_profit_trade": tp_trade,
        }
        self._track_trade(position_id=position_id, role="parent", intent=intent, trade=parent_trade)
        self._track_trade(position_id=position_id, role="stop", intent=intent, trade=stop_trade)
        self._track_trade(position_id=position_id, role="take_profit", intent=intent, trade=tp_trade)

        self._audit(
            "entry_live",
            {
                "position_id": position_id,
                "decision_id": intent.decision_id,
                "intent_id": intent.intent_id,
                "intent": asdict(intent),
                "state": asdict(state),
                "contract": self._contract_payload(contract),
                "orders": {
                    "parent": self._order_payload(parent),
                    "stop": self._order_payload(stop),
                    "take_profit": self._order_payload(take_profit),
                },
            },
        )
        return state

    def apply_risk_update(self, update: RiskUpdateIntent) -> bool:
        state = self.positions.get(update.position_id)
        if state is None or state.status != "OPEN":
            return False

        changed = False
        if update.new_stop_price is not None:
            if update.new_stop_price < state.current_stop:
                self._audit(
                    "risk_update_rejected",
                    {
                        "position_id": update.position_id,
                        "reason": "stop_downward",
                        "current_stop": state.current_stop,
                        "new_stop": update.new_stop_price,
                    },
                )
                return False
            if update.new_stop_price > state.current_stop:
                state.current_stop = float(update.new_stop_price)
                changed = True

        if update.new_take_profit_price is not None:
            if update.new_take_profit_price < state.current_take_profit:
                self._audit(
                    "risk_update_rejected",
                    {
                        "position_id": update.position_id,
                        "reason": "take_profit_downward",
                        "current_take_profit": state.current_take_profit,
                        "new_take_profit": update.new_take_profit_price,
                    },
                )
                return False
            if update.new_take_profit_price > state.current_take_profit:
                state.current_take_profit = float(update.new_take_profit_price)
                changed = True

        if not changed:
            return False

        state.updated_at = self._ts()
        if self.dry_run:
            self._audit(
                "risk_update_dry_run",
                {
                    "position_id": update.position_id,
                    "decision_id": update.decision_id,
                    "intent_id": update.intent_id,
                    "update": asdict(update),
                    "state": asdict(state),
                },
            )
            return True

        if not self.ib or not self.ib.isConnected():
            return False
        live = self._live_orders.get(update.position_id)
        if not live:
            return False

        stop: Order = live["stop"]
        tp: Order = live["take_profit"]
        stop.auxPrice = float(state.current_stop)
        tp.lmtPrice = float(state.current_take_profit)
        contract: Contract = live["contract"]
        self.ib.placeOrder(contract, stop)
        self.ib.placeOrder(contract, tp)
        self._audit(
            "risk_update_live",
            {
                "position_id": update.position_id,
                "decision_id": update.decision_id,
                "intent_id": update.intent_id,
                "update": asdict(update),
                "state": asdict(state),
                "orders": {
                    "stop": self._order_payload(stop),
                    "take_profit": self._order_payload(tp),
                },
            },
        )
        return True

    def _update_realized_pnl(self, state: ExecutionState, exit_price: float | None, reason: str) -> None:
        """Update session P&L tracking after a trade closes."""
        entry = state.fill_price or state.entry_price_reference
        if entry is None or entry <= 0:
            return
        if exit_price is None or exit_price <= 0:
            # No exit price available (e.g. stop filled but price unknown yet)
            # Use stop price as estimate for stop exits, entry for unknown
            if "stop" in reason.lower():
                exit_price = state.current_stop
            else:
                return
        trade_pnl_pct = (exit_price - entry) / entry
        trade_pnl_dollars = trade_pnl_pct * entry * state.qty * 100  # SPX multiplier
        self.session_pnl_dollars += trade_pnl_dollars
        self.realized_pnl_pct = self.session_pnl_dollars / self.starting_capital
        self.trades_closed += 1
        self._audit("pnl_update", {
            "position_id": state.position_id,
            "entry_price": entry,
            "exit_price": exit_price,
            "trade_pnl_pct": round(trade_pnl_pct, 4),
            "trade_pnl_dollars": round(trade_pnl_dollars, 2),
            "session_pnl_dollars": round(self.session_pnl_dollars, 2),
            "realized_pnl_pct": round(self.realized_pnl_pct, 4),
            "reason": reason,
        })
        # Hard kill: if session loss exceeds 10%, activate kill switch
        if self.realized_pnl_pct <= -0.10:
            if self.kill_switch_path:
                with open(self.kill_switch_path, "w") as f:
                    f.write("kill")
            self._audit("hard_kill_triggered", {
                "realized_pnl_pct": round(self.realized_pnl_pct, 4),
                "session_pnl_dollars": round(self.session_pnl_dollars, 2),
                "trades_closed": self.trades_closed,
            })

    def flatten_position(self, position_id: str, reason: str = "model_exit",
                         exit_price: float | None = None) -> bool:
        state = self.positions.get(position_id)
        if state is None or state.status != "OPEN":
            return False
        state.status = "CLOSED"
        state.updated_at = self._ts()
        state.notes.append(reason)

        # Update realized P&L for circuit breaker
        self._update_realized_pnl(state, exit_price, reason)

        if self.dry_run:
            self._audit(
                "flatten_dry_run",
                {
                    "position_id": position_id,
                    "reason": reason,
                    "decision_id": state.decision_id,
                    "intent_id": state.intent_id,
                    "realized_pnl_pct": round(self.realized_pnl_pct, 4),
                    "session_pnl_dollars": round(self.session_pnl_dollars, 2),
                },
            )
            return True

        if not self.ib or not self.ib.isConnected():
            return False
        live = self._live_orders.get(position_id)
        if not live:
            return False

        stop: Order = live["stop"]
        tp: Order = live["take_profit"]
        contract: Contract = live["contract"]
        self.ib.cancelOrder(stop)
        self.ib.cancelOrder(tp)
        flatten_order = MarketOrder("SELL", state.qty)
        flatten_trade = self.ib.placeOrder(contract, flatten_order)
        self._audit(
            "flatten_live",
            {
                "position_id": position_id,
                "reason": reason,
                "decision_id": state.decision_id,
                "intent_id": state.intent_id,
                "contract": self._contract_payload(contract),
                "flatten_order": self._order_payload(flatten_order),
                "flatten_order_id": int(getattr(flatten_order, "orderId", 0)),
                "flatten_trade_status": str(getattr(getattr(flatten_trade, "orderStatus", None), "status", "")),
                "realized_pnl_pct": round(self.realized_pnl_pct, 4),
                "session_pnl_dollars": round(self.session_pnl_dollars, 2),
            },
        )
        return True
