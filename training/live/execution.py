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
    ) -> None:
        self.ib = ib
        self.dry_run = dry_run
        self.max_position_size = max(1, int(max_position_size))
        self.daily_loss_limit_pct = float(daily_loss_limit_pct)
        self.kill_switch_path = kill_switch_path
        self.audit_path = audit_path
        self.positions: dict[str, ExecutionState] = {}
        self._next_local_id = 1_000_000
        self._live_orders: dict[str, dict[str, Any]] = {}
        self.realized_pnl_pct: float = 0.0

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
            "payload": payload,
        }
        with open(self.audit_path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")

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
                metadata={"dry_run": True, "entry_order": intent.entry_order},
            )
            self.positions[position_id] = state
            self._audit("entry_dry_run", {"intent": asdict(intent), "state": asdict(state)})
            return state

        if not self.ib or not self.ib.isConnected():
            raise RuntimeError("IB is not connected for live order placement")

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

        self.ib.placeOrder(contract, parent)
        self.ib.placeOrder(contract, stop)
        self.ib.placeOrder(contract, take_profit)

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
            metadata={"entry_order": intent.entry_order},
        )
        self.positions[position_id] = state
        self._live_orders[position_id] = {
            "contract": contract,
            "parent": parent,
            "stop": stop,
            "take_profit": take_profit,
        }
        self._audit("entry_live", {"intent": asdict(intent), "state": asdict(state)})
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
            self._audit("risk_update_dry_run", {"update": asdict(update), "state": asdict(state)})
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
        self._audit("risk_update_live", {"update": asdict(update), "state": asdict(state)})
        return True

    def flatten_position(self, position_id: str, reason: str = "model_exit") -> bool:
        state = self.positions.get(position_id)
        if state is None or state.status != "OPEN":
            return False
        state.status = "CLOSED"
        state.updated_at = self._ts()
        state.notes.append(reason)

        if self.dry_run:
            self._audit("flatten_dry_run", {"position_id": position_id, "reason": reason})
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
        self.ib.placeOrder(contract, MarketOrder("SELL", state.qty))
        self._audit("flatten_live", {"position_id": position_id, "reason": reason})
        return True

