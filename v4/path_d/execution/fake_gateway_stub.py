"""Fake-only gateway seam. The real IBKR adapter is explicitly deferred to step 7."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FakeGatewayStub:
    connected: bool = True
    submitted_order_ids: list[str] = field(default_factory=list)
    cancelled_order_ids: list[str] = field(default_factory=list)

    def submit(self, order_id: str) -> None:
        if not self.connected:
            raise ConnectionError("fake gateway disconnected")
        self.submitted_order_ids.append(order_id)

    def cancel(self, order_id: str) -> None:
        if not self.connected:
            raise ConnectionError("fake gateway disconnected")
        self.cancelled_order_ids.append(order_id)

