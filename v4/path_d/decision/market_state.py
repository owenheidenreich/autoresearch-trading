"""Received-clock canonical market state; contains no vendor or broker clients."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from v4.path_d.contracts import CanonicalMarketEventV1
from v4.path_d.features.source_neutral import IndexObservation


@dataclass
class CanonicalMarketState:
    option_quotes: dict[str, CanonicalMarketEventV1] = field(default_factory=dict)
    spx_observations: list[IndexObservation] = field(default_factory=list)
    last_received_timestamp_utc: str | None = None

    def ingest(self, event: CanonicalMarketEventV1) -> None:
        event_clock = _timestamp(event.received_timestamp_utc)
        if self.last_received_timestamp_utc is not None and event_clock < _timestamp(self.last_received_timestamp_utc):
            raise ValueError("canonical events must be replayed in received_timestamp_utc order")
        self.last_received_timestamp_utc = event.received_timestamp_utc
        if event.event_type in {"OPTION_QUOTE", "BROKER_QUOTE"}:
            assert event.osi_symbol is not None
            self.option_quotes[event.osi_symbol] = event
        elif event.event_type == "SPX_INDEX":
            assert event.index_price_micros is not None
            self.spx_observations.append(
                IndexObservation(
                    received_timestamp_utc=event.received_timestamp_utc,
                    close=event.index_price_micros / 1_000_000.0,
                    volume=float(event.volume or 0),
                )
            )

    def option_quote(self, osi_symbol: str) -> CanonicalMarketEventV1:
        try:
            return self.option_quotes[osi_symbol]
        except KeyError as exc:
            raise ValueError(f"no canonical quote for {osi_symbol!r}") from exc

    @property
    def spx_watermark(self) -> str:
        if not self.spx_observations:
            raise ValueError("SPX context is not available")
        return self.spx_observations[-1].received_timestamp_utc


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)

