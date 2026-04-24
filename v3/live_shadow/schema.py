from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class IBKRContractSpec:
    symbol: str = "SPX"
    sec_type: str = "OPT"
    exchange: str = "SMART"
    currency: str = "USD"
    trading_class: str = "SPXW"
    multiplier: str = "100"
    last_trade_date_or_contract_month: str = ""
    strike: float = 0.0
    right: str = "C"


@dataclass(frozen=True)
class QuoteSnapshot:
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    mid: float | None = None
    timestamp_ms: int | None = None
    stale: bool = False


@dataclass(frozen=True)
class GreekSnapshot:
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    implied_vol: float | None = None
    underlying_price: float | None = None
    source_tick: str | None = None


@dataclass(frozen=True)
class ContractCandidateSnapshot:
    action_id: int
    slot: int
    side: str
    strike: float
    risk_band: str
    moneyness: str
    premium: float | None
    spread_fraction: float | None
    model_score: float | None = None
    win_prob: float | None = None
    stopout_prob: float | None = None
    quote: QuoteSnapshot | None = None
    greeks: GreekSnapshot | None = None
    contract: IBKRContractSpec | None = None


@dataclass(frozen=True)
class DecisionSnapshot:
    session_id: str
    timestamp_ms: int
    day: str
    completed_bar_index: int
    model_artifact: str
    dataset_fingerprint: str | None
    feature_schema: list[str]
    features: dict[str, float]
    candidates: list[ContractCandidateSnapshot]
    selected_action_id: int
    selected_contract: IBKRContractSpec | None
    no_order_reason: str
    scores: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

