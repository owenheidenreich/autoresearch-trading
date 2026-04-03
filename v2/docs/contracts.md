# v2 Core Contract: TradeIntent

## Purpose

TradeIntent is the single executable object at the center of the v2 system.
Replay scores it. Live executes it. Training learns to emit it.

In v1, the trade representation was split across multiple incompatible shapes:
- Training labels: separate arrays (y_pred, y_v18) with proxy signals
- Replay: reconstructed trades from model outputs + hardcoded rules
- Live: DecisionIntent dataclass -> ExecutionState dataclass

v2 collapses all of these into one frozen object.

---

## TradeIntent Definition

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class TradeIntent:
    """The atomic unit of the v2 trading system."""

    # --- Decision ---
    trade: bool                          # False = no-trade (most bars)

    # --- Contract identity (None when trade=False) ---
    expiry: str | None                   # "20260403" YYYYMMDD (0DTE)
    strike: float | None                 # e.g. 5200.0 (5-point grid)
    right: str | None                    # "C" or "P"

    # --- Sizing ---
    qty: int                             # 0 when trade=False, else >= 1

    # --- Entry ---
    entry_ref_price: float | None        # option mid-price at decision bar
    order_style: str                     # "MKT" | "LMT" | "ADAPTIVE"
    limit_price: float | None            # None for MKT orders
    tif: str                             # "DAY" | "IOC"

    # --- Risk management ---
    stop_price: float                    # hard stop loss (option premium)
    take_profit_price: float             # profit target (option premium)
    max_hold_bars: int                   # max duration in bars (1-390)
    exit_policy: str                     # "STOP_TP_TIME" | "TRAILING" | "MODEL_EXIT"

    # --- Metadata ---
    confidence: float                    # model confidence [0, 1]
    reason_codes: tuple[str, ...]        # immutable decision trail
    bar_index: int                       # bar within trading day (0-389)
    timestamp: str                       # ISO 8601 decision timestamp
    intent_id: str                       # UUID for tracking

    # --- Quote provenance (for audit) ---
    bid_at_decision: float | None        # option bid when decision was made
    ask_at_decision: float | None        # option ask when decision was made
    underlying_price: float | None       # SPX spot at decision time

    # --- Versioning ---
    policy_version: str                  # e.g. "v2.0.1"
```

---

## Validation Rules

When `trade=True`, all of the following must hold:
- `expiry` is non-None and matches YYYYMMDD format
- `strike` is non-None, positive, on the 5-point grid
- `right` is "C" or "P"
- `qty >= 1`
- `entry_ref_price > 0`
- `stop_price > 0` and `stop_price < entry_ref_price` (for long options, stop is below entry)
- `take_profit_price > entry_ref_price` (target is above entry for long options)
- `max_hold_bars` is in range [1, 390]
- `exit_policy` is one of: "STOP_TP_TIME", "TRAILING", "MODEL_EXIT"
- `confidence` is in [0.0, 1.0]
- `bar_index` is in [0, 389]
- `intent_id` is a valid UUID string

When `trade=False`:
- `qty == 0`
- Contract fields (expiry, strike, right) may be None
- Risk fields default to safe values (stop_price=0, take_profit_price=0, max_hold_bars=0)

---

## RiskAdjustment (Separate Object)

TradeIntent is frozen. Mid-trade risk updates use a separate object:

```python
@dataclass(frozen=True)
class RiskAdjustment:
    """Adjusts stop/TP on an open position. Immutable."""
    intent_id: str                       # references the original TradeIntent
    new_stop_price: float | None         # None = no change
    new_take_profit_price: float | None  # None = no change
    reason_codes: tuple[str, ...]
    adjustment_id: str                   # UUID
    timestamp: str                       # ISO 8601
```

---

## Flow Through the System

```
Training:
  model.forward(features) -> raw outputs
  -> construct TradeIntent (or trade=False)
  -> loss computed against oracle TradeIntents (from labels.py)

Replay:
  load model -> for each bar, emit TradeIntent
  -> simulator.py executes against historical option prices
  -> metrics.py scores the sequence of trades

Live:
  market.py provides features -> model.forward()
  -> decision.py constructs TradeIntent
  -> execution.py resolves to IBKR contract and places orders
```

---

## Serialization

**For audit logs (JSON):**
```python
import dataclasses, json
json.dumps(dataclasses.asdict(intent), sort_keys=True)
```

Fields are deterministically ordered (sort_keys=True) so that identical intents
produce identical JSON strings. This matters for replay checksums.

**For replay data (torch):**
```python
torch.save(dataclasses.asdict(intent), path)
```

**reason_codes is a tuple** (not list) to enforce immutability and allow hashing.

---

## v1 DecisionIntent Delta

| v1 DecisionIntent | v2 TradeIntent | Change |
|-------------------|----------------|--------|
| action: int (0-15) | trade: bool + strike/right | Explicit contract identity replaces action enum |
| contract: Any (ib_insync.Contract) | expiry/strike/right: str/float/str | Broker-agnostic, serializable |
| qty: int | qty: int | Same |
| entry_order: str | order_style: str | Renamed for clarity |
| stop_price: float | stop_price: float | Same |
| take_profit_price: float | take_profit_price: float | Same |
| confidence: float | confidence: float | Same |
| reason_codes: list[str] | reason_codes: tuple[str, ...] | Immutable |
| entry_limit_price: float | limit_price: float | Renamed |
| reference_price: float | entry_ref_price: float | Renamed |
| decision_id: str | (dropped) | Replaced by intent_id |
| intent_id: str | intent_id: str | Same |
| metadata: dict | (dropped) | Structured fields replace opaque dict |
| (missing) | tif: str | Added: time-in-force |
| (missing) | max_hold_bars: int | Added: max hold duration |
| (missing) | exit_policy: str | Added: exit strategy |
| (missing) | bid_at_decision: float | Added: quote provenance |
| (missing) | ask_at_decision: float | Added: quote provenance |
| (missing) | underlying_price: float | Added: SPX spot reference |
| (missing) | policy_version: str | Added: versioning |
| (missing) | bar_index: int | Added: time context |
| (missing) | timestamp: str | Added: ISO 8601 |

---

## Broker Resolution

The conversion from `expiry/strike/right` to a broker-executable contract is a
one-way function that lives in `v2/live/execution.py`:

```python
def resolve_to_ibkr(intent: TradeIntent) -> ib_insync.Option:
    return ib_insync.Option(
        symbol="SPX",
        lastTradeDateOrContractMonth=intent.expiry,
        strike=intent.strike,
        right=intent.right,
        exchange="SMART",
        multiplier="100",
    )
```

This function is never called in replay or training. The TradeIntent itself is
broker-agnostic and fully serializable without any ib_insync dependency.
