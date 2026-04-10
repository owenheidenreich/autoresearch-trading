# Section 5: IBKR Paper Trading

## Scope
The live/paper trading subsystem: IBKR market data streaming, real-time feature computation, model inference, order execution, position management, safety controls, and the bridge from backtested model to real-time paper trading. All files in this section are currently stubs with detailed specifications -- implementation has not yet begun.

## Critical Files

| File | Role | Lines | Mutable? | Status |
|------|------|------:|----------|--------|
| `v2/live/service.py` | Main RTH trading loop (9:30-16:00 ET). Orchestrates: market data -> features -> model inference -> execution. Bar-by-bar loop, EOD flatten, kill switch. | 12 | Yes | Stub |
| `v2/live/market.py` | IBKR market data streaming. 5-second bar aggregation to 1-minute bars. VIX subscription, option chain snapshots. LiveFeatureEngine for real-time feature computation. | 14 | Yes | Stub |
| `v2/live/decision.py` | Model inference to TradeIntent. Scores candidates from live option chain, applies DecisionPolicy filters, constructs TradeIntent. | 15 | Yes | Stub |
| `v2/live/execution.py` | IBKR order placement and lifecycle. Resolves TradeIntent to ib_insync.Option, places OCO bracket orders, manages state machine. | 17 | Yes | Stub |
| `v2/live/adoption.py` | Manual position adoption (future). Allows operator to assign existing IBKR positions to the system for stop/TP management. | 9 | Yes | Stub |
| `v2/core/schema.py` | TradeIntent contract. Shared between training/replay and live. Includes `resolve_to_ibkr()` spec for contract resolution. | 206 | No (harness) | Implemented |
| `v2/core/policy.py` | DecisionPolicy. Shared gate threshold, risk ranges, cooldown, time blocks. Same parameters used by training labels and live decisions. | 83 | Yes (research) | Implemented |
| `v2/docs/execution.md` | Order lifecycle state machine spec. Full state diagram, reconnect recovery, orphan detection, audit trail, safety rules. | doc | No (spec) | Written |
| `v2/docs/contracts.md` | TradeIntent schema spec. Includes `resolve_to_ibkr()` function converting to ib_insync.Option. | doc | No (spec) | Written |
| `v2/docs/goal.md` | Mission statement. Paper trading completion gate: "Full RTH session 9:30-16:00 ET without crashes or orphans." | doc | No (spec) | Written |
| `pyproject.toml` | Project dependencies. Includes `ib_insync>=0.9.86` for IBKR API. | config | No | Present |

## Data Flow

```
IBKR TWS/Gateway (paper account)
    |
    v
market.py (IBKRMarketStream)
    |  subscribes: SPX 5-sec bars, VIX, option chain
    |  aggregates: 5-sec bars -> 1-min bars (FiveSecondMinuteAggregator)
    |  computes: 47 features in real-time (LiveFeatureEngine)
    |  snapshots: option chain candidates (OptionChainSnapshot)
    |
    v
decision.py (DecisionEngine)
    |  loads: model.pt (best model from ART2)
    |  runs: model forward pass on live features
    |  generates: candidate contracts via core/candidates.py
    |  scores: candidates against model predictions
    |  applies: DecisionPolicy filters (gate, cooldown, time blocks, daily loss cap)
    |  emits: TradeIntent (or no-trade)
    |
    v
execution.py (OCOExecutionEngine)
    |  resolves: TradeIntent -> ib_insync.Option (SPX, strike, right, expiry)
    |  places: OCO bracket order (entry + stop + target)
    |  manages: order state machine
    |    INTENT_RECEIVED -> ORDER_PLACED -> ACKNOWLEDGED -> FILLED
    |    FILLED -> BRACKET_LIVE -> CLOSING -> CLOSED
    |  handles: reconnect recovery, orphan detection
    |  logs: all transitions to audit.jsonl
    |
    v
service.py (TradingService)
    |  loop: bar-by-bar during RTH (9:30-16:00 ET)
    |  safety: daily loss cap (-5% blocks new entries)
    |  safety: max 1 concurrent position
    |  safety: EOD hard flatten at bar 389 (15:59 ET)
    |  safety: file-based kill switch
    |  output: results/live/audit.jsonl
```

## Order Lifecycle State Machine

```
INTENT_RECEIVED
    |
    v
ORDER_PLACED --timeout(LMT 60s, MKT 30s)--> ANOMALY
    |
    v
ACKNOWLEDGED
    |
    v
FILLED
    |
    v
BRACKET_LIVE
    |  exit triggers: stop / target / model_exit / max_hold / EOD
    |
    v
CLOSING
    |
    v
CLOSED
```

## Key Interfaces

**Inputs:**
- `model.pt` -- best model from ART2 pipeline
- `core/policy.py` -- DecisionPolicy (same parameters as training)
- IBKR TWS/Gateway connection (paper account)

**Outputs:**
- Paper trades executed on IBKR
- `results/live/audit.jsonl` -- complete audit trail of all state transitions
- P&L tracked by IBKR paper account

**Shared contracts with training/replay:**
- TradeIntent (core/schema.py) -- identical format in training labels, replay simulation, and live execution
- DecisionPolicy (core/policy.py) -- same gate/risk parameters
- Feature schema (core/features.py) -- same 47 features computed live

## Safety Rules (from execution.md spec)

| Rule | Detail |
|------|--------|
| Daily loss limit | -5% account value blocks new entries |
| Max concurrent | 1 position at a time |
| Order timeout | LMT: 60 sec, MKT: 30 sec flags anomaly |
| EOD hard flatten | Bar 389 (15:59 ET), non-negotiable |
| Kill switch | File-based emergency stop, closes all positions |
| Reconnect recovery | Reconcile positions after IBKR disconnect |
| Orphan detection | Detect and handle positions not tracked by system |

## Dependencies on Other Sections

| Section | Dependency |
|---------|------------|
| Market Data | `core/features.py` defines the 47-feature schema computed in real-time |
| Training Runs | `model.pt` is the trained model loaded for inference |
| Validation | Replay simulator and live execution must produce consistent behavior (same spread cost, same stop logic) |
| ART2 Pipeline | ART2 produces the model artifact deployed to paper trading |

## Audit Surface Area

- Parity: does live execution match replay simulation? Same spread model, same stop logic, same fill assumptions?
- Feature parity: are live-computed features identical to historical features in data.pt?
- TradeIntent consistency: is the same TradeIntent contract used across training labels, replay, and live?
- DecisionPolicy consistency: are policy parameters identical between training and live inference?
- Safety rules: are all 7 safety controls implemented and tested?
- State machine completeness: are all transitions handled, including edge cases (partial fills, disconnects)?
- Audit trail: is every state transition logged to audit.jsonl?
- Kill switch: can it reliably halt trading and flatten positions?
- Implementation status: all live/ files are stubs -- what is the implementation priority order?
- IBKR API dependency: `ib_insync>=0.9.86` -- is this maintained and compatible?

---

## Audit Questions -- Direct Improvements

**1. TradeIntent is missing `bid_at_decision` and `ask_at_decision` in replay's `model_to_intent()` -- no slippage audit baseline.**
`replay.py:123-142` constructs TradeIntent without `bid_at_decision`, `ask_at_decision`, or `limit_price` (default None). `contracts.md:56-59` defines these as quote provenance fields "for audit." Populate these from the option OHLCV data in replay now so that when live goes online, slippage analysis is possible from day one.

**2. `resolve_to_ibkr()` spec uses `exchange="SMART"` -- SPX options trade on CBOE.**
`contracts.md:190` shows `exchange="SMART"`. SPXW 0DTE options are exclusively listed on CBOE. SMART routing adds indirection and potential misrouting. The stub should specify `exchange="CBOE"` and `tradingClass="SPXW"` to target weeklies.

**3. Simulator assumes stop/TP fill AT the exact stop/TP price -- 0DTE options gap through levels.**
`simulator.py:154-155` fills stops at exactly `entry_px * (1 - stop_pct)`. In live 0DTE, gamma causes large jumps. A stop at $2.00 might fill at $1.50 on a fast SPX move. Replay overstates stop-loss outcomes. Adding a slippage model to simulator.py would make scores more predictive of live P&L.

**4. `daily_loss_cap_pct` uses `starting_equity` -- live needs current account value.**
`simulator.py:285` checks against `starting_equity`. In live trading the account grows/shrinks. A 5% cap on $10K starting is $500, but if account is $15K, that same $500 is only 3.3%. Live service needs to query IBKR account value.

**5. `MIN_HOLD_BARS=2` in features.py is undocumented in execution.md.**
`features.py:30` defines `MIN_HOLD_BARS = 2`. `simulator.py:149` suppresses exits for 2 bars after entry. `execution.md` never mentions a minimum hold period. Live implementation must replicate this or there is a parity gap.

**6. `STOP_COOLDOWN_BARS=5` is used in simulator but absent from the execution state machine.**
After a stop loss, `simulator.py:281` enforces 5-bar cooldown. The execution state machine in `execution.md` has no COOLDOWN state or transition. The live `decision.py` must implement this cooldown but it is not in the spec it follows.

## Audit Questions -- Deeper Planning

**7. Feature parity between Polygon training data and IBKR live data is the highest risk -- no mapping exists for all 47 features.**
Training features from Polygon SPX bars, SPY volume, VIX, and wide-grid SPXW option chains. 8 volume/flow features (`log_near_call_volume`, `log_near_put_volume`, `call_put_flow_ratio`) require real-time option chain volume per strike. IBKR does not stream option volume per strike the same way Polygon provides historical bars. The v1 `archive/tools/live_feature_parity_report.py` mapped an older 32-feature set. The v2 47-feature set has no equivalent mapping.

**8. Rolling z-score normalization requires 60-day lookback buffer -- how does live bootstrap?**
`core/features.py:97-100` defines 60-day rolling z-score. `market.py:14` mentions "Context bootstrap (download recent history for normalization buffer)" as TODO. Without this buffer, first 60 days of live features have different normalization characteristics. You need 60 trading days of Polygon-equivalent data from IBKR historical bars before the model sees its first bar.

**9. Trailing stop tiers cannot use IBKR native TRAIL orders -- requires custom implementation with race conditions.**
Simulator implements 4-tier trailing stop (30%/50%/80%/120% gain locking 0%/25%/50%/80%). IBKR's native trailing stop is a simple offset, not tiered. Live must: monitor prices itself, cancel/replace stop orders at each tier transition, track tier state internally. If price moves through two tiers in one bar while modifying orders, there is a race. This is the hardest parity problem in the system.

**10. State machine has no TIMEOUT state and no retry logic -- fast markets are normal for 0DTE.**
`execution.md:118`: "No retry" on rejection. For LMT, 60-second timeout. But 0DTE options during fast moves (FOMC, open) regularly see IBKR latency spikes of 5-15 seconds. No provision for order modification (chasing a fill), re-quote after stale-price rejection, or IBKR's "PendingCancel" state. These are normal conditions during the highest-signal periods.

**11. No Greeks-based position risk guard -- nothing prevents entering extreme-gamma options near expiry.**
`policy.py` defines risk as stop/target percentages and hold bars only. No check on the Greeks of the traded option. A $0.50 option at 3:30 PM has extreme theta decay (~$0.10/minute) and gamma. The model may learn to avoid these in training, but live safety rules have no Greeks guard. A simple check like "do not enter if theta_per_bar > X% of premium" would prevent worst-case scenarios.

**12. All live/ files are stubs (9-17 lines) while the model is at experiment 70+ -- when does implementation begin?**
The gap between "model works in replay" and "model works in live" is historically where trading systems fail. The v1 archive shows a complete IBKR integration existed. The question: can v2 reuse that code or does the TradeIntent contract change require a rewrite? A phased plan (market data first, then inference, then paper orders) with integration tests at each phase would reduce parity-bug risk.

**13. `RiskAdjustment` contract (schema.py:156-165) is defined but never referenced in simulator or execution.md -- dead code or unspecified feature?**
`schema.py` defines `RiskAdjustment` for mid-trade stop/TP modification. Simulator has no code path using it. Execution.md does not describe how it triggers state transitions. If trailing stops are the only mid-trade modification, the tiered logic already exists without `RiskAdjustment`. Either dead code or an undesigned feature.

## Related Documentation

- `v2/docs/execution.md` -- order lifecycle state machine, safety rules
- `v2/docs/contracts.md` -- TradeIntent schema, `resolve_to_ibkr()` function
- `v2/docs/goal.md` -- mission statement, paper trading completion gate
- `v2/docs/migration.md` -- v1 to v2 live trading migration notes
- `v2/docs/archive_map.md` -- maps v1 live trading files to v2 disposition
