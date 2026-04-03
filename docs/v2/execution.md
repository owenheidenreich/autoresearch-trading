# v2 Execution State Machine

## Purpose

Defines the order lifecycle from TradeIntent to completed trade.
This spec governs `v2/live/execution.py` behavior.

---

## State Machine

```
                    TradeIntent received
                          |
                          v
                    [INTENT_RECEIVED]
                          |
                    resolve to IBKR contract
                          |
                    +-----+-----+
                    |           |
               resolution    resolution
                succeeds      fails
                    |           |
                    v           v
              [ORDER_PLACED]  [REJECTED]
                    |           (terminal)
                    |
              +-----+-----+
              |           |
         acknowledged   rejected
              |           |
              v           v
        [ACKNOWLEDGED]  [REJECTED]
              |           (terminal)
              |
         +----+----+
         |         |
      partial    full
       fill      fill
         |         |
         v         v
   [PARTIALLY   [FILLED]
    FILLED]        |
         |    place stop + TP
         |    bracket orders
      full         |
      fill         v
         |   [BRACKET_LIVE]
         +-------->|
                   |
         +---------+---------+
         |         |         |
       stop      TP hit    model    max hold   EOD
       hit         |       exit     reached    flatten
         |         |         |         |         |
         v         v         v         v         v
                  [CLOSING]
                     |
                exit fill received
                     |
                     v
                  [CLOSED]
                  (terminal)
```

---

## States

| State | Description | Entry Condition |
|-------|-------------|-----------------|
| INTENT_RECEIVED | TradeIntent accepted, pending resolution | New TradeIntent with trade=True |
| ORDER_PLACED | Entry order sent to IBKR | Contract resolved, order transmitted |
| ACKNOWLEDGED | IBKR accepted the order | Order status callback: Submitted |
| PARTIALLY_FILLED | Partial entry fill received | Execution callback with remaining > 0 |
| FILLED | Entry fully filled | Execution callback with remaining == 0 |
| BRACKET_LIVE | Stop and TP orders live in market | Stop + TP orders acknowledged |
| CLOSING | Exit triggered, awaiting fill | Stop/TP/model exit/max hold/EOD |
| CLOSED | Trade complete, P&L recorded | Exit order filled |
| REJECTED | Order rejected by IBKR | At any pre-FILLED stage |
| CANCELLED | Order cancelled (timeout, kill switch) | Manual or automatic cancellation |

---

## Transitions

### Happy Path

1. `INTENT_RECEIVED` -> `ORDER_PLACED`: Resolve expiry/strike/right to ib_insync.Option.
   Qualify contract with IBKR. Place entry order (MKT or LMT per order_style).

2. `ORDER_PLACED` -> `ACKNOWLEDGED`: IBKR confirms order receipt.

3. `ACKNOWLEDGED` -> `FILLED`: Entry fill received. Record fill_price, fill_time,
   slippage (fill_price vs entry_ref_price).

4. `FILLED` -> `BRACKET_LIVE`: Place stop loss order at TradeIntent.stop_price.
   Place take profit order at TradeIntent.take_profit_price.
   Both as OCA (One Cancels All) group.

5. `BRACKET_LIVE` -> `CLOSING`: One of:
   - Stop order filled
   - TP order filled
   - Model exit signal (cancel remaining bracket, place market exit)
   - Max hold reached (cancel remaining bracket, place market exit)
   - EOD flatten (cancel remaining bracket, place market exit)
   - Kill switch activated

6. `CLOSING` -> `CLOSED`: Exit fill received. Record exit_price, exit_time,
   compute realized P&L.

### Error Paths

- `ORDER_PLACED` -> `REJECTED`: IBKR rejects (insufficient margin, invalid contract,
  market closed). Log rejection reason. No retry.

- `ACKNOWLEDGED` -> `CANCELLED`: Entry not filled within timeout (60 seconds for LMT).
  Cancel order. No trade recorded.

- `PARTIALLY_FILLED` -> `FILLED`: Continue waiting for remaining fill.
  If timeout, accept partial fill and place bracket on filled qty.

- Any state -> `CANCELLED`: Kill switch activated. Cancel all pending orders,
  flatten any open position at market.

---

## Reconnect and Restart Recovery

### Connection Drop During Active Position

1. On reconnect, query all open positions from IBKR.
2. Match against in-memory ExecutionState by contract identity.
3. For each matched position:
   - Verify stop and TP orders are still live in IBKR
   - If missing, re-place them from ExecutionState
4. For unmatched IBKR positions: log as ORPHANED, do not touch.

### Service Restart With Open Positions

1. Load last ExecutionState from audit log (JSONL).
2. Connect to IBKR. Query open positions and orders.
3. Reconcile:
   - ExecutionState says BRACKET_LIVE + IBKR has matching position: resume management.
   - ExecutionState says BRACKET_LIVE + IBKR has no position: mark CLOSED, compute P&L from fills.
   - IBKR has position + no ExecutionState: ORPHANED. Do not adopt unless manual override.

### Orphaned Bracket Reconciliation

An orphaned bracket is a stop/TP order pair in IBKR with no corresponding
ExecutionState. This happens when:
- Service crashed between entry fill and state persistence
- Manual IBKR TWS interaction created orders outside the system

Resolution:
1. Log orphan details (contract, qty, stop/TP prices, order IDs).
2. Do NOT cancel or modify automatically.
3. Alert human via audit log entry with severity=CRITICAL.
4. Human must manually resolve via TWS or explicit override command.

---

## Manual Position Adoption

**Not implemented in initial v2.** Planned for `v2/live/adoption.py`.

When implemented:
- Separate workflow from autonomous trading
- Operator explicitly assigns a position to the system
- System places stop/TP brackets on the adopted position
- Model manages exits but does not attribute entry to itself
- Adopted positions are tracked separately in metrics

---

## Audit Trail

Every state transition is logged to JSONL:

```json
{
  "timestamp": "2026-04-03T14:30:15.123Z",
  "event": "STATE_TRANSITION",
  "intent_id": "uuid",
  "from_state": "FILLED",
  "to_state": "BRACKET_LIVE",
  "details": {
    "fill_price": 3.45,
    "stop_order_id": 12345,
    "tp_order_id": 12346
  }
}
```

Audit log location: `results/live/audit.jsonl`

Every order sent to IBKR is logged before transmission. Every callback
from IBKR is logged on receipt. This provides a complete reconstruction
path if anything goes wrong.

---

## Safety Rules

1. **Daily loss limit:** If session P&L < -5% of account, block new entries.
   Open positions may still exit normally.

2. **Kill switch:** File-based kill switch. If file exists, cancel all pending
   orders and flatten all positions at market. No new entries.

3. **Max position count:** 1 concurrent position initially. Configurable.

4. **Order timeout:** LMT orders cancelled after 60 seconds if not filled.
   MKT orders should fill within seconds; if not filled in 30 seconds, flag as anomaly.

5. **Pre-market check:** Verify IBKR connection, market data subscription,
   and account permissions before entering the trading loop.

6. **EOD hard flatten:** All positions closed by bar 389 (15:59 ET).
   This is non-negotiable. No overnight risk on 0DTE options.
