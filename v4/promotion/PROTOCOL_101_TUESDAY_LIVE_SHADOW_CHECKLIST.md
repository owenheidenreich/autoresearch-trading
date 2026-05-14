# Protocol 101 Tuesday No-Order Live Shadow Checklist

Purpose: data parity and operational rehearsal only. This is not a profit experiment.

## Non-Negotiables

- `live_orders_enabled = false` for the whole run.
- `broker_endpoint_called = false` for the whole run.
- Protocol 101 remains frozen.
- Paper account starts at `$10,000`; the real `$500` IBKR reserve is not trading capital.
- Max initial paper size remains `1` contract and max concurrent positions remains `1`.
- Stop and ask for explicit approval before enabling any paper-order endpoint.

## Run Sequence

1. Connect IB Gateway.
2. Confirm SPX, VIX, and SPXW NBBO are live and fresh.
3. Run no-order Protocol101 shadow capture with live_orders_enabled=false.
4. Validate protocol101_shadow_v2 schema, feature parity, quote freshness, SPXW root, PM settlement, and account state.
5. Run order-state rehearsal on captured live shadow rows.
6. Stop and ask for explicit approval before any paper-order endpoint is enabled.

## Stop Conditions

- IBKR connection fails or reconnects repeatedly.
- SPX or VIX context is missing, stale, or not timestamped.
- SPXW option NBBO is missing, stale, locked, crossed, zero, or wrong root/settlement.
- Live feature values cannot be mapped to the historical Protocol101 schema.
- Any event has live_orders_enabled=true or broker_endpoint_called=true.
- Paper account risk gate blocks for a hard reason.
- Order-state rehearsal shows overlap, unaffordable trade, or non-flat session.

## Required JSONL Events

- `market_snapshot`
- `candidate_set`
- `model_decision`
- `risk_gate`
- `paper_account_state`
- `exit_decision`

## Current Decision

`ready_for_tuesday_no_order_live_shadow_only`
