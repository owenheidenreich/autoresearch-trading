# Protocol 101 Tuesday Live-Shadow Checklist

This checklist is for no-order live shadow only. The $500 in IBKR is the access/data reserve. The paper account baseline is $10,000. Do not place paper orders until the no-order live-data parity gate passes and the user explicitly approves paper order testing.

## 1. Pre-Open Guardrails

- Confirm `live_orders_enabled = false` and `broker_endpoint_called = false` in the router config.
- Confirm no paid historical download job is running.
- Confirm the frozen Protocol101 artifacts and Protocol051/054/081 fallback artifacts are unchanged.
- Confirm the JSONL output path is new for the session and will not overwrite historical artifacts.

## 2. Market-Data Connection

- Connect to IB Gateway/TWS.
- Verify fresh SPX context, VIX context, and SPXW option NBBO are available.
- Reject AM-settled `SPX` contracts; accept PM-settled `SPXW` only.
- Record quote/context freshness and root/settlement metadata in every row.

## 3. No-Order Shadow Capture

- Emit JSONL rows matching `protocol101_shadow_v1`.
- Include market snapshot, model decision, selected contract, intended no-order order reference, fill assumption, exit plan, and account state.
- Keep `intended_order.mode = no_order_shadow` and `will_submit_to_broker = false`.
- Run long enough to capture both entry opportunities and flat/no-entry periods.

## 4. Live Parity Review

- Validate feature columns against the frozen Protocol101 feature schema.
- Validate live SPX/VIX/option quote timestamps are fresh.
- Validate one contract max, no overlaps, flat-before-close behavior, and $10,000 affordability checks.
- Compare live feature distributions against historical replay distributions before trusting decisions.

## 5. Paper-Order Rehearsal Gate

- Only after no-order live parity passes, rerun order-state rehearsal on captured live shadow rows.
- Require max concurrent positions = 1, all order intents affordable, and no stale quotes.
- Ask for explicit user approval before enabling any paper order endpoint.

## Blockers

- Missing SPX/VIX context.
- Missing OPRA/SPXW NBBO.
- Stale quote/context rows.
- Any row with `live_orders_enabled = true` before approval.
- Any selected contract with root `SPX` instead of `SPXW`.
- Any unaffordable one-contract trade under the $10,000 paper-account baseline.
