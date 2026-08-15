# Observability Remaining Gaps

Date: 2026-05-25

## Not Proved By This Implementation

- Current-day IBKR paper connection.
- Current DU paper account identity.
- Current SPX/VIX/SPXW entitlements.
- A live no-order session passing the new observability validator.
- A real nonzero Protocol101 candidate set.
- A real Protocol101 BUY/SELL intent.
- A real Protocol101 paper-dry-run intent chain.
- A real paper-submit order id.
- Real broker status, fill, cancel, or timeout.
- Real open-position lifecycle state.
- Real exit or forced-flat behavior.
- Real end-to-end reconstruction from live durable logs.

## Implementation Gaps That Remain Narrowly Technical

- Candidate diagnostics still expose `top_rejected_contracts` inside `candidate_gate_diagnostics`; the validator fail-closes on `filter_reason` but does not independently require every diagnostic subfield.
- Account-prefix guard booleans are fully available in executor guard rows, but pre-executor `risk_gate` rows only know intent validation, not IBKR account permission. This preserves behavior and avoids inventing account evidence before the executor guard exists.
- No-candidate decisions log empty raw logits and null wait logit because Protocol101 does not run the model forward when no candidates exist. That is truthful reconstruction, not a model output.
- Broker order id/status/fill/cancel fields are schema-supported and fake-tested, but no live paper order evidence exists.
- Lifecycle fields are schema-supported and fake-tested, but real lifecycle evidence requires a held paper position or an approved fixture-specific lifecycle run.

## Operational Gaps

- Stage 1 remains UNKNOWN.
- Stage 2 remains UNKNOWN until a no-order live session passes observability validation.
- Stage 3 remains NOT TESTED until a real Protocol101 intent reaches paper-dry-run with no broker call.
- Stage 4 remains BLOCKED until explicit approval and one-contract paper-submit evidence exists.
- Stage 5 remains UNKNOWN until open-position and exit evidence exists.
- Stage 7 remains BLOCKED until operational and Section 3 model gates pass.

