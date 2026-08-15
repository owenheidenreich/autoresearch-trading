# Tuesday Stage Evidence Plan

Date: 2026-05-25

Target session: Tuesday 2026-05-26, 06:30 to 13:00 PT market window

Purpose: collect evidence, not force-pass readiness.

## What We Changed Today

1. Confirmed the Tuesday Gateway job uses the automatic IBC paper Gateway login script.
2. Removed the unnecessary early 2FA watch.
3. Kept the existing installed launchd stack as the source of truth.
4. Added Codex read-only checkpoints at 06:30, 07:15, 12:50, and 13:45.
5. Added a pre-submit dry-run validation step to the Tuesday paper-fill runner so Stage 3 has a chance to collect real dry-run evidence before any guarded paper submit.
6. Added the fail-closed observability validator to the Tuesday runner summary.
7. Added reconstruction fields to the Tuesday runner so Stage 6 can be judged against the observability contract.

## Simple Phase Plan

### Phase 1: Gateway Starts

Time: 06:05 PT

What should happen:

- The automatic IBC script starts IB Gateway paper.
- It uses the stored paper credentials.
- It waits for an API handshake.

Stage helped:

- Stage 1, preflight readiness.

What would count as useful evidence:

- Gateway started.
- API port accepted a connection.
- No credential failure.

### Phase 2: Preflight

Time: 06:20 PT

What should happen:

- The system checks that IBKR paper API access and market data are available.
- It should not place orders.

Stage helped:

- Stage 1, preflight readiness.

What would count as useful evidence:

- Paper account is identified safely.
- SPX/VIX/SPXW data access works.
- Required paper-only flags are present.

### Phase 3: No-Order Surface Check

Time: starts around 06:31 PT

What should happen:

- The system collects SPX, VIX, and SPXW quotes.
- It builds the option ladder.
- It scores the surface without placing orders.

Stage helped:

- Stage 2, no-order observation readiness.
- Stage 6, observability readiness.

What would count as useful evidence:

- Fresh quotes.
- Quote timestamps and quote age.
- Candidate set or clear no-candidate reason.
- No broker order call during this phase.

### Phase 4: Dry-Run Before Submit

Time: during each bounded probe

What should happen:

- Before any paper order, the runner validates the same intended order as a dry-run.
- The dry-run must not call the broker order endpoint.

Stage helped:

- Stage 3, paper-dry-run readiness.
- Stage 6, observability readiness.

What would count as useful evidence:

- A dry-run row exists.
- The guard passed or blocked with clear reasons.
- `broker_order_endpoint_called=false`.

### Phase 5: Bounded Paper Submit

Time: only after dry-run and guard pass

What should happen:

- The runner may submit up to the approved limits.
- Quantity is one contract.
- Paper account only.
- No real money.

Stage helped:

- Stage 4, paper-submit readiness.

What would count as useful evidence:

- Guard pass before submit.
- Paper submit row.
- Broker status.
- Fill, cancel, or timeout.
- No raw account id.

### Phase 6: Exit Or Forced Exit

Time: immediately after an entry fill

What should happen:

- If entry fills, the runner immediately attempts a paper exit.
- If normal exit does not fill, it tries the forced exit offset.

Stage helped:

- Stage 5, lifecycle/exit readiness.

What would count as useful evidence:

- Entry fill detected.
- Exit intent created.
- Exit fill, cancel, or timeout logged.
- No open-position risk at the end.

### Phase 7: Post-Session Review

Time: 13:15 launchd evidence job, 13:45 Codex review

What should happen:

- Evidence packets are regenerated.
- Codex reviews the logs and summaries.
- No more orders are submitted.

Stages helped:

- Stage 6, observability/reconstruction readiness.
- Stage 7, model-improvement gate review.

What would count as useful evidence:

- Observability validation status.
- Execution observation count.
- Filled round-trip count.
- Remaining blockers listed honestly.

## What Can Move After Tuesday

- Stage 1 can move only if current Gateway, account, API, and market-data proof exists.
- Stage 2 can move only if no-order quote/context/candidate evidence exists.
- Stage 3 can move only if dry-run rows exist and show no broker endpoint call.
- Stage 4 can move only if submit/status/fill/cancel/timeout evidence exists.
- Stage 5 can move only if a filled position is exited or safely resolved.
- Stage 6 can move only if the new observability validator passes on current logs.
- Stage 7 can improve only if the evidence also satisfies model-gate blockers. It may still remain blocked.

## What We Should Not Do Tuesday

- Do not manually submit extra orders.
- Do not change launchd during the run.
- Do not change runtime flags during the run.
- Do not train or tune models.
- Do not treat fills as proof of profitability.
- Do not mark a stage PASS just because code exists.

