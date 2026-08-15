# Missing Evidence List

Date: 2026-05-25

System: Protocol101 / `PAPER_DEFAULT_PROTOCOL101`

Rule used: if evidence is missing, the stage is not treated as PASS.

## Stage 0: Repo/control readiness

Status: PASS

Missing evidence:

- A single canonical document that reconciles current guarded `paper-submit` launch configuration with older no-order and not-paper-approved promotion guidance.
- Explicit retirement or supersession notes for stale Protocol101 promotion docs.

## Stage 1: Preflight readiness

Status: UNKNOWN

Missing evidence:

- Current-day IBKR paper connection proof.
- Current paper account id evidence, safely redacted but proving DU/paper account prefix.
- Current market data entitlement proof for SPX, VIX, and SPXW NBBO.
- Current proof that required env vars and runtime flags were present at session start.
- Current launchd loaded/enabled state. This assessment intentionally did not run launchd commands.
- Preflight output from `run_protocol101_paper_preflight.sh`. This assessment intentionally did not run broker scripts.

## Stage 2: No-order observation readiness

Status: UNKNOWN

Missing evidence:

- Observed nonzero Protocol101 candidate set in live no-order mode.
- Observed Protocol101 entry decision in live no-order mode.
- Raw quote timestamp and quote age preserved for every option quote used in every decision.
- Candidate set hash.
- Feature vector hash.
- Raw logits or full model-score vector for every decision.
- Action mask or equivalent explanation of available actions.
- Complete reason chain for no-candidate and no-entry outcomes.
- Live no-order full-action parity pass.

## Stage 3: Paper-dry-run readiness

Status: NOT TESTED

Missing evidence:

- Real live Protocol101 BUY/SELL intent in paper-dry-run mode.
- Guard validation result for that exact intent.
- No `placeOrder` call for that exact intent.
- Would-submit order payload with contract, side, quantity, limit, quote age, context age, account cash, and guard result.
- Paper-dry-run log row from the live Protocol101 path, not only synthetic executor smoke.

## Stage 4: Paper-submit readiness

Status: BLOCKED

Missing evidence:

- Explicitly approved one-contract paper-submit probe.
- Guard-passed BUY or SELL paper intent from Protocol101.
- Current DU/paper account id evidence.
- Account affordability evidence at order time.
- Real quote freshness evidence at order time.
- Broker order id.
- Broker status row.
- Fill, cancel, or timeout row.
- Confirmation that no real-money path was available during the run.
- Evidence that only one contract and one open position were possible in the actual session.

## Stage 5: Lifecycle/exit readiness

Status: UNKNOWN

Missing evidence:

- Open position detection after an entry fill.
- Runtime lifecycle state after entry.
- Lifecycle row built from a real held position.
- Exit intent from a real held position.
- Forced-flat intent or proof of scheduled forced-flat behavior.
- Disconnect/failure behavior while holding a position.
- Exit submission, fill, cancel, or timeout rows.
- Reconstructable final position state after exit or forced-flat.

## Stage 6: Observability/reconstruction readiness

Status: FAIL

Missing evidence:

- Raw decision timestamp.
- Received timestamp.
- Raw quote timestamp.
- Quote age for every quote used by every decision.
- Candidate set hash.
- Feature vector hash.
- Raw logits.
- Margins and thresholds for all action comparisons.
- Action mask.
- Selected action and selected contract payload for every decision.
- Guard result with every blocking reason.
- Order intent payload.
- Broker endpoint called flag on all broker-adjacent rows.
- Broker order id and status timeline.
- Fill/cancel/timeout status.
- Latency in milliseconds.
- Lifecycle state before and after exit decisions.
- Complete account snapshot at decision and order time.

## Stage 7: Evidence required before model tweaking

Status: BLOCKED

Missing evidence:

- Minimum fill observations required by fill-model readiness.
- Minimum execution observations required by fill-model readiness.
- At least one closed paper round trip.
- Untouched holdout data availability.
- Live no-order full-action parity pass.
- Complete reconstruction pass from Stage 6.
- Preregistered model-improvement hypothesis packet.
- Explicit approval that Section 3 model gates are reopened.

## Cross-Cutting Missing Evidence

- A current operational truth packet from a regular market session after observability repair.
- A clean separation between stale docs and current approved operating defaults.
- A fail-closed audit that marks the run not ready when reconstruction fields are missing.
- A CEO-readable single page that states current allowed mode before every session.

