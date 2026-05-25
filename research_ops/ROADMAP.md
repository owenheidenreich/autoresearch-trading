# Research Ops Roadmap

Last updated: 2026-05-24

This roadmap prioritizes information gain, falsification power, reduction of
hidden assumptions, replay-only edge risk, and operational realism.

## Immediate Audits

1. Execution realism verifier for Protocol101.
   - Why: replay edge is not tradable until fills, quote age, latency, and
     cancellation behavior are measured.
   - Evidence: paper/no-order decision table joined to quotes, order intents,
     guard rejects, submit attempts, fills, cancels, and quote age.
   - Falsifier: fills are materially unavailable or worse than replay
     assumptions for selected trades.

2. Replay/live candidate parity cartography.
   - Why: live decisions must use the same causal features as training and
     replay.
   - Evidence: matched snapshot comparison of candidate inclusion, features,
     quote age, spread, side, premium, and moneyness.
   - Falsifier: live admits, drops, or transforms candidates differently enough
     to change selected actions.

3. Lifecycle parity verifier.
   - Why: entry edge can be invalidated by live exit state mismatch.
   - Evidence: known replay trajectories passed through live lifecycle builders
     with action-by-action diffs.
   - Falsifier: live hold/exit/stop/forced-flat actions diverge without an
     approved reason.

## Observability Work

1. Decision trace schema for no-order and paper-submit sessions.
2. Explicit order-intent, guard-decision, quote-age, and account-state joins.
3. Dashboard-safe summaries that exclude credentials, account identifiers, and
   raw paid-data contents.

## Validation Hardening

1. Mark exposed windows and prevent repeated benchmark mining from being called
   final validation.
2. Define untouched evaluation rules before any new candidate is trained.
3. Separate diagnostic, selection, promotion, and final-evidence metrics.

## Architecture Research

Architecture work is deferred until execution and parity gates reduce the risk
that new models are optimizing simulator artifacts. Future RFCs may evaluate
hierarchical policies, trajectory models, or uncertainty-aware execution models,
but only after their specific assumptions and falsifiers are written down.

## Long-Term ML Research

1. Execution-aware utility modeling.
2. Probabilistic fill and slippage modeling.
3. Causal controls for liquidity and quote mechanics.
4. Sequential policy formulations only after state, action, reward, and
   observability assumptions are explicitly tested.
