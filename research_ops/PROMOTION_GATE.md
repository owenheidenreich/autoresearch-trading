# Promotion Gate

Last updated: 2026-05-24

This gate governs any change to `PAPER_DEFAULT_PROTOCOL101` or any operational
default. It does not approve real-money trading.

## Minimum Required Evidence

1. Frozen candidate identity and artifact manifest.
2. Cartography report proving the candidate's code, data, labels, and runtime
   surfaces.
3. Experiment RFC written before results are known.
4. Replay/live parity verifier.
5. Execution realism verifier.
6. Lifecycle parity verifier when exits or hold logic are affected.
7. Cost, fee, slippage, spread, and non-fill sensitivity.
8. Validation memo separating diagnostic, selection, and untouched evidence.
9. Operational rollback plan.
10. Decision memo explicitly approving the default change.
11. CEO dashboard update.

## Blocking Conditions

Promotion is blocked if any of these are true:

- The candidate depends on open critical assumptions without an accepted-risk
  decision.
- Fill realism is unmeasured or materially worse than replay.
- Live features are not proven causal and semantically equivalent to replay.
- The validation window was repeatedly mined and no untouched path remains.
- Runtime behavior, launchd, broker mode, or paid-data behavior changed outside
  the approved packet.

## Decision Memo Fields

```text
Decision:
Candidate:
Control:
Does this change PAPER_DEFAULT_PROTOCOL101:
Does this authorize model training:
Does this authorize broker/data/runtime action:
Evidence reviewed:
Assumptions accepted:
Rollback trigger:
Owner:
Review date:
```
