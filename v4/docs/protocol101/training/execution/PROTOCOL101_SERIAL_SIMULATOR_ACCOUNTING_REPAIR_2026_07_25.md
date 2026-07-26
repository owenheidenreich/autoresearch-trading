# Protocol101 Serial Simulator Accounting Repair

Date: 2026-07-25

## Finding

The Stage-1 readiness reconciliation found two account-state defects in
`protocol101_serial_simulator_v2`:

1. The final pending trade from each session was not realized before the first
   candidate of the next session. Buying power therefore appeared to reset to
   the fold's starting cash at each session boundary.
2. Affordability reserved the option premium but not the $3 round-trip fee,
   even though the fee was already included in realized trade PnL.

These defects were economically material. They could admit trades that a
strict one-account replay could not afford and distort fold PnL, drawdown,
ruin, null, and heuristic results.

## Repair

The current simulator contract is:

`protocol101_serial_simulator_v4_account_continuity_fee_reserve`

It:

- realizes the prior session's pending trade before granting buying power to
  the next session;
- carries account cash continuously across sessions within each fold;
- records session-starting equity for the 5% daily breaker;
- reserves the configured round-trip fee in the affordability check.

Regression tests cover cross-session cash carry-forward and fee-reserve
affordability.

## Evidence Consequence

- Exact current-contract null and fixed-heuristic packets were regenerated
  after the repair.
- The current fixed G3 heuristic is policy 5 at `$4,592` pooled
  fee-adjusted PnL. It is not an eligible candidate: only 3/5 folds are
  profitable and weak-fold equity violates G4 v2.
- Any Stage-1 packet naming simulator v2/v3, or lacking the fee reserve, is
  non-authoritative for current G1/G3/G4 decisions and must be replayed before
  reuse.
- This does not automatically invalidate every historical campaign: older
  campaigns may have used different simulators. Their curves remain
  methodology/provenance evidence unless independently replayed under the
  current contract.

No model training, threshold selection, holdout read, recorder read, broker
call, paper submission, or runtime mutation occurred during this repair.
