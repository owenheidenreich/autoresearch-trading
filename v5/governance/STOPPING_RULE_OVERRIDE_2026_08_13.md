# Override of the 2026-08-12 stopping rule

**Status: SIGNED AND IN FORCE — owner signed 2026-08-13.**

## What was pre-committed

On 2026-08-12, before any outcome existed, the plan recorded:

> **If the Phase 3/4 screen returns no edge, the research programme closes.** The work is written up as a
> negative result... It binds whether the screen fails at the known-answer campaign or at the economics.

Phase 3 ran 2026-08-13 01:00–01:41 and stopped: both nulls clean at 0.0000 false pass, recovery at the
declared threshold 8% against 80% required. **The rule bound.**

## What the owner overrode, and why it is not simply goalpost-moving

The owner overrode it on 2026-08-13. The reason this is recorded rather than quietly done is that
overriding a pre-commitment is exactly the move that destroys the value of pre-commitment, and the only
defence is that the *premise changed measurably*, not that the answer was unwelcome.

What changed, all measured after the rule was written and none of it by inspecting an outcome:

1. **The threshold the screen failed against was derived from the wrong contract universe.** The 65.73%
   bar came from `option_correct_call_dollars` / `option_wrong_call_dollars`, which were measured on the
   1,031 Phase-1 trajectories. Those sit in the OTM band that a `$3.00–$8.00` price filter selects. The
   whole-ladder measurement on 232 sessions and 82,709 observations puts the **near-ATM** break-even at
   **50.60%** against OTM's 54.50%, and the loop bar at **61.05%** against **68.29%**.
2. **The price filter was never a decision.** It was an undeclared proxy that had been binding research
   for months. It is now an explicit `moneyness_band` knob, released by a signed charter amendment.
3. **The sample constraint is removable at zero cost.** SPXW `ohlcv-1m` is $0.00 at every era back to
   2013, and daily 0DTE expiries make roughly 1,050 further sessions usable — against the 447 the
   synthetic run showed are needed to certify a 62% edge after a 24-candidate sweep.

A screen on ~1,300 option sessions at a bar of 61.05%, with the contract universe declared rather than
inherited, is a materially different experiment from one on 251 sessions at 68.29%. That is what
"genuinely new" is meant to mean.

## What the override does not license

- **It does not re-open the failed screen.** The 2026-08-13 campaign stands as recorded, and its ledger
  row is not amended.
- **It does not lower any standard.** The known-answer campaign runs again in full, with the same three
  criteria and the same order: nulls and recovery **before** any economics. If recovery fails again, the
  screen stops again.
- **It does not permit a second null repair.** That remains spent.
- **It is not a second override.** A further failure closes the programme, and this document is the record
  that one override has been used.

## Signature

- Drafted **2026-08-13** by Claude Opus 5.
- **Signed: repository owner, 2026-08-13.** Instruction given in conversation: *"yes override the stopping
  rule"*, after the rule and the case against overriding it were read back in full.

*Signing this contacted nothing and changed no runtime state.*
