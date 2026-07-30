# FT2-08 Repair Independent Completion Audit

Terminal outcome: `accepted`

The stable FT2-04, FT2-05, and FT2-08 producer packets satisfy the bounded
FT2-08-REPAIR Goal under the owner's next-completed-minute fill convention.
This is independent acceptance of the producer repair only. It is not model,
training, holdout, paper, live, or promotion approval.

## What Passed

- The amended authority is SHA-256
  `2363d3f986daba20bd5087ed751dc5b2d839e76cd6413aeca0bcd255eb98857a`.
  It contains A1-A4, the deferred sub-minute decision, and the deferred
  component-freeze and tail-guard owners. The preserved prior authority hashes
  to `893aa0664944680e053ffd12a4d44c8a798397cbeed6ad56cd682fd864d9f832`.
- FT2-04 freezes t+1 entry and exit fills, post-fill path windows, exact 15:55
  forced flat, no-bid full loss, MNAR sensitivity, and the 45-session role
  firewall. Independent recomputation found zero overlap with 225 outer-test,
  30 holdout, and five embargo sessions. Its 12 focused tests passed.
- FT2-05 contains 45 checkpoint Parquets and 460,937 rows. Every entry-fill
  timestamp is exactly one minute after its decision, decisions span
  09:32-15:29 ET, remaining-session path counts include no-bid states, no-bid
  forced-flat rows use zero value and full loss, and all hold-flat clocks use
  exact 15:55 ET. The five V2 tests passed.
- FT2-05 includes the required V1/V2 impact and MNAR artifacts, pins builder
  SHA-256 `eada84800b5141591995c29898e7d5c2a596bd68d5fa99712738126e9bd913f6`,
  and explicitly chains preserved V1 receipt
  `9085a09cbc1583973fdb372c78d78568e3fb1baa0fa1d9b42c23b05543ee24de`.
- FT2-08 covers Step 4(a-i): the physical label firewall, cent-exact account
  ledger, source-neutral contract identity, complete-ladder safe action,
  dedicated open-contract state, signed-amendment activation gate, executable
  timing and warmup/reset laws, and replay-v5.1 design requirements.
- The repaired tensor contains the signed 17 alpha features exactly once:
  nine invariant fields on the market axis and eight contract-dependent fields
  on the exact 42-contract axis. The V1 19-market plus four-contract layout is
  fully cross-walked; contract-free E-family Greeks are prohibited.
- The contract validator independently passed 17/17 with its write intercepted,
  leaving the producer's `validation.json` unchanged.
- All current receipt inventories reproduce: 7 FT2-04 deliverables, 228 FT2-05
  deliverables, 12 FT2-08 deliverables, and every FT2-08 pinned input.
- All 28 FT2-20 findings are present with their original severities
  (18 blocking, 10 material), explicit owners, and an exact repair, partial
  repair with retained gate, or named FT2-10/11 deferral.
- Simulator v5 remains byte-identical to prior independent acceptance:
  `7296a437577ed006326d2ad35ad1f3499c4925334556d64d8c5fb75e4985f548`.

The active stale-language scan found no affirmative same-minute-fill,
46-session-current, or neutral-censoring claim. Matches were historical V1/V2
comparisons, explicit negations, or the required no-bid-excluded diagnostic.

## Scope

No producer file was modified. The focused tests used no pytest cache or Python
bytecode writes, and the contract validator's output write was intercepted.
No training, fitting, tuning, protected market data, recorder, broker, paid
data, simulator, runtime, promotion, or paper action occurred.

Per the producer's explicit sequencing, this audit does not evaluate the
top-level aggregate receipt that still preserves the Step-0 terminal state; the
aggregate receipt may now hash this acceptance result.

## Highest Allowed Claim

> FT2-08 (with its FT2-04/05 dependencies) is repaired against the FT2-20
> findings under the owner's t+1 fill convention; FT2-10 repair is unblocked.
