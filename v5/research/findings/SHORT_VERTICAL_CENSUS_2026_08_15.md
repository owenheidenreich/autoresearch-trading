# The short-vertical census: nothing clears even fee-only — the owner's STOP condition fires

**2026-08-15.** Job 44, owner-authorized: one preregistered $0 census of defined-risk short
verticals on the owned quote corpus, with the pre-committed instruction *"STOP the options program
if nothing clears even fee-only execution."*

## One sentence for the bot

Selling defined-risk premium — the one structure every measurement pointed at — shows the expected
positive sign at perfect midpoint execution but nowhere near strongly enough to certify, and it is
gone entirely at real execution; per the owner's pre-committed decision, options-strategy
development stops.

## What was measured

36 preregistered cells — call/put credit spreads × widths 5/10 × short strike 0/10/20 points OTM ×
entries 13:00/14:00/15:00 — on the 243 coverage-included sessions, both legs priced causally from
the entry snapshot, held to validated cash settlement (so the spread toll is paid once, at entry),
a $500 maximum-loss abstention law (the 5% breaker on the $10,000 account), abstentions scoring
zero, 20,000-draw session bootstrap with a one-sided bound corrected for the family of 36. Zero
fitted parameters. Declaration V2 `45f78cae…` (V1's run was voided by an aggregation crash before
any outcome was read; identical law).

## The result

**Decision: `STOP_CONDITION_MET_NO_CELL_CLEARS_FEE_ONLY`.** No cell clears at fee-only; none
clears at touch. 4,104 trades across 8,748 cell-sessions.

| Best cells (by fee-only mean/session) | fee-only | corrected LCB | folds+ | touch | trades |
|---|---:|---:|---:|---:|---:|
| call credit, w5, ATM, 13:00 | **+$14.85** | −$30.59 | 3/5 | +$1.78 | 243 |
| call credit, w5, 10-OTM, 14:00 | +$12.69 | −$23.09 | 2/5 | +$2.53 | 239 |
| call credit, w5, 10-OTM, 13:00 | +$10.50 | −$29.93 | 3/5 | +$0.17 | 243 |
| … worst: put credit, w5, ATM, 14:00 | −$27.53 | −$72.22 | 1/5 | −$42.02 | 240 |

Three honest observations:

1. **The mechanism's sign is there.** 10 of 36 cells are positive at fee-only, all of them call
   credits, consistent with the variance premium measured in every year of the corpus (ledger row
   332). The put side is deeply negative throughout — over this corpus the market's rises made put
   credits collect too little for their settlement losses.
2. **The size is far below certifiability.** The best mean, +$14.85/session (~$3,700/year on the
   account), sits $30+ above its family-corrected lower bound on 243 sessions. Even uncorrected,
   no cell approaches a clean pass; fold signs never exceed 3/5.
3. **Execution erases even the sign.** At the touch the best cell keeps +$2.53/session and most go
   negative. Fee-only is the *best possible* execution — midpoint fills with fees — so no
   execution improvement can rescue what fails there. This kills the "measure sub-second fills
   next" branch: there is nothing behind the execution wall to reach.

## What this closes

Combined with the prior closures, both sides of retail SPXW 0DTE at this account size are now
measured: buying loses to the variance premium in every state examined (rows 332, 338, 340, 341);
selling it in defined-risk vertical form shows a sign too small to certify at perfect execution
and negative at real execution; the 4-leg fly lost outright; naked shorts fail survival. **The
owner's pre-committed STOP applies: options-strategy development is stopped.**

## Scope limits, stated plainly

- 243 sessions bound what is detectable: a true ~$15/session call-credit edge might exist below
  this census's detection floor. That would be roughly a 37%/year return on the account if real —
  but certifying it would need substantially more independent sessions than daily 0DTE history
  contains, or years of forward confirmation, for an edge whose existence this census cannot
  distinguish from zero.
- These 243 sessions are development-exposed; even a pass here could not have grounded a tradable
  claim without fresh confirmation.
- The census prices one entry per cell per day with no management; a managed variant is a
  different, unauthorized question and does not reopen this one.

## Evidence

- Receipt: `v4/audit/autoresearch/short_vertical_census_2026_08_15/receipt.json` (tracked);
  row table `v5/work/short-vertical-census/run_2026_08_15/session_cell_results.parquet`.
- Declarations V1/V2 and log: `v5/history/jobs/short-vertical-census/`.
- Implementation: `v5/research/short_vertical_census.py`,
  `v5/ops/run_short_vertical_census.py`, tests `v5/tests/test_short_vertical_census.py`.
