# Settlement-source law for the parity-settled corpus

**Decided 2026-08-22 on owner delegation.** This was carried as "the oldest unmade decision" in job
46. It turns out not to have been unmade — it was **written and never ratified, and its one binding
requirement was never measured**. Both are fixed here.

## 1. The decision in one line

**The law already exists in the learning-content design §4.3 and is ratified here verbatim.** Its
zero-recovery twin requirement is now **measured and discharged for Member P**, and remains **binding
and un-discharged for Member Q**.

## 2. What the law is

Plain English first. Most SPX 0DTE options in this corpus expire without anyone buying them back. To
know what a position was worth at the end, you need the index's official closing value. **We only own
that official number for 251 sessions — about 24% of the corpus.** For the other 76% the closing
value is *derived* from option prices by put-call parity ("parity spot") rather than published.

The signed design's answer, ratified unchanged:

1. **Settlement source is a first-class per-session column** — `official_1600` or `parity_close` —
   and appears in every receipt. **A cash settlement is never called a fill.**
2. **Every terminal-dependent number carries a zero-recovery twin**: the same number recomputed with
   the terminal intrinsic forced to $0. **Any result whose sign differs between the twins is not
   bankable.**
3. **The session grid is read from the data, never assumed.** Backfill sessions end at 15:58, not
   16:00. The share of exits resolving as executable bid / delayed first-later bid / validated cash
   settlement / blocked is a **per-era QC gate reported before any fit**.
4. Base rates are re-measured per era; the target itself does not move, being account arithmetic.
5. Era is confounded with chronology and the design says so; composition-matched controls are drawn
   **within era**.

## 3. The measurement that was missing, now made

Computed 2026-08-22 over 24 sessions spanning both eras, comparing the settled and zero-recovery
twins on the same actions. Receipt: `zero_recovery_twin_2026_08_22.json`.

| Quantity | Measured |
|---|---:|
| Exit-matrix cells resolved by the settlement branch | **39.67%** |
| **Model-fired entries** whose bracket exit used settlement | **0.00%** (0 of 4,856) |
| Model-fired entries: settled vs zero-recovery mean | −$17.33 vs **−$17.33** |
| All priced actions (n=77,966): settled vs zero-recovery mean | −$27.18 vs **−$27.18** |
| Ordering edge: settled vs zero-recovery | +$9.86 vs **+$9.86** |
| **Delta, every stream** | **exactly $0.00** |

**Nearly 40% of the exit matrix does rest on a settlement number, and none of the economics does.**
The reason is structural rather than lucky: Member P's bracket exits at the +50%/−30% touch minute or
at the 60-minute horizon, and those minutes sit inside the session where executable bids exist. The
settlement branch fires for late-session minutes and for contracts that stop quoting — regions the
60-minute bracket never exits into.

**A logical bound worth stating, because it makes half the question moot in advance.** Zero recovery
replaces a terminal intrinsic, which is never negative, with $0 — so the twin can only move a value
*down*. A negative headline therefore **cannot flip sign**, and the 2026-08-21 entry result of
−$15.66 per entry was settlement-robust before anything was computed. Only a *difference* between two
streams could have moved, which is why the ordering edge was the number actually worth measuring. It
did not move either.

## 4. What is discharged and what is not

- **DISCHARGED — Member P (first touch, +50%/−30%, 60 minutes).** The twin is identical to the cent
  on both streams. No P-derived number in this project is settlement-dependent, and re-reporting a
  twin for it would be ceremony. Any future P-family result on **this** corpus and **this** horizon
  inherits the discharge, and must cite this file rather than re-deriving it.
- **NOT DISCHARGED — Member Q (ENTER vs WAIT, 120 minutes), and anything else reaching past the
  entry window.** A 120-minute horizon from a late-morning entry lands squarely in the
  settlement-exposed region where 39.67% of cells resolve by intrinsic. The design already requires
  the twin to flow through Q's *targets* as a declared evaluation variant rather than through the
  reporting layer, and that requirement stands untouched. **Q may not be fitted without it.**
- **NOT DISCHARGED — any change of horizon, exit law, or contract universe.** The discharge is a
  measurement about where the 60-minute bracket exits, not a property of the corpus. Change the exit
  and the exposure changes with it.

## 5. Consequence for future declarations

Any declaration covering a terminal-dependent member must state which case it is in: **discharged by
this file** (P-family, 60-minute bracket, this corpus), or **carrying its own twin as a declared
evaluation variant**. A declaration that is silent on settlement exposure is incomplete, and the
Outcome Run Gate's reviewer should refuse it.
