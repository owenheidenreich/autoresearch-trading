# Scoped reopening of ledger rows 340/341 — drawdown-ordered lifecycle experiment

**Status: SIGNED 2026-08-15**, together with the companion
[`STOP_OVERRIDE_DRAWDOWN_LIFECYCLE_2026_08_15.md`](STOP_OVERRIDE_DRAWDOWN_LIFECYCLE_2026_08_15.md);
the owner signed in conversation ("signed.", given twice, one per document) after reading both
drafts and the advisory pricing. The document text is byte-identical to the draft the owner read;
only this status block and §7 changed to record the signature. Rows 340/341 are now reopened
**only** for the experiment defined below; the preflight condition remains binding before any real
fit.

## 1. What is reopened, exactly

[`DO_NOT_RETEST.md`](../research/history/DO_NOT_RETEST.md) row 340 bars any further selective
long-side entry model on the owned quote corpus; row 341 bars any further exit formulation for a
long 0DTE position on it. This reopening releases **one** experiment and nothing else:

- **Entry label family (3 members, Bonferroni 0.05/3):** for an exact causally eligible SPXW 0DTE
  contract at decision time, does its quote-mid path reach **+30% / +50% / +100%** of entry mid
  **before** ever touching **−30%**, within a frozen 60-minute frame? First-touch ordering is the
  label — this is the owner's "least predicted drawdown" design, and it is the one label axis
  (path ordering, MAE-conditioned) never previously fitted here.
- **Entry model:** one architecture within the conservative 29–50-parameter budget (row 343).
- **Exit in phase one:** none is fitted. Selected entries are scored through the declared law —
  exit at the bid on the first mid-touch of −30% (the thesis-violation stop, which matches the
  label's own ordering), otherwise a fixed 60-minute clock. Fixed, declared, never tuned.
- **Exit model (phase two, only if phase one survives every kill):** fitted solely on surviving
  out-of-fold selected entries, per row 341's own reopening condition, against duration-matched
  controls and the labeled-unattainable oracle.
- **Corpus:** the owned pre-cutoff quote corpus (sessions through 2026-07-31). No reserved
  sessions, no purchase.

## 2. Advisory requirement surface (2026-08-15; to be recomputed under the frozen declaration)

Measured on 32,976 random-entry 60-minute quote paths over 251 sessions (mid-based ordering,
ask-in/bid-out dollars, $3.08 fees; advisory pending the receipted rerun):

| Label cell | Base rate | W (hold-60) | L (hold-60) | Break-even precision | Gap |
|---|---:|---:|---:|---:|---:|
| +30% before −30% | 44.5% | +$395 | −$366 | 48.1% | +3.6pp |
| +50% before −30% | 31.8% | +$636 | −$338 | 34.7% | +2.9pp |
| +100% before −30% | 14.8% | +$1,195 | −$241 | 16.8% | +2.0pp |

Median path drawdown is −44%, so the −30% floor is selective by construction. **Detectability
honesty:** ~500 selected trades on this corpus can certify only a precision lift of roughly
+6–10pp — about 2–4× the break-even gap. A pass requires an edge of that size; a negative result
therefore reads "no large edge," never "no edge."

## 3. Kill conditions (pre-committed, in order)

1. The known-answer preflight fails → `UNDERPOWERED`; stop before the real fit; reopening spent.
2. Mid-to-mid gross on selected trades ≤ 0 → closed; reopening spent.
3. Family-corrected one-sided lower bound at ask-in/bid-out ≤ 0, or fewer than 4/5 chronological
   folds positive (absolute and against both controls) → closed as no large edge; reopening spent.
4. Positive at mid, negative only at the touch → `EXECUTION_BINDING`; predictive claim stands,
   trading claim does not; reopening spent.
5. Any timestamp leak, post-entry filter, survivorship defect, or reserved-session contact →
   `ARTIFACT`; reopening spent without repair-and-rerun.

No neighboring threshold, seed, horizon, architecture, subgroup, or operating-point retry follows
any kill. One shot.

## 4. Controls and machinery (unchanged from project law)

Chronological folds; session-level bootstrap inference; composition-matched control; identically
fitted shuffled-label null; per-feature timestamp audit; serial $10,000 one-position account with
the $500 risk law and WAIT; zero-P&L no-trade sessions in the primary estimand; declaration hashed
before any outcome is read.

## 5. What this does not touch

Rows 332/333/334/338/339/342/343 stand. The print corpus stays closed to selective models. The
short side and spreads stay closed per the census result. G1's general prohibition stands except
for this experiment via the same per-job release route used by jobs 29/32/36/37/41.

## 6. Prior odds, stated before signature

Recorded so the signature is informed, per project practice: the 08-14 ITM-depth rank fit showed
real ranking skill and failed chronological stability; the 08-14 enter-vs-WAIT fit lost −$15.93 per
trade mid-to-mid; the census closed both retail sides at fee-only. The one feature this experiment
has that none of those had: its label, stop, and clock are the same object (first-touch ordering),
and its break-even gap (+2–3.6pp) is the smallest this project has ever priced.

## 7. Signature

- Owner: **Owen Heidenreich** — signed in conversation ("signed."), 2026-08-15.
