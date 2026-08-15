# Job 39 is inconclusive because its declared selector was defective

**Specification correction, 2026-08-14. No promotion, paper order, live order, external contact or
reserved session was involved.**

## Correct status

The economic conclusion previously attached to Job 39's V4 primary cell is **void**. The correct status
is:

> **INCONCLUSIVE — SPECIFICATION DEFECT.**

The fitted ranking is retained. The zero-trade V4 economic receipt is retained as immutable audit
history, but it is not a negative result and does not spend the reopening.

## The defect

V4 trained a smooth-L1 regression on a label clipped to `[-25, 30]` points and normalized by 30. Its
declared selector then required a prediction of at least **30 points**, exactly the upper boundary of the
target support. A conditional-mean regression can rank high-depth opportunities while remaining below
that boundary. Requiring the boundary itself therefore made policy activation depend on a degenerate
ceiling case or model overshoot rather than on the learned ranking.

That defect is derivable from the declaration's label arithmetic before reading any economic outcome.
Correcting the selector is therefore a specification repair, not post-outcome operating-point shopping.

The V4 primary model's maximum out-of-fold score was **29.8785**, so the defective selector chose zero
trades from 423,053 predictions. With no selected trades, mid-to-mid economics were never measured. A
zero-trade result under an invalid selector cannot establish non-positive gross P&L.

## What remains valid

- The width-3 parameter counts remain 226 / 245 / 322 / 341 for the four permitted architectures.
- The 24 chronological real/shuffled fits and their out-of-fold predictions remain valid model evidence.
- The 61-feature causal timestamp audit, whole-chain equality proof, no-post-entry-filter assertion,
  reserved-session cutoff and cache integrity checks remain valid.
- The generous evidence budget is **377 parameters**. The conservative design-effect budget remains
  **29–50 parameters**, admitting none of the declared family, and must accompany every result.

What is void is only the V4 selector and every economic interpretation derived from its failure to fire.
No Job 39 negative finding belongs in the do-not-retest ledger.

## Corrective V5 law

V5 must freeze a causal rank-based operating point before reading P&L. For each chronological fold, its
cutoff is calibrated only from strictly earlier training sessions; the scored session is processed
minute by minute under the same two-trade cap and 120-minute occupancy clock. A whole-day top-N rule that
uses later scores from the scored day is forbidden.

V5 must also run a pre-economics attainability gate. Any absolute prediction threshold at or above the
label's clip ceiling is refused. A rank selector must prove that its cutoff is finite, derived without
outcomes or P&L, and frozen from the training prefix.

Kill condition #1 remains unchanged once V5 actually selects trades: mean gross mid-to-mid P&L per trade,
with spread and fees removed, must be strictly positive. If the measured value is non-positive, the
corrected run is negative and stops.

## Evidence retained and superseded

- immutable V4 declaration: `v5/work/entry-exit-attribution/DECLARATION_V4.json`
- valid causal feature audit: `v4/audit/autoresearch/causal_day_fit_feature_audit_2026_08_14_attempt001/receipt.json`
- valid 24-fit receipt: `v4/audit/autoresearch/causal_day_magnitude_fit_2026_08_14_attempt001/receipt.json`
- immutable but economically void V4 selector receipt: `v4/audit/autoresearch/causal_day_magnitude_primary_economics_2026_08_14_attempt001/receipt.json`

The owner is concurrently maintaining `STATUS.md`, the causal-day policy gate, frozen knobs, statistics
and the do-not-retest ledger. None is changed by this correction.

## Corrected successor result

V5 subsequently applied the frozen causal training-prefix rank law. It selected 24 trades and therefore
confirmed that V4's zero-trade interpretation was void. The corrected policy passed gross mid-to-mid and
both pooled control point estimates, but failed the chronological fold and corrected-confidence standard.
That distinct result is recorded in
[`CAUSAL_DAY_MAGNITUDE_RANK_POLICY_2026_08_14.md`](CAUSAL_DAY_MAGNITUDE_RANK_POLICY_2026_08_14.md).
