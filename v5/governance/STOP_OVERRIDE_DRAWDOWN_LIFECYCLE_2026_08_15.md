# Override of the 2026-08-15 options-program STOP — narrow, preflight-gated

**Status: SIGNED 2026-08-15.** The owner signed in conversation with the single word "signed.",
given twice for this document and its companion reopening, after reading both drafts and the
advisory pricing. The document text is byte-identical to the draft the owner read; only this
status block and §5 changed to record the signature.

## 1. What is being overridden

On 2026-08-15 the owner pre-committed: *"Proceed only with the preregistered $0 defined-risk
short-vertical census on owned data … and STOP the options program if nothing clears even fee-only
execution."* The census closed the same day as `STOP_CONDITION_MET_NO_CELL_CLEARS_FEE_ONLY`
(receipt `v4/audit/autoresearch/short_vertical_census_2026_08_15/receipt.json`), so options-strategy
development stopped by that instruction.

Later the same day, after being shown the STOP conflict explicitly, the owner instructed:

> "continue trying to create a model that can trade 0dte options long calls and long puts"

and specified the design: an entry model selecting for the least predicted post-entry drawdown
(not beyond −30%), and an exit model that cuts trades violating that thesis early while capturing
profits on trades that run. This document makes that continuation lawful, on the record, and
narrow — the same mechanism as `STOPPING_RULE_OVERRIDE_2026_08_13.md`.

## 2. Scope of the override

Signing this reopens options-strategy development **only** for the single preregistered
drawdown-ordered lifecycle experiment defined in
`LEDGER_REOPENING_DRAWDOWN_LIFECYCLE_2026_08_15.md`. It does not reopen the short side, spreads,
other labels, other corpora, or any general search. If that experiment ends at any of its kill
conditions, this override is spent and the 2026-08-15 STOP resumes in full force.

## 3. What signing does NOT authorize

No data purchase, no vendor or broker contact, no paper or live orders, no promotion, no runtime
mutation, no reserved (post-2026-08-05) sessions. The fit itself additionally requires the signed
ledger reopening and a passed known-answer preflight (§4).

## 4. Preflight condition (binding)

No real fit may run until an end-to-end known-answer rehearsal of the exact pipeline — production
training law, SHA-256 seeds, best-checkpoint restore, per-trial records — recovers a planted edge
at the smallest size the gate must accept in at least 80% of trials with clean nulls. The 2026-08-15
capacity campaigns V1–V3 do not satisfy this; a new rehearsal against this experiment's declaration
is required. A failed preflight ends the experiment as `UNDERPOWERED` before any real outcome is
opened, and this override is spent.

## 5. Signature

Signing acknowledges: the census STOP fired and is being overridden knowingly; the measured prior
odds are poor (two path-label fits failed preregistered kills on 2026-08-14; the corpus certifies
only a large entry edge); and a negative result is final for this corpus.

- Owner: **Owen Heidenreich** — signed in conversation ("signed."), 2026-08-15.
