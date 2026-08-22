# The protocol-v2 amendments are adopted, and their own header cannot say so

**Decided 2026-08-22 on owner delegation.** This resolves a textual conflict a cold review raised,
and the resolution is not the obvious one.

## The conflict

[`work/entry-exit-attribution/PROTOCOL_V2_AMENDMENT_DRAFT_2026_08_15.md`](../work/entry-exit-attribution/PROTOCOL_V2_AMENDMENT_DRAFT_2026_08_15.md)
opens with **"Status: DRAFT. Nothing here is adopted."**

[`DEVELOPMENT_CHARTER_2026_08.md`](DEVELOPMENT_CHARTER_2026_08.md) §3, **signed 2026-08-15**, says:
*"The owner adopts amendments A1–A13, revision 2, in [that file], for job 46."*

So the repository asserted both that these amendments bind every job-46 fit — the per-fit capacity
statement, the self-hashed runner-verified declaration, deterministic seeds, best-checkpoint
restoration, the exposure ledger, session-level inference, the divergence gate — and that nothing in
them was adopted.

## The ruling

**The charter governs. The amendments are ADOPTED for job 46 as of 2026-08-15.** The header is stale
text that was never re-read after signing, not a live reservation.

## Why the header was not simply corrected

Editing it was attempted on 2026-08-22 and **correctly refused by the project's own guard**.

The file is pinned by
[`PREACQUISITION_SEMANTIC_FREEZE_V1.json`](../work/lifecycle-training/PREACQUISITION_SEMANTIC_FREEZE_V1.json)
(`71463cc0…`), whose `post_contact_rule` reads: *"No listed source, population, target, optimizer,
architecture, chronology, control, inference, or execution rule may change after the preflight."*
`test_preflight_prices_every_session_schema_before_writing_passing_receipt` failed on
`semantic freeze source drift`, and the edit was reverted.

**That guard is right and the header stays.** This file is pinned because it *defines the training
law that governed a $19.2450 vendor purchase*. The freeze's job is to prove that the semantics under
which money was spent are byte-identical to the semantics on disk today. A cosmetic header fix that
breaks that proof is a bad trade: it would repair a sentence nobody acts on at the cost of the
evidence that the purchase was governed by what we say it was.

The project's own working rule says the same thing in general form: **a pinned file's hashes are a
historical record, not a value to regenerate.** The sanctioned move when a pinned file must change is
to seal its phase against re-running — and sealing the completed acquisition phase to fix a stale
header would be disproportionate by a wide margin.

## Therefore

- **This note is the authoritative adoption record.** Anything reading that file's header must read
  this alongside it.
- **The header and the filename both stay as they are.** The filename's `DRAFT` is likewise
  historical: the charter adopts the file by exact path, and renaming it would break a signed
  reference to fix a cosmetic one.
- **Nothing about the amendments' substance is in question.** All thirteen (A1–A13, revision 2) are
  in force for job 46 and were verified textually unchanged during this episode.
