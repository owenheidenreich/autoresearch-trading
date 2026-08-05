# Path-D — Programme Restart Record

> **Dated decision record.** This records the 2026-08-05 restart. Later Track-A scheduling and
> execution changes are recorded only in [STATUS.md](../STATUS.md), which controls current state.

**Status: OWNER-DECIDED — the research programme is REOPENED as of 2026-08-05.**

Owner instruction, 2026-08-05: *"un-do the stand down i authorized accidentally, reopen the
investigation so docs are consistent."*

This record exists because the programme was stood down and restarted within two hours on 2026-08-04, and
the documents were never reconciled. It resolves the conflict. It does not rewrite history: the
stand-down record stays exactly as written, because its findings are correct and still binding.

---

## 1. What happened, in order

| When (2026-08-04 PT) | Event | Evidence |
|---|---|---|
| 11:21 | Track-A capture authorized, wrapper built, launchd jobs installed | `d9ed0c0c` |
| 13:30 | Build-order amendment §2 stands the capture down; owner asked to run `bootout` | `017e5692` |
| 15:14 | Stand-down record (Option C) written; live audit finds Track-A jobs **gone** — accurate at that moment | `3848f430` |
| 15:23 | Owner asks for a path to a trained model; **mechanism-first master plan reopens the programme** | `b78c1d3c` |
| 19:15 | Track-A launchd plists **re-created** — the capture was re-armed during the reopening | plist birth time |
| 18:41–19:38 | Admission-ledger enforcement, exit-side ledger, capture hardening all proceed | `3acafb85` … `883f8185` |
| 2026-08-05 08:04 | launchd entry point moved out of the iCloud-synced tree after the open window died | `a6594ebb` |

**Nothing was violated.** The stand-down was real and was executed; the restart was a real owner decision
two hours later. Only the paperwork lagged.

## 2. The owner's stated reason for the stand-down no longer holds

The stand-down was accepted partly on a misunderstanding the owner has now stated plainly: the term
"stood down" was not understood to mean "stopped," and the decision was signed without that being clear.
That is a governance defect in **how the decision was presented**, not in the decision itself, and §5 of
the ground rules now addresses it directly.

## 3. What this record changes

| Document | Effect |
|---|---|
| [`PATHD_BUILD_ORDER_AMENDMENT_2026_08_04.md`](../../v4/docs/protocol101/training/contracts/PATHD_BUILD_ORDER_AMENDMENT_2026_08_04.md) §2 | **SUPERSEDED.** The Track-A capture is authorized and running. Do **not** run the `bootout` block. |
| [`PATHD_PROGRAMME_STAND_DOWN_RECORD_2026_08_04.md`](../../v4/docs/protocol101/training/contracts/PATHD_PROGRAMME_STAND_DOWN_RECORD_2026_08_04.md) §4 | **SUPERSEDED** as a live-state audit — it is a snapshot of 2026-08-04 15:14 and is no longer current. |
| Stand-down record §5, "Phase-0 feature certification queue — frozen" | **SUPERSEDED.** The queue is unfrozen; Phase-0 certification is the active work. |
| Stand-down record §1, §2, §3, §6, §7 | **STILL IN FORCE.** The five negatives stand, the preserved assets stand, the holdout is still SPENT, the restart conditions still govern, and the lessons still bind. |

**The restart conditions in stand-down §6 are not waived by this record — they are the standard the
reopened programme must meet.** Specifically: a stated mechanism rather than a correlation; economics
computed on raw data before any modelling apparatus; a confirmation path that actually exists; and nulls
that pass a known-answer gate.

## 4. Capture scope, as narrowed by the owner

Owner instruction, 2026-08-05, given **before** the 09:10 PT window observed any data:

- **Certified sample: 2026-08-06 and 2026-08-07 only**, both windows each — 4 windows, 2 open bursts.
- **2026-08-05 is excluded from the certified sample.** Its 09:28 PT open window was lost to an iCloud
  file-eviction fault (`launchd.open.err`, 06:28 PT); its 09:10 PT midday window runs only to confirm the
  launcher repair and is recorded as an **infrastructure verification run**, not evidence.

The exclusion decision was made **pre-observation**, which is what keeps it lawful under declaration v5's
`selection_law` ("never drop or substitute after observing data"). Narrowing sits inside the existing
`authorization.json` scope, whose own precedent reads: *"Narrowing within an authorized window requires no
re-authorization; widening it would."*

**Declared limitation, recorded before the data is seen:** two sessions give two open-burst samples. The
emission lag `L` is set from the worst observed p99, so a thinner sample is a weaker worst-case estimate.
This is an accepted, owner-chosen limitation, to be reviewed 2026-08-08 alongside the results.

Declaration v6 recording this is written **after** the 2026-08-05 midday window completes, so that
narrowing the session list does not cause the wrapper's session gate to refuse the verification run.

## 5. Still outstanding for the owner

The two `shadowasof` launchd jobs remain loaded with zero-byte logs since 2026-07-18. They contact no
broker and no vendor and cost nothing. They are unrelated to Track-A and need no action for the restart.

*Signed: Claude Opus 5 — 2026-08-05. Owner decision: reopen, 2026-08-05.*
