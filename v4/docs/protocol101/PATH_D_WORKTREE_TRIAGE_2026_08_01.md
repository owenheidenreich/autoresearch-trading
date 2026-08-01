# Path-D worktree triage — 2026-08-01

Status: **completed classification; no deletion or quarantine move performed**

- Branch: `v4/phase-0`
- Snapshot basis: `git status --porcelain=v1 -z -uall` before ignore hardening
Snapshot total: **28,799 dirty paths** (the earlier 1,886 figure was Git's
collapsed-directory presentation, not the expanded file count).

## Classification

| Category | Paths | Disposition |
|---|---:|---|
| Path-D / signed A5–A7 record | 51 | Commit deliberately in logical, secret-scanned commits. |
| Data and generated evidence | 27,692 | Ignore; do not delete or commit. |
| iCloud-style `… 2…` duplicates | 106 | Ignore now; preserve in place; use the dated candidate list before any later manifest-backed quarantine. |
| Unrelated pre-existing work | 950 | Leave untouched and report after the pivot commits. |
| **Total** | **28,799** | Every path is classified by the precedence rules below. |

The 51 in-scope paths comprise 25 modified tracked authority/spec/checker files
and 26 untracked Path-D/A6/A7 proposals, reviews, scripts, and receipts. The
governed record includes the A5 metadata state required by A6, the signed A6
entry-gate convexity amendment, the signed A7/D59 staged sub-minute amendment,
the walking-skeleton learnings ledger, and the Path-D planning/review packet.

## Deterministic precedence rules

Each path receives the first matching category:

1. **Path-D / A5–A7:** exact allowlist of the signed authority chain, the two
   governed amendment packet directories, and the named Path-D/walking-skeleton
   documents.
2. **iCloud duplicate:** an untracked path with a component ending in
   ` 2` or ` 2.<extension>`.
3. **Data/generated:** an untracked path under `v4/raw/`,
   `v4/normalized*/`, or a run-scoped
   `v4/audit/autoresearch/<packet>/`; a deployment conflict copy; a recorder
   `market_events.jsonl`; or a data/model suffix covered by the new ignore
   rules.
4. **Unrelated/preserve:** every remaining tracked or untracked path.

This makes the categories exhaustive and mutually exclusive. The 950-path
preserve set contains 71 modified tracked files, one tracked deletion, and 878
untracked files. It includes older v2/v3 work, runtime/broker edits, general
docs, model/scripts/tests, and other research material that this pivot record
does not own.

## Ignore result

Conservative root ignore rules were added for:

- local `.claude/` state and iCloud conflict-copy basenames;
- `.deploy-state <n>` conflict copies;
- `v4/raw/` and `v4/normalized*/` data while retaining tracked sentinels;
- Parquet/DBN/model/compressed artifacts and `market_events.jsonl`;
- untracked run-scoped autoresearch packet trees.

The signed A6 and A7 packet directories are explicit exceptions. Existing
tracked audit evidence remains tracked regardless of ignore rules. The expanded
visible status fell from 28,799 to **1,002** paths: the 51 in-scope paths, the
950 preserved paths, and the `.gitignore` edit itself. Therefore **27,798
pre-existing paths became newly ignored**, exactly the 27,692 data/generated
paths plus the 106 duplicate candidates.

No real source tree, `v4/docs/` contract tree, or tracked authority artifact
was broadly ignored. No file was deleted, moved, overwritten, or quarantined.

## Later quarantine boundary

The duplicate candidates are recorded in
`PATH_D_ICLOUD_DUPLICATE_QUARANTINE_CANDIDATES_2026_08_01.txt`.
That file is a review list, not authorization to move or delete. A later cleanup
must re-check current imports/references, create a dated move manifest with
original and destination paths, and provide rollback instructions before moving
anything.
