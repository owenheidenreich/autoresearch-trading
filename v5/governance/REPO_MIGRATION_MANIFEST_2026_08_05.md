# Repository Migration Manifest — moving out of `~/Documents`

**Status: DRAFT procedure — owner-executed, Tier 1. Drafted 2026-08-05 on the owner's decision of
the same day. Do not execute before the Track-A capture sessions (2026-08-06/07) are banked.**

## Why

macOS protects `~/Documents`: a scheduled job has no parent whose read permission it can inherit,
so no launchd job can run this repository's code (STATUS job 4a — the 08-05 capture window died on
exactly this). The block covers unattended Track-A capture now and G7 live shadow / G8 guarded
paper later. Moving the repository out of any TCC-protected folder (`Documents`, `Desktop`,
`Downloads`, iCloud-synced trees) removes the whole class of failure, including the iCloud
file-eviction fault recorded in the restart record.

## Target

`/Users/gduby/autoresearch-trading` — inside the home directory, outside every TCC-protected and
iCloud-synced path. The external data trees are unaffected: `~/.autoresearch-trading` is already
outside `Documents`, and `/Volumes/AR_TRADING_DATA` does not move.

## Owner-executed procedure

1. **Wait** until both Track-A sessions are banked and no attended runner is live.
2. **Snapshot** (from the old location):
   `git -C ~/Documents/autoresearch-trading status --porcelain > /tmp/pre_move_status.txt`
   and `git fsck --no-dangling` must report no missing objects.
3. **Stop watchers:** confirm no launchd job, editor, or agent has the tree open. The two stale
   `shadowasof` jobs from 07-18 should be booted out first (separate owner decision, same sitting).
4. **Move, do not copy:** `mv ~/Documents/autoresearch-trading /Users/gduby/autoresearch-trading`.
   A move on the same volume preserves inodes and takes seconds; never leave two live copies.
5. **Leave a marker, not a symlink,** at the old path: a single `MOVED_TO.txt` naming the new
   path. A symlink back into `Documents` would silently reintroduce the TCC problem for anything
   that follows it.
6. **Re-point externals:** IDE workspaces, shell profiles, and any launchd plist that names the old
   absolute path (currently the Track-A plists; inventory with
   `grep -rl 'Documents/autoresearch-trading' ~/Library/LaunchAgents`).
7. **Verify** (from the new location):
   - `git status --porcelain` matches the pre-move snapshot byte for byte;
   - `./.venv/bin/python -m pytest v5/tests -q` — all green (the venv may need
     `python -m venv --upgrade-deps` or recreation if it hard-codes the old path — check
     `.venv/bin/python` runs at all first);
   - `./.venv/bin/python v5/ops/check_project.py` — clean;
   - one scheduled no-op job that reads a file in the new tree fires successfully — this is the
     acceptance test the whole move exists for.
8. **Record** the completed move in STATUS (flip job 4a) with the date and the verification output.

## Signed evidence and absolute paths — the rule

Signed receipts and audit files under `v4/audit/` and elsewhere record absolute paths beginning
`/Users/gduby/Documents/autoresearch-trading/...`. **They are never edited** — a signed file's
bytes are its identity. The path mapping is recorded here instead:

> Any absolute path beginning `/Users/gduby/Documents/autoresearch-trading/` in a receipt written
> before the move date resolves to `/Users/gduby/autoresearch-trading/` after it.

Verifiers that resolve receipt paths should try the recorded path first and apply this mapping on
miss. (`v5/research/feature_admission.py` resolves relative paths against the repository root and
is unaffected; only fully absolute recorded paths need the mapping.)

## What can go wrong

| Risk | Control |
|---|---|
| Two live copies diverge | Move, never copy; marker file at the old path |
| venv breaks (hard-coded shebangs) | Verification step 7 catches it; recreate `.venv` if needed |
| A plist still points at the old path | Step 6 inventory grep; step 7 scheduled-job acceptance test |
| Signed evidence "fixed" by find-and-replace | Forbidden by the rule above; mapping lives here instead |
| Move mid-capture corrupts a session | Step 1: never move while a capture or runner is live |

*Drafted by Claude Fable 5, 2026-08-05. This document changes nothing by itself; every step above
is owner-executed.*
