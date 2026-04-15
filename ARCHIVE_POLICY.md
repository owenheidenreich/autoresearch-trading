# Archive Policy

Two archive directories exist. Neither is searched by active tooling.

## `archive/`

Long-term historical storage. Contains v1 and v3 RL artifacts, old plans, old docs, old results logs. Pre-dates the current `v2/` system entirely.

- **When to consult:** Only when explicitly asked about pre-v2 history.
- **Never modify.** Read-only reference.

## `archive_quarantine/`

Quarantined files from the 2026-04-15 whitelist-first repo reduction. Contains files that were part of `v2/` but are not reachable from any canonical pipeline entrypoint.

- **Contents:** stale model checkpoints, pre-baseline experiment artifacts, broken live-trading module, stale session handoff docs, stale code copies.
- **See:** `archive_quarantine/ARCHIVE_MANIFEST.md` for a detailed inventory.
- **When to consult:** When you need a specific old model checkpoint or experiment artifact.
- **How to restore:** Copy the file back to its original `v2/` location. Verify it compiles.
- **Cleanup schedule:** After 90 days with no retrieval, contents can be deleted.

## What does NOT go into archive

- Anything imported by a canonical pipeline entrypoint
- Anything referenced by `CLAUDE.md`, `pre_run_gate.py`, or `docs/README.md`
- The current promoted model (`v2/models/model.pt`)
- `data.pt` and `data_sidecars/`

## Rule of thumb

If you're unsure whether a file is active: `grep -r "filename" v2/ --include="*.py"`. If nothing imports or references it, it belongs in archive.
