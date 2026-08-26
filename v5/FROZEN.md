# v5 is frozen

**Frozen 2026-08-26.** Tagged `v5-final`.

This tree is evidence now, not a workspace. Read it. Do not build in it.

Active development moved to a new repository, `protocol101`, which reads the shared data volume and
treats everything here as a frozen archive. It never imports from this tree, and a check enforces that.

## What this tree holds

| | |
|---|---|
| Python | ~690,000 lines across v2, v3, v4 and v5 |
| Markdown | ~3,800 files |
| Last work | Job 55, completed 2026-08-25 |

**Job 55 was the final job.** It acquired 21 of 21 SPXW consolidated-quote sessions:
2,373,877,845 records, QC passed with zero gaps, disconnects or out-of-scope calls, at
`$1.900784897804` in committed quotes. The actual vendor invoice remains `UNKNOWN`. Its aggregate
receipt is on the data volume under `cmbp-tier0/job55/receipts/`.

## What carried forward

Nothing was copied. Each item below was re-derived against this tree's own tests as an oracle, and
those tests still pass unmodified.

- The detectable-effect arithmetic from `v5/research/statistics.py`.
- The greeks from `v5/research/greeks.py`. The new implementation passes `v5/tests/test_greeks.py`
  verbatim, changed only in its import line.
- The CMBP decoder from `v5/research/cmbp_tier0.py`, reproducing all 24 published session counters.
- `research/history/DO_NOT_RETEST.md`, ported into a machine-readable ledger that mechanically refuses
  a closed configuration rather than asking anyone to remember it.

The most important thing carried across is one record: **buying SPXW 0DTE premium at minute cadence is
closed as a class**, structurally, at -$13.00 per trade before any friction. It reopens on demonstrated
directional skill on the underlying, and nothing else.

## What was deliberately not done

No history rewrite. No object cleanup. No deletion. `.git` remains about 10 GiB and the working tree
about 143 GiB. iCloud sync conflict duplicates are ignored but still on disk.

The data on `/Volumes/AR_TRADING_DATA` is untouched and is shared with the new repository, read-only.

## If you are an agent reading this

You are in the wrong tree. Current work is in `protocol101`. Read its `AGENTS.md`.

This tree is safe to read and cite as evidence. It is not safe to extend: its governance was prose
enforced by asking the owner, which is the specific problem the new repository exists to fix.
