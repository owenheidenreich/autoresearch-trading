# Protocol101 Recorder Storage Maintenance - 2026-07-20

## Context

The Protocol101 IBKR recorder writes roughly 3 GiB per full trading day. On
2026-07-20 the data volume fell below the recorder safety floor:

```text
/System/Volumes/Data free space before cleanup: about 7.5 GiB
required recorder safety floor: 30 GiB
```

Low disk space previously contributed to recorder failures and write-health
risk. Raw IBKR capture evidence must not be deleted casually.

## Cleanup Performed

The cleanup intentionally avoided raw IBKR market-data deletion.

- Removed abandoned Git temporary object files under `.git/objects`.
  - `git count-objects -vH` reported about `26.36 GiB` of Git garbage before
    cleanup.
  - The largest files were abandoned `.git/objects/pack/tmp_pack_*` files from
    interrupted Git packing.
- Removed rebuildable local caches:
  - `/Users/gduby/.cache/codex-runtimes`
  - `/Users/gduby/.cache/uv`
  - `/Users/gduby/Library/Caches/com.openai.codex`
  - `/Users/gduby/Library/Caches/Codex`
  - `/Users/gduby/.autoresearch-trading/pycache-test`
  - `/Users/gduby/.autoresearch-trading/tmp`
  - `/Users/gduby/Documents/autoresearch-trading/.pytest_cache`
- Compressed one large derived research artifact:
  - `v4/audit/autoresearch/protocol101_live_v2_group2_geometry_uplift_attempt002/fold_predictions.csv`
  - compressed to `fold_predictions.csv.zst`
  - original and compressed SHA-256 files were written next to it.

Free space after cleanup was about `41 GiB`.

## Not Touched

- No raw IBKR market data was deleted.
- No sealed evidence directory was modified.
- No model training, threshold tuning, broker/API call, paper-submit,
  promotion/default change, runtime trading flag change, launchd schedule
  change, or real-money path change was performed.

## Remaining Risk

At roughly 3 GiB per recorder day, `41 GiB` free is enough for near-term
capture, but it is not enough margin for the entire July recorder packet while
preserving the 30 GiB safety floor.

The largest remaining project-owned storage buckets are:

```text
~/.autoresearch-trading/live_runtime/ibkr_capture         about 24 GiB
~/.autoresearch-trading/live_runtime/ibkr_capture_sealed  about 6 GiB
v4/audit/autoresearch                                     about 31 GiB
data/                                                     about 22 GiB
.git/                                                     about 16 GiB after cleanup
```

## Recommended Next Decision

Choose one evidence-preserving archival policy before the next several recorder
days accumulate:

1. Move older open/burned/development captures to an external volume, preserving
   directory structure and checksums.
2. Compress only non-sealed open/burned/development raw captures with `zstd`,
   writing restore notes and checksums for each compressed file.
3. Keep sealed captures untouched locally and only move/compress them under a
   separately signed sealed-evidence migration procedure.

Do not delete raw recorder evidence without an explicit owner-approved
retention rule.
