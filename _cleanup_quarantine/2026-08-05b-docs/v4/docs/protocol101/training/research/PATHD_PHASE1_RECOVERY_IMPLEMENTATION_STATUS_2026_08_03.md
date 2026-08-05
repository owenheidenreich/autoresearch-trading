# Path-D Phase-1 Recovery Implementation Status — 2026-08-03

## Verdict

`BLOCKED_MISSING_APPROVED_EXTERNAL_STORAGE`

The causal entry/exit campaign and no-submit runtime boundaries are implemented,
but no real development artifact has been trained. No approved external
encrypted APFS volume is mounted. The internal data volume is 90% used with
about 47 GiB available, so the 25% free-space contract correctly prohibits the
215-session development materialization.

This is not `TIER_S_SUPPORTED`, `TIER_S_NOT_SUPPORTED`, `UNDERPOWERED`, or
`INVALID`; those are replay verdicts and require completed real OOF artifacts.

## Implemented

- `v4/research/pathd_phase1_entry.py`
  - Rebuilds the signed-18 tuple from raw native CBBO-1m, exact-contract
    CBBO-1s, and official completed SPX only.
  - Inventories all 251 source sessions but permits decoding only the first 215;
    the final 36-session firewall (including the protected 30) is rejected.
  - Freezes entry emission at 2,336 ms and arrival at another 1,000 ms.
  - Uses the conservative one-tick-through limit fill law and separate labels.
  - Trains five expanding HGB folds with one-session embargo and chronological
    tail calibration.
  - Emits learned and deterministic-control evaluation receipts plus embargoed
    prequential initial-history receipts used only to train exit folds.
  - Prevents the full-development shadow artifact from generating exit-training
    receipts.
- `v4/research/phase1_exit_model.py`
  - Builds 52 one-second causal features with a 320 ms completed-second
    availability clock.
  - Keeps labels separate and supports 0/1/2/5-second latency and $3/$4
    round-trip fee label partitions.
  - Deterministically caps fit/calibration rows at 500,000/150,000 and fits
    HGB mean/q10/q50/q90 heads with the frozen utility composer.
- `v4/research/pathd_phase1_replay.py`
  - Runs the four entry/exit boxes, fixed comparators, eight matched-rate random
    exits, latency/fee sensitivities, session bootstrap, fold deltas, and four
    negative controls.
  - Emits only the frozen verdict vocabulary and never opens the holdout.
- `v4/research/phase1_storage.py` and `v4/scripts/run_phase1_exit_model.py`
  - Enforce encrypted external APFS, stable volume name, 25% free space, and a
    150 GB Phase-1 cap.
  - Provide checksum-preserving relocation and resumable materialize/train/
    trajectory/replay stages.
- `v4/scripts/run_pathd_candidate.py` and
  `v4/path_d/runtime/ibkr_paper_dry_run.py`
  - Expose only `databento-no-order` and `ibkr-paper-dry-run`.
  - Require frozen full-development shadow manifests.
  - Connect to IBKR read-only, require a `DU` paper account, qualify exact SPXW
    identity, and construct governed previews without a submission operation.

## Evidence completed

- One owned raw session (`2025-12-23`) materialized 1,150 causal entry rows in
  2.39 seconds without reading quarantined processed examples.
- A synthetic 100-session entry campaign produced five fold artifacts, 102
  evaluation receipts, and 46 initial-history training-only receipts; the
  full-development artifact reloaded successfully.
- A synthetic full entry-to-exit campaign fitted all five exit folds and the
  full-development 52-feature exit artifact; all five synthetic target-skill
  folds were positive.
- The zero-cost gate for the captured August 3 historical CBBO-1m slice returned
  exactly `$0.00`; 1,287 historical rows were downloaded and hash-receipted.
- Same-session live/historical comparison passed exact decoded-value identity
  for all 914 captured live CBBO-1m rows across every model/source column.
- Focused new/modified tests pass. The broader Path-D run passed 185 tests and
  failed four legacy Claude-release reconciliation tests before their test
  bodies because already-dirty registered source/test hashes differ from the
  immutable release receipt. That unrelated release was not rewritten.

## Hard stops preserved

- Protected holdout opened: `false`.
- Paper order submitted: `false`.
- Broker submission endpoint called by the Path-D candidate: `false`.
- Protocol160 modified: `false`.
- Paper-default registry modified: `false`.
- Promotion/default/runtime flags modified: `false`.
- `cmbp-1` purchased or downloaded: `false`.

## Resume sequence after the SSD is mounted

Set these roots to directories on `/Volumes/AR_TRADING_DATA`:

```bash
export AR_TRADING_DATA_ROOT=/Volumes/AR_TRADING_DATA
export AR_TRADING_SCRATCH_ROOT=/Volumes/AR_TRADING_DATA
export AR_TRADING_ARTIFACT_ROOT=/Volumes/AR_TRADING_DATA/artifacts
```

Then run, in order:

```bash
PYTHONPATH=. .venv/bin/python -m v4.scripts.run_phase1_exit_model storage-preflight --create-roots

PYTHONPATH=. .venv/bin/python -m v4.scripts.run_phase1_exit_model relocate-corpus \
  --source /Users/gduby/.autoresearch-trading/pathd_2025-08-01_2026-07-31

PYTHONPATH=. .venv/bin/python -m v4.scripts.run_phase1_exit_model materialize-entry --resume

PYTHONPATH=. .venv/bin/python -m v4.scripts.run_phase1_exit_model train-entry

PYTHONPATH=. .venv/bin/python -m v4.scripts.run_phase1_exit_model build-trajectories --resume \
  --entry-campaign /Volumes/AR_TRADING_DATA/artifacts/entry_v2/campaign.json \
  --emission-lag-receipt v4/audit/autoresearch/thetadata_completed_minute_timing_2026_08_03/shared_emission_lag.json

PYTHONPATH=. .venv/bin/python -m v4.scripts.run_phase1_exit_model train-exit \
  --entry-campaign /Volumes/AR_TRADING_DATA/artifacts/entry_v2/campaign.json

PYTHONPATH=. .venv/bin/python -m v4.scripts.run_phase1_exit_model replay \
  --entry-campaign /Volumes/AR_TRADING_DATA/artifacts/entry_v2/campaign.json \
  --exit-campaign /Volumes/AR_TRADING_DATA/artifacts/exit_v1/campaign.json
```

Only after those complete should the separate candidate runner use the two
full-development shadow manifests for the next no-order Databento session and
read-only IBKR paper dry-run.
